"""Fit the feature pipeline and per-entity baselines.

Run it with::

    python -m training.build_baselines

It reads the **training split only** and writes to ``artifacts/``: encoders, the numeric scaler,
the sequence vocabulary, corpus statistics, per-entity behavioral profiles, behavioral cohorts
with their priors, and a global prior.

The ordering below is not arbitrary; it is what keeps the fit honest.

**Vocabulary and encoders are fitted first**, in a cheap pre-pass. They only need to know which
categories and tokens exist, which is ordinary vocabulary construction.

**Then a single streaming pass** replays training events in time order. Each event is featurized
against the profile assembled from *strictly earlier* events, then folded in. This is the same
sequence the online scorer follows, so offline features equal online features by construction.

A batch alternative -- build all profiles, then featurize everything against them -- would be
much faster and quietly wrong: every event would be compared to a baseline that already includes
it, so "unusual for this entity" would be systematically understated and offline metrics would
be unreachable in production.

**The scaler is fitted last**, on the raw feature matrix that pass produced, because it needs
the feature distribution that the pipeline actually generates.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.artifacts import read_manifest, write_manifest
from common.config import settings
from common.models import Event
from common.seed import set_global_seed
from data_generator.generate import dataframe_to_events, load_events
from features.cohorts import CohortModel, build_cohort_priors, build_global_prior
from features.encoders import CategoricalEncoder, EncoderBundle, NumericScaler
from features.entity_window import BehaviorProfile, ProfileAccumulator, ProfileStore
from features.event_features import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    CorpusStats,
)
from features.featurize import FeaturePipeline, FeatureVector
from features.sequences import SequenceVocab

logger = logging.getLogger(__name__)

#: Cap on distinct categories per field. Resources are the only high-cardinality field here, and
#: a code space wider than the data can support just adds noise for the tree model.
MAX_CATEGORIES: Dict[str, int] = {
    "resource_accessed": 200,
    "device_os": 60,
    "geo_country": 60,
    "entity_type": 10,
    "auth_method": 10,
    "device_protocol": 20,
    "cohort": 64,
}

#: Minimum times a command must appear to earn a vocabulary slot; rarer ones map to ``<unk>``.
MIN_TOKEN_COUNT = 3


def load_split_events(split: str = "train", directory: Optional[Path] = None) -> List[Event]:
    """Load one split's events as :class:`Event` objects, in time order.

    Labels are never loaded. This script has no reason to see them, and not loading them makes
    target leakage impossible rather than merely unlikely.
    """
    frame = load_events(directory)
    subset = frame[frame["split"] == split]
    if subset.empty:
        raise ValueError(
            f"No events in split '{split}'. Run: python -m data_generator.generate --seed 42"
        )
    subset = subset.sort_values(["timestamp", "event_id"])
    return dataframe_to_events(subset)


def fit_encoders(events: Sequence[Event]) -> EncoderBundle:
    """Fit categorical encoders over the training events.

    The scaler is left unset here: it needs the raw feature matrix, which does not exist until
    the streaming pass has run.
    """
    columns: Dict[str, List[Any]] = {name: [] for name in CATEGORICAL_FEATURE_NAMES}

    for event in events:
        columns["entity_type"].append(event.entity_type.value)
        columns["auth_method"].append(event.auth_method.value)
        columns["geo_country"].append(event.geo.country)
        columns["device_protocol"].append(event.device_fingerprint.protocol)
        columns["device_os"].append(event.device_fingerprint.os)
        columns["resource_accessed"].append(event.resource_accessed)

    # Cohort codes are the cohort ids themselves plus "unknown" for an unassigned entity. Fitted
    # from the configured cohort count rather than from data, so the encoder is stable even if a
    # cohort happens to end up empty.
    columns["cohort"] = [str(index) for index in range(settings.cohort_count)] + ["unknown"]

    encoders = {
        name: CategoricalEncoder.fit(
            name, values, max_categories=MAX_CATEGORIES.get(name), min_count=1
        )
        for name, values in columns.items()
    }

    return EncoderBundle(
        categorical=encoders,
        scaler=None,
        numeric_names=list(NUMERIC_FEATURE_NAMES),
        categorical_names=list(CATEGORICAL_FEATURE_NAMES),
    )


def fit_vocabulary(events: Sequence[Event]) -> SequenceVocab:
    """Build the command vocabulary and global n-gram statistics."""
    return SequenceVocab.fit(
        (event.command_sequence for event in events),
        min_count=MIN_TOKEN_COUNT,
        max_len=settings.sequence_max_len,
        ngram_n=settings.sequence_ngram_n,
    )


def streaming_pass(
    pipeline: FeaturePipeline, events: Sequence[Event]
) -> Tuple[List[FeatureVector], Dict[str, ProfileAccumulator]]:
    """Replay events in time order, featurizing each against strictly earlier history.

    Returns the feature vectors and the per-entity accumulators, which become the persisted
    profiles and the cohort priors.
    """
    pipeline.reset_state()
    vectors: List[FeatureVector] = []

    started = time.perf_counter()
    for position, event in enumerate(events, start=1):
        vectors.append(pipeline.featurize(event, update_state=True, use_live_profile=True))
        if position % 25_000 == 0:
            rate = position / max(1e-9, time.perf_counter() - started)
            logger.info("  featurized %d/%d events (%.0f/s)", position, len(events), rate)

    elapsed = time.perf_counter() - started
    logger.info(
        "Streaming pass complete: %d events in %.1fs (%.0f events/s, %.2f ms/event)",
        len(events),
        elapsed,
        len(events) / max(1e-9, elapsed),
        1000.0 * elapsed / max(1, len(events)),
    )
    return vectors, dict(pipeline._accumulators)


def compute_entity_feature_stats(
    profiles: Dict[str, BehaviorProfile],
    vectors: Sequence[FeatureVector],
    numeric_names: Sequence[str],
) -> None:
    """Attach per-entity raw feature means and standard deviations, in place.

    The Phase 3 baseline model scores statistical deviation against these, so they must be the
    *raw* values: a per-entity mean of globally-scaled features would measure deviation from the
    population, which is a different and much weaker question.
    """
    grouped: Dict[str, List[np.ndarray]] = {}
    for vector in vectors:
        row = np.asarray(
            [vector.raw.get(name, 0.0) for name in numeric_names], dtype=float
        )
        grouped.setdefault(vector.entity_id, []).append(row)

    for entity_id, rows in grouped.items():
        profile = profiles.get(entity_id)
        if profile is None:
            continue
        matrix = np.vstack(rows)
        profile.feature_names = list(numeric_names)
        profile.feature_means = np.nan_to_num(matrix.mean(axis=0)).tolist()
        profile.feature_stds = np.nan_to_num(matrix.std(axis=0)).tolist()


def attach_prior_feature_stats(
    prior: BehaviorProfile,
    member_ids: Sequence[str],
    vectors: Sequence[FeatureVector],
    numeric_names: Sequence[str],
) -> None:
    """Attach pooled feature statistics to a cohort or global prior.

    Pooled over every member's events, so a cold-start entity inherits a variance estimate that
    reflects its cohort's real spread rather than a handful of its own observations.
    """
    members = set(member_ids)
    rows = [
        np.asarray([vector.raw.get(name, 0.0) for name in numeric_names], dtype=float)
        for vector in vectors
        if vector.entity_id in members
    ]
    if not rows:
        return
    matrix = np.vstack(rows)
    prior.feature_names = list(numeric_names)
    prior.feature_means = np.nan_to_num(matrix.mean(axis=0)).tolist()
    prior.feature_stds = np.nan_to_num(matrix.std(axis=0)).tolist()


def build(
    split: str = "train",
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit the whole feature pipeline and write it to ``artifacts/``.

    Returns a summary dictionary suitable for logging and for the manifest.
    """
    set_global_seed()
    target_dir = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)

    logger.info("Loading '%s' split", split)
    events = load_split_events(split, dataset_dir)
    logger.info("Loaded %d events for %d entities", len(events), len({e.entity_id for e in events}))

    # --- pre-pass: vocabulary, encoders, corpus statistics ---
    logger.info("Fitting vocabulary, encoders and corpus statistics")
    vocab = fit_vocabulary(events)
    encoders = fit_encoders(events)
    corpus = CorpusStats.fit(list(events))
    logger.info(
        "  vocabulary %d tokens, %d n-grams; %d resources in corpus",
        vocab.size,
        len(vocab.ngram_counts),
        len(corpus.resource_frequency),
    )

    # --- streaming pass: profiles and raw features together ---
    pipeline = FeaturePipeline(
        encoders=encoders, vocab=vocab, profiles=ProfileStore(), cohorts=CohortModel(), corpus=corpus
    )
    logger.info("Streaming pass: building profiles and raw features")
    vectors, accumulators = streaming_pass(pipeline, events)

    profiles = {
        entity_id: accumulator.build() for entity_id, accumulator in accumulators.items()
    }

    # --- scaler, fitted on the distribution the pipeline actually produces ---
    logger.info("Fitting the numeric scaler")
    numeric_names = list(NUMERIC_FEATURE_NAMES)
    raw_matrix = FeaturePipeline.raw_matrix(vectors, numeric_names)
    scaler = NumericScaler.fit(numeric_names, raw_matrix)
    encoders.scaler = scaler

    # --- cohorts and priors ---
    logger.info("Clustering entities into behavioral cohorts")
    cohort_model = CohortModel.fit(list(profiles.values()))
    assignments: Dict[str, int] = {}
    for entity_id, profile in profiles.items():
        cohort = cohort_model.assign(profile)
        if cohort is not None:
            profile.cohort = cohort
            assignments[entity_id] = cohort

    cohort_priors = build_cohort_priors(accumulators, assignments)
    global_prior = build_global_prior(accumulators)

    # --- per-entity and per-prior feature statistics ---
    logger.info("Computing per-entity feature statistics")
    compute_entity_feature_stats(profiles, vectors, numeric_names)
    for cohort, prior in cohort_priors.items():
        members = [
            entity_id for entity_id, assigned in assignments.items() if assigned == cohort
        ]
        attach_prior_feature_stats(prior, members, vectors, numeric_names)
    attach_prior_feature_stats(global_prior, list(profiles), vectors, numeric_names)

    store = ProfileStore(
        profiles=profiles,
        cohort_priors=cohort_priors,
        global_prior=global_prior,
        type_cohorts=cohort_model.type_cohorts,
    )

    pipeline.profiles = store
    pipeline.cohorts = cohort_model
    pipeline.encoders = encoders

    logger.info("Writing artifacts to %s", target_dir)
    paths = pipeline.save(target_dir)
    _update_manifest(paths, target_dir)

    cold_start_count = sum(1 for profile in profiles.values() if profile.cold_start)
    summary: Dict[str, Any] = {
        "split": split,
        "n_events": len(events),
        "n_entities": len(profiles),
        "n_features": pipeline.n_features,
        "n_numeric": len(numeric_names),
        "n_categorical": len(CATEGORICAL_FEATURE_NAMES),
        "vocab_size": vocab.size,
        "n_cohorts": cohort_model.n_cohorts,
        "cohort_sizes": dict(sorted(cohort_model.sizes.items())),
        "cohort_labels": {key: value for key, value in sorted(cohort_model.labels.items())},
        "n_cold_start_profiles": cold_start_count,
        "type_cohorts": cohort_model.type_cohorts,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    return summary


def _update_manifest(paths: Dict[str, Path], target_dir: Path) -> None:
    """Record the feature artifacts in the manifest without disturbing later slots."""
    manifest = read_manifest(target_dir / settings.manifest_filename)
    slots = manifest.get("artifacts") or {}
    for key in ("encoders", "sequence_vocab", "entity_profiles", "cohorts"):
        if key in paths:
            slots[key] = Path(paths[key]).name
    slots["scaler"] = Path(paths["encoders"]).name  # the scaler lives inside the encoder bundle
    manifest["artifacts"] = slots
    write_manifest(manifest, target_dir / settings.manifest_filename)


def format_report(summary: Dict[str, Any]) -> str:
    """Human-readable summary printed after fitting."""
    lines = [
        "",
        "=" * 74,
        " Feature pipeline and entity baselines",
        "=" * 74,
        f" fitted on          : {summary['split']} split "
        f"({summary['n_events']:,} events, {summary['n_entities']} entities)",
        f" features           : {summary['n_features']} "
        f"({summary['n_numeric']} numeric + {summary['n_categorical']} categorical)",
        f" command vocabulary : {summary['vocab_size']} tokens",
        f" cold-start profiles: {summary['n_cold_start_profiles']} of {summary['n_entities']}",
        "",
        f" behavioral cohorts : {summary['n_cohorts']}",
        " " + "-" * 60,
    ]
    for cohort, size in summary["cohort_sizes"].items():
        label = summary["cohort_labels"].get(cohort, "")
        lines.append(f"   cohort {cohort}  {size:>4} entities   {label}")

    lines += [
        "",
        " default cohort per entity type (used before any behavior is seen)",
        " " + "-" * 60,
    ]
    for entity_type, cohort in sorted(summary["type_cohorts"].items()):
        lines.append(f"   {entity_type:<20} -> cohort {cohort}")

    lines += ["", " artifacts written", " " + "-" * 60]
    for name, path in summary["artifacts"].items():
        lines.append(f"   {name:<16} {Path(path).name}")
    lines += ["=" * 74, ""]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.build_baselines",
        description="Fit the feature pipeline, entity baselines and behavioral cohorts.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test"),
        help="Split to fit on. Always 'train' outside of debugging.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    try:
        summary = build(
            split=args.split,
            dataset_dir=args.dataset_dir,
            artifacts_dir=args.artifacts_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.quiet:
        print(format_report(summary))

    # Fail loudly in dev: a pipeline missing cohorts or a scaler would silently degrade every
    # downstream tier, so refuse to report success.
    if summary["n_cohorts"] < 2:
        logger.error("Only %d cohort(s) fitted; cold-start priors need at least 2.", summary["n_cohorts"])
        return 1
    if summary["n_features"] != len(NUMERIC_FEATURE_NAMES) + len(CATEGORICAL_FEATURE_NAMES):
        logger.error("Feature count mismatch: %d", summary["n_features"])
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
