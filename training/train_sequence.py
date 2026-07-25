"""Train the Tier 2 GRU sequence model and write it to ``artifacts/``.

Run it with::

    python -m training.train_sequence

Prerequisite: ``python -m training.build_baselines`` has fitted the feature pipeline, whose
sequence vocabulary this model reuses. This script never refits the vocabulary.

Command-sequence encoding is **stateless** -- it depends only on the fitted vocabulary, not on any
entity's rolling history -- so unlike the baseline there is no leakage distinction between the
training and validation paths. Each event's ``command_sequence`` is encoded exactly the way the
online scorer encodes it (``vocab.encode`` producing the same left-padded, ``<bos>``-prefixed ids
carried on every :class:`FeatureVector`), which keeps train/serve parity by construction.

Labels are loaded for evaluation only. The model is trained unsupervised on the (mostly-normal)
training split. The report highlights recall on ``lateral_movement`` and ``low_and_slow_exfil``,
the two classes whose signal is most about *order and breadth of actions* -- exactly what a
sequence model is meant to add over the tabular baseline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.artifacts import read_manifest, write_manifest
from common.config import settings
from common.models import Event
from common.seed import set_global_seed
from data_generator.generate import dataframe_to_events, load_events, load_labels
from features.featurize import FeaturePipeline
from features.sequences import SequenceVocab
from models.sequence import SEQUENCE_FILE, SequenceModel, SequenceTrainConfig

logger = logging.getLogger(__name__)

#: Classes whose signal is primarily about the ordering/breadth of actions -- the sequence
#: model's reason to exist. Recall on these is reported explicitly.
SEQUENCE_SENSITIVE_CLASSES = ("lateral_movement", "low_and_slow_exfil")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_split(split: str, dataset_dir: Optional[Path] = None) -> List[Event]:
    """Load one split's events as :class:`Event` objects in time order (labels never loaded)."""
    frame = load_events(dataset_dir)
    subset = frame[frame["split"] == split]
    if subset.empty:
        raise ValueError(
            f"No events in split '{split}'. Run: python -m data_generator.generate --seed 42"
        )
    subset = subset.sort_values(["timestamp", "event_id"])
    return dataframe_to_events(subset)


def load_label_map(split: str, dataset_dir: Optional[Path] = None) -> Dict[str, str]:
    """Map ``event_id -> label`` (class name) for one split, for evaluation only."""
    labels = load_labels(dataset_dir)
    subset = labels[labels["split"] == split]
    return {
        str(event_id): str(label)
        for event_id, label in zip(subset["event_id"], subset["label"])
    }


def encode_sequences(events: Sequence[Event], vocab: SequenceVocab) -> np.ndarray:
    """Encode each event's command sequence to a fixed-length id row (as the scorer would)."""
    if not events:
        return np.zeros((0, vocab.max_len), dtype=np.int64)
    return np.asarray([vocab.encode(event.command_sequence) for event in events], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_scores(
    scores: np.ndarray,
    labels: Sequence[str],
    budget_pct: float,
) -> Dict[str, Any]:
    """Overall ranking metrics plus per-class recall at the alert budget."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, dtype=float)
    label_arr = np.asarray(list(labels), dtype=object)
    y = (label_arr != "normal").astype(int)
    n = y.size
    positives = int(y.sum())
    prevalence = positives / n if n else 0.0

    metrics: Dict[str, Any] = {
        "n": float(n),
        "n_positives": float(positives),
        "prevalence": prevalence,
    }
    if positives == 0 or positives == n:
        metrics.update({"pr_auc": float("nan"), "roc_auc": float("nan"), "recall_at_budget": float("nan")})
        return metrics

    metrics["pr_auc"] = float(average_precision_score(y, scores))
    metrics["roc_auc"] = float(roc_auc_score(y, scores))
    metrics["pr_auc_uplift"] = metrics["pr_auc"] / prevalence if prevalence else float("nan")

    k = max(1, int(round(n * budget_pct)))
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    in_budget = np.zeros(n, dtype=bool)
    in_budget[top_idx] = True

    metrics["budget_k"] = float(k)
    metrics["recall_at_budget"] = float(y[top_idx].sum() / positives)

    # Per-class recall within the same alert budget -- how many of each attack class's events land
    # in the top-k by sequence surprise.
    per_class: Dict[str, Dict[str, float]] = {}
    for cls in sorted(set(label_arr.tolist())):
        if cls == "normal":
            continue
        cls_mask = label_arr == cls
        total = int(cls_mask.sum())
        caught = int((cls_mask & in_budget).sum())
        per_class[cls] = {
            "n": float(total),
            "recall_at_budget": float(caught / total) if total else float("nan"),
        }
    metrics["per_class"] = per_class
    return metrics


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def train_sequence(
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    config: Optional[SequenceTrainConfig] = None,
) -> Dict[str, Any]:
    """Train the GRU sequence model, evaluate on validation, and persist it."""
    set_global_seed()
    config = config or SequenceTrainConfig()
    target_dir = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)

    logger.info("Loading fitted feature pipeline (for the sequence vocabulary) from %s", target_dir)
    pipeline = FeaturePipeline.load(target_dir)
    vocab = pipeline.vocab
    if vocab.size <= len(("<pad>", "<unk>", "<bos>")):
        raise ValueError(
            "Sequence vocabulary is empty. Run: python -m training.build_baselines"
        )

    logger.info("Encoding the training split's command sequences (vocab size %d)", vocab.size)
    train_events = load_split("train", dataset_dir)
    train_matrix = encode_sequences(train_events, vocab)
    logger.info("Training matrix: %s", train_matrix.shape)

    logger.info("Training the GRU")
    model = SequenceModel.train(train_matrix, vocab, config)

    logger.info("Encoding and scoring the validation split")
    val_events = load_split("val", dataset_dir)
    val_matrix = encode_sequences(val_events, vocab)
    label_map = load_label_map("val", dataset_dir)
    val_labels = [label_map.get(event.event_id, "normal") for event in val_events]
    val_scores = np.atleast_1d(model.score_sequence(val_matrix))

    metrics = evaluate_scores(val_scores, val_labels, settings.alert_budget_pct)

    path = model.save(target_dir / SEQUENCE_FILE)
    _update_manifest(path, target_dir)

    summary: Dict[str, Any] = {
        "n_train_events": len(train_events),
        "n_val_events": len(val_events),
        "vocab_size": vocab.size,
        "max_len": model.max_len,
        "normalizer_center": model.normalizer.center,
        "normalizer_scale": model.normalizer.scale,
        "val_metrics": metrics,
        "artifact": str(path),
    }
    return summary


def _update_manifest(path: Path, target_dir: Path) -> None:
    """Record the sequence model in the manifest without disturbing other slots."""
    manifest_path = target_dir / settings.manifest_filename
    manifest = read_manifest(manifest_path)
    slots = manifest.get("artifacts") or {}
    slots["sequence_model"] = Path(path).name
    manifest["artifacts"] = slots
    write_manifest(manifest, manifest_path)


def format_report(summary: Dict[str, Any]) -> str:
    """Human-readable summary printed after training."""
    metrics = summary["val_metrics"]

    def fmt(value: Any, spec: str = ".4f") -> str:
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "",
        "=" * 74,
        " GRU sequence model (Tier 2, Deliverable #3)",
        "=" * 74,
        f" trained on         : {summary['n_train_events']:,} train events",
        f" vocabulary / steps : {summary['vocab_size']} tokens / {summary['max_len']} max len",
        f" normalizer         : center={fmt(summary['normalizer_center'], '.5f')}  "
        f"scale={fmt(summary['normalizer_scale'], '.5f')}",
        "",
        f" validation ({int(metrics['n']):,} events, "
        f"{int(metrics['n_positives'])} anomalies, "
        f"prevalence {fmt(metrics['prevalence'], '.4%')})",
        " " + "-" * 60,
        f"   PR-AUC             : {fmt(metrics.get('pr_auc'))}  "
        f"(random {fmt(metrics.get('prevalence'))}, uplift x{fmt(metrics.get('pr_auc_uplift'), '.1f')})",
        f"   ROC-AUC            : {fmt(metrics.get('roc_auc'))}",
        f"   recall @ {settings.alert_budget_pct:.0%} budget : {fmt(metrics.get('recall_at_budget'))}",
        "",
        " per-class recall @ budget (sequence-sensitive classes highlighted)",
        " " + "-" * 60,
    ]
    per_class = metrics.get("per_class", {})
    for cls in sorted(per_class):
        marker = "  <-- sequence-sensitive" if cls in SEQUENCE_SENSITIVE_CLASSES else ""
        row = per_class[cls]
        lines.append(
            f"   {cls:<22} {fmt(row['recall_at_budget'])}  "
            f"(n={int(row['n'])}){marker}"
        )

    lines += [
        "",
        f" artifact           : {Path(summary['artifact']).name}",
        "=" * 74,
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_sequence",
        description="Train the Tier 2 GRU next-event sequence model over command sequences.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Artifacts directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    config = SequenceTrainConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    try:
        summary = train_sequence(
            dataset_dir=args.dataset_dir,
            artifacts_dir=args.artifacts_dir,
            config=config,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.quiet:
        print(format_report(summary))

    # Fail loudly if the sequence model shows no ranking signal at all: a PR-AUC that does not clear
    # the random floor means the tier is broken, not merely weak.
    metrics = summary["val_metrics"]
    pr_auc = metrics.get("pr_auc")
    prevalence = metrics.get("prevalence", 0.0)
    if pr_auc is not None and prevalence and not (pr_auc > prevalence):
        logger.error(
            "Sequence PR-AUC %.4f does not clear the random floor %.4f; refusing to report success.",
            pr_auc,
            prevalence,
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
