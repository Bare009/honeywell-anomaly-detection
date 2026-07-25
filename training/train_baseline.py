"""Train the Tier 1 baseline autoencoder and write it to ``artifacts/``.

Run it with::

    python -m training.train_baseline

Prerequisite: ``python -m training.build_baselines`` has already fitted the feature pipeline
(encoders, scaler, profiles, cohorts, vocabulary). This script loads that pipeline and never
refits it.

Two featurization paths are used deliberately, and they are not the same.

**Training features are built leakage-free**, exactly the way ``build_baselines`` built them: a
fresh pipeline with the fitted transforms but an *empty* profile store, streamed in time order so
each event is compared only against strictly earlier history. Loading the persisted profiles here
instead would compare every training event against a baseline that already contains it -- the
autoencoder would learn from features no online scorer could ever reproduce.

**Validation features are built the way serving builds them**: against the persisted (train-fitted)
profiles plus live accumulation over the validation stream. Validation entities are absent from the
train profiles, so there is no leakage, and the resulting feature distribution is exactly what the
deployed model will see. That is the honest surface on which to report PR-AUC.

Only aggregate metrics are quoted. Labels are loaded here for evaluation only -- never for training,
which is unsupervised.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.artifacts import read_manifest, write_manifest
from common.config import settings
from common.models import Event
from common.seed import set_global_seed
from data_generator.generate import dataframe_to_events, load_events, load_labels
from features.event_features import NUMERIC_FEATURE_NAMES
from features.entity_window import ProfileStore
from features.featurize import FeaturePipeline, FeatureVector
from models.baseline import BASELINE_FILE, BaselineModel, BaselineTrainConfig
from training.build_baselines import streaming_pass

logger = logging.getLogger(__name__)


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


def load_anomaly_flags(split: str, dataset_dir: Optional[Path] = None) -> Dict[str, bool]:
    """Map ``event_id -> is_anomaly`` for one split, for evaluation only."""
    labels = load_labels(dataset_dir)
    subset = labels[labels["split"] == split]
    return {
        str(event_id): bool(is_anomaly)
        for event_id, is_anomaly in zip(subset["event_id"], subset["is_anomaly"])
    }


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #


def numeric_matrix(vectors: Sequence[FeatureVector], numeric_names: Sequence[str]) -> np.ndarray:
    """Stack the leading numeric block (the autoencoder's input) from feature vectors."""
    width = len(numeric_names)
    if not vectors:
        return np.zeros((0, width), dtype=float)
    return np.vstack([vector.values[:width] for vector in vectors])


def featurize_training_split(
    fitted: FeaturePipeline, events: Sequence[Event]
) -> List[FeatureVector]:
    """Leakage-free training features: fitted transforms, empty profiles, streamed in time order."""
    stream_pipeline = FeaturePipeline(
        encoders=fitted.encoders,
        vocab=fitted.vocab,
        profiles=ProfileStore(),
        cohorts=fitted.cohorts,
        corpus=fitted.corpus,
    )
    vectors, _ = streaming_pass(stream_pipeline, events)
    return vectors


def featurize_serving_split(
    fitted: FeaturePipeline, events: Sequence[Event]
) -> List[FeatureVector]:
    """Serving-like features: against the persisted profiles plus live accumulation."""
    return fitted.featurize_events(events, reset=True, use_live_profile=True)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_scores(
    scores: np.ndarray, y_true: np.ndarray, budget_pct: float
) -> Dict[str, float]:
    """PR-AUC, ROC-AUC and recall at the alert budget, plus the random-baseline reference."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = y_true.size
    positives = int(y_true.sum())
    prevalence = positives / n if n else 0.0

    metrics: Dict[str, float] = {
        "n": float(n),
        "n_positives": float(positives),
        "prevalence": prevalence,
    }
    if positives == 0 or positives == n:
        # Degenerate label vector: ranking metrics are undefined. Report NaN honestly.
        metrics.update({"pr_auc": float("nan"), "roc_auc": float("nan"), "recall_at_budget": float("nan")})
        return metrics

    metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    metrics["roc_auc"] = float(roc_auc_score(y_true, scores))

    # Recall within the top-`budget_pct` events by score -- the analyst alert budget (D4).
    k = max(1, int(round(n * budget_pct)))
    top_idx = np.argsort(scores)[::-1][:k]
    metrics["recall_at_budget"] = float(y_true[top_idx].sum() / positives)
    metrics["budget_k"] = float(k)
    metrics["pr_auc_uplift"] = metrics["pr_auc"] / prevalence if prevalence else float("nan")
    return metrics


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def train_baseline(
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    config: Optional[BaselineTrainConfig] = None,
) -> Dict[str, Any]:
    """Train the baseline autoencoder, evaluate on validation, and persist it."""
    set_global_seed()
    config = config or BaselineTrainConfig()
    target_dir = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)

    logger.info("Loading fitted feature pipeline from %s", target_dir)
    fitted = FeaturePipeline.load(target_dir)
    numeric_names = list(fitted.numeric_names) or list(NUMERIC_FEATURE_NAMES)

    logger.info("Featurizing the training split (leakage-free stream)")
    train_events = load_split("train", dataset_dir)
    train_vectors = featurize_training_split(fitted, train_events)
    train_matrix = numeric_matrix(train_vectors, numeric_names)
    logger.info("Training matrix: %s", train_matrix.shape)

    logger.info("Training the autoencoder")
    model = BaselineModel.train(train_matrix, numeric_names, config)

    logger.info("Featurizing the validation split (serving-like)")
    val_events = load_split("val", dataset_dir)
    val_vectors = featurize_serving_split(fitted, val_events)
    val_matrix = numeric_matrix(val_vectors, numeric_names)

    flags = load_anomaly_flags("val", dataset_dir)
    y_val = np.array([1 if flags.get(v.event_id, False) else 0 for v in val_vectors], dtype=int)
    val_scores = np.atleast_1d(model.score_baseline(val_matrix))

    metrics = evaluate_scores(val_scores, y_val, settings.alert_budget_pct)

    # Training-split score distribution, for the report and a sanity check on the normalizer.
    train_scores = np.atleast_1d(model.score_baseline(train_matrix))

    path = model.save(target_dir / BASELINE_FILE)
    _update_manifest(path, target_dir)

    summary: Dict[str, Any] = {
        "n_train_events": len(train_events),
        "n_val_events": len(val_events),
        "input_dim": model.input_dim,
        "normalizer_center": model.normalizer.center,
        "normalizer_scale": model.normalizer.scale,
        "train_score_mean": float(train_scores.mean()) if train_scores.size else float("nan"),
        "train_score_p95": float(np.quantile(train_scores, 0.95)) if train_scores.size else float("nan"),
        "val_metrics": metrics,
        "artifact": str(path),
    }
    return summary


def _update_manifest(path: Path, target_dir: Path) -> None:
    """Record the baseline model in the manifest without disturbing other slots."""
    manifest_path = target_dir / settings.manifest_filename
    manifest = read_manifest(manifest_path)
    slots = manifest.get("artifacts") or {}
    slots["baseline_model"] = Path(path).name
    slots["autoencoder"] = Path(path).name
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
        " Baseline autoencoder (Tier 1, Deliverable #2)",
        "=" * 74,
        f" trained on         : {summary['n_train_events']:,} train events "
        f"(input dim {summary['input_dim']})",
        f" normalizer         : center={fmt(summary['normalizer_center'], '.5f')}  "
        f"scale={fmt(summary['normalizer_scale'], '.5f')}",
        f" train score        : mean {fmt(summary['train_score_mean'])}  "
        f"p95 {fmt(summary['train_score_p95'])}",
        "",
        f" validation ({int(metrics['n']):,} events, "
        f"{int(metrics['n_positives'])} anomalies, "
        f"prevalence {fmt(metrics['prevalence'], '.4%')})",
        " " + "-" * 60,
        f"   PR-AUC             : {fmt(metrics.get('pr_auc'))}",
        f"   random PR-AUC      : {fmt(metrics.get('prevalence'))}  "
        f"(uplift x{fmt(metrics.get('pr_auc_uplift'), '.1f')})",
        f"   ROC-AUC            : {fmt(metrics.get('roc_auc'))}",
        f"   recall @ {settings.alert_budget_pct:.0%} budget : "
        f"{fmt(metrics.get('recall_at_budget'))}  "
        f"(top {int(metrics.get('budget_k', 0))} events)",
        "",
        f" artifact           : {Path(summary['artifact']).name}",
        "=" * 74,
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_baseline",
        description="Train the Tier 1 baseline autoencoder over entity-relative features.",
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

    config = BaselineTrainConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    try:
        summary = train_baseline(
            dataset_dir=args.dataset_dir,
            artifacts_dir=args.artifacts_dir,
            config=config,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.quiet:
        print(format_report(summary))

    # Fail loudly in dev if the baseline shows no signal: a PR-AUC that does not clear the random
    # floor means something is wrong upstream, and every later tier would inherit the problem.
    metrics = summary["val_metrics"]
    pr_auc = metrics.get("pr_auc")
    prevalence = metrics.get("prevalence", 0.0)
    if pr_auc is not None and prevalence and not (pr_auc > 2.0 * prevalence):
        logger.error(
            "Baseline PR-AUC %.4f does not clear twice the random floor %.4f; refusing to report success.",
            pr_auc,
            prevalence,
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
