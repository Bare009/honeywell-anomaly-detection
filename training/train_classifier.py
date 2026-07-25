"""Train the anomaly-type classifier and fit risk fusion.

Run it with::

    python -m training.train_classifier

Prerequisites: the feature pipeline (``build_baselines``), the baseline autoencoder
(``train_baseline``) and the GRU (``train_sequence``) are all built, because the classifier consumes
their scores as features and the risk layer fuses all three.

The flow mirrors the rest of the training plane. Training features are built **leakage-free** (fitted
transforms, empty profiles, streamed in time order); validation features are built the way serving
builds them (against the persisted profiles). The classifier is supervised, so it sees labels on the
training split; the two unsupervised tiers never did. Fusion weights, calibration and the alert-budget
threshold are all tuned on validation -- the designated tuning split -- and the honest test-set numbers
come later in Phase 9.

What it writes: the calibrated classifier (``classifier.json``) and the tuned risk model
(``risk_model.json``), and the corresponding manifest slots.
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
from common.models import ANOMALY_CLASS_INDEX, ANOMALY_CLASSES, AnomalyType
from common.seed import set_global_seed
from features.featurize import FeaturePipeline
from models.baseline import BaselineModel
from models.calibration import expected_calibration_error
from models.classifier import CLASSIFIER_FILE, ClassifierModel, ClassifierTrainConfig, assemble_matrix
from models.detectors import DetectorBank, attack_probability, resolve_anomaly_type
from models.risk import RISK_FILE, RiskModel
from models.sequence import SequenceModel
from training.train_baseline import (
    featurize_serving_split,
    featurize_training_split,
    load_split,
)
from training.train_sequence import load_label_map

logger = logging.getLogger(__name__)

_NORMAL = AnomalyType.NORMAL.value
_NORMAL_INDEX = ANOMALY_CLASS_INDEX[_NORMAL]


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #


def tier_scores(
    vectors: Sequence[Any],
    baseline: BaselineModel,
    sequence: SequenceModel,
) -> Dict[str, np.ndarray]:
    """Baseline and sequence scores for a list of feature vectors."""
    return {
        "baseline": np.atleast_1d(baseline.score_baseline(vectors)),
        "sequence": np.atleast_1d(sequence.score_sequence(vectors)),
    }


def classifier_matrix(
    vectors: Sequence[Any], baseline_scores: np.ndarray, sequence_scores: np.ndarray
) -> np.ndarray:
    """Assemble the classifier's feature matrix: pipeline features plus the two tier scores."""
    pipeline_matrix = FeaturePipeline.to_matrix(vectors)
    return assemble_matrix(pipeline_matrix, baseline_scores, sequence_scores)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    risk: RiskModel,
    classifier: ClassifierModel,
    vectors: Sequence[Any],
    baseline_scores: np.ndarray,
    sequence_scores: np.ndarray,
    attack_probs: np.ndarray,
    class_probs: List[Dict[str, float]],
    class_types: List[str],
    true_labels: Sequence[str],
    budget_pct: float,
) -> Dict[str, Any]:
    """Compute macro-F1, calibration ECE, recall@budget, detector precision and uncertainty."""
    from sklearn.metrics import f1_score

    bank = DetectorBank()
    detector_results = [bank.evaluate(vector.raw) for vector in vectors]

    y_true_label = np.asarray(list(true_labels), dtype=object)
    y_anom = (y_true_label != _NORMAL).astype(int)
    n = y_anom.size
    positives = int(y_anom.sum())

    # --- resolved type predictions (classifier + confident detector overrides) ---
    predicted: List[str] = []
    for base_type, probs, results in zip(class_types, class_probs, detector_results):
        resolved, _, _ = resolve_anomaly_type(AnomalyType(base_type), probs, results)
        predicted.append(resolved.value)
    macro_f1 = float(
        f1_score(list(y_true_label), predicted, labels=ANOMALY_CLASSES, average="macro", zero_division=0)
    )
    # A class absent from validation contributes an unavoidable F1 of 0 to the 9-class macro, which
    # caps it regardless of model quality. Report the macro over the classes actually present too,
    # so the number reflects the model rather than the split's composition; Phase 9 evaluates the
    # full 9-class macro on the test split, which does contain every class.
    present_classes = sorted(set(y_true_label.tolist()))
    macro_f1_present = float(
        f1_score(list(y_true_label), predicted, labels=present_classes, average="macro", zero_division=0)
    )
    missing_classes = [cls for cls in ANOMALY_CLASSES if cls not in present_classes]

    # --- fused risk ---
    fused = risk.fuse(baseline_scores, sequence_scores, attack_probs)
    risk_scores = risk.to_risk(fused)
    probability = risk_scores / 100.0

    ece = expected_calibration_error(probability, y_anom, n_bins=10)

    k = max(1, int(round(n * budget_pct)))
    top = np.argsort(risk_scores)[::-1][:k]
    recall_at_budget = float(y_anom[top].sum() / positives) if positives else float("nan")
    in_budget = risk_scores >= risk.budget_threshold
    budget_fraction = float(in_budget.mean())

    # --- deterministic detector precision on their target classes ---
    # Two views: "type" precision (fired on exactly this class) and "anomaly" precision (fired on
    # any true anomaly). The gap is campaign-stage labelling -- a genuinely impossible hop inside a
    # multi-stage campaign carries that stage's label, not `impossible_travel` -- so anomaly
    # precision is the safety-relevant number, and it is what should be near 1.0.
    detector_precision: Dict[str, Dict[str, float]] = {}
    target = {"impossible_travel": "impossible_travel", "brute_force": "brute_force"}
    for name, cls in target.items():
        fired = np.array(
            [any(r.name == name and r.fired for r in results) for results in detector_results]
        )
        n_fired = int(fired.sum())
        if n_fired:
            correct_type = int(((y_true_label == cls) & fired).sum())
            on_anomaly = int(((y_true_label != _NORMAL) & fired).sum())
            detector_precision[name] = {
                "type_precision": correct_type / n_fired,
                "anomaly_precision": on_anomaly / n_fired,
                "n_fired": float(n_fired),
                "recall": float(correct_type / max(1, int((y_true_label == cls).sum()))),
            }
        else:
            detector_precision[name] = {
                "type_precision": float("nan"),
                "anomaly_precision": float("nan"),
                "n_fired": 0.0,
                "recall": 0.0,
            }

    # --- uncertainty: cold-start should be wider ---
    cold = np.array([1.0 if getattr(v, "cold_start", False) else 0.0 for v in vectors])
    bands = risk.uncertainty(baseline_scores, sequence_scores, attack_probs, cold)
    cold_mask = cold >= 0.5
    uncertainty_cold = float(bands[cold_mask].mean()) if cold_mask.any() else float("nan")
    uncertainty_warm = float(bands[~cold_mask].mean()) if (~cold_mask).any() else float("nan")

    # --- per-class recall within budget (for the report) ---
    per_class: Dict[str, Dict[str, float]] = {}
    for cls in ANOMALY_CLASSES:
        if cls == _NORMAL:
            continue
        cls_mask = y_true_label == cls
        total = int(cls_mask.sum())
        caught = int((cls_mask & in_budget).sum())
        per_class[cls] = {
            "n": float(total),
            "recall_at_budget": float(caught / total) if total else float("nan"),
        }

    return {
        "n": float(n),
        "n_positives": float(positives),
        "prevalence": float(positives / n) if n else 0.0,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "missing_classes": missing_classes,
        "calibration_ece": ece,
        "recall_at_budget": recall_at_budget,
        "budget_fraction": budget_fraction,
        "detector_precision": detector_precision,
        "uncertainty_cold": uncertainty_cold,
        "uncertainty_warm": uncertainty_warm,
        "per_class": per_class,
        "weights": dict(risk.weights),
        "budget_threshold": risk.budget_threshold,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def train_classifier(
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    config: Optional[ClassifierTrainConfig] = None,
) -> Dict[str, Any]:
    """Train the classifier, tune risk fusion on validation, and persist both."""
    set_global_seed()
    config = config or ClassifierTrainConfig()
    target_dir = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)

    logger.info("Loading pipeline and the baseline/sequence models from %s", target_dir)
    pipeline = FeaturePipeline.load(target_dir)
    baseline = BaselineModel.load(target_dir / "baseline_model.json")
    sequence = SequenceModel.load(target_dir / "sequence_model.json")

    feature_names = list(pipeline.feature_names) + ["baseline_score", "sequence_score"]
    categorical_indices = list(pipeline.categorical_indices)

    # --- training features (leakage-free) + labels ---
    logger.info("Featurizing the training split (leakage-free stream)")
    train_events = load_split("train", dataset_dir)
    train_vectors = featurize_training_split(pipeline, train_events)
    train_scores = tier_scores(train_vectors, baseline, sequence)
    x_train = classifier_matrix(train_vectors, train_scores["baseline"], train_scores["sequence"])

    train_label_map = load_label_map("train", dataset_dir)
    train_labels = [train_label_map.get(vector.event_id, _NORMAL) for vector in train_vectors]

    logger.info("Training the LightGBM classifier (%s)", x_train.shape)
    classifier = ClassifierModel.train(
        x_train, train_labels, feature_names, categorical_indices, config
    )

    # --- validation features (serving-like) ---
    logger.info("Featurizing the validation split (serving-like) and scoring all tiers")
    val_events = load_split("val", dataset_dir)
    val_vectors = featurize_serving_split(pipeline, val_events)
    val_scores = tier_scores(val_vectors, baseline, sequence)
    x_val = classifier_matrix(val_vectors, val_scores["baseline"], val_scores["sequence"])

    val_label_map = load_label_map("val", dataset_dir)
    val_labels = [val_label_map.get(vector.event_id, _NORMAL) for vector in val_vectors]
    y_val = np.array([0 if label == _NORMAL else 1 for label in val_labels], dtype=int)

    class_types, class_probs = classifier.classify_matrix(x_val)
    attack_probs = np.array([attack_probability(probs) for probs in class_probs])

    # --- tune fusion on validation ---
    logger.info("Tuning risk fusion weights, calibration and the alert-budget threshold")
    risk = RiskModel.tune(
        val_scores["baseline"],
        val_scores["sequence"],
        attack_probs,
        y_val,
        budget_pct=settings.alert_budget_pct,
    )

    metrics = evaluate(
        risk,
        classifier,
        val_vectors,
        val_scores["baseline"],
        val_scores["sequence"],
        attack_probs,
        class_probs,
        class_types,
        val_labels,
        settings.alert_budget_pct,
    )

    classifier_path = classifier.save(target_dir / CLASSIFIER_FILE)
    risk_path = risk.save(target_dir / RISK_FILE)
    _update_manifest(classifier_path, risk_path, target_dir)

    return {
        "n_train_events": len(train_events),
        "n_val_events": len(val_events),
        "n_features": len(feature_names),
        "val_metrics": metrics,
        "classifier_artifact": str(classifier_path),
        "risk_artifact": str(risk_path),
    }


def _update_manifest(classifier_path: Path, risk_path: Path, target_dir: Path) -> None:
    """Record the classifier and risk model in the manifest without disturbing other slots."""
    manifest_path = target_dir / settings.manifest_filename
    manifest = read_manifest(manifest_path)
    slots = manifest.get("artifacts") or {}
    slots["classifier"] = Path(classifier_path).name
    slots["calibrator"] = Path(classifier_path).name  # per-class calibration lives in the classifier
    slots["fusion"] = Path(risk_path).name
    slots["thresholds"] = Path(risk_path).name
    manifest["artifacts"] = slots
    write_manifest(manifest, manifest_path)


def format_report(summary: Dict[str, Any]) -> str:
    """Human-readable summary comparing measured metrics to the plan's targets."""
    m = summary["val_metrics"]

    def fmt(value: Any, spec: str = ".4f") -> str:
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)

    def gate(value: Any, target: float, good_high: bool = True) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "?"
        ok = v >= target if good_high else v <= target
        return "PASS" if ok else "MISS"

    weights = m["weights"]
    lines = [
        "",
        "=" * 74,
        " Classifier + risk fusion (Tier 3, Deliverable #4; D4/D5)",
        "=" * 74,
        f" trained on         : {summary['n_train_events']:,} train events, "
        f"{summary['n_features']} features",
        f" fusion weights     : baseline {fmt(weights['baseline'], '.2f')}  "
        f"sequence {fmt(weights['sequence'], '.2f')}  classifier {fmt(weights['classifier'], '.2f')}",
        f" budget threshold   : risk >= {fmt(m['budget_threshold'], '.2f')}",
        "",
        f" validation ({int(m['n']):,} events, {int(m['n_positives'])} anomalies, "
        f"prevalence {fmt(m['prevalence'], '.4%')})",
        " " + "-" * 66,
        f"   macro-F1 (9-class): {fmt(m['macro_f1'])}   target >= 0.85   [{gate(m['macro_f1'], 0.85)}]"
        + (f"   (missing in val: {', '.join(m['missing_classes'])})" if m.get("missing_classes") else ""),
        f"   macro-F1 (present): {fmt(m['macro_f1_present'])}   target >= 0.85   "
        f"[{gate(m['macro_f1_present'], 0.85)}]",
        f"   calibration ECE   : {fmt(m['calibration_ece'])}   target <= 0.05   "
        f"[{gate(m['calibration_ece'], 0.05, good_high=False)}]",
        f"   recall @ {settings.alert_budget_pct:.0%} budget: {fmt(m['recall_at_budget'])}   "
        f"target >= 0.80   [{gate(m['recall_at_budget'], 0.80)}]",
        f"   budget volume     : {fmt(m['budget_fraction'], '.4%')}   target ~ "
        f"{settings.alert_budget_pct:.0%}",
        "",
        " deterministic detectors (precision on their target class)",
        " " + "-" * 66,
    ]
    for name, stats in m["detector_precision"].items():
        lines.append(
            f"   {name:<20} anomaly-precision {fmt(stats['anomaly_precision'])}  "
            f"type-precision {fmt(stats['type_precision'])}  (fired {int(stats['n_fired'])})"
        )

    lines += [
        "",
        f" uncertainty band   : cold-start {fmt(m['uncertainty_cold'], '.2f')}  vs  "
        f"established {fmt(m['uncertainty_warm'], '.2f')}   "
        f"[{'PASS' if (m['uncertainty_cold'] or 0) > (m['uncertainty_warm'] or 0) else 'MISS'}]",
        "",
        " per-class recall @ budget",
        " " + "-" * 66,
    ]
    for cls in sorted(m["per_class"]):
        row = m["per_class"][cls]
        lines.append(f"   {cls:<22} {fmt(row['recall_at_budget'])}  (n={int(row['n'])})")

    lines += [
        "",
        f" artifacts          : {Path(summary['classifier_artifact']).name}, "
        f"{Path(summary['risk_artifact']).name}",
        "=" * 74,
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_classifier",
        description="Train the anomaly-type classifier and tune risk fusion.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Artifacts directory.")
    parser.add_argument("--num-boost-round", type=int, default=None, help="LightGBM rounds.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    config = ClassifierTrainConfig()
    if args.num_boost_round is not None:
        config.num_boost_round = args.num_boost_round

    try:
        summary = train_classifier(
            dataset_dir=args.dataset_dir,
            artifacts_dir=args.artifacts_dir,
            config=config,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.quiet:
        print(format_report(summary))

    # Fail loudly if the fused detector shows no budgeted signal: something upstream is broken.
    metrics = summary["val_metrics"]
    recall = metrics.get("recall_at_budget")
    if recall is not None and not (recall > 0.5):
        logger.error("Recall@budget %.4f is implausibly low; refusing to report success.", recall)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
