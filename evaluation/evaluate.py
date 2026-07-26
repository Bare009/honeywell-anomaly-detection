"""End-to-end evaluation on the held-out test split.

Computes every headline metric the report quotes -- and computes them the imbalance-aware way:
PR-AUC and recall within the alert budget lead, ROC-AUC and accuracy are reported but never leaned
on (at ~1% prevalence they flatter). Writes ``artifacts/metrics.json`` and, if MongoDB is reachable,
persists a ``model_metrics`` document so the dashboard's performance page fills in.

Run it with::

    python -m evaluation.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.artifacts import read_manifest
from common.config import settings
from common.models import (
    ANOMALY_CLASSES,
    AnomalyType,
    DatasetSummary,
    ModelMetrics,
)
from common.seed import set_global_seed
from models.calibration import expected_calibration_error
from evaluation.scoring import ScoredSplit, load_models, score_split

logger = logging.getLogger(__name__)

METRICS_FILE = "metrics.json"
_NORMAL = AnomalyType.NORMAL.value


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #


def recall_at_budget(risk: np.ndarray, y_true: np.ndarray, budget: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    k = max(1, int(round(len(risk) * budget)))
    top = np.argsort(risk)[::-1][:k]
    return float(y_true[top].sum() / positives)


def precision_at_k_curve(risk: np.ndarray, y_true: np.ndarray, points: int = 200) -> List[float]:
    """Cumulative precision as the top-k grows -- the alert-budget curve (D4)."""
    order = np.argsort(risk)[::-1]
    hits = np.cumsum(y_true[order])
    ks = np.arange(1, len(order) + 1)
    precision = hits / ks
    if len(precision) <= points:
        return [float(p) for p in precision]
    idx = np.linspace(0, len(precision) - 1, points).astype(int)
    return [float(precision[i]) for i in idx]


def per_class_metrics(labels: List[str], predicted: List[str]) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=list(ANOMALY_CLASSES), zero_division=0
    )
    return {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(ANOMALY_CLASSES)
    }


def detector_precision(scored: ScoredSplit) -> Dict[str, Dict[str, float]]:
    labels = np.asarray(scored.labels, dtype=object)
    out: Dict[str, Dict[str, float]] = {}
    for name, cls in (("impossible_travel", "impossible_travel"), ("brute_force", "brute_force")):
        fired = np.array(
            [any(r.name == name and r.fired for r in results) for results in scored.detector_results]
        )
        n_fired = int(fired.sum())
        if n_fired == 0:
            out[name] = {"anomaly_precision": float("nan"), "type_precision": float("nan"), "n_fired": 0.0}
            continue
        out[name] = {
            "anomaly_precision": float(((labels[fired] != _NORMAL)).mean()),
            "type_precision": float((labels[fired] == cls).mean()),
            "n_fired": float(n_fired),
        }
    return out


def compute_metrics(scored: ScoredSplit) -> Dict[str, Any]:
    """All headline metrics for one scored split."""
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

    y = scored.y_true
    risk = scored.risk
    prob = risk / 100.0
    labels = scored.labels
    predicted = scored.predicted_type
    positives = int(y.sum())
    n = int(y.size)

    present = sorted(set(labels))
    missing = [c for c in ANOMALY_CLASSES if c not in present]

    metrics: Dict[str, Any] = {
        "n": n,
        "prevalence": float(positives / n) if n else 0.0,
        "pr_auc": float(average_precision_score(y, risk)) if 0 < positives < n else float("nan"),
        "roc_auc": float(roc_auc_score(y, risk)) if 0 < positives < n else float("nan"),
        "recall_at_1pct_budget": recall_at_budget(risk, y, settings.alert_budget_pct),
        "precision_at_k": precision_at_k_curve(risk, y),
        "macro_f1": float(f1_score(labels, predicted, labels=list(ANOMALY_CLASSES), average="macro", zero_division=0)),
        "macro_f1_present": float(f1_score(labels, predicted, labels=present, average="macro", zero_division=0)),
        "missing_classes": missing,
        "calibration_ece": expected_calibration_error(prob, y),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(ANOMALY_CLASSES)).astype(int).tolist(),
        "per_class": per_class_metrics(labels, predicted),
        "detector_precision": detector_precision(scored),
    }
    if scored.latency_ms:
        arr = np.asarray(scored.latency_ms)
        metrics["latency_median_ms"] = float(np.median(arr))
        metrics["latency_p95_ms"] = float(np.percentile(arr, 95))
    return metrics


def dataset_summary(scored: ScoredSplit) -> DatasetSummary:
    counts = Counter(scored.labels)
    return DatasetSummary(
        n_events=len(scored.labels),
        n_entities=len(set(scored.entity_ids)),
        anomaly_rate=float(scored.y_true.mean()) if scored.y_true.size else 0.0,
        per_class_counts=dict(counts),
        split=scored.split,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_model_metrics(scored: ScoredSplit, metrics: Dict[str, Any]) -> ModelMetrics:
    manifest = read_manifest()
    return ModelMetrics(
        artifact_schema_version=manifest.get("schema_version"),
        git_sha=manifest.get("git_sha"),
        seed=settings.random_seed,
        dataset_summary=dataset_summary(scored),
        pr_auc=metrics["pr_auc"],
        roc_auc=metrics["roc_auc"],
        recall_at_1pct_budget=metrics["recall_at_1pct_budget"],
        precision_at_k=metrics["precision_at_k"],
        macro_f1=metrics["macro_f1"],
        calibration_ece=metrics["calibration_ece"],
        confusion_matrix=metrics["confusion_matrix"],
        class_order=list(ANOMALY_CLASSES),
        per_class=metrics["per_class"],
        notes=(
            f"macro_f1 over classes present in {scored.split}: "
            f"{metrics['macro_f1_present']:.4f}"
            + (f"; missing: {', '.join(metrics['missing_classes'])}" if metrics["missing_classes"] else "")
        ),
    )


def _persist_to_mongo(model_metrics: ModelMetrics) -> bool:
    """Best-effort write to the model_metrics collection; returns success."""
    import asyncio

    async def _write() -> None:
        from common.database import Collections, get_collection

        await get_collection(Collections.MODEL_METRICS).insert_one(
            model_metrics.model_dump(mode="json")
        )

    try:
        asyncio.run(_write())
        return True
    except Exception as exc:  # noqa: BLE001 - Mongo is optional for evaluation
        logger.warning("Could not persist metrics to MongoDB: %s", exc)
        return False


def evaluate(
    split: str = "test",
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Score the split, compute metrics, and write ``metrics.json`` (+ optional Mongo)."""
    set_global_seed()
    target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    models = load_models(target)

    logger.info("Scoring the %s split", split)
    scored = score_split(split, *models, dataset_dir=dataset_dir, limit=limit, measure_latency=True)
    metrics = compute_metrics(scored)
    model_metrics = build_model_metrics(scored, metrics)

    # metrics.json carries the clean ModelMetrics plus a few report-only extras (detector precision,
    # present-class macro-F1, latency) that have no home in the persisted schema.
    payload = model_metrics.model_dump(mode="json")
    payload["detector_precision"] = metrics["detector_precision"]
    payload["macro_f1_present"] = metrics["macro_f1_present"]
    payload["latency_median_ms"] = metrics.get("latency_median_ms")
    payload["latency_p95_ms"] = metrics.get("latency_p95_ms")
    metrics_path = target / METRICS_FILE
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    logger.info("Wrote %s", metrics_path)

    persisted = _persist_to_mongo(model_metrics) if persist else False

    return {
        "metrics": metrics,
        "model_metrics": model_metrics,
        "metrics_path": str(metrics_path),
        "persisted_to_mongo": persisted,
    }


def format_report(summary: Dict[str, Any]) -> str:
    m = summary["metrics"]

    def fmt(value: Any, spec: str = ".4f") -> str:
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)

    det = m["detector_precision"]
    lines = [
        "",
        "=" * 74,
        " Test-set evaluation (Deliverable #7)",
        "=" * 74,
        f" events             : {m['n']:,}  (prevalence {fmt(m['prevalence'], '.4%')})",
        " " + "-" * 60,
        f"   PR-AUC             : {fmt(m['pr_auc'])}   target >= 0.90",
        f"   ROC-AUC            : {fmt(m['roc_auc'])}   (uninformative at 1% prevalence)",
        f"   recall @ 1% budget : {fmt(m['recall_at_1pct_budget'])}   target >= 0.80",
        f"   macro-F1 (9-class) : {fmt(m['macro_f1'])}   target >= 0.85",
        f"   macro-F1 (present) : {fmt(m['macro_f1_present'])}",
        f"   calibration ECE    : {fmt(m['calibration_ece'])}   target <= 0.05",
        "",
        f"   impossible_travel  : anomaly-precision {fmt(det['impossible_travel']['anomaly_precision'])} "
        f"(fired {int(det['impossible_travel']['n_fired'])})",
        f"   brute_force        : anomaly-precision {fmt(det['brute_force']['anomaly_precision'])} "
        f"(fired {int(det['brute_force']['n_fired'])})",
    ]
    if "latency_median_ms" in m:
        lines.append(f"   latency            : median {fmt(m['latency_median_ms'], '.2f')} ms")
    lines += [f" mongo persisted     : {summary['persisted_to_mongo']}", "=" * 74, ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.evaluate")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-persist", action="store_true", help="Skip the MongoDB write.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        summary = evaluate(split=args.split, limit=args.limit, persist=not args.no_persist)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(format_report(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
