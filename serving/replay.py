"""Replay a dataset split through the scoring pipeline.

Used two ways: to drive the live demo (stream historical events past the scorer so the dashboard
fills with ranked, explained, campaign-linked alerts) and to sanity-check end-to-end scoring and
latency offline. By default it runs the pipeline in-process against an in-memory store, so it needs
no database and no running service; point it at a real store for a full demo.

Deterministic under a fixed seed: the same replay produces the same detections every time.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.models import Detection, Event
from common.seed import set_global_seed
from serving.pipeline import ScoringPipeline
from serving.store import DetectionStore, InMemoryStore
from training.train_baseline import load_split

logger = logging.getLogger(__name__)


async def _run(
    pipeline: ScoringPipeline, events: List[Event]
) -> Tuple[List[Detection], List[float]]:
    detections: List[Detection] = []
    latencies_ms: List[float] = []
    for event in events:
        started = time.perf_counter()
        detection = await pipeline.score_event(event)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        detections.append(detection)
    return detections, latencies_ms


def replay(
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 42,
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    store: Optional[DetectionStore] = None,
    enable_explanations: bool = True,
) -> Dict[str, Any]:
    """Replay a split and return a summary (detections, latency, campaigns)."""
    set_global_seed(seed)
    store = store or InMemoryStore()
    pipeline = ScoringPipeline.load(
        artifacts_dir=artifacts_dir, store=store, enable_explanations=enable_explanations
    )

    events = load_split(split, dataset_dir)
    if limit is not None:
        events = events[:limit]

    detections, latencies = asyncio.run(_run(pipeline, events))
    latency_arr = np.asarray(latencies, dtype=float)

    campaigns = asyncio.run(store.list_campaigns(limit=10_000))
    anomalies = [d for d in detections if d.is_anomaly]

    return {
        "split": split,
        "n_events": len(events),
        "n_anomalies": len(anomalies),
        "n_in_budget": sum(1 for d in detections if d.in_alert_budget),
        "n_campaigns": len(campaigns),
        "latency_median_ms": float(np.median(latency_arr)) if latency_arr.size else 0.0,
        "latency_p95_ms": float(np.percentile(latency_arr, 95)) if latency_arr.size else 0.0,
        "pipeline": pipeline,
        "store": store,
        "detections": detections,
    }


def format_report(summary: Dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 74,
        " Replay through the scoring pipeline",
        "=" * 74,
        f" split              : {summary['split']}",
        f" events scored      : {summary['n_events']:,}",
        f" anomalies          : {summary['n_anomalies']:,}",
        f" in alert budget    : {summary['n_in_budget']:,}",
        f" campaigns linked   : {summary['n_campaigns']:,}",
        f" latency            : median {summary['latency_median_ms']:.2f} ms  "
        f"p95 {summary['latency_p95_ms']:.2f} ms",
        "=" * 74,
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m serving.replay",
        description="Replay a dataset split through the scoring pipeline (in-process).",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N events.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument(
        "--no-explanations",
        action="store_true",
        help="Skip SHAP/counterfactual explanations (faster; measures raw scoring latency).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    try:
        summary = replay(
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            dataset_dir=args.dataset_dir,
            artifacts_dir=args.artifacts_dir,
            enable_explanations=not args.no_explanations,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(format_report(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
