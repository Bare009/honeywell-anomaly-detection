"""Cold-start ablation: do cohort priors actually help new entities? (D3 / cold-start target)

An entity with little history is scored against a blend of its thin history and its cohort's prior.
This experiment measures whether that prior earns its place: it scores the test split twice, once
with cohort priors and once with them removed (the store then falls back to the global prior), and
compares recall on the *cold-start anomalies* -- the events from entities below the history
threshold. The difference is the uplift the peer prior buys.

Both passes featurize the same events in the same order, so the cold-start mask (which depends only
on history depth, not on the prior) aligns across them. The number is reported honestly whichever
way it comes out.

Run it with::

    python -m evaluation.coldstart_experiment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.config import settings
from common.models import ColdStartMetrics
from common.seed import set_global_seed
from features.entity_window import ProfileStore
from features.featurize import FeaturePipeline
from evaluation.scoring import ScoredSplit, load_models, score_split

logger = logging.getLogger(__name__)

COLDSTART_FILE = "coldstart_metrics.json"


def _without_cohort_priors(pipeline: FeaturePipeline) -> FeaturePipeline:
    """A twin pipeline whose profile store has no cohort priors (falls back to the global prior)."""
    src = pipeline.profiles
    stripped = ProfileStore(
        profiles=src.profiles,
        cohort_priors={},  # remove peer priors; resolve() now falls back to the global prior
        global_prior=src.global_prior,
        type_cohorts=src.type_cohorts,
    )
    return FeaturePipeline(
        encoders=pipeline.encoders,
        vocab=pipeline.vocab,
        profiles=stripped,
        cohorts=pipeline.cohorts,
        corpus=pipeline.corpus,
    )


def run(
    split: str = "test",
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the cold-start ablation and write ``coldstart_metrics.json``."""
    set_global_seed()
    target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    pipeline, baseline, sequence, classifier, risk = load_models(target)

    logger.info("Scoring %s with cohort priors", split)
    with_priors = score_split(split, pipeline, baseline, sequence, classifier, risk, dataset_dir, limit)

    logger.info("Scoring %s without cohort priors (global fallback)", split)
    no_prior_pipeline = _without_cohort_priors(pipeline)
    without_priors = score_split(
        split, no_prior_pipeline, baseline, sequence, classifier, risk, dataset_dir, limit
    )

    cold = with_priors.cold_start
    y = with_priors.y_true
    cold_anomaly = cold & (y == 1)
    threshold = risk.alert_threshold

    def recall(scored: ScoredSplit) -> float:
        if not cold_anomaly.any():
            return float("nan")
        detected = np.asarray(scored.risk) >= threshold
        return float(detected[cold_anomaly].mean())

    recall_with = recall(with_priors)
    recall_without = recall(without_priors)
    uplift = (
        float(recall_with - recall_without)
        if np.isfinite(recall_with) and np.isfinite(recall_without)
        else float("nan")
    )
    n_cold_entities = len({eid for eid, is_cold in zip(with_priors.entity_ids, cold) if is_cold})

    result = ColdStartMetrics(
        recall_with_priors=recall_with,
        recall_without_priors=recall_without,
        uplift=uplift,
        n_cold_entities=n_cold_entities,
    )

    payload = {
        **result.model_dump(mode="json"),
        "n_cold_events": int(cold.sum()),
        "n_cold_anomalies": int(cold_anomaly.sum()),
        "alert_threshold": threshold,
        "split": split,
    }
    path = target / COLDSTART_FILE
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return {"result": result, "payload": payload, "path": str(path)}


def format_report(summary: Dict[str, Any]) -> str:
    p = summary["payload"]

    def fmt(value: Any) -> str:
        try:
            return format(float(value), ".4f")
        except (TypeError, ValueError):
            return str(value)

    return "\n".join(
        [
            "",
            "=" * 74,
            " Cold-start ablation (cohort priors vs global fallback)",
            "=" * 74,
            f" cold-start entities : {p['n_cold_entities']:,}",
            f" cold-start anomalies: {p['n_cold_anomalies']:,}",
            f" recall WITH priors  : {fmt(p['recall_with_priors'])}   target >= 0.70",
            f" recall WITHOUT      : {fmt(p['recall_without_priors'])}",
            f" uplift              : {fmt(p['uplift'])}",
            "=" * 74,
            "",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.coldstart_experiment")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        summary = run(split=args.split, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(format_report(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
