"""Orchestrate the full offline artifact build.

Run it with::

    python -m training.build_artifacts

This is the single entry point that turns a generated dataset into a complete, servable
``artifacts/`` directory. It runs each training stage in dependency order and stops on the first
failure, because a later stage that trained against half-built inputs would be silently wrong.

Stages, in order:

1. **Feature pipeline** (``training.build_baselines``) -- encoders, scaler, sequence vocabulary,
   entity profiles, behavioral cohorts and priors. Everything downstream consumes these.
2. **Baseline autoencoder** (``training.train_baseline``) -- Tier 1.
3. **GRU sequence model** (``training.train_sequence``) -- Tier 2.

Later phases append their stages here (classifier, fusion, calibration, SHAP background), so the
demo `make seed` step stays a single command as the system grows.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common.config import settings
from common.seed import set_global_seed
from models.baseline import BaselineTrainConfig
from models.sequence import SequenceTrainConfig
from training import build_baselines, train_baseline, train_sequence

logger = logging.getLogger(__name__)


def build_all(
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    skip_features: bool = False,
    baseline_config: Optional[BaselineTrainConfig] = None,
    sequence_config: Optional[SequenceTrainConfig] = None,
) -> Dict[str, Any]:
    """Run every artifact-building stage in order and return a combined summary."""
    set_global_seed()
    target_dir = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    results: Dict[str, Any] = {}

    if skip_features:
        logger.info("Skipping the feature-pipeline stage (--skip-features)")
    else:
        logger.info("[1/3] Building the feature pipeline and entity baselines")
        started = time.perf_counter()
        results["features"] = build_baselines.build(
            split="train", dataset_dir=dataset_dir, artifacts_dir=target_dir
        )
        logger.info("      done in %.1fs", time.perf_counter() - started)

    logger.info("[2/3] Training the baseline autoencoder")
    started = time.perf_counter()
    results["baseline"] = train_baseline.train_baseline(
        dataset_dir=dataset_dir,
        artifacts_dir=target_dir,
        config=baseline_config,
    )
    logger.info("      done in %.1fs", time.perf_counter() - started)

    logger.info("[3/3] Training the GRU sequence model")
    started = time.perf_counter()
    results["sequence"] = train_sequence.train_sequence(
        dataset_dir=dataset_dir,
        artifacts_dir=target_dir,
        config=sequence_config,
    )
    logger.info("      done in %.1fs", time.perf_counter() - started)

    return results


def format_report(results: Dict[str, Any]) -> str:
    """Combine each stage's report into one summary block."""
    blocks: List[str] = []
    if "features" in results:
        blocks.append(build_baselines.format_report(results["features"]))
    if "baseline" in results:
        blocks.append(train_baseline.format_report(results["baseline"]))
    if "sequence" in results:
        blocks.append(train_sequence.format_report(results["sequence"]))
    return "\n".join(blocks)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.build_artifacts",
        description="Build the complete artifacts/ directory from a generated dataset.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Artifacts directory.")
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Reuse an already-fitted feature pipeline and only (re)train the models.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    try:
        results = build_all(
            dataset_dir=args.dataset_dir,
            artifacts_dir=args.artifacts_dir,
            skip_features=args.skip_features,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.quiet:
        print(format_report(results))

    # A baseline that shows no signal means the build is not servable; surface that as a failure.
    metrics = results.get("baseline", {}).get("val_metrics", {})
    pr_auc = metrics.get("pr_auc")
    prevalence = metrics.get("prevalence", 0.0)
    if pr_auc is not None and prevalence and not (pr_auc > 2.0 * prevalence):
        logger.error("Baseline PR-AUC %.4f does not clear the random floor; build not servable.", pr_auc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
