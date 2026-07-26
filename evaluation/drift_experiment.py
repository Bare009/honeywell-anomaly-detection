"""Concept-drift experiment (D3): adaptation absorbs benign drift, abrupt shifts still fire.

Two controlled streams, deterministic under the fixed seed so the report reproduces exactly:

* a **benign gradual drift** (a slowly ramping mean, like a team migrating to a new schedule), run
  once with adaptation off and once on -- the false-positive (DRIFTING) rate collapses when the
  baseline re-profiles, which is the whole point;
* an **abrupt shift** (a sudden jump to a new regime), which spikes PSI above the threshold and is
  flagged rather than silently accepted.

The PSI series for both are saved so the report can plot the adaptation curve. The stream shapes are
parameterized to mirror the benign-drift cohort's gradual schedule change; using synthetic streams
keeps the experiment reproducible rather than hostage to which entities a regeneration produces.

Run it with::

    python -m evaluation.drift_experiment
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
from common.models import DriftMetrics, DriftStatus
from common.seed import set_global_seed
from models.drift import DriftMonitor

logger = logging.getLogger(__name__)

DRIFT_FILE = "drift_metrics.json"


def _baseline_sample(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, 600)


def _run_stream(alpha: float, values: np.ndarray) -> Dict[str, Any]:
    """Feed a stream through a monitor; return the PSI series and drift statistics."""
    monitor = DriftMonitor.from_baseline(
        "experiment",
        _baseline_sample(),
        n_bins=10,
        min_samples=30,
        window_size=100,
        threshold=settings.drift_psi_threshold,
        alpha=alpha,
    )
    psi_series: List[float] = []
    drifting = 0
    adapted = 0
    for value in values:
        reading = monitor.update(float(value))
        psi_series.append(round(reading.psi, 4))
        if reading.status == DriftStatus.DRIFTING:
            drifting += 1
        if reading.status == DriftStatus.ADAPTED:
            adapted += 1
    scored = [p for p in psi_series if p > 0.0] or [0.0]
    return {
        "psi_series": psi_series,
        "drift_fraction": drifting / len(values),
        "adaptation_events": adapted,
        "mean_psi": float(np.mean(scored)),
    }


def run(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run the drift experiment and write ``drift_metrics.json``."""
    set_global_seed()
    target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)

    rng = np.random.default_rng(7)
    steps = 600
    benign = np.array([2.5 * i / steps + rng.normal(0.0, 1.0) for i in range(steps)])

    abrupt_rng = np.random.default_rng(8)
    abrupt = np.concatenate(
        [abrupt_rng.normal(0.0, 1.0, 300), abrupt_rng.normal(6.0, 1.0, 300)]
    )

    without_adapt = _run_stream(alpha=0.0, values=benign)
    with_adapt = _run_stream(alpha=settings.drift_refresh_alpha, values=benign)
    abrupt_run = _run_stream(alpha=settings.drift_refresh_alpha, values=abrupt)

    result = DriftMetrics(
        fp_rate_before=without_adapt["drift_fraction"],
        fp_rate_after=with_adapt["drift_fraction"],
        adaptation_events=with_adapt["adaptation_events"],
        mean_psi=with_adapt["mean_psi"],
    )
    payload = {
        **result.model_dump(mode="json"),
        "abrupt_max_psi": float(max(abrupt_run["psi_series"])),
        "abrupt_flagged": abrupt_run["drift_fraction"] > 0.0,
        "series": {
            "benign_with_adaptation": with_adapt["psi_series"],
            "benign_without_adaptation": without_adapt["psi_series"],
            "abrupt": abrupt_run["psi_series"],
            "threshold": settings.drift_psi_threshold,
        },
    }
    path = target / DRIFT_FILE
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
            " Drift adaptation experiment (D3)",
            "=" * 74,
            f" benign drift, no adaptation : DRIFTING {fmt(p['fp_rate_before'])} of updates",
            f" benign drift, adaptation on : DRIFTING {fmt(p['fp_rate_after'])} of updates",
            f" adaptation events           : {p['adaptation_events']}",
            f" abrupt shift max PSI        : {fmt(p['abrupt_max_psi'])}  "
            f"(threshold {fmt(p['series']['threshold'])}, flagged: {p['abrupt_flagged']})",
            "=" * 74,
            "",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.drift_experiment")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    summary = run()
    print(format_report(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
