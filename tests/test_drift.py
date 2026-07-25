"""Concept-drift tests (Phase 6, D3).

PSI behaves as expected on known distributions, and the monitor distinguishes the two cases that
matter: a gradual benign shift is absorbed (PSI stays low, the entity is not flagged), while an
abrupt shift spikes PSI above the threshold and is flagged. The contrast between adaptation on and
off is asserted directly, since "adapts to benign drift" is the whole point.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.models import DriftStatus
from models.drift import DriftMonitor, population_stability_index


class TestPSI:
    def test_identical_distributions_score_zero(self) -> None:
        probs = [0.2, 0.3, 0.5]
        assert population_stability_index(probs, probs) == pytest.approx(0.0, abs=1e-9)

    def test_different_distributions_score_positive(self) -> None:
        assert population_stability_index([0.8, 0.1, 0.1], [0.1, 0.1, 0.8]) > 0.5

    def test_is_symmetric_enough_and_finite(self) -> None:
        a, b = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
        assert np.isfinite(population_stability_index(a, b))
        assert population_stability_index(a, b) > 0.0

    def test_handles_empty_bins_without_nan(self) -> None:
        assert np.isfinite(population_stability_index([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))


def _monitor(baseline_seed: int = 0, **overrides):
    rng = np.random.default_rng(baseline_seed)
    baseline = rng.normal(0.0, 1.0, 600)
    params = dict(n_bins=10, min_samples=30, window_size=100, threshold=0.25, alpha=0.05)
    params.update(overrides)
    n_bins = params.pop("n_bins")
    return DriftMonitor.from_baseline("entity_x", baseline, n_bins=n_bins, **params)


class TestDriftMonitor:
    def test_stable_stream_stays_stable(self) -> None:
        monitor = _monitor()
        rng = np.random.default_rng(1)
        for value in rng.normal(0.0, 1.0, 300):
            reading = monitor.update(value)
        assert reading.status == DriftStatus.STABLE
        assert monitor.psi < monitor.threshold

    def test_abrupt_shift_is_flagged(self) -> None:
        monitor = _monitor()
        rng = np.random.default_rng(2)
        statuses = []
        for value in rng.normal(0.0, 1.0, 120):
            statuses.append(monitor.update(value).status)
        for value in rng.normal(6.0, 1.0, 120):  # sudden jump to a new regime
            statuses.append(monitor.update(value).status)
        assert DriftStatus.DRIFTING in statuses

    @staticmethod
    def _run_ramp(alpha: float) -> "tuple[float, DriftMonitor]":
        monitor = _monitor(alpha=alpha)
        rng = np.random.default_rng(3)
        drift_hits = 0
        steps = 600
        for step in range(steps):
            mean = 2.0 * step / steps  # slow ramp 0 -> 2
            if monitor.update(mean + rng.normal(0.0, 1.0)).status == DriftStatus.DRIFTING:
                drift_hits += 1
        return drift_hits / steps, monitor

    def test_gradual_benign_drift_is_absorbed(self) -> None:
        """With EWMA re-profiling, a slow shift is absorbed: drift is rare and self-corrects."""
        drift_fraction, monitor = self._run_ramp(alpha=0.15)
        assert drift_fraction < 0.1
        assert monitor.status != DriftStatus.DRIFTING
        assert monitor.psi < monitor.threshold

    def test_adaptation_matters(self) -> None:
        """The same ramp trips PSI far more often without adaptation than with it."""
        adaptive_fraction, _ = self._run_ramp(alpha=0.15)
        static_fraction, _ = self._run_ramp(alpha=0.0)
        assert static_fraction > adaptive_fraction
        assert static_fraction > 0.1

    def test_sustained_shift_eventually_adapts(self) -> None:
        monitor = _monitor(adapt_after=3)
        rng = np.random.default_rng(4)
        for value in rng.normal(0.0, 1.0, 120):
            monitor.update(value)
        statuses = [monitor.update(value).status for value in rng.normal(6.0, 1.0, 400)]
        assert DriftStatus.ADAPTED in statuses  # a lasting shift becomes the new normal

    def test_below_min_samples_is_stable(self) -> None:
        monitor = _monitor(min_samples=50)
        reading = monitor.update(0.0)
        assert reading.status == DriftStatus.STABLE
        assert reading.psi == 0.0

    def test_dict_round_trip(self) -> None:
        monitor = _monitor()
        rng = np.random.default_rng(9)
        for value in rng.normal(0.0, 1.0, 60):
            monitor.update(value)
        restored = DriftMonitor.from_dict(monitor.to_dict())
        assert restored.baseline_probs == monitor.baseline_probs
        assert restored.samples_seen == monitor.samples_seen
        assert restored.status == monitor.status
