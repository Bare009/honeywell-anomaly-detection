"""Deterministic detector tests (Phase 5).

The detectors are pure functions over the raw feature values, so they need no training and no
dataset. The properties that matter: they fire on the geometric signature (fast long hop, failed-auth
burst) and stay silent otherwise, their confidence rises with the margin over the threshold, and a
confident detector overrides the classifier's type while moving probability mass onto its class.
"""

from __future__ import annotations

import math
from typing import Dict

import pytest

from common.config import settings
from common.models import AnomalyType
from models.detectors import (
    OVERRIDE_CONFIDENCE,
    BruteForceDetector,
    DetectorBank,
    ImpossibleTravelDetector,
    attack_probability,
    resolve_anomaly_type,
    strongest_override,
)


def make_raw(
    velocity_kmh: float = 0.0,
    distance_km: float = 0.0,
    is_first: bool = False,
    auth_failures: float = 0.0,
    new_country: bool = True,
) -> Dict[str, float]:
    """Build the raw-feature dict the detectors read, in the pipeline's log1p units."""
    return {
        "log_geo_velocity_kmh": math.log1p(velocity_kmh),
        "log_distance_from_prev_km": math.log1p(distance_km),
        "is_first_event": 1.0 if is_first else 0.0,
        "is_new_country": 1.0 if new_country else 0.0,
        "window_auth_failures": auth_failures,
    }


class TestImpossibleTravel:
    def test_fires_on_fast_long_hop(self) -> None:
        detector = ImpossibleTravelDetector()
        result = detector.evaluate(make_raw(velocity_kmh=3000.0, distance_km=8000.0))
        assert result.fired is True
        assert result.anomaly_type == AnomalyType.IMPOSSIBLE_TRAVEL
        assert 0.5 <= result.confidence <= 1.0

    def test_silent_on_short_hop(self) -> None:
        """A high implied speed over a few km is geolocation jitter, not travel."""
        detector = ImpossibleTravelDetector()
        result = detector.evaluate(make_raw(velocity_kmh=3000.0, distance_km=50.0))
        assert result.fired is False

    def test_silent_on_first_event(self) -> None:
        detector = ImpossibleTravelDetector()
        result = detector.evaluate(
            make_raw(velocity_kmh=3000.0, distance_km=8000.0, is_first=True)
        )
        assert result.fired is False

    def test_silent_below_threshold(self) -> None:
        detector = ImpossibleTravelDetector()
        slow = settings.impossible_travel_kmh * 0.5
        result = detector.evaluate(make_raw(velocity_kmh=slow, distance_km=8000.0))
        assert result.fired is False

    def test_silent_when_country_is_familiar(self) -> None:
        """Snapping back to a known country -- or a benign multi-location entity -- is not travel."""
        detector = ImpossibleTravelDetector()
        result = detector.evaluate(
            make_raw(velocity_kmh=3000.0, distance_km=8000.0, new_country=False)
        )
        assert result.fired is False

    def test_confidence_rises_with_speed(self) -> None:
        detector = ImpossibleTravelDetector()
        near = detector.evaluate(
            make_raw(velocity_kmh=settings.impossible_travel_kmh * 1.1, distance_km=8000.0)
        )
        far = detector.evaluate(
            make_raw(velocity_kmh=settings.impossible_travel_kmh * 5.0, distance_km=8000.0)
        )
        assert far.confidence > near.confidence


class TestBruteForce:
    def test_fires_on_failure_burst(self) -> None:
        detector = BruteForceDetector()
        result = detector.evaluate(make_raw(auth_failures=settings.brute_force_threshold + 3))
        assert result.fired is True
        assert result.anomaly_type == AnomalyType.BRUTE_FORCE

    def test_silent_below_threshold(self) -> None:
        detector = BruteForceDetector()
        result = detector.evaluate(make_raw(auth_failures=settings.brute_force_threshold - 1))
        assert result.fired is False

    def test_confidence_rises_with_failures(self) -> None:
        detector = BruteForceDetector()
        few = detector.evaluate(make_raw(auth_failures=settings.brute_force_threshold))
        many = detector.evaluate(make_raw(auth_failures=settings.brute_force_threshold * 4))
        assert many.confidence > few.confidence


class TestDetectorBank:
    def test_evaluate_returns_all_detectors(self) -> None:
        bank = DetectorBank()
        results = bank.evaluate(make_raw())
        assert {r.name for r in results} == {"impossible_travel", "brute_force"}

    def test_fired_only_returns_fired_sorted(self) -> None:
        bank = DetectorBank()
        raw = make_raw(velocity_kmh=5000.0, distance_km=9000.0, auth_failures=20)
        fired = bank.fired(raw)
        assert len(fired) == 2
        assert fired[0].confidence >= fired[1].confidence

    def test_no_fire_on_benign(self) -> None:
        bank = DetectorBank()
        assert bank.fired(make_raw(velocity_kmh=400.0, distance_km=600.0, auth_failures=1)) == []


class TestTypeResolution:
    def test_confident_detector_overrides_classifier(self) -> None:
        detector = BruteForceDetector()
        result = detector.evaluate(make_raw(auth_failures=settings.brute_force_threshold * 4))
        assert result.overrides is True

        probs = {"normal": 0.2, "credential_stuffing": 0.6, "brute_force": 0.2}
        resolved, adjusted, hits = resolve_anomaly_type(
            AnomalyType.CREDENTIAL_STUFFING, probs, [result]
        )
        assert resolved == AnomalyType.BRUTE_FORCE
        assert adjusted["brute_force"] == pytest.approx(result.confidence)
        assert "brute_force" in hits

    def test_no_override_when_detector_silent(self) -> None:
        probs = {"normal": 0.1, "lateral_movement": 0.9}
        resolved, adjusted, hits = resolve_anomaly_type(
            AnomalyType.LATERAL_MOVEMENT, probs, DetectorBank().evaluate(make_raw())
        )
        assert resolved == AnomalyType.LATERAL_MOVEMENT
        assert hits == []

    def test_weak_detector_does_not_override(self) -> None:
        """A marginal fire (confidence < override threshold) informs but does not override."""
        detector = BruteForceDetector()
        marginal = detector.evaluate(make_raw(auth_failures=settings.brute_force_threshold))
        assert marginal.fired is True
        assert marginal.confidence < OVERRIDE_CONFIDENCE
        assert strongest_override([marginal]) is None


class TestAttackProbability:
    def test_complement_of_normal(self) -> None:
        assert attack_probability({"normal": 0.7, "brute_force": 0.3}) == pytest.approx(0.3)

    def test_clamped_to_unit_interval(self) -> None:
        assert 0.0 <= attack_probability({"normal": 1.2}) <= 1.0
        assert attack_probability({}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Precision on the real validation split (needs only the feature pipeline)
# --------------------------------------------------------------------------- #

import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_PIPELINE = (_ARTIFACTS / "encoders.json").exists()


@pytest.mark.integration
@pytest.mark.metrics
@pytest.mark.skipif(not _HAS_PIPELINE, reason="run python -m training.build_baselines first")
class TestBuiltDetectorPrecision:
    """Deterministic-detector precision on injected cases (Deliverable #4 acceptance)."""

    def _fire_and_labels(self):
        from features.featurize import FeaturePipeline
        from training.train_baseline import featurize_serving_split, load_split
        from training.train_sequence import load_label_map

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        vectors = featurize_serving_split(pipeline, load_split("val"))
        label_map = load_label_map("val")
        labels = np.array([label_map.get(v.event_id, "normal") for v in vectors], dtype=object)
        bank = DetectorBank()
        results = [bank.evaluate(v.raw) for v in vectors]
        return labels, results

    def test_brute_force_precision_is_near_perfect(self) -> None:
        labels, results = self._fire_and_labels()
        fired = np.array([any(r.name == "brute_force" and r.fired for r in res) for res in results])
        assert fired.sum() > 0
        precision = ((labels[fired] != "normal")).mean()
        assert precision >= 0.95

    def test_impossible_travel_fires_mostly_on_anomalies(self) -> None:
        """Requiring a genuinely new destination country keeps benign fires near zero."""
        labels, results = self._fire_and_labels()
        fired = np.array(
            [any(r.name == "impossible_travel" and r.fired for r in res) for res in results]
        )
        assert fired.sum() > 0
        anomaly_precision = (labels[fired] != "normal").mean()
        benign_fires = int((labels[fired] == "normal").sum())
        assert anomaly_precision >= 0.85
        assert benign_fires <= 2  # near-zero false alarms on benign traffic
