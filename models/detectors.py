"""Deterministic detectors: the geometric certainties.

Two attack classes have a physical signature that needs no learning. *Impossible travel* is a
speed a human cannot achieve between two sessions; *brute force* is a burst of failed
authentications in a short window. A learned model can approximate these, but a rule states them
exactly, so when a rule fires with high confidence it is a near-certain call and should override a
probabilistic guess.

Both detectors read the **same features the rest of the system uses** -- the raw, unscaled values
on a :class:`~features.featurize.FeatureVector` -- rather than recomputing geometry from the event.
That keeps them in lock-step with the feature pipeline: the impossible-travel rule and the
``log_geo_velocity_kmh`` feature can never disagree because they are the same number. It also means
the detectors are trivially reused offline and online with no second code path.

Their role in the stack is twofold: they contribute to the risk score, and when they fire
confidently they override the classifier's anomaly *type* (a model that mislabels an obvious brute
force as ``credential_stuffing`` should defer to the rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.config import settings
from common.models import ANOMALY_CLASS_INDEX, AnomalyType

#: Distance floor for impossible travel. Two points a few hundred km apart can imply a silly speed
#: from geolocation jitter or ordinary regional travel over a short gap, so only genuinely
#: long-haul hops fire. Injected impossible-travel jumps are >6000 km, so this floor keeps every
#: attack while discarding short benign hops.
MIN_TRAVEL_DISTANCE_KM = 2000.0

#: A detector at or above this confidence overrides the classifier's predicted type.
OVERRIDE_CONFIDENCE = 0.75


@dataclass
class DetectorResult:
    """The outcome of one deterministic check on one event."""

    name: str
    anomaly_type: AnomalyType
    fired: bool
    confidence: float
    detail: str = ""

    @property
    def overrides(self) -> bool:
        """Whether this result is confident enough to override the classifier's type."""
        return self.fired and self.confidence >= OVERRIDE_CONFIDENCE


def _confidence_over(value: float, threshold: float) -> float:
    """Map "how far past the threshold" to a confidence in ``[0.5, 1.0]`` for a fired detector.

    A value exactly at the threshold is a marginal fire (0.5); at or beyond twice the threshold it
    is unambiguous (1.0). Below the threshold the detector does not fire at all.
    """
    if threshold <= 0.0:
        return 1.0
    ratio = (value - threshold) / threshold
    return float(min(1.0, max(0.5, 0.5 + 0.5 * ratio)))


class ImpossibleTravelDetector:
    """Fires when the implied speed between consecutive events exceeds a physical limit."""

    def __init__(
        self,
        threshold_kmh: Optional[float] = None,
        min_distance_km: float = MIN_TRAVEL_DISTANCE_KM,
    ) -> None:
        self.threshold_kmh = (
            settings.impossible_travel_kmh if threshold_kmh is None else threshold_kmh
        )
        self.min_distance_km = min_distance_km

    def evaluate(self, raw: Mapping[str, float]) -> DetectorResult:
        # Features are stored log1p-compressed; invert to recover physical units.
        velocity = math.expm1(float(raw.get("log_geo_velocity_kmh", 0.0)))
        distance = math.expm1(float(raw.get("log_distance_from_prev_km", 0.0)))
        is_first = float(raw.get("is_first_event", 0.0)) >= 0.5
        # The destination must be a country this entity has never used. Without this, the rule also
        # fires on the physically-equal-but-benign "snap back home" event right after a real jump,
        # and on entities that legitimately connect from more than one familiar location -- neither
        # of which is an intrusion. An attacker's impossible hop lands somewhere genuinely new.
        into_new_country = float(raw.get("is_new_country", 0.0)) >= 0.5

        fired = (
            not is_first
            and into_new_country
            and distance >= self.min_distance_km
            and velocity > self.threshold_kmh
        )
        confidence = _confidence_over(velocity, self.threshold_kmh) if fired else 0.0
        detail = (
            f"{velocity:,.0f} km/h over {distance:,.0f} km (limit {self.threshold_kmh:,.0f} km/h)"
            if fired
            else ""
        )
        return DetectorResult(
            name="impossible_travel",
            anomaly_type=AnomalyType.IMPOSSIBLE_TRAVEL,
            fired=fired,
            confidence=confidence,
            detail=detail,
        )


class BruteForceDetector:
    """Fires when failed authentications in the rolling window exceed a burst threshold."""

    def __init__(self, threshold: Optional[int] = None) -> None:
        self.threshold = settings.brute_force_threshold if threshold is None else threshold

    def evaluate(self, raw: Mapping[str, float]) -> DetectorResult:
        failures = float(raw.get("window_auth_failures", 0.0))
        fired = failures >= self.threshold
        confidence = _confidence_over(failures, float(self.threshold)) if fired else 0.0
        detail = (
            f"{int(failures)} failed auths in the window (threshold {self.threshold})"
            if fired
            else ""
        )
        return DetectorResult(
            name="brute_force",
            anomaly_type=AnomalyType.BRUTE_FORCE,
            fired=fired,
            confidence=confidence,
            detail=detail,
        )


class DetectorBank:
    """The full set of deterministic detectors, evaluated together."""

    def __init__(
        self,
        impossible_travel: Optional[ImpossibleTravelDetector] = None,
        brute_force: Optional[BruteForceDetector] = None,
    ) -> None:
        self.detectors = [
            impossible_travel or ImpossibleTravelDetector(),
            brute_force or BruteForceDetector(),
        ]

    def evaluate(self, raw: Mapping[str, float]) -> List[DetectorResult]:
        """Run every detector; returns a result per detector (fired or not)."""
        return [detector.evaluate(raw) for detector in self.detectors]

    def fired(self, raw: Mapping[str, float]) -> List[DetectorResult]:
        """Only the detectors that fired, strongest confidence first."""
        results = [result for result in self.evaluate(raw) if result.fired]
        results.sort(key=lambda result: result.confidence, reverse=True)
        return results


def strongest_override(results: Sequence[DetectorResult]) -> Optional[DetectorResult]:
    """The most confident detector result eligible to override the classifier, if any."""
    eligible = [result for result in results if result.overrides]
    if not eligible:
        return None
    return max(eligible, key=lambda result: result.confidence)


def resolve_anomaly_type(
    classifier_type: AnomalyType,
    classifier_probs: Mapping[str, float],
    detector_results: Sequence[DetectorResult],
) -> Tuple[AnomalyType, Dict[str, float], List[str]]:
    """Reconcile the classifier's type with the deterministic detectors.

    A confident detector wins: an obvious brute force or impossible-travel event should carry that
    label even if the classifier guessed a neighbouring class. When it overrides, the probability
    mass is moved onto the detector's class so the reported probabilities stay consistent with the
    reported type. Returns the resolved type, the (possibly adjusted) probabilities, and the names
    of every detector that fired.
    """
    probs: Dict[str, float] = {name: float(value) for name, value in classifier_probs.items()}
    hits = [result.name for result in detector_results if result.fired]

    override = strongest_override(detector_results)
    if override is None:
        return classifier_type, probs, hits

    forced = override.anomaly_type
    # Blend the detector's confidence into the probability of its class, keeping the rest ranked.
    forced_key = forced.value
    remaining = max(0.0, 1.0 - override.confidence)
    other_total = sum(value for key, value in probs.items() if key != forced_key) or 1.0
    adjusted = {
        key: (override.confidence if key == forced_key else remaining * value / other_total)
        for key, value in probs.items()
    }
    if forced_key not in adjusted:
        adjusted[forced_key] = override.confidence
    return forced, adjusted, hits


def attack_probability(classifier_probs: Mapping[str, float]) -> float:
    """Probability the event is *any* attack -- the classifier's contribution to fusion."""
    normal = float(classifier_probs.get(AnomalyType.NORMAL.value, 0.0))
    return float(min(1.0, max(0.0, 1.0 - normal)))


__all__ = [
    "MIN_TRAVEL_DISTANCE_KM",
    "OVERRIDE_CONFIDENCE",
    "DetectorResult",
    "ImpossibleTravelDetector",
    "BruteForceDetector",
    "DetectorBank",
    "strongest_override",
    "resolve_anomaly_type",
    "attack_probability",
]
