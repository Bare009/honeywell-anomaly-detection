"""Risk fusion: three tier scores into one calibrated, explainable number.

The three detectors answer different questions -- "is this an unfamiliar combination?" (baseline),
"is this an unfamiliar order of actions?" (sequence), "does this look like a known attack type?"
(classifier). Fusion combines them into a single ``0-100`` risk that an analyst can rank on, with
three properties that make it trustworthy:

* **Calibrated (D5).** The fused score is mapped through isotonic regression to an actual
  probability of anomaly, so ``risk = 80`` means roughly an 80% chance this is malicious. That is
  what keeps the Expected Calibration Error low.
* **An uncertainty band (D5).** When the tiers disagree, or the entity is cold-start with a thin
  profile, the score is less trustworthy and the band widens -- surfacing "we are not sure" instead
  of a false-precision number.
* **An alert budget (D4).** The threshold is set so that flagging everything above it spends roughly
  the analyst's review budget (the top ``alert_budget_pct`` of events), tuned on validation to catch
  as many true anomalies as that budget allows.

The fusion weights and the calibration are **tuned on validation to maximise recall within the
budget**, not hand-set. A confident deterministic detector overrides the fused number upward: a
physically impossible login is not a probabilistic maybe.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.config import settings
from models.calibration import IsotonicCalibrator
from models.detectors import DetectorResult, strongest_override

logger = logging.getLogger(__name__)

#: Artifact filename for the persisted risk/fusion model.
RISK_FILE = "risk_model.json"

#: A confident detector override lifts risk to at least this, scaled by its confidence.
DETECTOR_RISK_FLOOR = 90.0


@dataclass
class RiskAssessment:
    """The fused verdict for one event."""

    risk_score: float
    risk_uncertainty: float
    fused_raw: float
    classifier_confidence: float
    in_alert_budget: bool
    is_anomaly: bool
    detector_hits: List[str] = field(default_factory=list)


@dataclass
class RiskModel:
    """Holds the fusion weights, calibration, thresholds and uncertainty parameters."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(settings.fusion_weights))
    calibrator: IsotonicCalibrator = field(
        default_factory=lambda: IsotonicCalibrator(x=[0.0, 1.0], y=[0.0, 1.0])
    )
    budget_threshold: float = 60.0
    alert_threshold: float = field(default_factory=lambda: settings.risk_alert_threshold)
    uncertainty_scale: float = 0.8
    coldstart_multiplier: float = field(
        default_factory=lambda: settings.coldstart_uncertainty_multiplier
    )
    max_uncertainty: float = 50.0

    # ------------------------------------------------------------------ #
    # The fusion computation
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        baseline: np.ndarray,
        sequence: np.ndarray,
        classifier: np.ndarray,
    ) -> np.ndarray:
        """Weighted combination of the three tier scores into ``fused_raw`` in ``[0, 1]``."""
        b = np.asarray(baseline, dtype=float)
        s = np.asarray(sequence, dtype=float)
        c = np.asarray(classifier, dtype=float)
        fused = (
            self.weights.get("baseline", 0.0) * b
            + self.weights.get("sequence", 0.0) * s
            + self.weights.get("classifier", 0.0) * c
        )
        return np.clip(fused, 0.0, 1.0)

    def to_risk(self, fused_raw: np.ndarray) -> np.ndarray:
        """Map fused score to a calibrated ``0-100`` risk."""
        probability = self.calibrator.transform(np.asarray(fused_raw, dtype=float))
        return 100.0 * probability

    def uncertainty(
        self,
        baseline: np.ndarray,
        sequence: np.ndarray,
        classifier: np.ndarray,
        cold_start: np.ndarray,
    ) -> np.ndarray:
        """Half-width of the risk band, in risk points.

        Driven by tier disagreement (their standard deviation) and widened for cold-start entities,
        whose thin profiles make every score less certain.
        """
        stacked = np.vstack(
            [np.asarray(baseline, dtype=float), np.asarray(sequence, dtype=float), np.asarray(classifier, dtype=float)]
        )
        disagreement = stacked.std(axis=0)
        band = 100.0 * self.uncertainty_scale * disagreement
        cold = np.asarray(cold_start, dtype=float) >= 0.5
        band = np.where(cold, band * self.coldstart_multiplier, band)
        return np.clip(band, 0.0, self.max_uncertainty)

    # ------------------------------------------------------------------ #
    # Per-event assessment (used by serving)
    # ------------------------------------------------------------------ #

    def assess(
        self,
        baseline: float,
        sequence: float,
        classifier: float,
        cold_start: bool = False,
        detector_results: Optional[Sequence[DetectorResult]] = None,
    ) -> RiskAssessment:
        """Fuse one event's tier scores into a full :class:`RiskAssessment`."""
        fused = float(self.fuse(np.array([baseline]), np.array([sequence]), np.array([classifier]))[0])
        risk = float(self.to_risk(np.array([fused]))[0])
        band = float(
            self.uncertainty(
                np.array([baseline]), np.array([sequence]), np.array([classifier]), np.array([cold_start])
            )[0]
        )

        results = list(detector_results or [])
        hits = [result.name for result in results if result.fired]
        override = strongest_override(results)
        if override is not None:
            risk = max(risk, DETECTOR_RISK_FLOOR * override.confidence)

        in_budget = risk >= self.budget_threshold
        is_anomaly = risk >= self.alert_threshold or override is not None
        return RiskAssessment(
            risk_score=round(risk, 4),
            risk_uncertainty=round(band, 4),
            fused_raw=round(fused, 6),
            classifier_confidence=float(classifier),
            in_alert_budget=bool(in_budget or override is not None),
            is_anomaly=bool(is_anomaly),
            detector_hits=hits,
        )

    # ------------------------------------------------------------------ #
    # Tuning on validation
    # ------------------------------------------------------------------ #

    @classmethod
    def tune(
        cls,
        baseline: Sequence[float],
        sequence: Sequence[float],
        classifier: Sequence[float],
        is_anomaly: Sequence[int],
        budget_pct: Optional[float] = None,
        weight_step: float = 0.05,
        **overrides: Any,
    ) -> "RiskModel":
        """Fit fusion weights, calibration and the budget threshold on validation data.

        Weights are grid-searched over the simplex to maximise recall within the alert budget; the
        calibration maps the best fused score to an anomaly probability; the budget threshold is the
        risk cutoff that spends exactly the budget.
        """
        b = np.asarray(baseline, dtype=float)
        s = np.asarray(sequence, dtype=float)
        c = np.asarray(classifier, dtype=float)
        y = np.asarray(is_anomaly, dtype=int)
        budget = settings.alert_budget_pct if budget_pct is None else budget_pct

        positives = int(y.sum())
        n = y.size
        k = max(1, int(round(n * budget)))

        best_weights = dict(settings.fusion_weights)
        best_recall = -1.0
        if positives > 0:
            steps = int(round(1.0 / weight_step))
            for i in range(steps + 1):
                for j in range(steps + 1 - i):
                    wb = i * weight_step
                    ws = j * weight_step
                    wc = 1.0 - wb - ws
                    if wc < -1e-9:
                        continue
                    wc = max(0.0, wc)
                    fused = wb * b + ws * s + wc * c
                    top = np.argsort(fused)[::-1][:k]
                    recall = float(y[top].sum() / positives)
                    if recall > best_recall:
                        best_recall = recall
                        best_weights = {"baseline": wb, "sequence": ws, "classifier": wc}

        model = cls(weights=best_weights, **overrides)
        fused = model.fuse(b, s, c)
        model.calibrator = IsotonicCalibrator.fit(fused, y)

        risk = model.to_risk(fused)
        # Threshold that flags the top `budget` fraction. Guard the degenerate all-equal case.
        model.budget_threshold = float(np.quantile(risk, max(0.0, 1.0 - budget)))
        logger.info(
            "Risk tuned: weights=%s, recall@budget=%.4f, budget_threshold=%.2f",
            {key: round(value, 3) for key, value in best_weights.items()},
            best_recall,
            model.budget_threshold,
        )
        return model

    # ------------------------------------------------------------------ #
    # Persistence (JSON)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "risk_fusion",
            "weights": dict(self.weights),
            "calibrator": self.calibrator.to_dict(),
            "budget_threshold": self.budget_threshold,
            "alert_threshold": self.alert_threshold,
            "uncertainty_scale": self.uncertainty_scale,
            "coldstart_multiplier": self.coldstart_multiplier,
            "max_uncertainty": self.max_uncertainty,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RiskModel":
        return cls(
            weights=dict(payload.get("weights", settings.fusion_weights)),
            calibrator=IsotonicCalibrator.from_dict(payload.get("calibrator", {})),
            budget_threshold=float(payload.get("budget_threshold", 60.0)),
            alert_threshold=float(payload.get("alert_threshold", settings.risk_alert_threshold)),
            uncertainty_scale=float(payload.get("uncertainty_scale", 0.8)),
            coldstart_multiplier=float(
                payload.get("coldstart_multiplier", settings.coldstart_uncertainty_multiplier)
            ),
            max_uncertainty=float(payload.get("max_uncertainty", 50.0)),
        )

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else Path(settings.artifacts_dir) / RISK_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        logger.info("Wrote risk model to %s", target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RiskModel":
        source = Path(path) if path else Path(settings.artifacts_dir) / RISK_FILE
        if not source.exists():
            raise FileNotFoundError(
                f"No risk model at {source}. Run: python -m training.train_classifier"
            )
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


__all__ = [
    "RISK_FILE",
    "DETECTOR_RISK_FLOOR",
    "RiskAssessment",
    "RiskModel",
]
