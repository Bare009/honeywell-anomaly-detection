"""Probability calibration, persisted as JSON.

A raw model score is only loosely a probability. Isotonic regression fixes that: it learns the
monotonic mapping from score to empirical frequency, so a calibrated 0.8 really does mean "about
80% of events like this were anomalous". That property is what makes a risk score trustworthy to an
analyst and what the Expected Calibration Error measures.

Isotonic regression is chosen over Platt scaling because it makes no shape assumption -- it fits
whatever monotonic curve the data shows -- and, crucially here, it reduces to a small table of
``(x, y)`` knots that persists as plain JSON and evaluates with a single ``numpy`` interpolation.
No pickled scikit-learn estimator, so the artifact survives a library upgrade like every other one
in the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class IsotonicCalibrator:
    """A fitted monotonic score -> probability map, stored as interpolation knots."""

    x: List[float]
    y: List[float]

    @classmethod
    def fit(cls, scores: Sequence[float], targets: Sequence[float]) -> "IsotonicCalibrator":
        """Fit isotonic regression mapping ``scores`` to binary ``targets`` in ``[0, 1]``."""
        from sklearn.isotonic import IsotonicRegression

        score_arr = np.asarray(scores, dtype=float)
        target_arr = np.asarray(targets, dtype=float)
        finite = np.isfinite(score_arr)
        score_arr, target_arr = score_arr[finite], target_arr[finite]

        if score_arr.size == 0 or len(np.unique(target_arr)) < 2:
            # Nothing to calibrate against: fall back to the identity, clamped to [0, 1].
            return cls(x=[0.0, 1.0], y=[0.0, 1.0])

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(score_arr, target_arr)
        knots_x = np.asarray(model.X_thresholds_, dtype=float)
        knots_y = np.asarray(model.y_thresholds_, dtype=float)
        if knots_x.size < 2:
            # Degenerate fit (a single distinct score): keep a flat, valid two-point map.
            value = float(knots_y[0]) if knots_y.size else float(target_arr.mean())
            return cls(x=[0.0, 1.0], y=[value, value])
        return cls(x=knots_x.tolist(), y=knots_y.tolist())

    def transform(self, scores: Any) -> np.ndarray:
        """Map scores to calibrated probabilities via linear interpolation between knots.

        ``numpy.interp`` clamps to the endpoint values outside the fitted range, matching the
        isotonic ``out_of_bounds="clip"`` behaviour.
        """
        values = np.asarray(scores, dtype=float)
        calibrated = np.interp(values, self.x, self.y)
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IsotonicCalibrator":
        return cls(x=list(payload.get("x", [0.0, 1.0])), y=list(payload.get("y", [0.0, 1.0])))


def expected_calibration_error(
    probabilities: Sequence[float], targets: Sequence[float], n_bins: int = 10
) -> float:
    """Expected Calibration Error: the average gap between confidence and accuracy.

    Predictions are grouped into ``n_bins`` equal-width confidence bins; within each bin the mean
    predicted probability (confidence) is compared to the observed frequency of positives
    (accuracy), and the absolute gaps are averaged weighted by bin population. Zero is perfect.
    """
    prob = np.asarray(probabilities, dtype=float)
    target = np.asarray(targets, dtype=float)
    if prob.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = prob.size
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        # Include the right edge in the final bin so probability 1.0 is counted.
        if upper >= 1.0:
            mask = (prob >= lower) & (prob <= upper)
        else:
            mask = (prob >= lower) & (prob < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(target[mask].mean())
        error += (count / total) * abs(confidence - accuracy)
    return float(error)


__all__ = ["IsotonicCalibrator", "expected_calibration_error"]
