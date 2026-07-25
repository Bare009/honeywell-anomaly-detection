"""Concept-drift detection and adaptation (D3).

Normal behaviour evolves -- new schedules, new devices, new locations. A detector that treats every
such change as an attack is useless, and one that never notices change goes stale. This module walks
the line with the **Population Stability Index (PSI)**: it compares the recent distribution of a
behavioural signal against a learned baseline, and

* a **gradual, benign** shift is absorbed -- the baseline is re-profiled with an EWMA each time PSI
  stays under the threshold, so it tracks the new normal and PSI stays low;
* an **abrupt** shift spikes PSI above the threshold and is flagged as drift rather than silently
  accepted;
* a shift that persists is eventually accepted as the new baseline (``ADAPTED``), so the system does
  not alarm forever on a real, lasting change.

PSI is a standard, interpretable measure (``< 0.1`` little change, ``0.1-0.25`` moderate, ``> 0.25``
major). The threshold and the adaptation rate come from configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from common.config import settings
from common.models import DriftStatus, utc_now


def population_stability_index(
    baseline: Sequence[float], recent: Sequence[float], eps: float = 1e-6
) -> float:
    """PSI between two binned distributions (probability vectors of equal length)."""
    b = np.clip(np.asarray(baseline, dtype=float), eps, None)
    r = np.clip(np.asarray(recent, dtype=float), eps, None)
    b = b / b.sum()
    r = r / r.sum()
    return float(np.sum((r - b) * np.log(r / b)))


@dataclass
class DriftReading:
    """The outcome of folding one observation into a monitor."""

    psi: float
    status: DriftStatus
    samples_seen: int
    adapted: bool = False


@dataclass
class DriftMonitor:
    """Tracks drift of a scalar behavioural signal for one entity.

    Construct with :meth:`from_baseline` from an initial sample of the signal, then feed new values
    with :meth:`update`.
    """

    entity_id: str
    edges: List[float]
    baseline_probs: List[float]
    threshold: float = field(default_factory=lambda: settings.drift_psi_threshold)
    alpha: float = field(default_factory=lambda: settings.drift_refresh_alpha)
    min_samples: int = field(default_factory=lambda: settings.drift_min_samples)
    window_size: int = field(default_factory=lambda: settings.drift_window_size)
    adapt_after: int = 3

    status: DriftStatus = DriftStatus.STABLE
    psi: float = 0.0
    samples_seen: int = 0
    _drift_streak: int = 0
    _recent: Deque[float] = field(default_factory=deque)
    last_refresh: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not isinstance(self._recent, deque) or self._recent.maxlen != self.window_size:
            self._recent = deque(self._recent, maxlen=self.window_size)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_edges(values: Sequence[float], n_bins: int) -> np.ndarray:
        """Quantile bin edges from a baseline sample, widened if the signal is near-constant."""
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return np.linspace(-1.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(data, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:  # constant signal
            center = float(edges[0]) if edges.size else 0.0
            return np.array([center - 0.5, center + 0.5])
        return edges

    @classmethod
    def from_baseline(
        cls,
        entity_id: str,
        baseline_values: Sequence[float],
        n_bins: Optional[int] = None,
        **overrides: Any,
    ) -> "DriftMonitor":
        """Build a monitor from an initial sample of the tracked signal."""
        bins = settings.drift_psi_bins if n_bins is None else n_bins
        edges = cls._make_edges(baseline_values, bins)
        probs = cls._histogram(baseline_values, edges)
        return cls(
            entity_id=entity_id,
            edges=edges.tolist(),
            baseline_probs=probs.tolist(),
            **overrides,
        )

    @staticmethod
    def _histogram(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
        counts, _ = np.histogram(np.asarray(values, dtype=float), bins=np.asarray(edges, dtype=float))
        total = counts.sum()
        if total == 0:
            return np.full(len(counts), 1.0 / max(1, len(counts)))
        return counts / total

    # ------------------------------------------------------------------ #
    # Streaming update
    # ------------------------------------------------------------------ #

    def update(self, value: float) -> DriftReading:
        """Fold one observation in; return the current PSI and drift status."""
        self._recent.append(float(value))
        self.samples_seen += 1

        if len(self._recent) < self.min_samples:
            self.status = DriftStatus.STABLE
            self.psi = 0.0
            return DriftReading(self.psi, self.status, self.samples_seen)

        recent_probs = self._histogram(list(self._recent), self.edges)
        self.psi = population_stability_index(self.baseline_probs, recent_probs)

        adapted = False
        if self.psi >= self.threshold:
            self._drift_streak += 1
            self.status = DriftStatus.DRIFTING
            if self._drift_streak >= self.adapt_after:
                # A lasting shift becomes the new normal rather than a permanent alarm.
                self._absorb(recent_probs)
                self.status = DriftStatus.ADAPTED
                self._drift_streak = 0
                adapted = True
        else:
            # Benign, gradual change: track it so PSI stays low.
            self._absorb(recent_probs)
            self.status = DriftStatus.STABLE
            self._drift_streak = 0
            adapted = True

        return DriftReading(self.psi, self.status, self.samples_seen, adapted=adapted)

    def _absorb(self, recent_probs: np.ndarray) -> None:
        """EWMA re-profiling: nudge the baseline toward the recent distribution."""
        baseline = np.asarray(self.baseline_probs, dtype=float)
        updated = (1.0 - self.alpha) * baseline + self.alpha * np.asarray(recent_probs, dtype=float)
        total = updated.sum()
        self.baseline_probs = (updated / total if total else updated).tolist()
        self.last_refresh = utc_now()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "edges": list(self.edges),
            "baseline_probs": list(self.baseline_probs),
            "threshold": self.threshold,
            "alpha": self.alpha,
            "min_samples": self.min_samples,
            "window_size": self.window_size,
            "adapt_after": self.adapt_after,
            "status": self.status.value,
            "psi": self.psi,
            "samples_seen": self.samples_seen,
            "recent": list(self._recent),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DriftMonitor":
        monitor = cls(
            entity_id=payload["entity_id"],
            edges=list(payload.get("edges", [])),
            baseline_probs=list(payload.get("baseline_probs", [])),
            threshold=float(payload.get("threshold", settings.drift_psi_threshold)),
            alpha=float(payload.get("alpha", settings.drift_refresh_alpha)),
            min_samples=int(payload.get("min_samples", settings.drift_min_samples)),
            window_size=int(payload.get("window_size", settings.drift_window_size)),
            adapt_after=int(payload.get("adapt_after", 3)),
            status=DriftStatus(payload.get("status", DriftStatus.STABLE.value)),
            psi=float(payload.get("psi", 0.0)),
            samples_seen=int(payload.get("samples_seen", 0)),
        )
        monitor._recent = deque(payload.get("recent", []), maxlen=monitor.window_size)
        refresh = payload.get("last_refresh")
        monitor.last_refresh = datetime.fromisoformat(refresh) if refresh else None
        return monitor


__all__ = ["population_stability_index", "DriftReading", "DriftMonitor"]
