"""Behavioral cohorts -- the mechanism behind cold start.

An entity with no history cannot be scored against its own baseline. The usual fallback is a
global average, which is nearly useless here: the "average" of a night-batch service account
and a 9-to-5 office worker resembles neither, so a new office worker's first morning login
looks anomalous against it.

So entities are clustered by **how they behave** -- when they work, where from, how much data
they move, how they authenticate -- and a new entity is scored against its cohort's prior
instead. The generator happens to build its population from latent archetypes, but nothing here
is told that: the clustering is unsupervised over observed behavior, and it would work the same
on real telemetry.

The fitted model persists as **centroids in JSON**, not a pickled estimator. Assignment is a
nearest-centroid lookup, so serving needs no scikit-learn version match to reproduce training's
cohort assignment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.config import settings
from common.models import EntityType
from features.entity_window import BehaviorProfile, ProfileAccumulator

#: Ordered names of the behavioral summary dimensions used for clustering. Documented so a
#: cohort can be described to an analyst ("this cohort works nights and moves a lot of data")
#: rather than being an opaque integer.
SUMMARY_FEATURE_NAMES: List[str] = (
    [f"hour_{hour:02d}" for hour in range(24)]
    + ["weekend_share"]
    + ["type_user", "type_service_account", "type_edge_device"]
    + [
        "log_bytes_out_mean",
        "log_bytes_out_std",
        "log_duration_mean",
        "log_interval_mean",
        "sequence_len_mean",
        "auth_failure_rate",
        "distinct_countries",
        "distinct_resources",
        "geo_spread_scaled",
        "auth_password_share",
        "auth_token_share",
        "auth_certificate_share",
        "auth_mfa_share",
        "auth_biometric_share",
    ]
)


def behavior_summary(profile: BehaviorProfile) -> np.ndarray:
    """Turn a learned profile into a fixed-length vector for clustering.

    Deliberately built from *shape* rather than volume: the hour histogram is normalized, and
    counts appear only as logs or ratios. Otherwise clustering would separate busy entities from
    quiet ones, which says nothing about behavior.
    """
    hour_hist = list(profile.hour_hist) if profile.hour_hist else [0.0] * 24
    if len(hour_hist) != 24:
        hour_hist = (hour_hist + [0.0] * 24)[:24]

    dow = list(profile.dow_hist) if profile.dow_hist else [0.0] * 7
    dow = (dow + [0.0] * 7)[:7]
    weekend_share = float(sum(dow[5:7]))

    entity_type = profile.entity_type or ""
    type_flags = [
        1.0 if entity_type == EntityType.USER.value else 0.0,
        1.0 if entity_type == EntityType.SERVICE_ACCOUNT.value else 0.0,
        1.0 if entity_type == EntityType.EDGE_DEVICE.value else 0.0,
    ]

    auth = profile.auth_method_dist or {}

    values = (
        hour_hist
        + [weekend_share]
        + type_flags
        + [
            profile.bytes_out_log.mean,
            profile.bytes_out_log.std,
            profile.duration_log.mean,
            profile.interval_log.mean,
            profile.sequence_len.mean,
            profile.auth_failure_rate,
            # Counts of distinct values are log-scaled: the difference between 1 and 3 countries
            # matters far more than between 30 and 32 resources.
            math.log1p(len(profile.country_dist)),
            math.log1p(len(profile.resource_dist)),
            math.log1p(max(0.0, profile.geo_spread_km)) / 10.0,
            auth.get("password", 0.0),
            auth.get("token", 0.0),
            auth.get("certificate", 0.0),
            auth.get("mfa", 0.0),
            auth.get("biometric", 0.0),
        ]
    )

    vector = np.asarray(values, dtype=float)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class CohortModel:
    """Nearest-centroid cohort assignment over standardized behavioral summaries."""

    centroids: List[List[float]] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=lambda: list(SUMMARY_FEATURE_NAMES))
    #: Human-readable description per cohort, derived from its centroid.
    labels: Dict[int, str] = field(default_factory=dict)
    #: Most common cohort per entity type, for an entity with no behavior at all yet.
    type_cohorts: Dict[str, int] = field(default_factory=dict)
    sizes: Dict[int, int] = field(default_factory=dict)

    @property
    def n_cohorts(self) -> int:
        return len(self.centroids)

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(
        cls,
        profiles: Sequence[BehaviorProfile],
        n_cohorts: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "CohortModel":
        """Cluster entity profiles into behavioral cohorts.

        Only entities with real history are used for fitting. Including thin profiles would let
        the clusters be shaped by entities we know nothing about, defeating the purpose.
        """
        from sklearn.cluster import KMeans

        resolved_seed = settings.random_seed if seed is None else seed
        target = settings.cohort_count if n_cohorts is None else n_cohorts

        usable = [profile for profile in profiles if profile.event_count >= 10]
        if len(usable) < 2:
            usable = list(profiles)
        if not usable:
            return cls()

        matrix = np.vstack([behavior_summary(profile) for profile in usable])

        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        scaled = (matrix - means) / stds

        clusters = int(min(target, len(usable)))
        kmeans = KMeans(
            n_clusters=clusters,
            n_init=10,
            random_state=resolved_seed,
            algorithm="lloyd",
        )
        assignments = kmeans.fit_predict(scaled)

        sizes: Dict[int, int] = {}
        for cohort in assignments:
            sizes[int(cohort)] = sizes.get(int(cohort), 0) + 1

        # Which cohort to hand a brand-new entity that has produced nothing yet: the most
        # populous cohort among entities of the same type.
        type_counts: Dict[str, Dict[int, int]] = {}
        for profile, cohort in zip(usable, assignments):
            key = profile.entity_type or "unknown"
            bucket = type_counts.setdefault(key, {})
            bucket[int(cohort)] = bucket.get(int(cohort), 0) + 1
        type_cohorts = {
            entity_type: max(counts.items(), key=lambda item: item[1])[0]
            for entity_type, counts in type_counts.items()
        }

        model = cls(
            centroids=kmeans.cluster_centers_.tolist(),
            means=means.tolist(),
            stds=stds.tolist(),
            feature_names=list(SUMMARY_FEATURE_NAMES),
            type_cohorts=type_cohorts,
            sizes=sizes,
        )
        model.labels = {
            cohort: model.describe(cohort) for cohort in range(model.n_cohorts)
        }
        return model

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #

    def _scale(self, vector: np.ndarray) -> np.ndarray:
        if not self.means or not self.stds:
            return vector
        means = np.asarray(self.means, dtype=float)
        stds = np.asarray(self.stds, dtype=float)
        if vector.shape[0] != means.shape[0]:
            raise ValueError(
                f"summary has {vector.shape[0]} dimensions, model expects {means.shape[0]}"
            )
        return (vector - means) / stds

    def assign(self, profile: BehaviorProfile) -> Optional[int]:
        """Nearest cohort for a profile, or ``None`` if the model is unfitted."""
        if not self.centroids:
            return None
        scaled = self._scale(behavior_summary(profile))
        centroids = np.asarray(self.centroids, dtype=float)
        distances = np.linalg.norm(centroids - scaled, axis=1)
        return int(np.argmin(distances))

    def assign_for_new_entity(
        self, entity_type: Optional[str], profile: Optional[BehaviorProfile] = None
    ) -> Optional[int]:
        """Cohort for an entity we know little or nothing about.

        With even a few events, nearest-centroid on the partial profile is used -- a handful of
        logins already reveals whether something is a nightly batch job or a daytime user. With
        nothing at all, fall back to the most common cohort for its entity type.
        """
        if profile is not None and profile.event_count >= 3:
            assigned = self.assign(profile)
            if assigned is not None:
                return assigned
        if entity_type and entity_type in self.type_cohorts:
            return self.type_cohorts[entity_type]
        if self.sizes:
            return max(self.sizes.items(), key=lambda item: item[1])[0]
        return None

    # ------------------------------------------------------------------ #
    # Description
    # ------------------------------------------------------------------ #

    def describe(self, cohort: int) -> str:
        """Plain-language summary of a cohort, for the dashboard and the report."""
        if cohort < 0 or cohort >= self.n_cohorts or not self.means:
            return f"cohort {cohort}"

        centroid = np.asarray(self.centroids[cohort], dtype=float)
        means = np.asarray(self.means, dtype=float)
        stds = np.asarray(self.stds, dtype=float)
        raw = centroid * stds + means  # back to interpretable units

        index = {name: position for position, name in enumerate(self.feature_names)}
        hours = np.asarray([raw[index[f"hour_{hour:02d}"]] for hour in range(24)])
        peak_hour = int(np.argmax(hours))
        night_share = float(hours[list(range(0, 6)) + [22, 23]].sum())

        type_names = ["user", "service_account", "edge_device"]
        type_values = [raw[index[f"type_{name}"]] for name in type_names]
        dominant_type = type_names[int(np.argmax(type_values))]

        schedule = "around the clock" if night_share > 0.28 else f"peaking at {peak_hour:02d}:00"
        weekend = raw[index["weekend_share"]]
        weekday_note = "including weekends" if weekend > 0.22 else "on weekdays"

        return f"{dominant_type}s active {schedule}, {weekday_note}"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroids": self.centroids,
            "means": self.means,
            "stds": self.stds,
            "feature_names": list(self.feature_names),
            "labels": {str(key): value for key, value in self.labels.items()},
            "type_cohorts": dict(self.type_cohorts),
            "sizes": {str(key): value for key, value in self.sizes.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CohortModel":
        return cls(
            centroids=[list(row) for row in payload.get("centroids") or []],
            means=list(payload.get("means") or []),
            stds=list(payload.get("stds") or []),
            feature_names=list(payload.get("feature_names") or SUMMARY_FEATURE_NAMES),
            labels={int(key): value for key, value in (payload.get("labels") or {}).items()},
            type_cohorts={
                key: int(value) for key, value in (payload.get("type_cohorts") or {}).items()
            },
            sizes={int(key): int(value) for key, value in (payload.get("sizes") or {}).items()},
        )

    def save(self, path: Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return target

    @classmethod
    def load(cls, path: Path) -> "CohortModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def build_cohort_priors(
    accumulators: Dict[str, ProfileAccumulator],
    assignments: Dict[str, int],
) -> Dict[int, BehaviorProfile]:
    """Aggregate a prior profile per cohort by **merging counts**, not averaging profiles.

    Merging raw accumulators is exact. Averaging finished profiles would give a quiet entity's
    distribution the same weight as a busy one's, so a single low-volume outlier could distort
    the prior its whole cohort depends on.
    """
    merged: Dict[int, ProfileAccumulator] = {}

    for entity_id, accumulator in accumulators.items():
        cohort = assignments.get(entity_id)
        if cohort is None:
            continue
        target = merged.get(cohort)
        if target is None:
            target = ProfileAccumulator(f"__cohort_{cohort}__", accumulator.entity_type)
            merged[cohort] = target
        target.merge(accumulator)

    priors: Dict[int, BehaviorProfile] = {}
    for cohort, accumulator in merged.items():
        profile = accumulator.build(cohort=cohort)
        profile.cold_start = False
        profile.confidence = 1.0
        priors[cohort] = profile
    return priors


def build_global_prior(accumulators: Dict[str, ProfileAccumulator]) -> BehaviorProfile:
    """Last-resort prior: every entity merged together.

    Used only when an entity's cohort cannot be determined. Weak by nature -- averaging a plant
    sensor with an analyst produces a shape resembling neither -- which is exactly why cohorts
    exist and why the cold-start ablation in Phase 9 compares against this.
    """
    combined = ProfileAccumulator("__global__")
    for accumulator in accumulators.values():
        combined.merge(accumulator)
    profile = combined.build()
    profile.entity_id = "__global__"
    profile.cold_start = False
    profile.confidence = 1.0
    return profile


__all__ = [
    "SUMMARY_FEATURE_NAMES",
    "behavior_summary",
    "CohortModel",
    "build_cohort_priors",
    "build_global_prior",
]
