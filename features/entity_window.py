"""Per-entity context: the learned baseline and the rolling window.

Behavioral detection needs two different kinds of memory about an entity, and they are easy to
confuse:

**The learned baseline** (:class:`BehaviorProfile`) is the long-run summary of what this entity
normally does -- which hours, which countries, which resources, how much data. It is fitted
offline, persisted to ``artifacts/``, and answers "is this unusual *for them*".

**The rolling window** (:class:`EntityState`) is short-term memory of the last hour of activity.
It answers questions a single event cannot: how fast did they appear to travel since the
previous event, how many auth failures just happened, how many distinct resources are they
touching right now.

Both are updated by the same code offline and online, which is what makes train/serve parity
achievable rather than aspirational.

Cold start is handled here too, by :class:`ProfileStore`. An entity with little history is not
scored against an empty profile -- it is scored against a **blend** of its own thin history and
its cohort's prior, weighted by how much history it actually has. That shrinkage is the
mechanism behind the cold-start recall target, and the blend weight is exposed as a feature so
downstream tiers know how much to trust the comparison.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from common.config import settings
from common.models import Event
from features.geo import centroid, geo_velocity_kmh, max_distance_from_km
from features.sequences import NGRAM_SEPARATOR, ngrams

#: Keep only the most frequent N values in high-cardinality distributions. Unbounded
#: dictionaries would make the persisted profiles enormous while adding nothing: the tail is
#: indistinguishable from novelty anyway.
TOP_K_DEFAULT = 40

#: How much evidence a cohort prior is worth, in sessions. An entity with this many sessions is
#: trusted about half on its own history and half on its cohort's.
PRIOR_STRENGTH_SESSIONS = 12

#: Cap on rolling-window length, independent of the time window. Bounds memory and per-event
#: cost even if an entity produces a burst of thousands of events.
MAX_WINDOW_EVENTS = 400

#: How many events may pass before an entity's live profile is rebuilt. Finalizing a profile
#: normalizes a dozen distributions and recomputes a geographic centroid, which is wasted work
#: per event: one more event barely moves a baseline built from hundreds.
#:
#: Applied identically offline and online, so train/serve parity is preserved. It also mirrors
#: how real systems behave -- baselines refresh on a schedule, not on every request. A cached
#: profile is only ever *older* than the true one, never newer, so it cannot leak future
#: information into the comparison.
LIVE_PROFILE_REFRESH_EVENTS = 20


def _normalize(counts: Dict[str, float], top_k: Optional[int] = TOP_K_DEFAULT) -> Dict[str, float]:
    """Turn raw counts into a probability distribution, keeping the top K keys."""
    if not counts:
        return {}
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        items = items[:top_k]
    total = sum(value for _, value in items)
    if total <= 0:
        return {}
    return {key: value / total for key, value in items}


def _blend_dist(
    primary: Dict[str, float], prior: Dict[str, float], weight: float
) -> Dict[str, float]:
    """Convex blend of two distributions."""
    if weight >= 1.0:
        return dict(primary)
    if weight <= 0.0:
        return dict(prior)
    keys = set(primary) | set(prior)
    return {
        key: weight * primary.get(key, 0.0) + (1.0 - weight) * prior.get(key, 0.0)
        for key in keys
    }


def _blend_list(primary: Sequence[float], prior: Sequence[float], weight: float) -> List[float]:
    """Convex blend of two equal-length numeric vectors."""
    if not primary:
        return list(prior)
    if not prior or len(prior) != len(primary):
        return list(primary)
    return [
        weight * left + (1.0 - weight) * right for left, right in zip(primary, prior)
    ]


def _blend_scalar(primary: Optional[float], prior: Optional[float], weight: float) -> float:
    """Convex blend of two scalars, tolerating either being absent."""
    if primary is None:
        return float(prior or 0.0)
    if prior is None:
        return float(primary)
    return weight * float(primary) + (1.0 - weight) * float(prior)


@dataclass
class RunningStat:
    """Streaming mean and standard deviation.

    Sum-of-squares rather than Welford: the magnitudes here are small (log-scaled bytes,
    log-scaled durations) so numerical stability is not a concern, and this form serializes to
    three numbers and can be merged across entities by addition -- which is exactly what
    building a cohort prior needs.
    """

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, value: float) -> None:
        if value is None or math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        self.total += float(value)
        self.total_sq += float(value) * float(value)

    def merge(self, other: "RunningStat") -> None:
        """Absorb another accumulator (used to build cohort and global priors)."""
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self.total_sq / self.count) - (self.mean**2)
        return math.sqrt(max(0.0, variance))

    def zscore(self, value: float, fallback_std: float = 1.0) -> float:
        """Standard deviations from the mean, with a floor on the divisor.

        An entity with near-constant behavior would otherwise produce enormous z-scores from
        trivial variation, drowning out every other feature.
        """
        if self.count < 2:
            return 0.0
        spread = max(self.std, fallback_std * 0.1, 1e-3)
        return (float(value) - self.mean) / spread

    def to_dict(self) -> Dict[str, float]:
        return {"count": self.count, "total": self.total, "total_sq": self.total_sq}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunningStat":
        return cls(
            count=int(payload.get("count", 0)),
            total=float(payload.get("total", 0.0)),
            total_sq=float(payload.get("total_sq", 0.0)),
        )


# --------------------------------------------------------------------------- #
# Learned baseline
# --------------------------------------------------------------------------- #


@dataclass
class BehaviorProfile:
    """What the system has learned about one entity's normal behavior.

    Also used to represent a **cohort prior** and the **global prior**, which are the same
    shape aggregated over many entities. Keeping one type for all three is what lets the
    cold-start fallback be a blend rather than a special case with its own code path.
    """

    entity_id: str
    entity_type: Optional[str] = None
    cohort: Optional[int] = None

    session_count: int = 0
    event_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    hour_hist: List[float] = field(default_factory=lambda: [0.0] * 24)
    dow_hist: List[float] = field(default_factory=lambda: [0.0] * 7)

    country_dist: Dict[str, float] = field(default_factory=dict)
    ip_prefix_dist: Dict[str, float] = field(default_factory=dict)
    resource_dist: Dict[str, float] = field(default_factory=dict)
    auth_method_dist: Dict[str, float] = field(default_factory=dict)
    device_mac_dist: Dict[str, float] = field(default_factory=dict)
    device_os_dist: Dict[str, float] = field(default_factory=dict)
    protocol_dist: Dict[str, float] = field(default_factory=dict)
    token_dist: Dict[str, float] = field(default_factory=dict)
    ngram_dist: Dict[str, float] = field(default_factory=dict)

    home_lat: Optional[float] = None
    home_lon: Optional[float] = None
    geo_spread_km: float = 0.0

    auth_failure_rate: float = 0.0

    bytes_out_log: RunningStat = field(default_factory=RunningStat)
    bytes_in_log: RunningStat = field(default_factory=RunningStat)
    duration_log: RunningStat = field(default_factory=RunningStat)
    interval_log: RunningStat = field(default_factory=RunningStat)
    sequence_len: RunningStat = field(default_factory=RunningStat)

    #: Raw feature-vector statistics, filled in after the feature pass. The baseline model
    #: (Phase 3) uses these for per-entity statistical deviation scoring.
    feature_names: List[str] = field(default_factory=list)
    feature_means: List[float] = field(default_factory=list)
    feature_stds: List[float] = field(default_factory=list)

    #: True when this profile is too thin to trust on its own.
    cold_start: bool = True
    #: How much of this profile is its own history versus a prior (1.0 = entirely its own).
    confidence: float = 0.0

    # ------------------------------------------------------------------ #
    # Lookups used by the feature functions
    # ------------------------------------------------------------------ #

    def hour_likelihood(self, hour: int) -> float:
        """Relative likelihood of this hour, scaled so 1.0 means "as likely as uniform"."""
        if not self.hour_hist or sum(self.hour_hist) <= 0:
            return 1.0
        return float(self.hour_hist[hour % 24]) * 24.0

    def dow_likelihood(self, weekday: int) -> float:
        """Relative likelihood of this day of week, 1.0 meaning uniform."""
        if not self.dow_hist or sum(self.dow_hist) <= 0:
            return 1.0
        return float(self.dow_hist[weekday % 7]) * 7.0

    @staticmethod
    def likelihood(dist: Dict[str, float], key: Optional[str]) -> float:
        """Probability of a categorical value under a learned distribution."""
        if not dist or key is None:
            return 0.0
        return float(dist.get(key, 0.0))

    @staticmethod
    def is_new(dist: Dict[str, float], key: Optional[str]) -> bool:
        """Whether this value has never been observed for this entity.

        An empty distribution means "no history", which is *not* the same as "novel value" --
        reporting novelty there would make every first event of every entity look anomalous.
        """
        if not dist:
            return False
        return key is None or key not in dist

    def distance_from_home_km(self, lat: float, lon: float) -> float:
        """Distance from the entity's usual location, 0 when no home is known."""
        if self.home_lat is None or self.home_lon is None:
            return 0.0
        from features.geo import haversine_km

        return haversine_km(self.home_lat, self.home_lon, lat, lon)

    # ------------------------------------------------------------------ #
    # Blending (cold-start shrinkage)
    # ------------------------------------------------------------------ #

    def blend_with(self, prior: "BehaviorProfile", weight: float) -> "BehaviorProfile":
        """Shrink this profile toward a prior.

        ``weight`` is how much to trust this entity's own history: 1.0 keeps it unchanged, 0.0
        replaces it entirely with the prior. Every distribution, histogram and scalar is
        blended, so downstream feature code sees an ordinary profile and needs no cold-start
        branch of its own.
        """
        weight = float(min(1.0, max(0.0, weight)))
        if weight >= 1.0:
            return self

        merged = BehaviorProfile(
            entity_id=self.entity_id,
            entity_type=self.entity_type or prior.entity_type,
            cohort=self.cohort if self.cohort is not None else prior.cohort,
            session_count=self.session_count,
            event_count=self.event_count,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            hour_hist=_blend_list(self.hour_hist, prior.hour_hist, weight),
            dow_hist=_blend_list(self.dow_hist, prior.dow_hist, weight),
            country_dist=_blend_dist(self.country_dist, prior.country_dist, weight),
            ip_prefix_dist=_blend_dist(self.ip_prefix_dist, prior.ip_prefix_dist, weight),
            resource_dist=_blend_dist(self.resource_dist, prior.resource_dist, weight),
            auth_method_dist=_blend_dist(self.auth_method_dist, prior.auth_method_dist, weight),
            device_mac_dist=_blend_dist(self.device_mac_dist, prior.device_mac_dist, weight),
            device_os_dist=_blend_dist(self.device_os_dist, prior.device_os_dist, weight),
            protocol_dist=_blend_dist(self.protocol_dist, prior.protocol_dist, weight),
            token_dist=_blend_dist(self.token_dist, prior.token_dist, weight),
            ngram_dist=_blend_dist(self.ngram_dist, prior.ngram_dist, weight),
            auth_failure_rate=_blend_scalar(
                self.auth_failure_rate, prior.auth_failure_rate, weight
            ),
            geo_spread_km=_blend_scalar(self.geo_spread_km, prior.geo_spread_km, weight),
            feature_names=self.feature_names or list(prior.feature_names),
            feature_means=_blend_list(self.feature_means, prior.feature_means, weight)
            if self.feature_means
            else list(prior.feature_means),
            feature_stds=_blend_list(self.feature_stds, prior.feature_stds, weight)
            if self.feature_stds
            else list(prior.feature_stds),
            cold_start=True,
            confidence=weight,
        )

        # Home location: keep the entity's own if known, otherwise inherit the prior's.
        if self.home_lat is not None and self.home_lon is not None:
            merged.home_lat, merged.home_lon = self.home_lat, self.home_lon
        else:
            merged.home_lat, merged.home_lon = prior.home_lat, prior.home_lon

        # Numeric statistics: an entity with very few observations has unusable variance, so
        # the prior's accumulator is merged in to give the z-scores a sane scale.
        for attribute in ("bytes_out_log", "bytes_in_log", "duration_log", "interval_log", "sequence_len"):
            own: RunningStat = getattr(self, attribute)
            prior_stat: RunningStat = getattr(prior, attribute)
            combined = RunningStat(count=own.count, total=own.total, total_sq=own.total_sq)
            if own.count < 8:
                combined.merge(prior_stat)
            setattr(merged, attribute, combined)

        return merged

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form.

        Values are written at **full precision**, deliberately. Rounding to six decimals shrank
        the artifact noticeably but broke exact train/serve parity: training computed features
        from full-precision profiles while serving loaded rounded ones, so the two disagreed in
        the sixth decimal on every likelihood feature. Python's JSON encoder round-trips floats
        exactly, so this is lossless.
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "cohort": self.cohort,
            "session_count": self.session_count,
            "event_count": self.event_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "hour_hist": list(self.hour_hist),
            "dow_hist": list(self.dow_hist),
            "country_dist": dict(self.country_dist),
            "ip_prefix_dist": dict(self.ip_prefix_dist),
            "resource_dist": dict(self.resource_dist),
            "auth_method_dist": dict(self.auth_method_dist),
            "device_mac_dist": dict(self.device_mac_dist),
            "device_os_dist": dict(self.device_os_dist),
            "protocol_dist": dict(self.protocol_dist),
            "token_dist": dict(self.token_dist),
            "ngram_dist": dict(self.ngram_dist),
            "home_lat": self.home_lat,
            "home_lon": self.home_lon,
            "geo_spread_km": self.geo_spread_km,
            "auth_failure_rate": self.auth_failure_rate,
            "bytes_out_log": self.bytes_out_log.to_dict(),
            "bytes_in_log": self.bytes_in_log.to_dict(),
            "duration_log": self.duration_log.to_dict(),
            "interval_log": self.interval_log.to_dict(),
            "sequence_len": self.sequence_len.to_dict(),
            "feature_names": list(self.feature_names),
            "feature_means": list(self.feature_means),
            "feature_stds": list(self.feature_stds),
            "cold_start": self.cold_start,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BehaviorProfile":
        def _parse_time(value: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(value) if value else None

        profile = cls(
            entity_id=payload["entity_id"],
            entity_type=payload.get("entity_type"),
            cohort=payload.get("cohort"),
            session_count=int(payload.get("session_count", 0)),
            event_count=int(payload.get("event_count", 0)),
            first_seen=_parse_time(payload.get("first_seen")),
            last_seen=_parse_time(payload.get("last_seen")),
            hour_hist=list(payload.get("hour_hist") or [0.0] * 24),
            dow_hist=list(payload.get("dow_hist") or [0.0] * 7),
            country_dist=dict(payload.get("country_dist") or {}),
            ip_prefix_dist=dict(payload.get("ip_prefix_dist") or {}),
            resource_dist=dict(payload.get("resource_dist") or {}),
            auth_method_dist=dict(payload.get("auth_method_dist") or {}),
            device_mac_dist=dict(payload.get("device_mac_dist") or {}),
            device_os_dist=dict(payload.get("device_os_dist") or {}),
            protocol_dist=dict(payload.get("protocol_dist") or {}),
            token_dist=dict(payload.get("token_dist") or {}),
            ngram_dist=dict(payload.get("ngram_dist") or {}),
            home_lat=payload.get("home_lat"),
            home_lon=payload.get("home_lon"),
            geo_spread_km=float(payload.get("geo_spread_km", 0.0)),
            auth_failure_rate=float(payload.get("auth_failure_rate", 0.0)),
            feature_names=list(payload.get("feature_names") or []),
            feature_means=list(payload.get("feature_means") or []),
            feature_stds=list(payload.get("feature_stds") or []),
            cold_start=bool(payload.get("cold_start", True)),
            confidence=float(payload.get("confidence", 0.0)),
        )
        for attribute in ("bytes_out_log", "bytes_in_log", "duration_log", "interval_log", "sequence_len"):
            setattr(profile, attribute, RunningStat.from_dict(payload.get(attribute) or {}))
        return profile


# --------------------------------------------------------------------------- #
# Streaming profile builder
# --------------------------------------------------------------------------- #


class ProfileAccumulator:
    """Builds a :class:`BehaviorProfile` from a stream of events.

    Streaming rather than batch on purpose. The offline fit replays events in time order and
    featurizes each one against the profile built from *strictly earlier* events, which is
    exactly what happens online. A batch build would let each event influence the baseline it
    is judged against -- a subtle leak that would make offline metrics unreachable in
    production.
    """

    def __init__(self, entity_id: str, entity_type: Optional[str] = None) -> None:
        self.entity_id = entity_id
        self.entity_type = entity_type

        self.event_count = 0
        self.sessions: set = set()
        self.first_seen: Optional[datetime] = None
        self.last_seen: Optional[datetime] = None

        self.hour_counts = [0.0] * 24
        self.dow_counts = [0.0] * 7
        self.country_counts: Dict[str, float] = {}
        self.ip_prefix_counts: Dict[str, float] = {}
        self.resource_counts: Dict[str, float] = {}
        self.auth_method_counts: Dict[str, float] = {}
        self.device_mac_counts: Dict[str, float] = {}
        self.device_os_counts: Dict[str, float] = {}
        self.protocol_counts: Dict[str, float] = {}
        self.token_counts: Dict[str, float] = {}
        self.ngram_counts: Dict[str, float] = {}

        self.geo_points: List[Tuple[float, float]] = []
        self.auth_failures = 0

        self.bytes_out_log = RunningStat()
        self.bytes_in_log = RunningStat()
        self.duration_log = RunningStat()
        self.interval_log = RunningStat()
        self.sequence_len = RunningStat()

        self._previous_timestamp: Optional[datetime] = None
        self._geo_sample_limit = 250  # bound the spread computation
        self._cached_profile: Optional[BehaviorProfile] = None
        self._cached_at_count: int = -1

    def update(self, event: Event) -> None:
        """Fold one event into the accumulator."""
        self.event_count += 1
        self.entity_type = self.entity_type or event.entity_type.value

        if event.session_id:
            self.sessions.add(event.session_id)

        timestamp = event.timestamp
        if self.first_seen is None or timestamp < self.first_seen:
            self.first_seen = timestamp
        if self.last_seen is None or timestamp > self.last_seen:
            self.last_seen = timestamp

        self.hour_counts[timestamp.hour] += 1.0
        self.dow_counts[timestamp.weekday()] += 1.0

        self.country_counts[event.geo.country] = self.country_counts.get(event.geo.country, 0.0) + 1.0
        prefix = ip_prefix(event.source_ip)
        self.ip_prefix_counts[prefix] = self.ip_prefix_counts.get(prefix, 0.0) + 1.0
        self.resource_counts[event.resource_accessed] = (
            self.resource_counts.get(event.resource_accessed, 0.0) + 1.0
        )
        method = event.auth_method.value
        self.auth_method_counts[method] = self.auth_method_counts.get(method, 0.0) + 1.0

        fingerprint = event.device_fingerprint
        self.device_mac_counts[fingerprint.mac] = self.device_mac_counts.get(fingerprint.mac, 0.0) + 1.0
        self.device_os_counts[fingerprint.os] = self.device_os_counts.get(fingerprint.os, 0.0) + 1.0
        self.protocol_counts[fingerprint.protocol] = (
            self.protocol_counts.get(fingerprint.protocol, 0.0) + 1.0
        )

        for token in event.command_sequence:
            self.token_counts[token] = self.token_counts.get(token, 0.0) + 1.0
        for gram in ngrams(list(event.command_sequence), settings.sequence_ngram_n):
            self.ngram_counts[gram] = self.ngram_counts.get(gram, 0.0) + 1.0

        if len(self.geo_points) < self._geo_sample_limit:
            self.geo_points.append((event.geo.lat, event.geo.lon))

        if not event.auth_success:
            self.auth_failures += 1

        self.bytes_out_log.update(math.log1p(max(0.0, event.bytes_out)))
        self.bytes_in_log.update(math.log1p(max(0.0, event.bytes_in)))
        self.duration_log.update(math.log1p(max(0.0, event.session_duration)))
        self.sequence_len.update(float(len(event.command_sequence)))

        if self._previous_timestamp is not None:
            gap = (timestamp - self._previous_timestamp).total_seconds()
            self.interval_log.update(math.log1p(max(0.0, gap)))
        self._previous_timestamp = timestamp

    def merge(self, other: "ProfileAccumulator") -> None:
        """Absorb another accumulator, for building cohort and global priors."""
        self.event_count += other.event_count
        self.sessions |= other.sessions
        self.auth_failures += other.auth_failures

        if other.first_seen and (self.first_seen is None or other.first_seen < self.first_seen):
            self.first_seen = other.first_seen
        if other.last_seen and (self.last_seen is None or other.last_seen > self.last_seen):
            self.last_seen = other.last_seen

        for index in range(24):
            self.hour_counts[index] += other.hour_counts[index]
        for index in range(7):
            self.dow_counts[index] += other.dow_counts[index]

        for attribute in (
            "country_counts",
            "ip_prefix_counts",
            "resource_counts",
            "auth_method_counts",
            "device_mac_counts",
            "device_os_counts",
            "protocol_counts",
            "token_counts",
            "ngram_counts",
        ):
            target: Dict[str, float] = getattr(self, attribute)
            for key, value in getattr(other, attribute).items():
                target[key] = target.get(key, 0.0) + value

        for attribute in ("bytes_out_log", "bytes_in_log", "duration_log", "interval_log", "sequence_len"):
            getattr(self, attribute).merge(getattr(other, attribute))

        remaining = self._geo_sample_limit - len(self.geo_points)
        if remaining > 0:
            self.geo_points.extend(other.geo_points[:remaining])

    def build(self, cohort: Optional[int] = None) -> BehaviorProfile:
        """Finalize into a :class:`BehaviorProfile`."""
        session_count = len(self.sessions)
        home = centroid(self.geo_points)
        min_sessions = settings.entity_history_min_sessions

        profile = BehaviorProfile(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            cohort=cohort,
            session_count=session_count,
            event_count=self.event_count,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            hour_hist=_normalize_list(self.hour_counts),
            dow_hist=_normalize_list(self.dow_counts),
            country_dist=_normalize(self.country_counts, top_k=20),
            ip_prefix_dist=_normalize(self.ip_prefix_counts, top_k=TOP_K_DEFAULT),
            resource_dist=_normalize(self.resource_counts, top_k=TOP_K_DEFAULT),
            auth_method_dist=_normalize(self.auth_method_counts, top_k=None),
            device_mac_dist=_normalize(self.device_mac_counts, top_k=12),
            device_os_dist=_normalize(self.device_os_counts, top_k=12),
            protocol_dist=_normalize(self.protocol_counts, top_k=None),
            token_dist=_normalize(self.token_counts, top_k=80),
            ngram_dist=_normalize(self.ngram_counts, top_k=160),
            home_lat=home[0] if home else None,
            home_lon=home[1] if home else None,
            geo_spread_km=(
                max_distance_from_km(self.geo_points, home[0], home[1]) if home else 0.0
            ),
            auth_failure_rate=(
                self.auth_failures / self.event_count if self.event_count else 0.0
            ),
            bytes_out_log=self.bytes_out_log,
            bytes_in_log=self.bytes_in_log,
            duration_log=self.duration_log,
            interval_log=self.interval_log,
            sequence_len=self.sequence_len,
            cold_start=session_count < min_sessions,
            confidence=min(1.0, session_count / max(1, min_sessions)),
        )
        return profile

    def build_cached(
        self,
        refresh_every: int = LIVE_PROFILE_REFRESH_EVENTS,
        cohort: Optional[int] = None,
    ) -> BehaviorProfile:
        """Finalized profile, rebuilt only every ``refresh_every`` events.

        See :data:`LIVE_PROFILE_REFRESH_EVENTS` for why this is safe. Callers that need an
        exactly-current profile (the offline artifact writer) should call :meth:`build` instead.
        """
        if (
            self._cached_profile is not None
            and (self.event_count - self._cached_at_count) < refresh_every
        ):
            return self._cached_profile

        self._cached_profile = self.build(cohort=cohort)
        self._cached_at_count = self.event_count
        return self._cached_profile


def _normalize_list(counts: Sequence[float]) -> List[float]:
    """Normalize a histogram to sum to 1, or return zeros if empty."""
    total = sum(counts)
    if total <= 0:
        return [0.0] * len(counts)
    return [value / total for value in counts]


def ip_prefix(source_ip: str, octets: int = 3) -> str:
    """The network portion of an address.

    Full addresses are effectively unique, so treating each one as a category would make every
    event novel. The /24 prefix is the useful granularity: it changes when someone connects
    from a genuinely different network.
    """
    parts = str(source_ip).split(".")
    if len(parts) < octets:
        return str(source_ip)
    return ".".join(parts[:octets])


# --------------------------------------------------------------------------- #
# Rolling window
# --------------------------------------------------------------------------- #


@dataclass
class EventSummary:
    """The fields of a past event that later events need. Kept small deliberately."""

    timestamp: datetime
    lat: float
    lon: float
    country: str
    resource: str
    auth_method: str
    auth_success: bool
    session_id: Optional[str]
    mac: str
    device_os: str
    protocol: str
    bytes_out: float
    ip_prefix: str

    @classmethod
    def from_event(cls, event: Event) -> "EventSummary":
        return cls(
            timestamp=event.timestamp,
            lat=event.geo.lat,
            lon=event.geo.lon,
            country=event.geo.country,
            resource=event.resource_accessed,
            auth_method=event.auth_method.value,
            auth_success=bool(event.auth_success),
            session_id=event.session_id,
            mac=event.device_fingerprint.mac,
            device_os=event.device_fingerprint.os,
            protocol=event.device_fingerprint.protocol,
            bytes_out=float(event.bytes_out),
            ip_prefix=ip_prefix(event.source_ip),
        )


class EntityState:
    """Short-term memory of one entity's recent activity.

    Holds a time-bounded window of recent events plus the immediately previous event. This is
    what makes rate, breadth and velocity features computable from a single event, and it is
    updated identically in the offline replay and the online scorer.
    """

    def __init__(
        self,
        entity_id: str,
        window_minutes: Optional[int] = None,
        max_events: int = MAX_WINDOW_EVENTS,
    ) -> None:
        self.entity_id = entity_id
        self.window = timedelta(
            minutes=settings.entity_window_minutes if window_minutes is None else window_minutes
        )
        self.max_events = max_events
        self.events: Deque[EventSummary] = deque()
        self.previous: Optional[EventSummary] = None
        self.total_events = 0
        self.sessions_seen: set = set()

    # -- maintenance -------------------------------------------------- #

    def prune(self, now: datetime) -> None:
        """Drop events that have fallen outside the window."""
        cutoff = now - self.window
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()
        while len(self.events) > self.max_events:
            self.events.popleft()

    def update(self, event: Event) -> None:
        """Record an event.

        Always called *after* featurizing it, so a feature never sees the event it describes in
        its own history.
        """
        summary = EventSummary.from_event(event)
        self.events.append(summary)
        # Prune after appending, not before: pruning first lets the deque reach max_events + 1,
        # so the cap would not actually hold.
        self.prune(event.timestamp)
        self.previous = summary
        self.total_events += 1
        if event.session_id:
            self.sessions_seen.add(event.session_id)

    # -- window measures ---------------------------------------------- #

    def window_events(self, now: datetime) -> List[EventSummary]:
        """Events inside the window as of ``now``, excluding the current event."""
        cutoff = now - self.window
        return [summary for summary in self.events if summary.timestamp >= cutoff]

    def velocity_since_previous(self, lat: float, lon: float, at: datetime) -> Optional[float]:
        """Implied travel speed since the previous event, or ``None`` if there is none."""
        if self.previous is None:
            return None
        return geo_velocity_kmh(
            self.previous.lat, self.previous.lon, self.previous.timestamp, lat, lon, at
        )

    def seconds_since_previous(self, at: datetime) -> Optional[float]:
        """Gap since the previous event in seconds, or ``None`` for a first event."""
        if self.previous is None:
            return None
        return max(0.0, (at - self.previous.timestamp).total_seconds())

    def session_events(self, session_id: Optional[str]) -> List[EventSummary]:
        """Events from the window belonging to one session."""
        if not session_id:
            return []
        return [summary for summary in self.events if summary.session_id == session_id]

    def is_known_session(self, session_id: Optional[str]) -> bool:
        """Whether this session has been seen before."""
        return bool(session_id) and session_id in self.sessions_seen


class StateStore:
    """Rolling windows for many entities, created on demand."""

    def __init__(self, window_minutes: Optional[int] = None) -> None:
        self.window_minutes = window_minutes
        self._states: Dict[str, EntityState] = {}

    def get(self, entity_id: str) -> EntityState:
        """Return this entity's state, creating an empty one if it is new."""
        state = self._states.get(entity_id)
        if state is None:
            state = EntityState(entity_id, window_minutes=self.window_minutes)
            self._states[entity_id] = state
        return state

    def __len__(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        self._states.clear()


# --------------------------------------------------------------------------- #
# Profile store with hierarchical cold-start fallback
# --------------------------------------------------------------------------- #


@dataclass
class ResolvedProfile:
    """The baseline actually used to score one event, plus how much to trust it."""

    profile: BehaviorProfile
    cold_start: bool
    confidence: float
    cohort: Optional[int]
    source: str  # "entity" | "entity+cohort" | "cohort" | "global"


class ProfileStore:
    """Per-entity baselines with a cohort-then-global fallback.

    Resolution order for an entity with thin history:

    1. its own profile, shrunk toward
    2. its cohort's prior, falling back to
    3. the global prior.

    Shrinkage rather than a hard switch: an entity with 5 sessions has *some* signal, and
    throwing it away in favour of a cohort average would be as wrong as trusting it fully. The
    blend weight is carried through as ``confidence`` and exposed as a feature, so the risk
    fusion tier can widen its uncertainty band for cold-start entities rather than guessing.
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, BehaviorProfile]] = None,
        cohort_priors: Optional[Dict[int, BehaviorProfile]] = None,
        global_prior: Optional[BehaviorProfile] = None,
        type_cohorts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.profiles: Dict[str, BehaviorProfile] = profiles or {}
        self.cohort_priors: Dict[int, BehaviorProfile] = cohort_priors or {}
        self.global_prior: BehaviorProfile = global_prior or BehaviorProfile(entity_id="__global__")
        #: Default cohort per entity type, used when a brand-new entity has no behavior yet.
        self.type_cohorts: Dict[str, int] = type_cohorts or {}

    # -- lookups ------------------------------------------------------- #

    def has(self, entity_id: str) -> bool:
        return entity_id in self.profiles

    def get(self, entity_id: str) -> Optional[BehaviorProfile]:
        return self.profiles.get(entity_id)

    def put(self, profile: BehaviorProfile) -> None:
        self.profiles[profile.entity_id] = profile

    def cohort_for(self, entity_id: str, entity_type: Optional[str] = None) -> Optional[int]:
        """Best available cohort for an entity, falling back to its entity type's default."""
        profile = self.profiles.get(entity_id)
        if profile is not None and profile.cohort is not None:
            return profile.cohort
        if entity_type and entity_type in self.type_cohorts:
            return self.type_cohorts[entity_type]
        return None

    def prior_for(
        self, cohort: Optional[int], entity_type: Optional[str] = None
    ) -> Tuple[BehaviorProfile, str]:
        """The prior to shrink toward, and a label describing where it came from."""
        if cohort is not None and cohort in self.cohort_priors:
            return self.cohort_priors[cohort], "cohort"
        if entity_type and entity_type in self.type_cohorts:
            fallback = self.type_cohorts[entity_type]
            if fallback in self.cohort_priors:
                return self.cohort_priors[fallback], "cohort"
        return self.global_prior, "global"

    # -- resolution ---------------------------------------------------- #

    def resolve(
        self,
        entity_id: str,
        entity_type: Optional[str] = None,
        live_profile: Optional[BehaviorProfile] = None,
    ) -> ResolvedProfile:
        """Produce the baseline to score an event against.

        Parameters
        ----------
        live_profile:
            A profile assembled from events seen so far in this process, used by the offline
            replay and by long-running online sessions. Preferred over the persisted profile
            when it has more evidence, so an entity that arrives mid-stream starts building its
            own baseline immediately instead of staying on cohort priors forever.
        """
        stored = self.profiles.get(entity_id)

        own = stored
        if live_profile is not None:
            if stored is None or live_profile.session_count >= stored.session_count:
                own = live_profile

        cohort = self.cohort_for(entity_id, entity_type)
        if own is not None and own.cohort is not None:
            cohort = own.cohort

        if own is None or own.event_count == 0:
            prior, source = self.prior_for(cohort, entity_type)
            resolved = prior.blend_with(prior, 1.0)  # copy semantics without mutation
            return ResolvedProfile(
                profile=resolved,
                cold_start=True,
                confidence=0.0,
                cohort=cohort,
                source=source,
            )

        weight = own.session_count / (own.session_count + PRIOR_STRENGTH_SESSIONS)
        min_sessions = settings.entity_history_min_sessions

        if own.session_count >= min_sessions:
            # Enough history of its own; no shrinkage needed.
            own.cohort = cohort if own.cohort is None else own.cohort
            own.cold_start = False
            own.confidence = 1.0
            return ResolvedProfile(
                profile=own, cold_start=False, confidence=1.0, cohort=cohort, source="entity"
            )

        prior, source = self.prior_for(cohort, entity_type)
        blended = own.blend_with(prior, weight)
        blended.cohort = cohort
        return ResolvedProfile(
            profile=blended,
            cold_start=True,
            confidence=weight,
            cohort=cohort,
            source=f"entity+{source}",
        )

    # -- persistence --------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {
                entity_id: profile.to_dict() for entity_id, profile in self.profiles.items()
            },
            "cohort_priors": {
                str(cohort): profile.to_dict() for cohort, profile in self.cohort_priors.items()
            },
            "global_prior": self.global_prior.to_dict(),
            "type_cohorts": dict(self.type_cohorts),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProfileStore":
        return cls(
            profiles={
                entity_id: BehaviorProfile.from_dict(value)
                for entity_id, value in (payload.get("entities") or {}).items()
            },
            cohort_priors={
                int(cohort): BehaviorProfile.from_dict(value)
                for cohort, value in (payload.get("cohort_priors") or {}).items()
            },
            global_prior=BehaviorProfile.from_dict(
                payload.get("global_prior") or {"entity_id": "__global__"}
            ),
            type_cohorts={
                key: int(value) for key, value in (payload.get("type_cohorts") or {}).items()
            },
        )

    def save(self, path) -> Any:
        import json
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1)
            handle.write("\n")
        return target

    @classmethod
    def load(cls, path) -> "ProfileStore":
        import json
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def build_profiles(
    events: Iterable[Event],
) -> Tuple[Dict[str, BehaviorProfile], Dict[str, ProfileAccumulator]]:
    """Build final per-entity profiles from a stream of events.

    Returns both the finished profiles and the raw accumulators, because cohort priors are
    built by *merging* accumulators rather than averaging finished profiles -- merging counts is
    exact, whereas averaging normalized distributions would over-weight low-volume entities.
    """
    accumulators: Dict[str, ProfileAccumulator] = {}
    for event in events:
        accumulator = accumulators.get(event.entity_id)
        if accumulator is None:
            accumulator = ProfileAccumulator(event.entity_id, event.entity_type.value)
            accumulators[event.entity_id] = accumulator
        accumulator.update(event)

    return (
        {entity_id: acc.build() for entity_id, acc in accumulators.items()},
        accumulators,
    )


__all__ = [
    "TOP_K_DEFAULT",
    "PRIOR_STRENGTH_SESSIONS",
    "MAX_WINDOW_EVENTS",
    "RunningStat",
    "BehaviorProfile",
    "ProfileAccumulator",
    "EventSummary",
    "EntityState",
    "StateStore",
    "ResolvedProfile",
    "ProfileStore",
    "build_profiles",
    "ip_prefix",
]
