"""Per-event feature computation.

Every feature here answers one question: **how unusual is this for this entity?** Almost none
of them are absolute. `bytes_out = 4 MB` means nothing on its own; `bytes_out is 3.4 standard
deviations above what this service account normally sends` is a signal.

That framing drives three conventions:

* **Likelihoods, not one-hot membership.** Instead of "has this entity used this country
  before", features carry the *probability* of the value under the entity's learned
  distribution. A country used 2% of the time is different from one used 60% of the time, and
  a binary flag throws that away.
* **Log scaling for heavy tails.** Bytes, durations and inter-event gaps span orders of
  magnitude. Untransformed, a single large transfer would dominate the scaled vector.
* **No hardcoded domain lists.** Resource sensitivity is *learned* from the corpus (how rare
  is it, how few entities touch it) rather than matched against a list of paths. A hardcoded
  list would be the generator's knowledge leaking into the detector, and would not transfer to
  real telemetry at all.

Features are returned raw and unscaled; :mod:`features.encoders` standardizes them later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from common.models import Event
from features.entity_window import (
    BehaviorProfile,
    EntityState,
    ResolvedProfile,
    ip_prefix,
)
from features.geo import haversine_km
from features.sequences import SequenceVocab, profile_ngram_novelty

#: Ordered numeric feature names. This order is a contract: it defines column positions in the
#: model matrices, the scaler and the SHAP output, so append rather than reorder.
NUMERIC_FEATURE_NAMES: List[str] = [
    # --- temporal ---
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "hour_likelihood",
    "dow_likelihood",
    "log_seconds_since_prev",
    "is_first_event",
    # --- geo / network ---
    "log_geo_velocity_kmh",
    "log_distance_from_prev_km",
    "log_distance_from_home_km",
    "country_likelihood",
    "is_new_country",
    "ip_prefix_likelihood",
    "is_new_ip_prefix",
    # --- resource / access ---
    "resource_likelihood",
    "is_new_resource",
    "resource_global_rarity",
    "resource_entity_share",
    "window_distinct_resources",
    "window_foreign_resource_ratio",
    # --- authentication ---
    "auth_method_likelihood",
    "is_new_auth_method",
    "auth_success",
    "window_auth_failures",
    "window_auth_failure_ratio",
    "auth_failure_rate_delta",
    # --- command sequence ---
    "sequence_length",
    "sequence_length_zscore",
    "seq_token_novelty_entity",
    "seq_ngram_novelty_entity",
    "seq_ngram_novelty_global",
    "seq_token_rarity_global",
    # --- device ---
    "is_new_device_mac",
    "is_new_device_os",
    "protocol_likelihood",
    "device_changed_from_prev",
    "device_os_global_rarity",
    # --- volume ---
    "log_bytes_out",
    "bytes_out_zscore",
    "log_bytes_in",
    "bytes_in_zscore",
    "log_session_duration",
    "session_duration_zscore",
    "log_window_bytes_out",
    "window_bytes_out_ratio",
    # --- rate ---
    "window_event_count",
    "window_events_per_minute",
    "window_distinct_sessions",
    "interval_zscore",
    # --- session ---
    "session_event_index",
    "session_distinct_resources",
    "session_auth_failures",
    "is_new_session",
    # --- profile meta ---
    "log_entity_session_count",
    "log_entity_event_count",
    "cold_start",
    "profile_confidence",
]

#: Categorical fields, encoded to integer codes and appended after the numeric block.
CATEGORICAL_FEATURE_NAMES: List[str] = [
    "entity_type",
    "auth_method",
    "geo_country",
    "device_protocol",
    "device_os",
    "resource_accessed",
    "cohort",
]


@dataclass
class CorpusStats:
    """Corpus-wide statistics that make "rare" and "narrowly used" measurable.

    Fitted from the training split. Two things a per-entity profile cannot express:

    * ``resource_frequency`` -- how common a resource is across the whole organization.
    * ``resource_entity_share`` -- what fraction of entities ever touch it. A resource used by
      1% of entities is structurally sensitive; this is the learned replacement for a
      hand-written list of "sensitive paths".
    """

    n_events: int = 0
    n_entities: int = 0
    resource_frequency: Dict[str, float] = field(default_factory=dict)
    resource_entity_share: Dict[str, float] = field(default_factory=dict)
    device_os_frequency: Dict[str, float] = field(default_factory=dict)
    country_frequency: Dict[str, float] = field(default_factory=dict)

    def rarity(self, distribution: Dict[str, float], key: Optional[str]) -> float:
        """Surprisal of a value under a corpus distribution, in nats.

        An unseen value is scored as if it occurred once in the corpus, which keeps the value
        finite. An unbounded rarity would swamp every other feature after scaling.
        """
        if key is None:
            return 0.0
        floor = 1.0 / max(1, self.n_events)
        share = distribution.get(key, 0.0)
        return -math.log(max(share, floor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_events": self.n_events,
            "n_entities": self.n_entities,
            "resource_frequency": self.resource_frequency,
            "resource_entity_share": self.resource_entity_share,
            "device_os_frequency": self.device_os_frequency,
            "country_frequency": self.country_frequency,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CorpusStats":
        return cls(
            n_events=int(payload.get("n_events", 0)),
            n_entities=int(payload.get("n_entities", 0)),
            resource_frequency=dict(payload.get("resource_frequency") or {}),
            resource_entity_share=dict(payload.get("resource_entity_share") or {}),
            device_os_frequency=dict(payload.get("device_os_frequency") or {}),
            country_frequency=dict(payload.get("country_frequency") or {}),
        )

    @classmethod
    def fit(cls, events: List[Event]) -> "CorpusStats":
        """Compute corpus statistics from training events."""
        resource_counts: Dict[str, float] = {}
        resource_entities: Dict[str, set] = {}
        os_counts: Dict[str, float] = {}
        country_counts: Dict[str, float] = {}
        entities: set = set()

        for event in events:
            entities.add(event.entity_id)
            resource_counts[event.resource_accessed] = (
                resource_counts.get(event.resource_accessed, 0.0) + 1.0
            )
            resource_entities.setdefault(event.resource_accessed, set()).add(event.entity_id)
            fingerprint_os = event.device_fingerprint.os
            os_counts[fingerprint_os] = os_counts.get(fingerprint_os, 0.0) + 1.0
            country_counts[event.geo.country] = country_counts.get(event.geo.country, 0.0) + 1.0

        total = float(len(events)) or 1.0
        entity_total = float(len(entities)) or 1.0

        return cls(
            n_events=len(events),
            n_entities=len(entities),
            resource_frequency={
                key: value / total for key, value in resource_counts.items()
            },
            resource_entity_share={
                key: len(members) / entity_total for key, members in resource_entities.items()
            },
            device_os_frequency={key: value / total for key, value in os_counts.items()},
            country_frequency={key: value / total for key, value in country_counts.items()},
        )


def _safe_log1p(value: Optional[float]) -> float:
    """``log1p`` that tolerates ``None`` and negative inputs."""
    if value is None:
        return 0.0
    return math.log1p(max(0.0, float(value)))


def compute_event_features(
    event: Event,
    resolved: ResolvedProfile,
    state: EntityState,
    vocab: Optional[SequenceVocab] = None,
    corpus: Optional[CorpusStats] = None,
) -> Dict[str, float]:
    """Compute every numeric feature for one event.

    Parameters
    ----------
    resolved:
        The baseline to compare against, already blended for cold start.
    state:
        The entity's rolling window **as of before this event**. The caller must update the
        state afterwards, never before, or a feature would see the event in its own history.

    Returns
    -------
    dict
        Raw, unscaled values keyed by :data:`NUMERIC_FEATURE_NAMES`.
    """
    profile: BehaviorProfile = resolved.profile
    corpus = corpus or CorpusStats()
    timestamp: datetime = event.timestamp
    features: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Temporal
    # ------------------------------------------------------------------ #
    hour = timestamp.hour + timestamp.minute / 60.0
    weekday = timestamp.weekday()

    # Cyclical encoding: hour 23 and hour 0 are adjacent, which a raw integer cannot express.
    features["hour_sin"] = math.sin(2.0 * math.pi * hour / 24.0)
    features["hour_cos"] = math.cos(2.0 * math.pi * hour / 24.0)
    features["dow_sin"] = math.sin(2.0 * math.pi * weekday / 7.0)
    features["dow_cos"] = math.cos(2.0 * math.pi * weekday / 7.0)
    features["is_weekend"] = 1.0 if weekday >= 5 else 0.0
    features["hour_likelihood"] = profile.hour_likelihood(timestamp.hour)
    features["dow_likelihood"] = profile.dow_likelihood(weekday)

    gap_seconds = state.seconds_since_previous(timestamp)
    features["log_seconds_since_prev"] = _safe_log1p(gap_seconds)
    features["is_first_event"] = 1.0 if gap_seconds is None else 0.0

    # ------------------------------------------------------------------ #
    # Geo / network
    # ------------------------------------------------------------------ #
    velocity = state.velocity_since_previous(event.geo.lat, event.geo.lon, timestamp)
    features["log_geo_velocity_kmh"] = _safe_log1p(velocity)

    if state.previous is not None:
        previous_distance = haversine_km(
            state.previous.lat, state.previous.lon, event.geo.lat, event.geo.lon
        )
    else:
        previous_distance = 0.0
    features["log_distance_from_prev_km"] = _safe_log1p(previous_distance)
    features["log_distance_from_home_km"] = _safe_log1p(
        profile.distance_from_home_km(event.geo.lat, event.geo.lon)
    )

    features["country_likelihood"] = profile.likelihood(profile.country_dist, event.geo.country)
    features["is_new_country"] = 1.0 if profile.is_new(profile.country_dist, event.geo.country) else 0.0

    prefix = ip_prefix(event.source_ip)
    features["ip_prefix_likelihood"] = profile.likelihood(profile.ip_prefix_dist, prefix)
    features["is_new_ip_prefix"] = 1.0 if profile.is_new(profile.ip_prefix_dist, prefix) else 0.0

    # ------------------------------------------------------------------ #
    # Resource / access
    # ------------------------------------------------------------------ #
    resource = event.resource_accessed
    features["resource_likelihood"] = profile.likelihood(profile.resource_dist, resource)
    features["is_new_resource"] = 1.0 if profile.is_new(profile.resource_dist, resource) else 0.0
    features["resource_global_rarity"] = corpus.rarity(corpus.resource_frequency, resource)
    features["resource_entity_share"] = corpus.resource_entity_share.get(resource, 0.0)

    window = state.window_events(timestamp)
    window_resources = {summary.resource for summary in window}
    window_resources.add(resource)
    features["window_distinct_resources"] = float(len(window_resources))

    # Breadth *outside* the entity's usual set is the lateral-movement signal: touching many
    # resources is normal for some roles, touching many unfamiliar ones is not.
    if profile.resource_dist:
        foreign = sum(1 for name in window_resources if name not in profile.resource_dist)
        features["window_foreign_resource_ratio"] = foreign / len(window_resources)
    else:
        features["window_foreign_resource_ratio"] = 0.0

    # ------------------------------------------------------------------ #
    # Authentication
    # ------------------------------------------------------------------ #
    method = event.auth_method.value
    features["auth_method_likelihood"] = profile.likelihood(profile.auth_method_dist, method)
    features["is_new_auth_method"] = 1.0 if profile.is_new(profile.auth_method_dist, method) else 0.0
    features["auth_success"] = 1.0 if event.auth_success else 0.0

    window_failures = sum(1 for summary in window if not summary.auth_success)
    if not event.auth_success:
        window_failures += 1
    features["window_auth_failures"] = float(window_failures)
    window_size = len(window) + 1
    window_failure_ratio = window_failures / window_size
    features["window_auth_failure_ratio"] = window_failure_ratio
    # How far the current burst departs from this entity's habitual failure rate. A service
    # account that never fails is more suspicious at 3 failures than a user who often typos.
    features["auth_failure_rate_delta"] = window_failure_ratio - profile.auth_failure_rate

    # ------------------------------------------------------------------ #
    # Command sequence
    # ------------------------------------------------------------------ #
    tokens = [token for token in event.command_sequence if token]
    features["sequence_length"] = float(len(tokens))
    features["sequence_length_zscore"] = profile.sequence_len.zscore(float(len(tokens)))

    if tokens and profile.token_dist:
        unseen_tokens = sum(1 for token in tokens if token not in profile.token_dist)
        features["seq_token_novelty_entity"] = unseen_tokens / len(tokens)
    else:
        features["seq_token_novelty_entity"] = 0.0

    ngram_n = vocab.ngram_n if vocab else 2
    features["seq_ngram_novelty_entity"] = (
        profile_ngram_novelty(tokens, profile.ngram_dist, ngram_n) if profile.ngram_dist else 0.0
    )
    features["seq_ngram_novelty_global"] = vocab.ngram_novelty(tokens) if vocab else 0.0
    features["seq_token_rarity_global"] = vocab.mean_token_rarity(tokens) if vocab else 0.0

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    fingerprint = event.device_fingerprint
    features["is_new_device_mac"] = (
        1.0 if profile.is_new(profile.device_mac_dist, fingerprint.mac) else 0.0
    )
    features["is_new_device_os"] = (
        1.0 if profile.is_new(profile.device_os_dist, fingerprint.os) else 0.0
    )
    features["protocol_likelihood"] = profile.likelihood(profile.protocol_dist, fingerprint.protocol)
    features["device_changed_from_prev"] = (
        1.0 if state.previous is not None and state.previous.mac != fingerprint.mac else 0.0
    )
    features["device_os_global_rarity"] = corpus.rarity(corpus.device_os_frequency, fingerprint.os)

    # ------------------------------------------------------------------ #
    # Volume
    # ------------------------------------------------------------------ #
    log_bytes_out = _safe_log1p(event.bytes_out)
    log_bytes_in = _safe_log1p(event.bytes_in)
    log_duration = _safe_log1p(event.session_duration)

    features["log_bytes_out"] = log_bytes_out
    features["bytes_out_zscore"] = profile.bytes_out_log.zscore(log_bytes_out)
    features["log_bytes_in"] = log_bytes_in
    features["bytes_in_zscore"] = profile.bytes_in_log.zscore(log_bytes_in)
    features["log_session_duration"] = log_duration
    features["session_duration_zscore"] = profile.duration_log.zscore(log_duration)

    window_bytes_out = sum(summary.bytes_out for summary in window) + float(event.bytes_out)
    features["log_window_bytes_out"] = _safe_log1p(window_bytes_out)
    # Sustained volume relative to this entity's typical single-event volume. This is what makes
    # low-and-slow exfiltration visible: each event is ordinary, the running total is not.
    typical = math.expm1(profile.bytes_out_log.mean) if profile.bytes_out_log.count else 0.0
    features["window_bytes_out_ratio"] = (
        window_bytes_out / typical if typical > 1.0 else 0.0
    )

    # ------------------------------------------------------------------ #
    # Rate
    # ------------------------------------------------------------------ #
    features["window_event_count"] = float(window_size)
    window_minutes = max(1.0, state.window.total_seconds() / 60.0)
    features["window_events_per_minute"] = window_size / window_minutes
    window_sessions = {summary.session_id for summary in window if summary.session_id}
    if event.session_id:
        window_sessions.add(event.session_id)
    features["window_distinct_sessions"] = float(len(window_sessions))
    features["interval_zscore"] = (
        profile.interval_log.zscore(_safe_log1p(gap_seconds)) if gap_seconds is not None else 0.0
    )

    # ------------------------------------------------------------------ #
    # Profile meta
    # ------------------------------------------------------------------ #
    features["log_entity_session_count"] = _safe_log1p(profile.session_count)
    features["log_entity_event_count"] = _safe_log1p(profile.event_count)
    features["cold_start"] = 1.0 if resolved.cold_start else 0.0
    features["profile_confidence"] = float(resolved.confidence)

    return features


def categorical_values(event: Event, cohort: Optional[int]) -> Dict[str, Any]:
    """The categorical fields for one event, ready for encoding."""
    return {
        "entity_type": event.entity_type.value,
        "auth_method": event.auth_method.value,
        "geo_country": event.geo.country,
        "device_protocol": event.device_fingerprint.protocol,
        "device_os": event.device_fingerprint.os,
        "resource_accessed": event.resource_accessed,
        "cohort": "unknown" if cohort is None else str(cohort),
    }


__all__ = [
    "NUMERIC_FEATURE_NAMES",
    "CATEGORICAL_FEATURE_NAMES",
    "CorpusStats",
    "compute_event_features",
    "categorical_values",
]
