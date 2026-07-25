"""Benign session and event generation.

This is the most important module in the generator. Everything the models learn about
"normal" comes from here, and the quality of the whole project rests on this traffic being
*plausibly messy*: if benign behavior is too clean, every attack stands out and the reported
metrics are meaningless.

So benign traffic deliberately includes the things that make real detection hard:

* occasional legitimate off-hours access,
* occasional legitimate travel to a second office,
* occasional first-time access to a new resource,
* occasional failed logins (mistyped passwords),
* rare legitimate access to sensitive resources,
* command sequences that vary in length and order rather than repeating a fixed template,
* gradual benign drift for a subset of entities (see :mod:`data_generator.drift`).

Every event produced here is labeled ``normal``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.models import (
    AnomalyType,
    AuthMethod,
    DeviceFingerprint,
    Event,
    GeoLocation,
)
from data_generator.drift import (
    drifted_resource_pool,
    effective_city,
    effective_device,
    effective_hour_weights,
)
from data_generator.profiles import (
    City,
    EntityGenProfile,
    GeneratorConfig,
    World,
)

#: Probability a session starts outside the entity's normal hours despite the schedule --
#: real people occasionally work late. Without this, "off-hours" alone would perfectly
#: separate attacks from normal traffic.
OFF_HOURS_PROBABILITY = 0.035

#: Probability a session touches a resource the entity has never used before. Novelty must
#: be suspicious-but-not-conclusive.
NEW_RESOURCE_PROBABILITY = 0.030

#: Probability a benign session includes a legitimate sensitive-resource access, for
#: cohorts that have any. Set high enough that a sensitive resource name is *not* close to a
#: label by itself: what should matter is which entity touched it, when, and from where --
#: not the resource in isolation.
SENSITIVE_ACCESS_PROBABILITY = 0.060


def _sample_hour(
    hour_weights: Sequence[float], rng: np.random.Generator
) -> int:
    """Draw an hour of day from the entity's activity distribution."""
    return int(rng.choice(24, p=np.asarray(hour_weights, dtype=float)))


def _jitter_geo(city: City, rng: np.random.Generator) -> GeoLocation:
    """City-level location with a little positional noise.

    Coordinates in real telemetry come from IP geolocation, which is never exact. The jitter
    keeps geo features from being perfectly discrete lookups.
    """
    return GeoLocation(
        country=city.country,
        city=city.name,
        lat=float(np.clip(city.lat + rng.normal(0.0, 0.045), -90.0, 90.0)),
        lon=float(np.clip(city.lon + rng.normal(0.0, 0.045), -180.0, 180.0)),
    )


def _sample_ip(entity: EntityGenProfile, rng: np.random.Generator) -> str:
    """An address from the entity's usual subnet."""
    return f"{entity.ip_prefix}{int(rng.integers(2, 254))}"


def _sample_auth_method(entity: EntityGenProfile, rng: np.random.Generator) -> AuthMethod:
    """Draw an auth method from the entity's personal distribution."""
    methods = list(entity.auth_weights.keys())
    probabilities = np.asarray([entity.auth_weights[m] for m in methods], dtype=float)
    probabilities = probabilities / probabilities.sum()
    return methods[int(rng.choice(len(methods), p=probabilities))]


def _sample_resources(
    entity: EntityGenProfile,
    world: World,
    moment: datetime,
    count: int,
    rng: np.random.Generator,
) -> List[str]:
    """Pick the resources touched in one session.

    Draws from the entity's own weighted pool, extended by any resources it has
    legitimately taken on through drift, with small probabilities of genuine novelty and of
    a rare sensitive access.
    """
    pool = dict(entity.resource_weights)
    for resource in drifted_resource_pool(entity, moment):
        pool.setdefault(resource, 0.12)

    names = list(pool.keys())
    weights = np.asarray([pool[name] for name in names], dtype=float)
    weights = weights / weights.sum()

    chosen = [names[int(rng.choice(len(names), p=weights))] for _ in range(count)]

    cohort = entity.cohort
    if cohort.rare_sensitive and rng.random() < SENSITIVE_ACCESS_PROBABILITY:
        pick = cohort.rare_sensitive[int(rng.integers(0, len(cohort.rare_sensitive)))]
        chosen[int(rng.integers(0, len(chosen)))] = pick

    if rng.random() < NEW_RESOURCE_PROBABILITY:
        novel = world.foreign_resources(entity, rng, count=1)
        if novel:
            chosen[int(rng.integers(0, len(chosen)))] = novel[0]

    return chosen


def _sample_commands(
    entity: EntityGenProfile, length: int, rng: np.random.Generator
) -> List[str]:
    """Build a command sequence with realistic variation.

    Starts from one of the cohort's templates, then perturbs it: templates give the sequence
    model learnable n-gram structure, and the perturbations stop it from memorizing a tiny
    set of exact strings.
    """
    templates = entity.cohort.command_templates
    template = list(templates[int(rng.integers(0, len(templates)))])

    # Repeat a middle step now and then (a user re-reading a document, a service retrying).
    if len(template) > 2 and rng.random() < 0.30:
        position = int(rng.integers(1, len(template) - 1))
        template.insert(position, template[position])

    # Occasionally drop a middle step (a session that ends without a clean logout).
    if len(template) > 3 and rng.random() < 0.18:
        template.pop(int(rng.integers(1, len(template) - 1)))

    # Occasionally swap two adjacent middle steps.
    if len(template) > 3 and rng.random() < 0.12:
        position = int(rng.integers(1, len(template) - 2))
        template[position], template[position + 1] = template[position + 1], template[position]

    if length <= len(template):
        return template[:length]

    # Pad by cycling the middle of the template, keeping the terminal step last.
    body = template[1:-1] or template
    padded = template[:-1]
    while len(padded) < length - 1:
        padded.append(body[len(padded) % len(body)])
    padded.append(template[-1])
    return padded[:length]


def generate_session(
    entity: EntityGenProfile,
    world: World,
    started_at: datetime,
    rng: np.random.Generator,
    session_index: int,
) -> List[Event]:
    """Generate all events for one benign session.

    A session is a coherent unit: one location, one device, one auth method, one shared
    duration, and an ordered command sequence spread across its events.
    """
    cohort = entity.cohort
    low, high = cohort.events_per_session
    event_count = int(rng.integers(low, high + 1))

    city = effective_city(entity, started_at, rng)
    device = effective_device(entity, started_at, rng)
    geo = _jitter_geo(city, rng)

    # Legitimate travel: connect from the secondary office instead of home.
    if cohort.travel_probability > 0 and rng.random() < cohort.travel_probability:
        geo = _jitter_geo(entity.secondary_city, rng)

    source_ip = _sample_ip(entity, rng)
    auth_method = _sample_auth_method(entity, rng)

    mu, sigma = entity.session_seconds
    duration = float(np.clip(rng.lognormal(mu, sigma), 5.0, 6 * 3600.0))

    resources = _sample_resources(entity, world, started_at, event_count, rng)
    commands = _sample_commands(entity, event_count, rng)

    session_id = f"ses_{entity.entity_id}_{session_index:05d}"
    bytes_mu, bytes_sigma = entity.bytes_out_lognormal

    fingerprint = DeviceFingerprint(**device.as_dict())

    events: List[Event] = []
    # A mistyped password: one or two failures immediately before a successful login.
    failed_prefix = 0
    if rng.random() < entity.auth_fail_rate:
        failed_prefix = int(rng.integers(1, 3))

    offset_seconds = 0.0
    for attempt in range(failed_prefix):
        events.append(
            Event(
                event_id=f"evt_{session_id}_f{attempt}",
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                timestamp=started_at + timedelta(seconds=offset_seconds),
                source_ip=source_ip,
                geo=geo,
                resource_accessed="/auth/login",
                auth_method=auth_method,
                auth_success=False,
                session_id=session_id,
                session_duration=duration,
                command_sequence=["auth_attempt"],
                device_fingerprint=fingerprint,
                bytes_out=float(rng.uniform(200, 900)),
                bytes_in=float(rng.uniform(200, 900)),
                label=AnomalyType.NORMAL,
            )
        )
        offset_seconds += float(rng.uniform(3.0, 25.0))

    # Spread the session's events across its duration in time order.
    step = duration / max(1, event_count)
    for index in range(event_count):
        offset_seconds += step * float(rng.uniform(0.55, 1.45))
        events.append(
            Event(
                event_id=f"evt_{session_id}_{index:03d}",
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                timestamp=started_at + timedelta(seconds=offset_seconds),
                source_ip=source_ip,
                geo=geo,
                resource_accessed=resources[index],
                auth_method=auth_method,
                auth_success=True,
                session_id=session_id,
                session_duration=duration,
                # Each event carries the sequence up to and including its own step, so a
                # single event is scoreable on its own while the order is still visible.
                command_sequence=commands[: index + 1][-8:],
                device_fingerprint=fingerprint,
                bytes_out=float(np.clip(rng.lognormal(bytes_mu, bytes_sigma), 64.0, 5e7)),
                bytes_in=float(np.clip(rng.lognormal(bytes_mu - 0.8, bytes_sigma), 64.0, 2e7)),
                label=AnomalyType.NORMAL,
            )
        )

    return events


def _session_start_times(
    entity: EntityGenProfile,
    day_start: datetime,
    is_weekend: bool,
    rng: np.random.Generator,
) -> List[datetime]:
    """Choose when this entity's sessions begin on one day."""
    cohort = entity.cohort
    rate = entity.sessions_per_day
    if is_weekend:
        rate *= cohort.weekend_activity

    session_count = int(rng.poisson(max(0.0, rate)))
    if session_count == 0:
        return []

    hour_weights = effective_hour_weights(entity, day_start)
    starts: List[datetime] = []
    for _ in range(session_count):
        if rng.random() < OFF_HOURS_PROBABILITY:
            hour = int(rng.integers(0, 24))  # genuine off-hours work
        else:
            hour = _sample_hour(hour_weights, rng)
        starts.append(
            day_start
            + timedelta(
                hours=hour,
                minutes=int(rng.integers(0, 60)),
                seconds=int(rng.integers(0, 60)),
            )
        )
    return sorted(starts)


def generate_benign_events(
    world: World,
    rng: np.random.Generator,
    config: Optional[GeneratorConfig] = None,
) -> List[Event]:
    """Generate the full benign event stream for every entity across the timeline.

    Returns
    -------
    list of Event
        Time-sorted benign events, each labeled ``normal``.
    """
    config = config or world.config
    events: List[Event] = []

    for entity in world.entities:
        session_index = 0
        for day_offset in range(config.resolved_days()):
            day_start = config.start_date + timedelta(days=day_offset)

            # Cold-start entities simply do not exist before they are onboarded.
            if day_start + timedelta(days=1) <= entity.active_from:
                continue

            is_weekend = day_start.weekday() >= 5
            for started_at in _session_start_times(entity, day_start, is_weekend, rng):
                if started_at < entity.active_from:
                    continue
                events.extend(
                    generate_session(entity, world, started_at, rng, session_index)
                )
                session_index += 1

    events.sort(key=lambda event: event.timestamp)
    return events


def benign_summary(events: Sequence[Event]) -> Dict[str, float]:
    """Descriptive statistics used by TAXONOMY.md and the generator's console report."""
    if not events:
        return {"n_events": 0}

    failures = sum(1 for event in events if not event.auth_success)
    off_hours = sum(1 for event in events if event.timestamp.hour < 6 or event.timestamp.hour >= 22)
    countries = {event.geo.country for event in events}
    entities = {event.entity_id for event in events}
    sessions = {event.session_id for event in events}

    return {
        "n_events": len(events),
        "n_entities": len(entities),
        "n_sessions": len(sessions),
        "auth_failure_rate": failures / len(events),
        "off_hours_rate": off_hours / len(events),
        "n_countries": len(countries),
        "mean_session_duration": float(
            np.mean([event.session_duration for event in events])
        ),
    }


__all__ = [
    "OFF_HOURS_PROBABILITY",
    "NEW_RESOURCE_PROBABILITY",
    "SENSITIVE_ACCESS_PROBABILITY",
    "generate_session",
    "generate_benign_events",
    "benign_summary",
]
