"""Attack injectors -- one per anomaly class.

Eight behaviors are injected into the benign stream. The plan's summary lists seven; the
label space in section 7.1 has eight attack classes, so ``credential_misuse`` is included
here as well. Every class in :data:`common.models.ATTACK_CLASSES` has an injector, which is
what lets the supervised classifier learn all of them.

Two design rules govern this module.

**Each class has a distinct primary signal.** Otherwise the classifier would be guessing
between overlapping classes and macro-F1 would be unreachable regardless of model quality:

===========================  =========================================================
Class                        Primary signal
===========================  =========================================================
``brute_force``              Failed-auth burst against one entity in a short window
``credential_stuffing``      One source IP spraying many entities, few attempts each
``impossible_travel``        Geo velocity between consecutive events exceeds physics
``credential_misuse``        Valid auth, but several behavioral facets deviate at once
``lateral_movement``         Access breadth across resources outside the entity's cohort
``device_spoofing``          Device fingerprint inconsistent with the entity's history
``low_and_slow_exfil``       Sustained small elevated transfers over many hours
``insider_drift``            Gradual, self-directed shift toward sensitive resources
===========================  =========================================================

**Subtlety is a single dial.** ``config.subtlety`` in ``[0, 1]`` interpolates every injector
between blatant and near-invisible. If the dataset turns out too easy (inflated metrics) or
too hard, that one number is tuned rather than eight separate injectors -- and the value
used is recorded in ``TAXONOMY.md`` and the dataset metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.models import (
    ANOMALY_CLASSES,
    AnomalyType,
    AuthMethod,
    DeviceFingerprint,
    EntityType,
    Event,
    GeoLocation,
)
from data_generator.profiles import (
    HOSTILE_CITIES,
    HOSTILE_COMMANDS,
    City,
    DeviceSpec,
    EntityGenProfile,
    GeneratorConfig,
    World,
    random_mac,
    user_agent_for,
)

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres.

    Used here only to pick a city far enough away that the travel is physically impossible.
    The canonical implementation used for *features* lives in ``features/geo.py``.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Incident container
# --------------------------------------------------------------------------- #


@dataclass
class AttackIncident:
    """One injected attack: a labeled, time-ordered burst of events for one entity."""

    anomaly_type: AnomalyType
    entity_id: str
    events: List[Event] = field(default_factory=list)
    campaign_id: Optional[str] = None
    stage: Optional[int] = None

    @property
    def started_at(self) -> datetime:
        """Timestamp of the first event in the incident."""
        return min(event.timestamp for event in self.events)

    @property
    def ended_at(self) -> datetime:
        """Timestamp of the last event in the incident."""
        return max(event.timestamp for event in self.events)

    def tag_campaign(self, campaign_id: str, stage: int) -> None:
        """Attach campaign ground truth to every event in the incident (D1)."""
        self.campaign_id = campaign_id
        self.stage = stage
        for event in self.events:
            event.campaign_id = campaign_id
            event.stage = stage


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _lerp(blatant: float, subtle: float, subtlety: float) -> float:
    """Interpolate a parameter between its blatant and subtle extremes."""
    return blatant + (subtle - blatant) * float(np.clip(subtlety, 0.0, 1.0))


def _hostile_city(rng: np.random.Generator) -> City:
    """Pick an attack origin."""
    return HOSTILE_CITIES[int(rng.integers(0, len(HOSTILE_CITIES)))]


def _distant_city(entity: EntityGenProfile, rng: np.random.Generator, min_km: float) -> City:
    """Pick a hostile city at least ``min_km`` from the entity's home.

    Guarantees the impossible-travel injector actually produces impossible travel rather
    than relying on chance.
    """
    home = entity.home_city
    candidates = [
        city
        for city in HOSTILE_CITIES
        if haversine_km(home.lat, home.lon, city.lat, city.lon) >= min_km
    ]
    if not candidates:
        candidates = sorted(
            HOSTILE_CITIES,
            key=lambda city: -haversine_km(home.lat, home.lon, city.lat, city.lon),
        )[:2]
    return candidates[int(rng.integers(0, len(candidates)))]


def _geo(city: City, rng: np.random.Generator) -> GeoLocation:
    """City-level geolocation with small positional noise."""
    return GeoLocation(
        country=city.country,
        city=city.name,
        lat=float(np.clip(city.lat + rng.normal(0.0, 0.05), -90.0, 90.0)),
        lon=float(np.clip(city.lon + rng.normal(0.0, 0.05), -180.0, 180.0)),
    )


def _hostile_ip(rng: np.random.Generator) -> str:
    """A public-looking source address for attacker traffic."""
    blocks = (45, 89, 91, 103, 185, 193, 196, 197)
    first = blocks[int(rng.integers(0, len(blocks)))]
    return f"{first}.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"


def _hostile_device(entity: EntityGenProfile, rng: np.random.Generator) -> DeviceSpec:
    """An unfamiliar device, plausible for the entity's protocol but not its own."""
    attacker_os = ("Kali Linux 2024.1", "Windows Server 2019", "Ubuntu 20.04", "Android 13")
    os_name = attacker_os[int(rng.integers(0, len(attacker_os)))]
    protocols = entity.cohort.protocols
    return DeviceSpec(
        os=os_name,
        mac=random_mac(rng),
        protocol=protocols[int(rng.integers(0, len(protocols)))],
        user_agent=user_agent_for(os_name),
    )


def _unfamiliar_device(entity: EntityGenProfile, rng: np.random.Generator) -> DeviceSpec:
    """A device that is plausible for this cohort but has never been seen for this entity.

    Used where the attacker is trying to blend in. A blatantly hostile OS string (``Kali
    Linux``) would be a *global* giveaway learnable from one event without any behavioral
    profiling -- which is exactly the shortcut we do not want the models to take. An
    unfamiliar MAC on an ordinary corporate OS can only be spotted against the entity's own
    device history.
    """
    cohort = entity.cohort
    os_name = cohort.os_pool[int(rng.integers(0, len(cohort.os_pool)))]
    return DeviceSpec(
        os=os_name,
        mac=random_mac(rng),
        protocol=entity.primary_device().protocol,
        user_agent=user_agent_for(os_name),
    )


def _entity_device(entity: EntityGenProfile) -> DeviceSpec:
    """The entity's own primary device."""
    return entity.primary_device()


def _off_hours_moment(base: datetime, rng: np.random.Generator) -> datetime:
    """Move a timestamp into the small hours, where activity is genuinely unusual."""
    return base.replace(
        hour=int(rng.integers(0, 5)),
        minute=int(rng.integers(0, 60)),
        second=int(rng.integers(0, 60)),
        microsecond=0,
    )


def _typical_bytes(entity: EntityGenProfile) -> float:
    """Median bytes-out for this entity, used as the reference for exfil sizing."""
    mu, _ = entity.bytes_out_lognormal
    return float(math.exp(mu))


def _make_event(
    entity: EntityGenProfile,
    *,
    event_id: str,
    timestamp: datetime,
    resource: str,
    geo: GeoLocation,
    source_ip: str,
    auth_method: AuthMethod,
    auth_success: bool,
    session_id: str,
    session_duration: float,
    commands: Sequence[str],
    device: DeviceSpec,
    bytes_out: float,
    bytes_in: float,
    label: AnomalyType,
) -> Event:
    """Construct one labeled attack event."""
    return Event(
        event_id=event_id,
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        timestamp=timestamp,
        source_ip=source_ip,
        geo=geo,
        resource_accessed=resource,
        auth_method=auth_method,
        auth_success=auth_success,
        session_id=session_id,
        session_duration=float(session_duration),
        command_sequence=list(commands)[-8:],
        device_fingerprint=DeviceFingerprint(**device.as_dict()),
        bytes_out=float(max(0.0, bytes_out)),
        bytes_in=float(max(0.0, bytes_in)),
        label=label,
    )


def _dominant_auth(entity: EntityGenProfile) -> AuthMethod:
    """The auth method this entity uses most often."""
    return max(entity.auth_weights.items(), key=lambda item: item[1])[0]


def _unusual_auth(entity: EntityGenProfile, rng: np.random.Generator) -> AuthMethod:
    """An auth method this entity rarely or never uses."""
    used = set(entity.auth_weights.keys())
    unused = [method for method in AuthMethod if method not in used]
    if unused:
        return unused[int(rng.integers(0, len(unused)))]
    rarest = min(entity.auth_weights.items(), key=lambda item: item[1])[0]
    return rarest


def _recon_commands(rng: np.random.Generator, count: int) -> List[str]:
    """A short run of attacker reconnaissance commands."""
    picks = rng.choice(len(HOSTILE_COMMANDS), size=min(count, len(HOSTILE_COMMANDS)), replace=False)
    return [HOSTILE_COMMANDS[int(i)] for i in picks]


# --------------------------------------------------------------------------- #
# Injectors
# --------------------------------------------------------------------------- #


def inject_brute_force(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """Repeated failed authentication against one entity from one source.

    Sized to stay above the deterministic detector's threshold (``brute_force_threshold``
    attempts inside ``brute_force_window_minutes``) even at maximum subtlety -- a brute force
    that does not burst is not a brute force, so the floor is a definition, not a cheat.
    """
    attempts = int(round(_lerp(22, 7, config.subtlety)))
    window_minutes = _lerp(3.5, 8.5, config.subtlety)
    city = _hostile_city(rng)
    geo = _geo(city, rng)
    source_ip = _hostile_ip(rng)
    device = _hostile_device(entity, rng)
    auth_method = AuthMethod.PASSWORD if AuthMethod.PASSWORD in entity.auth_weights else _dominant_auth(entity)
    session_id = f"ses_atk_{incident_id}"
    duration = window_minutes * 60.0

    events: List[Event] = []
    for index in range(attempts):
        offset = window_minutes * 60.0 * (index / max(1, attempts - 1)) if attempts > 1 else 0.0
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_{index:03d}",
                timestamp=start + timedelta(seconds=offset + float(rng.uniform(0, 2.0))),
                resource="/auth/login",
                geo=geo,
                source_ip=source_ip,
                auth_method=auth_method,
                auth_success=False,
                session_id=session_id,
                session_duration=duration,
                commands=["auth_attempt"],
                device=device,
                bytes_out=float(rng.uniform(180, 700)),
                bytes_in=float(rng.uniform(180, 700)),
                label=AnomalyType.BRUTE_FORCE,
            )
        )

    # Roughly half of brute-force attempts eventually succeed. The successful login is part
    # of the same incident, which is what makes a follow-on campaign stage plausible.
    if rng.random() < 0.5:
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_hit",
                timestamp=events[-1].timestamp + timedelta(seconds=float(rng.uniform(5, 40))),
                resource="/auth/login",
                geo=geo,
                source_ip=source_ip,
                auth_method=auth_method,
                auth_success=True,
                session_id=session_id,
                session_duration=duration,
                commands=["auth_attempt", "login"],
                device=device,
                bytes_out=float(rng.uniform(400, 1200)),
                bytes_in=float(rng.uniform(400, 1200)),
                label=AnomalyType.BRUTE_FORCE,
            )
        )

    return AttackIncident(AnomalyType.BRUTE_FORCE, entity.entity_id, events)


def inject_credential_stuffing(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """One source IP tried against many accounts, only a couple of attempts each.

    The inverse shape of brute force: breadth across entities rather than depth against one.
    Distinguishing the two is exactly the kind of thing the classifier exists for, so both
    are present with the same auth-failure surface but opposite fan-out.

    ``entity`` is the first victim; further victims are drawn from the same cohort, because
    real credential dumps are usually organizationally clustered.
    """
    victim_count = int(round(_lerp(26, 9, config.subtlety)))
    attempts_each = int(round(_lerp(3, 1, config.subtlety)))

    peers = [
        candidate
        for candidate in world.cohort_members(entity.cohort_id)
        if candidate.entity_id != entity.entity_id and candidate.active_from <= start
    ]
    rng.shuffle(peers)
    victims = [entity] + peers[: max(0, victim_count - 1)]

    city = _hostile_city(rng)
    geo = _geo(city, rng)
    source_ip = _hostile_ip(rng)
    device = _hostile_device(entity, rng)
    spread_minutes = _lerp(12.0, 90.0, config.subtlety)

    events: List[Event] = []
    for victim_index, victim in enumerate(victims):
        auth_method = (
            AuthMethod.PASSWORD if AuthMethod.PASSWORD in victim.auth_weights else _dominant_auth(victim)
        )
        session_id = f"ses_atk_{incident_id}_{victim_index:03d}"
        for attempt in range(max(1, attempts_each)):
            offset = float(rng.uniform(0, spread_minutes * 60.0))
            # A single account in the spray happens to have a reused password.
            success = victim_index == 0 and attempt == attempts_each - 1 and rng.random() < 0.35
            events.append(
                _make_event(
                    victim,
                    event_id=f"evt_atk_{incident_id}_{victim_index:03d}_{attempt}",
                    timestamp=start + timedelta(seconds=offset),
                    resource="/auth/login",
                    geo=geo,
                    source_ip=source_ip,
                    auth_method=auth_method,
                    auth_success=success,
                    session_id=session_id,
                    session_duration=spread_minutes * 60.0,
                    commands=["auth_attempt"],
                    device=device,
                    bytes_out=float(rng.uniform(150, 600)),
                    bytes_in=float(rng.uniform(150, 600)),
                    label=AnomalyType.CREDENTIAL_STUFFING,
                )
            )

    events.sort(key=lambda event: event.timestamp)
    return AttackIncident(AnomalyType.CREDENTIAL_STUFFING, entity.entity_id, events)


def inject_impossible_travel(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """Two authenticated sessions from locations no traveller could cover in the gap.

    Deliberately unambiguous: the deterministic detector must reach ~1.0 precision on this
    class, so the implied velocity always exceeds the configured threshold by a wide margin.
    Subtlety only widens the time gap, never enough to make the travel physically possible.
    """
    home_geo = _geo(entity.home_city, rng)
    far_city = _distant_city(entity, rng, min_km=6000.0)
    far_geo = _geo(far_city, rng)

    gap_minutes = _lerp(9.0, 55.0, config.subtlety)
    auth_method = _dominant_auth(entity)
    own_device = _entity_device(entity)
    stolen_device = _hostile_device(entity, rng)

    resources = list(entity.resource_weights.keys()) or list(entity.cohort.resources)
    session_home = f"ses_atk_{incident_id}_home"
    session_far = f"ses_atk_{incident_id}_far"

    events = [
        _make_event(
            entity,
            event_id=f"evt_atk_{incident_id}_home",
            timestamp=start,
            resource=resources[int(rng.integers(0, len(resources)))],
            geo=home_geo,
            source_ip=f"{entity.ip_prefix}{int(rng.integers(2, 254))}",
            auth_method=auth_method,
            auth_success=True,
            session_id=session_home,
            session_duration=float(rng.uniform(120, 900)),
            commands=["login", "view_document"],
            device=own_device,
            bytes_out=_typical_bytes(entity) * float(rng.uniform(0.6, 1.4)),
            bytes_in=_typical_bytes(entity) * float(rng.uniform(0.3, 0.8)),
            label=AnomalyType.IMPOSSIBLE_TRAVEL,
        )
    ]

    second_at = start + timedelta(minutes=gap_minutes)
    for index in range(int(rng.integers(2, 4))):
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_far{index}",
                timestamp=second_at + timedelta(seconds=index * float(rng.uniform(20, 120))),
                resource=resources[int(rng.integers(0, len(resources)))],
                geo=far_geo,
                source_ip=_hostile_ip(rng),
                auth_method=auth_method,
                auth_success=True,
                session_id=session_far,
                session_duration=float(rng.uniform(120, 1200)),
                commands=["login", "search_docs", "download_report"][: index + 2],
                device=stolen_device,
                bytes_out=_typical_bytes(entity) * float(rng.uniform(1.0, 2.5)),
                bytes_in=_typical_bytes(entity) * float(rng.uniform(0.3, 0.9)),
                label=AnomalyType.IMPOSSIBLE_TRAVEL,
            )
        )

    return AttackIncident(AnomalyType.IMPOSSIBLE_TRAVEL, entity.entity_id, events)


def inject_credential_misuse(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """Valid credentials, wrong behavior.

    Authentication succeeds and travel is physically plausible, so no single deterministic
    rule catches it. What gives it away is several facets deviating from the entity's own
    baseline at once -- off-hours, unfamiliar country, unusual auth method, sensitive
    resources, unfamiliar device. Subtlety controls how many facets deviate, which is what
    forces the baseline and fusion tiers to do real work.
    """
    facets = int(round(_lerp(5, 2, config.subtlety)))

    # Deep small-hours access is itself a strong single-event signal, so it is one of the
    # facets rather than a constant. The subtler variants land in the late evening, which
    # plenty of legitimate work also does.
    if facets >= 4:
        moment = _off_hours_moment(start, rng)
    else:
        moment = start.replace(
            hour=int(rng.integers(19, 24)),
            minute=int(rng.integers(0, 60)),
            second=int(rng.integers(0, 60)),
            microsecond=0,
        )

    # Only the most blatant variant connects from a hostile country. At default subtlety the
    # attacker comes from a location the entity legitimately uses, so "unfamiliar country" is
    # unavailable as a shortcut and the deviation has to be found in the entity's own
    # behavioral baseline: off-hours, sensitive resources, unfamiliar device, higher volume.
    use_foreign_geo = facets >= 4
    city = _hostile_city(rng) if use_foreign_geo else entity.secondary_city
    geo = _geo(city, rng)

    auth_method = _unusual_auth(entity, rng) if facets >= 5 else _dominant_auth(entity)
    device = _unfamiliar_device(entity, rng) if facets >= 2 else _entity_device(entity)
    source_ip = _hostile_ip(rng) if use_foreign_geo else f"{entity.ip_prefix}{int(rng.integers(2, 254))}"

    sensitive = list(world.sensitive_resources)
    own = list(entity.resource_weights.keys()) or list(entity.cohort.resources)
    event_count = int(rng.integers(4, 9))
    session_id = f"ses_atk_{incident_id}"
    duration = float(rng.uniform(600, 5400))

    commands = ["login", "search_docs", "view_document", "download_report", "export_result"]
    events: List[Event] = []
    offset = 0.0
    for index in range(event_count):
        offset += float(rng.uniform(20, 240))
        # Some crown jewels, but mostly ordinary traffic as cover -- a session that touches
        # nothing but sensitive resources would be trivially separable on resource name alone.
        if rng.random() < 0.35 and sensitive:
            resource = sensitive[int(rng.integers(0, len(sensitive)))]
        else:
            resource = own[int(rng.integers(0, len(own)))]
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_{index:03d}",
                timestamp=moment + timedelta(seconds=offset),
                resource=resource,
                geo=geo,
                source_ip=source_ip,
                auth_method=auth_method,
                auth_success=True,
                session_id=session_id,
                session_duration=duration,
                commands=commands[: min(index + 2, len(commands))],
                device=device,
                bytes_out=_typical_bytes(entity) * float(rng.uniform(1.5, 4.0)),
                bytes_in=_typical_bytes(entity) * float(rng.uniform(0.2, 0.7)),
                label=AnomalyType.CREDENTIAL_MISUSE,
            )
        )

    return AttackIncident(AnomalyType.CREDENTIAL_MISUSE, entity.entity_id, events)


def inject_lateral_movement(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """Rapid fan-out across resources belonging to other cohorts.

    The signal is *breadth outside the entity's own pool* plus recon commands that never
    appear in benign traffic. This is the class the sequence model should win on: the
    individual accesses can look ordinary, but their order and variety do not.
    """
    breadth = int(round(_lerp(13, 5, config.subtlety)))
    targets = world.foreign_resources(entity, rng, count=breadth)
    if not targets:
        targets = list(world.sensitive_resources[:breadth])

    geo = _geo(entity.home_city, rng)  # already inside the network
    source_ip = f"{entity.ip_prefix}{int(rng.integers(2, 254))}"
    device = _entity_device(entity)
    auth_method = _dominant_auth(entity)
    session_id = f"ses_atk_{incident_id}"

    recon = _recon_commands(rng, count=max(3, breadth // 2))
    commands: List[str] = ["login"]
    duration = float(rng.uniform(900, 5400))

    events: List[Event] = []
    offset = 0.0
    for index, resource in enumerate(targets):
        offset += float(rng.uniform(8, 70))  # fast pivoting, unlike human browsing
        commands.append(recon[index % len(recon)])
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_{index:03d}",
                timestamp=start + timedelta(seconds=offset),
                resource=resource,
                geo=geo,
                source_ip=source_ip,
                auth_method=auth_method,
                auth_success=True,
                session_id=session_id,
                session_duration=duration,
                commands=commands,
                device=device,
                bytes_out=_typical_bytes(entity) * float(rng.uniform(0.8, 2.0)),
                bytes_in=_typical_bytes(entity) * float(rng.uniform(0.5, 1.5)),
                label=AnomalyType.LATERAL_MOVEMENT,
            )
        )

    return AttackIncident(AnomalyType.LATERAL_MOVEMENT, entity.entity_id, events)


def inject_device_spoofing(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """The entity's identity presented from a device that is not its device.

    The fingerprint is always **plausible for the cohort** but never one this entity has used:
    an unseen MAC, and an OS/protocol drawn from what its peers legitimately run. That is
    deliberate. An earlier version borrowed an OS and protocol from a *different* cohort -- a
    plant sensor reporting Windows over gRPC -- which a naive model detected at 100% recall
    from ``entity_type`` plus ``protocol`` alone, with no device history involved. The class
    then proved nothing about behavioral profiling. Now it can only be caught by knowing what
    this specific entity has connected from before.

    Especially meaningful for edge devices: a plant sensor's hardware identity does not change
    on its own.
    """
    own = _entity_device(entity)
    cohort = entity.cohort

    # A different build from the cohort's own pool where possible, otherwise the same OS with
    # an unseen MAC.
    os_options = [name for name in cohort.os_pool if name != own.os] or [own.os]
    spoof_os = os_options[int(rng.integers(0, len(os_options)))]

    # The more blatant variant also switches to another protocol the cohort uses; the subtler
    # one keeps the entity's usual protocol so only the MAC and OS build differ.
    protocol_options = [p for p in cohort.protocols if p != own.protocol]
    if config.subtlety < 0.6 and protocol_options:
        spoof_protocol = protocol_options[int(rng.integers(0, len(protocol_options)))]
    else:
        spoof_protocol = own.protocol

    device = DeviceSpec(
        os=spoof_os,
        mac=random_mac(rng),
        protocol=spoof_protocol,
        user_agent=user_agent_for(spoof_os),
    )

    # Mostly from inside the network: a spoofed device connecting from Brazil would be caught
    # by the geo signal instead, which is a different class's job.
    city = entity.home_city if rng.random() < 0.75 else _hostile_city(rng)
    geo = _geo(city, rng)
    source_ip = (
        f"{entity.ip_prefix}{int(rng.integers(2, 254))}"
        if city.name == entity.home_city.name
        else _hostile_ip(rng)
    )
    auth_method = _dominant_auth(entity)
    resources = list(entity.resource_weights.keys()) or list(entity.cohort.resources)
    session_id = f"ses_atk_{incident_id}"
    duration = float(rng.uniform(180, 2400))
    event_count = int(rng.integers(3, 8))

    events: List[Event] = []
    offset = 0.0
    template = list(cohort.command_templates[int(rng.integers(0, len(cohort.command_templates)))])
    for index in range(event_count):
        offset += float(rng.uniform(15, 200))
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_{index:03d}",
                timestamp=start + timedelta(seconds=offset),
                resource=resources[int(rng.integers(0, len(resources)))],
                geo=geo,
                source_ip=source_ip,
                auth_method=auth_method,
                auth_success=True,
                session_id=session_id,
                session_duration=duration,
                # Mimics the entity's usual commands: the device is the only tell.
                commands=template[: min(index + 1, len(template))],
                device=device,
                bytes_out=_typical_bytes(entity) * float(rng.uniform(0.7, 1.8)),
                bytes_in=_typical_bytes(entity) * float(rng.uniform(0.4, 1.2)),
                label=AnomalyType.DEVICE_SPOOFING,
            )
        )

    return AttackIncident(AnomalyType.DEVICE_SPOOFING, entity.entity_id, events)


def inject_low_and_slow_exfil(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """Data leaving in small pieces over many hours.

    Every single event is only mildly larger than normal -- the per-event deviation is
    intentionally too small to alert on. What makes it detectable is the sustained
    repetition, which is why this class is a test of windowed and sequence features rather
    than of per-event thresholds.
    """
    transfer_count = int(round(_lerp(34, 16, config.subtlety)))
    # Kept close to normal on purpose. An individual transfer must be unremarkable: if a single
    # event were large enough to alert on, this would just be a threshold rule and there would
    # be nothing "low and slow" about it. The detectable property is the sustained repetition.
    size_multiplier = _lerp(2.4, 1.25, config.subtlety)
    span_hours = _lerp(6.0, 30.0, config.subtlety)

    geo = _geo(entity.home_city if rng.random() < 0.85 else _hostile_city(rng), rng)
    source_ip = f"{entity.ip_prefix}{int(rng.integers(2, 254))}"
    device = _entity_device(entity)
    auth_method = _dominant_auth(entity)
    baseline_bytes = _typical_bytes(entity)

    own = list(entity.resource_weights.keys()) or list(entity.cohort.resources)
    export_resources = [
        resource
        for resource in own + list(world.sensitive_resources)
        if any(token in resource for token in ("export", "download", "dump", "backup", "bulk", "vault"))
    ] or list(world.sensitive_resources)

    events: List[Event] = []
    for index in range(transfer_count):
        # Sessions are short and separated -- each looks like an ordinary small download.
        offset_hours = span_hours * (index / max(1, transfer_count - 1))
        timestamp = start + timedelta(hours=offset_hours, seconds=float(rng.uniform(0, 240)))
        # Mostly the entity's ordinary resources, so the resource name is not a label either.
        if rng.random() < 0.35:
            resource = export_resources[int(rng.integers(0, len(export_resources)))]
        else:
            resource = own[int(rng.integers(0, len(own)))]
        events.append(
            _make_event(
                entity,
                event_id=f"evt_atk_{incident_id}_{index:03d}",
                timestamp=timestamp,
                resource=resource,
                geo=geo,
                source_ip=source_ip,
                auth_method=auth_method,
                auth_success=True,
                session_id=f"ses_atk_{incident_id}_{index // 4:03d}",
                session_duration=float(rng.uniform(120, 900)),
                commands=["login", "search_docs", "download_report"],
                device=device,
                bytes_out=baseline_bytes * size_multiplier * float(rng.uniform(0.8, 1.3)),
                bytes_in=baseline_bytes * float(rng.uniform(0.1, 0.4)),
                label=AnomalyType.LOW_AND_SLOW_EXFIL,
            )
        )

    return AttackIncident(AnomalyType.LOW_AND_SLOW_EXFIL, entity.entity_id, events)


def inject_insider_drift(
    entity: EntityGenProfile,
    world: World,
    rng: np.random.Generator,
    start: datetime,
    config: GeneratorConfig,
    incident_id: str,
) -> AttackIncident:
    """A trusted entity slowly turning malicious.

    The hardest class by construction, and the reason benign drift exists in this dataset:
    both are gradual behavioral change. The difference is *direction*. Benign drift moves
    toward a new but ordinary routine; insider drift converges on sensitive resources and
    off-hours access. A system that flags all change would fail here, and so would one that
    adapts to all change.
    """
    span_days = _lerp(3.0, 9.0, config.subtlety)
    sessions = int(round(_lerp(9, 5, config.subtlety)))

    geo = _geo(entity.home_city, rng)
    source_ip = f"{entity.ip_prefix}{int(rng.integers(2, 254))}"
    device = _entity_device(entity)
    auth_method = _dominant_auth(entity)
    own = list(entity.resource_weights.keys()) or list(entity.cohort.resources)
    sensitive = list(world.sensitive_resources)

    events: List[Event] = []
    for session_index in range(sessions):
        # Escalation: later sessions are more sensitive and later at night.
        progress = session_index / max(1, sessions - 1)
        day_offset = span_days * progress
        session_start = start + timedelta(days=day_offset)
        if progress > 0.45:
            session_start = _off_hours_moment(session_start, rng)
        else:
            session_start = session_start.replace(
                hour=int(rng.integers(18, 23)), minute=int(rng.integers(0, 60))
            )

        session_id = f"ses_atk_{incident_id}_{session_index:03d}"
        duration = float(rng.uniform(300, 3000))
        event_count = int(rng.integers(2, 5))
        offset = 0.0
        for index in range(event_count):
            offset += float(rng.uniform(30, 300))
            if rng.random() < 0.25 + 0.65 * progress and sensitive:
                resource = sensitive[int(rng.integers(0, len(sensitive)))]
            else:
                resource = own[int(rng.integers(0, len(own)))]
            events.append(
                _make_event(
                    entity,
                    event_id=f"evt_atk_{incident_id}_{session_index:03d}_{index:03d}",
                    timestamp=session_start + timedelta(seconds=offset),
                    resource=resource,
                    geo=geo,
                    source_ip=source_ip,
                    auth_method=auth_method,
                    auth_success=True,
                    session_id=session_id,
                    session_duration=duration,
                    commands=["login", "search_docs", "view_document", "archive_files"][
                        : min(index + 2, 4)
                    ],
                    device=device,
                    bytes_out=_typical_bytes(entity) * float(rng.uniform(1.2, 2.6 + 1.5 * progress)),
                    bytes_in=_typical_bytes(entity) * float(rng.uniform(0.1, 0.5)),
                    label=AnomalyType.INSIDER_DRIFT,
                )
            )

    return AttackIncident(AnomalyType.INSIDER_DRIFT, entity.entity_id, events)


# --------------------------------------------------------------------------- #
# Registry and allocation
# --------------------------------------------------------------------------- #

Injector = Callable[
    [EntityGenProfile, World, np.random.Generator, datetime, GeneratorConfig, str],
    AttackIncident,
]

#: Injector plus the entity types the behavior makes sense for. An "insider" is a person, so
#: insider drift never targets a container or a sensor.
INJECTORS: Dict[str, Tuple[Injector, Tuple[EntityType, ...]]] = {
    AnomalyType.BRUTE_FORCE.value: (
        inject_brute_force,
        (EntityType.USER, EntityType.SERVICE_ACCOUNT, EntityType.EDGE_DEVICE),
    ),
    AnomalyType.CREDENTIAL_STUFFING.value: (
        inject_credential_stuffing,
        (EntityType.USER, EntityType.SERVICE_ACCOUNT),
    ),
    AnomalyType.IMPOSSIBLE_TRAVEL.value: (
        inject_impossible_travel,
        (EntityType.USER, EntityType.SERVICE_ACCOUNT),
    ),
    AnomalyType.CREDENTIAL_MISUSE.value: (
        inject_credential_misuse,
        (EntityType.USER, EntityType.SERVICE_ACCOUNT),
    ),
    AnomalyType.LATERAL_MOVEMENT.value: (
        inject_lateral_movement,
        (EntityType.USER, EntityType.SERVICE_ACCOUNT, EntityType.EDGE_DEVICE),
    ),
    AnomalyType.DEVICE_SPOOFING.value: (
        inject_device_spoofing,
        (EntityType.USER, EntityType.EDGE_DEVICE, EntityType.SERVICE_ACCOUNT),
    ),
    AnomalyType.LOW_AND_SLOW_EXFIL.value: (
        inject_low_and_slow_exfil,
        (EntityType.USER, EntityType.SERVICE_ACCOUNT),
    ),
    AnomalyType.INSIDER_DRIFT.value: (
        inject_insider_drift,
        (EntityType.USER,),
    ),
}


#: Hours an incident of each class needs to finish. Long-running classes must start early
#: enough that they do not run off the end of the window they are placed in.
_TAIL_HOURS: Dict[str, float] = {
    # Low-and-slow spans up to ~30 simulated hours; insider drift escalates over ~9 days.
    # These must reflect the real span, otherwise an incident placed near the end of a split
    # dumps most of its events into the following one and skews the split densities.
    AnomalyType.LOW_AND_SLOW_EXFIL.value: 34.0,
    AnomalyType.INSIDER_DRIFT.value: 168.0,
}


def eligible_entities(
    world: World, anomaly_type: str, active_by: Optional[datetime] = None
) -> List[EntityGenProfile]:
    """Entities this attack class can plausibly target.

    Parameters
    ----------
    active_by:
        When given, only entities that already exist by this moment are returned. Needed
        because cold-start entities are onboarded late and cannot be attacked before then.
    """
    _, allowed_types = INJECTORS[anomaly_type]
    return [
        entity
        for entity in world.entities
        if entity.entity_type in allowed_types
        and (active_by is None or entity.active_from <= active_by)
    ]


def split_windows(config: GeneratorConfig) -> Dict[str, Tuple[datetime, datetime]]:
    """The ``[start, end)`` time window of each split."""
    train_end, val_end = config.split_boundaries()
    return {
        "train": (config.start_date, train_end),
        "val": (train_end, val_end),
        "test": (val_end, config.end_date()),
    }


SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


def sample_split(
    rng: np.random.Generator,
    config: GeneratorConfig,
    split_weights: Optional[Dict[str, float]] = None,
) -> str:
    """Draw the split an incident should be placed in.

    Attacks are placed **proportionally across splits** rather than uniformly over the
    timeline. Uniform placement produced badly uneven anomaly density (1.3% in validation
    against 3.1% in test), and that matters concretely: the alert-budget threshold is tuned
    on validation and then applied to test, so if the two carry different anomaly density the
    tuned threshold is calibrated for the wrong world.

    Parameters
    ----------
    split_weights:
        Share of benign events in each split. Passed in by the orchestrator, which knows the
        real volumes -- weighting by *time* is not enough, because cold-start entities are
        onboarded late and inflate the later splits' event counts.
    """
    if split_weights:
        raw = [max(0.0, float(split_weights.get(name, 0.0))) for name in SPLIT_NAMES]
    else:
        test_fraction = max(0.0, 1.0 - config.train_fraction - config.val_fraction)
        raw = [config.train_fraction, config.val_fraction, test_fraction]

    probabilities = np.asarray(raw, dtype=float)
    if probabilities.sum() <= 0:
        probabilities = np.ones(3, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return SPLIT_NAMES[int(rng.choice(3, p=probabilities))]


def _moment_in_window(
    entity: EntityGenProfile,
    window: Tuple[datetime, datetime],
    rng: np.random.Generator,
    tail_hours: float,
) -> Optional[datetime]:
    """A start time inside ``window`` that leaves room for the incident to complete.

    Returns ``None`` when the entity does not exist for long enough inside this window.
    """
    window_start, window_end = window
    earliest = max(entity.active_from, window_start)
    latest = window_end - timedelta(hours=tail_hours)
    if latest <= earliest:
        # Long incidents are allowed to overflow the window rather than be dropped; their
        # per-event labels stay correct wherever the events land.
        latest = min(window_end, earliest + timedelta(hours=1))
        if latest <= earliest:
            return None
    span = (latest - earliest).total_seconds()
    return earliest + timedelta(seconds=float(rng.uniform(0, max(1.0, span))))


def run_injector(
    anomaly_type: str,
    world: World,
    rng: np.random.Generator,
    config: GeneratorConfig,
    incident_id: str,
    entity: Optional[EntityGenProfile] = None,
    start: Optional[datetime] = None,
    split: Optional[str] = None,
) -> Optional[AttackIncident]:
    """Run one injector by class name.

    Used both for standalone incidents and, from :mod:`data_generator.campaigns`, for the
    individual stages of a campaign.

    Parameters
    ----------
    split:
        Place the incident inside this split's time window. Ignored when ``start`` is given.
    """
    injector, _ = INJECTORS[anomaly_type]
    tail_hours = _TAIL_HOURS.get(anomaly_type, 2.0)

    if start is None:
        window = split_windows(config)[split or "train"] if split else (
            config.start_date,
            config.end_date(),
        )
        if entity is None:
            # Pick from entities that exist inside the target window, so a cold-start entity
            # is never attacked before it is onboarded.
            candidates = eligible_entities(
                world, anomaly_type, active_by=window[1] - timedelta(hours=tail_hours)
            )
            if not candidates:
                candidates = eligible_entities(world, anomaly_type)
            if not candidates:
                return None
            entity = candidates[int(rng.integers(0, len(candidates)))]

        start = _moment_in_window(entity, window, rng, tail_hours)
        if start is None:
            return None

    if entity is None:
        candidates = eligible_entities(world, anomaly_type, active_by=start)
        if not candidates:
            return None
        entity = candidates[int(rng.integers(0, len(candidates)))]

    incident = injector(entity, world, rng, start, config, incident_id)
    return incident if incident.events else None


def inject_attacks(
    world: World,
    rng: np.random.Generator,
    config: GeneratorConfig,
    target_events: int,
    split_weights: Optional[Dict[str, float]] = None,
) -> List[AttackIncident]:
    """Inject standalone attack incidents until the event budget is spent.

    The budget is split across classes by ``config.attack_class_weights``. Because incidents
    have variable length, each class is filled until it reaches its share rather than by a
    fixed incident count -- which keeps the overall anomaly rate on target.
    """
    if target_events <= 0:
        return []

    weights = config.attack_class_weights
    total_weight = sum(weights.values()) or 1.0
    windows = split_windows(config)

    if split_weights:
        shares = {name: max(0.0, float(split_weights.get(name, 0.0))) for name in SPLIT_NAMES}
    else:
        test_fraction = max(0.0, 1.0 - config.train_fraction - config.val_fraction)
        shares = {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "test": test_fraction,
        }
    share_total = sum(shares.values()) or 1.0
    shares = {name: value / share_total for name, value in shares.items()}

    incidents: List[AttackIncident] = []
    counter = 0

    for anomaly_type in ANOMALY_CLASSES[1:]:  # skip 'normal'
        class_share = weights.get(anomaly_type, 0.0) / total_weight
        class_budget = target_events * class_share
        if class_budget <= 0:
            continue

        # Per-split budget for this class, spent by where events actually *land*. Long-running
        # incidents spill across boundaries, so an incident placed in train can deposit events
        # in val; those are debited from val's budget too. Without that, each split fills its
        # own quota and then receives spillover on top, pushing the overall rate above target.
        remaining = {
            split: int(round(class_budget * shares[split])) for split in SPLIT_NAMES
        }

        guard = 0
        last_size = 0
        while guard < 500:
            guard += 1
            open_splits = [split for split in SPLIT_NAMES if remaining[split] > 0]
            if not open_splits:
                break

            # Incidents are chunky -- a credential-stuffing spray is ~34 events. Iterating a
            # (class, split) grid forced at least one full incident into every cell, which at
            # small scale produced 24 incidents no matter how small the budget and overshot the
            # target rate by 67%. Instead, pick whichever split still has room, weighted by how
            # much room, and stop once no split has space for a meaningful addition.
            if last_size and max(remaining[split] for split in open_splits) < last_size * 0.5:
                break

            room = np.asarray([remaining[split] for split in open_splits], dtype=float)
            split = open_splits[int(rng.choice(len(open_splits), p=room / room.sum()))]

            counter += 1
            incident = run_injector(
                anomaly_type,
                world,
                rng,
                config,
                incident_id=f"{anomaly_type[:6]}{counter:05d}",
                split=split,
            )
            if incident is None:
                remaining[split] = 0  # this split cannot host the class; try the others
                continue

            incidents.append(incident)
            last_size = len(incident.events)
            for landed_split, window in windows.items():
                remaining[landed_split] -= sum(
                    1
                    for event in incident.events
                    if window[0] <= event.timestamp < window[1]
                )

    return incidents


def attack_summary(incidents: Sequence[AttackIncident]) -> Dict[str, object]:
    """Per-class incident and event counts, for the taxonomy and the console report."""
    per_class: Dict[str, Dict[str, int]] = {}
    for incident in incidents:
        bucket = per_class.setdefault(
            incident.anomaly_type.value, {"incidents": 0, "events": 0}
        )
        bucket["incidents"] += 1
        bucket["events"] += len(incident.events)
    return {
        "n_incidents": len(incidents),
        "n_events": sum(len(incident.events) for incident in incidents),
        "per_class": per_class,
    }


__all__ = [
    "AttackIncident",
    "haversine_km",
    "inject_brute_force",
    "inject_credential_stuffing",
    "inject_impossible_travel",
    "inject_credential_misuse",
    "inject_lateral_movement",
    "inject_device_spoofing",
    "inject_low_and_slow_exfil",
    "inject_insider_drift",
    "INJECTORS",
    "eligible_entities",
    "split_windows",
    "sample_split",
    "run_injector",
    "inject_attacks",
    "attack_summary",
]
