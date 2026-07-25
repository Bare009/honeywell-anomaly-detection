"""Entity profiles, latent cohorts and the simulated world.

Everything the generator knows about "normal" lives here. Each entity gets a profile
describing when it works, where it connects from, what it touches, how it authenticates
and which device it uses. Profiles are drawn from one of six **latent cohorts** -- groups of
entities that behave alike.

Cohorts matter well beyond realism: they are the priors the cold-start path falls back on
when an entity has too little history of its own (section 14 of the plan). They exist in
the data because they exist in real organizations, and the models later rediscover them by
clustering rather than being told.

Nothing here writes files or consumes global random state: :func:`build_world` takes an
explicit generator so the whole world is reproducible from a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.models import AuthMethod, EntityType

# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class City:
    """A location an entity can legitimately connect from."""

    name: str
    country: str
    lat: float
    lon: float


#: Home locations for the simulated organization. Clustered in India with a few
#: international offices, so cross-country access is plausible but notable.
HOME_CITIES: Tuple[City, ...] = (
    City("Bengaluru", "India", 12.9716, 77.5946),
    City("Hyderabad", "India", 17.3850, 78.4867),
    City("Pune", "India", 18.5204, 73.8567),
    City("Chennai", "India", 13.0827, 80.2707),
    City("Gurugram", "India", 28.4595, 77.0266),
    City("Singapore", "Singapore", 1.3521, 103.8198),
    City("London", "United Kingdom", 51.5074, -0.1278),
    City("Phoenix", "United States", 33.4484, -112.0740),
)

#: Locations attacks originate from. Every one is >1500 km from *every* home city, which is
#: what makes "unfamiliar country" and geo-velocity carry information. Jakarta was removed
#: from this list during Phase 1: it sits only 905 km from the Singapore office, so attacks
#: staged from there would have looked like ordinary regional travel.
HOSTILE_CITIES: Tuple[City, ...] = (
    City("Sao Paulo", "Brazil", -23.5505, -46.6333),
    City("Lagos", "Nigeria", 6.5244, 3.3792),
    City("Kyiv", "Ukraine", 50.4501, 30.5234),
    City("Bogota", "Colombia", 4.7110, -74.0721),
    City("Caracas", "Venezuela", 10.4806, -66.9036),
    City("Minsk", "Belarus", 53.9006, 27.5590),
    City("Ankara", "Turkey", 39.9334, 32.8597),
)


# --------------------------------------------------------------------------- #
# Hour-of-day activity shapes
# --------------------------------------------------------------------------- #


def _hour_weights(active: Sequence[int], peak: Sequence[int], baseline: float) -> Tuple[float, ...]:
    """Build a 24-slot activity distribution.

    Parameters
    ----------
    active:
        Hours during which the entity is normally working.
    peak:
        Subset of ``active`` that carries extra load.
    baseline:
        Relative weight for every other hour. Never zero -- a genuinely impossible hour
        would make any off-hours access trivially detectable, which is not realistic and
        would inflate our metrics.
    """
    weights = [baseline] * 24
    for hour in active:
        weights[hour % 24] = 1.0
    for hour in peak:
        weights[hour % 24] = 1.6
    total = sum(weights)
    return tuple(value / total for value in weights)


BUSINESS_HOURS = _hour_weights(range(9, 19), (10, 11, 14, 15), 0.02)
EXTENDED_HOURS = _hour_weights(range(8, 22), (10, 11, 15, 16, 20), 0.04)
SHIFTED_HOURS = _hour_weights(range(12, 22), (14, 15, 18), 0.03)
ALWAYS_ON = _hour_weights(range(24), (2, 3, 22), 1.0)
BATCH_HOURS = _hour_weights((0, 1, 2, 3, 4, 5, 12, 13, 22, 23), (1, 2, 3), 0.3)


# --------------------------------------------------------------------------- #
# Resource catalog
# --------------------------------------------------------------------------- #

#: Resources that represent crown jewels. Legitimate access is rare and concentrated in a
#: few cohorts; attacks converge on them, which is what makes resource sensitivity a
#: genuine signal rather than a giveaway.
SENSITIVE_RESOURCES: Tuple[str, ...] = (
    "/vault/payroll/export",
    "/vault/customer-pii/bulk",
    "/vault/contracts/signed",
    "/db/prod/credentials",
    "/admin/iam/roles",
    "/plant/control/setpoints",
    "/vault/source/crypto-keys",
    "/db/prod/full-dump",
)


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CohortSpec:
    """Behavioral archetype shared by a group of entities."""

    cohort_id: int
    name: str
    entity_type: EntityType
    hour_weights: Tuple[float, ...]
    weekend_activity: float
    sessions_per_day: float
    events_per_session: Tuple[int, int]
    session_seconds: Tuple[float, float]
    resources: Tuple[str, ...]
    rare_sensitive: Tuple[str, ...]
    auth_dist: Dict[AuthMethod, float]
    protocols: Tuple[str, ...]
    os_pool: Tuple[str, ...]
    command_templates: Tuple[Tuple[str, ...], ...]
    bytes_out_lognormal: Tuple[float, float]
    auth_fail_rate: float
    travel_probability: float

    def sensitive_share(self) -> float:
        """Share of accesses that legitimately touch a sensitive resource."""
        return 0.02 if self.rare_sensitive else 0.0


COHORTS: Tuple[CohortSpec, ...] = (
    CohortSpec(
        cohort_id=0,
        name="office_staff",
        entity_type=EntityType.USER,
        hour_weights=BUSINESS_HOURS,
        weekend_activity=0.12,
        sessions_per_day=2.6,
        events_per_session=(3, 8),
        session_seconds=(6.0, 0.7),
        resources=(
            "/portal/home",
            "/portal/timesheet",
            "/hr/leave-request",
            "/hr/policies",
            "/finance/expense-claim",
            "/docs/team-drive",
            "/docs/meeting-notes",
            "/mail/inbox",
            "/reports/monthly-summary",
            "/portal/directory",
        ),
        rare_sensitive=("/vault/contracts/signed",),
        auth_dist={AuthMethod.PASSWORD: 0.62, AuthMethod.MFA: 0.33, AuthMethod.BIOMETRIC: 0.05},
        protocols=("https",),
        os_pool=("Windows 11 22H2", "Windows 10 21H2", "macOS 14.4"),
        command_templates=(
            ("login", "open_portal", "view_document", "logout"),
            ("login", "open_portal", "submit_timesheet", "view_document", "logout"),
            ("login", "search_docs", "view_document", "download_report", "logout"),
            ("login", "open_mail", "view_document", "reply_mail", "logout"),
        ),
        bytes_out_lognormal=(8.6, 0.9),
        auth_fail_rate=0.018,
        travel_probability=0.010,
    ),
    CohortSpec(
        cohort_id=1,
        name="engineering",
        entity_type=EntityType.USER,
        hour_weights=EXTENDED_HOURS,
        weekend_activity=0.30,
        sessions_per_day=3.4,
        events_per_session=(4, 11),
        session_seconds=(7.1, 0.8),
        resources=(
            "/git/platform-core",
            "/git/edge-firmware",
            "/ci/pipeline/build",
            "/ci/pipeline/logs",
            "/artifacts/registry",
            "/k8s/staging/pods",
            "/k8s/staging/logs",
            "/db/staging/query",
            "/docs/architecture",
            "/monitoring/dashboards",
            "/jira/board",
        ),
        rare_sensitive=("/db/prod/credentials", "/vault/source/crypto-keys"),
        auth_dist={AuthMethod.MFA: 0.46, AuthMethod.TOKEN: 0.32, AuthMethod.PASSWORD: 0.22},
        protocols=("https", "ssh"),
        os_pool=("Ubuntu 22.04", "macOS 14.4", "Windows 11 22H2"),
        command_templates=(
            ("login", "git_pull", "run_build", "view_logs", "logout"),
            ("login", "ssh_connect", "tail_logs", "restart_service", "logout"),
            ("login", "git_pull", "run_tests", "push_artifact", "view_logs", "logout"),
            ("login", "query_db", "export_result", "logout"),
            ("login", "open_dashboard", "view_logs", "logout"),
        ),
        bytes_out_lognormal=(9.4, 1.0),
        auth_fail_rate=0.022,
        travel_probability=0.016,
    ),
    CohortSpec(
        cohort_id=2,
        name="business_analytics",
        entity_type=EntityType.USER,
        hour_weights=BUSINESS_HOURS,
        weekend_activity=0.08,
        sessions_per_day=2.2,
        events_per_session=(3, 9),
        session_seconds=(6.8, 0.7),
        resources=(
            "/bi/dashboards/sales",
            "/bi/dashboards/ops",
            "/bi/report-builder",
            "/db/warehouse/query",
            "/db/warehouse/export",
            "/docs/analysis",
            "/portal/home",
            "/reports/monthly-summary",
            "/reports/forecast",
        ),
        rare_sensitive=("/vault/customer-pii/bulk",),
        auth_dist={AuthMethod.MFA: 0.52, AuthMethod.PASSWORD: 0.43, AuthMethod.BIOMETRIC: 0.05},
        protocols=("https",),
        os_pool=("Windows 11 22H2", "macOS 14.4"),
        command_templates=(
            ("login", "open_dashboard", "run_query", "export_result", "logout"),
            ("login", "run_query", "build_report", "download_report", "logout"),
            ("login", "open_dashboard", "view_document", "logout"),
        ),
        bytes_out_lognormal=(10.1, 1.1),
        auth_fail_rate=0.015,
        travel_probability=0.008,
    ),
    CohortSpec(
        cohort_id=3,
        name="batch_services",
        entity_type=EntityType.SERVICE_ACCOUNT,
        hour_weights=BATCH_HOURS,
        weekend_activity=0.95,
        sessions_per_day=5.5,
        events_per_session=(2, 5),
        session_seconds=(5.4, 0.5),
        resources=(
            "/etl/nightly/extract",
            "/etl/nightly/load",
            "/db/warehouse/write",
            "/queue/ingest",
            "/storage/backup/write",
            "/metrics/push",
        ),
        rare_sensitive=("/db/prod/full-dump",),
        auth_dist={AuthMethod.TOKEN: 0.68, AuthMethod.CERTIFICATE: 0.32},
        protocols=("https", "grpc"),
        os_pool=("Debian 12 (container)", "Alpine 3.19 (container)"),
        command_templates=(
            ("authenticate", "extract_batch", "load_batch", "emit_metrics"),
            ("authenticate", "read_queue", "write_warehouse"),
            ("authenticate", "snapshot", "write_backup"),
        ),
        bytes_out_lognormal=(11.6, 0.8),
        auth_fail_rate=0.006,
        travel_probability=0.0,
    ),
    CohortSpec(
        cohort_id=4,
        name="integration_services",
        entity_type=EntityType.SERVICE_ACCOUNT,
        hour_weights=ALWAYS_ON,
        weekend_activity=0.90,
        sessions_per_day=6.0,
        events_per_session=(2, 4),
        session_seconds=(4.6, 0.5),
        resources=(
            "/api/gateway/orders",
            "/api/gateway/inventory",
            "/api/partner/sync",
            "/queue/events",
            "/cache/warm",
            "/metrics/push",
        ),
        rare_sensitive=(),
        auth_dist={AuthMethod.TOKEN: 0.74, AuthMethod.CERTIFICATE: 0.26},
        protocols=("https", "grpc"),
        os_pool=("Debian 12 (container)", "Alpine 3.19 (container)"),
        command_templates=(
            ("authenticate", "sync_partner", "emit_metrics"),
            ("authenticate", "read_queue", "publish_event"),
            ("authenticate", "warm_cache"),
        ),
        bytes_out_lognormal=(9.9, 0.7),
        auth_fail_rate=0.005,
        travel_probability=0.0,
    ),
    CohortSpec(
        cohort_id=5,
        name="plant_devices",
        entity_type=EntityType.EDGE_DEVICE,
        hour_weights=ALWAYS_ON,
        weekend_activity=1.0,
        sessions_per_day=7.0,
        events_per_session=(2, 4),
        session_seconds=(4.2, 0.4),
        resources=(
            "/plant/telemetry/push",
            "/plant/sensor/read",
            "/plant/firmware/check",
            "/plant/heartbeat",
            "/plant/diagnostics",
        ),
        rare_sensitive=("/plant/control/setpoints",),
        auth_dist={AuthMethod.CERTIFICATE: 0.88, AuthMethod.TOKEN: 0.12},
        protocols=("modbus", "mqtt", "https"),
        os_pool=(
            "FreeRTOS 10.4 / fw 2.1.3",
            "Yocto Linux / fw 3.0.1",
            "QNX 7.1 / fw 1.8.9",
        ),
        command_templates=(
            ("handshake", "push_telemetry", "heartbeat"),
            ("handshake", "read_sensor", "push_telemetry"),
            ("handshake", "check_firmware", "heartbeat"),
        ),
        bytes_out_lognormal=(7.4, 0.5),
        auth_fail_rate=0.004,
        travel_probability=0.0,
    ),
)

#: Fast lookup by cohort id.
COHORT_BY_ID: Dict[int, CohortSpec] = {cohort.cohort_id: cohort for cohort in COHORTS}

#: Commands that only appear in attack behavior. Kept separate from the benign vocabulary
#: so the sequence model has genuinely unseen tokens to be surprised by.
HOSTILE_COMMANDS: Tuple[str, ...] = (
    "whoami",
    "enum_shares",
    "net_view",
    "mount_share",
    "psexec",
    "dump_creds",
    "disable_audit",
    "add_local_admin",
    "port_scan",
    "archive_files",
    "stage_payload",
    "clear_logs",
)


# --------------------------------------------------------------------------- #
# Generator configuration
# --------------------------------------------------------------------------- #


@dataclass
class GeneratorConfig:
    """Every knob of the simulation, in one auditable place.

    The defaults produce roughly 80-90k events over 35 simulated days at a ~2% anomaly
    rate -- large enough for the sequence model to learn from, small enough to regenerate
    and retrain in minutes on a CPU.
    """

    seed: int = 42

    # --- population ---
    n_entities: int = 260
    entity_type_mix: Tuple[float, float, float] = (0.60, 0.25, 0.15)  # user, service, device

    # --- timeline ---
    days: int = 45
    start_date: datetime = field(
        default_factory=lambda: datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    )
    train_fraction: float = 0.60
    val_fraction: float = 0.20  # test takes the remainder

    # --- cold start (entities that appear late, so they have little history) ---
    coldstart_entity_fraction: float = 0.12
    coldstart_appearance_point: float = 0.82  # fraction of the timeline

    # --- benign concept drift (D3) ---
    drift_entity_fraction: float = 0.14
    drift_start_point: float = 0.55
    drift_ramp_days: float = 9.0

    # --- attacks ---
    #: Inside the mandated 0.5-3% band, chosen at the low end for a specific reason: the
    #: headline metric is recall inside a 1%-of-events analyst alert budget. If anomalies are
    #: 2% of events, the top 1% cannot physically contain more than half of them and the
    #: target is unreachable no matter how good the model is. At 0.8% the budget can hold
    #: essentially all anomalies, so the metric measures the ranking rather than arithmetic.
    target_anomaly_rate: float = 0.008
    #: Relative share of the anomaly budget per class. Geometric classes need fewer events
    #: to be unambiguous; behavioral classes need more to be learnable.
    #: Weighted toward the behaviorally hard classes. Brute force and credential stuffing are
    #: genuinely easy -- an auth-failure burst is visible in a single event -- so letting them
    #: dominate the anomaly budget would make the headline metrics look good while proving
    #: nothing about the layered detectors.
    attack_class_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "brute_force": 0.11,
            "credential_stuffing": 0.10,
            "impossible_travel": 0.09,
            "credential_misuse": 0.15,
            "lateral_movement": 0.19,
            "device_spoofing": 0.14,
            "low_and_slow_exfil": 0.10,
            "insider_drift": 0.12,
        }
    )
    #: Fraction of the anomaly budget spent on multi-stage campaigns (D1) rather than
    #: standalone incidents. Campaigns are large, indivisible units (~35 events each), so this
    #: share also determines how many campaigns exist -- and therefore how precisely campaign
    #: reconstruction accuracy can be measured in Phase 9.
    campaign_budget_fraction: float = 0.42

    #: 0 = blatant attacks, 1 = nearly indistinguishable from normal. This is the single
    #: dial that controls how hard the dataset is; see TAXONOMY.md.
    subtlety: float = 0.55

    def resolved_days(self) -> int:
        """Timeline length, guaranteed to be at least one day."""
        return max(1, int(self.days))

    def end_date(self) -> datetime:
        """Exclusive end of the simulated timeline."""
        return self.start_date + timedelta(days=self.resolved_days())

    def split_boundaries(self) -> Tuple[datetime, datetime]:
        """Return the two timestamps that separate train | val | test.

        The split is by **time**, never at random: a model that trains on the future and
        predicts the past would report metrics it could never achieve in production.
        """
        total = timedelta(days=self.resolved_days())
        return (
            self.start_date + total * self.train_fraction,
            self.start_date + total * (self.train_fraction + self.val_fraction),
        )


# --------------------------------------------------------------------------- #
# Entity profiles
# --------------------------------------------------------------------------- #


@dataclass
class DeviceSpec:
    """One device an entity legitimately connects from."""

    os: str
    mac: str
    protocol: str
    user_agent: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        """Shape expected by :class:`common.models.DeviceFingerprint`."""
        return {
            "os": self.os,
            "mac": self.mac,
            "protocol": self.protocol,
            "user_agent": self.user_agent,
        }


@dataclass
class EntityGenProfile:
    """The generator's private notion of an entity's normal behavior.

    Richer than the runtime :class:`common.models.EntityProfile`: this is ground truth
    about how the entity was simulated, whereas the runtime profile is what the system
    *learns* from observed events.
    """

    entity_id: str
    entity_type: EntityType
    cohort_id: int
    home_city: City
    secondary_city: City
    ip_prefix: str
    devices: List[DeviceSpec]
    resource_weights: Dict[str, float]
    auth_weights: Dict[AuthMethod, float]
    sessions_per_day: float
    hour_weights: Tuple[float, ...]
    session_seconds: Tuple[float, float]
    bytes_out_lognormal: Tuple[float, float]
    auth_fail_rate: float
    active_from: datetime
    is_coldstart: bool = False
    drift_plan: Optional["DriftPlan"] = None  # populated by data_generator.drift

    @property
    def cohort(self) -> CohortSpec:
        """The archetype this entity was drawn from."""
        return COHORT_BY_ID[self.cohort_id]

    def primary_device(self) -> DeviceSpec:
        """The device the entity uses most of the time."""
        return self.devices[0]


@dataclass
class World:
    """The complete simulated population plus shared catalogs."""

    config: GeneratorConfig
    entities: List[EntityGenProfile]
    resource_catalog: Tuple[str, ...]
    sensitive_resources: Tuple[str, ...]

    def by_id(self, entity_id: str) -> EntityGenProfile:
        """Look up one entity, raising a clear error on a bad id."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(f"Unknown entity_id: {entity_id}")

    def cohort_members(self, cohort_id: int) -> List[EntityGenProfile]:
        """Every entity belonging to one cohort."""
        return [e for e in self.entities if e.cohort_id == cohort_id]

    def foreign_resources(self, entity: EntityGenProfile, rng: np.random.Generator, count: int) -> List[str]:
        """Resources this entity has no business touching.

        Used by lateral movement: breadth *outside* the entity's own pool is the signal,
        so the resources must come from other cohorts.
        """
        own = set(entity.cohort.resources)
        candidates = [r for r in self.resource_catalog if r not in own]
        if not candidates:
            return []
        picks = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        return [candidates[int(i)] for i in picks]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def random_mac(rng: np.random.Generator) -> str:
    """A locally-administered synthetic MAC address."""
    octets = rng.integers(0, 256, size=5)
    return "02:" + ":".join(f"{int(o):02x}" for o in octets)


def user_agent_for(os_name: str) -> str:
    """A plausible user-agent string derived from the OS (deterministic)."""
    if os_name.startswith("Windows"):
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0"
    if os_name.startswith("macOS"):
        return "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) Safari/17.4"
    if os_name.startswith("Ubuntu"):
        return "Mozilla/5.0 (X11; Linux x86_64) Firefox/124.0"
    return f"{os_name.split()[0]}-agent/1.0"


def _dirichlet_weights(
    keys: Sequence[str], rng: np.random.Generator, concentration: float
) -> Dict[str, float]:
    """Per-entity preference weights over a set of keys.

    Drawn from a Dirichlet so every entity in a cohort has its *own* favourites rather
    than an identical distribution -- otherwise per-entity baselines would be redundant
    and the whole premise of entity-level profiling would collapse.
    """
    if not keys:
        return {}
    weights = rng.dirichlet(np.full(len(keys), concentration))
    return {key: float(weight) for key, weight in zip(keys, weights)}


def _assign_cohorts(config: GeneratorConfig, rng: np.random.Generator) -> List[int]:
    """Assign a cohort to every entity, honoring the configured entity-type mix."""
    user_share, service_share, device_share = config.entity_type_mix
    total = config.n_entities

    counts = {
        EntityType.USER: int(round(total * user_share)),
        EntityType.SERVICE_ACCOUNT: int(round(total * service_share)),
        EntityType.EDGE_DEVICE: int(round(total * device_share)),
    }
    # Absorb rounding drift into the largest group so the population size is exact.
    counts[EntityType.USER] += total - sum(counts.values())

    by_type: Dict[EntityType, List[int]] = {}
    for cohort in COHORTS:
        by_type.setdefault(cohort.entity_type, []).append(cohort.cohort_id)

    assignments: List[int] = []
    for entity_type, count in counts.items():
        options = by_type[entity_type]
        for index in range(max(0, count)):
            # Round-robin with a light random perturbation: keeps cohorts balanced while
            # avoiding a rigid id-to-cohort pattern.
            if rng.random() < 0.75:
                assignments.append(options[index % len(options)])
            else:
                assignments.append(options[int(rng.integers(0, len(options)))])

    rng.shuffle(assignments)
    return assignments


def build_world(config: Optional[GeneratorConfig] = None, rng: Optional[np.random.Generator] = None) -> World:
    """Create the full population of entity profiles.

    Parameters
    ----------
    config:
        Simulation knobs. Defaults to :class:`GeneratorConfig`.
    rng:
        Explicit generator. Defaults to one seeded from ``config.seed`` so the world is
        reproducible without touching global random state.
    """
    config = config or GeneratorConfig()
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    cohort_ids = _assign_cohorts(config, rng)
    timeline_days = config.resolved_days()
    coldstart_count = int(round(config.n_entities * config.coldstart_entity_fraction))
    coldstart_at = config.start_date + timedelta(
        days=timeline_days * config.coldstart_appearance_point
    )

    # Which entities are cold-start is decided up front so the choice is independent of
    # cohort assignment order.
    coldstart_indices = set(
        int(i)
        for i in rng.choice(config.n_entities, size=min(coldstart_count, config.n_entities), replace=False)
    )

    type_counters: Dict[EntityType, int] = {
        EntityType.USER: 0,
        EntityType.SERVICE_ACCOUNT: 0,
        EntityType.EDGE_DEVICE: 0,
    }
    prefixes = {
        EntityType.USER: "user",
        EntityType.SERVICE_ACCOUNT: "svc",
        EntityType.EDGE_DEVICE: "dev",
    }

    entities: List[EntityGenProfile] = []
    for index, cohort_id in enumerate(cohort_ids):
        cohort = COHORT_BY_ID[cohort_id]
        type_counters[cohort.entity_type] += 1
        entity_id = f"{prefixes[cohort.entity_type]}_{type_counters[cohort.entity_type]:04d}"

        home = HOME_CITIES[int(rng.integers(0, len(HOME_CITIES)))]
        # A secondary city the entity plausibly travels to (never the same as home).
        others = [city for city in HOME_CITIES if city.name != home.name]
        secondary = others[int(rng.integers(0, len(others)))]

        n_devices = 1 if cohort.entity_type != EntityType.USER else int(rng.integers(1, 3))
        devices: List[DeviceSpec] = []
        for _ in range(n_devices):
            os_name = cohort.os_pool[int(rng.integers(0, len(cohort.os_pool)))]
            devices.append(
                DeviceSpec(
                    os=os_name,
                    mac=random_mac(rng),
                    protocol=cohort.protocols[int(rng.integers(0, len(cohort.protocols)))],
                    user_agent=user_agent_for(os_name),
                )
            )

        resource_weights = _dirichlet_weights(cohort.resources, rng, concentration=0.9)
        auth_weights = {
            method: float(share) for method, share in cohort.auth_dist.items()
        }

        is_coldstart = index in coldstart_indices
        entities.append(
            EntityGenProfile(
                entity_id=entity_id,
                entity_type=cohort.entity_type,
                cohort_id=cohort_id,
                home_city=home,
                secondary_city=secondary,
                ip_prefix=f"10.{int(rng.integers(1, 240))}.{int(rng.integers(0, 255))}.",
                devices=devices,
                resource_weights=resource_weights,
                auth_weights=auth_weights,
                # Per-entity rate spread around the cohort mean, floored so nobody is idle.
                sessions_per_day=max(
                    0.4, float(cohort.sessions_per_day * rng.normal(1.0, 0.22))
                ),
                hour_weights=cohort.hour_weights,
                session_seconds=cohort.session_seconds,
                bytes_out_lognormal=cohort.bytes_out_lognormal,
                auth_fail_rate=float(np.clip(cohort.auth_fail_rate * rng.normal(1.0, 0.3), 0.0, 0.2)),
                active_from=coldstart_at if is_coldstart else config.start_date,
                is_coldstart=is_coldstart,
            )
        )

    catalog = sorted({resource for cohort in COHORTS for resource in cohort.resources})
    return World(
        config=config,
        entities=entities,
        resource_catalog=tuple(catalog),
        sensitive_resources=SENSITIVE_RESOURCES,
    )


__all__ = [
    "City",
    "HOME_CITIES",
    "HOSTILE_CITIES",
    "SENSITIVE_RESOURCES",
    "HOSTILE_COMMANDS",
    "CohortSpec",
    "COHORTS",
    "COHORT_BY_ID",
    "GeneratorConfig",
    "DeviceSpec",
    "EntityGenProfile",
    "World",
    "build_world",
    "random_mac",
    "user_agent_for",
    "BUSINESS_HOURS",
    "EXTENDED_HOURS",
    "SHIFTED_HOURS",
    "ALWAYS_ON",
    "BATCH_HOURS",
]
