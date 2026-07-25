"""Pydantic v2 domain contracts shared by every plane of the system.

These models are the single definition of "what a thing is" across the data generator,
the feature pipeline, the scoring service, the read API and (mirrored in TypeScript) the
dashboard. Anything persisted to MongoDB is produced by ``model_dump(mode="json")``, so
all datetimes serialize to ISO-8601 strings and all enums to their string values.

Design notes
------------
* ``Event.label`` / ``campaign_id`` / ``stage`` are **optional**. Ground truth is stored
  separately from features and is never required to score an event -- the serving path
  must work on unlabeled traffic.
* ``Lenient*`` annotated types coerce loose strings ("0.83", "true", "3") into real
  numbers/bools. They exist for the optional LLM narrator, which returns JSON with
  inconsistent typing, and for tolerant ingest of replayed data.
* Every model is ``extra="forbid"`` free-form only where a nested payload genuinely
  varies (probability maps, dataset summaries); elsewhere fields are explicit.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def utc_now() -> datetime:
    """Timezone-aware current UTC timestamp (used as a default factory)."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Return a short, prefixed, collision-safe identifier, e.g. ``det_1a2b3c4d``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _coerce_int(value: Any) -> Any:
    """Best-effort string -> int coercion; passes anything else through untouched."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return value
        try:
            return int(float(text))
        except ValueError:
            return value
    if isinstance(value, float):
        return int(value)
    return value


def _coerce_float(value: Any) -> Any:
    """Best-effort string -> float coercion, tolerating ``%`` and thousands separators."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").rstrip("%")
        if not text:
            return value
        try:
            return float(text)
        except ValueError:
            return value
    return value


_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "on"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "off", "none", "null", ""}


def _coerce_bool(value: Any) -> Any:
    """Best-effort string -> bool coercion for LLM/JSON payloads."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_TOKENS:
            return True
        if text in _FALSE_TOKENS:
            return False
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return value


#: ``int`` that also accepts ``"3"`` / ``3.0``.
LenientInt = Annotated[int, BeforeValidator(_coerce_int)]
#: ``float`` that also accepts ``"0.83"`` / ``"83%"``.
LenientFloat = Annotated[float, BeforeValidator(_coerce_float)]
#: ``bool`` that also accepts ``"yes"`` / ``"true"`` / ``1``.
LenientBool = Annotated[bool, BeforeValidator(_coerce_bool)]


class BaseSchema(BaseModel):
    """Common base: enum values on dump, populate by field name, strip strings."""

    model_config = ConfigDict(
        use_enum_values=False,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class EntityType(str, Enum):
    """The kind of actor generating a behavioral event."""

    USER = "user"
    SERVICE_ACCOUNT = "service_account"
    EDGE_DEVICE = "edge_device"


class AuthMethod(str, Enum):
    """How the actor authenticated for this access/connection."""

    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MFA = "mfa"


class AnomalyType(str, Enum):
    """The label space: ``normal`` plus the eight injected attack behaviors."""

    NORMAL = "normal"
    CREDENTIAL_MISUSE = "credential_misuse"
    LATERAL_MOVEMENT = "lateral_movement"
    BRUTE_FORCE = "brute_force"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    CREDENTIAL_STUFFING = "credential_stuffing"
    DEVICE_SPOOFING = "device_spoofing"
    LOW_AND_SLOW_EXFIL = "low_and_slow_exfil"
    INSIDER_DRIFT = "insider_drift"


#: Canonical, **ordered** class list. Model output columns, confusion-matrix axes and the
#: dashboard legend all index into this exact order -- never reorder it.
ANOMALY_CLASSES: List[str] = [
    AnomalyType.NORMAL.value,
    AnomalyType.CREDENTIAL_MISUSE.value,
    AnomalyType.LATERAL_MOVEMENT.value,
    AnomalyType.BRUTE_FORCE.value,
    AnomalyType.IMPOSSIBLE_TRAVEL.value,
    AnomalyType.CREDENTIAL_STUFFING.value,
    AnomalyType.DEVICE_SPOOFING.value,
    AnomalyType.LOW_AND_SLOW_EXFIL.value,
    AnomalyType.INSIDER_DRIFT.value,
]

#: Attack classes only (everything except ``normal``).
ATTACK_CLASSES: List[str] = ANOMALY_CLASSES[1:]

#: Stable index lookup for the class list.
ANOMALY_CLASS_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ANOMALY_CLASSES)}


class DetectionStatus(str, Enum):
    """Analyst triage state of a detection."""

    NEW = "new"
    REVIEWING = "reviewing"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class CampaignStatus(str, Enum):
    """Lifecycle of a reconstructed multi-stage attack campaign."""

    OPEN = "open"
    CLOSED = "closed"


class AnalystVerdict(str, Enum):
    """Feedback an analyst can give on a detection."""

    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"


class DriftStatus(str, Enum):
    """Per-entity concept-drift state derived from PSI."""

    STABLE = "stable"
    DRIFTING = "drifting"
    ADAPTED = "adapted"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


class GeoLocation(BaseSchema):
    """Where an access originated."""

    country: str = Field(..., description="ISO-ish country name or code.")
    city: Optional[str] = Field(default=None)
    lat: LenientFloat = Field(..., ge=-90.0, le=90.0)
    lon: LenientFloat = Field(..., ge=-180.0, le=180.0)


class DeviceFingerprint(BaseSchema):
    """Stable-ish identity of the connecting device."""

    os: str = Field(..., description="OS or firmware string.")
    mac: str = Field(..., description="MAC address (synthetic).")
    protocol: str = Field(..., description="Access protocol, e.g. https, ssh, modbus.")
    user_agent: Optional[str] = Field(default=None)

    @field_validator("mac")
    @classmethod
    def _normalize_mac(cls, value: str) -> str:
        return value.strip().lower()


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #


class Event(BaseSchema):
    """One access/connection event -- the atomic unit the system scores.

    Ground-truth fields (``label``, ``campaign_id``, ``stage``) are optional and are
    stripped before the event reaches any model.
    """

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    entity_id: str = Field(..., min_length=1)
    entity_type: EntityType
    timestamp: datetime

    source_ip: str = Field(..., min_length=3)
    geo: GeoLocation

    resource_accessed: str = Field(..., min_length=1)
    auth_method: AuthMethod
    auth_success: LenientBool = Field(default=True)

    session_id: Optional[str] = Field(default=None)
    session_duration: LenientFloat = Field(
        default=0.0, ge=0.0, description="Connection length in seconds."
    )

    command_sequence: List[str] = Field(
        default_factory=list, description="Ordered commands/actions in this event."
    )
    device_fingerprint: DeviceFingerprint

    bytes_out: LenientFloat = Field(default=0.0, ge=0.0)
    bytes_in: LenientFloat = Field(default=0.0, ge=0.0)

    # --- ground truth (training/eval only; never used at inference) ---
    label: Optional[AnomalyType] = Field(default=None)
    campaign_id: Optional[str] = Field(default=None)
    stage: Optional[LenientInt] = Field(default=None, ge=0)

    ingested_at: datetime = Field(default_factory=utc_now)

    def to_unlabeled(self) -> "Event":
        """Return a copy with ground truth removed (used on the serving path)."""
        return self.model_copy(update={"label": None, "campaign_id": None, "stage": None})


class Session(BaseSchema):
    """A group of events belonging to one login/connection session."""

    session_id: str = Field(default_factory=lambda: new_id("ses"))
    entity_id: str
    entity_type: EntityType
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration: LenientFloat = Field(default=0.0, ge=0.0)
    event_count: LenientInt = Field(default=0, ge=0)
    resources: List[str] = Field(default_factory=list)
    command_sequence: List[str] = Field(default_factory=list)
    auth_failures: LenientInt = Field(default=0, ge=0)
    label: Optional[AnomalyType] = None


# --------------------------------------------------------------------------- #
# Entity profile
# --------------------------------------------------------------------------- #


class DriftState(BaseSchema):
    """Rolling drift bookkeeping for one entity."""

    psi: LenientFloat = Field(default=0.0, ge=0.0)
    status: DriftStatus = DriftStatus.STABLE
    last_refresh: Optional[datetime] = None
    samples_seen: LenientInt = Field(default=0, ge=0)


class EntityProfile(BaseSchema):
    """The learned notion of "normal" for a single entity."""

    entity_id: str
    entity_type: EntityType
    cohort: Optional[LenientInt] = Field(
        default=None, ge=0, description="Behavioral cohort id used for cold-start priors."
    )

    session_count: LenientInt = Field(default=0, ge=0)
    event_count: LenientInt = Field(default=0, ge=0)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    typical_login_hours: List[int] = Field(default_factory=list)
    typical_geo: List[GeoLocation] = Field(default_factory=list)
    typical_countries: List[str] = Field(default_factory=list)
    typical_resources: Dict[str, float] = Field(
        default_factory=dict, description="resource -> access frequency share."
    )
    auth_method_dist: Dict[str, float] = Field(default_factory=dict)
    device_fingerprints: List[str] = Field(default_factory=list)

    feature_names: List[str] = Field(default_factory=list)
    feature_means: List[float] = Field(default_factory=list)
    feature_stds: List[float] = Field(default_factory=list)
    sequence_ngram_profile: Dict[str, float] = Field(default_factory=dict)

    cold_start: LenientBool = Field(default=True)
    drift: DriftState = Field(default_factory=DriftState)
    feedback_threshold_adjust: LenientFloat = Field(
        default=0.0,
        description="Signed risk offset learned from analyst feedback (D6).",
    )

    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("typical_login_hours")
    @classmethod
    def _validate_hours(cls, value: List[int]) -> List[int]:
        for hour in value:
            if not 0 <= int(hour) <= 23:
                raise ValueError(f"login hour out of range: {hour}")
        return [int(h) for h in value]


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #


class FeatureAttribution(BaseSchema):
    """One SHAP-style feature contribution on a single detection."""

    feature: str
    value: Optional[Any] = Field(default=None, description="Observed feature value.")
    contribution: LenientFloat = Field(
        ..., description="Signed push toward (positive) or away from anomalous."
    )
    direction: str = Field(default="increases_risk")
    baseline_value: Optional[Any] = Field(
        default=None, description="The entity's usual value for this feature."
    )
    description: Optional[str] = Field(default=None)

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, value: str) -> str:
        allowed = {"increases_risk", "decreases_risk", "neutral"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"direction must be one of {sorted(allowed)}")
        return normalized


class CounterfactualChange(BaseSchema):
    """A single "change this to make it normal" edit (D2)."""

    feature: str
    actual: Optional[Any] = None
    suggested: Optional[Any] = None
    description: Optional[str] = None


class Counterfactual(BaseSchema):
    """The minimal set of changes that would flip the verdict to benign (D2)."""

    changes: List[CounterfactualChange] = Field(default_factory=list)
    resulting_risk: Optional[LenientFloat] = Field(default=None, ge=0.0, le=100.0)
    original_risk: Optional[LenientFloat] = Field(default=None, ge=0.0, le=100.0)
    found: LenientBool = Field(default=False)
    summary: Optional[str] = None


class MitreTechnique(BaseSchema):
    """A MITRE ATT&CK technique mapped from the predicted anomaly type."""

    technique_id: str = Field(..., description="e.g. T1110")
    name: str
    tactic: Optional[str] = None
    url: Optional[str] = None
    confidence: LenientFloat = Field(default=1.0, ge=0.0, le=1.0)


class SequenceStepAttribution(BaseSchema):
    """Per-step surprise from the sequence model."""

    position: LenientInt = Field(..., ge=0)
    token: str
    score: LenientFloat = Field(..., description="Normalized per-step anomaly weight.")


class BaselineComparison(BaseSchema):
    """Structured diff of this event against the entity's learned profile."""

    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="field -> {observed, typical, deviates} triples.",
    )
    summary: Optional[str] = None


class Explanation(BaseSchema):
    """Everything an analyst needs to understand one detection."""

    top_features: List[FeatureAttribution] = Field(default_factory=list)
    counterfactual: Optional[Counterfactual] = None
    sequence_attribution: List[SequenceStepAttribution] = Field(default_factory=list)
    mitre: List[MitreTechnique] = Field(default_factory=list)
    baseline_comparison: Optional[BaselineComparison] = None
    narrative: Optional[str] = None
    narrative_source: str = Field(
        default="template", description="template | llm -- never affects the score."
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


class DetectionScores(BaseSchema):
    """The raw per-tier scores that feed risk fusion."""

    baseline: LenientFloat = Field(default=0.0, ge=0.0, le=1.0)
    sequence: LenientFloat = Field(default=0.0, ge=0.0, le=1.0)
    classifier_confidence: LenientFloat = Field(default=0.0, ge=0.0, le=1.0)
    fused_raw: Optional[LenientFloat] = Field(default=None, ge=0.0, le=1.0)


class Detection(BaseSchema):
    """The scored output for one event: risk, type, explanation, campaign link."""

    detection_id: str = Field(default_factory=lambda: new_id("det"))
    entity_id: str
    entity_type: EntityType
    timestamp: datetime
    event_ref: Optional[str] = Field(default=None, description="Source event_id.")
    session_id: Optional[str] = None

    scores: DetectionScores = Field(default_factory=DetectionScores)
    risk_score: LenientFloat = Field(default=0.0, ge=0.0, le=100.0)
    risk_uncertainty: LenientFloat = Field(
        default=0.0, ge=0.0, description="Half-width of the risk confidence band (D5)."
    )
    in_alert_budget: LenientBool = Field(default=False)
    is_anomaly: LenientBool = Field(default=False)

    anomaly_type: AnomalyType = AnomalyType.NORMAL
    anomaly_type_probs: Dict[str, float] = Field(default_factory=dict)
    detector_hits: List[str] = Field(
        default_factory=list, description="Deterministic detectors that fired."
    )

    explanation: Explanation = Field(default_factory=Explanation)

    campaign_id: Optional[str] = None
    cold_start: LenientBool = Field(default=False)
    drift_flag: LenientBool = Field(default=False)

    status: DetectionStatus = DetectionStatus.NEW
    ground_truth_label: Optional[AnomalyType] = Field(
        default=None, description="Eval only; never shown to the models."
    )
    analyst_feedback: Optional[AnalystVerdict] = None

    created_at: datetime = Field(default_factory=utc_now)

    @property
    def risk_band(self) -> str:
        """Coarse severity band used for dashboard colouring."""
        if self.risk_score >= 80:
            return "critical"
        if self.risk_score >= 60:
            return "high"
        if self.risk_score >= 40:
            return "medium"
        return "low"


# --------------------------------------------------------------------------- #
# Campaign (D1)
# --------------------------------------------------------------------------- #


class CampaignStage(BaseSchema):
    """One linked step in a reconstructed attack storyline."""

    anomaly_type: AnomalyType
    detection_id: str
    timestamp: datetime
    risk_score: LenientFloat = Field(default=0.0, ge=0.0, le=100.0)


class Campaign(BaseSchema):
    """A multi-stage attack narrative reconstructed from linked detections (D1)."""

    campaign_id: str = Field(default_factory=lambda: new_id("cmp"))
    entity_id: str
    entity_type: Optional[EntityType] = None
    started_at: datetime
    last_activity: datetime
    stages: List[CampaignStage] = Field(default_factory=list)
    detection_ids: List[str] = Field(default_factory=list)
    kill_chain: List[str] = Field(
        default_factory=list, description="Ordered technique/stage names."
    )
    max_risk: LenientFloat = Field(default=0.0, ge=0.0, le=100.0)
    status: CampaignStatus = CampaignStatus.OPEN
    ground_truth_campaign_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def stage_count(self) -> int:
        """Number of linked stages in this campaign."""
        return len(self.stages)


# --------------------------------------------------------------------------- #
# Feedback (D6)
# --------------------------------------------------------------------------- #


class FeedbackAdjustment(BaseSchema):
    """What the system actually changed in response to analyst feedback."""

    scope: str = Field(default="entity", description="entity | cohort | global")
    scope_id: Optional[str] = None
    adjustment: LenientFloat = Field(default=0.0)
    previous_value: Optional[LenientFloat] = None
    new_value: Optional[LenientFloat] = None


class Feedback(BaseSchema):
    """An analyst verdict on a detection, plus the adjustment it triggered."""

    feedback_id: str = Field(default_factory=lambda: new_id("fbk"))
    detection_id: str
    entity_id: str
    analyst_verdict: AnalystVerdict
    note: Optional[str] = None
    applied: Optional[FeedbackAdjustment] = None
    created_at: datetime = Field(default_factory=utc_now)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class DatasetSummary(BaseSchema):
    """Shape of the dataset a training run was evaluated on."""

    n_events: LenientInt = Field(default=0, ge=0)
    n_entities: LenientInt = Field(default=0, ge=0)
    anomaly_rate: LenientFloat = Field(default=0.0, ge=0.0, le=1.0)
    per_class_counts: Dict[str, int] = Field(default_factory=dict)
    split: Optional[str] = None


class ColdStartMetrics(BaseSchema):
    """Cold-start ablation results (with vs without cohort priors)."""

    recall_with_priors: Optional[LenientFloat] = None
    recall_without_priors: Optional[LenientFloat] = None
    uplift: Optional[LenientFloat] = None
    n_cold_entities: LenientInt = Field(default=0, ge=0)


class DriftMetrics(BaseSchema):
    """Drift-experiment results before/after adaptation."""

    fp_rate_before: Optional[LenientFloat] = None
    fp_rate_after: Optional[LenientFloat] = None
    adaptation_events: LenientInt = Field(default=0, ge=0)
    mean_psi: Optional[LenientFloat] = None


class CampaignMetrics(BaseSchema):
    """Campaign-reconstruction accuracy (D1)."""

    stages_linked_correctly: Optional[LenientFloat] = None
    campaigns_reconstructed: LenientInt = Field(default=0, ge=0)
    campaigns_expected: LenientInt = Field(default=0, ge=0)


class ModelMetrics(BaseSchema):
    """One evaluation run: the numbers reported in the final report."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    created_at: datetime = Field(default_factory=utc_now)
    artifact_schema_version: Optional[str] = None
    git_sha: Optional[str] = None
    seed: LenientInt = Field(default=42)

    dataset_summary: DatasetSummary = Field(default_factory=DatasetSummary)

    pr_auc: Optional[LenientFloat] = Field(default=None, ge=0.0, le=1.0)
    roc_auc: Optional[LenientFloat] = Field(default=None, ge=0.0, le=1.0)
    recall_at_1pct_budget: Optional[LenientFloat] = Field(default=None, ge=0.0, le=1.0)
    precision_at_k: List[float] = Field(default_factory=list)
    macro_f1: Optional[LenientFloat] = Field(default=None, ge=0.0, le=1.0)
    calibration_ece: Optional[LenientFloat] = Field(default=None, ge=0.0, le=1.0)

    confusion_matrix: List[List[int]] = Field(default_factory=list)
    class_order: List[str] = Field(default_factory=lambda: list(ANOMALY_CLASSES))
    per_class: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    coldstart: ColdStartMetrics = Field(default_factory=ColdStartMetrics)
    drift: DriftMetrics = Field(default_factory=DriftMetrics)
    campaigns: CampaignMetrics = Field(default_factory=CampaignMetrics)

    notes: Optional[str] = None


# --------------------------------------------------------------------------- #
# API payloads
# --------------------------------------------------------------------------- #


class HealthStatus(BaseSchema):
    """Health of one dependency or of a whole service."""

    status: str = Field(default="ok", description="ok | degraded | error")
    detail: Optional[str] = None
    latency_ms: Optional[LenientFloat] = None


class ServiceHealth(BaseSchema):
    """Aggregate health payload returned by ``/health`` endpoints."""

    service: str
    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
    artifact_schema_version: Optional[str] = None
    artifacts_ready: LenientBool = Field(default=False)
    dependencies: Dict[str, HealthStatus] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=utc_now)


__all__ = [
    # helpers
    "utc_now",
    "new_id",
    "BaseSchema",
    "LenientInt",
    "LenientFloat",
    "LenientBool",
    # enums
    "EntityType",
    "AuthMethod",
    "AnomalyType",
    "ANOMALY_CLASSES",
    "ATTACK_CLASSES",
    "ANOMALY_CLASS_INDEX",
    "DetectionStatus",
    "CampaignStatus",
    "AnalystVerdict",
    "DriftStatus",
    # value objects
    "GeoLocation",
    "DeviceFingerprint",
    # telemetry
    "Event",
    "Session",
    # profile
    "DriftState",
    "EntityProfile",
    # explainability
    "FeatureAttribution",
    "CounterfactualChange",
    "Counterfactual",
    "MitreTechnique",
    "SequenceStepAttribution",
    "BaselineComparison",
    "Explanation",
    # detection
    "DetectionScores",
    "Detection",
    # campaign
    "CampaignStage",
    "Campaign",
    # feedback
    "FeedbackAdjustment",
    "Feedback",
    # metrics
    "DatasetSummary",
    "ColdStartMetrics",
    "DriftMetrics",
    "CampaignMetrics",
    "ModelMetrics",
    # api
    "HealthStatus",
    "ServiceHealth",
]
