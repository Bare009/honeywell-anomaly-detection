"""Central configuration for the behavioral anomaly detection system.

Every tunable in the system lives here so that the offline training plane, the online
serving plane and the read API all agree on the same numbers. Values are read from the
environment with the ``ADP_`` prefix (``ADP_`` = Anomaly Detection Platform), optionally
via a ``.env`` file, and every field has a sensible default so the system runs with no
configuration at all.

``extra="ignore"`` is deliberate: stray ``ADP_*`` variables left over in a shell or CI
runner must never crash startup.

Example
-------
>>> from common.config import settings
>>> settings.random_seed
42
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Repository root: <root>/common/config.py -> parents[1] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The one and only global seed. Referenced by :mod:`common.seed`.
RANDOM_SEED = 42

#: List-valued settings that must accept a plain comma-separated string from the
#: environment. Docker Compose and CI runners can only supply strings, and demanding JSON
#: for something like a CORS origin list is a needless trap.
_CSV_LIST_FIELDS = frozenset({"api_cors_origins"})


class _CsvTolerantMixin:
    """Skip pydantic-settings' JSON decoding for the fields in ``_CSV_LIST_FIELDS``.

    By default an env source JSON-decodes any value destined for a list field and raises
    before field validators run, so ``ADP_API_CORS_ORIGINS=http://a,http://b`` would be a
    hard startup failure. Handing the raw string through lets the field validator split it.
    """

    def prepare_field_value(
        self, field_name: str, field: Any, value: Any, value_is_complex: bool
    ) -> Any:
        if field_name in _CSV_LIST_FIELDS and isinstance(value, str):
            return value
        return super().prepare_field_value(  # type: ignore[misc]
            field_name, field, value, value_is_complex
        )


class _CsvTolerantEnvSource(_CsvTolerantMixin, EnvSettingsSource):
    """Environment-variable source that tolerates CSV list values."""


class _CsvTolerantDotEnvSource(_CsvTolerantMixin, DotEnvSettingsSource):
    """``.env`` file source that tolerates CSV list values."""


class Settings(BaseSettings):
    """Typed, environment-driven settings for every plane of the system."""

    model_config = SettingsConfigDict(
        env_prefix="ADP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #
    random_seed: int = Field(
        default=RANDOM_SEED,
        description="Global seed applied to random, NumPy, PyTorch and LightGBM.",
    )

    # ------------------------------------------------------------------ #
    # Environment / meta
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="behavioral-anomaly-detection")
    environment: str = Field(default="local", description="local | docker | ci")
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------ #
    # MongoDB
    # ------------------------------------------------------------------ #
    mongo_url: str = Field(default="mongodb://localhost:27017")
    mongo_db_name: str = Field(default="anomaly_detection")
    mongo_timeout_ms: int = Field(default=3000, ge=100)

    # ------------------------------------------------------------------ #
    # Redis (optional streaming transport)
    # ------------------------------------------------------------------ #
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_enabled: bool = Field(
        default=False,
        description="When False the streaming path is skipped entirely (demo-safe).",
    )
    redis_stream_events: str = Field(default="adp:events")
    redis_stream_detections: str = Field(default="adp:detections")
    redis_consumer_group: str = Field(default="adp-scorers")
    redis_consumer_name: str = Field(default="scorer-1")

    # ------------------------------------------------------------------ #
    # Artifacts contract
    # ------------------------------------------------------------------ #
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    dataset_dirname: str = Field(default="dataset")
    manifest_filename: str = Field(default="manifest.json")
    metrics_filename: str = Field(default="metrics.json")
    artifact_schema_version: str = Field(
        default="1.0.0",
        description="Bumped when the artifact layout changes; serving refuses a mismatch.",
    )

    # ------------------------------------------------------------------ #
    # Alerting / risk
    # ------------------------------------------------------------------ #
    alert_budget_pct: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Fraction of events an analyst can review (top-N by risk).",
    )
    anomaly_gate_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Fused unsupervised score above which the classifier tier runs.",
    )
    risk_alert_threshold: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="0-100 risk at or above which a detection is treated as an alert.",
    )

    # Fusion weights -- must sum to 1.0 (validated below).
    fusion_weight_baseline: float = Field(default=0.35, ge=0.0, le=1.0)
    fusion_weight_sequence: float = Field(default=0.30, ge=0.0, le=1.0)
    fusion_weight_classifier: float = Field(default=0.35, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ #
    # Cold start
    # ------------------------------------------------------------------ #
    entity_history_min_sessions: int = Field(
        default=15,
        ge=1,
        description="Below this session count an entity is cold-start (cohort priors used).",
    )
    coldstart_uncertainty_multiplier: float = Field(default=2.0, ge=1.0)

    # ------------------------------------------------------------------ #
    # Concept drift
    # ------------------------------------------------------------------ #
    drift_window_size: int = Field(default=200, ge=10)
    drift_min_samples: int = Field(default=50, ge=5)
    drift_psi_threshold: float = Field(default=0.20, gt=0.0)
    drift_psi_bins: int = Field(default=10, ge=2)
    drift_refresh_alpha: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="EWMA rate at which benign drift is absorbed into the baseline.",
    )

    # ------------------------------------------------------------------ #
    # Feature pipeline
    # ------------------------------------------------------------------ #
    sequence_max_len: int = Field(default=20, ge=2)
    sequence_ngram_n: int = Field(default=2, ge=1)
    cohort_count: int = Field(default=6, ge=2)
    impossible_travel_kmh: float = Field(
        default=900.0,
        gt=0.0,
        description="Geo velocity above which travel is physically implausible.",
    )
    brute_force_threshold: int = Field(
        default=5,
        ge=2,
        description="Failed auth attempts within the window that constitute a burst.",
    )
    brute_force_window_minutes: int = Field(default=10, ge=1)
    entity_window_minutes: int = Field(
        default=60,
        ge=1,
        description="Rolling window for per-entity volume/rate features.",
    )

    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    scoring_auth_token: str = Field(
        default="dev-scoring-token",
        description="Bearer token required by the scoring service write endpoints.",
    )
    scoring_auth_enabled: bool = Field(default=True)

    # ------------------------------------------------------------------ #
    # Optional LLM narrator (never affects a score)
    # ------------------------------------------------------------------ #
    llm_enabled: bool = Field(default=False)
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=220, ge=16)
    llm_timeout_seconds: float = Field(default=8.0, gt=0.0)

    # ------------------------------------------------------------------ #
    # MITRE mapping source
    # ------------------------------------------------------------------ #
    mitre_map_source: str = Field(default="static", description="static | qdrant")
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_collection: str = Field(default="mitre_techniques")

    # ------------------------------------------------------------------ #
    # Services
    # ------------------------------------------------------------------ #
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    serving_host: str = Field(default="0.0.0.0")
    serving_port: int = Field(default=8100, ge=1, le=65535)
    api_cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
        ]
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Swap in env sources that accept comma-separated lists.

        Precedence is unchanged: explicit arguments, then environment, then ``.env``, then
        secrets files.
        """
        return (
            init_settings,
            _CsvTolerantEnvSource(settings_cls),
            _CsvTolerantDotEnvSource(settings_cls),
            file_secret_settings,
        )

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: Any) -> Any:
        """Accept a comma-separated string or a JSON array as well as a real list."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass  # fall through to CSV splitting
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("mitre_map_source")
    @classmethod
    def _validate_mitre_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"static", "qdrant"}:
            raise ValueError("mitre_map_source must be 'static' or 'qdrant'")
        return normalized

    @field_validator("artifacts_dir", mode="before")
    @classmethod
    def _resolve_artifacts_dir(cls, value: Any) -> Any:
        """Resolve a relative artifacts path against the project root."""
        if value is None:
            return value
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @model_validator(mode="after")
    def _validate_fusion_weights(self) -> "Settings":
        total = (
            self.fusion_weight_baseline
            + self.fusion_weight_sequence
            + self.fusion_weight_classifier
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Fusion weights must sum to 1.0 "
                f"(baseline={self.fusion_weight_baseline}, "
                f"sequence={self.fusion_weight_sequence}, "
                f"classifier={self.fusion_weight_classifier}, sum={total})"
            )
        return self

    # ------------------------------------------------------------------ #
    # Derived paths
    # ------------------------------------------------------------------ #
    @property
    def dataset_dir(self) -> Path:
        """Directory holding the generated dataset parquet/json files."""
        return self.artifacts_dir / self.dataset_dirname

    @property
    def manifest_path(self) -> Path:
        """Path to the version-stamped artifacts manifest."""
        return self.artifacts_dir / self.manifest_filename

    @property
    def metrics_path(self) -> Path:
        """Path to the evaluation metrics dump written by the training plane."""
        return self.artifacts_dir / self.metrics_filename

    @property
    def fusion_weights(self) -> dict[str, float]:
        """Fusion weights as a mapping, in tier order."""
        return {
            "baseline": self.fusion_weight_baseline,
            "sequence": self.fusion_weight_sequence,
            "classifier": self.fusion_weight_classifier,
        }

    def ensure_dirs(self) -> None:
        """Create the artifact directories if they do not already exist."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()


#: Convenience module-level singleton for the common ``from common.config import settings``.
settings = get_settings()

__all__ = ["PROJECT_ROOT", "RANDOM_SEED", "Settings", "get_settings", "settings"]
