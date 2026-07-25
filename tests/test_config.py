"""Configuration contract tests.

Config is loaded by every process in the system, so a bad default or a silent parsing
failure here breaks everything downstream. These tests pin the invariants that other
phases rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from common.config import PROJECT_ROOT, RANDOM_SEED, Settings, get_settings, settings


class TestDefaults:
    """Defaults must produce a working system with no environment at all."""

    def test_seed_is_42(self) -> None:
        assert settings.random_seed == RANDOM_SEED == 42

    def test_mongo_defaults(self) -> None:
        assert settings.mongo_db_name == "anomaly_detection"
        assert settings.mongo_url.startswith("mongodb://")

    def test_alert_budget_is_one_percent(self) -> None:
        assert settings.alert_budget_pct == pytest.approx(0.01)

    def test_redis_disabled_by_default(self) -> None:
        """Streaming is a bonus path; the demo must not require Redis."""
        assert settings.redis_enabled is False

    def test_llm_disabled_by_default(self) -> None:
        """Nothing in the demo path may depend on the internet."""
        assert settings.llm_enabled is False

    def test_mitre_source_defaults_to_static(self) -> None:
        assert settings.mitre_map_source == "static"

    def test_scoring_auth_enabled_by_default(self) -> None:
        """The only service with a write path must be authenticated out of the box."""
        assert settings.scoring_auth_enabled is True
        assert settings.scoring_auth_token


class TestFusionWeights:
    """Fusion weights are a probability-like split and must sum to exactly 1.0."""

    def test_default_weights_sum_to_one(self) -> None:
        assert sum(settings.fusion_weights.values()) == pytest.approx(1.0)

    def test_fusion_weights_mapping_has_three_tiers(self) -> None:
        assert set(settings.fusion_weights) == {"baseline", "sequence", "classifier"}

    def test_invalid_weights_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            Settings(
                fusion_weight_baseline=0.5,
                fusion_weight_sequence=0.5,
                fusion_weight_classifier=0.5,
            )

    def test_custom_weights_summing_to_one_are_accepted(self) -> None:
        custom = Settings(
            fusion_weight_baseline=0.2,
            fusion_weight_sequence=0.3,
            fusion_weight_classifier=0.5,
        )
        assert custom.fusion_weights["classifier"] == pytest.approx(0.5)


class TestPaths:
    """Artifact paths must resolve to absolute locations inside the repository."""

    def test_project_root_contains_common_package(self) -> None:
        assert (PROJECT_ROOT / "common" / "config.py").exists()

    def test_artifacts_dir_is_absolute(self) -> None:
        assert settings.artifacts_dir.is_absolute()

    def test_relative_artifacts_dir_resolves_against_root(self) -> None:
        custom = Settings(artifacts_dir=Path("custom_artifacts"))
        assert custom.artifacts_dir == PROJECT_ROOT / "custom_artifacts"

    def test_absolute_artifacts_dir_is_preserved(self, tmp_path: Path) -> None:
        custom = Settings(artifacts_dir=tmp_path)
        assert custom.artifacts_dir == tmp_path

    def test_derived_paths(self) -> None:
        assert settings.dataset_dir == settings.artifacts_dir / "dataset"
        assert settings.manifest_path == settings.artifacts_dir / "manifest.json"
        assert settings.metrics_path == settings.artifacts_dir / "metrics.json"

    def test_ensure_dirs_creates_layout(self, tmp_path: Path) -> None:
        custom = Settings(artifacts_dir=tmp_path / "art")
        custom.ensure_dirs()
        assert custom.artifacts_dir.is_dir()
        assert custom.dataset_dir.is_dir()


class TestValidation:
    """Bad values fail loudly at construction, not deep inside a training run."""

    def test_alert_budget_must_be_a_fraction(self) -> None:
        with pytest.raises(ValidationError):
            Settings(alert_budget_pct=1.5)

    def test_alert_budget_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Settings(alert_budget_pct=0.0)

    def test_unknown_mitre_source_rejected(self) -> None:
        with pytest.raises(ValidationError, match="static"):
            Settings(mitre_map_source="pinecone")

    def test_mitre_source_is_case_insensitive(self) -> None:
        assert Settings(mitre_map_source="QDRANT").mitre_map_source == "qdrant"

    def test_min_sessions_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Settings(entity_history_min_sessions=0)


class TestEnvironmentLoading:
    """Env parsing must be tolerant of real-world shell and container behavior."""

    def test_env_prefix_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ADP_MONGO_DB_NAME", "test_db")
        assert Settings().mongo_db_name == "test_db"

    def test_unknown_env_vars_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stray ADP_* variable must never crash startup."""
        monkeypatch.setenv("ADP_TOTALLY_MADE_UP_SETTING", "boom")
        assert Settings().random_seed == 42

    def test_cors_origins_accept_comma_separated_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Docker and CI can only pass strings, so a CSV list must parse."""
        monkeypatch.setenv(
            "ADP_API_CORS_ORIGINS", "http://a.test, http://b.test ,http://c.test"
        )
        assert Settings().api_cors_origins == [
            "http://a.test",
            "http://b.test",
            "http://c.test",
        ]

    def test_cors_origins_accept_json_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ADP_API_CORS_ORIGINS", '["http://a.test"]')
        assert Settings().api_cors_origins == ["http://a.test"]

    def test_boolean_env_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ADP_REDIS_ENABLED", "true")
        assert Settings().redis_enabled is True


class TestSingleton:
    """The settings singleton is cached so all modules see the same object."""

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_module_level_singleton_matches(self) -> None:
        assert settings is get_settings()
