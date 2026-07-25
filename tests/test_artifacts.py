"""Artifacts contract tests.

The manifest is what stops a stale or incompatible model from silently scoring traffic
after a schema change. Reading it must never raise (a missing manifest just means "nothing
trained yet"), and a version mismatch must be detectable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.artifacts import (
    ARTIFACT_KEYS,
    ArtifactSchemaMismatch,
    artifacts_ready,
    check_schema_version,
    empty_manifest,
    manifest_schema_version,
    read_manifest,
    write_manifest,
)
from common.config import settings


class TestEmptyManifest:
    """The placeholder manifest shape the repository ships with."""

    def test_has_expected_top_level_keys(self) -> None:
        manifest = empty_manifest()
        for key in ("schema_version", "seed", "created_at", "git_sha", "artifacts"):
            assert key in manifest

    def test_all_artifact_slots_present_and_unfilled(self) -> None:
        slots = empty_manifest()["artifacts"]
        assert set(slots) == set(ARTIFACT_KEYS)
        assert all(value is None for value in slots.values())

    def test_stamped_with_current_schema_and_seed(self) -> None:
        manifest = empty_manifest()
        assert manifest["schema_version"] == settings.artifact_schema_version
        assert manifest["seed"] == settings.random_seed


class TestReadManifest:
    """Reading is defensive: it degrades instead of raising."""

    def test_missing_file_returns_placeholder(self, tmp_path: Path) -> None:
        manifest = read_manifest(tmp_path / "does_not_exist.json")
        assert manifest["created_at"] is None

    def test_corrupt_json_returns_placeholder(self, tmp_path: Path) -> None:
        """A truncated manifest must not crash a service on startup."""
        broken = tmp_path / "manifest.json"
        broken.write_text("{ not json", encoding="utf-8")
        assert read_manifest(broken)["created_at"] is None

    def test_non_object_json_returns_placeholder(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_manifest(path)["artifacts"] == {key: None for key in ARTIFACT_KEYS}

    def test_reads_real_content(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"schema_version": "9.9.9"}), encoding="utf-8")
        assert read_manifest(path)["schema_version"] == "9.9.9"


class TestWriteManifest:
    """Writing always stamps provenance so an artifact set is traceable."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest({"artifacts": {"classifier": "classifier.txt"}}, path)
        manifest = read_manifest(path)
        assert manifest["artifacts"]["classifier"] == "classifier.txt"

    def test_stamps_created_at_and_version(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest({}, path)
        manifest = read_manifest(path)
        assert manifest["created_at"] is not None
        assert manifest["schema_version"] == settings.artifact_schema_version
        assert manifest["seed"] == settings.random_seed

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "manifest.json"
        write_manifest({}, path)
        assert path.exists()

    def test_written_json_is_human_readable(self, tmp_path: Path) -> None:
        """Indented output keeps the tracked manifest reviewable in a diff."""
        path = tmp_path / "manifest.json"
        write_manifest({}, path)
        assert "\n  " in path.read_text(encoding="utf-8")


class TestSchemaVersionCheck:
    """A mismatch must be detectable and, when asked, fatal."""

    def test_matching_version_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest({}, path)
        assert check_schema_version(path) is True

    def test_mismatch_returns_false_when_not_strict(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"schema_version": "0.0.1"}), encoding="utf-8")
        assert check_schema_version(path) is False

    def test_mismatch_raises_when_strict(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"schema_version": "0.0.1"}), encoding="utf-8")
        with pytest.raises(ArtifactSchemaMismatch, match="schema mismatch"):
            check_schema_version(path, strict=True)

    def test_version_reported_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"schema_version": "2.5.0"}), encoding="utf-8")
        assert manifest_schema_version(path) == "2.5.0"


class TestArtifactsReady:
    """Readiness means training has actually produced something loadable."""

    def test_placeholder_is_not_ready(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(empty_manifest()), encoding="utf-8")
        assert artifacts_ready(path) is False

    def test_missing_manifest_is_not_ready(self, tmp_path: Path) -> None:
        assert artifacts_ready(tmp_path / "nope.json") is False

    def test_stamped_but_empty_slots_is_not_ready(self, tmp_path: Path) -> None:
        """A timestamp alone does not mean a model exists."""
        path = tmp_path / "manifest.json"
        write_manifest({"artifacts": {key: None for key in ARTIFACT_KEYS}}, path)
        assert artifacts_ready(path) is False

    def test_named_but_missing_files_is_not_ready(self, tmp_path: Path) -> None:
        """The manifest is tracked in git while the artifacts are not.

        A fresh clone therefore has a manifest naming files it does not have. Trusting the
        manifest alone would let serving report readiness and then fail on the first request.
        """
        path = tmp_path / "manifest.json"
        write_manifest({"artifacts": {"classifier": "classifier.txt"}}, path)
        assert artifacts_ready(path) is False

    def test_populated_manifest_with_real_files_is_ready(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        (tmp_path / "classifier.txt").write_text("model", encoding="utf-8")
        write_manifest({"artifacts": {"classifier": "classifier.txt"}}, path)
        assert artifacts_ready(path) is True


class TestRepositoryManifest:
    """The manifest tracked in git must be a valid, unfilled placeholder."""

    def test_tracked_manifest_exists_and_parses(self, project_root: Path) -> None:
        path = project_root / "artifacts" / "manifest.json"
        assert path.exists()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == settings.artifact_schema_version
        assert manifest["seed"] == settings.random_seed

    def test_gitkeep_present(self, project_root: Path) -> None:
        assert (project_root / "artifacts" / ".gitkeep").exists()

    def test_tracked_manifest_reports_a_consistent_state(self, project_root: Path) -> None:
        """The manifest must agree with itself about whether artifacts exist.

        Readiness is not asserted either way: the manifest starts as a placeholder and becomes
        populated once ``training/build_baselines.py`` has run locally. What must always hold is
        the internal consistency -- a manifest claiming readiness has to name at least one
        artifact, and one that names none must not claim readiness.
        """
        path = project_root / "artifacts" / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        slots = manifest.get("artifacts") or {}
        filled = [key for key, value in slots.items() if value]

        if artifacts_ready(path):
            assert filled, "manifest claims readiness but names no artifacts"
            assert manifest.get("created_at")
        else:
            assert not filled or manifest.get("created_at") is None
