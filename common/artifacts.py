"""The artifacts contract: one place that reads and writes ``artifacts/manifest.json``.

Every piece of trained state lives under ``artifacts/`` and is described by a
version-stamped manifest. The serving plane checks that manifest at startup and refuses to
run against an incompatible layout, which is what stops a stale model silently scoring
production traffic after a schema change.

Both the training plane (writer) and the serving/read planes (readers) go through this
module so the manifest shape is defined exactly once.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config import settings

logger = logging.getLogger(__name__)

#: Artifact slots the training plane is expected to fill in, in build order.
ARTIFACT_KEYS: List[str] = [
    "encoders",
    "scaler",
    "entity_profiles",
    "cohorts",
    "sequence_vocab",
    "baseline_model",
    "autoencoder",
    "sequence_model",
    "classifier",
    "calibrator",
    "fusion",
    "thresholds",
    "shap_background",
]


class ArtifactSchemaMismatch(RuntimeError):
    """Raised when a manifest's ``schema_version`` does not match the running code."""


def empty_manifest() -> Dict[str, Any]:
    """Return a fresh placeholder manifest with all slots unfilled."""
    return {
        "schema_version": settings.artifact_schema_version,
        "seed": settings.random_seed,
        "created_at": None,
        "git_sha": None,
        "dataset": None,
        "artifacts": {key: None for key in ARTIFACT_KEYS},
        "metrics": None,
        "notes": "Placeholder manifest. Populated by training/build_artifacts.py.",
    }


def current_git_sha() -> Optional[str]:
    """Return the short git SHA of the working tree, or ``None`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def read_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the manifest from disk, returning a placeholder if it is missing/corrupt.

    Reading never raises: a missing manifest simply means "nothing trained yet", which the
    health endpoints report as ``artifacts_ready = False``.
    """
    manifest_path = path or settings.manifest_path
    if not manifest_path.exists():
        logger.debug("No manifest at %s; returning placeholder", manifest_path)
        return empty_manifest()
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable manifest at %s: %s", manifest_path, exc)
        return empty_manifest()
    if not isinstance(data, dict):
        logger.warning("Manifest at %s is not an object; ignoring", manifest_path)
        return empty_manifest()
    return data


def write_manifest(manifest: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Write the manifest, stamping ``created_at``, ``git_sha``, seed and schema version."""
    manifest_path = path or settings.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    stamped = dict(manifest)
    stamped.setdefault("artifacts", {})
    stamped["schema_version"] = settings.artifact_schema_version
    stamped["seed"] = settings.random_seed
    stamped["created_at"] = datetime.now(timezone.utc).isoformat()
    stamped["git_sha"] = current_git_sha()

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(stamped, handle, indent=2, sort_keys=False)
        handle.write("\n")

    logger.info("Wrote manifest to %s", manifest_path)
    return manifest_path


def manifest_schema_version(path: Optional[Path] = None) -> Optional[str]:
    """Return the ``schema_version`` recorded in the manifest, if any."""
    version = read_manifest(path).get("schema_version")
    return str(version) if version is not None else None


def check_schema_version(path: Optional[Path] = None, strict: bool = False) -> bool:
    """Verify the on-disk manifest matches ``settings.artifact_schema_version``.

    Parameters
    ----------
    path:
        Manifest location. Defaults to the configured path.
    strict:
        When True, raise :class:`ArtifactSchemaMismatch` instead of returning False.
        Serving uses ``strict=True`` at startup so a mismatch fails loudly and early.
    """
    found = manifest_schema_version(path)
    expected = settings.artifact_schema_version
    if found == expected:
        return True
    message = (
        f"Artifact schema mismatch: manifest reports {found!r}, "
        f"code expects {expected!r}. Re-run training/build_artifacts.py."
    )
    if strict:
        raise ArtifactSchemaMismatch(message)
    logger.warning(message)
    return False


def artifacts_ready(path: Optional[Path] = None) -> bool:
    """True when trained state is actually loadable.

    Checks both that the manifest names artifacts **and that at least one of those files exists
    on disk**. The file check is not redundant: the manifest is tracked in git while the artifact
    binaries are not, so a fresh clone carries a manifest describing files it does not have.
    Trusting the manifest alone would let the serving readiness gate report success and then fail
    on the first scoring request.

    A placeholder manifest with every slot ``None`` counts as not ready.
    """
    manifest_path = path or settings.manifest_path
    manifest = read_manifest(manifest_path)
    if manifest.get("created_at") is None:
        return False

    slots = manifest.get("artifacts") or {}
    if not isinstance(slots, dict):
        return False

    named = [value for value in slots.values() if value]
    if not named:
        return False

    directory = Path(manifest_path).parent
    return any((directory / str(name)).exists() for name in named)


def artifact_path(name: str) -> Path:
    """Resolve a filename inside ``artifacts/``, creating the directory if needed."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings.artifacts_dir / name


def dataset_path(name: str) -> Path:
    """Resolve a filename inside ``artifacts/dataset/``, creating the directory if needed."""
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    return settings.dataset_dir / name


__all__ = [
    "ARTIFACT_KEYS",
    "ArtifactSchemaMismatch",
    "empty_manifest",
    "current_git_sha",
    "read_manifest",
    "write_manifest",
    "manifest_schema_version",
    "check_schema_version",
    "artifacts_ready",
    "artifact_path",
    "dataset_path",
]
