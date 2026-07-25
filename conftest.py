"""Root pytest configuration.

Puts the project root on ``sys.path`` so tests can ``import common`` (and later
``import features``, ``import models``, ...) without the package being installed, and
provides the fixtures every phase's tests reuse.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import settings  # noqa: E402
from common.models import (  # noqa: E402
    AuthMethod,
    DeviceFingerprint,
    EntityType,
    Event,
    GeoLocation,
)
from common.seed import set_global_seed  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Reseed every source of randomness before each test.

    Autouse: no test may depend on the random state left behind by another test.
    """
    set_global_seed(settings.random_seed)


@pytest.fixture
def fixed_timestamp() -> datetime:
    """A stable reference timestamp so assertions never depend on wall-clock time."""
    return datetime(2026, 3, 2, 9, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_geo() -> GeoLocation:
    """A benign home location (Bengaluru)."""
    return GeoLocation(country="India", city="Bengaluru", lat=12.9716, lon=77.5946)


@pytest.fixture
def sample_device() -> DeviceFingerprint:
    """A consistent corporate laptop fingerprint."""
    return DeviceFingerprint(
        os="Windows 11",
        mac="a4:5e:60:1b:2c:3d",
        protocol="https",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    )


@pytest.fixture
def sample_event(
    fixed_timestamp: datetime,
    sample_geo: GeoLocation,
    sample_device: DeviceFingerprint,
) -> Event:
    """A single well-formed benign event."""
    return Event(
        event_id="evt_test000001",
        entity_id="user_0001",
        entity_type=EntityType.USER,
        timestamp=fixed_timestamp,
        source_ip="10.24.8.15",
        geo=sample_geo,
        resource_accessed="/api/reports/quarterly",
        auth_method=AuthMethod.PASSWORD,
        auth_success=True,
        session_id="ses_test000001",
        session_duration=412.5,
        command_sequence=["login", "list_reports", "open_report", "logout"],
        device_fingerprint=sample_device,
        bytes_out=18432.0,
        bytes_in=2048.0,
    )


@pytest.fixture
def sample_event_payload(sample_event: Event) -> dict:
    """The same benign event as a JSON-ready dict (what the scoring API receives)."""
    return sample_event.model_dump(mode="json")


@pytest.fixture
def sample_event_sequence(sample_event: Event) -> list[Event]:
    """Five sequential events for one entity, ten minutes apart."""
    events = []
    for offset in range(5):
        events.append(
            sample_event.model_copy(
                update={
                    "event_id": f"evt_seq{offset:06d}",
                    "timestamp": sample_event.timestamp + timedelta(minutes=10 * offset),
                }
            )
        )
    return events


@pytest.fixture
def tmp_artifacts_dir(tmp_path: Path) -> Path:
    """An isolated ``artifacts/`` directory so tests never touch real trained state."""
    artifacts = tmp_path / "artifacts"
    (artifacts / "dataset").mkdir(parents=True, exist_ok=True)
    return artifacts
