"""Health reporting for the read API.

Health checks must never raise and never hang. Each dependency is probed independently, so
a missing Redis (which is optional) degrades the report rather than failing it, and the
endpoint still answers 200 with an honest per-dependency status.
"""

from __future__ import annotations

import logging
from typing import Dict

from common.artifacts import artifacts_ready, manifest_schema_version
from common.config import settings
from common.database import check_mongo_health, check_redis_health
from common.models import HealthStatus, ServiceHealth

logger = logging.getLogger(__name__)

#: Reported by ``/health`` so the dashboard can show which build it is talking to.
SERVICE_VERSION = "0.1.0"


def _aggregate_status(dependencies: Dict[str, HealthStatus]) -> str:
    """Roll per-dependency statuses into one service status.

    ``disabled`` is not a failure -- optional components (Redis streaming) report it by
    design, so they are excluded from the rollup.
    """
    relevant = [
        dep.status for dep in dependencies.values() if dep.status != "disabled"
    ]
    if any(status == "error" for status in relevant):
        return "degraded"
    return "ok"


async def get_liveness() -> ServiceHealth:
    """Cheap liveness probe: does the process respond at all?

    Deliberately touches no external dependency, so it stays fast and stays 200 even while
    MongoDB is still starting up.
    """
    return ServiceHealth(
        service="api",
        status="ok",
        version=SERVICE_VERSION,
        artifact_schema_version=manifest_schema_version(),
        artifacts_ready=artifacts_ready(),
        dependencies={},
    )


async def get_readiness() -> ServiceHealth:
    """Full dependency probe: MongoDB, Redis (if enabled) and the artifacts manifest."""
    dependencies: Dict[str, HealthStatus] = {
        "mongodb": await check_mongo_health(),
        "redis": await check_redis_health(),
    }

    ready = artifacts_ready()
    dependencies["artifacts"] = HealthStatus(
        status="ok" if ready else "degraded",
        detail=(
            f"schema {manifest_schema_version()} at {settings.artifacts_dir}"
            if ready
            else "no trained artifacts yet -- run training/build_artifacts.py"
        ),
    )

    return ServiceHealth(
        service="api",
        status=_aggregate_status(dependencies),
        version=SERVICE_VERSION,
        artifact_schema_version=manifest_schema_version(),
        artifacts_ready=ready,
        dependencies=dependencies,
    )


__all__ = ["SERVICE_VERSION", "get_liveness", "get_readiness"]
