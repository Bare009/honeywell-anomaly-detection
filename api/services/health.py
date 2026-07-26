"""Health reporting for the read API.

Health checks must never raise and never hang. Each dependency is probed independently, so
a missing Redis (which is optional) degrades the report rather than failing it, and the
endpoint still answers 200 with an honest per-dependency status.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

from common.artifacts import artifacts_ready, manifest_schema_version
from common.config import settings
from common.database import check_mongo_health
from common.models import HealthStatus, ServiceHealth

logger = logging.getLogger(__name__)

#: Reported by ``/health`` so the dashboard can show which build it is talking to.
SERVICE_VERSION = "0.1.0"


async def _check_redis_container() -> HealthStatus:
    """Raw reachability ping for the Redis container (independent of streaming being enabled).

    System Health reports whether the *container* is up, not whether the app uses it, so this
    pings regardless of ``redis_enabled``.
    """
    try:
        import redis.asyncio as redis  # local import: redis is only needed here and by streaming

        client = redis.from_url(settings.redis_url)
        try:
            await asyncio.wait_for(client.ping(), timeout=2.0)
            return HealthStatus(status="ok", detail="reachable")
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001 - unreachable is a status, not a crash
        return HealthStatus(status="error", detail=f"unreachable: {type(exc).__name__}")


async def _check_http(url: str, name: str) -> HealthStatus:
    """Reachability check for an HTTP container (the scorer, the dashboard)."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
        ok = response.status_code < 500
        return HealthStatus(
            status="ok" if ok else "error",
            detail=f"HTTP {response.status_code}",
        )
    except Exception as exc:  # noqa: BLE001
        return HealthStatus(status="error", detail=f"unreachable: {type(exc).__name__}")


#: Only these are core to the read API answering correctly; the rest are shown for visibility
#: but do not, by themselves, mark the service degraded.
_CORE_DEPENDENCIES = ("mongodb", "artifacts")


def _aggregate_status(dependencies: Dict[str, HealthStatus]) -> str:
    """Roll the core dependency statuses into one service status."""
    core = [dependencies[name].status for name in _CORE_DEPENDENCIES if name in dependencies]
    if any(status == "error" for status in core):
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
    """Probe every container in the stack plus the artifacts manifest.

    Reports the docker-compose services (mongodb, redis, scorer, dashboard, read-api) by
    reachability, so System Health mirrors ``docker compose ps``.
    """
    mongodb, redis, scorer, dashboard = await asyncio.gather(
        check_mongo_health(),
        _check_redis_container(),
        _check_http(f"{settings.scorer_url}/health", "scorer"),
        _check_http(settings.ui_url, "dashboard"),
    )
    dependencies: Dict[str, HealthStatus] = {
        "mongodb": mongodb,
        "redis": redis,
        "scorer": scorer,
        "dashboard": dashboard,
        "read-api": HealthStatus(status="ok", detail="serving this response"),
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
