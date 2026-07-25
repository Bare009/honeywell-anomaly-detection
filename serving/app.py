"""Scoring service application.

Phase 0 scope: a health endpoint plus the artifact-manifest check that every later phase
depends on. The scoring endpoints (``/score``, ``/score/batch``) and the full pipeline
arrive in Phase 7 -- this file is intentionally the place they will attach.

Two rules this service holds to from the start:

* **It never trains.** Trained state is loaded read-only from ``artifacts/``.
* **It fails loudly on a schema mismatch, but only where it is safe to.** Startup logs a
  clear warning if the manifest version does not match the code; the scoring endpoints
  (Phase 7) will refuse to serve rather than score with a mismatched model.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from common.artifacts import (
    artifacts_ready,
    check_schema_version,
    manifest_schema_version,
    read_manifest,
)
from common.config import settings
from common.database import check_mongo_health, check_redis_health, close_all
from common.models import HealthStatus, ServiceHealth
from common.seed import set_global_seed

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "0.1.0"

#: Populated once at startup and read by request handlers. Loading artifacts is expensive,
#: so it happens exactly once per process rather than per request.
STATE: Dict[str, Any] = {
    "artifacts_loaded": False,
    "schema_ok": False,
    "manifest": None,
}


async def verify_scoring_token(
    authorization: str | None = Header(default=None),
) -> None:
    """Require a bearer token on write endpoints.

    The scoring service accepts data, so it is the only service with a write path and the
    only one that needs authentication. Disable only for local, non-networked runs via
    ``ADP_SCORING_AUTH_ENABLED=false``.
    """
    if not settings.scoring_auth_enabled:
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1].strip()
    if token != settings.scoring_auth_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid scoring token",
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Seed determinism, verify the artifact contract, and cache the manifest."""
    set_global_seed()
    logger.info("Scoring service starting (seed=%s)", settings.random_seed)

    manifest = read_manifest()
    STATE["manifest"] = manifest
    STATE["artifacts_loaded"] = artifacts_ready()

    # Non-strict at startup so the service can boot before training has ever run; the
    # Phase 7 scoring endpoints enforce this hard before scoring anything.
    STATE["schema_ok"] = check_schema_version(strict=False)

    if not STATE["artifacts_loaded"]:
        logger.warning(
            "No trained artifacts found in %s. Scoring will be unavailable until "
            "training/build_artifacts.py has run.",
            settings.artifacts_dir,
        )
    elif not STATE["schema_ok"]:
        logger.error(
            "Artifact schema mismatch (manifest=%s, expected=%s). Refusing to score "
            "until artifacts are rebuilt.",
            manifest.get("schema_version"),
            settings.artifact_schema_version,
        )
    else:
        logger.info(
            "Artifacts ready (schema %s, git %s)",
            manifest.get("schema_version"),
            manifest.get("git_sha"),
        )

    yield

    await close_all()
    logger.info("Scoring service stopped")


def create_app() -> FastAPI:
    """Build and configure the scoring FastAPI application."""
    app = FastAPI(
        title="Behavioral Anomaly Detection - Scoring Service",
        description=(
            "Stateless per-event scorer. Loads trained artifacts once at startup and "
            "never retrains. Write endpoints require a bearer token."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=ServiceHealth, tags=["system"])
    async def health() -> ServiceHealth:
        """Report readiness of the artifact contract and backing stores.

        Answers 200 even when artifacts are missing -- the payload says so explicitly,
        which is more useful to an operator than a failed request.
        """
        dependencies: Dict[str, HealthStatus] = {
            "mongodb": await check_mongo_health(),
            "redis": await check_redis_health(),
        }

        loaded = bool(STATE["artifacts_loaded"])
        schema_ok = bool(STATE["schema_ok"])

        if not loaded:
            artifacts_status, detail = "degraded", "no trained artifacts yet"
        elif not schema_ok:
            artifacts_status, detail = "error", "artifact schema version mismatch"
        else:
            artifacts_status, detail = "ok", "artifacts loaded"
        dependencies["artifacts"] = HealthStatus(status=artifacts_status, detail=detail)

        overall = "ok"
        if artifacts_status == "error" or any(
            dep.status == "error" for dep in dependencies.values()
        ):
            overall = "degraded"

        return ServiceHealth(
            service="serving",
            status=overall,
            version=SERVICE_VERSION,
            artifact_schema_version=manifest_schema_version(),
            artifacts_ready=loaded and schema_ok,
            dependencies=dependencies,
        )

    @app.get("/manifest", tags=["system"])
    async def manifest() -> Dict[str, Any]:
        """Return the artifact manifest this process is serving from.

        Lets an operator confirm which model build is live without shell access.
        """
        return STATE["manifest"] or read_manifest()

    @app.get("/ready", tags=["system"], dependencies=[Depends(verify_scoring_token)])
    async def ready() -> Dict[str, Any]:
        """Authenticated readiness gate for the scoring path.

        Returns 503 when the service could not score an event, so a load balancer or the
        replay script can wait instead of pushing traffic at an unusable scorer.
        """
        if not STATE["artifacts_loaded"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained artifacts loaded. Run training/build_artifacts.py.",
            )
        if not STATE["schema_ok"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Artifact schema mismatch: manifest reports "
                    f"{manifest_schema_version()}, code expects "
                    f"{settings.artifact_schema_version}."
                ),
            )
        return {"ready": True, "schema_version": settings.artifact_schema_version}

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(
        "serving.app:app",
        host=settings.serving_host,
        port=settings.serving_port,
        reload=False,
    )
