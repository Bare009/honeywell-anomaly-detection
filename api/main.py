"""Read API application factory.

Serves the analyst dashboard. Read-only by design: there is no write path here, so the
scoring service owns all mutation and this service can be scaled or restarted freely.

Index creation at startup is **best-effort**. The dashboard must come up and render an
honest "MongoDB unreachable" state rather than crash-looping while Mongo boots.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import system
from api.services.health import SERVICE_VERSION, get_liveness
from common.config import settings
from common.database import close_all, ensure_indexes
from common.models import ServiceHealth
from common.seed import set_global_seed

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Seed determinism and bootstrap indexes on startup; close clients on shutdown."""
    set_global_seed()
    logger.info("Read API starting (seed=%s)", settings.random_seed)

    try:
        created = await ensure_indexes()
        logger.info("Ensured indexes on %d collections", len(created))
    except Exception as exc:  # noqa: BLE001 - startup must survive an absent database
        logger.warning(
            "Index bootstrap skipped (%s: %s). The API will still serve and report "
            "MongoDB as unhealthy.",
            type(exc).__name__,
            exc,
        )

    yield

    await close_all()
    logger.info("Read API stopped")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Behavioral Anomaly Detection - Read API",
        description=(
            "Read-only API powering the analyst dashboard: ranked detections, entity "
            "profiles, attack campaigns, model metrics and drift state."
        ),
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(system.router)

    @app.get("/api/v1/health", response_model=ServiceHealth, tags=["system"])
    async def health() -> ServiceHealth:
        """Liveness probe. Touches no dependency, so it answers instantly."""
        return await get_liveness()

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
