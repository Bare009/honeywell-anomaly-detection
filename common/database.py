"""MongoDB and Redis access for the serving and read planes.

Both clients are **lazy singletons**: importing this module never opens a socket, so unit
tests and offline training scripts can import it freely without a database running. The
connection is created on first use and reused for the process lifetime.

Everything here is defensive by design. Redis is optional (streaming is a bonus path), and
``ensure_indexes`` is idempotent so it can run on every service start.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from common.config import settings
from common.models import HealthStatus

logger = logging.getLogger(__name__)

# Module-level singletons (created on first use).
_mongo_client: Optional[Any] = None
_redis_client: Optional[Any] = None


class Collections:
    """Canonical MongoDB collection names.

    Referencing these constants instead of raw strings keeps the schema in one place.
    """

    EVENTS = "events"
    ENTITY_PROFILES = "entity_profiles"
    DETECTIONS = "detections"
    CAMPAIGNS = "campaigns"
    FEEDBACK = "feedback"
    MODEL_METRICS = "model_metrics"
    DRIFT_STATE = "drift_state"

    @classmethod
    def all(cls) -> List[str]:
        """Every collection name managed by this system."""
        return [
            cls.EVENTS,
            cls.ENTITY_PROFILES,
            cls.DETECTIONS,
            cls.CAMPAIGNS,
            cls.FEEDBACK,
            cls.MODEL_METRICS,
            cls.DRIFT_STATE,
        ]


#: Index specification per collection: ``(keys, unique)``.
#: Mirrors section 12 of the implementation plan.
INDEX_SPEC: Dict[str, List[Tuple[List[Tuple[str, int]], bool]]] = {
    Collections.DETECTIONS: [
        ([("detection_id", 1)], True),
        ([("risk_score", -1)], False),
        ([("entity_id", 1), ("timestamp", -1)], False),
        ([("anomaly_type", 1)], False),
        ([("campaign_id", 1)], False),
        ([("created_at", -1)], False),
    ],
    Collections.ENTITY_PROFILES: [
        ([("entity_id", 1)], True),
        ([("cohort", 1)], False),
    ],
    Collections.CAMPAIGNS: [
        ([("campaign_id", 1)], True),
        ([("entity_id", 1), ("last_activity", -1)], False),
    ],
    Collections.EVENTS: [
        ([("event_id", 1)], True),
        ([("entity_id", 1), ("timestamp", -1)], False),
    ],
    Collections.FEEDBACK: [
        ([("feedback_id", 1)], True),
        ([("detection_id", 1)], False),
        ([("created_at", -1)], False),
    ],
    Collections.MODEL_METRICS: [
        ([("run_id", 1)], True),
        ([("created_at", -1)], False),
    ],
    Collections.DRIFT_STATE: [
        ([("entity_id", 1)], True),
        ([("updated_at", -1)], False),
    ],
}


# --------------------------------------------------------------------------- #
# MongoDB
# --------------------------------------------------------------------------- #


def get_mongo_client() -> Any:
    """Return the process-wide Motor client, creating it on first call."""
    global _mongo_client
    if _mongo_client is None:
        from motor.motor_asyncio import AsyncIOMotorClient

        _mongo_client = AsyncIOMotorClient(
            settings.mongo_url,
            serverSelectionTimeoutMS=settings.mongo_timeout_ms,
            connectTimeoutMS=settings.mongo_timeout_ms,
            uuidRepresentation="standard",
        )
        logger.info("Created MongoDB client for %s", settings.mongo_url)
    return _mongo_client


def get_database() -> Any:
    """Return the configured application database handle."""
    return get_mongo_client()[settings.mongo_db_name]


def get_collection(name: str) -> Any:
    """Return one collection handle by name.

    Raises
    ------
    ValueError
        If ``name`` is not a collection this system manages -- catches typos early.
    """
    if name not in Collections.all():
        raise ValueError(f"Unknown collection '{name}'. Known: {Collections.all()}")
    return get_database()[name]


async def ensure_indexes() -> Dict[str, List[str]]:
    """Create every index from :data:`INDEX_SPEC`. Idempotent.

    Returns
    -------
    dict
        ``collection -> [created index names]``. Safe to call on every startup;
        MongoDB treats an existing identical index as a no-op.
    """
    database = get_database()
    created: Dict[str, List[str]] = {}

    for collection_name, specs in INDEX_SPEC.items():
        collection = database[collection_name]
        names: List[str] = []
        for keys, unique in specs:
            index_name = await collection.create_index(keys, unique=unique)
            names.append(index_name)
        created[collection_name] = names

    logger.info("Ensured indexes on %d collections", len(created))
    return created


async def check_mongo_health() -> HealthStatus:
    """Ping MongoDB and report status without raising."""
    started = time.perf_counter()
    try:
        await get_mongo_client().admin.command("ping")
    except Exception as exc:  # noqa: BLE001 - health checks must never raise
        return HealthStatus(
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return HealthStatus(
        status="ok",
        detail=f"connected to {settings.mongo_db_name}",
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


async def close_mongo_client() -> None:
    """Close and drop the Mongo singleton (used on service shutdown)."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        logger.info("Closed MongoDB client")


# --------------------------------------------------------------------------- #
# Redis (optional)
# --------------------------------------------------------------------------- #


def get_redis_client() -> Any:
    """Return the process-wide async Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as aioredis

        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Created Redis client for %s", settings.redis_url)
    return _redis_client


async def check_redis_health() -> HealthStatus:
    """Ping Redis and report status without raising.

    Redis is optional: when ``redis_enabled`` is False this reports ``disabled`` rather
    than an error, so the demo path stays green with no Redis running.
    """
    if not settings.redis_enabled:
        return HealthStatus(status="disabled", detail="streaming disabled by config")

    started = time.perf_counter()
    try:
        await get_redis_client().ping()
    except Exception as exc:  # noqa: BLE001 - health checks must never raise
        return HealthStatus(
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return HealthStatus(
        status="ok",
        detail="connected",
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


async def close_redis_client() -> None:
    """Close and drop the Redis singleton (used on service shutdown)."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.close()
        except Exception:  # noqa: BLE001 - shutdown must never raise
            logger.warning("Redis client close raised; ignoring", exc_info=True)
        _redis_client = None
        logger.info("Closed Redis client")


async def close_all() -> None:
    """Close every database connection (convenience for service shutdown)."""
    await close_mongo_client()
    await close_redis_client()


def reset_clients() -> None:
    """Drop client singletons without awaiting -- for test isolation only."""
    global _mongo_client, _redis_client
    _mongo_client = None
    _redis_client = None


__all__ = [
    "Collections",
    "INDEX_SPEC",
    "get_mongo_client",
    "get_database",
    "get_collection",
    "ensure_indexes",
    "check_mongo_health",
    "close_mongo_client",
    "get_redis_client",
    "check_redis_health",
    "close_redis_client",
    "close_all",
    "reset_clients",
]
