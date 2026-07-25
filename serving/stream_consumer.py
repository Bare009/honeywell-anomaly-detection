"""Optional Redis Streams consumer for real-time scoring.

Demonstrates the "scalable / streaming" criterion: instead of (or alongside) the HTTP endpoint,
events can arrive on a Redis stream and be scored as they land, with detections published to a
downstream stream. It is entirely optional -- ``redis_enabled`` is False by default, and nothing in
the demo path depends on it. The offline HTTP scorer is the safety net.

Run it (with Redis up and ``ADP_REDIS_ENABLED=true``)::

    python -m serving.stream_consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

from common.config import settings
from common.database import get_redis_client
from common.models import Event
from serving.pipeline import ScoringPipeline
from serving.store import MongoStore

logger = logging.getLogger(__name__)


async def _ensure_group(redis: object, stream: str, group: str) -> None:
    """Create the consumer group if it does not exist (idempotent)."""
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - BUSYGROUP means it already exists
        if "BUSYGROUP" not in str(exc):
            raise


async def consume(
    pipeline: ScoringPipeline,
    block_ms: int = 5000,
    count: int = 32,
    max_batches: Optional[int] = None,
) -> int:
    """Consume events from the configured Redis stream and score them.

    Returns the number of events scored. ``max_batches`` bounds the loop (used by tests/demos);
    left as ``None`` it runs until cancelled.
    """
    redis = get_redis_client()
    stream = settings.redis_stream_events
    detections_stream = settings.redis_stream_detections
    group = settings.redis_consumer_group
    consumer = settings.redis_consumer_name

    await _ensure_group(redis, stream, group)
    scored = 0
    batches = 0

    while max_batches is None or batches < max_batches:
        batches += 1
        response = await redis.xreadgroup(  # type: ignore[attr-defined]
            group, consumer, {stream: ">"}, count=count, block=block_ms
        )
        if not response:
            continue
        for _stream_name, messages in response:
            for message_id, fields in messages:
                try:
                    event = Event.model_validate_json(fields["event"])
                    detection = await pipeline.score_event(event)
                    await redis.xadd(  # type: ignore[attr-defined]
                        detections_stream,
                        {"detection": json.dumps(detection.model_dump(mode="json"))},
                    )
                    scored += 1
                except Exception as exc:  # noqa: BLE001 - a bad message must not kill the loop
                    logger.warning("Skipping bad stream message %s: %s", message_id, exc)
                finally:
                    await redis.xack(stream, group, message_id)  # type: ignore[attr-defined]
    return scored


def main(argv: Optional[list] = None) -> int:  # pragma: no cover - manual entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    if not settings.redis_enabled:
        logger.error("Streaming is disabled. Set ADP_REDIS_ENABLED=true to run the consumer.")
        return 1
    pipeline = ScoringPipeline.load(store=MongoStore())
    logger.info("Stream consumer started on '%s'", settings.redis_stream_events)
    try:
        asyncio.run(consume(pipeline))
    except KeyboardInterrupt:
        logger.info("Stream consumer stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
