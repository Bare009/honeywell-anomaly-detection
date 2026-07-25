"""WebSocket for the dashboard's live view.

A minimal push channel: on connection it streams the dashboard summary periodically so the UI can
show activity without polling. Phase 8 builds on this; here it is deliberately simple and resilient
-- any backend hiccup closes the socket cleanly rather than crashing the API.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.services.store import get_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])

#: Seconds between live summary pushes.
PUSH_INTERVAL_SECONDS = 3.0


@router.websocket("/api/v1/ws")
async def live_updates(websocket: WebSocket) -> None:
    """Push the dashboard summary to a connected client until it disconnects."""
    await websocket.accept()
    try:
        while True:
            try:
                summary = await get_store().dashboard_summary()
            except Exception as exc:  # noqa: BLE001 - report, do not crash the socket
                summary = {"error": f"{type(exc).__name__}: {exc}"}
            await websocket.send_json(summary)
            await asyncio.sleep(PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")


__all__ = ["router"]
