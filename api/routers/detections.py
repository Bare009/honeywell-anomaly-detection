"""Ranked detections, entity history and the dashboard summary (read-only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from api.services.store import get_store

router = APIRouter(prefix="/api/v1", tags=["detections"])


@router.get("/detections", summary="Ranked detections with filters")
async def list_detections(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    sort: str = Query("risk", pattern="^(risk|time)$"),
    anomaly_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    cold_start: Optional[bool] = None,
    min_risk: Optional[float] = Query(None, ge=0.0, le=100.0),
) -> List[Dict[str, Any]]:
    """List detections, sorted by risk (default) or time, with optional filters."""
    store = get_store()
    detections = await store.list_detections(
        skip=skip,
        limit=limit,
        anomaly_type=anomaly_type,
        entity_type=entity_type,
        cold_start=cold_start,
        min_risk=min_risk,
        sort=sort,
    )
    return [d.model_dump(mode="json") for d in detections]


@router.get("/detections/{detection_id}", summary="One detection with full explanation")
async def get_detection(detection_id: str) -> Dict[str, Any]:
    detection = await get_store().get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")
    return detection.model_dump(mode="json")


@router.get("/entities/{entity_id}", summary="Entity detection history")
async def get_entity(entity_id: str, limit: int = Query(100, ge=1, le=500)) -> Dict[str, Any]:
    detections = await get_store().entity_detections(entity_id, limit=limit)
    return {
        "entity_id": entity_id,
        "n_detections": len(detections),
        "detections": [d.model_dump(mode="json") for d in detections],
    }


@router.get("/dashboard/summary", summary="Overview counts for the dashboard")
async def dashboard_summary() -> Dict[str, Any]:
    return await get_store().dashboard_summary()


__all__ = ["router"]
