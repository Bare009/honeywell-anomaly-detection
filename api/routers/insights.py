"""Model metrics, drift state and the analyst feedback endpoint."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, status

from api.services.store import get_store
from common.artifacts import read_manifest
from common.config import settings
from common.database import Collections, get_collection
from common.models import AnalystVerdict
from serving.feedback import FeedbackProcessor

router = APIRouter(prefix="/api/v1", tags=["insights"])


@router.get("/metrics", summary="Latest evaluation metrics")
async def get_metrics() -> Dict[str, Any]:
    """Return the most recent ``model_metrics`` document, or the manifest's metrics as a fallback."""
    try:
        collection = get_collection(Collections.MODEL_METRICS)
        doc = await collection.find_one({}, sort=[("created_at", -1)])
        if doc:
            doc.pop("_id", None)
            return doc
    except Exception:  # noqa: BLE001 - fall back to the on-disk manifest if Mongo is unavailable
        pass
    manifest = read_manifest()
    return {"source": "manifest", "metrics": manifest.get("metrics"), "git_sha": manifest.get("git_sha")}


@router.get("/drift", summary="Per-entity drift state")
async def get_drift(limit: int = Query(100, ge=1, le=1000)) -> List[Dict[str, Any]]:
    collection = get_collection(Collections.DRIFT_STATE)
    cursor = collection.find({}).sort("updated_at", -1).limit(limit)
    results: List[Dict[str, Any]] = []
    async for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)
    return results


@router.post("/feedback", summary="Record an analyst verdict (D6)")
async def post_feedback(detection_id: str, verdict: AnalystVerdict, note: str | None = None) -> Dict[str, Any]:
    """Apply analyst feedback, adjusting the entity's alert threshold.

    This is the one mutating endpoint on the read API, kept here because the dashboard posts it; it
    delegates to the same :class:`FeedbackProcessor` the scoring service uses.
    """
    processor = FeedbackProcessor(get_store())
    try:
        record = await processor.apply(detection_id, verdict, note=note)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return record.model_dump(mode="json")


__all__ = ["router"]
