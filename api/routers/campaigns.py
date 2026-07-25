"""Reconstructed attack campaigns / kill chains (read-only, D1)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from api.services.store import get_store

router = APIRouter(prefix="/api/v1/campaigns", tags=["campaigns"])


@router.get("", summary="List reconstructed campaigns")
async def list_campaigns(
    entity_id: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    campaigns = await get_store().list_campaigns(entity_id=entity_id, skip=skip, limit=limit)
    return [c.model_dump(mode="json") for c in campaigns]


@router.get("/{campaign_id}", summary="One campaign with its stage timeline")
async def get_campaign(campaign_id: str) -> Dict[str, Any]:
    campaign = await get_store().get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found")
    return campaign.model_dump(mode="json")


__all__ = ["router"]
