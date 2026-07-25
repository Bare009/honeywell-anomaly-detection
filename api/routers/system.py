"""System and configuration endpoints for the read API."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from api.services.health import get_readiness
from common.config import settings
from common.models import ANOMALY_CLASSES, ServiceHealth

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/health", response_model=ServiceHealth, summary="Dependency health")
async def system_health() -> ServiceHealth:
    """Report MongoDB, Redis and artifact readiness.

    Always returns 200 with a per-dependency breakdown -- an outage is data, not an HTTP
    error, so the dashboard can render exactly what is wrong.
    """
    return await get_readiness()


@router.get("/config", summary="Non-secret runtime configuration")
async def system_config() -> Dict[str, Any]:
    """Expose the operational knobs the dashboard needs to render correctly.

    Only non-secret values are included. Tokens, API keys and connection strings are
    deliberately omitted.
    """
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "seed": settings.random_seed,
        "artifact_schema_version": settings.artifact_schema_version,
        "alert_budget_pct": settings.alert_budget_pct,
        "risk_alert_threshold": settings.risk_alert_threshold,
        "anomaly_gate_threshold": settings.anomaly_gate_threshold,
        "fusion_weights": settings.fusion_weights,
        "entity_history_min_sessions": settings.entity_history_min_sessions,
        "drift": {
            "window_size": settings.drift_window_size,
            "psi_threshold": settings.drift_psi_threshold,
            "refresh_alpha": settings.drift_refresh_alpha,
        },
        "anomaly_classes": ANOMALY_CLASSES,
        "streaming_enabled": settings.redis_enabled,
        "llm_narrative_enabled": settings.llm_enabled,
        "mitre_map_source": settings.mitre_map_source,
    }


__all__ = ["router"]
