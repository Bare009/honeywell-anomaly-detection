"""Read-only FastAPI service that powers the analyst dashboard.

Kept separate from the scoring service on purpose: dashboard queries must never compete
with scoring for latency, and this service has no write path at all. Routers for
detections, entities, campaigns, metrics, drift, feedback and the live WebSocket arrive in
Phase 7; Phase 0 ships health only.
"""
