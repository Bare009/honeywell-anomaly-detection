"""Service health endpoint tests.

Both services must start and answer without a database, because the dashboard has to render
an honest "MongoDB unreachable" state rather than fail to load. These tests use FastAPI's
TestClient, so no external process is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from common.config import settings


@pytest.fixture(scope="module")
def api_client():
    """Read API test client (runs lifespan, tolerating an absent MongoDB)."""
    from api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def serving_client():
    """Scoring service test client."""
    from serving.app import app

    with TestClient(app) as client:
        yield client


class TestReadApiHealth:
    """`GET /api/v1/health` is the liveness probe the dashboard polls."""

    def test_returns_200(self, api_client: TestClient) -> None:
        assert api_client.get("/api/v1/health").status_code == 200

    def test_reports_service_identity(self, api_client: TestClient) -> None:
        payload = api_client.get("/api/v1/health").json()
        assert payload["service"] == "api"
        assert payload["status"] == "ok"

    def test_liveness_touches_no_dependency(self, api_client: TestClient) -> None:
        """Liveness must stay fast and stay 200 while MongoDB is still booting."""
        assert api_client.get("/api/v1/health").json()["dependencies"] == {}

    def test_reports_artifact_state(self, api_client: TestClient) -> None:
        payload = api_client.get("/api/v1/health").json()
        assert payload["artifact_schema_version"] == settings.artifact_schema_version
        assert isinstance(payload["artifacts_ready"], bool)

    def test_checked_at_is_iso_string(self, api_client: TestClient) -> None:
        assert isinstance(api_client.get("/api/v1/health").json()["checked_at"], str)


class TestReadApiSystemRouter:
    """`/api/v1/system/*` exposes dependency health and non-secret config."""

    def test_system_health_returns_200_without_mongo(
        self, api_client: TestClient
    ) -> None:
        """A dependency outage is data, not an HTTP error."""
        response = api_client.get("/api/v1/system/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}

    def test_system_health_lists_dependencies(self, api_client: TestClient) -> None:
        dependencies = api_client.get("/api/v1/system/health").json()["dependencies"]
        assert {"mongodb", "redis", "artifacts"} <= set(dependencies)

    def test_redis_reported_disabled_when_off(self, api_client: TestClient) -> None:
        dependencies = api_client.get("/api/v1/system/health").json()["dependencies"]
        assert dependencies["redis"]["status"] == "disabled"

    def test_config_endpoint_exposes_operational_knobs(
        self, api_client: TestClient
    ) -> None:
        payload = api_client.get("/api/v1/system/config").json()
        assert payload["seed"] == settings.random_seed
        assert payload["alert_budget_pct"] == pytest.approx(settings.alert_budget_pct)
        assert len(payload["anomaly_classes"]) == 9
        assert set(payload["fusion_weights"]) == {"baseline", "sequence", "classifier"}

    def test_config_endpoint_leaks_no_secrets(self, api_client: TestClient) -> None:
        """The dashboard needs configuration, not credentials."""
        body = api_client.get("/api/v1/system/config").text.lower()
        assert settings.scoring_auth_token.lower() not in body
        for forbidden in ("token", "api_key", "password", "mongo_url"):
            assert forbidden not in body


class TestServingHealth:
    """The scoring service reports whether it could actually score an event."""

    def test_returns_200(self, serving_client: TestClient) -> None:
        assert serving_client.get("/health").status_code == 200

    def test_reports_service_identity(self, serving_client: TestClient) -> None:
        payload = serving_client.get("/health").json()
        assert payload["service"] == "serving"

    def test_artifacts_not_ready_before_training(
        self, serving_client: TestClient
    ) -> None:
        """With no trained models the service must say so rather than imply readiness."""
        payload = serving_client.get("/health").json()
        assert payload["artifacts_ready"] is False
        assert payload["dependencies"]["artifacts"]["status"] == "degraded"

    def test_manifest_endpoint_returns_contract(self, serving_client: TestClient) -> None:
        payload = serving_client.get("/manifest").json()
        assert payload["schema_version"] == settings.artifact_schema_version
        assert "artifacts" in payload


class TestServingAuth:
    """The only service with a write path must be authenticated."""

    def test_missing_token_is_401(self, serving_client: TestClient) -> None:
        assert serving_client.get("/ready").status_code == 401

    def test_wrong_scheme_is_401(self, serving_client: TestClient) -> None:
        response = serving_client.get(
            "/ready", headers={"Authorization": f"Basic {settings.scoring_auth_token}"}
        )
        assert response.status_code == 401

    def test_wrong_token_is_403(self, serving_client: TestClient) -> None:
        response = serving_client.get(
            "/ready", headers={"Authorization": "Bearer not-the-token"}
        )
        assert response.status_code == 403

    def test_valid_token_reaches_readiness_gate(self, serving_client: TestClient) -> None:
        """Authenticated but untrained: 503, not 401/403 -- auth passed, artifacts did not."""
        response = serving_client.get(
            "/ready",
            headers={"Authorization": f"Bearer {settings.scoring_auth_token}"},
        )
        assert response.status_code == 503
        assert "artifacts" in response.json()["detail"].lower()


class TestErrorHandling:
    """Bad requests must be 4xx, never 500."""

    def test_unknown_route_is_404(self, api_client: TestClient) -> None:
        assert api_client.get("/api/v1/nope").status_code == 404

    def test_wrong_method_is_405(self, api_client: TestClient) -> None:
        assert api_client.post("/api/v1/health").status_code == 405
