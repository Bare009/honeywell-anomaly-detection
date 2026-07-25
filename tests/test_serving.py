"""Serving pipeline and scoring API tests (Phase 7).

These exercise the full online path against the real trained artifacts, using an in-memory store so
no database is required. They check the acceptance criteria: events score into well-formed
detections, known attacks are caught, batch scoring equals event-by-event scoring, the HTTP endpoint
authenticates and rejects malformed input, and median latency stays within budget.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import pytest

from common.config import settings
from common.models import ANOMALY_CLASSES, Detection, Event

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_ALL = all(
    (_ARTIFACTS / name).exists()
    for name in ("encoders.json", "baseline_model.json", "sequence_model.json", "classifier.json", "risk_model.json")
)

pytestmark = pytest.mark.skipif(not _HAS_ALL, reason="run the full training pipeline first")

REPLAY_LIMIT = 3000


@pytest.fixture(scope="module")
def replayed():
    """Replay a validation slice once through a real pipeline; reused across tests."""
    from serving.replay import replay
    from serving.store import InMemoryStore
    from training.train_sequence import load_label_map

    store = InMemoryStore()
    summary = replay(split="val", limit=REPLAY_LIMIT, store=store, enable_explanations=True)
    labels = load_label_map("val")
    return summary, labels


class TestReplayScoring:
    def test_every_event_scored(self, replayed) -> None:
        summary, _ = replayed
        assert len(summary["detections"]) == summary["n_events"] == REPLAY_LIMIT

    def test_detections_are_well_formed(self, replayed) -> None:
        summary, _ = replayed
        for det in summary["detections"][:500]:
            assert 0.0 <= det.risk_score <= 100.0
            assert 0.0 <= det.scores.baseline <= 1.0
            assert 0.0 <= det.scores.sequence <= 1.0
            assert det.anomaly_type.value in ANOMALY_CLASSES

    def test_some_anomalies_are_flagged(self, replayed) -> None:
        summary, _ = replayed
        assert summary["n_anomalies"] > 0
        assert summary["n_in_budget"] > 0

    def test_alerts_carry_an_explanation(self, replayed) -> None:
        summary, _ = replayed
        alerts = [d for d in summary["detections"] if d.is_anomaly]
        assert alerts, "expected at least one alert"
        assert any(d.explanation.top_features for d in alerts)

    def test_brute_force_is_detected(self, replayed) -> None:
        """Brute-force events (deterministic signature) should be caught and typed correctly."""
        summary, labels = replayed
        bf_flagged = [
            d
            for d in summary["detections"]
            if labels.get(d.event_ref) == "brute_force" and d.is_anomaly
        ]
        assert bf_flagged, "no brute_force events were flagged"
        assert any(d.anomaly_type.value == "brute_force" for d in bf_flagged)

    def test_median_latency_within_budget(self, replayed) -> None:
        summary, _ = replayed
        assert summary["latency_median_ms"] < 50.0, summary["latency_median_ms"]


class TestBatchParity:
    def test_batch_equals_event_by_event(self) -> None:
        """Scoring a batch equals scoring each event on its own through an identical pipeline."""
        from serving.pipeline import ScoringPipeline
        from serving.store import InMemoryStore
        from training.train_baseline import load_split

        events: List[Event] = load_split("val")[:200]
        pipeline_a = ScoringPipeline.load(store=InMemoryStore(), enable_explanations=False)
        pipeline_b = ScoringPipeline.load(store=InMemoryStore(), enable_explanations=False)

        batch = asyncio.run(pipeline_a.score_batch(events, persist=False))

        async def _loop() -> List[Detection]:
            return [await pipeline_b.score_event(event, persist=False) for event in events]

        individual = asyncio.run(_loop())

        for left, right in zip(batch, individual):
            assert left.risk_score == pytest.approx(right.risk_score, abs=1e-9)
            assert left.anomaly_type == right.anomaly_type
            assert left.is_anomaly == right.is_anomaly


class TestScoringEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from serving.app import create_app
        from serving.pipeline import ScoringPipeline
        from serving.store import InMemoryStore

        pipeline = ScoringPipeline.load(store=InMemoryStore(), enable_explanations=False)
        app = create_app(pipeline=pipeline)
        with TestClient(app) as client:
            yield client

    @staticmethod
    def _payload() -> dict:
        from tests.test_features import make_event

        return make_event(entity_id="user_endpoint").model_dump(mode="json")

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {settings.scoring_auth_token}"}

    def test_score_returns_detection(self, client) -> None:
        response = client.post("/score", json=self._payload(), headers=self._auth())
        assert response.status_code == 200
        body = response.json()
        assert "risk_score" in body and "anomaly_type" in body

    def test_missing_token_is_rejected(self, client) -> None:
        response = client.post("/score", json=self._payload())
        assert response.status_code == 401

    def test_malformed_input_is_4xx(self, client) -> None:
        response = client.post("/score", json={"entity_id": "x"}, headers=self._auth())
        assert response.status_code == 422

    def test_batch_endpoint(self, client) -> None:
        payloads = [self._payload(), self._payload()]
        response = client.post("/score/batch", json=payloads, headers=self._auth())
        assert response.status_code == 200
        assert len(response.json()) == 2
