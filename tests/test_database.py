"""Database contract tests.

No live MongoDB or Redis is needed here: these tests pin the collection names and the index
specification (section 12 of the plan) and prove the clients are genuinely lazy, so that
importing the module in a training script never opens a socket.
"""

from __future__ import annotations

import pytest

from common.database import (
    INDEX_SPEC,
    Collections,
    check_redis_health,
    get_collection,
    reset_clients,
)


class TestCollections:
    """Collection names are referenced across three services; typos must be caught."""

    def test_all_returns_every_collection(self) -> None:
        assert set(Collections.all()) == {
            "events",
            "entity_profiles",
            "detections",
            "campaigns",
            "feedback",
            "model_metrics",
            "drift_state",
        }

    def test_names_are_unique(self) -> None:
        names = Collections.all()
        assert len(names) == len(set(names))


class TestIndexSpec:
    """Index coverage drives dashboard query performance."""

    def test_every_collection_has_indexes(self) -> None:
        assert set(INDEX_SPEC) == set(Collections.all())

    def test_unique_identifier_indexes(self) -> None:
        """Each document type needs a unique id index to make writes idempotent."""
        expected = {
            Collections.DETECTIONS: "detection_id",
            Collections.ENTITY_PROFILES: "entity_id",
            Collections.CAMPAIGNS: "campaign_id",
            Collections.EVENTS: "event_id",
            Collections.FEEDBACK: "feedback_id",
            Collections.MODEL_METRICS: "run_id",
            Collections.DRIFT_STATE: "entity_id",
        }
        for collection, field in expected.items():
            unique_keys = [keys for keys, unique in INDEX_SPEC[collection] if unique]
            assert [(field, 1)] in unique_keys, f"{collection} missing unique {field}"

    def test_detections_indexed_for_ranked_queries(self) -> None:
        """The dashboard's primary query is 'top detections by risk'."""
        keys = [keys for keys, _ in INDEX_SPEC[Collections.DETECTIONS]]
        assert [("risk_score", -1)] in keys
        assert [("entity_id", 1), ("timestamp", -1)] in keys
        assert [("campaign_id", 1)] in keys

    def test_index_directions_are_valid(self) -> None:
        for specs in INDEX_SPEC.values():
            for keys, _ in specs:
                for _, direction in keys:
                    assert direction in (1, -1)


class TestGetCollection:
    """Unknown names fail immediately rather than creating a stray collection."""

    def test_unknown_collection_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown collection"):
            get_collection("detectionz")


class TestLaziness:
    """Importing the module must not connect to anything."""

    def test_reset_clients_is_safe_to_call(self) -> None:
        reset_clients()  # must not raise even with no clients created

    async def test_redis_health_reports_disabled_without_connecting(self) -> None:
        """Redis is optional, so a disabled config is 'disabled', not 'error'."""
        health = await check_redis_health()
        assert health.status == "disabled"
