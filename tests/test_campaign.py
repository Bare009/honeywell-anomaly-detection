"""Campaign reconstruction tests (Phase 7, D1).

The linker groups an entity's anomalies into time-windowed campaigns. Unit tests cover the linking
rules directly; a controlled multi-stage scenario shows that stages of the same campaign land in one
reconstructed campaign (the >= 90% acceptance, on well-formed input). Real-data reconstruction is
measured in the Phase 9 campaign experiment.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from common.models import AnomalyType, Detection, DetectionScores, EntityType
from serving.campaign import DEFAULT_WINDOW_SECONDS, CampaignLinker
from serving.store import InMemoryStore

BASE = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)


def _detection(
    entity_id: str,
    when: datetime,
    anomaly_type: AnomalyType = AnomalyType.BRUTE_FORCE,
    risk: float = 75.0,
    is_anomaly: bool = True,
) -> Detection:
    return Detection(
        entity_id=entity_id,
        entity_type=EntityType.USER,
        timestamp=when,
        risk_score=risk,
        is_anomaly=is_anomaly,
        anomaly_type=anomaly_type,
        scores=DetectionScores(),
    )


def _run(coro):
    return asyncio.run(coro)


class TestLinker:
    def test_benign_detection_joins_no_campaign(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store)
        result = _run(linker.link(_detection("u1", BASE, is_anomaly=False)))
        assert result is None

    def test_same_entity_within_window_shares_campaign(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store)
        first = _run(linker.link(_detection("u1", BASE, AnomalyType.BRUTE_FORCE)))
        second = _run(linker.link(_detection("u1", BASE + timedelta(hours=2), AnomalyType.LATERAL_MOVEMENT)))
        assert first == second

    def test_same_entity_outside_window_starts_new_campaign(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store, window_seconds=3600.0)
        first = _run(linker.link(_detection("u1", BASE)))
        second = _run(linker.link(_detection("u1", BASE + timedelta(hours=5))))
        assert first != second

    def test_different_entities_never_merge(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store)
        a = _run(linker.link(_detection("u1", BASE)))
        b = _run(linker.link(_detection("u2", BASE + timedelta(minutes=1))))
        assert a != b

    def test_kill_chain_accumulates_and_collapses_repeats(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store)
        cid = _run(linker.link(_detection("u1", BASE, AnomalyType.BRUTE_FORCE)))
        _run(linker.link(_detection("u1", BASE + timedelta(minutes=5), AnomalyType.BRUTE_FORCE)))
        _run(linker.link(_detection("u1", BASE + timedelta(minutes=10), AnomalyType.LATERAL_MOVEMENT)))
        campaign = _run(store.get_campaign(cid))
        assert campaign.kill_chain == ["brute_force", "lateral_movement"]
        assert campaign.stage_count == 3

    def test_max_risk_tracks_peak(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store)
        cid = _run(linker.link(_detection("u1", BASE, risk=60.0)))
        _run(linker.link(_detection("u1", BASE + timedelta(minutes=5), risk=92.0)))
        _run(linker.link(_detection("u1", BASE + timedelta(minutes=10), risk=70.0)))
        assert _run(store.get_campaign(cid)).max_risk == pytest.approx(92.0)


class TestReconstruction:
    def test_stages_of_a_campaign_reconstruct_together(self) -> None:
        """Interleave several multi-stage campaigns; each entity's stages must land in one campaign."""
        store = InMemoryStore()
        linker = CampaignLinker(store)

        stages = [AnomalyType.BRUTE_FORCE, AnomalyType.CREDENTIAL_MISUSE, AnomalyType.LATERAL_MOVEMENT, AnomalyType.LOW_AND_SLOW_EXFIL]
        # 5 entities, each a 4-stage campaign, stages 30 min apart; interleaved in time.
        events = []
        for step in range(4):
            for entity in range(5):
                events.append((f"ent_{entity}", BASE + timedelta(minutes=30 * step + entity), stages[step]))
        events.sort(key=lambda e: e[1])

        entity_to_campaign = {}
        correct = 0
        total = 0
        for entity_id, when, atype in events:
            cid = _run(linker.link(_detection(entity_id, when, atype)))
            total += 1
            if entity_id not in entity_to_campaign:
                entity_to_campaign[entity_id] = cid
            elif entity_to_campaign[entity_id] == cid:
                correct += 1
        # Every stage after the first should link to its entity's existing campaign.
        linked_fraction = correct / (total - len(entity_to_campaign))
        assert linked_fraction >= 0.9

    def test_each_entity_yields_one_campaign(self) -> None:
        store = InMemoryStore()
        linker = CampaignLinker(store)
        for step in range(3):
            for entity in range(4):
                _run(linker.link(_detection(f"e{entity}", BASE + timedelta(minutes=20 * step), AnomalyType.LATERAL_MOVEMENT)))
        campaigns = _run(store.list_campaigns(limit=100))
        assert len({c.entity_id for c in campaigns}) == 4
        assert len(campaigns) == 4
