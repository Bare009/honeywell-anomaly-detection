"""Analyst feedback loop tests (Phase 7, D6).

The processor moves a per-entity offset the right way, bounds it, persists the record and refuses
unknown detections. An integration test shows the offset actually changes the alert decision in the
scoring pipeline -- feedback shifts the threshold, not the reported risk.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from common.config import settings
from common.models import AnalystVerdict, Detection, DetectionScores, EntityType
from serving.feedback import FEEDBACK_RISK_DELTA, OFFSET_BOUND, FeedbackProcessor, apply_offset
from serving.store import InMemoryStore


def _run(coro):
    return asyncio.run(coro)


def _detection(entity_id: str = "u1", risk: float = 70.0) -> Detection:
    return Detection(
        entity_id=entity_id,
        entity_type=EntityType.USER,
        timestamp=datetime(2026, 3, 2, 9, tzinfo=timezone.utc),
        risk_score=risk,
        is_anomaly=True,
        scores=DetectionScores(),
    )


class TestApplyOffset:
    def test_clamped_to_range(self) -> None:
        assert apply_offset(95.0, 20.0) == 100.0
        assert apply_offset(5.0, -20.0) == 0.0

    def test_offset_shifts_risk(self) -> None:
        assert apply_offset(50.0, -10.0) == pytest.approx(40.0)


class TestFeedbackProcessor:
    def test_false_positive_lowers_offset(self) -> None:
        store = InMemoryStore()
        det = _detection()
        _run(store.save_detection(det))
        processor = FeedbackProcessor(store)
        feedback = _run(processor.apply(det.detection_id, AnalystVerdict.FALSE_POSITIVE))
        assert feedback.applied.new_value == pytest.approx(-FEEDBACK_RISK_DELTA)
        assert _run(store.get_entity_offset("u1")) == pytest.approx(-FEEDBACK_RISK_DELTA)

    def test_confirmed_raises_offset(self) -> None:
        store = InMemoryStore()
        det = _detection()
        _run(store.save_detection(det))
        processor = FeedbackProcessor(store)
        feedback = _run(processor.apply(det.detection_id, AnalystVerdict.CONFIRMED))
        assert feedback.applied.new_value == pytest.approx(FEEDBACK_RISK_DELTA)

    def test_offset_is_bounded(self) -> None:
        store = InMemoryStore()
        det = _detection()
        _run(store.save_detection(det))
        processor = FeedbackProcessor(store)
        for _ in range(20):  # far more than enough to hit the bound
            _run(processor.apply(det.detection_id, AnalystVerdict.FALSE_POSITIVE))
        assert _run(store.get_entity_offset("u1")) == pytest.approx(-OFFSET_BOUND)

    def test_feedback_is_persisted(self) -> None:
        store = InMemoryStore()
        det = _detection()
        _run(store.save_detection(det))
        processor = FeedbackProcessor(store)
        _run(processor.apply(det.detection_id, AnalystVerdict.CONFIRMED, note="looks real"))
        records = _run(store.list_feedback())
        assert len(records) == 1
        assert records[0].note == "looks real"

    def test_unknown_detection_raises(self) -> None:
        processor = FeedbackProcessor(InMemoryStore())
        with pytest.raises(KeyError):
            _run(processor.apply("det_missing", AnalystVerdict.CONFIRMED))


_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_ALL = all(
    (_ARTIFACTS / name).exists()
    for name in ("encoders.json", "baseline_model.json", "sequence_model.json", "classifier.json", "risk_model.json")
)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_ALL, reason="run the full training pipeline first")
class TestFeedbackChangesDecision:
    """Feedback visibly moves the alert decision for an entity in the real pipeline."""

    def test_negative_offset_suppresses_an_alert(self) -> None:
        from serving.pipeline import ScoringPipeline
        from serving.store import InMemoryStore as Store
        from training.train_baseline import load_split

        store = Store()
        pipeline = ScoringPipeline.load(store=store, enable_explanations=False)

        # Find an event that scores as an alert on first pass.
        events = load_split("val")[:4000]
        alert_event = None
        for event in events:
            detection = _run(pipeline.score_event(event))
            if detection.is_anomaly:
                alert_event = event
                alert_detection = detection
                break
        assert alert_event is not None, "expected at least one alert in the slice"

        # Pile on false-positive feedback for that entity, then re-score the same event.
        for _ in range(10):
            _run(store.set_entity_offset(alert_event.entity_id, -OFFSET_BOUND))
        rescored = _run(pipeline.score_event(alert_event))

        # The model risk is unchanged, but the offset lowers the effective decision.
        assert rescored.risk_score == pytest.approx(alert_detection.risk_score, abs=1e-6)
        if alert_detection.risk_score < 100.0 and not rescored.detector_hits:
            assert rescored.is_anomaly is False
