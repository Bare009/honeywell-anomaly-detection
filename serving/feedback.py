"""Analyst feedback / active-learning loop (D6).

An analyst who marks an alert a false positive -- or confirms a real one -- should change what the
system does next for that entity. This applies that verdict as a per-entity risk offset: a false
positive nudges the entity's effective risk down (fewer alerts for behaviour the analyst has judged
benign), a confirmation nudges it up (more sensitive). The offset is bounded so no amount of
feedback can pin an entity permanently on or off, and every verdict is persisted with the exact
adjustment it caused, so the loop is auditable.

The scoring pipeline reads the offset and applies it to the alert decision only -- the model's
calibrated ``risk_score`` is reported unchanged, so feedback shifts the threshold, not the science.
"""

from __future__ import annotations

import logging
from typing import Optional

from common.models import (
    AnalystVerdict,
    Feedback,
    FeedbackAdjustment,
)
from serving.store import DetectionStore

logger = logging.getLogger(__name__)

#: Risk points a single verdict moves the entity's effective threshold.
FEEDBACK_RISK_DELTA = 10.0

#: Cap on the accumulated offset, so repeated feedback cannot silence or pin an entity forever.
OFFSET_BOUND = 40.0


class FeedbackProcessor:
    """Applies analyst verdicts to per-entity risk offsets and records them."""

    def __init__(
        self,
        store: DetectionStore,
        delta: float = FEEDBACK_RISK_DELTA,
        bound: float = OFFSET_BOUND,
    ) -> None:
        self.store = store
        self.delta = delta
        self.bound = bound

    async def apply(
        self, detection_id: str, verdict: AnalystVerdict, note: Optional[str] = None
    ) -> Feedback:
        """Apply a verdict to the detection's entity and persist the feedback record.

        Raises
        ------
        KeyError
            If the detection is unknown -- feedback must reference a real detection.
        """
        detection = await self.store.get_detection(detection_id)
        if detection is None:
            raise KeyError(f"Unknown detection '{detection_id}'")

        entity_id = detection.entity_id
        previous = await self.store.get_entity_offset(entity_id)

        step = -self.delta if verdict == AnalystVerdict.FALSE_POSITIVE else self.delta
        new_offset = float(max(-self.bound, min(self.bound, previous + step)))
        await self.store.set_entity_offset(entity_id, new_offset)

        adjustment = FeedbackAdjustment(
            scope="entity",
            scope_id=entity_id,
            adjustment=step,
            previous_value=previous,
            new_value=new_offset,
        )
        feedback = Feedback(
            detection_id=detection_id,
            entity_id=entity_id,
            analyst_verdict=verdict,
            note=note,
            applied=adjustment,
        )
        await self.store.save_feedback(feedback)
        logger.info(
            "Feedback %s on %s: entity %s offset %.1f -> %.1f",
            verdict.value,
            detection_id,
            entity_id,
            previous,
            new_offset,
        )
        return feedback


def apply_offset(risk_score: float, offset: float) -> float:
    """Effective risk after an entity's feedback offset, clamped to ``[0, 100]``.

    A false-positive-heavy entity has a negative offset (its effective risk drops); a confirmed one
    has a positive offset. Used by the pipeline for the alert decision only.
    """
    return float(max(0.0, min(100.0, risk_score + offset)))


__all__ = ["FEEDBACK_RISK_DELTA", "OFFSET_BOUND", "FeedbackProcessor", "apply_offset"]
