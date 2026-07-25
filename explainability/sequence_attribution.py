"""Surface the sequence model's per-step attribution for the explanation layer.

The GRU already produces per-step surprise as a by-product of scoring (the same forward pass yields
both). This module is a thin adapter: it asks the model for the per-step weights and, optionally,
trims them to the few steps that carry most of the surprise -- the span an analyst should look at --
and phrases them as a short sentence.
"""

from __future__ import annotations

from typing import List, Optional

from common.models import SequenceStepAttribution
from features.featurize import FeatureVector
from models.sequence import SequenceModel


def sequence_attribution(
    model: SequenceModel, vector: FeatureVector, top_k: Optional[int] = None
) -> List[SequenceStepAttribution]:
    """Per-step attribution for one event, optionally trimmed to the ``top_k`` most surprising."""
    steps = model.attribute_sequence(vector)
    if top_k is None or top_k >= len(steps):
        return steps
    ranked = sorted(steps, key=lambda step: step.score, reverse=True)[:top_k]
    # Keep chronological order for display once the top-k are selected.
    ranked.sort(key=lambda step: step.position)
    return ranked


def summarize(steps: List[SequenceStepAttribution], max_tokens: int = 3) -> str:
    """A short plain-text description of where the sequence surprise concentrated."""
    if not steps:
        return "No command-sequence signal for this event."
    ranked = sorted(steps, key=lambda step: step.score, reverse=True)[:max_tokens]
    parts = [f"'{step.token}'" for step in ranked]
    if len(parts) == 1:
        return f"The command {parts[0]} was the most surprising step in the sequence."
    joined = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return f"The commands {joined} were the most surprising steps in the sequence."


__all__ = ["sequence_attribution", "summarize"]
