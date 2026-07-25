"""Counterfactual "nearest-normal" explanations (D2).

For an alert, the most useful question an analyst can ask is: *what would have made this benign?*
This search answers it by finding the smallest set of feature changes that drop the risk below the
alert threshold -- "this would have scored benign if the geo-velocity and the failed-auth burst had
been at their usual levels".

It works because the classifier's numeric features are **entity-relative and standardized**: a value
of zero is, by construction, "typical for this entity/population". So moving a risk-driving feature
to zero is exactly "make this behaviour ordinary", and re-scoring through the classifier and risk
model shows whether that would have cleared the alert. The search is greedy over the features SHAP
flags as most risk-increasing, keeps only changes that actually help, and stops as soon as the
verdict flips -- yielding a minimal, plain-language explanation with no extra artifact to persist.

Only the standardized numeric features are perturbed. Categorical identities and the two upstream
tier scores are left alone: "log in from a different country" is a change an analyst can reason
about, "have a lower autoencoder reconstruction error" is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.models import ANOMALY_CLASS_INDEX, AnomalyType, Counterfactual, CounterfactualChange
from models.classifier import ClassifierModel
from models.risk import RiskModel

_NORMAL_INDEX = ANOMALY_CLASS_INDEX[AnomalyType.NORMAL.value]


class CounterfactualSearch:
    """Greedy minimal-change search that flips an alert to benign."""

    def __init__(
        self,
        classifier: ClassifierModel,
        risk_model: RiskModel,
        mutable_features: Sequence[str],
        max_changes: int = 4,
        candidate_k: int = 8,
    ) -> None:
        self.classifier = classifier
        self.risk_model = risk_model
        self.max_changes = max_changes
        self.candidate_k = candidate_k
        mutable = set(mutable_features)
        # Column positions of the standardized numeric features we are allowed to move to typical.
        self.mutable_indices: List[int] = [
            index
            for index, name in enumerate(classifier.feature_names)
            if name in mutable
        ]

    def _risk_of(self, row: np.ndarray, baseline_score: float, sequence_score: float) -> float:
        proba = self.classifier.predict_proba(row[None, :])[0]
        attack = float(min(1.0, max(0.0, 1.0 - proba[_NORMAL_INDEX])))
        fused = self.risk_model.fuse(
            np.array([baseline_score]), np.array([sequence_score]), np.array([attack])
        )
        return float(self.risk_model.to_risk(fused)[0])

    def search(
        self,
        row: np.ndarray,
        baseline_score: float,
        sequence_score: float,
        shap_pushes: Optional[Dict[str, float]] = None,
        raw: Optional[Dict[str, float]] = None,
    ) -> Counterfactual:
        """Find the minimal feature resets that bring risk below the alert threshold."""
        row = np.asarray(row, dtype=float).copy()
        threshold = self.risk_model.alert_threshold
        original_risk = self._risk_of(row, baseline_score, sequence_score)

        if original_risk < threshold:
            return Counterfactual(
                changes=[],
                original_risk=original_risk,
                resulting_risk=original_risk,
                found=True,
                summary="Already below the alert threshold; no change needed.",
            )

        # Rank candidate features: by SHAP risk-push if available, else by how far the standardized
        # value sits from typical (larger magnitude = more unusual).
        def rank_key(index: int) -> float:
            name = self.classifier.feature_names[index]
            if shap_pushes and name in shap_pushes:
                return shap_pushes[name]
            return abs(float(row[index]))

        candidates = sorted(self.mutable_indices, key=rank_key, reverse=True)[: self.candidate_k]

        changes: List[CounterfactualChange] = []
        current_risk = original_risk
        for index in candidates:
            if len(changes) >= self.max_changes:
                break
            old = float(row[index])
            if abs(old) < 1e-9:
                continue  # already typical, nothing to change
            row[index] = 0.0  # standardized mean == "typical"
            new_risk = self._risk_of(row, baseline_score, sequence_score)
            if new_risk < current_risk - 1e-6:
                name = self.classifier.feature_names[index]
                actual = raw.get(name) if (raw and name in raw) else round(old, 4)
                changes.append(
                    CounterfactualChange(
                        feature=name,
                        actual=actual,
                        suggested="typical",
                        description=f"return {name} to its typical level",
                    )
                )
                current_risk = new_risk
                if current_risk < threshold:
                    break
            else:
                row[index] = old  # revert a change that did not help

        found = current_risk < threshold
        summary = self._summarize(changes, original_risk, current_risk, found)
        return Counterfactual(
            changes=changes,
            original_risk=round(original_risk, 4),
            resulting_risk=round(current_risk, 4),
            found=found,
            summary=summary,
        )

    @staticmethod
    def _summarize(
        changes: Sequence[CounterfactualChange],
        original_risk: float,
        resulting_risk: float,
        found: bool,
    ) -> str:
        if not changes:
            return "No small set of behavioural changes would clear this alert."
        names = ", ".join(change.feature for change in changes)
        if found:
            return (
                f"Would have scored benign ({resulting_risk:.0f} vs {original_risk:.0f}) "
                f"with typical {names}."
            )
        return (
            f"Adjusting {names} lowered risk to {resulting_risk:.0f} (from {original_risk:.0f}) "
            "but did not clear the alert."
        )


__all__ = ["CounterfactualSearch"]
