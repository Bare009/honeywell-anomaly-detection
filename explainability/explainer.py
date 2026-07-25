"""Assemble a complete :class:`~common.models.Explanation` for one detection.

This is the single place that gathers every strand of the explainability layer -- SHAP feature
attributions, the counterfactual "nearest-normal" edit, the sequence model's per-step surprise, the
MITRE mapping, a structured baseline comparison, and the plain-language narrative -- into the object
the dashboard renders and the report cites.

It is constructed once from the trained models and reused per event. The serving pipeline (Phase 7)
calls :meth:`explain`; the phase's tests call it directly. Nothing here can change a score: it reads
the models' outputs and describes them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from common.models import (
    ANOMALY_CLASS_INDEX,
    AnomalyType,
    BaselineComparison,
    Explanation,
    FeatureAttribution,
)
from explainability.counterfactual import CounterfactualSearch
from explainability.mitre_map import map_anomaly
from explainability.narrative import NarrativeGenerator
from explainability.sequence_attribution import sequence_attribution
from explainability.shap_explainer import ShapExplainer
from features.featurize import FeatureVector
from models.classifier import ClassifierModel
from models.risk import RiskAssessment, RiskModel
from models.sequence import SequenceModel


class DetectionExplainer:
    """Builds full explanations from the trained classifier, risk model and sequence model."""

    def __init__(
        self,
        classifier: ClassifierModel,
        risk_model: RiskModel,
        sequence_model: SequenceModel,
        mutable_features: Sequence[str],
        use_llm: Optional[bool] = None,
    ) -> None:
        self.shap = ShapExplainer(classifier)
        self.counterfactual = CounterfactualSearch(classifier, risk_model, mutable_features)
        self.sequence_model = sequence_model
        self.narrator = NarrativeGenerator(use_llm)

    def explain(
        self,
        entity_id: str,
        vector: FeatureVector,
        classifier_row,
        predicted_type: AnomalyType,
        risk: RiskAssessment,
        detector_hits: Optional[Sequence[str]] = None,
        baseline_values: Optional[Dict[str, float]] = None,
        top_k: int = 6,
    ) -> Explanation:
        """Assemble the explanation for one scored event."""
        class_index = ANOMALY_CLASS_INDEX[predicted_type.value]
        detector_hits = list(detector_hits or [])

        top_features = self.shap.local(
            classifier_row,
            class_index=class_index,
            top_k=top_k,
            raw=vector.raw,
            baseline_values=baseline_values,
        )
        shap_pushes = {attr.feature: float(attr.contribution) for attr in top_features}

        # The counterfactual needs the two tier scores that fed fusion; they are the trailing two
        # columns of the classifier row (baseline_score, sequence_score).
        baseline_score = float(classifier_row[-2])
        sequence_score = float(classifier_row[-1])
        counterfactual = self.counterfactual.search(
            classifier_row,
            baseline_score=baseline_score,
            sequence_score=sequence_score,
            shap_pushes=shap_pushes,
            raw=vector.raw,
        )

        sequence_steps = sequence_attribution(self.sequence_model, vector, top_k=top_k)
        mitre = map_anomaly(predicted_type)
        baseline_comparison = self._baseline_comparison(top_features, baseline_values)

        narrative, source = self.narrator.generate(
            entity_id=entity_id,
            anomaly_type=predicted_type,
            risk_score=risk.risk_score,
            top_features=top_features,
            mitre=mitre,
            cold_start=bool(getattr(vector, "cold_start", False)),
            detector_hits=detector_hits,
        )

        return Explanation(
            top_features=top_features,
            counterfactual=counterfactual,
            sequence_attribution=sequence_steps,
            mitre=mitre,
            baseline_comparison=baseline_comparison,
            narrative=narrative,
            narrative_source=source,
        )

    @staticmethod
    def _baseline_comparison(
        top_features: Sequence[FeatureAttribution],
        baseline_values: Optional[Dict[str, float]],
    ) -> BaselineComparison:
        """A structured diff of the top features against the entity's typical values."""
        fields: Dict[str, Dict[str, object]] = {}
        for attr in top_features:
            typical = baseline_values.get(attr.feature) if baseline_values else attr.baseline_value
            fields[attr.feature] = {
                "observed": attr.value,
                "typical": typical,
                "deviates": attr.direction == "increases_risk",
            }
        return BaselineComparison(fields=fields, summary=None)


__all__ = ["DetectionExplainer"]
