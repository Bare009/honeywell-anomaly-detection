"""Explainability tests (Phase 6, Deliverable #5).

Unit tests cover the pieces that need no dataset: the static MITRE map, the deterministic narrative
and its graceful fallback, sequence-attribution summarising, and SHAP local/global attribution on a
small trained classifier. The integration test assembles a full explanation for a real, high-risk
validation event and asserts the acceptance criteria: non-empty top features, a counterfactual, a
MITRE mapping and a narrative.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from common.config import settings
from common.models import (
    ANOMALY_CLASS_INDEX,
    ANOMALY_CLASSES,
    AnomalyType,
    FeatureAttribution,
    SequenceStepAttribution,
)
from explainability.mitre_map import map_anomaly, map_static
from explainability.narrative import NarrativeGenerator, template_narrative
from explainability.sequence_attribution import summarize
from explainability.shap_explainer import ShapExplainer
from models.classifier import ClassifierModel, ClassifierTrainConfig


# --------------------------------------------------------------------------- #
# MITRE mapping
# --------------------------------------------------------------------------- #


class TestMitreMap:
    def test_every_attack_class_maps_to_a_technique(self) -> None:
        for cls in ANOMALY_CLASSES:
            anomaly = AnomalyType(cls)
            techniques = map_anomaly(anomaly)
            if anomaly == AnomalyType.NORMAL:
                assert techniques == []
            else:
                assert techniques, f"{cls} has no MITRE mapping"
                assert all(t.technique_id.startswith("T") for t in techniques)
                assert all(t.url and "attack.mitre.org" in t.url for t in techniques)

    def test_brute_force_maps_to_t1110(self) -> None:
        assert map_anomaly(AnomalyType.BRUTE_FORCE)[0].technique_id == "T1110"

    def test_returns_independent_copies(self) -> None:
        first = map_static(AnomalyType.LATERAL_MOVEMENT)
        first[0].confidence = 0.1
        second = map_static(AnomalyType.LATERAL_MOVEMENT)
        assert second[0].confidence == 1.0


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #


class TestNarrative:
    def _features(self) -> List[FeatureAttribution]:
        return [
            FeatureAttribution(feature="window_auth_failures", value=8.0, contribution=1.2, direction="increases_risk"),
            FeatureAttribution(feature="hour_likelihood", value=0.01, contribution=0.5, direction="increases_risk"),
        ]

    def test_template_mentions_type_and_risk(self) -> None:
        text = template_narrative("user_1", AnomalyType.BRUTE_FORCE, 82.0, self._features())
        assert "brute force" in text
        assert "82" in text

    def test_template_flags_cold_start(self) -> None:
        text = template_narrative("dev_9", AnomalyType.CREDENTIAL_MISUSE, 70.0, self._features(), cold_start=True)
        assert "little history" in text

    def test_normal_reads_as_normal(self) -> None:
        text = template_narrative("user_1", AnomalyType.NORMAL, 5.0, [])
        assert "normal" in text.lower()

    def test_generator_falls_back_to_template_without_key(self) -> None:
        """With the LLM disabled (default) the source is always the deterministic template."""
        generator = NarrativeGenerator(use_llm=False)
        text, source = generator.generate("user_1", AnomalyType.LATERAL_MOVEMENT, 75.0, self._features())
        assert source == "template"
        assert text


# --------------------------------------------------------------------------- #
# Sequence attribution summary
# --------------------------------------------------------------------------- #


class TestSequenceSummary:
    def test_summarize_highlights_top_token(self) -> None:
        steps = [
            SequenceStepAttribution(position=1, token="login", score=0.1),
            SequenceStepAttribution(position=2, token="escalate", score=0.7),
            SequenceStepAttribution(position=3, token="exfiltrate", score=0.2),
        ]
        text = summarize(steps, max_tokens=1)
        assert "escalate" in text

    def test_summarize_empty(self) -> None:
        assert "No command-sequence signal" in summarize([])


# --------------------------------------------------------------------------- #
# SHAP on a small trained classifier
# --------------------------------------------------------------------------- #

_SHAP_FEATURES = ["driver", "n1", "n2", "n3"]


def _tiny_classifier() -> ClassifierModel:
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for _ in range(300):
        rows.append([rng.normal(0, 0.3), rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)])
        labels.append("normal")
    for _ in range(300):
        rows.append([rng.normal(5, 0.5), rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)])
        labels.append("brute_force")
    return ClassifierModel.train(
        np.array(rows), labels, _SHAP_FEATURES, categorical_indices=[],
        config=ClassifierTrainConfig(num_boost_round=60, min_data_in_leaf=10),
    )


@pytest.fixture(scope="module")
def shap_explainer() -> ShapExplainer:
    return ShapExplainer(_tiny_classifier())


class TestShapExplainer:
    def test_local_returns_ranked_attributions(self, shap_explainer: ShapExplainer) -> None:
        row = np.array([5.0, 0.0, 0.0, 0.0])
        attributions = shap_explainer.local(row, top_k=3)
        assert 0 < len(attributions) <= 3
        assert all(np.isfinite(a.contribution) for a in attributions)
        assert all(a.direction in {"increases_risk", "decreases_risk", "neutral"} for a in attributions)

    def test_driver_feature_dominates_for_attack(self, shap_explainer: ShapExplainer) -> None:
        attack_index = ANOMALY_CLASS_INDEX["brute_force"]
        attributions = shap_explainer.local(np.array([5.0, 0.0, 0.0, 0.0]), class_index=attack_index, top_k=4)
        top = max(attributions, key=lambda a: abs(a.contribution))
        assert top.feature == "driver"
        assert top.direction == "increases_risk"

    def test_uses_raw_values_for_display(self, shap_explainer: ShapExplainer) -> None:
        attributions = shap_explainer.local(
            np.array([5.0, 0.0, 0.0, 0.0]), top_k=1, raw={"driver": "8 failures"}
        )
        assert attributions[0].value == "8 failures"

    def test_global_importance_ranks_driver_first(self, shap_explainer: ShapExplainer) -> None:
        rng = np.random.default_rng(1)
        sample = np.vstack([
            np.column_stack([rng.normal(0, 0.3, 50), rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng.normal(0, 1, 50)]),
            np.column_stack([rng.normal(5, 0.5, 50), rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng.normal(0, 1, 50)]),
        ])
        importance = shap_explainer.global_importance(sample, top_k=4)
        assert importance[0][0] == "driver"


# --------------------------------------------------------------------------- #
# Full explanation on a real high-risk validation event
# --------------------------------------------------------------------------- #

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_ALL = all(
    (_ARTIFACTS / name).exists()
    for name in ("encoders.json", "baseline_model.json", "sequence_model.json", "classifier.json", "risk_model.json")
)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_ALL, reason="run the full training pipeline first")
class TestBuiltExplainer:
    """Assemble a complete explanation for a real, high-risk validation event."""

    def test_explanation_has_every_component(self) -> None:
        from explainability.explainer import DetectionExplainer
        from features.featurize import FeaturePipeline
        from models.baseline import BaselineModel
        from models.classifier import ClassifierModel as CM
        from models.detectors import DetectorBank, attack_probability
        from models.risk import RiskModel
        from models.sequence import SequenceModel
        from training.train_baseline import featurize_serving_split, load_split
        from training.train_classifier import classifier_matrix, tier_scores

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        baseline = BaselineModel.load(_ARTIFACTS / "baseline_model.json")
        sequence = SequenceModel.load(_ARTIFACTS / "sequence_model.json")
        classifier = CM.load(_ARTIFACTS / "classifier.json")
        risk = RiskModel.load(_ARTIFACTS / "risk_model.json")

        # Score a slice of validation and pick the highest-risk event to explain.
        events = load_split("val")[:4000]
        vectors = featurize_serving_split(pipeline, events)
        scores = tier_scores(vectors, baseline, sequence)
        matrix = classifier_matrix(vectors, scores["baseline"], scores["sequence"])
        _, probs = classifier.classify_matrix(matrix)
        attack = np.array([attack_probability(p) for p in probs])
        risk_scores = risk.to_risk(risk.fuse(scores["baseline"], scores["sequence"], attack))
        idx = int(np.argmax(risk_scores))

        vector = vectors[idx]
        row = matrix[idx]
        predicted_type, _ = classifier.classify(row)
        bank = DetectorBank()
        detector_results = bank.evaluate(vector.raw)
        assessment = risk.assess(
            float(scores["baseline"][idx]),
            float(scores["sequence"][idx]),
            float(attack[idx]),
            cold_start=vector.cold_start,
            detector_results=detector_results,
        )

        explainer = DetectionExplainer(classifier, risk, sequence, mutable_features=pipeline.numeric_names)
        explanation = explainer.explain(
            entity_id=vector.entity_id,
            vector=vector,
            classifier_row=row,
            predicted_type=predicted_type,
            risk=assessment,
            detector_hits=assessment.detector_hits,
        )

        assert explanation.top_features, "every detection must have top features"
        assert explanation.counterfactual is not None
        assert explanation.narrative
        if predicted_type != AnomalyType.NORMAL:
            assert explanation.mitre, "an attack detection must map to MITRE"
        # The highest-risk event should be an alert, and its explanation should name risk drivers.
        assert any(a.direction == "increases_risk" for a in explanation.top_features)
