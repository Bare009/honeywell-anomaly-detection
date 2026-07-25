"""The online scoring pipeline: one event in, one explained detection out.

This is where all six phases meet. For each event it runs the same steps, in the same order, the
training plane used -- so an online score equals the offline score for the same event:

1. featurize (the one shared ``featurize``), building live per-entity state;
2. the two unsupervised tiers (autoencoder baseline, GRU sequence);
3. the classifier (type + calibrated probabilities) and the deterministic detectors;
4. risk fusion into a calibrated 0-100 score, uncertainty band and alert-budget flag;
5. the analyst feedback offset applied to the *decision* (not the reported risk);
6. an explanation -- but only for alerts, since SHAP and the counterfactual search are the
   expensive part and the vast majority of events are benign (this is the latency gate);
7. campaign linking and drift tracking;
8. persistence through the store.

The pipeline holds per-entity state in-process (rolling windows, live profiles, drift monitors), so
a long-lived scorer keeps learning as it runs, exactly as the batch replay does.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.config import settings
from common.models import (
    AnomalyType,
    Detection,
    DetectionScores,
    Event,
)
from features.featurize import FeaturePipeline, FeatureVector
from models.baseline import BaselineModel
from models.classifier import ClassifierModel
from models.detectors import DetectorBank, attack_probability, resolve_anomaly_type
from models.drift import DriftMonitor
from models.risk import RiskAssessment, RiskModel
from models.sequence import SequenceModel
from serving.campaign import CampaignLinker
from serving.feedback import FeedbackProcessor, apply_offset
from serving.store import DetectionStore, InMemoryStore

logger = logging.getLogger(__name__)

_NORMAL = AnomalyType.NORMAL


class DriftManager:
    """Per-entity drift tracking on a scalar behavioural signal, bootstrapped lazily.

    A monitor needs a baseline sample before it can judge drift, so the first
    ``bootstrap_samples`` values for an entity are collected, then a monitor is built and every
    later value is folded in.
    """

    def __init__(self, bootstrap_samples: int = 50) -> None:
        self.bootstrap_samples = bootstrap_samples
        self.monitors: Dict[str, DriftMonitor] = {}
        self._buffers: Dict[str, List[float]] = defaultdict(list)

    def update(self, entity_id: str, value: float) -> Tuple[bool, float]:
        """Return ``(drift_flag, psi)`` after folding one value in."""
        monitor = self.monitors.get(entity_id)
        if monitor is not None:
            reading = monitor.update(value)
            from common.models import DriftStatus

            return reading.status == DriftStatus.DRIFTING, reading.psi

        buffer = self._buffers[entity_id]
        buffer.append(float(value))
        if len(buffer) >= self.bootstrap_samples:
            self.monitors[entity_id] = DriftMonitor.from_baseline(entity_id, buffer)
            self._buffers.pop(entity_id, None)
        return False, 0.0


class ScoringPipeline:
    """Loads the trained stack once and scores events through it."""

    def __init__(
        self,
        features: FeaturePipeline,
        baseline: BaselineModel,
        sequence: SequenceModel,
        classifier: ClassifierModel,
        risk: RiskModel,
        store: DetectionStore,
        explainer: Optional[Any] = None,
        campaign_linker: Optional[CampaignLinker] = None,
        feedback: Optional[FeedbackProcessor] = None,
        drift_manager: Optional[DriftManager] = None,
        enable_explanations: bool = True,
    ) -> None:
        self.features = features
        self.baseline = baseline
        self.sequence = sequence
        self.classifier = classifier
        self.risk = risk
        self.store = store
        self.explainer = explainer
        self.campaign_linker = campaign_linker or CampaignLinker(store)
        self.feedback = feedback or FeedbackProcessor(store)
        self.drift_manager = drift_manager or DriftManager()
        self.enable_explanations = enable_explanations

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        artifacts_dir: Optional[Path] = None,
        store: Optional[DetectionStore] = None,
        enable_explanations: bool = True,
    ) -> "ScoringPipeline":
        """Load every trained artifact and assemble a ready-to-score pipeline."""
        target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
        features = FeaturePipeline.load(target)
        baseline = BaselineModel.load(target / "baseline_model.json")
        sequence = SequenceModel.load(target / "sequence_model.json")
        classifier = ClassifierModel.load(target / "classifier.json")
        risk = RiskModel.load(target / "risk_model.json")
        store = store or InMemoryStore()

        explainer = None
        if enable_explanations:
            from explainability.explainer import DetectionExplainer

            explainer = DetectionExplainer(
                classifier, risk, sequence, mutable_features=features.numeric_names
            )

        return cls(
            features=features,
            baseline=baseline,
            sequence=sequence,
            classifier=classifier,
            risk=risk,
            store=store,
            explainer=explainer,
            enable_explanations=enable_explanations,
        )

    # ------------------------------------------------------------------ #
    # Synchronous compute (no I/O)
    # ------------------------------------------------------------------ #

    def _compute(self, event: Event) -> Dict[str, Any]:
        """Run the model stack for one event. Updates in-process feature state."""
        vector = self.features.featurize(event, update_state=True, use_live_profile=True)
        baseline_score = float(self.baseline.score_baseline(vector))
        sequence_score = float(self.sequence.score_sequence(vector))

        row = np.concatenate([vector.values, [baseline_score, sequence_score]])
        classifier_type, probs = self.classifier.classify(row)
        attack = attack_probability(probs)

        detector_results = DetectorBank().evaluate(vector.raw)
        assessment = self.risk.assess(
            baseline_score,
            sequence_score,
            attack,
            cold_start=vector.cold_start,
            detector_results=detector_results,
        )
        resolved_type, adjusted_probs, hits = resolve_anomaly_type(
            classifier_type, probs, detector_results
        )
        return {
            "vector": vector,
            "row": row,
            "baseline_score": baseline_score,
            "sequence_score": sequence_score,
            "attack": attack,
            "assessment": assessment,
            "resolved_type": resolved_type,
            "probs": adjusted_probs,
            "hits": hits,
        }

    @staticmethod
    def _display_type(resolved_type: AnomalyType, probs: Dict[str, float], is_anomaly: bool) -> AnomalyType:
        """Reconcile the reported type with the alert decision.

        A benign detection reports ``normal``. An alert that the classifier nonetheless called
        ``normal`` (risk driven by the unsupervised tiers) is labelled with its most likely attack
        class, so the type and the verdict never contradict each other.
        """
        if not is_anomaly:
            return _NORMAL
        if resolved_type != _NORMAL:
            return resolved_type
        attack_probs = {k: v for k, v in probs.items() if k != _NORMAL.value}
        if attack_probs:
            return AnomalyType(max(attack_probs, key=lambda k: attack_probs[k]))
        return resolved_type

    # ------------------------------------------------------------------ #
    # Async scoring (persistence + side effects)
    # ------------------------------------------------------------------ #

    async def score_event(self, event: Event, persist: bool = True) -> Detection:
        """Score one event into a persisted, explained :class:`Detection`."""
        computed = self._compute(event)
        vector: FeatureVector = computed["vector"]
        assessment: RiskAssessment = computed["assessment"]

        # Feedback offset shifts the *decision*, never the reported model risk.
        offset = await self.store.get_entity_offset(event.entity_id)
        effective_risk = apply_offset(assessment.risk_score, offset)
        override = bool(assessment.detector_hits)
        in_budget = (effective_risk >= self.risk.budget_threshold) or override
        is_anomaly = (effective_risk >= self.risk.alert_threshold) or override

        display_type = self._display_type(computed["resolved_type"], computed["probs"], is_anomaly)

        drift_flag, _psi = self.drift_manager.update(event.entity_id, computed["baseline_score"])

        detection = Detection(
            entity_id=event.entity_id,
            entity_type=event.entity_type,
            timestamp=event.timestamp,
            event_ref=event.event_id,
            session_id=event.session_id,
            scores=DetectionScores(
                baseline=computed["baseline_score"],
                sequence=computed["sequence_score"],
                classifier_confidence=computed["attack"],
                fused_raw=assessment.fused_raw,
            ),
            risk_score=assessment.risk_score,
            risk_uncertainty=assessment.risk_uncertainty,
            in_alert_budget=in_budget,
            is_anomaly=is_anomaly,
            anomaly_type=display_type,
            anomaly_type_probs=computed["probs"],
            detector_hits=assessment.detector_hits,
            cold_start=bool(vector.cold_start),
            drift_flag=drift_flag,
            ground_truth_label=event.label,
        )

        # Explanations are the expensive step, so compute them only for alerts (the latency gate).
        if is_anomaly and self.enable_explanations and self.explainer is not None:
            detection.explanation = self.explainer.explain(
                entity_id=event.entity_id,
                vector=vector,
                classifier_row=computed["row"],
                predicted_type=display_type,
                risk=assessment,
                detector_hits=computed["hits"],
            )

        if is_anomaly:
            detection.campaign_id = await self.campaign_linker.link(detection)

        if persist:
            await self.store.save_detection(detection)
        return detection

    async def score_batch(self, events: List[Event], persist: bool = True) -> List[Detection]:
        """Score a batch in arrival order -- identical to scoring each event on its own."""
        return [await self.score_event(event, persist=persist) for event in events]

    def reset_state(self) -> None:
        """Forget all per-entity state (between independent replays)."""
        self.features.reset_state()
        self.drift_manager = DriftManager()


__all__ = ["ScoringPipeline", "DriftManager"]
