"""Score a dataset split through the full stack, into plain arrays for evaluation.

This is the offline counterpart to the serving pipeline: it runs the same featurize -> tiers ->
classifier + detectors -> risk path, but returns per-event arrays instead of persisting detections,
which is what the metric functions and experiments consume. It reuses the exact training-plane
helpers (``featurize_serving_split``, ``tier_scores``, ``classifier_matrix``) so an evaluation score
equals a served score for the same event.

Ground-truth labels and campaign ids are loaded here for scoring only -- never fed into the models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.models import ANOMALY_CLASSES, AnomalyType
from data_generator.generate import load_labels
from features.featurize import FeaturePipeline, FeatureVector
from models.baseline import BaselineModel
from models.classifier import ClassifierModel
from models.detectors import DetectorBank, DetectorResult, attack_probability, resolve_anomaly_type
from models.risk import RiskModel
from models.sequence import SequenceModel
from training.train_baseline import featurize_serving_split, load_split
from training.train_classifier import classifier_matrix, tier_scores

_NORMAL = AnomalyType.NORMAL.value


@dataclass
class ScoredSplit:
    """Per-event evaluation arrays for one split, aligned by index."""

    split: str
    vectors: List[FeatureVector]
    baseline: np.ndarray
    sequence: np.ndarray
    attack: np.ndarray
    risk: np.ndarray
    predicted_type: List[str]
    class_probs: List[Dict[str, float]]
    detector_results: List[List[DetectorResult]]
    labels: List[str]
    campaign_ids: List[Optional[str]]
    event_ids: List[str]
    entity_ids: List[str]
    latency_ms: List[float] = field(default_factory=list)

    @property
    def y_true(self) -> np.ndarray:
        """1 for a ground-truth anomaly, 0 for normal."""
        return np.array([0 if label == _NORMAL else 1 for label in self.labels], dtype=int)

    @property
    def cold_start(self) -> np.ndarray:
        return np.array([bool(v.cold_start) for v in self.vectors])


def load_label_frame(split: str, dataset_dir: Optional[Path] = None) -> Dict[str, Tuple[str, Optional[str]]]:
    """Map ``event_id -> (label, campaign_id)`` for one split (evaluation only)."""
    labels = load_labels(dataset_dir)
    subset = labels[labels["split"] == split]
    mapping: Dict[str, Tuple[str, Optional[str]]] = {}
    for event_id, label, campaign_id in zip(
        subset["event_id"], subset["label"], subset["campaign_id"]
    ):
        cid = None if campaign_id is None or (isinstance(campaign_id, float) and np.isnan(campaign_id)) else str(campaign_id)
        mapping[str(event_id)] = (str(label), cid)
    return mapping


def score_split(
    split: str,
    pipeline: FeaturePipeline,
    baseline: BaselineModel,
    sequence: SequenceModel,
    classifier: ClassifierModel,
    risk: RiskModel,
    dataset_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    measure_latency: bool = False,
) -> ScoredSplit:
    """Score a split end-to-end and return aligned per-event arrays."""
    events = load_split(split, dataset_dir)
    if limit is not None:
        events = events[:limit]

    vectors = featurize_serving_split(pipeline, events)
    scores = tier_scores(vectors, baseline, sequence)
    baseline_scores = scores["baseline"]
    sequence_scores = scores["sequence"]

    matrix = classifier_matrix(vectors, baseline_scores, sequence_scores)
    base_types, class_probs = classifier.classify_matrix(matrix)
    attack = np.array([attack_probability(p) for p in class_probs])

    bank = DetectorBank()
    detector_results = [bank.evaluate(v.raw) for v in vectors]

    fused = risk.fuse(baseline_scores, sequence_scores, attack)
    risk_scores = risk.to_risk(fused)

    predicted: List[str] = []
    for base_type, probs, results in zip(base_types, class_probs, detector_results):
        resolved, _, _ = resolve_anomaly_type(AnomalyType(base_type), probs, results)
        predicted.append(resolved.value)

    label_frame = load_label_frame(split, dataset_dir)
    labels = [label_frame.get(v.event_id, (_NORMAL, None))[0] for v in vectors]
    campaign_ids = [label_frame.get(v.event_id, (_NORMAL, None))[1] for v in vectors]

    latency_ms: List[float] = []
    if measure_latency:
        # Per-event compute latency, measured on a sample to keep the pass quick.
        sample = vectors[: min(2000, len(vectors))]
        for vector in sample:
            started = time.perf_counter()
            b = float(baseline.score_baseline(vector))
            s = float(sequence.score_sequence(vector))
            row = np.concatenate([vector.values, [b, s]])
            classifier.classify(row)
            latency_ms.append((time.perf_counter() - started) * 1000.0)

    return ScoredSplit(
        split=split,
        vectors=vectors,
        baseline=baseline_scores,
        sequence=sequence_scores,
        attack=attack,
        risk=risk_scores,
        predicted_type=predicted,
        class_probs=class_probs,
        detector_results=detector_results,
        labels=labels,
        campaign_ids=campaign_ids,
        event_ids=[v.event_id for v in vectors],
        entity_ids=[v.entity_id for v in vectors],
        latency_ms=latency_ms,
    )


def load_models(artifacts_dir: Optional[Path] = None):
    """Load the full stack (pipeline + four models) for evaluation."""
    from common.config import settings

    target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    pipeline = FeaturePipeline.load(target)
    baseline = BaselineModel.load(target / "baseline_model.json")
    sequence = SequenceModel.load(target / "sequence_model.json")
    classifier = ClassifierModel.load(target / "classifier.json")
    risk = RiskModel.load(target / "risk_model.json")
    return pipeline, baseline, sequence, classifier, risk


__all__ = ["ScoredSplit", "load_label_frame", "score_split", "load_models"]
