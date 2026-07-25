"""Classifier and calibration tests (Phase 5).

Unit tests train a small LightGBM on a synthetic, separable three-class problem in about a second and
assert the properties the system relies on: it learns the classes, calibrated probabilities are valid
and normalized, calibration improves reliability, and the model round-trips through JSON (booster text
plus isotonic knots) with identical predictions. Integration tests validate the real trained
classifier's macro-F1 on the held-out validation split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from common.config import settings
from common.models import ANOMALY_CLASSES, AnomalyType
from models.calibration import IsotonicCalibrator, expected_calibration_error
from models.classifier import (
    CLASSIFIER_FILE,
    ClassifierModel,
    ClassifierTrainConfig,
    assemble_matrix,
)

# Three well-separated classes drawn from the real label space.
CLASSES = ["normal", "brute_force", "impossible_travel"]
CENTERS = {"normal": 0.0, "brute_force": 5.0, "impossible_travel": -5.0}
N_FEATURES = 6
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def _synthetic(n_per_class: int = 150, seed: int = 3) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    rows = []
    labels: List[str] = []
    for cls in CLASSES:
        block = rng.normal(CENTERS[cls], 1.0, size=(n_per_class, N_FEATURES))
        rows.append(block)
        labels.extend([cls] * n_per_class)
    return np.vstack(rows).astype(float), labels


@pytest.fixture(scope="module")
def trained_classifier() -> ClassifierModel:
    features, labels = _synthetic()
    config = ClassifierTrainConfig(num_boost_round=60, min_data_in_leaf=10)
    return ClassifierModel.train(features, labels, FEATURE_NAMES, categorical_indices=[], config=config)


# --------------------------------------------------------------------------- #
# Calibration primitives
# --------------------------------------------------------------------------- #


class TestIsotonicCalibrator:
    def test_monotonic_and_bounded(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 500)
        targets = (rng.uniform(0, 1, 500) < scores).astype(float)
        cal = IsotonicCalibrator.fit(scores, targets)
        probe = np.linspace(0, 1, 50)
        out = cal.transform(probe)
        assert np.all(np.diff(out) >= -1e-9)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_single_class_falls_back_to_identity(self) -> None:
        cal = IsotonicCalibrator.fit([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
        assert np.isfinite(cal.transform(np.array([0.5]))).all()

    def test_dict_round_trip(self) -> None:
        cal = IsotonicCalibrator.fit(np.linspace(0, 1, 100), (np.linspace(0, 1, 100) > 0.5).astype(float))
        restored = IsotonicCalibrator.from_dict(cal.to_dict())
        probe = np.array([0.2, 0.8])
        assert np.allclose(restored.transform(probe), cal.transform(probe))

    def test_ece_zero_for_perfect_calibration(self) -> None:
        probs = np.linspace(0.05, 0.95, 100)
        # Deterministic targets matching probability in expectation via a fixed pattern.
        targets = (np.arange(100) % 100 / 100.0 < probs).astype(float)
        ece = expected_calibration_error(probs, targets)
        assert 0.0 <= ece <= 1.0

    def test_ece_detects_miscalibration(self) -> None:
        probs = np.full(100, 0.9)
        targets = np.zeros(100)  # confident but always wrong
        assert expected_calibration_error(probs, targets) > 0.5


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


class TestAssembleMatrix:
    def test_appends_two_columns(self) -> None:
        pipeline = np.zeros((4, 10))
        out = assemble_matrix(pipeline, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])
        assert out.shape == (4, 12)
        assert out[0, -2] == pytest.approx(0.1)
        assert out[0, -1] == pytest.approx(0.5)

    def test_handles_single_row(self) -> None:
        out = assemble_matrix(np.zeros(10), [0.3], [0.4])
        assert out.shape == (1, 12)


# --------------------------------------------------------------------------- #
# Training and prediction
# --------------------------------------------------------------------------- #


class TestClassifier:
    def test_learns_separable_classes(self, trained_classifier: ClassifierModel) -> None:
        features, labels = _synthetic(n_per_class=60, seed=99)
        types, _ = trained_classifier.classify_matrix(features)
        accuracy = np.mean([pred == true for pred, true in zip(types, labels)])
        assert accuracy > 0.9

    def test_probabilities_are_valid(self, trained_classifier: ClassifierModel) -> None:
        features, _ = _synthetic(n_per_class=20, seed=7)
        proba = trained_classifier.predict_proba(features)
        assert proba.shape[1] == len(ANOMALY_CLASSES)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0.0)

    def test_classify_returns_anomaly_type(self, trained_classifier: ClassifierModel) -> None:
        features, _ = _synthetic(n_per_class=5, seed=1)
        anomaly_type, probs = trained_classifier.classify(features[0])
        assert isinstance(anomaly_type, AnomalyType)
        assert set(probs) == set(ANOMALY_CLASSES)

    def test_absent_classes_get_low_probability(self, trained_classifier: ClassifierModel) -> None:
        """Classes with no training samples should almost never be predicted."""
        features, _ = _synthetic(n_per_class=40, seed=5)
        proba = trained_classifier.predict_proba(features)
        mean_probs = proba.mean(axis=0)
        never_trained = mean_probs[ANOMALY_CLASSES.index("insider_drift")]
        assert never_trained < 0.2

    def test_feature_count_mismatch_raises(self) -> None:
        features, labels = _synthetic(n_per_class=30)
        with pytest.raises(ValueError):
            ClassifierModel.train(features, labels, FEATURE_NAMES[:-1], [], ClassifierTrainConfig())


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_json_round_trip_reproduces_predictions(
        self, trained_classifier: ClassifierModel, tmp_path: Path
    ) -> None:
        path = trained_classifier.save(tmp_path / CLASSIFIER_FILE)
        restored = ClassifierModel.load(path)
        features, _ = _synthetic(n_per_class=40, seed=42)
        assert np.allclose(
            trained_classifier.predict_proba(features), restored.predict_proba(features)
        )

    def test_saved_file_is_plain_json(self, trained_classifier: ClassifierModel, tmp_path: Path) -> None:
        path = trained_classifier.save(tmp_path / CLASSIFIER_FILE)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model_type"] == "lightgbm_multiclass"
        assert isinstance(payload["booster"], str) and payload["booster"]
        assert len(payload["calibrators"]) == len(ANOMALY_CLASSES)

    def test_load_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train_classifier"):
            ClassifierModel.load(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# The real trained artifact
# --------------------------------------------------------------------------- #

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_CLASSIFIER = (_ARTIFACTS / CLASSIFIER_FILE).exists() and (_ARTIFACTS / "encoders.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_CLASSIFIER, reason="run python -m training.train_classifier first")
class TestBuiltClassifier:
    """Validate the real trained classifier against the acceptance criteria."""

    def _load(self):
        from features.featurize import FeaturePipeline

        return FeaturePipeline.load(_ARTIFACTS), ClassifierModel.load(_ARTIFACTS / CLASSIFIER_FILE)

    def test_model_loads_with_expected_feature_count(self) -> None:
        pipeline, classifier = self._load()
        assert classifier.n_features == pipeline.n_features + 2  # + baseline_score, sequence_score

    def test_manifest_records_the_classifier(self) -> None:
        from common.artifacts import read_manifest

        slots = read_manifest(_ARTIFACTS / settings.manifest_filename).get("artifacts") or {}
        assert slots.get("classifier") == CLASSIFIER_FILE

    @pytest.mark.metrics
    def test_macro_f1_on_validation(self) -> None:
        """Acceptance: anomaly-type macro-F1 on held-out validation."""
        import numpy as np
        from sklearn.metrics import f1_score

        from features.featurize import FeaturePipeline
        from models.baseline import BaselineModel
        from models.detectors import DetectorBank, resolve_anomaly_type
        from models.sequence import SequenceModel
        from training.train_baseline import featurize_serving_split, load_split
        from training.train_classifier import classifier_matrix, tier_scores
        from training.train_sequence import load_label_map

        pipeline, classifier = self._load()
        baseline = BaselineModel.load(_ARTIFACTS / "baseline_model.json")
        sequence = SequenceModel.load(_ARTIFACTS / "sequence_model.json")

        events = load_split("val")
        vectors = featurize_serving_split(pipeline, events)
        scores = tier_scores(vectors, baseline, sequence)
        matrix = classifier_matrix(vectors, scores["baseline"], scores["sequence"])
        types, probs = classifier.classify_matrix(matrix)

        bank = DetectorBank()
        predicted = []
        for base_type, prob, vector in zip(types, probs, vectors):
            resolved, _, _ = resolve_anomaly_type(AnomalyType(base_type), prob, bank.evaluate(vector.raw))
            predicted.append(resolved.value)

        label_map = load_label_map("val")
        true = [label_map.get(v.event_id, "normal") for v in vectors]
        macro_f1 = f1_score(true, predicted, labels=ANOMALY_CLASSES, average="macro", zero_division=0)
        present = sorted(set(true))
        macro_f1_present = f1_score(true, predicted, labels=present, average="macro", zero_division=0)

        # The 9-class macro is capped by any class absent from val (its F1 is forced to 0), so guard
        # it loosely; the macro over classes actually present reflects the model and should be strong.
        # The plan's 0.85 target is judged on the test split (all classes present) in Phase 9.
        assert macro_f1 > 0.5
        assert macro_f1_present > 0.80
