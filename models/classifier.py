"""Tier 3 -- the anomaly-type classifier (Deliverable #4).

A calibrated LightGBM multi-class model that names the anomaly. LightGBM is the right tool for
imbalanced tabular data: it is fast on CPU, handles the categorical codes natively, and its
probabilities calibrate cleanly. It reads the full engineered feature vector **plus the two
unsupervised tier scores** (``baseline_score`` and ``sequence_score``), so the classifier can lean
on "this reconstructs badly" and "this command order is surprising" as evidence, not just the raw
behavioural features.

Imbalance is handled by the strength of the feature signal rather than by class weighting. Class
weights are available (``weight_mode``) but default to off: measured on validation, inverse-frequency
weighting *hurt* both type accuracy and detection, because it pushed the model to over-predict rare
classes (a device-novelty signal became "device_spoofing" on benign new devices). The rare attack
classes are separable enough on the engineered features that the unweighted model detects them well
while keeping precision. Reported probabilities are made trustworthy by **per-class isotonic
calibration** fitted on a held-out slice -- calibrating on the training fit would just relearn its
overconfidence.

Persistence is JSON: the booster serializes to its own text format (stable across versions, unlike
a pickle) and the calibrators are small knot tables. No pickled estimator anywhere.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.config import settings
from common.models import ANOMALY_CLASS_INDEX, ANOMALY_CLASSES, AnomalyType
from common.seed import lightgbm_params, set_global_seed
from models.calibration import IsotonicCalibrator

logger = logging.getLogger(__name__)

#: Artifact filename for the persisted classifier.
CLASSIFIER_FILE = "classifier.json"

#: The unsupervised tier scores appended to the pipeline feature block, in this order.
TIER_SCORE_NAMES: List[str] = ["baseline_score", "sequence_score"]


def assemble_matrix(
    pipeline_matrix: np.ndarray,
    baseline_scores: Sequence[float],
    sequence_scores: Sequence[float],
) -> np.ndarray:
    """Append the two tier scores as trailing columns of the classifier feature matrix."""
    base = np.asarray(pipeline_matrix, dtype=float)
    if base.ndim == 1:
        base = base[None, :]
    extra = np.column_stack(
        [np.asarray(baseline_scores, dtype=float), np.asarray(sequence_scores, dtype=float)]
    )
    return np.hstack([base, extra])


@dataclass
class ClassifierTrainConfig:
    """Training hyper-parameters for the LightGBM classifier."""

    num_boost_round: int = 300
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    #: "none" (default, best on validation), "balanced" (inverse frequency) or "sqrt" (tempered).
    #: Inverse-frequency weighting over-predicts rare classes here and lowers precision, so it is off.
    weight_mode: str = "none"
    #: Fraction of the training rows held out to fit the isotonic calibrators.
    calibration_fraction: float = 0.15
    seed: Optional[int] = None


def _class_weights(labels: np.ndarray, mode: str) -> Dict[int, float]:
    """Per-class sample weight so the rare attack classes are not swamped by ``normal``."""
    n_classes = len(ANOMALY_CLASSES)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights: Dict[int, float] = {}
    total = float(labels.size)
    for index in range(n_classes):
        count = counts[index]
        if count <= 0:
            weights[index] = 1.0
            continue
        balanced = total / (n_classes * count)
        if mode == "balanced":
            weights[index] = balanced
        elif mode == "sqrt":
            weights[index] = float(np.sqrt(balanced))
        else:
            weights[index] = 1.0
    return weights


class ClassifierModel:
    """A trained LightGBM multi-class booster plus per-class isotonic calibration."""

    def __init__(
        self,
        booster: Any,
        calibrators: Sequence[IsotonicCalibrator],
        feature_names: Sequence[str],
        categorical_indices: Sequence[int],
        class_order: Optional[Sequence[str]] = None,
    ) -> None:
        self.booster = booster
        self.calibrators = list(calibrators)
        self.feature_names = list(feature_names)
        self.categorical_indices = list(categorical_indices)
        self.class_order = list(class_order) if class_order else list(ANOMALY_CLASSES)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def _as_matrix(self, features: Any) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        return array[None, :] if array.ndim == 1 else array

    def raw_proba(self, features: Any) -> np.ndarray:
        """Uncalibrated LightGBM class probabilities, shape ``(n, n_classes)``."""
        matrix = self._as_matrix(features)
        proba = self.booster.predict(matrix)
        return np.atleast_2d(np.asarray(proba, dtype=float))

    def predict_proba(self, features: Any) -> np.ndarray:
        """Calibrated, renormalized class probabilities, shape ``(n, n_classes)``."""
        raw = self.raw_proba(features)
        if not self.calibrators:
            return raw
        calibrated = np.empty_like(raw)
        for index in range(raw.shape[1]):
            calibrated[:, index] = self.calibrators[index].transform(raw[:, index])
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return calibrated / row_sums

    def classify_matrix(self, features: Any) -> Tuple[List[str], List[Dict[str, float]]]:
        """Return the predicted class name and calibrated probability dict for each row."""
        proba = self.predict_proba(features)
        types: List[str] = []
        dicts: List[Dict[str, float]] = []
        for row in proba:
            best = int(np.argmax(row))
            types.append(self.class_order[best])
            dicts.append({name: float(row[i]) for i, name in enumerate(self.class_order)})
        return types, dicts

    def classify(self, features: Any) -> Tuple[AnomalyType, Dict[str, float]]:
        """Classify a single feature row into an :class:`AnomalyType` and probability dict."""
        types, dicts = self.classify_matrix(self._as_matrix(features))
        return AnomalyType(types[0]), dicts[0]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @classmethod
    def train(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
        feature_names: Sequence[str],
        categorical_indices: Sequence[int],
        config: Optional[ClassifierTrainConfig] = None,
    ) -> "ClassifierModel":
        """Fit the booster and per-class calibrators.

        A calibration slice is held out from the training rows; the booster fits on the rest and the
        isotonic calibrators are fitted on the slice, so calibration reflects unseen data.
        """
        import lightgbm as lgb

        config = config or ClassifierTrainConfig()
        seed = set_global_seed(config.seed)

        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
        y = np.asarray([ANOMALY_CLASS_INDEX[label] for label in labels], dtype=int)
        if matrix.shape[0] != y.size:
            raise ValueError("feature/label count mismatch")
        if matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"matrix has {matrix.shape[1]} columns but {len(feature_names)} names given"
            )

        # Deterministic calibration hold-out.
        rng = np.random.default_rng(seed)
        order = rng.permutation(matrix.shape[0])
        n_calib = max(1, int(round(matrix.shape[0] * config.calibration_fraction)))
        n_calib = min(n_calib, matrix.shape[0] - 1)
        calib_idx = order[:n_calib]
        fit_idx = order[n_calib:]

        weight_map = _class_weights(y[fit_idx], config.weight_mode)
        sample_weight = np.array([weight_map[label] for label in y[fit_idx]], dtype=float)

        params = lightgbm_params(
            seed=seed,
            objective="multiclass",
            num_class=len(ANOMALY_CLASSES),
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            min_data_in_leaf=config.min_data_in_leaf,
            feature_fraction=config.feature_fraction,
            bagging_fraction=config.bagging_fraction,
            bagging_freq=config.bagging_freq,
            lambda_l2=config.lambda_l2,
        )

        train_set = lgb.Dataset(
            matrix[fit_idx],
            label=y[fit_idx],
            weight=sample_weight,
            categorical_feature=list(categorical_indices),
            free_raw_data=False,
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=config.num_boost_round,
        )

        model = cls(booster, [], feature_names, categorical_indices, ANOMALY_CLASSES)

        # Per-class isotonic calibration on the held-out slice (one-vs-rest).
        raw_calib = model.raw_proba(matrix[calib_idx])
        calibrators: List[IsotonicCalibrator] = []
        for index in range(len(ANOMALY_CLASSES)):
            target = (y[calib_idx] == index).astype(float)
            calibrators.append(IsotonicCalibrator.fit(raw_calib[:, index], target))
        model.calibrators = calibrators

        logger.info(
            "Classifier trained: %d features, %d classes, %d fit / %d calib rows",
            matrix.shape[1],
            len(ANOMALY_CLASSES),
            fit_idx.size,
            calib_idx.size,
        )
        return model

    # ------------------------------------------------------------------ #
    # Persistence (JSON, never pickle)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "lightgbm_multiclass",
            "feature_names": list(self.feature_names),
            "categorical_indices": list(self.categorical_indices),
            "class_order": list(self.class_order),
            "booster": self.booster.model_to_string(),
            "calibrators": [calibrator.to_dict() for calibrator in self.calibrators],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ClassifierModel":
        import lightgbm as lgb

        booster = lgb.Booster(model_str=payload["booster"])
        calibrators = [
            IsotonicCalibrator.from_dict(item) for item in payload.get("calibrators", [])
        ]
        return cls(
            booster=booster,
            calibrators=calibrators,
            feature_names=list(payload.get("feature_names", [])),
            categorical_indices=list(payload.get("categorical_indices", [])),
            class_order=list(payload.get("class_order", ANOMALY_CLASSES)),
        )

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else Path(settings.artifacts_dir) / CLASSIFIER_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        logger.info("Wrote classifier to %s", target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ClassifierModel":
        source = Path(path) if path else Path(settings.artifacts_dir) / CLASSIFIER_FILE
        if not source.exists():
            raise FileNotFoundError(
                f"No classifier at {source}. Run: python -m training.train_classifier"
            )
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


__all__ = [
    "CLASSIFIER_FILE",
    "TIER_SCORE_NAMES",
    "assemble_matrix",
    "ClassifierTrainConfig",
    "ClassifierModel",
]
