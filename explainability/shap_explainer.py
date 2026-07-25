"""SHAP explanations for the anomaly-type classifier (Deliverable #5).

SHAP gives an exact, additive attribution of a tree model's output to its input features. For the
classifier that means: for this event, which behaviours pushed the model toward calling it an
attack, and by how much. Those become the ranked "top contributing features" in the alert drawer.

``TreeExplainer`` with the tree-path-dependent perturbation is exact and fast and needs no
background dataset, so a local explanation costs a single call and adds no artifact. A global
feature-importance summary (for the model-performance page and the report) is available from a
sample of rows.

Direction is expressed relative to **risk**, not to a raw class: a feature that pushes toward the
predicted attack increases risk; the same signed value would decrease risk if the predicted class
were ``normal``. That keeps the analyst-facing language consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.models import ANOMALY_CLASS_INDEX, AnomalyType, FeatureAttribution
from models.classifier import ClassifierModel

_NORMAL = AnomalyType.NORMAL.value


class ShapExplainer:
    """Wraps a :class:`ClassifierModel` with a SHAP TreeExplainer."""

    def __init__(self, classifier: ClassifierModel) -> None:
        import shap

        self.classifier = classifier
        self.feature_names: List[str] = list(classifier.feature_names)
        self.class_order: List[str] = list(classifier.class_order)
        self.explainer = shap.TreeExplainer(classifier.booster)

    # ------------------------------------------------------------------ #
    # SHAP values, shape-normalized to (n, features, classes)
    # ------------------------------------------------------------------ #

    def _shap_values(self, matrix: np.ndarray) -> np.ndarray:
        values = self.explainer.shap_values(matrix)
        if isinstance(values, list):
            # Older API: a list of (n, features) arrays, one per class.
            return np.stack([np.asarray(v) for v in values], axis=-1)
        array = np.asarray(values)
        if array.ndim == 2:  # single output
            return array[:, :, None]
        return array  # already (n, features, classes)

    # ------------------------------------------------------------------ #
    # Local explanation
    # ------------------------------------------------------------------ #

    def local(
        self,
        row: np.ndarray,
        class_index: Optional[int] = None,
        top_k: int = 6,
        raw: Optional[Dict[str, float]] = None,
        baseline_values: Optional[Dict[str, float]] = None,
    ) -> List[FeatureAttribution]:
        """Top-``k`` feature attributions for one event's classifier row.

        Parameters
        ----------
        class_index:
            Which class to explain. Defaults to the model's predicted class.
        raw:
            Optional map of feature name -> human-readable observed value (e.g. the unscaled
            numeric features), used for display instead of the scaled model input.
        baseline_values:
            Optional map of feature name -> the entity's typical value, for the "vs baseline" column.
        """
        row = np.asarray(row, dtype=float)
        row2d = row[None, :] if row.ndim == 1 else row
        proba = self.classifier.predict_proba(row2d)[0]
        if class_index is None:
            class_index = int(np.argmax(proba))
        predicted_is_normal = self.class_order[class_index] == _NORMAL

        contributions = self._shap_values(row2d)[0][:, class_index]
        # Express as a push toward *risk*: toward an attack raises risk, toward normal lowers it.
        risk_push = -contributions if predicted_is_normal else contributions

        order = np.argsort(np.abs(risk_push))[::-1][: max(0, top_k)]
        attributions: List[FeatureAttribution] = []
        for idx in order:
            name = self.feature_names[idx]
            push = float(risk_push[idx])
            if push > 1e-9:
                direction = "increases_risk"
            elif push < -1e-9:
                direction = "decreases_risk"
            else:
                direction = "neutral"
            value = raw.get(name) if (raw and name in raw) else float(row2d[0, idx])
            baseline_value = baseline_values.get(name) if baseline_values else None
            attributions.append(
                FeatureAttribution(
                    feature=name,
                    value=value,
                    contribution=push,
                    direction=direction,
                    baseline_value=baseline_value,
                    description=self._describe(name, value, push, direction),
                )
            )
        return attributions

    @staticmethod
    def _describe(name: str, value: Any, push: float, direction: str) -> str:
        verb = {
            "increases_risk": "raised",
            "decreases_risk": "lowered",
            "neutral": "did not move",
        }[direction]
        return f"{name} = {value} {verb} the risk (SHAP {push:+.3f})"

    # ------------------------------------------------------------------ #
    # Global importance
    # ------------------------------------------------------------------ #

    def global_importance(
        self, matrix: np.ndarray, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """Mean absolute SHAP value per feature over a sample, ranked (for the report/dashboard)."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        values = self._shap_values(matrix)  # (n, features, classes)
        importance = np.abs(values).mean(axis=(0, 2))  # average over samples and classes
        order = np.argsort(importance)[::-1][: max(0, top_k)]
        return [(self.feature_names[i], float(importance[i])) for i in order]


__all__ = ["ShapExplainer"]
