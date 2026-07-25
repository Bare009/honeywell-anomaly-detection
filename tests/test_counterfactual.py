"""Counterfactual search tests (Phase 6, D2).

A small classifier is trained where one feature drives an attack class. The search should find that
moving that feature to typical flips the verdict to benign, keep only changes that help, and report
the result honestly when no small change suffices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from explainability.counterfactual import CounterfactualSearch
from models.classifier import ClassifierModel, ClassifierTrainConfig
from models.calibration import IsotonicCalibrator
from models.risk import RiskModel

FEATURE_NAMES = ["driver", "n1", "n2", "baseline_score", "sequence_score"]
MUTABLE = ["driver", "n1", "n2"]


def _train_classifier() -> ClassifierModel:
    """A classifier where a large ``driver`` value means brute_force, near-zero means normal."""
    rng = np.random.default_rng(0)
    rows: List[np.ndarray] = []
    labels: List[str] = []
    for _ in range(400):
        rows.append([rng.normal(0, 0.3), rng.normal(0, 1), rng.normal(0, 1), rng.uniform(0, 0.3), rng.uniform(0, 0.3)])
        labels.append("normal")
    for _ in range(400):
        rows.append([rng.normal(5, 0.5), rng.normal(0, 1), rng.normal(0, 1), rng.uniform(0, 0.3), rng.uniform(0, 0.3)])
        labels.append("brute_force")
    return ClassifierModel.train(
        np.array(rows), labels, FEATURE_NAMES, categorical_indices=[],
        config=ClassifierTrainConfig(num_boost_round=60, min_data_in_leaf=10),
    )


def _risk_model() -> RiskModel:
    """Classifier-only fusion, identity calibration, so risk == 100 * attack probability."""
    return RiskModel(
        weights={"baseline": 0.0, "sequence": 0.0, "classifier": 1.0},
        calibrator=IsotonicCalibrator(x=[0.0, 1.0], y=[0.0, 1.0]),
        alert_threshold=50.0,
        budget_threshold=50.0,
    )


@pytest.fixture(scope="module")
def search() -> CounterfactualSearch:
    return CounterfactualSearch(_train_classifier(), _risk_model(), MUTABLE)


class TestCounterfactual:
    def test_flips_attack_to_benign(self, search: CounterfactualSearch) -> None:
        row = np.array([5.0, 0.0, 0.0, 0.1, 0.1])  # strong attack signal on 'driver'
        result = search.search(row, baseline_score=0.1, sequence_score=0.1)
        assert result.found is True
        assert result.resulting_risk < result.original_risk
        assert any(change.feature == "driver" for change in result.changes)

    def test_already_benign_needs_no_change(self, search: CounterfactualSearch) -> None:
        row = np.array([0.0, 0.0, 0.0, 0.1, 0.1])
        result = search.search(row, baseline_score=0.1, sequence_score=0.1)
        assert result.found is True
        assert result.changes == []

    def test_changes_are_minimal(self, search: CounterfactualSearch) -> None:
        """Only helpful changes are kept; the irrelevant noise features are not touched."""
        row = np.array([5.0, 2.0, -2.0, 0.1, 0.1])
        result = search.search(row, baseline_score=0.1, sequence_score=0.1)
        assert result.found is True
        # 'driver' is the only feature that matters, so it must be among the (few) changes.
        assert result.changes[0].feature == "driver"
        assert len(result.changes) <= search.max_changes

    def test_does_not_touch_tier_scores(self, search: CounterfactualSearch) -> None:
        row = np.array([5.0, 0.0, 0.0, 0.9, 0.9])
        result = search.search(row, baseline_score=0.9, sequence_score=0.9)
        changed = {change.feature for change in result.changes}
        assert "baseline_score" not in changed and "sequence_score" not in changed

    def test_reports_original_and_resulting_risk(self, search: CounterfactualSearch) -> None:
        row = np.array([5.0, 0.0, 0.0, 0.1, 0.1])
        result = search.search(row, baseline_score=0.1, sequence_score=0.1)
        assert result.original_risk is not None and result.resulting_risk is not None
        assert result.summary
