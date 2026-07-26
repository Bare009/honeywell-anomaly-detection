"""Evaluation tests (Phase 9).

Unit tests cover the metric helpers, the drift experiment (deterministic, no models) and the report
assembler (from synthetic artifacts). An integration test computes the headline metrics on a real
validation slice when the trained stack is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common.config import settings
from common.models import ANOMALY_CLASSES

from evaluation.evaluate import (
    per_class_metrics,
    precision_at_k_curve,
    recall_at_budget,
)


class TestMetricHelpers:
    def test_recall_at_budget_perfect_ranking(self) -> None:
        risk = np.array([95.0, 90.0, 85.0, 10.0, 5.0, 1.0])
        y = np.array([1, 1, 0, 0, 0, 0])
        # Top ~33% (k=2) captures both positives.
        assert recall_at_budget(risk, y, 0.34) == pytest.approx(1.0)

    def test_recall_at_budget_no_positives_is_nan(self) -> None:
        assert np.isnan(recall_at_budget(np.array([1.0, 2.0]), np.array([0, 0]), 0.5))

    def test_precision_at_k_curve_bounds_and_length(self) -> None:
        rng = np.random.default_rng(0)
        risk = rng.uniform(0, 100, 500)
        y = (rng.uniform(0, 1, 500) < 0.1).astype(int)
        curve = precision_at_k_curve(risk, y, points=50)
        assert len(curve) == 50
        assert all(0.0 <= p <= 1.0 for p in curve)

    def test_per_class_metrics_shape(self) -> None:
        labels = ["normal", "brute_force", "normal", "lateral_movement"]
        predicted = ["normal", "brute_force", "normal", "normal"]
        result = per_class_metrics(labels, predicted)
        assert set(result) == set(ANOMALY_CLASSES)
        assert result["brute_force"]["recall"] == pytest.approx(1.0)
        assert result["lateral_movement"]["recall"] == pytest.approx(0.0)

    def test_detector_precision_counts_fires(self) -> None:
        from evaluation.evaluate import detector_precision
        from evaluation.scoring import ScoredSplit
        from models.detectors import DetectorResult
        from common.models import AnomalyType

        fired = DetectorResult("brute_force", AnomalyType.BRUTE_FORCE, fired=True, confidence=1.0)
        silent = DetectorResult("brute_force", AnomalyType.BRUTE_FORCE, fired=False, confidence=0.0)
        it_silent = DetectorResult("impossible_travel", AnomalyType.IMPOSSIBLE_TRAVEL, fired=False, confidence=0.0)

        scored = ScoredSplit(
            split="test",
            vectors=[],
            baseline=np.array([]),
            sequence=np.array([]),
            attack=np.array([]),
            risk=np.array([]),
            predicted_type=[],
            class_probs=[],
            detector_results=[[fired, it_silent], [silent, it_silent]],
            labels=["brute_force", "normal"],
            campaign_ids=[None, None],
            event_ids=["a", "b"],
            entity_ids=["e1", "e2"],
        )
        result = detector_precision(scored)
        assert result["brute_force"]["n_fired"] == 1.0
        assert result["brute_force"]["anomaly_precision"] == pytest.approx(1.0)


class TestDriftExperiment:
    def test_adaptation_reduces_false_positives(self, tmp_path: Path) -> None:
        from evaluation.drift_experiment import run

        summary = run(artifacts_dir=tmp_path)
        p = summary["payload"]
        assert p["fp_rate_before"] > p["fp_rate_after"]
        assert p["abrupt_flagged"] is True
        assert p["abrupt_max_psi"] >= p["series"]["threshold"]
        assert (tmp_path / "drift_metrics.json").exists()


class TestReportAssembly:
    def test_report_renders_sections(self, tmp_path: Path) -> None:
        from evaluation.report import assemble

        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset_summary": {"split": "test", "n_events": 100, "n_entities": 10, "anomaly_rate": 0.01},
                    "pr_auc": 0.93,
                    "roc_auc": 0.98,
                    "recall_at_1pct_budget": 0.85,
                    "macro_f1": 0.78,
                    "calibration_ece": 0.03,
                    "per_class": {c: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5} for c in ANOMALY_CLASSES},
                    "detector_precision": {
                        "impossible_travel": {"anomaly_precision": 0.9, "type_precision": 0.4, "n_fired": 11},
                        "brute_force": {"anomaly_precision": 1.0, "type_precision": 1.0, "n_fired": 20},
                    },
                    "notes": "macro_f1 present 0.85",
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "coldstart_metrics.json").write_text(
            json.dumps({"recall_with_priors": 0.75, "recall_without_priors": 0.6, "uplift": 0.15,
                        "n_cold_entities": 12, "n_cold_anomalies": 20}),
            encoding="utf-8",
        )
        out = tmp_path / "REPORT.md"
        assemble(artifacts_dir=tmp_path, output=out)
        text = out.read_text(encoding="utf-8")
        assert "Headline metrics" in text
        assert "PR-AUC" in text
        assert "Cold-start ablation" in text
        assert "Deliverable" in text
        assert "brute_force" in text


_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_ALL = all(
    (_ARTIFACTS / name).exists()
    for name in ("encoders.json", "baseline_model.json", "sequence_model.json", "classifier.json", "risk_model.json")
)


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_ALL, reason="run the full training pipeline first")
class TestBuiltEvaluation:
    def test_headline_metrics_on_validation_slice(self) -> None:
        from evaluation.evaluate import compute_metrics
        from evaluation.scoring import load_models, score_split

        models = load_models()
        scored = score_split("val", *models, limit=3000)
        m = compute_metrics(scored)
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert 0.0 <= m["recall_at_1pct_budget"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert 0.0 <= m["calibration_ece"] <= 1.0
        assert len(m["confusion_matrix"]) == len(ANOMALY_CLASSES)
