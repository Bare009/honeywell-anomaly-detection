"""Risk fusion tests (Phase 5; D4/D5).

Unit tests exercise the fusion arithmetic on synthetic tier scores: weights tuned on labelled data
concentrate on the informative tier, the calibrated risk stays in ``[0, 100]``, the alert-budget
threshold spends roughly the budget, the uncertainty band widens for cold-start entities, and a
confident deterministic detector overrides the fused number upward. Integration tests check the real
tuned risk model's recall within the budget and calibration on the held-out validation split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.config import settings
from common.models import AnomalyType
from models.calibration import expected_calibration_error
from models.detectors import DetectorResult
from models.risk import DETECTOR_RISK_FLOOR, RISK_FILE, RiskModel


def _synthetic(n: int = 4000, seed: int = 0):
    """Tier scores where the classifier carries the signal and baseline/sequence are noisier."""
    rng = np.random.default_rng(seed)
    y = (rng.uniform(0, 1, n) < 0.02).astype(int)
    classifier = np.clip(0.15 * rng.standard_normal(n) + 0.7 * y + 0.1, 0, 1)
    baseline = np.clip(0.3 * rng.uniform(0, 1, n) + 0.2 * y, 0, 1)
    sequence = np.clip(0.3 * rng.uniform(0, 1, n) + 0.2 * y, 0, 1)
    return baseline, sequence, classifier, y


class TestFusion:
    def test_fused_in_unit_interval(self) -> None:
        model = RiskModel(weights={"baseline": 0.3, "sequence": 0.3, "classifier": 0.4})
        fused = model.fuse(np.array([0.9, 0.1]), np.array([0.2, 0.8]), np.array([0.5, 0.5]))
        assert np.all((fused >= 0.0) & (fused <= 1.0))

    def test_risk_in_zero_to_hundred(self) -> None:
        b, s, c, y = _synthetic()
        model = RiskModel.tune(b, s, c, y)
        risk = model.to_risk(model.fuse(b, s, c))
        assert risk.min() >= 0.0 and risk.max() <= 100.0


class TestTuning:
    def test_weights_favour_informative_tier(self) -> None:
        b, s, c, y = _synthetic()
        model = RiskModel.tune(b, s, c, y)
        assert model.weights["classifier"] >= model.weights["baseline"]
        assert model.weights["classifier"] >= model.weights["sequence"]

    def test_weights_sum_to_one(self) -> None:
        b, s, c, y = _synthetic()
        model = RiskModel.tune(b, s, c, y)
        assert sum(model.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_budget_threshold_spends_roughly_the_budget(self) -> None:
        b, s, c, y = _synthetic()
        model = RiskModel.tune(b, s, c, y, budget_pct=0.01)
        risk = model.to_risk(model.fuse(b, s, c))
        flagged = float((risk >= model.budget_threshold).mean())
        assert 0.003 <= flagged <= 0.03

    def test_recall_at_budget_beats_chance(self) -> None:
        b, s, c, y = _synthetic()
        model = RiskModel.tune(b, s, c, y, budget_pct=0.01)
        risk = model.to_risk(model.fuse(b, s, c))
        k = max(1, int(round(y.size * 0.01)))
        top = np.argsort(risk)[::-1][:k]
        recall = y[top].sum() / y.sum()
        assert recall > 0.2  # far above the 1% a random ranking would catch


class TestUncertainty:
    def test_cold_start_band_is_wider(self) -> None:
        model = RiskModel(uncertainty_scale=0.8, coldstart_multiplier=2.0)
        warm = model.assess(0.1, 0.5, 0.9, cold_start=False)
        cold = model.assess(0.1, 0.5, 0.9, cold_start=True)
        assert cold.risk_uncertainty > warm.risk_uncertainty

    def test_band_is_bounded(self) -> None:
        model = RiskModel(max_uncertainty=50.0)
        cold = model.assess(0.0, 0.5, 1.0, cold_start=True)
        assert 0.0 <= cold.risk_uncertainty <= 50.0

    def test_agreement_gives_low_uncertainty(self) -> None:
        model = RiskModel()
        agreed = model.assess(0.8, 0.8, 0.8, cold_start=False)
        assert agreed.risk_uncertainty == pytest.approx(0.0, abs=1e-6)


class TestAssess:
    def test_detector_override_lifts_risk(self) -> None:
        model = RiskModel(alert_threshold=60.0)
        fired = DetectorResult(
            name="brute_force", anomaly_type=AnomalyType.BRUTE_FORCE, fired=True, confidence=1.0
        )
        low = model.assess(0.0, 0.0, 0.0, detector_results=[fired])
        assert low.risk_score >= DETECTOR_RISK_FLOOR
        assert low.is_anomaly is True
        assert low.in_alert_budget is True
        assert "brute_force" in low.detector_hits

    def test_no_detector_leaves_risk_from_fusion(self) -> None:
        model = RiskModel()
        result = model.assess(0.1, 0.1, 0.1)
        assert result.detector_hits == []
        assert 0.0 <= result.risk_score <= 100.0


class TestPersistence:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        b, s, c, y = _synthetic()
        model = RiskModel.tune(b, s, c, y)
        restored = RiskModel.load(model.save(tmp_path / RISK_FILE))
        assert restored.weights == pytest.approx(model.weights)
        probe = model.to_risk(model.fuse(b[:10], s[:10], c[:10]))
        restored_probe = restored.to_risk(restored.fuse(b[:10], s[:10], c[:10]))
        assert np.allclose(probe, restored_probe)

    def test_load_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train_classifier"):
            RiskModel.load(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# The real tuned artifact
# --------------------------------------------------------------------------- #

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_RISK = (_ARTIFACTS / RISK_FILE).exists() and (_ARTIFACTS / "classifier.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_RISK, reason="run python -m training.train_classifier first")
class TestBuiltRisk:
    """Validate the real tuned risk model on the held-out validation split."""

    def _score_val(self):
        import numpy as np

        from features.featurize import FeaturePipeline
        from models.baseline import BaselineModel
        from models.classifier import ClassifierModel
        from models.detectors import attack_probability
        from models.sequence import SequenceModel
        from training.train_baseline import featurize_serving_split, load_split
        from training.train_classifier import classifier_matrix, tier_scores
        from training.train_sequence import load_label_map

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        baseline = BaselineModel.load(_ARTIFACTS / "baseline_model.json")
        sequence = SequenceModel.load(_ARTIFACTS / "sequence_model.json")
        classifier = ClassifierModel.load(_ARTIFACTS / "classifier.json")
        risk = RiskModel.load(_ARTIFACTS / RISK_FILE)

        events = load_split("val")
        vectors = featurize_serving_split(pipeline, events)
        scores = tier_scores(vectors, baseline, sequence)
        matrix = classifier_matrix(vectors, scores["baseline"], scores["sequence"])
        _, probs = classifier.classify_matrix(matrix)
        attack = np.array([attack_probability(p) for p in probs])

        label_map = load_label_map("val")
        y = np.array(
            [0 if label_map.get(v.event_id, "normal") == "normal" else 1 for v in vectors]
        )
        risk_scores = risk.to_risk(risk.fuse(scores["baseline"], scores["sequence"], attack))
        return risk, risk_scores, y

    @pytest.mark.metrics
    def test_recall_at_budget(self) -> None:
        risk, risk_scores, y = self._score_val()
        k = max(1, int(round(y.size * settings.alert_budget_pct)))
        top = np.argsort(risk_scores)[::-1][:k]
        recall = y[top].sum() / y.sum()
        assert recall > 0.5  # guard; the 0.80 plan target is reported by training on val

    @pytest.mark.metrics
    def test_calibration_is_reasonable(self) -> None:
        _, risk_scores, y = self._score_val()
        ece = expected_calibration_error(risk_scores / 100.0, y)
        assert ece < 0.15  # guard; the 0.05 plan target is reported by training on val

    def test_budget_volume_is_about_one_percent(self) -> None:
        risk, risk_scores, _ = self._score_val()
        flagged = float((risk_scores >= risk.budget_threshold).mean())
        assert 0.003 <= flagged <= 0.03
