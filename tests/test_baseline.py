"""Baseline autoencoder tests (Phase 3, Deliverable #2).

Two layers of coverage:

* **Self-contained unit tests** build a tiny synthetic manifold (normal points on a low-dimensional
  subspace, anomalies off it), train a small autoencoder in a fraction of a second, and assert the
  properties the rest of the system depends on: anomalies reconstruct worse than normal points, the
  score is bounded and monotonic in the raw error, training is deterministic, per-feature error is
  available for explanations, and the model round-trips through JSON with identical scores. None of
  these need the generated dataset.
* **Integration tests** (``TestBuiltBaseline``) load the *real* trained artifact and check the
  acceptance criteria on the held-out validation split. They are skipped until
  ``python -m training.train_baseline`` has been run.

The headline invariants: the score preserves ranking (so PR-AUC and recall@budget are meaningful),
cold-start inputs never produce NaNs, and a reloaded model is indistinguishable from the trained one
-- the serving container must score exactly as training did.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from common.config import settings
from common.models import AnomalyType
from features.event_features import CATEGORICAL_FEATURE_NAMES, NUMERIC_FEATURE_NAMES
from features.featurize import FeatureVector
from models.baseline import (
    BASELINE_FILE,
    DEFAULT_BOTTLENECK_DIM,
    DEFAULT_HIDDEN_DIM,
    BaselineModel,
    BaselineTrainConfig,
    ScoreNormalizer,
    TabularAutoencoder,
)


# --------------------------------------------------------------------------- #
# Synthetic data on a low-dimensional manifold
# --------------------------------------------------------------------------- #

INPUT_DIM = 12
LATENT_DIM = 3
FEATURE_NAMES = [f"f{i}" for i in range(INPUT_DIM)]


def _manifold_data(
    n_normal: int = 600, n_anomaly: int = 60, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """Normal points on a ``LATENT_DIM``-dimensional linear subspace; anomalies off it.

    An autoencoder with a bottleneck narrower than ``INPUT_DIM`` can learn the subspace and
    reconstruct normal points well, while off-manifold anomalies reconstruct poorly -- exactly the
    separation the baseline tier relies on.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(LATENT_DIM, INPUT_DIM))

    latent = rng.normal(size=(n_normal, LATENT_DIM))
    normal = latent @ basis + rng.normal(scale=0.05, size=(n_normal, INPUT_DIM))

    # Anomalies: broad uniform noise across the whole space, almost never on the subspace.
    anomaly = rng.uniform(-6.0, 6.0, size=(n_anomaly, INPUT_DIM))
    return normal.astype(np.float32), anomaly.astype(np.float32)


def _fast_config(**overrides) -> BaselineTrainConfig:
    """A small, fast training config for unit tests (real compression, few epochs)."""
    params = dict(
        hidden_dim=8,
        bottleneck_dim=LATENT_DIM,
        epochs=60,
        batch_size=64,
        patience=10,
        holdout_fraction=0.2,
    )
    params.update(overrides)
    return BaselineTrainConfig(**params)


@pytest.fixture(scope="module")
def trained_model() -> BaselineModel:
    """A small autoencoder trained on the synthetic normal manifold."""
    normal, _ = _manifold_data()
    return BaselineModel.train(normal, FEATURE_NAMES, _fast_config())


def _make_vector(values: np.ndarray, event_id: str = "evt_x") -> FeatureVector:
    """Wrap a raw numeric row in a FeatureVector (categorical codes appended as zeros)."""
    values = np.asarray(values, dtype=float)
    return FeatureVector(
        entity_id="entity_x",
        timestamp=datetime(2026, 3, 2, 9, tzinfo=timezone.utc),
        event_id=event_id,
        values=values,
        names=tuple(FEATURE_NAMES[: len(values)]),
    )


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class TestTabularAutoencoder:
    """The network geometry the plan specifies."""

    def test_forward_preserves_shape(self) -> None:
        import torch

        net = TabularAutoencoder(INPUT_DIM)
        out = net(torch.zeros((4, INPUT_DIM)))
        assert tuple(out.shape) == (4, INPUT_DIM)

    def test_default_geometry(self) -> None:
        net = TabularAutoencoder(INPUT_DIM)
        assert net.hidden_dim == DEFAULT_HIDDEN_DIM
        assert net.bottleneck_dim == DEFAULT_BOTTLENECK_DIM

    def test_bottleneck_is_narrower_than_input(self) -> None:
        """The compression is what forces the network to learn shared structure."""
        net = TabularAutoencoder(INPUT_DIM, hidden_dim=8, bottleneck_dim=3)
        assert net.bottleneck_dim < net.input_dim

    def test_rejects_nonpositive_input_dim(self) -> None:
        with pytest.raises(ValueError):
            TabularAutoencoder(0)


# --------------------------------------------------------------------------- #
# Score normalizer
# --------------------------------------------------------------------------- #


class TestScoreNormalizer:
    """The map from raw reconstruction error to a bounded, rank-preserving score."""

    def test_scores_are_bounded(self) -> None:
        normalizer = ScoreNormalizer.fit(np.linspace(0.0, 5.0, 100))
        scores = normalizer.normalize(np.array([-10.0, 0.0, 1.0, 100.0]))
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_is_monotonic_in_error(self) -> None:
        """Ranking must be preserved, or PR-AUC and recall@budget become meaningless."""
        normalizer = ScoreNormalizer.fit(np.linspace(0.0, 5.0, 200))
        errors = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0])
        scores = normalizer.normalize(errors)
        assert np.all(np.diff(scores) >= 0.0)

    def test_typical_error_scores_low_tail_scores_high(self) -> None:
        errors = np.abs(np.random.default_rng(0).normal(size=5000))
        normalizer = ScoreNormalizer.fit(errors)
        assert normalizer.normalize(np.median(errors)) < 0.5
        assert normalizer.normalize(np.quantile(errors, 0.999) * 3.0) > 0.9

    def test_empty_errors_is_safe(self) -> None:
        normalizer = ScoreNormalizer.fit([])
        assert 0.0 <= float(normalizer.normalize(1.0)) <= 1.0

    def test_constant_errors_do_not_divide_by_zero(self) -> None:
        normalizer = ScoreNormalizer.fit(np.full(50, 3.0))
        assert np.isfinite(normalizer.normalize(np.array([3.0, 10.0]))).all()

    def test_dict_round_trip(self) -> None:
        normalizer = ScoreNormalizer.fit(np.linspace(0.0, 2.0, 50))
        restored = ScoreNormalizer.from_dict(normalizer.to_dict())
        probe = np.array([0.3, 1.7, 5.0])
        assert np.allclose(restored.normalize(probe), normalizer.normalize(probe))


# --------------------------------------------------------------------------- #
# Training and scoring
# --------------------------------------------------------------------------- #


class TestTraining:
    """The model learns the normal manifold and separates anomalies from it."""

    def test_anomalies_score_higher_than_normal(self, trained_model: BaselineModel) -> None:
        normal, anomaly = _manifold_data()
        normal_scores = trained_model.score_baseline(normal)
        anomaly_scores = trained_model.score_baseline(anomaly)
        assert anomaly_scores.mean() > normal_scores.mean() + 0.2

    def test_anomalies_have_larger_reconstruction_error(
        self, trained_model: BaselineModel
    ) -> None:
        normal, anomaly = _manifold_data()
        assert (
            np.mean(trained_model.reconstruction_error(anomaly))
            > np.mean(trained_model.reconstruction_error(normal))
        )

    def test_scores_are_in_unit_interval(self, trained_model: BaselineModel) -> None:
        normal, anomaly = _manifold_data()
        scores = trained_model.score_baseline(np.vstack([normal, anomaly]))
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert np.isfinite(scores).all()

    def test_recall_at_budget_beats_chance(self, trained_model: BaselineModel) -> None:
        """The top-scoring events should be enriched for the injected anomalies."""
        normal, anomaly = _manifold_data()
        scores = np.concatenate(
            [trained_model.score_baseline(normal), trained_model.score_baseline(anomaly)]
        )
        y = np.concatenate([np.zeros(len(normal)), np.ones(len(anomaly))])
        k = len(anomaly)
        top = np.argsort(scores)[::-1][:k]
        recall = y[top].sum() / y.sum()
        assert recall > 0.5

    def test_training_is_deterministic(self) -> None:
        """Same seed, same data, identical scores -- the whole build is reproducible."""
        normal, anomaly = _manifold_data()
        first = BaselineModel.train(normal, FEATURE_NAMES, _fast_config())
        second = BaselineModel.train(normal, FEATURE_NAMES, _fast_config())
        assert np.allclose(
            first.score_baseline(anomaly), second.score_baseline(anomaly), rtol=0.0, atol=0.0
        )

    def test_rejects_too_few_rows(self) -> None:
        with pytest.raises(ValueError):
            BaselineModel.train(np.zeros((1, INPUT_DIM), dtype=np.float32), FEATURE_NAMES, _fast_config())

    def test_feature_name_count_must_match_network(self) -> None:
        """The wrapper refuses a name list that disagrees with the network's input width."""
        net = TabularAutoencoder(INPUT_DIM, hidden_dim=8, bottleneck_dim=LATENT_DIM)
        with pytest.raises(ValueError):
            BaselineModel(net, ScoreNormalizer(center=0.0, scale=1.0), FEATURE_NAMES[:-1])


# --------------------------------------------------------------------------- #
# Input handling and explanations
# --------------------------------------------------------------------------- #


class TestScoringInputs:
    """The model accepts a FeatureVector, a list of them, or a raw matrix, and slices numeric."""

    def test_single_feature_vector_returns_scalar(self, trained_model: BaselineModel) -> None:
        normal, _ = _manifold_data()
        score = trained_model.score_baseline(_make_vector(normal[0]))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_list_of_vectors_returns_array(self, trained_model: BaselineModel) -> None:
        normal, _ = _manifold_data()
        vectors = [_make_vector(row, f"evt_{i}") for i, row in enumerate(normal[:5])]
        scores = trained_model.score_baseline(vectors)
        assert scores.shape == (5,)

    def test_extra_categorical_columns_are_sliced_off(self, trained_model: BaselineModel) -> None:
        """A full feature vector carries categorical codes after the numeric block; ignore them."""
        normal, _ = _manifold_data()
        padded = np.concatenate([normal[0], np.array([3.0, 7.0, 1.0])])  # fake category codes
        from_padded = trained_model.score_baseline(padded)
        from_numeric = trained_model.score_baseline(normal[0])
        assert from_padded == pytest.approx(from_numeric)

    def test_too_few_columns_raise(self, trained_model: BaselineModel) -> None:
        with pytest.raises(ValueError):
            trained_model.score_baseline(np.zeros(INPUT_DIM - 2))

    def test_reconstruction_errors_cover_every_feature(
        self, trained_model: BaselineModel
    ) -> None:
        normal, _ = _manifold_data()
        errors = trained_model.reconstruction_errors(_make_vector(normal[0]))
        assert set(errors) == set(FEATURE_NAMES)
        assert all(np.isfinite(v) and v >= 0.0 for v in errors.values())

    def test_top_reconstruction_errors_are_sorted(self, trained_model: BaselineModel) -> None:
        _, anomaly = _manifold_data()
        top = trained_model.top_reconstruction_errors(_make_vector(anomaly[0]), k=3)
        assert len(top) == 3
        values = [value for _, value in top]
        assert values == sorted(values, reverse=True)

    def test_empty_input_returns_empty_array(self, trained_model: BaselineModel) -> None:
        assert trained_model.score_baseline([]).shape == (0,)


# --------------------------------------------------------------------------- #
# Persistence (JSON, never pickle)
# --------------------------------------------------------------------------- #


class TestPersistence:
    """A reloaded model must score identically to the trained one."""

    def test_json_round_trip_reproduces_scores(
        self, trained_model: BaselineModel, tmp_path: Path
    ) -> None:
        path = trained_model.save(tmp_path / BASELINE_FILE)
        restored = BaselineModel.load(path)

        _, anomaly = _manifold_data()
        assert np.array_equal(
            trained_model.score_baseline(anomaly), restored.score_baseline(anomaly)
        )

    def test_saved_file_is_plain_json(self, trained_model: BaselineModel, tmp_path: Path) -> None:
        """No pickle: the artifact must survive a torch/library upgrade."""
        path = trained_model.save(tmp_path / BASELINE_FILE)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model_type"] == "tabular_autoencoder"
        assert payload["input_dim"] == INPUT_DIM
        assert "state_dict" in payload and payload["state_dict"]

    def test_load_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train_baseline"):
            BaselineModel.load(tmp_path / "nope.json")

    def test_round_trip_preserves_feature_names(
        self, trained_model: BaselineModel, tmp_path: Path
    ) -> None:
        restored = BaselineModel.load(trained_model.save(tmp_path / BASELINE_FILE))
        assert restored.feature_names == trained_model.feature_names


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #


class TestColdStart:
    """Cold-start handling is inherited from the feature layer, not special-cased here."""

    def test_arbitrary_input_scores_finitely(self, trained_model: BaselineModel) -> None:
        """A cohort-blended cold-start vector must still produce a finite, comparable score."""
        blended = np.full(INPUT_DIM, 0.0, dtype=np.float32)  # a neutral, blended-toward-prior row
        score = trained_model.score_baseline(blended)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_extreme_input_does_not_overflow(self, trained_model: BaselineModel) -> None:
        score = trained_model.score_baseline(np.full(INPUT_DIM, 1e6, dtype=np.float32))
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# The real trained artifact
# --------------------------------------------------------------------------- #

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_BASELINE = (_ARTIFACTS / BASELINE_FILE).exists() and (_ARTIFACTS / "encoders.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_BASELINE, reason="run python -m training.train_baseline first"
)
class TestBuiltBaseline:
    """Validate the real trained baseline against the acceptance criteria."""

    def test_model_loads_with_expected_geometry(self) -> None:
        model = BaselineModel.load(_ARTIFACTS / BASELINE_FILE)
        assert model.input_dim == len(NUMERIC_FEATURE_NAMES)
        assert model.feature_names == list(NUMERIC_FEATURE_NAMES)

    def test_manifest_records_the_baseline(self) -> None:
        from common.artifacts import read_manifest

        slots = read_manifest(_ARTIFACTS / settings.manifest_filename).get("artifacts") or {}
        assert slots.get("baseline_model") == BASELINE_FILE
        assert slots.get("autoencoder") == BASELINE_FILE

    def test_scores_a_loaded_feature_vector(self) -> None:
        """End-to-end: load the pipeline, featurize an event, score it with the baseline."""
        from features.featurize import FeaturePipeline

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        model = BaselineModel.load(_ARTIFACTS / BASELINE_FILE)
        entity_id = next(iter(pipeline.profiles.profiles))

        from tests.test_features import make_event  # reuse the well-formed event builder

        vector = pipeline.featurize(make_event(entity_id=entity_id), update_state=False)
        score = model.score_baseline(vector)
        assert 0.0 <= score <= 1.0
        errors = model.reconstruction_errors(vector)
        assert len(errors) == len(NUMERIC_FEATURE_NAMES)

    @pytest.mark.metrics
    def test_pr_auc_uplift_over_random_on_validation(self) -> None:
        """Acceptance: the autoencoder alone gives a clear PR-AUC uplift over random on held-out."""
        from training.train_baseline import (
            evaluate_scores,
            featurize_serving_split,
            load_anomaly_flags,
            load_split,
            numeric_matrix,
        )
        from features.featurize import FeaturePipeline

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        model = BaselineModel.load(_ARTIFACTS / BASELINE_FILE)

        val_events = load_split("val")
        vectors = featurize_serving_split(pipeline, val_events)
        matrix = numeric_matrix(vectors, list(pipeline.numeric_names))
        flags = load_anomaly_flags("val")
        y = np.array([1 if flags.get(v.event_id, False) else 0 for v in vectors], dtype=int)

        scores = np.atleast_1d(model.score_baseline(matrix))
        metrics = evaluate_scores(scores, y, settings.alert_budget_pct)

        # "Clear uplift over random": PR-AUC well above the prevalence floor, and a real ranking
        # signal in ROC-AUC. Deliberately conservative thresholds -- the headline numbers live in
        # the Phase 9 report, this only guards against a broken tier.
        assert metrics["pr_auc"] > 3.0 * metrics["prevalence"]
        assert metrics["roc_auc"] > 0.6
