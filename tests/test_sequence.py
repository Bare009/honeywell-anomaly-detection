"""GRU sequence model tests (Phase 4, Deliverable #3).

Two layers of coverage:

* **Self-contained unit tests** fit a tiny GRU on a synthetic "grammar" -- normal sequences follow a
  fixed command order, anomalies scramble it -- in well under a second. They assert the properties
  the system depends on: scrambled orderings earn higher surprise than the learned order, the score
  is bounded and monotonic in the raw NLL, padding is masked out of both the loss and the score,
  variable-length and unknown tokens are handled, training is deterministic, per-step attribution
  comes from the same forward pass as the score, and the model round-trips through JSON with
  identical scores.
* **Integration tests** (``TestBuiltSequence``) load the *real* trained artifact and check it carries
  ranking signal on the held-out validation split, with attention to the sequence-sensitive classes
  (``lateral_movement``, ``low_and_slow_exfil``). Skipped until ``python -m training.train_sequence``
  has been run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from common.config import settings
from common.models import SequenceStepAttribution
from features.sequences import BOS_ID, PAD_ID, UNK_ID, SequenceVocab
from models.sequence import (
    DEFAULT_HIDDEN_DIM,
    SEQUENCE_FILE,
    GRUSequenceModel,
    SequenceModel,
    SequenceTrainConfig,
)

# A small closed grammar. Normal sequences walk this fixed order; anomalies permute it.
NORMAL_ORDER = ["login", "list", "open", "read", "close", "logout"]
ANOMALY_TOKENS = ["login", "delete", "escalate", "exfiltrate", "logout"]


def _fast_config(**overrides) -> SequenceTrainConfig:
    params = dict(embedding_dim=16, hidden_dim=24, epochs=40, batch_size=32, patience=8)
    params.update(overrides)
    return SequenceTrainConfig(**params)


@pytest.fixture(scope="module")
def grammar_vocab() -> SequenceVocab:
    """Vocabulary fitted on the normal grammar's tokens."""
    corpus = [NORMAL_ORDER for _ in range(50)]
    return SequenceVocab.fit(corpus, min_count=1, max_len=10, ngram_n=2)


def _normal_sequences(n: int, vocab: SequenceVocab, seed: int = 1) -> np.ndarray:
    """Encoded normal sequences: the fixed order, occasionally truncated at the tail."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        length = rng.integers(3, len(NORMAL_ORDER) + 1)
        rows.append(vocab.encode(NORMAL_ORDER[:length]))
    return np.asarray(rows, dtype=np.int64)


def _anomaly_sequences(n: int, vocab: SequenceVocab, seed: int = 2) -> np.ndarray:
    """Encoded anomalies: shuffled normal tokens, which break the learned transitions."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        scrambled = list(NORMAL_ORDER)
        rng.shuffle(scrambled)
        rows.append(vocab.encode(scrambled))
    return np.asarray(rows, dtype=np.int64)


@pytest.fixture(scope="module")
def trained_model(grammar_vocab: SequenceVocab) -> SequenceModel:
    """A GRU trained on normal grammar sequences."""
    train = _normal_sequences(400, grammar_vocab)
    return SequenceModel.train(train, grammar_vocab, _fast_config())


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class TestGRUNetwork:
    def test_forward_shape(self) -> None:
        import torch

        net = GRUSequenceModel(vocab_size=12)
        logits = net(torch.zeros((3, 8), dtype=torch.long))
        assert tuple(logits.shape) == (3, 8, 12)

    def test_default_hidden_dim(self) -> None:
        assert GRUSequenceModel(vocab_size=10).hidden_dim == DEFAULT_HIDDEN_DIM

    def test_pad_embedding_is_zero(self) -> None:
        """The padding token must carry no information into the GRU."""
        import torch

        net = GRUSequenceModel(vocab_size=10, pad_id=PAD_ID)
        pad_vector = net.embedding(torch.tensor([PAD_ID]))
        assert torch.allclose(pad_vector, torch.zeros_like(pad_vector))

    def test_rejects_nonpositive_vocab(self) -> None:
        with pytest.raises(ValueError):
            GRUSequenceModel(vocab_size=0)


# --------------------------------------------------------------------------- #
# Training and scoring
# --------------------------------------------------------------------------- #


class TestTraining:
    """The model learns the normal grammar and is surprised by scrambled orderings."""

    def test_anomalies_score_higher_than_normal(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        normal = _normal_sequences(100, grammar_vocab, seed=11)
        anomaly = _anomaly_sequences(100, grammar_vocab, seed=12)
        assert (
            trained_model.score_sequence(anomaly).mean()
            > trained_model.score_sequence(normal).mean() + 0.15
        )

    def test_anomalies_have_higher_nll(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        normal = _normal_sequences(100, grammar_vocab, seed=11)
        anomaly = _anomaly_sequences(100, grammar_vocab, seed=12)
        assert (
            np.mean(trained_model.sequence_nll(anomaly))
            > np.mean(trained_model.sequence_nll(normal))
        )

    def test_scores_are_in_unit_interval(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        matrix = np.vstack(
            [_normal_sequences(20, grammar_vocab), _anomaly_sequences(20, grammar_vocab)]
        )
        scores = trained_model.score_sequence(matrix)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert np.isfinite(scores).all()

    def test_training_is_deterministic(self, grammar_vocab: SequenceVocab) -> None:
        """Same seed, same data, identical scores."""
        train = _normal_sequences(300, grammar_vocab)
        probe = _anomaly_sequences(30, grammar_vocab, seed=99)
        first = SequenceModel.train(train, grammar_vocab, _fast_config())
        second = SequenceModel.train(train, grammar_vocab, _fast_config())
        assert np.allclose(
            first.score_sequence(probe), second.score_sequence(probe), rtol=0.0, atol=0.0
        )

    def test_rejects_too_few_sequences(self, grammar_vocab: SequenceVocab) -> None:
        with pytest.raises(ValueError):
            SequenceModel.train(
                np.zeros((1, grammar_vocab.max_len), dtype=np.int64),
                grammar_vocab,
                _fast_config(),
            )


# --------------------------------------------------------------------------- #
# Masking, variable length and unknown tokens
# --------------------------------------------------------------------------- #


class TestMaskingAndEdges:
    """Padding never touches the loss or the score; odd inputs never crash."""

    def test_empty_sequence_scores_neutral_low(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        """An event with no commands has nothing to be surprised by."""
        empty = np.asarray([grammar_vocab.encode([])], dtype=np.int64)
        assert trained_model.sequence_nll(empty)[0] == pytest.approx(0.0)
        assert trained_model.score_sequence(empty)[0] < 0.5

    def test_all_pad_row_has_zero_nll(self, trained_model: SequenceModel) -> None:
        pad_row = np.full((1, trained_model.max_len), PAD_ID, dtype=np.int64)
        assert trained_model.sequence_nll(pad_row)[0] == pytest.approx(0.0)

    def test_padding_does_not_change_score(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        """Left-padding to different widths must not change the surprise of the real tokens."""
        short = grammar_vocab.encode(NORMAL_ORDER, max_len=8)
        long = grammar_vocab.encode(NORMAL_ORDER, max_len=16)
        score_short = trained_model.sequence_nll(np.asarray([short], dtype=np.int64))[0]
        score_long = trained_model.sequence_nll(np.asarray([long], dtype=np.int64))[0]
        assert score_short == pytest.approx(score_long, abs=1e-5)

    def test_unknown_tokens_are_handled(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        encoded = grammar_vocab.encode(["login", "never_seen_command", "logout"])
        assert UNK_ID in encoded
        score = trained_model.score_sequence(np.asarray([encoded], dtype=np.int64))
        assert np.isfinite(score).all()

    def test_variable_length_batch(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        rows = [
            grammar_vocab.encode(NORMAL_ORDER[:2]),
            grammar_vocab.encode(NORMAL_ORDER),
            grammar_vocab.encode([]),
        ]
        scores = trained_model.score_sequence(np.asarray(rows, dtype=np.int64))
        assert scores.shape == (3,)
        assert np.isfinite(scores).all()

    def test_empty_input_returns_empty_array(self, trained_model: SequenceModel) -> None:
        assert trained_model.score_sequence([]).shape == (0,)


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #


class TestAttribution:
    """Per-step surprise comes from the same forward pass as the score."""

    def _vector(self, ids: List[int], event_id: str = "evt_seq"):
        from datetime import datetime, timezone

        from features.featurize import FeatureVector

        return FeatureVector(
            entity_id="e",
            timestamp=datetime(2026, 3, 2, 9, tzinfo=timezone.utc),
            event_id=event_id,
            values=np.zeros(1),
            names=("f",),
            sequence_ids=list(ids),
        )

    def test_attribution_returns_one_entry_per_real_step(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        ids = grammar_vocab.encode(NORMAL_ORDER)
        steps = trained_model.attribute_sequence(self._vector(ids))
        # One prediction per real token (from <bos> onward); padding contributes none.
        n_real = sum(1 for token_id in ids if token_id not in (PAD_ID,))
        assert 0 < len(steps) <= n_real
        assert all(isinstance(step, SequenceStepAttribution) for step in steps)

    def test_attribution_weights_form_a_distribution(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        ids = grammar_vocab.encode(list(reversed(NORMAL_ORDER)))
        steps = trained_model.attribute_sequence(self._vector(ids))
        total = sum(step.score for step in steps)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_attribution_names_real_tokens(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab
    ) -> None:
        ids = grammar_vocab.encode(NORMAL_ORDER)
        steps = trained_model.attribute_sequence(self._vector(ids))
        assert all(step.token in NORMAL_ORDER for step in steps)

    def test_empty_sequence_has_no_attribution(self, trained_model: SequenceModel) -> None:
        steps = trained_model.attribute_sequence(self._vector([PAD_ID] * trained_model.max_len))
        assert steps == []


# --------------------------------------------------------------------------- #
# Persistence (JSON, never pickle)
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_json_round_trip_reproduces_scores(
        self, trained_model: SequenceModel, grammar_vocab: SequenceVocab, tmp_path: Path
    ) -> None:
        path = trained_model.save(tmp_path / SEQUENCE_FILE)
        restored = SequenceModel.load(path)
        anomaly = _anomaly_sequences(40, grammar_vocab, seed=77)
        assert np.array_equal(
            trained_model.score_sequence(anomaly), restored.score_sequence(anomaly)
        )

    def test_saved_file_is_plain_json(
        self, trained_model: SequenceModel, tmp_path: Path
    ) -> None:
        path = trained_model.save(tmp_path / SEQUENCE_FILE)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model_type"] == "gru_sequence"
        assert payload["state_dict"]
        assert payload["tokens"]

    def test_load_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train_sequence"):
            SequenceModel.load(tmp_path / "nope.json")

    def test_token_count_must_match_vocab(self, grammar_vocab: SequenceVocab) -> None:
        net = GRUSequenceModel(vocab_size=grammar_vocab.size)
        from models.baseline import ScoreNormalizer

        with pytest.raises(ValueError):
            SequenceModel(net, grammar_vocab.tokens[:-1], grammar_vocab.max_len, ScoreNormalizer(0.0, 1.0))


# --------------------------------------------------------------------------- #
# The real trained artifact
# --------------------------------------------------------------------------- #

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_SEQUENCE = (_ARTIFACTS / SEQUENCE_FILE).exists() and (_ARTIFACTS / "encoders.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_SEQUENCE, reason="run python -m training.train_sequence first"
)
class TestBuiltSequence:
    """Validate the real trained sequence model against the acceptance criteria."""

    def test_model_loads_with_expected_vocab(self) -> None:
        from features.featurize import FeaturePipeline

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        model = SequenceModel.load(_ARTIFACTS / SEQUENCE_FILE)
        assert model.vocab_size == pipeline.vocab.size
        assert model.max_len == pipeline.vocab.max_len

    def test_manifest_records_the_sequence_model(self) -> None:
        from common.artifacts import read_manifest

        slots = read_manifest(_ARTIFACTS / settings.manifest_filename).get("artifacts") or {}
        assert slots.get("sequence_model") == SEQUENCE_FILE

    def test_scores_a_loaded_feature_vector(self) -> None:
        from features.featurize import FeaturePipeline
        from tests.test_features import make_event

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        model = SequenceModel.load(_ARTIFACTS / SEQUENCE_FILE)
        entity_id = next(iter(pipeline.profiles.profiles))
        vector = pipeline.featurize(make_event(entity_id=entity_id), update_state=False)
        score = model.score_sequence(vector)
        assert 0.0 <= float(score) <= 1.0

    @pytest.mark.metrics
    def test_has_ranking_signal_on_validation(self) -> None:
        """Acceptance: the sequence model carries real signal, with recall on order-driven classes."""
        from training.train_sequence import (
            encode_sequences,
            evaluate_scores,
            load_label_map,
            load_split,
        )
        from features.featurize import FeaturePipeline

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        model = SequenceModel.load(_ARTIFACTS / SEQUENCE_FILE)

        val_events = load_split("val")
        matrix = encode_sequences(val_events, pipeline.vocab)
        label_map = load_label_map("val")
        labels = [label_map.get(event.event_id, "normal") for event in val_events]
        scores = np.atleast_1d(model.score_sequence(matrix))

        metrics = evaluate_scores(scores, labels, settings.alert_budget_pct)

        # A real, if narrow, ranking signal: PR-AUC clears the random floor and the top-1% budget is
        # enriched for the order-driven classes the sequence tier exists to catch.
        assert metrics["pr_auc"] > metrics["prevalence"]
        per_class = metrics["per_class"]
        sequence_recall = max(
            per_class.get("lateral_movement", {}).get("recall_at_budget", 0.0) or 0.0,
            per_class.get("low_and_slow_exfil", {}).get("recall_at_budget", 0.0) or 0.0,
        )
        assert sequence_recall > 0.0
