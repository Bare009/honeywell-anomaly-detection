"""Tier 2 -- the GRU next-event sequence model (Deliverable #3).

The brief allows an LSTM, GRU, Transformer or a graph model. We build exactly one: a GRU trained
to predict the next command token. The reasoning is recorded in the plan -- the vocabulary is tiny
(~50 tokens) and sequences cap at 20 steps, so a Transformer's long-range attention has nothing to
exploit and would overfit at higher cost; a GRU has three gates against an LSTM's four (fewer
parameters, equal or better on short sequences with limited data); and a graph models adjacency,
not order, which is the whole point of a *sequence-aware* deliverable.

**One forward pass, one score, one explanation.** The model reads a command sequence and, at each
position, predicts the next token. The per-position negative log-likelihood (NLL) is the surprise of
that step. The sequence score is the mean NLL over the real positions; the per-step NLL *is* the
attribution the explainability layer highlights. Score and explanation are the same numbers, so they
can never disagree.

**Padding never touches the loss.** Sequences are left-padded to a fixed length with a reserved
``<pad>`` id and prefixed with ``<bos>`` so "what usually comes first" is learnable. The loss and the
score are computed only at positions whose *context* is a real token (``input != <pad>``); with
contiguous left padding that is exactly the set of positions predicting a real next token. A padded
or empty command sequence therefore contributes nothing and scores neutrally, rather than crashing.

**JSON, never pickle**, for the same reason as every other artifact: a pickled ``torch`` module is
tied to the library version that wrote it. The token list is stored alongside the weights so
attribution can name tokens without loading the vocabulary separately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.config import settings
from common.models import SequenceStepAttribution
from common.seed import set_global_seed
from features.featurize import FeatureVector
from features.sequences import BOS_ID, PAD_ID, SequenceVocab

from models.baseline import ScoreNormalizer

logger = logging.getLogger(__name__)

#: Artifact filename for the persisted sequence model.
SEQUENCE_FILE = "sequence_model.json"

DEFAULT_EMBEDDING_DIM = 32
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 1


def _right_align(
    matrix: np.ndarray, pad_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-align left-padded sequences so real tokens start at column 0 (for packing).

    Command sequences arrive left-padded, with the real tokens as a contiguous suffix. Returns
    ``(suffix, real_len, targets, orig_index)``: the right-aligned ids, the real length per row
    (including ``<bos>``), the next-token targets (``suffix`` shifted left by one), and the original
    left-padded position of each suffix column (so attribution can point back at the right step).
    """
    matrix = np.asarray(matrix, dtype=np.int64)
    batch = matrix.shape[0]
    real_positions = [np.where(row != pad_id)[0] for row in matrix]
    real_len = np.array([len(pos) for pos in real_positions], dtype=np.int64)
    width = int(real_len.max()) if batch and real_len.max() > 0 else 1

    suffix = np.full((batch, width), pad_id, dtype=np.int64)
    orig_index = np.full((batch, width), -1, dtype=np.int64)
    for row, positions in enumerate(real_positions):
        if positions.size:
            suffix[row, : positions.size] = matrix[row, positions]
            orig_index[row, : positions.size] = positions

    targets = np.full_like(suffix, pad_id)
    targets[:, :-1] = suffix[:, 1:]
    return suffix, real_len, targets, orig_index

# The mean NLL at this quantile of held-out real command sequences maps to a score of 0.5, so
# ordinary command flows score low and only surprising orderings climb toward one.
DEFAULT_NORMALIZER_QUANTILE = 0.95


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class GRUSequenceModel(nn.Module):
    """Embedding -> GRU -> next-token softmax.

    A small recurrent language model over command tokens. The embedding uses ``padding_idx`` so the
    pad token has a fixed zero vector and never accrues a gradient.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_id)
        self.gru = nn.GRU(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # noqa: D401 - nn.Module hook
        """Return next-token logits, shape ``(batch, steps, vocab)``.

        When ``lengths`` is given the sequence is packed, so the GRU only ever runs over the real
        tokens and never accumulates hidden state across padding. That makes the score invariant to
        how much a sequence was padded, which matters because a shorter command sequence would
        otherwise inherit a different recurrent "warm-up" purely from its leading ``<pad>`` run.
        """
        embedded = self.embedding(x)
        if lengths is None:
            hidden, _ = self.gru(embedded)
        else:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            hidden, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=embedded.shape[1]
            )
        return self.output(hidden)


# --------------------------------------------------------------------------- #
# Training config
# --------------------------------------------------------------------------- #


@dataclass
class SequenceTrainConfig:
    """Training hyper-parameters for the GRU, kept together so a run is fully described."""

    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    patience: int = 6
    holdout_fraction: float = 0.15
    normalizer_quantile: float = DEFAULT_NORMALIZER_QUANTILE
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# The model wrapper
# --------------------------------------------------------------------------- #


class SequenceModel:
    """A trained GRU language model plus its score normalizer and token list.

    Built by :meth:`train` offline, or by :meth:`load` in the serving container. The token list is
    carried with the model so :meth:`attribute_sequence` can name tokens without the vocabulary.
    """

    def __init__(
        self,
        net: GRUSequenceModel,
        tokens: Sequence[str],
        max_len: int,
        normalizer: ScoreNormalizer,
    ) -> None:
        self.net = net
        self.net.eval()
        self.tokens: List[str] = list(tokens)
        self.max_len = int(max_len)
        self.normalizer = normalizer
        if len(self.tokens) != net.vocab_size:
            raise ValueError(
                f"token list has {len(self.tokens)} entries but the network vocab is "
                f"{net.vocab_size}"
            )

    @property
    def vocab_size(self) -> int:
        return self.net.vocab_size

    @property
    def pad_id(self) -> int:
        return self.net.pad_id

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #

    def _as_id_matrix(
        self,
        data: Union[FeatureVector, Sequence[FeatureVector], Sequence[Sequence[int]], np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """Return ``(id_matrix, was_single)`` for any accepted input shape."""
        if isinstance(data, FeatureVector):
            return np.asarray([data.sequence_ids], dtype=np.int64), True
        if isinstance(data, np.ndarray):
            single = data.ndim == 1
            matrix = data[None, :] if single else data
            return matrix.astype(np.int64), single
        items = list(data)
        if not items:
            return np.zeros((0, self.max_len), dtype=np.int64), False
        if isinstance(items[0], FeatureVector):
            rows = [item.sequence_ids for item in items]
        else:
            rows = [list(item) for item in items]
        return np.asarray(rows, dtype=np.int64), False

    # ------------------------------------------------------------------ #
    # Core: one forward pass yields both the score and the attribution
    # ------------------------------------------------------------------ #

    def _prepare_batch(
        self, id_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Right-align each sequence's real tokens and describe the next-token targets.

        Command sequences arrive left-padded (real tokens as a contiguous suffix). Here they are
        re-aligned so the real tokens start at column 0, which is what ``pack_padded_sequence``
        needs. Returns:

        * ``suffix`` -- ``(batch, width)`` right-aligned token ids (``width`` = longest real length);
        * ``real_len`` -- number of real tokens per row (includes ``<bos>``);
        * ``targets`` -- ``suffix`` shifted left by one (the next token to predict at each step);
        * ``orig_index`` -- original left-padded position of each ``suffix`` column, for attribution.
        """
        matrix = np.asarray(id_matrix, dtype=np.int64)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D id matrix, got shape {matrix.shape}")
        return _right_align(matrix, self.pad_id)

    def _forward_nll(
        self, id_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-step NLL, validity mask, targets and original positions from one forward pass.

        Everything is aligned to the right-aligned suffix layout of width ``w``. ``nll[i, j]`` is the
        surprise of the token that follows suffix position ``j``; ``mask[i, j]`` is 1 for the
        ``real_len - 1`` real predictions in that row. ``orig_index`` maps a suffix column back to
        the caller's left-padded position, so attribution can point at the right step.
        """
        suffix, real_len, targets, orig_index = self._prepare_batch(id_matrix)
        batch, width = suffix.shape
        if width < 2:
            empty = np.zeros((batch, 0), dtype=float)
            return empty, empty, empty.astype(np.int64), orig_index

        self.net.eval()
        with torch.no_grad():
            suffix_t = torch.as_tensor(suffix, dtype=torch.long)
            lengths_t = torch.clamp(torch.as_tensor(real_len, dtype=torch.long), min=1)
            logits = self.net(suffix_t, lengths_t)
            log_probs = F.log_softmax(logits, dim=-1)
            target_t = torch.as_tensor(targets, dtype=torch.long)
            token_log_prob = log_probs.gather(-1, target_t.unsqueeze(-1)).squeeze(-1)
            nll = (-token_log_prob).cpu().numpy()

        columns = np.arange(width)
        mask = (columns[None, :] < (real_len[:, None] - 1)).astype(float)
        return nll, mask, targets, orig_index

    def sequence_nll(
        self,
        data: Union[FeatureVector, Sequence[FeatureVector], Sequence[Sequence[int]], np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Mean NLL over the real positions of each sequence (the raw, unnormalized surprise).

        A sequence with no real context (empty or all-padding) scores ``0.0`` -- neutral -- because
        there is nothing to be surprised by.
        """
        matrix, single = self._as_id_matrix(data)
        if matrix.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        nll, mask, _, _ = self._forward_nll(matrix)
        counts = mask.sum(axis=1)
        totals = (nll * mask).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_nll = np.where(counts > 0, totals / np.maximum(counts, 1.0), 0.0)
        return float(mean_nll[0]) if single else mean_nll

    def score_sequence(
        self,
        data: Union[FeatureVector, Sequence[FeatureVector], Sequence[Sequence[int]], np.ndarray],
    ) -> Union[float, np.ndarray]:
        """The normalized ``sequence_score`` in ``[0, 1]``; higher means more surprising."""
        raw = self.sequence_nll(data)
        if isinstance(raw, float):
            return float(self.normalizer.normalize(np.array([raw]))[0])
        if raw.shape[0] == 0:
            return raw
        return self.normalizer.normalize(raw)

    def attribute_sequence(self, vector: FeatureVector) -> List[SequenceStepAttribution]:
        """Per-step attribution: which command in the sequence carried the surprise.

        The weight on each step is its share of the sequence's total surprise, so the weights over
        the real positions form a distribution -- an analyst reads "this step accounts for most of
        the anomaly". Derived from the same NLL as the score, never a second computation.
        """
        matrix, _ = self._as_id_matrix(vector)
        nll, mask, targets, orig_index = self._forward_nll(matrix)
        if nll.shape[1] == 0:
            return []

        row_nll = nll[0]
        row_mask = mask[0]
        row_targets = targets[0]
        row_index = orig_index[0]
        total = float((row_nll * row_mask).sum())

        attributions: List[SequenceStepAttribution] = []
        for column in range(len(row_nll)):
            if row_mask[column] <= 0.0:
                continue
            token_id = int(row_targets[column])
            token = self.tokens[token_id] if 0 <= token_id < len(self.tokens) else "<unk>"
            weight = float(row_nll[column] / total) if total > 0 else 0.0
            # Report the predicted token's position in the caller's (left-padded) sequence.
            position = int(row_index[column + 1]) if column + 1 < len(row_index) else column
            attributions.append(
                SequenceStepAttribution(position=max(position, 0), token=token, score=weight)
            )
        return attributions

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @classmethod
    def train(
        cls,
        id_matrix: np.ndarray,
        vocab: SequenceVocab,
        config: Optional[SequenceTrainConfig] = None,
    ) -> "SequenceModel":
        """Fit the GRU on a matrix of encoded (mostly-normal) command sequences.

        A held-out slice is carved off for early stopping and for fitting the score normalizer, so
        the reference surprise distribution comes from sequences the weights never trained on. The
        loss is the mean NLL over real positions only -- padding is masked out.
        """
        config = config or SequenceTrainConfig()
        seed = set_global_seed(config.seed)

        matrix = np.asarray(id_matrix, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError(f"expected a 2-D id matrix with >=2 columns, got {matrix.shape}")
        if matrix.shape[0] < 2:
            raise ValueError("need at least 2 sequences to fit the sequence model")

        max_len = matrix.shape[1]
        pad_id = PAD_ID

        rng = np.random.default_rng(seed)
        order = rng.permutation(matrix.shape[0])
        n_holdout = max(1, int(round(matrix.shape[0] * config.holdout_fraction)))
        n_holdout = min(n_holdout, matrix.shape[0] - 1)
        holdout = matrix[order[:n_holdout]]
        fit = matrix[order[n_holdout:]]

        net = GRUSequenceModel(
            vocab.size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            pad_id=pad_id,
        )
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        def masked_mean_nll(batch_np: np.ndarray) -> torch.Tensor:
            """Mean NLL over real next-token predictions, padding packed out of the recurrence."""
            suffix, real_len, targets, _ = _right_align(batch_np, pad_id)
            if suffix.shape[1] < 2:
                return torch.zeros((), requires_grad=True)
            suffix_t = torch.as_tensor(suffix, dtype=torch.long)
            lengths_t = torch.clamp(torch.as_tensor(real_len, dtype=torch.long), min=1)
            logits = net(suffix_t, lengths_t)
            log_probs = F.log_softmax(logits, dim=-1)
            target_t = torch.as_tensor(targets, dtype=torch.long)
            token_log_prob = log_probs.gather(-1, target_t.unsqueeze(-1)).squeeze(-1)
            columns = torch.arange(suffix.shape[1])
            length_limit = torch.as_tensor(real_len, dtype=torch.long) - 1
            mask = (columns.unsqueeze(0) < length_limit.unsqueeze(1)).to(log_probs.dtype)
            total = (-token_log_prob * mask).sum()
            denom = mask.sum().clamp(min=1.0)
            return total / denom

        best_holdout = float("inf")
        best_state: Dict[str, torch.Tensor] = {
            key: value.detach().clone() for key, value in net.state_dict().items()
        }
        epochs_without_improvement = 0
        n_fit = fit.shape[0]

        for epoch in range(config.epochs):
            net.train()
            epoch_order = rng.permutation(n_fit)
            for start in range(0, n_fit, config.batch_size):
                batch_np = fit[epoch_order[start : start + config.batch_size]]
                optimizer.zero_grad()
                loss = masked_mean_nll(batch_np)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                holdout_loss = float(masked_mean_nll(holdout).item())

            if holdout_loss < best_holdout - 1e-6:
                best_holdout = holdout_loss
                best_state = {
                    key: value.detach().clone() for key, value in net.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    logger.info(
                        "Early stopping at epoch %d (best holdout NLL %.6f)",
                        epoch + 1,
                        best_holdout,
                    )
                    break

        net.load_state_dict(best_state)
        net.eval()

        model = cls(net, list(vocab.tokens), max_len, ScoreNormalizer(center=0.0, scale=1.0))

        # Fit the normalizer on held-out sequences that actually have command content, so the
        # reference distribution is real surprise rather than a pile of neutral zeros.
        holdout_nll = np.atleast_1d(model.sequence_nll(holdout))
        _, mask_matrix, _, _ = model._forward_nll(holdout)
        has_content = mask_matrix.sum(axis=1) > 0
        reference = holdout_nll[has_content] if has_content.any() else holdout_nll
        model.normalizer = ScoreNormalizer.fit(reference, quantile=config.normalizer_quantile)

        logger.info(
            "Sequence model trained: vocab=%d, max_len=%d, holdout NLL=%.6f, "
            "normalizer(center=%.5f, scale=%.5f)",
            vocab.size,
            max_len,
            best_holdout,
            model.normalizer.center,
            model.normalizer.scale,
        )
        return model

    # ------------------------------------------------------------------ #
    # Persistence (JSON, never pickle)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        state = {
            key: value.detach().cpu().numpy().tolist()
            for key, value in self.net.state_dict().items()
        }
        return {
            "model_type": "gru_sequence",
            "vocab_size": self.net.vocab_size,
            "embedding_dim": self.net.embedding_dim,
            "hidden_dim": self.net.hidden_dim,
            "num_layers": self.net.num_layers,
            "pad_id": self.net.pad_id,
            "max_len": self.max_len,
            "tokens": list(self.tokens),
            "normalizer": self.normalizer.to_dict(),
            "state_dict": state,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SequenceModel":
        net = GRUSequenceModel(
            vocab_size=int(payload["vocab_size"]),
            embedding_dim=int(payload.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
            hidden_dim=int(payload.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
            num_layers=int(payload.get("num_layers", DEFAULT_NUM_LAYERS)),
            pad_id=int(payload.get("pad_id", PAD_ID)),
        )
        state = {
            key: torch.tensor(np.asarray(value, dtype=np.float32))
            for key, value in payload["state_dict"].items()
        }
        net.load_state_dict(state)
        net.eval()
        normalizer = ScoreNormalizer.from_dict(payload.get("normalizer", {}))
        return cls(
            net,
            list(payload.get("tokens", [])),
            int(payload.get("max_len", settings.sequence_max_len)),
            normalizer,
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the model to JSON and return the path."""
        target = Path(path) if path else Path(settings.artifacts_dir) / SEQUENCE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        logger.info("Wrote sequence model to %s", target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SequenceModel":
        """Load a model previously written by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If the artifact is missing, with the command needed to build it.
        """
        source = Path(path) if path else Path(settings.artifacts_dir) / SEQUENCE_FILE
        if not source.exists():
            raise FileNotFoundError(
                f"No sequence model at {source}. Run: python -m training.train_sequence"
            )
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


__all__ = [
    "SEQUENCE_FILE",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_NORMALIZER_QUANTILE",
    "GRUSequenceModel",
    "SequenceTrainConfig",
    "SequenceModel",
]
