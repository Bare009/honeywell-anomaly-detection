"""Command-sequence vocabulary, encoding and n-gram novelty.

Two consumers depend on this module.

The **sequence model** (Phase 4) needs fixed-length integer tensors, so
:meth:`SequenceVocab.encode` pads and truncates. Truncation keeps the *tail* of a sequence,
because the most recent actions are the ones that matter for what happens next.

The **feature pipeline** needs scalar novelty measures that work per event without a neural
network: how much of this sequence has this entity done before, and how rare are these
transitions globally. Those give the tabular tiers order-awareness cheaply, and give the
baseline model something to catch lateral movement with even before the GRU exists.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.config import settings

#: Padding token, always id 0 so a padded position is trivially maskable.
PAD_TOKEN = "<pad>"
#: Everything not in the vocabulary. Unseen commands are signal, not an error.
UNK_TOKEN = "<unk>"
#: Sequence start marker, so "what usually comes first" is learnable.
BOS_TOKEN = "<bos>"

RESERVED_TOKENS: Tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2

#: Separator inside an n-gram key. Chosen because no command token contains it.
NGRAM_SEPARATOR = ">"


def ngrams(tokens: Sequence[str], n: int = 2) -> List[str]:
    """Contiguous n-grams of a token sequence, as ``"a>b"`` keys.

    A sequence shorter than ``n`` yields no n-grams rather than a padded one: inventing
    transitions that did not occur would make short sequences look artificially familiar.
    """
    if n <= 1:
        return list(tokens)
    if len(tokens) < n:
        return []
    return [
        NGRAM_SEPARATOR.join(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    ]


@dataclass
class SequenceVocab:
    """Token-to-id mapping for command sequences, plus global token statistics."""

    tokens: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    max_len: int = field(default_factory=lambda: settings.sequence_max_len)
    ngram_n: int = field(default_factory=lambda: settings.sequence_ngram_n)
    ngram_counts: Dict[str, int] = field(default_factory=dict)
    ngram_total: int = 0

    def __post_init__(self) -> None:
        # Reserved tokens always occupy the first ids, in a fixed order.
        for position, reserved in enumerate(RESERVED_TOKENS):
            if position >= len(self.tokens) or self.tokens[position] != reserved:
                self.tokens = list(RESERVED_TOKENS) + [
                    token for token in self.tokens if token not in RESERVED_TOKENS
                ]
                break
        self._index: Dict[str, int] = {token: i for i, token in enumerate(self.tokens)}

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(
        cls,
        sequences: Iterable[Sequence[str]],
        min_count: int = 2,
        max_len: Optional[int] = None,
        ngram_n: Optional[int] = None,
    ) -> "SequenceVocab":
        """Build a vocabulary from training sequences.

        Parameters
        ----------
        min_count:
            Tokens seen fewer times than this stay out of the vocabulary and map to
            ``<unk>``. That is deliberate: a token appearing once is better represented as
            "something unfamiliar" than as its own embedding fitted on a single example.
        """
        resolved_n = settings.sequence_ngram_n if ngram_n is None else ngram_n

        counts: Dict[str, int] = {}
        gram_counts: Dict[str, int] = {}
        for sequence in sequences:
            cleaned = [str(token) for token in sequence if token]
            for token in cleaned:
                counts[token] = counts.get(token, 0) + 1
            for gram in ngrams(cleaned, resolved_n):
                gram_counts[gram] = gram_counts.get(gram, 0) + 1

        kept = [token for token, count in counts.items() if count >= min_count]
        kept.sort(key=lambda token: (-counts[token], token))  # deterministic ordering

        return cls(
            tokens=list(RESERVED_TOKENS) + kept,
            counts={token: counts[token] for token in kept},
            total_count=sum(counts.values()),
            max_len=settings.sequence_max_len if max_len is None else max_len,
            ngram_n=resolved_n,
            ngram_counts=gram_counts,
            ngram_total=sum(gram_counts.values()),
        )

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Vocabulary size, including reserved tokens (the embedding dimension)."""
        return len(self.tokens)

    def token_id(self, token: str) -> int:
        """Id for one token, ``UNK_ID`` if unknown."""
        return self._index.get(str(token), UNK_ID)

    def encode(
        self,
        tokens: Sequence[str],
        max_len: Optional[int] = None,
        add_bos: bool = True,
    ) -> List[int]:
        """Encode a sequence to a fixed-length id list, left-padded.

        Left padding with truncation from the *front* keeps the most recent actions, which are
        the informative ones for next-event prediction.
        """
        length = self.max_len if max_len is None else max_len
        ids = [self.token_id(token) for token in tokens if token]
        if add_bos:
            ids = [BOS_ID] + ids

        if len(ids) >= length:
            return ids[-length:]
        return [PAD_ID] * (length - len(ids)) + ids

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> List[str]:
        """Turn ids back into tokens, for explanations and debugging."""
        output: List[str] = []
        for value in ids:
            index = int(value)
            token = self.tokens[index] if 0 <= index < len(self.tokens) else UNK_TOKEN
            if strip_special and token in RESERVED_TOKENS:
                continue
            output.append(token)
        return output

    # ------------------------------------------------------------------ #
    # Novelty measures
    # ------------------------------------------------------------------ #

    def unknown_ratio(self, tokens: Sequence[str]) -> float:
        """Share of tokens that are not in the vocabulary at all."""
        cleaned = [token for token in tokens if token]
        if not cleaned:
            return 0.0
        unknown = sum(1 for token in cleaned if self.token_id(token) == UNK_ID)
        return unknown / len(cleaned)

    def token_rarity(self, token: str) -> float:
        """Surprisal of a token globally, in nats.

        Higher means rarer. An unseen token gets the surprisal of a hypothetical
        single-occurrence token, so it scores as rare rather than as infinitely surprising --
        an unbounded value would dominate the scaled feature vector.
        """
        total = max(1, self.total_count)
        count = self.counts.get(str(token), 0)
        return -math.log(max(count, 1) / (total + 1))

    def mean_token_rarity(self, tokens: Sequence[str]) -> float:
        """Average global surprisal across a sequence."""
        cleaned = [token for token in tokens if token]
        if not cleaned:
            return 0.0
        return sum(self.token_rarity(token) for token in cleaned) / len(cleaned)

    def ngram_novelty(self, tokens: Sequence[str]) -> float:
        """Share of this sequence's n-grams never seen during training.

        This is the cheap, model-free version of "that ordering is unusual" -- the signal the
        sequence model learns properly in Phase 4.
        """
        grams = ngrams([token for token in tokens if token], self.ngram_n)
        if not grams:
            return 0.0
        unseen = sum(1 for gram in grams if gram not in self.ngram_counts)
        return unseen / len(grams)

    def mean_ngram_rarity(self, tokens: Sequence[str]) -> float:
        """Average surprisal of the transitions in a sequence, in nats."""
        grams = ngrams([token for token in tokens if token], self.ngram_n)
        if not grams:
            return 0.0
        total = max(1, self.ngram_total)
        return sum(
            -math.log(max(self.ngram_counts.get(gram, 0), 1) / (total + 1)) for gram in grams
        ) / len(grams)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "counts": dict(self.counts),
            "total_count": self.total_count,
            "max_len": self.max_len,
            "ngram_n": self.ngram_n,
            "ngram_counts": dict(self.ngram_counts),
            "ngram_total": self.ngram_total,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SequenceVocab":
        return cls(
            tokens=list(payload.get("tokens", [])),
            counts=dict(payload.get("counts", {})),
            total_count=int(payload.get("total_count", 0)),
            max_len=int(payload.get("max_len", settings.sequence_max_len)),
            ngram_n=int(payload.get("ngram_n", settings.sequence_ngram_n)),
            ngram_counts=dict(payload.get("ngram_counts", {})),
            ngram_total=int(payload.get("ngram_total", 0)),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "SequenceVocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def profile_ngram_novelty(tokens: Sequence[str], profile_ngrams: Dict[str, float], n: int = 2) -> float:
    """Share of a sequence's n-grams this *entity* has never produced before.

    The per-entity counterpart to :meth:`SequenceVocab.ngram_novelty`. An entity doing
    something normal-for-the-company but new-for-itself is the interesting case, and only this
    comparison catches it.
    """
    grams = ngrams([token for token in tokens if token], n)
    if not grams:
        return 0.0
    if not profile_ngrams:
        return 1.0  # nothing known about this entity yet: everything is new
    unseen = sum(1 for gram in grams if gram not in profile_ngrams)
    return unseen / len(grams)


__all__ = [
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "RESERVED_TOKENS",
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "NGRAM_SEPARATOR",
    "ngrams",
    "SequenceVocab",
    "profile_ngram_novelty",
]
