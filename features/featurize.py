"""The single shared feature entry point.

There is exactly one function that turns an event into model input:
:meth:`FeaturePipeline.featurize`. Training calls it, the online scorer calls it, and the batch
helper :meth:`FeaturePipeline.featurize_events` calls it in a loop rather than reimplementing it
in vectorized form.

That is a deliberate performance sacrifice. A pandas implementation would be much faster
offline, but it would be a *second* implementation, and the two would drift. Train/serve skew is
the most common way a working model becomes a broken deployment, and it is close to invisible in
metrics: offline numbers stay excellent while production quietly degrades. One slow correct path
beats two fast ones that disagree.

The other invariant enforced here: **state is updated after featurizing, never before**. An
event must never contribute to the baseline it is being judged against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from common.artifacts import artifact_path
from common.config import settings
from common.models import Event
from features.cohorts import CohortModel
from features.encoders import EncoderBundle, NumericScaler
from features.entity_window import (
    BehaviorProfile,
    EntityState,
    ProfileAccumulator,
    ProfileStore,
    ResolvedProfile,
    StateStore,
)
from features.event_features import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    CorpusStats,
    categorical_values,
    compute_event_features,
)
from features.session_features import compute_session_features
from features.sequences import SequenceVocab

logger = logging.getLogger(__name__)

# Artifact filenames written by training/build_baselines.py and read by serving.
ENCODERS_FILE = "encoders.json"
VOCAB_FILE = "sequence_vocab.json"
PROFILES_FILE = "entity_profiles.json"
COHORTS_FILE = "cohorts.json"
CORPUS_FILE = "corpus_stats.json"
FEATURE_SPACE_FILE = "feature_space.json"


@dataclass
class FeatureVector:
    """One event's model-ready representation, plus everything needed to explain it."""

    entity_id: str
    timestamp: datetime
    event_id: Optional[str]

    #: Scaled numeric features followed by categorical codes, ordered by ``names``.
    values: np.ndarray
    names: Tuple[str, ...]

    #: Unscaled values keyed by name. Explanations quote these, because "hour_likelihood was
    #: -2.3 standard deviations" means nothing to an analyst while "0.4% of their logins are at
    #: this hour" does.
    raw: Dict[str, float] = field(default_factory=dict)

    #: Fixed-length token ids for the sequence model (Phase 4).
    sequence_ids: List[int] = field(default_factory=list)
    sequence_tokens: List[str] = field(default_factory=list)

    cold_start: bool = False
    cohort: Optional[int] = None
    profile_confidence: float = 0.0
    profile_source: str = "global"
    novelty: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        """Scaled features keyed by name."""
        return {name: float(value) for name, value in zip(self.names, self.values)}

    def raw_value(self, name: str, default: float = 0.0) -> float:
        return float(self.raw.get(name, default))


class FeaturePipeline:
    """Holds every fitted transform and produces feature vectors.

    Constructed empty for fitting (``training/build_baselines.py``), or loaded from
    ``artifacts/`` for serving. Serving never fits anything.
    """

    def __init__(
        self,
        encoders: Optional[EncoderBundle] = None,
        vocab: Optional[SequenceVocab] = None,
        profiles: Optional[ProfileStore] = None,
        cohorts: Optional[CohortModel] = None,
        corpus: Optional[CorpusStats] = None,
    ) -> None:
        self.encoders = encoders or EncoderBundle(
            numeric_names=list(NUMERIC_FEATURE_NAMES),
            categorical_names=list(CATEGORICAL_FEATURE_NAMES),
        )
        self.vocab = vocab or SequenceVocab()
        self.profiles = profiles or ProfileStore()
        self.cohorts = cohorts or CohortModel()
        self.corpus = corpus or CorpusStats()

        # Per-entity live profiles built while replaying. Offline this accumulates the training
        # split; online it captures activity since the process started, so an entity that is new
        # to the persisted artifacts still develops its own baseline as it goes.
        self._accumulators: Dict[str, ProfileAccumulator] = {}
        self.states = StateStore()

    # ------------------------------------------------------------------ #
    # Feature space
    # ------------------------------------------------------------------ #

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Ordered feature names: numeric block, then categorical codes."""
        return tuple(self.encoders.feature_names)

    @property
    def numeric_names(self) -> Tuple[str, ...]:
        return tuple(self.encoders.numeric_names)

    @property
    def categorical_indices(self) -> List[int]:
        """Column positions of categorical codes, for LightGBM."""
        return self.encoders.categorical_indices

    @property
    def n_features(self) -> int:
        return len(self.encoders.feature_names)

    @property
    def is_fitted(self) -> bool:
        """Whether this pipeline can produce scaled vectors."""
        return self.encoders.scaler is not None

    # ------------------------------------------------------------------ #
    # Live profile bookkeeping
    # ------------------------------------------------------------------ #

    def live_profile(self, entity_id: str) -> Optional[BehaviorProfile]:
        """Profile built from events this pipeline has already seen, if any.

        Uses the cached build: finalizing a profile is expensive relative to the marginal
        information one more event adds. See
        :data:`features.entity_window.LIVE_PROFILE_REFRESH_EVENTS`.
        """
        accumulator = self._accumulators.get(entity_id)
        if accumulator is None or accumulator.event_count == 0:
            return None
        return accumulator.build_cached()

    def reset_state(self) -> None:
        """Forget all rolling windows and live profiles.

        Called between independent replays so one pass cannot leak context into the next.
        """
        self._accumulators.clear()
        self.states.clear()

    def _resolve(self, event: Event, use_live: bool) -> ResolvedProfile:
        """Pick the baseline for this event, with cold-start shrinkage applied."""
        live = self.live_profile(event.entity_id) if use_live else None
        if live is not None and live.cohort is None:
            live.cohort = self.cohorts.assign_for_new_entity(event.entity_type.value, live)

        resolved = self.profiles.resolve(
            event.entity_id, event.entity_type.value, live_profile=live
        )

        if resolved.cohort is None:
            # No stored cohort: derive one so the prior is at least type-appropriate rather than
            # the global average, which resembles no real entity.
            cohort = self.cohorts.assign_for_new_entity(event.entity_type.value, live)
            if cohort is not None:
                resolved = ResolvedProfile(
                    profile=resolved.profile,
                    cold_start=resolved.cold_start,
                    confidence=resolved.confidence,
                    cohort=cohort,
                    source=resolved.source,
                )
        return resolved

    # ------------------------------------------------------------------ #
    # The one entry point
    # ------------------------------------------------------------------ #

    def featurize(
        self,
        event: Event,
        update_state: bool = True,
        use_live_profile: bool = True,
    ) -> FeatureVector:
        """Turn one event into a feature vector.

        Used identically offline and online. Ground truth on the event is ignored entirely --
        ``label``, ``campaign_id`` and ``stage`` are never read here, so a labeled and an
        unlabeled copy of the same event produce identical output.

        Parameters
        ----------
        update_state:
            Fold the event into the rolling window and live profile afterwards. Set False to
            score a hypothetical event without disturbing history -- which is exactly what the
            counterfactual search in Phase 6 needs.
        """
        state: EntityState = self.states.get(event.entity_id)
        resolved = self._resolve(event, use_live=use_live_profile)

        # --- raw features (state is still "as of before this event") ---
        raw = compute_event_features(
            event, resolved, state, vocab=self.vocab, corpus=self.corpus
        )
        raw.update(compute_session_features(event, state))

        numeric_names = self.encoders.numeric_names or NUMERIC_FEATURE_NAMES
        numeric = np.asarray(
            [float(raw.get(name, 0.0)) for name in numeric_names], dtype=float
        )

        scaled = (
            self.encoders.scaler.transform(numeric)
            if self.encoders.scaler is not None
            else numeric
        )

        # --- categorical codes ---
        categorical_raw = categorical_values(event, resolved.cohort)
        codes, novelty = self.encoders.encode_categoricals(categorical_raw)

        values = np.concatenate([scaled, np.asarray(codes, dtype=float)])

        # --- sequence ---
        tokens = [token for token in event.command_sequence if token]
        sequence_ids = self.vocab.encode(tokens) if self.vocab.size else []

        vector = FeatureVector(
            entity_id=event.entity_id,
            timestamp=event.timestamp,
            event_id=event.event_id,
            values=values,
            names=self.feature_names,
            raw=raw,
            sequence_ids=sequence_ids,
            sequence_tokens=tokens,
            cold_start=resolved.cold_start,
            cohort=resolved.cohort,
            profile_confidence=resolved.confidence,
            profile_source=resolved.source,
            novelty=novelty,
        )

        if update_state:
            self._observe(event)

        return vector

    def _observe(self, event: Event) -> None:
        """Fold an event into the rolling window and the live profile."""
        accumulator = self._accumulators.get(event.entity_id)
        if accumulator is None:
            accumulator = ProfileAccumulator(event.entity_id, event.entity_type.value)
            self._accumulators[event.entity_id] = accumulator
        accumulator.update(event)
        self.states.get(event.entity_id).update(event)

    # ------------------------------------------------------------------ #
    # Batch replay
    # ------------------------------------------------------------------ #

    def featurize_events(
        self,
        events: Sequence[Event],
        reset: bool = True,
        use_live_profile: bool = True,
    ) -> List[FeatureVector]:
        """Featurize a sequence of events in time order.

        Events are sorted by timestamp first: rolling-window features are meaningless if events
        arrive out of order, and a caller passing an unsorted frame would silently get wrong
        values rather than an error.
        """
        if reset:
            self.reset_state()

        ordered = sorted(events, key=lambda event: (event.timestamp, event.event_id))
        return [
            self.featurize(event, update_state=True, use_live_profile=use_live_profile)
            for event in ordered
        ]

    @staticmethod
    def to_matrix(vectors: Sequence[FeatureVector]) -> np.ndarray:
        """Stack feature vectors into a 2-D matrix."""
        if not vectors:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([vector.values for vector in vectors])

    @staticmethod
    def raw_matrix(vectors: Sequence[FeatureVector], names: Sequence[str]) -> np.ndarray:
        """Stack the *unscaled* numeric features, for fitting the scaler and entity baselines."""
        if not vectors:
            return np.zeros((0, len(names)), dtype=float)
        return np.vstack(
            [
                np.asarray([vector.raw.get(name, 0.0) for name in names], dtype=float)
                for vector in vectors
            ]
        )

    @staticmethod
    def sequence_matrix(vectors: Sequence[FeatureVector]) -> np.ndarray:
        """Stack encoded sequences into an integer matrix for the sequence model."""
        if not vectors or not vectors[0].sequence_ids:
            return np.zeros((len(vectors), 0), dtype=np.int64)
        width = max(len(vector.sequence_ids) for vector in vectors)
        matrix = np.zeros((len(vectors), width), dtype=np.int64)
        for row, vector in enumerate(vectors):
            ids = vector.sequence_ids
            matrix[row, width - len(ids) :] = ids  # keep left padding
        return matrix

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, directory: Optional[Path] = None) -> Dict[str, Path]:
        """Write every fitted component to ``artifacts/`` and return the paths."""
        target = Path(directory) if directory else Path(settings.artifacts_dir)
        target.mkdir(parents=True, exist_ok=True)

        paths = {
            "encoders": self.encoders.save(target / ENCODERS_FILE),
            "sequence_vocab": self.vocab.save(target / VOCAB_FILE),
            "entity_profiles": Path(self.profiles.save(target / PROFILES_FILE)),
            "cohorts": self.cohorts.save(target / COHORTS_FILE),
        }

        corpus_path = target / CORPUS_FILE
        with corpus_path.open("w", encoding="utf-8") as handle:
            json.dump(self.corpus.to_dict(), handle, indent=2)
            handle.write("\n")
        paths["corpus_stats"] = corpus_path

        # A standalone description of the feature space, so a consumer can check column order
        # without loading the encoders.
        space_path = target / FEATURE_SPACE_FILE
        with space_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "feature_names": list(self.feature_names),
                    "numeric_names": list(self.numeric_names),
                    "categorical_names": list(self.encoders.categorical_names),
                    "categorical_indices": self.categorical_indices,
                    "n_features": self.n_features,
                    "sequence_max_len": self.vocab.max_len,
                    "sequence_vocab_size": self.vocab.size,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        paths["feature_space"] = space_path

        return paths

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "FeaturePipeline":
        """Load a fitted pipeline from ``artifacts/``.

        Raises
        ------
        FileNotFoundError
            If the encoders are missing, with the command needed to build them. Serving must
            fail loudly at startup rather than silently score with unfitted transforms.
        """
        target = Path(directory) if directory else Path(settings.artifacts_dir)

        encoders_path = target / ENCODERS_FILE
        if not encoders_path.exists():
            raise FileNotFoundError(
                f"No fitted encoders at {encoders_path}. "
                "Run: python -m training.build_baselines"
            )

        encoders = EncoderBundle.load(encoders_path)

        vocab_path = target / VOCAB_FILE
        vocab = SequenceVocab.load(vocab_path) if vocab_path.exists() else SequenceVocab()

        profiles_path = target / PROFILES_FILE
        profiles = ProfileStore.load(profiles_path) if profiles_path.exists() else ProfileStore()

        cohorts_path = target / COHORTS_FILE
        cohorts = CohortModel.load(cohorts_path) if cohorts_path.exists() else CohortModel()

        corpus_path = target / CORPUS_FILE
        if corpus_path.exists():
            with corpus_path.open("r", encoding="utf-8") as handle:
                corpus = CorpusStats.from_dict(json.load(handle))
        else:
            corpus = CorpusStats()

        pipeline = cls(
            encoders=encoders,
            vocab=vocab,
            profiles=profiles,
            cohorts=cohorts,
            corpus=corpus,
        )
        logger.info(
            "Loaded feature pipeline: %d features, %d profiles, %d cohorts, vocab %d",
            pipeline.n_features,
            len(profiles.profiles),
            cohorts.n_cohorts,
            vocab.size,
        )
        return pipeline


def load_pipeline(directory: Optional[Path] = None) -> FeaturePipeline:
    """Convenience loader used by the serving and training entry points."""
    return FeaturePipeline.load(directory)


__all__ = [
    "ENCODERS_FILE",
    "VOCAB_FILE",
    "PROFILES_FILE",
    "COHORTS_FILE",
    "CORPUS_FILE",
    "FEATURE_SPACE_FILE",
    "FeatureVector",
    "FeaturePipeline",
    "load_pipeline",
]
