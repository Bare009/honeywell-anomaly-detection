"""Feature pipeline tests.

The headline property here is **train/serve parity**: the features computed while training must
be bit-identical to those computed while scoring. Train/serve skew is close to invisible in
metrics -- offline numbers stay excellent while production silently degrades -- so it is asserted
directly rather than assumed from the fact that both paths call the same function.

Also guarded: encoder round-trips, the cold-start fallback producing usable vectors for entities
the system has never seen, geo math against known real-world distances, and the invariant that a
label can never influence a feature.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from common.config import settings
from common.models import (
    AnomalyType,
    AuthMethod,
    DeviceFingerprint,
    EntityType,
    Event,
    GeoLocation,
)
from features.cohorts import (
    SUMMARY_FEATURE_NAMES,
    CohortModel,
    behavior_summary,
    build_cohort_priors,
    build_global_prior,
)
from features.encoders import (
    MIN_STD,
    UNSEEN_CODE,
    CategoricalEncoder,
    EncoderBundle,
    NumericScaler,
)
from features.entity_window import (
    LIVE_PROFILE_REFRESH_EVENTS,
    PRIOR_STRENGTH_SESSIONS,
    BehaviorProfile,
    EntityState,
    ProfileAccumulator,
    ProfileStore,
    RunningStat,
    ip_prefix,
)
from features.event_features import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    CorpusStats,
    categorical_values,
    compute_event_features,
)
from features.featurize import FeaturePipeline, FeatureVector
from features.geo import (
    MIN_ELAPSED_SECONDS,
    centroid,
    elapsed_hours,
    geo_velocity_kmh,
    haversine_km,
    is_impossible_travel,
    max_distance_from_km,
)
from features.session_features import (
    SESSION_FEATURE_NAMES,
    compute_session_features,
    group_by_session,
    session_command_sequence,
    summarize_session,
)
from features.sequences import (
    BOS_ID,
    PAD_ID,
    UNK_ID,
    UNK_TOKEN,
    SequenceVocab,
    ngrams,
    profile_ngram_novelty,
)

# Real-world reference coordinates, for asserting the geo math against known distances.
LONDON = (51.5074, -0.1278)
PARIS = (48.8566, 2.3522)
BENGALURU = (12.9716, 77.5946)
SAO_PAULO = (-23.5505, -46.6333)
SINGAPORE = (1.3521, 103.8198)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_events() -> List[Event]:
    """A small generated dataset, replayed as Event objects in time order."""
    from data_generator.generate import generate_dataset
    from data_generator.profiles import GeneratorConfig

    config = GeneratorConfig(seed=42, n_entities=40, days=12, target_anomaly_rate=0.02)
    events_frame, _, _, _, _ = generate_dataset(config)

    from data_generator.generate import dataframe_to_events

    ordered = events_frame.sort_values(["timestamp", "event_id"])
    return dataframe_to_events(ordered)


@pytest.fixture(scope="module")
def fitted_pipeline(small_events: List[Event]) -> FeaturePipeline:
    """A fully fitted pipeline, built the same way ``training.build_baselines`` builds it."""
    from training.build_baselines import fit_encoders, fit_vocabulary, streaming_pass

    encoders = fit_encoders(small_events)
    vocab = fit_vocabulary(small_events)
    corpus = CorpusStats.fit(small_events)

    pipeline = FeaturePipeline(encoders=encoders, vocab=vocab, corpus=corpus)
    vectors, accumulators = streaming_pass(pipeline, small_events)

    profiles = {entity_id: acc.build() for entity_id, acc in accumulators.items()}
    encoders.scaler = NumericScaler.fit(
        list(NUMERIC_FEATURE_NAMES),
        FeaturePipeline.raw_matrix(vectors, list(NUMERIC_FEATURE_NAMES)),
    )

    cohort_model = CohortModel.fit(list(profiles.values()))
    assignments: Dict[str, int] = {}
    for entity_id, profile in profiles.items():
        cohort = cohort_model.assign(profile)
        if cohort is not None:
            profile.cohort = cohort
            assignments[entity_id] = cohort

    pipeline.profiles = ProfileStore(
        profiles=profiles,
        cohort_priors=build_cohort_priors(accumulators, assignments),
        global_prior=build_global_prior(accumulators),
        type_cohorts=cohort_model.type_cohorts,
    )
    pipeline.cohorts = cohort_model
    pipeline.encoders = encoders
    return pipeline


@pytest.fixture
def replay_events(small_events: List[Event]) -> List[Event]:
    """A slice of one busy entity's events plus some others, for replay parity checks."""
    counts: Dict[str, int] = {}
    for event in small_events:
        counts[event.entity_id] = counts.get(event.entity_id, 0) + 1
    busiest = max(counts.items(), key=lambda item: item[1])[0]

    selected = [event for event in small_events if event.entity_id == busiest][:120]
    others = [event for event in small_events if event.entity_id != busiest][:180]
    combined = selected + others
    combined.sort(key=lambda event: (event.timestamp, event.event_id))
    return combined


def make_event(
    entity_id: str = "user_0001",
    entity_type: EntityType = EntityType.USER,
    when: datetime = datetime(2026, 3, 2, 10, 15, tzinfo=timezone.utc),
    lat: float = BENGALURU[0],
    lon: float = BENGALURU[1],
    country: str = "India",
    resource: str = "/portal/home",
    auth: AuthMethod = AuthMethod.PASSWORD,
    success: bool = True,
    mac: str = "02:aa:bb:cc:dd:ee",
    device_os: str = "Windows 11 22H2",
    protocol: str = "https",
    ip: str = "10.20.30.40",
    bytes_out: float = 4096.0,
    duration: float = 300.0,
    commands: List[str] | None = None,
    session_id: str = "ses_x_0001",
) -> Event:
    """Construct a well-formed event with sensible defaults."""
    return Event(
        entity_id=entity_id,
        entity_type=entity_type,
        timestamp=when,
        source_ip=ip,
        geo=GeoLocation(country=country, city="City", lat=lat, lon=lon),
        resource_accessed=resource,
        auth_method=auth,
        auth_success=success,
        session_id=session_id,
        session_duration=duration,
        command_sequence=commands if commands is not None else ["login", "view_document"],
        device_fingerprint=DeviceFingerprint(os=device_os, mac=mac, protocol=protocol),
        bytes_out=bytes_out,
        bytes_in=1024.0,
    )


# --------------------------------------------------------------------------- #
# Geo
# --------------------------------------------------------------------------- #


class TestGeo:
    """Geo math checked against known real-world distances, not just self-consistency."""

    def test_haversine_london_paris(self) -> None:
        distance = haversine_km(*LONDON, *PARIS)
        assert 330.0 < distance < 360.0, f"London-Paris computed as {distance:.0f} km"

    def test_haversine_bengaluru_sao_paulo(self) -> None:
        distance = haversine_km(*BENGALURU, *SAO_PAULO)
        assert 14_000.0 < distance < 15_500.0, f"computed {distance:.0f} km"

    def test_haversine_is_symmetric(self) -> None:
        assert haversine_km(*LONDON, *PARIS) == pytest.approx(haversine_km(*PARIS, *LONDON))

    def test_haversine_zero_distance(self) -> None:
        assert haversine_km(*LONDON, *LONDON) == pytest.approx(0.0, abs=1e-9)

    def test_haversine_antipodal(self) -> None:
        """Half the Earth's circumference, and no domain error from floating-point drift."""
        assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(20_015.0, rel=0.01)

    def test_elapsed_hours_is_floored(self) -> None:
        """Without a floor, two events in the same second imply infinite velocity."""
        moment = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        assert elapsed_hours(moment, moment) == pytest.approx(MIN_ELAPSED_SECONDS / 3600.0)

    def test_elapsed_hours_handles_reversed_order(self) -> None:
        early = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        late = early + timedelta(hours=3)
        assert elapsed_hours(late, early) == pytest.approx(3.0)

    def test_velocity_plausible_flight(self) -> None:
        """London to Paris in two hours is an ordinary journey, not a red flag."""
        start = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)
        velocity = geo_velocity_kmh(*LONDON, start, *PARIS, start + timedelta(hours=2))
        assert velocity < settings.impossible_travel_kmh

    def test_velocity_impossible_hop(self) -> None:
        start = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)
        velocity = geo_velocity_kmh(
            *BENGALURU, start, *SAO_PAULO, start + timedelta(minutes=20)
        )
        assert velocity > settings.impossible_travel_kmh

    def test_impossible_travel_fires_on_long_fast_hop(self) -> None:
        start = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)
        fired, velocity = is_impossible_travel(
            *BENGALURU, start, *SAO_PAULO, start + timedelta(minutes=20)
        )
        assert fired is True
        assert velocity > settings.impossible_travel_kmh

    def test_impossible_travel_ignores_short_hops(self) -> None:
        """Geolocation jitter over a few km can imply silly speeds; distance floor prevents that."""
        start = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)
        nearby = (BENGALURU[0] + 0.4, BENGALURU[1] + 0.4)
        fired, _ = is_impossible_travel(
            *BENGALURU, start, *nearby, start + timedelta(seconds=30)
        )
        assert fired is False

    def test_impossible_travel_allows_real_travel(self) -> None:
        start = datetime(2026, 3, 2, 8, 0, tzinfo=timezone.utc)
        fired, _ = is_impossible_travel(
            *BENGALURU, start, *SINGAPORE, start + timedelta(hours=6)
        )
        assert fired is False

    def test_centroid_of_single_point(self) -> None:
        result = centroid([LONDON])
        assert result is not None
        assert result[0] == pytest.approx(LONDON[0], abs=1e-6)
        assert result[1] == pytest.approx(LONDON[1], abs=1e-6)

    def test_centroid_is_between_points(self) -> None:
        result = centroid([LONDON, PARIS])
        assert result is not None
        assert PARIS[0] < result[0] < LONDON[0]

    def test_centroid_handles_antimeridian(self) -> None:
        """Averaging degrees naively would place this in the middle of the wrong ocean."""
        result = centroid([(0.0, 179.0), (0.0, -179.0)])
        assert result is not None
        assert abs(result[1]) > 170.0

    def test_centroid_of_empty(self) -> None:
        assert centroid([]) is None

    def test_max_distance_from(self) -> None:
        spread = max_distance_from_km([LONDON, PARIS, BENGALURU], *LONDON)
        assert spread == pytest.approx(haversine_km(*LONDON, *BENGALURU))

    def test_max_distance_from_empty(self) -> None:
        assert max_distance_from_km([], *LONDON) == 0.0


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


class TestCategoricalEncoder:
    """Unseen categories are signal, so they must encode predictably rather than raise."""

    def test_codes_start_at_one(self) -> None:
        encoder = CategoricalEncoder.fit("country", ["India", "India", "Singapore"])
        code, novel = encoder.transform("India")
        assert code >= 1
        assert novel is False

    def test_unseen_value_gets_reserved_code_and_flag(self) -> None:
        encoder = CategoricalEncoder.fit("country", ["India"])
        code, novel = encoder.transform("Brazil")
        assert code == UNSEEN_CODE
        assert novel is True

    def test_none_is_treated_as_unseen(self) -> None:
        encoder = CategoricalEncoder.fit("country", ["India"])
        assert encoder.transform(None) == (UNSEEN_CODE, True)

    def test_nan_is_treated_as_unseen(self) -> None:
        encoder = CategoricalEncoder.fit("country", ["India"])
        assert encoder.transform(float("nan")) == (UNSEEN_CODE, True)

    def test_whitespace_is_normalized(self) -> None:
        encoder = CategoricalEncoder.fit("country", ["India"])
        assert encoder.transform("  India  ")[1] is False

    def test_frequency_ordering_is_deterministic(self) -> None:
        """Two fits on the same data must assign identical codes, or artifacts drift."""
        values = ["a", "b", "b", "c", "c", "c"]
        first = CategoricalEncoder.fit("x", values)
        second = CategoricalEncoder.fit("x", list(reversed(values)))
        assert first.categories == second.categories

    def test_most_frequent_gets_lowest_code(self) -> None:
        encoder = CategoricalEncoder.fit("x", ["rare", "common", "common", "common"])
        assert encoder.categories[0] == "common"

    def test_max_categories_truncates(self) -> None:
        encoder = CategoricalEncoder.fit(
            "x", ["a"] * 5 + ["b"] * 4 + ["c"] * 3, max_categories=2
        )
        assert encoder.categories == ["a", "b"]
        assert encoder.transform("c") == (UNSEEN_CODE, True)

    def test_min_count_excludes_rare_values(self) -> None:
        encoder = CategoricalEncoder.fit("x", ["a", "a", "b"], min_count=2)
        assert "b" not in encoder.categories

    def test_inverse_round_trip(self) -> None:
        encoder = CategoricalEncoder.fit("x", ["alpha", "beta"])
        code, _ = encoder.transform("beta")
        assert encoder.inverse(code) == "beta"

    def test_inverse_of_unseen_code(self) -> None:
        encoder = CategoricalEncoder.fit("x", ["alpha"])
        assert encoder.inverse(UNSEEN_CODE) is None

    def test_cardinality_includes_unseen_slot(self) -> None:
        encoder = CategoricalEncoder.fit("x", ["a", "b"])
        assert encoder.cardinality == 3

    def test_dict_round_trip(self) -> None:
        encoder = CategoricalEncoder.fit("x", ["a", "b", "b"])
        restored = CategoricalEncoder.from_dict(encoder.to_dict())
        assert restored.categories == encoder.categories
        assert restored.transform("b") == encoder.transform("b")


class TestNumericScaler:
    """Scaling must be stable, JSON-persistable and immune to degenerate columns."""

    def test_standardizes_to_zero_mean(self) -> None:
        matrix = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaler = NumericScaler.fit(["a", "b"], matrix)
        scaled = scaler.transform(matrix)
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)

    def test_unit_variance(self) -> None:
        matrix = np.random.default_rng(0).normal(5.0, 3.0, size=(200, 2))
        scaler = NumericScaler.fit(["a", "b"], matrix)
        assert np.allclose(scaler.transform(matrix).std(axis=0), 1.0, atol=0.02)

    def test_constant_column_becomes_zero(self) -> None:
        """A feature that never varies carries no information; it must not divide by zero."""
        matrix = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
        scaler = NumericScaler.fit(["a", "b"], matrix)
        scaled = scaler.transform(matrix)
        assert np.allclose(scaled[:, 1], 0.0)
        assert all(std >= MIN_STD for std in scaler.stds)

    def test_single_vector_transform(self) -> None:
        scaler = NumericScaler.fit(["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
        scaled = scaler.transform(np.array([2.0, 3.0]))
        assert scaled.shape == (2,)

    def test_nan_input_is_neutralized(self) -> None:
        scaler = NumericScaler.fit(["a"], np.array([[1.0], [2.0], [3.0]]))
        assert np.isfinite(scaler.transform(np.array([float("nan")]))).all()

    def test_extreme_values_are_clipped(self) -> None:
        """One pathological event must not swamp every other feature in the vector."""
        scaler = NumericScaler.fit(["a"], np.array([[1.0], [2.0], [3.0]]))
        assert abs(float(scaler.transform(np.array([1e12]))[0])) <= 12.0

    def test_wrong_width_raises(self) -> None:
        scaler = NumericScaler.fit(["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="expected 2 features"):
            scaler.transform(np.array([1.0, 2.0, 3.0]))

    def test_name_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            NumericScaler.fit(["a"], np.array([[1.0, 2.0]]))

    def test_inverse_transform_recovers_values(self) -> None:
        matrix = np.array([[1.0], [5.0], [9.0]])
        scaler = NumericScaler.fit(["a"], matrix)
        recovered = scaler.inverse_transform(scaler.transform(np.array([5.0])))
        assert recovered[0] == pytest.approx(5.0)

    def test_dict_round_trip(self) -> None:
        scaler = NumericScaler.fit(["a", "b"], np.array([[1.0, 2.0], [3.0, 8.0]]))
        restored = NumericScaler.from_dict(scaler.to_dict())
        probe = np.array([2.0, 5.0])
        assert np.allclose(restored.transform(probe), scaler.transform(probe))


class TestEncoderBundle:
    """The saveable container the serving path loads."""

    def test_feature_names_order(self) -> None:
        bundle = EncoderBundle(numeric_names=["a", "b"], categorical_names=["c"])
        assert bundle.feature_names == ["a", "b", "c_code"]

    def test_categorical_indices_follow_numeric_block(self) -> None:
        bundle = EncoderBundle(numeric_names=["a", "b", "c"], categorical_names=["x", "y"])
        assert bundle.categorical_indices == [3, 4]

    def test_encode_categoricals_reports_novelty(self) -> None:
        bundle = EncoderBundle(
            categorical={"country": CategoricalEncoder.fit("country", ["India"])},
            categorical_names=["country"],
        )
        codes, novelty = bundle.encode_categoricals({"country": "Brazil"})
        assert codes == [UNSEEN_CODE]
        assert novelty["country"] is True

    def test_missing_encoder_is_handled(self) -> None:
        bundle = EncoderBundle(categorical_names=["absent"])
        codes, novelty = bundle.encode_categoricals({"absent": "x"})
        assert codes == [UNSEEN_CODE]
        assert novelty["absent"] is True

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """Artifacts are JSON, not pickle, so they survive a library upgrade."""
        bundle = EncoderBundle(
            categorical={"country": CategoricalEncoder.fit("country", ["India", "Brazil"])},
            scaler=NumericScaler.fit(["a"], np.array([[1.0], [2.0]])),
            numeric_names=["a"],
            categorical_names=["country"],
        )
        path = bundle.save(tmp_path / "encoders.json")
        restored = EncoderBundle.load(path)

        assert restored.feature_names == bundle.feature_names
        assert restored.encode_categoricals({"country": "India"}) == bundle.encode_categoricals(
            {"country": "India"}
        )
        assert restored.scaler is not None
        assert np.allclose(
            restored.scaler.transform(np.array([1.5])),
            bundle.scaler.transform(np.array([1.5])),
        )

    def test_saved_file_is_readable_json(self, tmp_path: Path) -> None:
        bundle = EncoderBundle(numeric_names=["a"], categorical_names=[])
        path = bundle.save(tmp_path / "e.json")
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #


class TestSequences:
    """Vocabulary, fixed-length encoding, and the model-free novelty measures."""

    def test_ngrams_basic(self) -> None:
        assert ngrams(["a", "b", "c"], 2) == ["a>b", "b>c"]

    def test_ngrams_shorter_than_n(self) -> None:
        """Inventing transitions would make short sequences look artificially familiar."""
        assert ngrams(["a"], 2) == []

    def test_ngrams_unigram(self) -> None:
        assert ngrams(["a", "b"], 1) == ["a", "b"]

    def test_reserved_tokens_occupy_first_ids(self) -> None:
        vocab = SequenceVocab.fit([["login", "logout"]] * 5, min_count=1)
        assert vocab.tokens[PAD_ID] == "<pad>"
        assert vocab.tokens[UNK_ID] == UNK_TOKEN
        assert vocab.token_id("<bos>") == BOS_ID

    def test_rare_tokens_map_to_unk(self) -> None:
        vocab = SequenceVocab.fit([["common"] * 5, ["once"]], min_count=3)
        assert vocab.token_id("once") == UNK_ID
        assert vocab.token_id("common") != UNK_ID

    def test_encode_produces_fixed_length(self) -> None:
        vocab = SequenceVocab.fit([["a", "b", "c"]] * 5, min_count=1, max_len=6)
        assert len(vocab.encode(["a", "b"])) == 6

    def test_encode_left_pads(self) -> None:
        vocab = SequenceVocab.fit([["a", "b"]] * 5, min_count=1, max_len=5)
        encoded = vocab.encode(["a"], add_bos=False)
        assert encoded[0] == PAD_ID
        assert encoded[-1] == vocab.token_id("a")

    def test_encode_truncation_keeps_the_tail(self) -> None:
        """The most recent actions predict what comes next; the oldest are expendable."""
        tokens = [f"t{index}" for index in range(10)]
        vocab = SequenceVocab.fit([tokens] * 5, min_count=1, max_len=3)
        encoded = vocab.encode(tokens, add_bos=False)
        assert encoded == [vocab.token_id(name) for name in tokens[-3:]]

    def test_encode_adds_bos(self) -> None:
        vocab = SequenceVocab.fit([["a"]] * 5, min_count=1, max_len=4)
        assert BOS_ID in vocab.encode(["a"], add_bos=True)

    def test_decode_strips_special_tokens(self) -> None:
        vocab = SequenceVocab.fit([["a", "b"]] * 5, min_count=1, max_len=6)
        assert vocab.decode(vocab.encode(["a", "b"])) == ["a", "b"]

    def test_unknown_ratio(self) -> None:
        vocab = SequenceVocab.fit([["a", "b"]] * 5, min_count=1)
        assert vocab.unknown_ratio(["a", "zzz"]) == pytest.approx(0.5)

    def test_unknown_ratio_of_empty(self) -> None:
        vocab = SequenceVocab.fit([["a"]] * 5, min_count=1)
        assert vocab.unknown_ratio([]) == 0.0

    def test_rarity_is_higher_for_rare_tokens(self) -> None:
        vocab = SequenceVocab.fit([["common"] * 20 + ["rare"]], min_count=1)
        assert vocab.token_rarity("rare") > vocab.token_rarity("common")

    def test_unseen_token_rarity_is_finite(self) -> None:
        """An unbounded rarity would dominate the scaled feature vector."""
        vocab = SequenceVocab.fit([["a"] * 10], min_count=1)
        assert math.isfinite(vocab.token_rarity("never_seen"))

    def test_ngram_novelty_detects_new_transitions(self) -> None:
        vocab = SequenceVocab.fit([["login", "view", "logout"]] * 10, min_count=1)
        assert vocab.ngram_novelty(["login", "view"]) == pytest.approx(0.0)
        assert vocab.ngram_novelty(["logout", "login"]) == pytest.approx(1.0)

    def test_profile_ngram_novelty_empty_profile_is_all_new(self) -> None:
        """An entity with no history has genuinely never done any of this."""
        assert profile_ngram_novelty(["a", "b"], {}, 2) == 1.0

    def test_profile_ngram_novelty_known_transition(self) -> None:
        assert profile_ngram_novelty(["a", "b"], {"a>b": 1.0}, 2) == 0.0

    def test_json_round_trip(self, tmp_path: Path) -> None:
        vocab = SequenceVocab.fit([["a", "b", "c"]] * 8, min_count=1)
        restored = SequenceVocab.load(vocab.save(tmp_path / "vocab.json"))
        assert restored.tokens == vocab.tokens
        assert restored.encode(["a", "b"]) == vocab.encode(["a", "b"])
        assert restored.ngram_novelty(["a", "b"]) == vocab.ngram_novelty(["a", "b"])


# --------------------------------------------------------------------------- #
# Running statistics and profiles
# --------------------------------------------------------------------------- #


class TestRunningStat:
    def test_mean_and_std(self) -> None:
        stat = RunningStat()
        for value in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
            stat.update(value)
        assert stat.mean == pytest.approx(5.0)
        assert stat.std == pytest.approx(2.0)

    def test_ignores_non_finite(self) -> None:
        stat = RunningStat()
        stat.update(float("nan"))
        stat.update(float("inf"))
        assert stat.count == 0

    def test_zscore_needs_two_observations(self) -> None:
        stat = RunningStat()
        stat.update(5.0)
        assert stat.zscore(100.0) == 0.0

    def test_zscore_is_bounded_for_constant_data(self) -> None:
        """Near-zero variance would otherwise produce enormous z-scores from trivial noise."""
        stat = RunningStat()
        for _ in range(50):
            stat.update(3.0)
        assert abs(stat.zscore(3.5)) < 50.0

    def test_merge_combines_counts(self) -> None:
        left, right = RunningStat(), RunningStat()
        for value in (1.0, 2.0, 3.0):
            left.update(value)
        for value in (4.0, 5.0, 6.0):
            right.update(value)
        left.merge(right)
        assert left.count == 6
        assert left.mean == pytest.approx(3.5)

    def test_dict_round_trip(self) -> None:
        stat = RunningStat()
        stat.update(2.0)
        restored = RunningStat.from_dict(stat.to_dict())
        assert restored.mean == stat.mean


class TestIpPrefix:
    def test_takes_first_three_octets(self) -> None:
        assert ip_prefix("10.20.30.40") == "10.20.30"

    def test_handles_malformed_address(self) -> None:
        assert ip_prefix("garbage") == "garbage"


class TestBehaviorProfile:
    """The learned baseline, including its cold-start blending behavior."""

    def test_empty_distribution_does_not_report_novelty(self) -> None:
        """No history is not the same as a novel value; conflating them alerts on every first event."""
        profile = BehaviorProfile(entity_id="x")
        assert profile.is_new({}, "anything") is False

    def test_known_value_is_not_new(self) -> None:
        profile = BehaviorProfile(entity_id="x", country_dist={"India": 1.0})
        assert profile.is_new(profile.country_dist, "India") is False

    def test_unknown_value_is_new(self) -> None:
        profile = BehaviorProfile(entity_id="x", country_dist={"India": 1.0})
        assert profile.is_new(profile.country_dist, "Brazil") is True

    def test_hour_likelihood_uniform_when_unknown(self) -> None:
        assert BehaviorProfile(entity_id="x").hour_likelihood(3) == 1.0

    def test_hour_likelihood_scales_around_one(self) -> None:
        hist = [0.0] * 24
        hist[9] = 1.0
        profile = BehaviorProfile(entity_id="x", hour_hist=hist)
        assert profile.hour_likelihood(9) == pytest.approx(24.0)
        assert profile.hour_likelihood(3) == pytest.approx(0.0)

    def test_distance_from_home_without_home(self) -> None:
        assert BehaviorProfile(entity_id="x").distance_from_home_km(*LONDON) == 0.0

    def test_blend_with_full_weight_is_identity(self) -> None:
        profile = BehaviorProfile(entity_id="x", country_dist={"India": 1.0})
        prior = BehaviorProfile(entity_id="p", country_dist={"Brazil": 1.0})
        assert profile.blend_with(prior, 1.0).country_dist == {"India": 1.0}

    def test_blend_with_zero_weight_takes_prior(self) -> None:
        profile = BehaviorProfile(entity_id="x", country_dist={"India": 1.0})
        prior = BehaviorProfile(entity_id="p", country_dist={"Brazil": 1.0})
        blended = profile.blend_with(prior, 0.0)
        assert blended.country_dist.get("Brazil") == pytest.approx(1.0)

    def test_blend_mixes_proportionally(self) -> None:
        profile = BehaviorProfile(entity_id="x", country_dist={"India": 1.0})
        prior = BehaviorProfile(entity_id="p", country_dist={"Brazil": 1.0})
        blended = profile.blend_with(prior, 0.25)
        assert blended.country_dist["India"] == pytest.approx(0.25)
        assert blended.country_dist["Brazil"] == pytest.approx(0.75)

    def test_blend_marks_cold_start(self) -> None:
        blended = BehaviorProfile(entity_id="x").blend_with(BehaviorProfile(entity_id="p"), 0.4)
        assert blended.cold_start is True
        assert blended.confidence == pytest.approx(0.4)

    def test_blend_inherits_prior_home_when_unknown(self) -> None:
        prior = BehaviorProfile(entity_id="p", home_lat=LONDON[0], home_lon=LONDON[1])
        blended = BehaviorProfile(entity_id="x").blend_with(prior, 0.3)
        assert blended.home_lat == pytest.approx(LONDON[0])

    def test_blend_borrows_variance_for_thin_history(self) -> None:
        """A handful of observations gives unusable variance, so z-scores need the prior's scale."""
        thin = BehaviorProfile(entity_id="x")
        thin.bytes_out_log.update(5.0)
        prior = BehaviorProfile(entity_id="p")
        for value in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
            prior.bytes_out_log.update(value)

        blended = thin.blend_with(prior, 0.1)
        assert blended.bytes_out_log.count > thin.bytes_out_log.count

    def test_json_round_trip(self) -> None:
        profile = BehaviorProfile(
            entity_id="user_1",
            entity_type="user",
            cohort=3,
            session_count=20,
            event_count=200,
            first_seen=datetime(2026, 1, 5, tzinfo=timezone.utc),
            last_seen=datetime(2026, 2, 1, tzinfo=timezone.utc),
            country_dist={"India": 0.9, "Singapore": 0.1},
            home_lat=BENGALURU[0],
            home_lon=BENGALURU[1],
            feature_names=["a"],
            feature_means=[1.0],
            feature_stds=[0.5],
        )
        restored = BehaviorProfile.from_dict(profile.to_dict())
        assert restored.entity_id == "user_1"
        assert restored.cohort == 3
        assert restored.country_dist["India"] == pytest.approx(0.9)
        assert restored.first_seen == profile.first_seen
        assert restored.feature_means == [1.0]


class TestProfileAccumulator:
    """Streaming profile construction."""

    def test_accumulates_events(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(5):
            accumulator.update(make_event(when=base + timedelta(hours=index)))
        profile = accumulator.build()
        assert profile.event_count == 5
        assert profile.session_count == 1
        assert profile.country_dist["India"] == pytest.approx(1.0)

    def test_hour_histogram_normalizes(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(4):
            accumulator.update(make_event(when=base + timedelta(hours=index)))
        assert sum(accumulator.build().hour_hist) == pytest.approx(1.0)

    def test_auth_failure_rate(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(4):
            accumulator.update(
                make_event(when=base + timedelta(minutes=index), success=index != 0)
            )
        assert accumulator.build().auth_failure_rate == pytest.approx(0.25)

    def test_cold_start_flag_reflects_session_count(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        accumulator.update(make_event())
        assert accumulator.build().cold_start is True

    def test_merge_pools_two_entities(self) -> None:
        left = ProfileAccumulator("a", "user")
        right = ProfileAccumulator("b", "user")
        left.update(make_event(entity_id="a", country="India"))
        right.update(make_event(entity_id="b", country="Singapore", session_id="ses_b"))
        left.merge(right)
        profile = left.build()
        assert profile.event_count == 2
        assert set(profile.country_dist) == {"India", "Singapore"}

    def test_build_cached_reuses_within_refresh_window(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        accumulator.update(make_event(when=base))
        first = accumulator.build_cached()
        accumulator.update(make_event(when=base + timedelta(minutes=1)))
        assert accumulator.build_cached() is first

    def test_build_cached_refreshes_after_enough_events(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        accumulator.update(make_event(when=base))
        first = accumulator.build_cached()
        for index in range(LIVE_PROFILE_REFRESH_EVENTS + 1):
            accumulator.update(make_event(when=base + timedelta(minutes=index + 2)))
        assert accumulator.build_cached() is not first

    def test_cached_profile_is_never_newer_than_reality(self) -> None:
        """Caching may serve a stale baseline, but never one containing future events."""
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(5):
            accumulator.update(make_event(when=base + timedelta(minutes=index)))
        cached = accumulator.build_cached()
        assert cached.event_count <= accumulator.event_count


# --------------------------------------------------------------------------- #
# Rolling window
# --------------------------------------------------------------------------- #


class TestEntityState:
    """Short-term memory: rate, breadth and velocity features depend on this."""

    def test_starts_empty(self) -> None:
        state = EntityState("user_1")
        assert state.previous is None
        assert state.seconds_since_previous(datetime.now(timezone.utc)) is None

    def test_records_previous_event(self) -> None:
        state = EntityState("user_1")
        event = make_event()
        state.update(event)
        assert state.previous is not None
        assert state.previous.resource == event.resource_accessed

    def test_seconds_since_previous(self) -> None:
        state = EntityState("user_1")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base))
        assert state.seconds_since_previous(base + timedelta(seconds=90)) == pytest.approx(90.0)

    def test_velocity_since_previous(self) -> None:
        state = EntityState("user_1", window_minutes=600)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base, lat=BENGALURU[0], lon=BENGALURU[1]))
        velocity = state.velocity_since_previous(
            SAO_PAULO[0], SAO_PAULO[1], base + timedelta(minutes=15)
        )
        assert velocity is not None and velocity > settings.impossible_travel_kmh

    def test_window_prunes_old_events(self) -> None:
        state = EntityState("user_1", window_minutes=60)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base))
        state.update(make_event(when=base + timedelta(hours=5)))
        assert len(state.window_events(base + timedelta(hours=5))) == 1

    def test_window_keeps_recent_events(self) -> None:
        state = EntityState("user_1", window_minutes=60)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(4):
            state.update(make_event(when=base + timedelta(minutes=index * 5)))
        assert len(state.window_events(base + timedelta(minutes=20))) == 4

    def test_window_length_is_capped(self) -> None:
        state = EntityState("user_1", window_minutes=10_000, max_events=10)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(40):
            state.update(make_event(when=base + timedelta(seconds=index)))
        assert len(state.events) <= 10

    def test_session_tracking(self) -> None:
        state = EntityState("user_1")
        state.update(make_event(session_id="ses_a"))
        assert state.is_known_session("ses_a") is True
        assert state.is_known_session("ses_b") is False

    def test_session_events_filter(self) -> None:
        state = EntityState("user_1")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base, session_id="ses_a"))
        state.update(make_event(when=base + timedelta(minutes=1), session_id="ses_b"))
        assert len(state.session_events("ses_a")) == 1


# --------------------------------------------------------------------------- #
# Cold-start resolution
# --------------------------------------------------------------------------- #


class TestProfileStore:
    """Hierarchical fallback: entity, shrunk toward cohort, else global."""

    @staticmethod
    def _profile(entity_id: str, sessions: int, cohort: int = 1) -> BehaviorProfile:
        return BehaviorProfile(
            entity_id=entity_id,
            entity_type="user",
            cohort=cohort,
            session_count=sessions,
            event_count=sessions * 5,
            country_dist={"India": 1.0},
            cold_start=sessions < settings.entity_history_min_sessions,
        )

    def _store(self) -> ProfileStore:
        return ProfileStore(
            profiles={"known": self._profile("known", 60)},
            cohort_priors={1: self._profile("__cohort_1__", 500)},
            global_prior=self._profile("__global__", 5000, cohort=0),
            type_cohorts={"user": 1},
        )

    def test_established_entity_uses_its_own_profile(self) -> None:
        resolved = self._store().resolve("known", "user")
        assert resolved.source == "entity"
        assert resolved.cold_start is False
        assert resolved.confidence == pytest.approx(1.0)

    def test_unknown_entity_falls_back_to_cohort_prior(self) -> None:
        """The global average resembles no real entity; a cohort prior does."""
        resolved = self._store().resolve("brand_new", "user")
        assert resolved.cold_start is True
        assert resolved.confidence == 0.0
        assert resolved.source == "cohort"
        assert resolved.cohort == 1

    def test_unknown_entity_of_unknown_type_uses_global(self) -> None:
        resolved = self._store().resolve("brand_new", "martian")
        assert resolved.source == "global"

    def test_thin_history_is_blended_not_replaced(self) -> None:
        """Five sessions carry some signal; discarding them would be as wrong as trusting them."""
        store = self._store()
        store.put(self._profile("thin", 4))
        resolved = store.resolve("thin", "user")

        expected = 4 / (4 + PRIOR_STRENGTH_SESSIONS)
        assert resolved.cold_start is True
        assert resolved.confidence == pytest.approx(expected)
        assert resolved.source == "entity+cohort"

    def test_confidence_grows_with_history(self) -> None:
        store = self._store()
        store.put(self._profile("small", 2))
        store.put(self._profile("bigger", 10))
        assert (
            store.resolve("bigger", "user").confidence
            > store.resolve("small", "user").confidence
        )

    def test_live_profile_overrides_thin_stored_profile(self) -> None:
        """An entity new to the artifacts should start building its own baseline immediately."""
        store = self._store()
        live = self._profile("streaming", 30)
        resolved = store.resolve("streaming", "user", live_profile=live)
        assert resolved.cold_start is False
        assert resolved.source == "entity"

    def test_resolve_always_returns_usable_profile(self) -> None:
        """The scorer must never receive None; there is always some baseline to compare against."""
        empty = ProfileStore()
        resolved = empty.resolve("nobody", "user")
        assert isinstance(resolved.profile, BehaviorProfile)
        assert resolved.cold_start is True

    def test_json_round_trip(self, tmp_path: Path) -> None:
        store = self._store()
        restored = ProfileStore.load(store.save(tmp_path / "profiles.json"))
        assert set(restored.profiles) == set(store.profiles)
        assert restored.type_cohorts == store.type_cohorts
        assert restored.resolve("known", "user").source == "entity"


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #


class TestEventFeatures:
    """Every declared feature must actually be produced, and be finite."""

    def test_feature_name_lists_are_unique(self) -> None:
        assert len(NUMERIC_FEATURE_NAMES) == len(set(NUMERIC_FEATURE_NAMES))
        assert len(CATEGORICAL_FEATURE_NAMES) == len(set(CATEGORICAL_FEATURE_NAMES))

    def test_session_features_are_part_of_the_numeric_space(self) -> None:
        assert set(SESSION_FEATURE_NAMES) <= set(NUMERIC_FEATURE_NAMES)

    def test_event_and_session_features_cover_the_numeric_space(self) -> None:
        """A declared-but-never-computed feature would sit silently at zero forever."""
        store = ProfileStore()
        state = EntityState("user_1")
        event = make_event()
        produced = set(
            compute_event_features(event, store.resolve("user_1", "user"), state)
        )
        produced |= set(compute_session_features(event, state))
        assert set(NUMERIC_FEATURE_NAMES) == produced

    def test_all_values_are_finite(self) -> None:
        store = ProfileStore()
        state = EntityState("user_1")
        event = make_event()
        features = compute_event_features(event, store.resolve("user_1", "user"), state)
        for name, value in features.items():
            assert math.isfinite(value), f"{name} is not finite"

    def test_first_event_is_flagged(self) -> None:
        store = ProfileStore()
        state = EntityState("user_1")
        features = compute_event_features(make_event(), store.resolve("user_1", "user"), state)
        assert features["is_first_event"] == 1.0
        assert features["log_geo_velocity_kmh"] == 0.0

    def test_velocity_appears_on_second_event(self) -> None:
        store = ProfileStore()
        state = EntityState("user_1", window_minutes=600)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base, lat=BENGALURU[0], lon=BENGALURU[1]))

        features = compute_event_features(
            make_event(
                when=base + timedelta(minutes=20), lat=SAO_PAULO[0], lon=SAO_PAULO[1]
            ),
            store.resolve("user_1", "user"),
            state,
        )
        assert features["is_first_event"] == 0.0
        assert features["log_geo_velocity_kmh"] > math.log1p(settings.impossible_travel_kmh)

    def test_new_country_flag_against_a_real_profile(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(20):
            accumulator.update(make_event(when=base + timedelta(hours=index), country="India"))

        store = ProfileStore(profiles={"user_1": accumulator.build()})
        state = EntityState("user_1")
        features = compute_event_features(
            make_event(country="Brazil"), store.resolve("user_1", "user"), state
        )
        assert features["is_new_country"] == 1.0
        assert features["country_likelihood"] == pytest.approx(0.0)

    def test_auth_failures_accumulate_in_the_window(self) -> None:
        store = ProfileStore()
        state = EntityState("user_1", window_minutes=60)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(6):
            state.update(make_event(when=base + timedelta(seconds=index * 10), success=False))

        features = compute_event_features(
            make_event(when=base + timedelta(seconds=70), success=False),
            store.resolve("user_1", "user"),
            state,
        )
        assert features["window_auth_failures"] >= settings.brute_force_threshold
        assert features["window_auth_failure_ratio"] == pytest.approx(1.0)

    def test_foreign_resource_ratio_detects_breadth_outside_the_profile(self) -> None:
        accumulator = ProfileAccumulator("user_1", "user")
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        for index in range(20):
            accumulator.update(
                make_event(when=base + timedelta(hours=index), resource="/portal/home")
            )

        store = ProfileStore(profiles={"user_1": accumulator.build()})
        state = EntityState("user_1", window_minutes=60)
        for index in range(4):
            state.update(
                make_event(when=base + timedelta(minutes=index), resource=f"/foreign/{index}")
            )

        features = compute_event_features(
            make_event(when=base + timedelta(minutes=5), resource="/foreign/9"),
            store.resolve("user_1", "user"),
            state,
        )
        assert features["window_foreign_resource_ratio"] > 0.8

    def test_device_change_flag(self) -> None:
        store = ProfileStore()
        state = EntityState("user_1", window_minutes=60)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base, mac="02:11:11:11:11:11"))
        features = compute_event_features(
            make_event(when=base + timedelta(minutes=1), mac="02:99:99:99:99:99"),
            store.resolve("user_1", "user"),
            state,
        )
        assert features["device_changed_from_prev"] == 1.0

    def test_cold_start_features_reflect_resolution(self) -> None:
        store = ProfileStore()
        state = EntityState("brand_new")
        features = compute_event_features(
            make_event(entity_id="brand_new"), store.resolve("brand_new", "user"), state
        )
        assert features["cold_start"] == 1.0
        assert features["profile_confidence"] == pytest.approx(0.0)

    def test_categorical_values_shape(self) -> None:
        values = categorical_values(make_event(), cohort=3)
        assert set(values) == set(CATEGORICAL_FEATURE_NAMES)
        assert values["cohort"] == "3"

    def test_categorical_cohort_unknown(self) -> None:
        assert categorical_values(make_event(), cohort=None)["cohort"] == "unknown"


class TestCorpusStats:
    """Learned resource rarity, instead of a hardcoded list of sensitive paths."""

    def test_fit_computes_frequencies(self) -> None:
        events = [make_event(resource="/common") for _ in range(9)]
        events.append(make_event(resource="/rare", entity_id="user_2"))
        corpus = CorpusStats.fit(events)
        assert corpus.resource_frequency["/common"] == pytest.approx(0.9)
        assert corpus.n_entities == 2

    def test_entity_share_measures_breadth_of_use(self) -> None:
        """A resource used by one entity out of many is structurally sensitive."""
        events = [make_event(entity_id=f"user_{index}", resource="/shared") for index in range(10)]
        events.append(make_event(entity_id="user_0", resource="/private"))
        corpus = CorpusStats.fit(events)
        assert corpus.resource_entity_share["/shared"] == pytest.approx(1.0)
        assert corpus.resource_entity_share["/private"] == pytest.approx(0.1)

    def test_rarity_is_higher_for_rare_values(self) -> None:
        corpus = CorpusStats.fit(
            [make_event(resource="/common") for _ in range(20)]
            + [make_event(resource="/rare", entity_id="u2")]
        )
        assert corpus.rarity(corpus.resource_frequency, "/rare") > corpus.rarity(
            corpus.resource_frequency, "/common"
        )

    def test_unseen_value_rarity_is_finite(self) -> None:
        corpus = CorpusStats.fit([make_event() for _ in range(5)])
        assert math.isfinite(corpus.rarity(corpus.resource_frequency, "/never-seen"))

    def test_dict_round_trip(self) -> None:
        corpus = CorpusStats.fit([make_event() for _ in range(4)])
        restored = CorpusStats.from_dict(corpus.to_dict())
        assert restored.resource_frequency == corpus.resource_frequency


class TestSessionFeatures:
    def test_session_index_counts_prior_events(self) -> None:
        state = EntityState("user_1", window_minutes=60)
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        state.update(make_event(when=base, session_id="ses_a"))
        state.update(make_event(when=base + timedelta(minutes=1), session_id="ses_a"))

        features = compute_session_features(
            make_event(when=base + timedelta(minutes=2), session_id="ses_a"), state
        )
        assert features["session_event_index"] == 2.0
        assert features["is_new_session"] == 0.0

    def test_new_session_flag(self) -> None:
        state = EntityState("user_1")
        assert compute_session_features(make_event(session_id="fresh"), state)["is_new_session"] == 1.0

    def test_summarize_session(self) -> None:
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        events = [
            make_event(when=base, resource="/a", success=False),
            make_event(when=base + timedelta(minutes=2), resource="/b"),
        ]
        summary = summarize_session(events)
        assert summary["event_count"] == 2.0
        assert summary["distinct_resources"] == 2.0
        assert summary["auth_failures"] == 1.0
        assert summary["span_seconds"] == pytest.approx(120.0)

    def test_summarize_empty_session(self) -> None:
        assert summarize_session([]) == {}

    def test_group_by_session_sorts_within_groups(self) -> None:
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        events = [
            make_event(when=base + timedelta(minutes=5), session_id="s"),
            make_event(when=base, session_id="s"),
        ]
        grouped = group_by_session(events)
        assert grouped["s"][0].timestamp == base

    def test_session_command_sequence_deduplicates_rolling_windows(self) -> None:
        """Each event carries an overlapping window; naive concatenation would repeat tokens."""
        base = datetime(2026, 3, 2, 9, tzinfo=timezone.utc)
        events = [
            make_event(when=base, commands=["login"]),
            make_event(when=base + timedelta(minutes=1), commands=["login", "view"]),
            make_event(when=base + timedelta(minutes=2), commands=["login", "view", "logout"]),
        ]
        assert session_command_sequence(events) == ["login", "view", "logout"]


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #


class TestCohorts:
    """Cohorts are the cold-start priors, so their fit and assignment must be stable."""

    def test_summary_vector_matches_declared_names(self) -> None:
        vector = behavior_summary(BehaviorProfile(entity_id="x"))
        assert vector.shape[0] == len(SUMMARY_FEATURE_NAMES)

    def test_summary_is_finite_for_empty_profile(self) -> None:
        assert np.isfinite(behavior_summary(BehaviorProfile(entity_id="x"))).all()

    def test_summary_encodes_entity_type(self) -> None:
        index = SUMMARY_FEATURE_NAMES.index("type_edge_device")
        vector = behavior_summary(BehaviorProfile(entity_id="x", entity_type="edge_device"))
        assert vector[index] == 1.0

    def test_fit_produces_requested_cohort_count(self, fitted_pipeline: FeaturePipeline) -> None:
        assert fitted_pipeline.cohorts.n_cohorts == settings.cohort_count

    def test_assignment_is_deterministic(self, fitted_pipeline: FeaturePipeline) -> None:
        profile = next(iter(fitted_pipeline.profiles.profiles.values()))
        assert fitted_pipeline.cohorts.assign(profile) == fitted_pipeline.cohorts.assign(profile)

    def test_refit_is_reproducible(self, fitted_pipeline: FeaturePipeline) -> None:
        """Cohort ids must be stable across runs or persisted codes become meaningless."""
        profiles = list(fitted_pipeline.profiles.profiles.values())
        first = CohortModel.fit(profiles)
        second = CohortModel.fit(profiles)
        assert first.centroids == second.centroids

    def test_every_cohort_has_members(self, fitted_pipeline: FeaturePipeline) -> None:
        assert all(size > 0 for size in fitted_pipeline.cohorts.sizes.values())

    def test_type_cohorts_cover_all_entity_types(self, fitted_pipeline: FeaturePipeline) -> None:
        assert set(fitted_pipeline.cohorts.type_cohorts) >= {
            member.value for member in EntityType
        }

    def test_new_entity_gets_a_type_appropriate_cohort(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        cohort = fitted_pipeline.cohorts.assign_for_new_entity("edge_device", None)
        assert cohort is not None
        assert cohort == fitted_pipeline.cohorts.type_cohorts["edge_device"]

    def test_partial_profile_is_used_when_available(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        """A handful of logins already reveals a nightly batch job from a daytime user."""
        accumulator = ProfileAccumulator("newcomer", "user")
        base = datetime(2026, 3, 2, 3, tzinfo=timezone.utc)
        for index in range(6):
            accumulator.update(
                make_event(entity_id="newcomer", when=base + timedelta(minutes=index * 7))
            )
        assert fitted_pipeline.cohorts.assign_for_new_entity("user", accumulator.build()) is not None

    def test_unfitted_model_assigns_nothing(self) -> None:
        assert CohortModel().assign(BehaviorProfile(entity_id="x")) is None

    def test_describe_is_human_readable(self, fitted_pipeline: FeaturePipeline) -> None:
        description = fitted_pipeline.cohorts.describe(0)
        assert any(word in description for word in ("user", "service_account", "edge_device"))

    def test_json_round_trip(self, fitted_pipeline: FeaturePipeline, tmp_path: Path) -> None:
        model = fitted_pipeline.cohorts
        restored = CohortModel.load(model.save(tmp_path / "cohorts.json"))
        profile = next(iter(fitted_pipeline.profiles.profiles.values()))
        assert restored.assign(profile) == model.assign(profile)

    def test_cohort_priors_are_not_cold_start(self, fitted_pipeline: FeaturePipeline) -> None:
        for prior in fitted_pipeline.profiles.cohort_priors.values():
            assert prior.cold_start is False
            assert prior.event_count > 0

    def test_cohort_prior_is_richer_than_any_member(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        """Priors merge counts, so they must dominate their members in evidence."""
        priors = fitted_pipeline.profiles.cohort_priors
        assert priors
        cohort, prior = next(iter(priors.items()))
        members = [
            profile
            for profile in fitted_pipeline.profiles.profiles.values()
            if profile.cohort == cohort
        ]
        if len(members) > 1:
            assert prior.event_count > max(member.event_count for member in members)

    def test_global_prior_exists(self, fitted_pipeline: FeaturePipeline) -> None:
        assert fitted_pipeline.profiles.global_prior.event_count > 0


# --------------------------------------------------------------------------- #
# The shared entry point
# --------------------------------------------------------------------------- #


class TestFeaturize:
    """The one function both planes call."""

    def test_vector_width_matches_feature_space(self, fitted_pipeline: FeaturePipeline) -> None:
        vector = fitted_pipeline.featurize(make_event(), update_state=False)
        assert vector.values.shape[0] == fitted_pipeline.n_features
        assert len(vector.names) == fitted_pipeline.n_features

    def test_feature_count_is_numeric_plus_categorical(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        assert fitted_pipeline.n_features == len(NUMERIC_FEATURE_NAMES) + len(
            CATEGORICAL_FEATURE_NAMES
        )

    def test_all_values_finite(self, fitted_pipeline: FeaturePipeline) -> None:
        vector = fitted_pipeline.featurize(make_event(), update_state=False)
        assert np.isfinite(vector.values).all()

    def test_raw_values_are_kept_for_explanations(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        """Analysts need "0.4% of their logins are at this hour", not "-2.3 sigma"."""
        vector = fitted_pipeline.featurize(make_event(), update_state=False)
        assert set(vector.raw) >= set(NUMERIC_FEATURE_NAMES)

    def test_sequence_ids_are_fixed_length(self, fitted_pipeline: FeaturePipeline) -> None:
        vector = fitted_pipeline.featurize(make_event(), update_state=False)
        assert len(vector.sequence_ids) == fitted_pipeline.vocab.max_len

    def test_categorical_indices_point_at_code_columns(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        indices = fitted_pipeline.categorical_indices
        assert len(indices) == len(CATEGORICAL_FEATURE_NAMES)
        for index in indices:
            assert fitted_pipeline.feature_names[index].endswith("_code")

    def test_labels_never_influence_features(self, fitted_pipeline: FeaturePipeline) -> None:
        """The single most important invariant: ground truth cannot reach a feature."""
        plain = make_event()
        labeled = plain.model_copy(
            update={
                "label": AnomalyType.LOW_AND_SLOW_EXFIL,
                "campaign_id": "cmp_1",
                "stage": 2,
            }
        )
        first = fitted_pipeline.featurize(plain, update_state=False)
        second = fitted_pipeline.featurize(labeled, update_state=False)
        assert np.array_equal(first.values, second.values)

    def test_update_state_false_leaves_history_untouched(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        """The Phase 6 counterfactual search scores hypothetical events; it must not pollute state."""
        event = make_event(entity_id="probe_entity")
        before = len(fitted_pipeline.states.get("probe_entity").events)
        fitted_pipeline.featurize(event, update_state=False)
        assert len(fitted_pipeline.states.get("probe_entity").events) == before

    def test_update_state_true_records_history(self, fitted_pipeline: FeaturePipeline) -> None:
        event = make_event(entity_id="recording_entity")
        fitted_pipeline.featurize(event, update_state=True)
        assert fitted_pipeline.states.get("recording_entity").previous is not None

    def test_unseen_entity_produces_a_usable_vector(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        """Cold start must degrade gracefully, never raise and never emit NaNs."""
        vector = fitted_pipeline.featurize(
            make_event(entity_id="never_seen_before"), update_state=False
        )
        assert np.isfinite(vector.values).all()
        assert vector.cold_start is True
        assert vector.cohort is not None, "cold-start events still need a cohort prior"

    def test_unseen_categories_are_flagged_not_fatal(
        self, fitted_pipeline: FeaturePipeline
    ) -> None:
        vector = fitted_pipeline.featurize(
            make_event(country="Atlantis", resource="/unheard/of"), update_state=False
        )
        assert vector.novelty["geo_country"] is True
        assert vector.novelty["resource_accessed"] is True
        assert np.isfinite(vector.values).all()

    def test_established_entity_is_not_cold_start(
        self, fitted_pipeline: FeaturePipeline, small_events: List[Event]
    ) -> None:
        established = [
            entity_id
            for entity_id, profile in fitted_pipeline.profiles.profiles.items()
            if not profile.cold_start
        ]
        assert established, "expected some established entities"
        event = next(
            event for event in small_events if event.entity_id == established[0]
        )
        assert fitted_pipeline.featurize(event, update_state=False).cold_start is False

    def test_featurize_is_deterministic(self, fitted_pipeline: FeaturePipeline) -> None:
        event = make_event()
        first = fitted_pipeline.featurize(event, update_state=False)
        second = fitted_pipeline.featurize(event, update_state=False)
        assert np.array_equal(first.values, second.values)

    def test_as_dict_maps_names_to_values(self, fitted_pipeline: FeaturePipeline) -> None:
        vector = fitted_pipeline.featurize(make_event(), update_state=False)
        mapping = vector.as_dict()
        assert len(mapping) == fitted_pipeline.n_features
        assert mapping[fitted_pipeline.feature_names[0]] == pytest.approx(vector.values[0])


class TestTrainServeParity:
    """The acceptance criterion: offline features equal online features, exactly."""

    def test_batch_replay_equals_event_by_event(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        """Batch training and single-event serving must agree bit for bit.

        Two independent pipelines share the same fitted artifacts. One replays the events as a
        batch, the other one at a time as the online scorer would. Any divergence means a
        stateful feature behaves differently in the two paths -- the classic silent failure where
        offline metrics stay excellent while production degrades.
        """
        offline = FeaturePipeline(
            encoders=fitted_pipeline.encoders,
            vocab=fitted_pipeline.vocab,
            profiles=fitted_pipeline.profiles,
            cohorts=fitted_pipeline.cohorts,
            corpus=fitted_pipeline.corpus,
        )
        online = FeaturePipeline(
            encoders=fitted_pipeline.encoders,
            vocab=fitted_pipeline.vocab,
            profiles=fitted_pipeline.profiles,
            cohorts=fitted_pipeline.cohorts,
            corpus=fitted_pipeline.corpus,
        )

        batch = offline.featurize_events(replay_events)
        streamed = [
            online.featurize(event, update_state=True) for event in replay_events
        ]

        assert len(batch) == len(streamed)
        for offline_vector, online_vector in zip(batch, streamed):
            assert offline_vector.event_id == online_vector.event_id
            assert np.allclose(
                offline_vector.values, online_vector.values, rtol=0.0, atol=0.0
            ), f"divergence on event {offline_vector.event_id}"

    def test_parity_holds_for_raw_values_too(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        offline = FeaturePipeline(
            encoders=fitted_pipeline.encoders,
            vocab=fitted_pipeline.vocab,
            profiles=fitted_pipeline.profiles,
            cohorts=fitted_pipeline.cohorts,
            corpus=fitted_pipeline.corpus,
        )
        online = FeaturePipeline(
            encoders=fitted_pipeline.encoders,
            vocab=fitted_pipeline.vocab,
            profiles=fitted_pipeline.profiles,
            cohorts=fitted_pipeline.cohorts,
            corpus=fitted_pipeline.corpus,
        )
        batch = offline.featurize_events(replay_events)
        streamed = [online.featurize(event) for event in replay_events]

        for offline_vector, online_vector in zip(batch, streamed):
            for name in NUMERIC_FEATURE_NAMES:
                assert offline_vector.raw[name] == pytest.approx(
                    online_vector.raw[name], rel=0.0, abs=0.0
                ), f"{name} diverged on {offline_vector.event_id}"

    def test_replay_is_reproducible(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        first = fitted_pipeline.featurize_events(replay_events, reset=True)
        second = fitted_pipeline.featurize_events(replay_events, reset=True)
        assert np.array_equal(
            FeaturePipeline.to_matrix(first), FeaturePipeline.to_matrix(second)
        )

    def test_reset_clears_state_between_replays(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        """Without a reset, one replay's history would leak into the next."""
        fitted_pipeline.featurize_events(replay_events, reset=True)
        assert len(fitted_pipeline.states) > 0
        fitted_pipeline.reset_state()
        assert len(fitted_pipeline.states) == 0

    def test_events_are_sorted_before_replay(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        """Window features are meaningless out of order, so the batch helper must sort."""
        shuffled = list(reversed(replay_events))
        ordered_result = fitted_pipeline.featurize_events(replay_events, reset=True)
        shuffled_result = fitted_pipeline.featurize_events(shuffled, reset=True)
        assert [vector.event_id for vector in ordered_result] == [
            vector.event_id for vector in shuffled_result
        ]

    def test_no_future_information_in_features(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        """The first event of an entity cannot know its own session count."""
        pipeline = FeaturePipeline(
            encoders=fitted_pipeline.encoders,
            vocab=fitted_pipeline.vocab,
            profiles=ProfileStore(),  # no persisted history at all
            cohorts=fitted_pipeline.cohorts,
            corpus=fitted_pipeline.corpus,
        )
        vectors = pipeline.featurize_events(replay_events)

        seen: set = set()
        for vector in vectors:
            if vector.entity_id not in seen:
                assert vector.raw["is_first_event"] == 1.0
                assert vector.raw["log_entity_event_count"] == pytest.approx(0.0)
                seen.add(vector.entity_id)


class TestMatrixHelpers:
    def test_to_matrix_shape(self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]) -> None:
        vectors = fitted_pipeline.featurize_events(replay_events[:50])
        matrix = FeaturePipeline.to_matrix(vectors)
        assert matrix.shape == (len(vectors), fitted_pipeline.n_features)

    def test_to_matrix_of_empty(self) -> None:
        assert FeaturePipeline.to_matrix([]).shape == (0, 0)

    def test_raw_matrix_shape(self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]) -> None:
        vectors = fitted_pipeline.featurize_events(replay_events[:50])
        matrix = FeaturePipeline.raw_matrix(vectors, list(NUMERIC_FEATURE_NAMES))
        assert matrix.shape == (len(vectors), len(NUMERIC_FEATURE_NAMES))

    def test_sequence_matrix_shape(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        vectors = fitted_pipeline.featurize_events(replay_events[:50])
        matrix = FeaturePipeline.sequence_matrix(vectors)
        assert matrix.shape == (len(vectors), fitted_pipeline.vocab.max_len)
        assert matrix.dtype == np.int64

    def test_sequence_matrix_preserves_left_padding(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event]
    ) -> None:
        vectors = fitted_pipeline.featurize_events(replay_events[:20])
        matrix = FeaturePipeline.sequence_matrix(vectors)
        assert (matrix[:, -1] != PAD_ID).all(), "the newest token must sit at the end"


class TestPipelinePersistence:
    """A loaded pipeline must reproduce the fitted one exactly."""

    def test_save_writes_every_component(
        self, fitted_pipeline: FeaturePipeline, tmp_path: Path
    ) -> None:
        paths = fitted_pipeline.save(tmp_path)
        for name in ("encoders", "sequence_vocab", "entity_profiles", "cohorts", "corpus_stats"):
            assert name in paths
            assert paths[name].exists()

    def test_round_trip_reproduces_features(
        self, fitted_pipeline: FeaturePipeline, replay_events: List[Event], tmp_path: Path
    ) -> None:
        """This is what the serving container does at startup; it must match training exactly."""
        fitted_pipeline.save(tmp_path)
        loaded = FeaturePipeline.load(tmp_path)

        original = FeaturePipeline(
            encoders=fitted_pipeline.encoders,
            vocab=fitted_pipeline.vocab,
            profiles=fitted_pipeline.profiles,
            cohorts=fitted_pipeline.cohorts,
            corpus=fitted_pipeline.corpus,
        )
        expected = original.featurize_events(replay_events[:80])
        actual = loaded.featurize_events(replay_events[:80])

        # Exact equality, not approximate. Persisted profiles are written at full precision
        # precisely so that a reloaded pipeline is indistinguishable from the fitted one.
        for left, right in zip(expected, actual):
            assert np.array_equal(left.values, right.values), left.event_id

    def test_feature_space_file_matches_pipeline(
        self, fitted_pipeline: FeaturePipeline, tmp_path: Path
    ) -> None:
        paths = fitted_pipeline.save(tmp_path)
        with paths["feature_space"].open(encoding="utf-8") as handle:
            space = json.load(handle)
        assert space["n_features"] == fitted_pipeline.n_features
        assert space["feature_names"] == list(fitted_pipeline.feature_names)
        assert space["categorical_indices"] == fitted_pipeline.categorical_indices

    def test_load_without_encoders_fails_loudly(self, tmp_path: Path) -> None:
        """Serving must refuse to start rather than score with unfitted transforms."""
        with pytest.raises(FileNotFoundError, match="build_baselines"):
            FeaturePipeline.load(tmp_path / "nothing")

    def test_unfitted_pipeline_reports_itself(self) -> None:
        assert FeaturePipeline().is_fitted is False

    def test_fitted_pipeline_reports_itself(self, fitted_pipeline: FeaturePipeline) -> None:
        assert fitted_pipeline.is_fitted is True


# --------------------------------------------------------------------------- #
# The artifacts on disk
# --------------------------------------------------------------------------- #

_ARTIFACTS = Path(settings.artifacts_dir)
_HAS_ARTIFACTS = (_ARTIFACTS / "encoders.json").exists()


@pytest.mark.skipif(
    not _HAS_ARTIFACTS, reason="run python -m training.build_baselines first"
)
class TestBuiltArtifacts:
    """Validate the real fitted pipeline the later phases will train against."""

    def test_pipeline_loads(self) -> None:
        pipeline = FeaturePipeline.load(_ARTIFACTS)
        assert pipeline.is_fitted
        assert pipeline.n_features == len(NUMERIC_FEATURE_NAMES) + len(
            CATEGORICAL_FEATURE_NAMES
        )

    def test_profiles_were_built(self) -> None:
        pipeline = FeaturePipeline.load(_ARTIFACTS)
        assert len(pipeline.profiles.profiles) > 100

    def test_cohorts_support_cold_start(self) -> None:
        pipeline = FeaturePipeline.load(_ARTIFACTS)
        assert pipeline.cohorts.n_cohorts >= 2
        assert pipeline.profiles.cohort_priors
        assert set(pipeline.profiles.type_cohorts) >= {
            member.value for member in EntityType
        }

    def test_every_cohort_has_a_prior(self) -> None:
        pipeline = FeaturePipeline.load(_ARTIFACTS)
        for cohort in range(pipeline.cohorts.n_cohorts):
            if pipeline.cohorts.sizes.get(cohort, 0) > 0:
                assert cohort in pipeline.profiles.cohort_priors

    def test_priors_carry_feature_statistics(self) -> None:
        """Phase 3 scores statistical deviation against these, so they must be populated."""
        pipeline = FeaturePipeline.load(_ARTIFACTS)
        for prior in pipeline.profiles.cohort_priors.values():
            assert len(prior.feature_means) == len(NUMERIC_FEATURE_NAMES)
            assert len(prior.feature_stds) == len(NUMERIC_FEATURE_NAMES)

    def test_established_profiles_carry_feature_statistics(self) -> None:
        pipeline = FeaturePipeline.load(_ARTIFACTS)
        established = [
            profile
            for profile in pipeline.profiles.profiles.values()
            if not profile.cold_start
        ]
        assert established
        assert len(established[0].feature_means) == len(NUMERIC_FEATURE_NAMES)

    def test_manifest_records_feature_artifacts(self) -> None:
        from common.artifacts import read_manifest

        slots = read_manifest(_ARTIFACTS / settings.manifest_filename).get("artifacts") or {}
        for key in ("encoders", "scaler", "entity_profiles", "cohorts", "sequence_vocab"):
            assert slots.get(key), f"manifest slot '{key}' not recorded"

    def test_scoring_latency_is_well_within_budget(self) -> None:
        """Feature computation is one part of a <50 ms per-event budget (Phase 7)."""
        import time

        pipeline = FeaturePipeline.load(_ARTIFACTS)
        entity_id = next(iter(pipeline.profiles.profiles))
        event = make_event(entity_id=entity_id)

        pipeline.featurize(event, update_state=False)  # warm caches

        started = time.perf_counter()
        iterations = 200
        for _ in range(iterations):
            pipeline.featurize(event, update_state=False)
        per_event_ms = (time.perf_counter() - started) * 1000.0 / iterations

        assert per_event_ms < 15.0, f"featurize took {per_event_ms:.2f} ms/event"
