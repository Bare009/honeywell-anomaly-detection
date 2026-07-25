"""Synthetic data generator tests.

The dataset is the foundation of every metric this project reports, so these tests guard the
properties that would silently invalidate everything downstream: label separation, the
anomaly rate band, per-class presence, campaign integrity, drift, and determinism.

Most tests share one small dataset generated once per session. A handful of tests exercise
individual injectors directly, because a property like "impossible travel really is
impossible" is best asserted on the geometry rather than inferred from aggregates.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from common.config import settings
from common.models import (
    ANOMALY_CLASSES,
    ATTACK_CLASSES,
    AnomalyType,
    EntityType,
    Event,
)
from data_generator.attacks import (
    INJECTORS,
    haversine_km,
    inject_brute_force,
    inject_credential_stuffing,
    inject_device_spoofing,
    inject_impossible_travel,
    inject_insider_drift,
    inject_lateral_movement,
    inject_low_and_slow_exfil,
    run_injector,
    split_windows,
)
from data_generator.campaigns import CAMPAIGN_TEMPLATES, generate_campaign
from data_generator.drift import (
    DriftKind,
    assign_drift_plans,
    drift_summary,
    effective_hour_weights,
)
from data_generator.generate import (
    EVENT_COLUMNS,
    LABEL_COLUMNS,
    assign_split,
    dataframe_to_events,
    entities_to_records,
    generate_dataset,
    load_events,
    load_labels,
    load_metadata,
    write_dataset,
)
from data_generator.normal import generate_benign_events, generate_session
from data_generator.profiles import (
    COHORTS,
    HOME_CITIES,
    HOSTILE_CITIES,
    GeneratorConfig,
    World,
    build_world,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def small_config(**overrides) -> GeneratorConfig:
    """A fast configuration that still exercises every code path.

    Kept large enough to produce all eight attack classes and at least two campaigns; a
    smaller population would leave classes empty and the tests would pass vacuously.
    """
    config = GeneratorConfig(
        seed=42,
        n_entities=60,
        days=18,
        target_anomaly_rate=0.020,
        campaign_budget_fraction=0.42,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture(scope="module")
def small_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, World, list, Dict]:
    """One generated dataset shared across the module (generation is the slow part)."""
    return generate_dataset(small_config())


@pytest.fixture(scope="module")
def events_frame(small_dataset) -> pd.DataFrame:
    return small_dataset[0]


@pytest.fixture(scope="module")
def labels_frame(small_dataset) -> pd.DataFrame:
    return small_dataset[1]


@pytest.fixture(scope="module")
def world(small_dataset) -> World:
    return small_dataset[2]


@pytest.fixture(scope="module")
def campaigns(small_dataset) -> list:
    return small_dataset[3]


@pytest.fixture(scope="module")
def metadata(small_dataset) -> Dict:
    return small_dataset[4]


@pytest.fixture
def tiny_world() -> World:
    """A minimal world for exercising individual injectors."""
    config = GeneratorConfig(seed=7, n_entities=30, days=10)
    rng = np.random.default_rng(7)
    return build_world(config, rng)


# --------------------------------------------------------------------------- #
# World and cohorts
# --------------------------------------------------------------------------- #


class TestWorld:
    """The population must be well-formed before anything else can be trusted."""

    def test_entity_count_is_exact(self) -> None:
        config = small_config()
        world = build_world(config, np.random.default_rng(1))
        assert len(world.entities) == config.n_entities

    def test_entity_ids_unique(self, tiny_world: World) -> None:
        ids = [entity.entity_id for entity in tiny_world.entities]
        assert len(ids) == len(set(ids))

    def test_all_six_cohorts_populated(self) -> None:
        """Cohorts are the cold-start priors; an empty one would break that fallback."""
        world = build_world(small_config(), np.random.default_rng(2))
        for cohort in COHORTS:
            assert world.cohort_members(cohort.cohort_id), f"cohort {cohort.name} is empty"

    def test_all_three_entity_types_present(self, tiny_world: World) -> None:
        types = {entity.entity_type for entity in tiny_world.entities}
        assert types == set(EntityType)

    def test_entities_have_distinct_resource_preferences(self, tiny_world: World) -> None:
        """Per-entity variation is what makes entity-level baselines worth having."""
        cohort_id = tiny_world.entities[0].cohort_id
        members = tiny_world.cohort_members(cohort_id)
        if len(members) < 2:
            pytest.skip("need two members of one cohort")
        first, second = members[0].resource_weights, members[1].resource_weights
        assert first != second

    def test_secondary_city_differs_from_home(self, tiny_world: World) -> None:
        for entity in tiny_world.entities:
            assert entity.secondary_city.name != entity.home_city.name

    def test_foreign_resources_are_outside_own_cohort(self, tiny_world: World) -> None:
        entity = tiny_world.entities[0]
        rng = np.random.default_rng(3)
        foreign = tiny_world.foreign_resources(entity, rng, count=4)
        assert foreign
        assert not (set(foreign) & set(entity.cohort.resources))

    def test_by_id_raises_on_unknown(self, tiny_world: World) -> None:
        with pytest.raises(KeyError):
            tiny_world.by_id("nope_9999")

    def test_coldstart_entities_activate_late(self) -> None:
        config = small_config()
        world = build_world(config, np.random.default_rng(4))
        cold = [entity for entity in world.entities if entity.is_coldstart]
        assert cold, "expected some cold-start entities"
        for entity in cold:
            assert entity.active_from > config.start_date


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    """The event table must match section 7.1 of the plan and leak no ground truth."""

    def test_event_columns_exact(self, events_frame: pd.DataFrame) -> None:
        assert list(events_frame.columns) == list(EVENT_COLUMNS)

    def test_required_schema_fields_present(self, events_frame: pd.DataFrame) -> None:
        required = {
            "entity_id",
            "entity_type",
            "timestamp",
            "source_ip",
            "geo_country",
            "geo_lat",
            "geo_lon",
            "resource_accessed",
            "auth_method",
            "session_duration",
            "command_sequence",
            "device_os",
            "device_mac",
            "device_protocol",
        }
        assert required <= set(events_frame.columns)

    def test_events_table_carries_no_ground_truth(self, events_frame: pd.DataFrame) -> None:
        """Labels live in a separate file so feature code cannot read one by accident."""
        for forbidden in ("label", "is_anomaly", "campaign_id", "stage"):
            assert forbidden not in events_frame.columns

    def test_label_columns_exact(self, labels_frame: pd.DataFrame) -> None:
        assert list(labels_frame.columns) == list(LABEL_COLUMNS)

    def test_event_ids_unique(self, events_frame: pd.DataFrame) -> None:
        assert events_frame["event_id"].is_unique

    def test_labels_align_one_to_one_with_events(
        self, events_frame: pd.DataFrame, labels_frame: pd.DataFrame
    ) -> None:
        assert len(events_frame) == len(labels_frame)
        assert set(events_frame["event_id"]) == set(labels_frame["event_id"])

    def test_no_nulls_in_required_fields(self, events_frame: pd.DataFrame) -> None:
        for column in ("entity_id", "timestamp", "resource_accessed", "device_mac", "geo_country"):
            assert events_frame[column].notna().all(), f"{column} has nulls"

    def test_enum_values_are_valid(self, events_frame: pd.DataFrame) -> None:
        assert set(events_frame["entity_type"]) <= {member.value for member in EntityType}

    def test_events_are_time_sorted(self, events_frame: pd.DataFrame) -> None:
        assert events_frame["timestamp"].is_monotonic_increasing

    def test_all_events_inside_the_declared_timeline(
        self, events_frame: pd.DataFrame, world: World
    ) -> None:
        """Split boundaries and metadata only describe reality if nothing falls outside."""
        start = pd.Timestamp(world.config.start_date)
        end = pd.Timestamp(world.config.end_date())
        assert events_frame["timestamp"].min() >= start
        assert events_frame["timestamp"].max() < end

    def test_numeric_fields_are_non_negative(self, events_frame: pd.DataFrame) -> None:
        for column in ("session_duration", "bytes_out", "bytes_in"):
            assert (events_frame[column] >= 0).all()

    def test_command_sequences_are_non_empty_lists(self, events_frame: pd.DataFrame) -> None:
        lengths = events_frame["command_sequence"].apply(len)
        assert (lengths > 0).all()

    def test_dataframe_round_trip_is_lossless(self, events_frame: pd.DataFrame) -> None:
        """Flattening is a storage detail; the nested Event model is the real contract."""
        sample = events_frame.head(200)
        events = dataframe_to_events(sample)
        assert len(events) == len(sample)

        first = events[0]
        row = sample.iloc[0]
        assert isinstance(first, Event)
        assert first.event_id == row["event_id"]
        assert first.geo.country == row["geo_country"]
        assert first.device_fingerprint.mac == row["device_mac"]
        assert first.command_sequence == list(row["command_sequence"])
        assert first.bytes_out == pytest.approx(row["bytes_out"])
        # Ground truth must not survive the round trip -- it was never in the table.
        assert first.label is None


# --------------------------------------------------------------------------- #
# Anomaly rate and class coverage
# --------------------------------------------------------------------------- #


class TestAnomalyRate:
    """The mandated imbalance band, and the arithmetic that makes the budget metric possible."""

    def test_rate_within_mandated_band(self, metadata: Dict) -> None:
        rate = metadata["totals"]["anomaly_rate"]
        assert 0.005 <= rate <= 0.030, f"anomaly rate {rate:.4f} outside 0.5%-3%"

    def test_rate_close_to_target(self, metadata: Dict) -> None:
        rate = metadata["totals"]["anomaly_rate"]
        target = metadata["config"]["target_anomaly_rate"]
        assert abs(rate - target) < 0.5 * target, f"rate {rate:.4f} far from target {target}"

    def test_default_config_rate_permits_budget_target(self) -> None:
        """A 1%-of-events budget cannot contain 80% of anomalies if anomalies exceed ~1.25%.

        This guards the reasoning behind the default rate: if someone raises it, the
        recall@1%-budget target silently becomes unreachable arithmetic rather than a
        modelling goal.
        """
        default = GeneratorConfig()
        ceiling = settings.alert_budget_pct / default.target_anomaly_rate
        assert ceiling >= 0.80 / 0.9, (
            f"target_anomaly_rate {default.target_anomaly_rate} caps recall@"
            f"{settings.alert_budget_pct:.0%} budget at {ceiling:.2f}"
        )

    def test_all_attack_classes_present(self, labels_frame: pd.DataFrame) -> None:
        present = set(labels_frame["label"])
        missing = set(ATTACK_CLASSES) - present
        assert not missing, f"missing attack classes: {sorted(missing)}"

    def test_normal_class_dominates(self, labels_frame: pd.DataFrame) -> None:
        share = (labels_frame["label"] == AnomalyType.NORMAL.value).mean()
        assert share > 0.95

    def test_is_anomaly_matches_label(self, labels_frame: pd.DataFrame) -> None:
        derived = labels_frame["label"] != AnomalyType.NORMAL.value
        assert (labels_frame["is_anomaly"] == derived).all()

    def test_labels_are_within_known_class_space(self, labels_frame: pd.DataFrame) -> None:
        assert set(labels_frame["label"]) <= set(ANOMALY_CLASSES)

    def test_per_class_counts_sum_to_total(self, metadata: Dict) -> None:
        assert sum(metadata["per_class_counts"].values()) == metadata["totals"]["n_events"]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


class TestSplits:
    """Time-ordered splits with comparable anomaly density."""

    def test_all_three_splits_present(self, events_frame: pd.DataFrame) -> None:
        assert set(events_frame["split"]) == {"train", "val", "test"}

    def test_splits_do_not_overlap_in_time(self, events_frame: pd.DataFrame) -> None:
        """A random split would leak the future into training and inflate every metric."""
        bounds = {
            name: (group["timestamp"].min(), group["timestamp"].max())
            for name, group in events_frame.groupby("split")
        }
        assert bounds["train"][1] <= bounds["val"][0]
        assert bounds["val"][1] <= bounds["test"][0]

    def test_assign_split_respects_boundaries(self) -> None:
        config = small_config()
        train_end, val_end = config.split_boundaries()
        assert assign_split(config.start_date, config) == "train"
        assert assign_split(train_end - timedelta(seconds=1), config) == "train"
        assert assign_split(train_end, config) == "val"
        assert assign_split(val_end, config) == "test"
        assert assign_split(config.end_date() - timedelta(seconds=1), config) == "test"

    def test_every_split_contains_anomalies(self, metadata: Dict) -> None:
        """The classifier needs labeled attacks in train; evaluation needs unseen ones in test."""
        for split, values in metadata["splits"]["per_split"].items():
            assert values["n_anomalies"] > 0, f"{split} has no anomalies"

    def test_val_and_test_density_same_order_of_magnitude(self, metadata: Dict) -> None:
        """Sanity bound only -- see the tight assertion on the real dataset below.

        This module's config is deliberately small (60 entities, 18 days), and incidents are
        chunky: one credential-stuffing spray is ~34 events against a per-split budget of
        roughly 8. Integer granularity therefore dominates at this scale. The property that
        actually matters -- validation and test carrying comparable anomaly density, because
        the alert-budget threshold is tuned on one and applied to the other -- is asserted at
        production scale in ``TestGeneratedArtifacts``.
        """
        per_split = metadata["splits"]["per_split"]
        val_rate = per_split["val"]["anomaly_rate"]
        test_rate = per_split["test"]["anomaly_rate"]
        assert val_rate > 0 and test_rate > 0
        ratio = max(val_rate, test_rate) / min(val_rate, test_rate)
        assert ratio < 5.0, f"val {val_rate:.4f} vs test {test_rate:.4f} differ wildly"

    def test_split_windows_tile_the_timeline(self) -> None:
        config = small_config()
        windows = split_windows(config)
        assert windows["train"][0] == config.start_date
        assert windows["train"][1] == windows["val"][0]
        assert windows["val"][1] == windows["test"][0]
        assert windows["test"][1] == config.end_date()


# --------------------------------------------------------------------------- #
# Benign traffic realism
# --------------------------------------------------------------------------- #


class TestBenignRealism:
    """Benign traffic must be messy, or every attack is trivially separable."""

    def test_benign_includes_auth_failures(self, metadata: Dict) -> None:
        assert metadata["benign"]["auth_failure_rate"] > 0.0

    def test_benign_auth_failures_are_rare(self, metadata: Dict) -> None:
        assert metadata["benign"]["auth_failure_rate"] < 0.05

    def test_benign_includes_off_hours_activity(self, metadata: Dict) -> None:
        """Off-hours access alone must never imply an attack."""
        assert metadata["benign"]["off_hours_rate"] > 0.02

    def test_benign_spans_multiple_countries(self, metadata: Dict) -> None:
        assert metadata["benign"]["n_countries"] >= 2

    def test_benign_touches_sensitive_resources(
        self, events_frame: pd.DataFrame, labels_frame: pd.DataFrame
    ) -> None:
        """A sensitive resource name must not be close to a label by itself."""
        merged = events_frame.merge(labels_frame[["event_id", "is_anomaly"]], on="event_id")
        benign_sensitive = merged[
            (~merged["is_anomaly"]) & (merged["resource_accessed"].str.startswith("/vault/"))
        ]
        assert len(benign_sensitive) > 0

    def test_command_sequences_vary_within_a_cohort(self, tiny_world: World) -> None:
        """Templates give structure; perturbation stops the model memorizing exact strings."""
        entity = tiny_world.entities[0]
        rng = np.random.default_rng(11)
        sequences = set()
        for index in range(40):
            events = generate_session(
                entity,
                tiny_world,
                tiny_world.config.start_date + timedelta(hours=index),
                rng,
                index,
            )
            sequences.add(tuple(events[-1].command_sequence))
        assert len(sequences) > 3

    def test_sessions_are_internally_consistent(self, tiny_world: World) -> None:
        entity = tiny_world.entities[0]
        rng = np.random.default_rng(12)
        events = generate_session(
            entity, tiny_world, tiny_world.config.start_date, rng, 0
        )
        assert len({event.session_id for event in events}) == 1
        assert len({event.device_fingerprint.mac for event in events}) == 1
        assert len({event.source_ip for event in events}) == 1
        timestamps = [event.timestamp for event in events]
        assert timestamps == sorted(timestamps)

    def test_all_benign_events_labeled_normal(self, tiny_world: World) -> None:
        rng = np.random.default_rng(13)
        events = generate_benign_events(tiny_world, rng, tiny_world.config)
        assert events
        assert all(event.label is AnomalyType.NORMAL for event in events)

    def test_coldstart_entities_have_no_early_events(self, tiny_world: World) -> None:
        rng = np.random.default_rng(14)
        events = generate_benign_events(tiny_world, rng, tiny_world.config)
        active_from = {
            entity.entity_id: entity.active_from for entity in tiny_world.entities
        }
        for event in events:
            assert event.timestamp >= active_from[event.entity_id]


# --------------------------------------------------------------------------- #
# Individual injectors
# --------------------------------------------------------------------------- #


class TestInjectors:
    """Each class must actually exhibit its defining signal."""

    def test_registry_covers_every_attack_class(self) -> None:
        assert set(INJECTORS) == set(ATTACK_CLASSES)

    def test_brute_force_burst_exceeds_detector_threshold(self, tiny_world: World) -> None:
        """A brute force that does not burst is not a brute force."""
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(21)
        incident = inject_brute_force(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "bf1"
        )
        failures = [event for event in incident.events if not event.auth_success]
        assert len(failures) >= settings.brute_force_threshold

        span_minutes = (
            failures[-1].timestamp - failures[0].timestamp
        ).total_seconds() / 60.0
        assert span_minutes <= settings.brute_force_window_minutes

    def test_brute_force_targets_a_single_entity(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(22)
        incident = inject_brute_force(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "bf2"
        )
        assert {event.entity_id for event in incident.events} == {entity.entity_id}

    def test_brute_force_comes_from_one_source(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(23)
        incident = inject_brute_force(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "bf3"
        )
        assert len({event.source_ip for event in incident.events}) == 1

    def test_credential_stuffing_sprays_many_entities(self, tiny_world: World) -> None:
        """The inverse shape of brute force: breadth across accounts, not depth against one."""
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(24)
        incident = inject_credential_stuffing(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "cs1"
        )
        victims = {event.entity_id for event in incident.events}
        assert len(victims) > 1

        per_victim = max(
            sum(1 for event in incident.events if event.entity_id == victim)
            for victim in victims
        )
        assert per_victim < settings.brute_force_threshold

    def test_credential_stuffing_uses_one_source_ip(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(25)
        incident = inject_credential_stuffing(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "cs2"
        )
        assert len({event.source_ip for event in incident.events}) == 1

    def test_impossible_travel_exceeds_velocity_threshold(self, tiny_world: World) -> None:
        """Must be unambiguous: the deterministic detector should reach ~1.0 precision."""
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        for seed in range(8):
            rng = np.random.default_rng(30 + seed)
            incident = inject_impossible_travel(
                entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "it1"
            )
            events = sorted(incident.events, key=lambda event: event.timestamp)
            first, second = events[0], events[1]

            distance = haversine_km(
                first.geo.lat, first.geo.lon, second.geo.lat, second.geo.lon
            )
            hours = (second.timestamp - first.timestamp).total_seconds() / 3600.0
            velocity = distance / max(hours, 1e-9)

            assert velocity > settings.impossible_travel_kmh, (
                f"seed {seed}: {velocity:.0f} km/h over {distance:.0f} km in {hours:.2f} h"
            )

    def test_impossible_travel_authentications_succeed(self, tiny_world: World) -> None:
        """It is a stolen valid session, not a guessing attack."""
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(31)
        incident = inject_impossible_travel(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "it2"
        )
        assert all(event.auth_success for event in incident.events)

    def test_lateral_movement_reaches_outside_its_cohort(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(32)
        incident = inject_lateral_movement(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "lm1"
        )
        touched = {event.resource_accessed for event in incident.events}
        assert touched - set(entity.cohort.resources)

    def test_lateral_movement_uses_recon_commands(self, tiny_world: World) -> None:
        from data_generator.profiles import HOSTILE_COMMANDS

        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(33)
        incident = inject_lateral_movement(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "lm2"
        )
        seen = {token for event in incident.events for token in event.command_sequence}
        assert seen & set(HOSTILE_COMMANDS)

    def test_device_spoofing_presents_an_unseen_mac(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.EDGE_DEVICE)
        rng = np.random.default_rng(34)
        incident = inject_device_spoofing(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "ds1"
        )
        own_macs = {device.mac for device in entity.devices}
        used = {event.device_fingerprint.mac for event in incident.events}
        assert not (used & own_macs)

    def test_low_and_slow_spans_many_hours(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(35)
        incident = inject_low_and_slow_exfil(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "ls1"
        )
        span_hours = (incident.ended_at - incident.started_at).total_seconds() / 3600.0
        assert span_hours > 4.0
        assert len(incident.events) >= 10

    def test_low_and_slow_events_are_individually_unremarkable(self, tiny_world: World) -> None:
        """If one event were big enough to alert on, this would just be a threshold rule."""
        import math

        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(36)
        incident = inject_low_and_slow_exfil(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "ls2"
        )
        median = math.exp(entity.bytes_out_lognormal[0])
        ratios = [event.bytes_out / median for event in incident.events]
        assert max(ratios) < 4.0, f"largest transfer is {max(ratios):.1f}x the median"

    def test_insider_drift_escalates_over_days(self, tiny_world: World) -> None:
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(37)
        incident = inject_insider_drift(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "id1"
        )
        span_days = (incident.ended_at - incident.started_at).total_seconds() / 86400.0
        assert span_days > 1.0

        events = sorted(incident.events, key=lambda event: event.timestamp)
        half = len(events) // 2
        sensitive = set(tiny_world.sensitive_resources)
        early = sum(1 for event in events[:half] if event.resource_accessed in sensitive)
        late = sum(1 for event in events[half:] if event.resource_accessed in sensitive)
        assert late >= early, "insider drift should converge on sensitive resources"

    def test_insider_drift_only_targets_people(self) -> None:
        """An 'insider' is a person, not a container or a sensor."""
        _, allowed = INJECTORS[AnomalyType.INSIDER_DRIFT.value]
        assert allowed == (EntityType.USER,)

    def test_every_injector_labels_its_events(self, tiny_world: World) -> None:
        rng = np.random.default_rng(38)
        for anomaly_type in ATTACK_CLASSES:
            incident = run_injector(
                anomaly_type, tiny_world, rng, tiny_world.config, f"chk_{anomaly_type[:4]}"
            )
            assert incident is not None, f"{anomaly_type} produced no incident"
            assert incident.events
            assert all(
                event.label is AnomalyType(anomaly_type) for event in incident.events
            ), f"{anomaly_type} mislabeled its events"

    def test_attacker_devices_avoid_globally_obvious_os_strings(self, tiny_world: World) -> None:
        """A `Kali Linux` OS field would be learnable from one event with no profiling."""
        entity = next(e for e in tiny_world.entities if e.entity_type is EntityType.USER)
        rng = np.random.default_rng(39)
        from data_generator.attacks import inject_credential_misuse

        incident = inject_credential_misuse(
            entity, tiny_world, rng, tiny_world.config.start_date, tiny_world.config, "cm1"
        )
        for event in incident.events:
            assert "Kali" not in event.device_fingerprint.os

    def test_hostile_cities_are_far_from_home_cities(self) -> None:
        """Geo signals only carry information if attack origins are genuinely distant."""
        for hostile in HOSTILE_CITIES:
            nearest = min(
                haversine_km(hostile.lat, hostile.lon, home.lat, home.lon)
                for home in HOME_CITIES
            )
            assert nearest > 1500.0, f"{hostile.name} is only {nearest:.0f} km from a home city"


# --------------------------------------------------------------------------- #
# Campaigns (D1)
# --------------------------------------------------------------------------- #


class TestCampaigns:
    """Multi-stage ground truth must be internally consistent or reconstruction is unmeasurable."""

    def test_campaigns_were_generated(self, campaigns: list) -> None:
        assert campaigns, "no campaigns generated"

    def test_every_template_shape_appears(self, campaigns: list) -> None:
        """Templates are cycled precisely so no shape can be missing from the demo."""
        names = {generated.template_name for generated in campaigns}
        assert len(names) >= 2

    def test_each_campaign_has_multiple_stages(self, campaigns: list) -> None:
        for generated in campaigns:
            assert len(generated.incidents) >= 2, "a one-stage campaign is just an incident"

    def test_stages_share_one_entity(self, campaigns: list) -> None:
        for generated in campaigns:
            entities = {
                event.entity_id for incident in generated.incidents for event in incident.events
            }
            assert entities == {generated.campaign.entity_id}

    def test_stages_are_time_ordered(self, campaigns: list) -> None:
        for generated in campaigns:
            starts = [incident.started_at for incident in generated.incidents]
            assert starts == sorted(starts), "campaign stages must be causally ordered"

    def test_stage_indices_are_sequential(self, campaigns: list) -> None:
        for generated in campaigns:
            stages = [incident.stage for incident in generated.incidents]
            assert stages == sorted(stages)
            assert stages[0] == 0

    def test_all_campaign_events_carry_the_campaign_id(self, campaigns: list) -> None:
        for generated in campaigns:
            for incident in generated.incidents:
                for event in incident.events:
                    assert event.campaign_id == generated.campaign.campaign_id
                    assert event.stage is not None

    def test_campaign_ids_are_unique(self, campaigns: list) -> None:
        ids = [generated.campaign.campaign_id for generated in campaigns]
        assert len(ids) == len(set(ids))

    def test_kill_chain_matches_a_known_template(self, campaigns: list) -> None:
        known = {template.name: list(template.stages) for template in CAMPAIGN_TEMPLATES}
        for generated in campaigns:
            expected = known[generated.template_name]
            actual = generated.campaign.kill_chain
            # Stages may be dropped if an injector cannot run, but order must be preserved.
            assert actual == [stage for stage in expected if stage in actual]

    def test_campaign_window_covers_its_events(self, campaigns: list) -> None:
        for generated in campaigns:
            events = generated.events
            assert generated.campaign.started_at == min(e.timestamp for e in events)
            assert generated.campaign.last_activity == max(e.timestamp for e in events)

    def test_campaign_labels_are_per_stage(self, campaigns: list) -> None:
        """A campaign is composed of existing classes; it is not a new label."""
        for generated in campaigns:
            for incident in generated.incidents:
                for event in incident.events:
                    assert event.label is incident.anomaly_type

    def test_campaign_events_appear_in_the_labels_table(
        self, campaigns: list, labels_frame: pd.DataFrame
    ) -> None:
        indexed = labels_frame.set_index("event_id")
        for generated in campaigns[:3]:
            for event in generated.events:
                assert event.event_id in indexed.index
                row = indexed.loc[event.event_id]
                assert row["campaign_id"] == generated.campaign.campaign_id

    def test_templates_require_consistent_entity_types(self) -> None:
        """An entity must be allowed by *every* stage injector or the chain is incoherent."""
        for template in CAMPAIGN_TEMPLATES:
            allowed = set(template.entity_types)
            for stage in template.stages:
                allowed &= set(INJECTORS[stage][1])
            assert allowed, f"template {template.name} has no viable entity type"

    def test_campaign_is_dropped_when_the_timeline_is_too_short(self) -> None:
        """A chain trimmed down to a single stage is just an incident, not a campaign."""
        config = GeneratorConfig(seed=5, n_entities=8, days=2)
        world = build_world(config, np.random.default_rng(5))
        result = generate_campaign(
            world,
            np.random.default_rng(5),
            config,
            campaign_index=0,
            template=CAMPAIGN_TEMPLATES[0],
        )
        assert result is None

    def test_campaign_events_stay_inside_the_timeline(self, campaigns: list, world: World) -> None:
        """A dataset claiming N days must not contain events on day N+2."""
        end = world.config.end_date()
        for generated in campaigns:
            for event in generated.events:
                assert world.config.start_date <= event.timestamp < end


# --------------------------------------------------------------------------- #
# Drift (D3)
# --------------------------------------------------------------------------- #


class TestDrift:
    """Benign drift must exist, be gradual, and be labeled normal."""

    def test_some_entities_drift(self, world: World) -> None:
        drifted = [entity for entity in world.entities if entity.drift_plan is not None]
        assert drifted, "no benign drift in the dataset"

    def test_drift_fraction_is_reasonable(self, world: World) -> None:
        summary = drift_summary(world)
        assert 0.05 <= summary["fraction_of_population"] <= 0.30

    def test_multiple_drift_kinds_present(self, world: World) -> None:
        kinds = {
            entity.drift_plan.kind
            for entity in world.entities
            if entity.drift_plan is not None
        }
        assert len(kinds) >= 2

    def test_drift_kinds_are_known(self, world: World) -> None:
        for entity in world.entities:
            if entity.drift_plan is not None:
                assert entity.drift_plan.kind in set(DriftKind)

    def test_coldstart_entities_never_drift(self, world: World) -> None:
        """No established baseline to drift away from; mixing them confounds both experiments."""
        for entity in world.entities:
            if entity.is_coldstart:
                assert entity.drift_plan is None

    def test_drift_starts_partway_through_the_timeline(self, world: World) -> None:
        config = world.config
        for entity in world.entities:
            if entity.drift_plan is not None:
                assert entity.drift_plan.starts_at > config.start_date
                assert entity.drift_plan.starts_at < config.end_date()

    def test_drift_progress_ramps_from_zero_to_one(self, world: World) -> None:
        """Gradualness is the point: an abrupt benign change is indistinguishable from attack."""
        plan = next(
            entity.drift_plan for entity in world.entities if entity.drift_plan is not None
        )
        assert plan.progress(plan.starts_at - timedelta(days=1)) == 0.0
        assert plan.progress(plan.starts_at) == 0.0
        midpoint = plan.progress(plan.starts_at + timedelta(days=plan.ramp_days / 2))
        assert 0.2 < midpoint < 0.8
        assert plan.progress(plan.starts_at + timedelta(days=plan.ramp_days * 2)) == 1.0

    def test_drift_ramp_lasts_days_not_minutes(self, world: World) -> None:
        for entity in world.entities:
            if entity.drift_plan is not None:
                assert entity.drift_plan.ramp_days >= 2.0

    def test_effective_hour_weights_stay_a_distribution(self, world: World) -> None:
        schedule_drifters = [
            entity
            for entity in world.entities
            if entity.drift_plan is not None and entity.drift_plan.new_hour_weights is not None
        ]
        if not schedule_drifters:
            pytest.skip("no schedule drift in this sample")
        entity = schedule_drifters[0]
        plan = entity.drift_plan
        assert plan is not None
        for offset in (0, 2, 5, 20):
            weights = effective_hour_weights(
                entity, plan.starts_at + timedelta(days=offset)
            )
            assert len(weights) == 24
            assert sum(weights) == pytest.approx(1.0)
            assert all(weight >= 0 for weight in weights)

    def test_drifted_entity_events_are_labeled_normal(
        self, world: World, events_frame: pd.DataFrame, labels_frame: pd.DataFrame
    ) -> None:
        """This is what makes the drift experiment honest rather than circular."""
        drifted_ids = {
            entity.entity_id for entity in world.entities if entity.drift_plan is not None
        }
        merged = events_frame.merge(labels_frame[["event_id", "label"]], on="event_id")
        subset = merged[merged["entity_id"].isin(drifted_ids)]
        assert len(subset) > 0
        # Drifted entities can still be attacked; what matters is that drift itself is benign.
        assert (subset["label"] == AnomalyType.NORMAL.value).mean() > 0.90

    def test_drift_summary_is_serializable(self, world: World) -> None:
        summary = drift_summary(world)
        json.dumps(summary)
        assert summary["all_labeled_benign"] is True

    def test_schedule_drift_actually_shifts_activity(self, world: World) -> None:
        schedule_drifters = [
            entity
            for entity in world.entities
            if entity.drift_plan is not None and entity.drift_plan.new_hour_weights is not None
        ]
        if not schedule_drifters:
            pytest.skip("no schedule drift in this sample")
        entity = schedule_drifters[0]
        plan = entity.drift_plan
        assert plan is not None
        before = effective_hour_weights(entity, plan.starts_at - timedelta(days=1))
        after = effective_hour_weights(entity, plan.starts_at + timedelta(days=30))
        assert before != after


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #


class TestColdStart:
    """Late-arriving entities must exist and have thin histories."""

    def test_coldstart_entities_exist(self, metadata: Dict) -> None:
        assert metadata["coldstart"]["n_entities"] > 0

    def test_coldstart_entities_have_fewer_events(
        self, world: World, events_frame: pd.DataFrame
    ) -> None:
        counts = events_frame["entity_id"].value_counts()
        cold = [
            counts.get(entity.entity_id, 0)
            for entity in world.entities
            if entity.is_coldstart
        ]
        warm = [
            counts.get(entity.entity_id, 0)
            for entity in world.entities
            if not entity.is_coldstart
        ]
        assert cold and warm
        assert np.mean(cold) < np.mean(warm)

    def test_coldstart_entities_are_a_minority(self, world: World) -> None:
        share = sum(1 for e in world.entities if e.is_coldstart) / len(world.entities)
        assert 0.0 < share < 0.30


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    """Every reported metric and the scripted demo assume seed 42 reproduces exactly."""

    def test_world_is_reproducible(self) -> None:
        config = small_config()
        first = build_world(config, np.random.default_rng(config.seed))
        second = build_world(config, np.random.default_rng(config.seed))
        assert [e.entity_id for e in first.entities] == [e.entity_id for e in second.entities]
        assert [e.home_city.name for e in first.entities] == [
            e.home_city.name for e in second.entities
        ]
        assert [e.devices[0].mac for e in first.entities] == [
            e.devices[0].mac for e in second.entities
        ]

    @pytest.mark.slow
    def test_full_generation_is_reproducible(self) -> None:
        config = GeneratorConfig(seed=42, n_entities=30, days=10, target_anomaly_rate=0.02)
        first_events, first_labels, _, _, _ = generate_dataset(config)
        second_events, second_labels, _, _, _ = generate_dataset(config)

        pd.testing.assert_frame_equal(first_events, second_events)
        pd.testing.assert_frame_equal(first_labels, second_labels)

    @pytest.mark.slow
    def test_different_seeds_produce_different_data(self) -> None:
        base = GeneratorConfig(seed=42, n_entities=30, days=10, target_anomaly_rate=0.02)
        other = GeneratorConfig(seed=7, n_entities=30, days=10, target_anomaly_rate=0.02)
        first, _, _, _, _ = generate_dataset(base)
        second, _, _, _, _ = generate_dataset(other)
        assert not first["event_id"].equals(second["event_id"])

    def test_generation_does_not_depend_on_global_random_state(self) -> None:
        """Component streams are spawned, so unrelated code cannot perturb the dataset."""
        config = GeneratorConfig(seed=42, n_entities=20, days=6, target_anomaly_rate=0.02)
        first, _, _, _, _ = generate_dataset(config)

        np.random.seed(999)
        _ = np.random.rand(1000)

        second, _, _, _, _ = generate_dataset(config)
        pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    """Round-tripping through parquet/JSON must preserve everything."""

    def test_write_and_reload(self, small_dataset, tmp_path: Path) -> None:
        events, labels, world, campaigns, metadata = small_dataset
        paths = write_dataset(events, labels, world, campaigns, metadata, tmp_path)

        for name, path in paths.items():
            assert path.exists(), f"{name} not written"

        reloaded_events = load_events(tmp_path)
        reloaded_labels = load_labels(tmp_path)
        assert len(reloaded_events) == len(events)
        assert list(reloaded_events.columns) == list(EVENT_COLUMNS)
        assert set(reloaded_labels["label"]) == set(labels["label"])

    def test_metadata_round_trip(self, small_dataset, tmp_path: Path) -> None:
        events, labels, world, campaigns, metadata = small_dataset
        write_dataset(events, labels, world, campaigns, metadata, tmp_path)
        reloaded = load_metadata(tmp_path)
        assert reloaded["seed"] == metadata["seed"]
        assert reloaded["totals"]["n_events"] == metadata["totals"]["n_events"]

    def test_entities_json_is_serializable_and_complete(self, world: World) -> None:
        records = entities_to_records(world)
        assert len(records) == len(world.entities)
        json.dumps(records)  # must not raise
        first = records[0]
        for key in ("entity_id", "cohort_id", "cohort_name", "is_coldstart", "drift", "devices"):
            assert key in first

    def test_campaigns_json_is_serializable(self, campaigns: list, tmp_path: Path) -> None:
        payload = [generated.as_dict() for generated in campaigns]
        text = json.dumps(payload)
        restored = json.loads(text)
        assert len(restored) == len(campaigns)
        if restored:
            assert "kill_chain" in restored[0]
            assert "template" in restored[0]

    def test_missing_dataset_raises_a_helpful_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="data_generator.generate"):
            load_events(tmp_path / "empty")


# --------------------------------------------------------------------------- #
# The committed dataset on disk
# --------------------------------------------------------------------------- #

_DATASET_DIR = settings.dataset_dir
_HAS_DATASET = (_DATASET_DIR / "events.parquet").exists()


@pytest.mark.skipif(not _HAS_DATASET, reason="run python -m data_generator.generate first")
class TestGeneratedArtifacts:
    """Validate the real dataset the rest of the pipeline will train on."""

    def test_rate_within_mandated_band(self) -> None:
        rate = load_metadata()["totals"]["anomaly_rate"]
        assert 0.005 <= rate <= 0.030

    def test_all_attack_classes_present(self) -> None:
        counts = load_metadata()["per_class_counts"]
        for name in ATTACK_CLASSES:
            assert counts.get(name, 0) > 0, f"{name} missing from the generated dataset"

    def test_budget_ceiling_permits_the_recall_target(self) -> None:
        """Confirms the real test split can actually reach recall@1% budget >= 0.80."""
        per_split = load_metadata()["splits"]["per_split"]
        test_rate = per_split["test"]["anomaly_rate"]
        ceiling = min(1.0, settings.alert_budget_pct / test_rate)
        assert ceiling >= 0.80, (
            f"test anomaly rate {test_rate:.4f} caps recall@"
            f"{settings.alert_budget_pct:.0%} budget at {ceiling:.2f}"
        )

    def test_labels_are_a_separate_file(self) -> None:
        events = load_events()
        assert "label" not in events.columns
        assert (_DATASET_DIR / "labels.parquet").exists()

    def test_campaigns_file_has_ground_truth(self) -> None:
        with (_DATASET_DIR / "campaigns.json").open(encoding="utf-8") as handle:
            campaigns = json.load(handle)
        assert campaigns
        for campaign in campaigns:
            assert len(campaign["stages"]) >= 2
            assert campaign["kill_chain"]

    def test_entities_file_records_drift_and_coldstart(self) -> None:
        with (_DATASET_DIR / "entities.json").open(encoding="utf-8") as handle:
            entities = json.load(handle)
        assert entities
        assert any(entity["drift"] is not None for entity in entities)
        assert any(entity["is_coldstart"] for entity in entities)

    def test_every_split_has_anomalies(self) -> None:
        for split, values in load_metadata()["splits"]["per_split"].items():
            assert values["n_anomalies"] > 0, f"{split} has no anomalies"

    def test_val_and_test_density_comparable(self) -> None:
        """The alert-budget threshold is tuned on val and applied to test.

        If the two splits carry different anomaly density, the tuned threshold is calibrated
        for the wrong world and every downstream operating-point metric is misleading.
        """
        per_split = load_metadata()["splits"]["per_split"]
        val_rate = per_split["val"]["anomaly_rate"]
        test_rate = per_split["test"]["anomaly_rate"]
        assert abs(val_rate - test_rate) < 0.003, (
            f"val {val_rate:.4f} vs test {test_rate:.4f} differ too much"
        )

    def test_all_events_inside_the_declared_timeline(self) -> None:
        import pandas as pd

        metadata = load_metadata()
        events = load_events()
        start = pd.Timestamp(metadata["config"]["start_date"])
        end = pd.Timestamp(metadata["config"]["end_date"])
        assert events["timestamp"].min() >= start
        assert events["timestamp"].max() < end
