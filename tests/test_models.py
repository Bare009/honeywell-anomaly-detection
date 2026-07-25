"""Domain contract tests.

These Pydantic models are the interface between the data generator, the feature pipeline,
the scorer, the read API and the dashboard. A silent shape change here would surface as a
confusing failure three phases later, so the invariants are pinned explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from common.models import (
    ANOMALY_CLASS_INDEX,
    ANOMALY_CLASSES,
    ATTACK_CLASSES,
    AnalystVerdict,
    AnomalyType,
    AuthMethod,
    BaselineComparison,
    Campaign,
    CampaignStage,
    CampaignStatus,
    ColdStartMetrics,
    Counterfactual,
    CounterfactualChange,
    DatasetSummary,
    Detection,
    DetectionScores,
    DetectionStatus,
    DeviceFingerprint,
    DriftState,
    DriftStatus,
    EntityProfile,
    EntityType,
    Event,
    Explanation,
    FeatureAttribution,
    Feedback,
    FeedbackAdjustment,
    GeoLocation,
    HealthStatus,
    MitreTechnique,
    ModelMetrics,
    SequenceStepAttribution,
    ServiceHealth,
    Session,
    new_id,
    utc_now,
)


class TestHelpers:
    """Shared helpers used as default factories across the models."""

    def test_utc_now_is_timezone_aware(self) -> None:
        assert utc_now().tzinfo is not None

    def test_new_id_is_prefixed_and_unique(self) -> None:
        first, second = new_id("det"), new_id("det")
        assert first.startswith("det_")
        assert first != second


class TestLabelSpace:
    """The class list is an ordered contract shared with model outputs and the UI."""

    def test_nine_classes(self) -> None:
        assert len(ANOMALY_CLASSES) == 9

    def test_normal_is_first(self) -> None:
        """Column 0 of every probability vector is 'normal' -- never reorder this."""
        assert ANOMALY_CLASSES[0] == "normal"

    def test_matches_enum_members(self) -> None:
        assert set(ANOMALY_CLASSES) == {member.value for member in AnomalyType}

    def test_attack_classes_exclude_normal(self) -> None:
        assert len(ATTACK_CLASSES) == 8
        assert "normal" not in ATTACK_CLASSES

    def test_expected_attack_classes_present(self) -> None:
        assert set(ATTACK_CLASSES) == {
            "credential_misuse",
            "lateral_movement",
            "brute_force",
            "impossible_travel",
            "credential_stuffing",
            "device_spoofing",
            "low_and_slow_exfil",
            "insider_drift",
        }

    def test_index_lookup_is_consistent(self) -> None:
        for position, name in enumerate(ANOMALY_CLASSES):
            assert ANOMALY_CLASS_INDEX[name] == position


class TestLenientCoercion:
    """Loose JSON from the optional LLM narrator and replayed data must not crash."""

    def test_float_from_string(self, sample_event: Event) -> None:
        event = sample_event.model_copy(update={"session_duration": 0})
        coerced = Event(**{**event.model_dump(), "session_duration": "123.5"})
        assert coerced.session_duration == pytest.approx(123.5)

    def test_float_from_percent_string(self) -> None:
        assert Counterfactual(resulting_risk="42%").resulting_risk == pytest.approx(42.0)

    def test_int_from_float_string(self) -> None:
        assert SequenceStepAttribution(position="3.0", token="ls", score=0.4).position == 3

    def test_bool_from_yes(self) -> None:
        assert Counterfactual(found="yes").found is True

    def test_bool_from_no(self) -> None:
        assert Counterfactual(found="no").found is False

    def test_bool_from_int(self) -> None:
        assert Counterfactual(found=1).found is True

    def test_invalid_string_still_rejected(self) -> None:
        """Coercion is best-effort, not a licence to accept garbage."""
        with pytest.raises(ValidationError):
            Counterfactual(resulting_risk="not-a-number")


class TestValueObjects:
    """Geo and device fingerprints are compared field-by-field, so shape matters."""

    def test_geo_round_trip(self, sample_geo: GeoLocation) -> None:
        assert GeoLocation(**sample_geo.model_dump()) == sample_geo

    def test_latitude_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            GeoLocation(country="X", lat=91.0, lon=0.0)

    def test_longitude_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            GeoLocation(country="X", lat=0.0, lon=181.0)

    def test_mac_is_normalized_to_lowercase(self) -> None:
        """Device-spoofing detection compares MACs; casing must not create false novelty."""
        device = DeviceFingerprint(os="Linux", mac="AA:BB:CC:DD:EE:FF", protocol="ssh")
        assert device.mac == "aa:bb:cc:dd:ee:ff"


class TestEvent:
    """The atomic unit the system scores."""

    def test_sample_event_is_valid(self, sample_event: Event) -> None:
        assert sample_event.entity_type is EntityType.USER
        assert sample_event.auth_method is AuthMethod.PASSWORD
        assert sample_event.auth_success is True

    def test_ground_truth_is_optional(self, sample_event: Event) -> None:
        """Serving receives unlabeled traffic; labels must never be required."""
        assert sample_event.label is None
        assert sample_event.campaign_id is None
        assert sample_event.stage is None

    def test_event_id_autogenerated(self, sample_geo, sample_device) -> None:
        event = Event(
            entity_id="user_1",
            entity_type=EntityType.USER,
            timestamp=utc_now(),
            source_ip="10.0.0.1",
            geo=sample_geo,
            resource_accessed="/api/x",
            auth_method=AuthMethod.TOKEN,
            device_fingerprint=sample_device,
        )
        assert event.event_id.startswith("evt_")

    def test_to_unlabeled_strips_ground_truth(self, sample_event: Event) -> None:
        labeled = sample_event.model_copy(
            update={
                "label": AnomalyType.BRUTE_FORCE,
                "campaign_id": "cmp_1",
                "stage": 2,
            }
        )
        stripped = labeled.to_unlabeled()

        assert stripped.label is None
        assert stripped.campaign_id is None
        assert stripped.stage is None
        # The original is untouched -- stripping must not mutate the source event.
        assert labeled.label is AnomalyType.BRUTE_FORCE
        assert stripped.event_id == labeled.event_id

    def test_json_round_trip(self, sample_event: Event) -> None:
        payload = sample_event.model_dump(mode="json")
        assert isinstance(payload["timestamp"], str)
        assert Event(**payload).event_id == sample_event.event_id

    def test_enum_serializes_to_string(self, sample_event: Event) -> None:
        payload = sample_event.model_dump(mode="json")
        assert payload["entity_type"] == "user"
        assert payload["auth_method"] == "password"

    def test_missing_required_field_rejected(self, sample_geo, sample_device) -> None:
        with pytest.raises(ValidationError):
            Event(
                entity_type=EntityType.USER,
                timestamp=utc_now(),
                source_ip="10.0.0.1",
                geo=sample_geo,
                resource_accessed="/api/x",
                auth_method=AuthMethod.TOKEN,
                device_fingerprint=sample_device,
            )

    def test_negative_session_duration_rejected(self, sample_event: Event) -> None:
        with pytest.raises(ValidationError):
            Event(**{**sample_event.model_dump(), "session_duration": -1.0})

    def test_command_sequence_defaults_to_empty(self, sample_geo, sample_device) -> None:
        event = Event(
            entity_id="dev_1",
            entity_type=EntityType.EDGE_DEVICE,
            timestamp=utc_now(),
            source_ip="10.0.0.2",
            geo=sample_geo,
            resource_accessed="modbus:502",
            auth_method=AuthMethod.CERTIFICATE,
            device_fingerprint=sample_device,
        )
        assert event.command_sequence == []


class TestSession:
    """Sessions group events for session-level features."""

    def test_defaults(self, fixed_timestamp: datetime) -> None:
        session = Session(
            entity_id="user_1",
            entity_type=EntityType.USER,
            started_at=fixed_timestamp,
        )
        assert session.session_id.startswith("ses_")
        assert session.event_count == 0
        assert session.auth_failures == 0


class TestEntityProfile:
    """The learned notion of normal for one entity."""

    def test_new_profile_is_cold_start(self) -> None:
        """A profile with no history must default to cold start, not to 'trusted'."""
        profile = EntityProfile(entity_id="user_new", entity_type=EntityType.USER)
        assert profile.cold_start is True
        assert profile.session_count == 0
        assert profile.drift.status is DriftStatus.STABLE

    def test_login_hours_validated(self) -> None:
        with pytest.raises(ValidationError):
            EntityProfile(
                entity_id="u1",
                entity_type=EntityType.USER,
                typical_login_hours=[9, 24],
            )

    def test_login_hours_coerced_to_int(self) -> None:
        profile = EntityProfile(
            entity_id="u1",
            entity_type=EntityType.USER,
            typical_login_hours=[9, 10, 11],
        )
        assert profile.typical_login_hours == [9, 10, 11]

    def test_drift_state_nested(self) -> None:
        profile = EntityProfile(
            entity_id="u1",
            entity_type=EntityType.USER,
            drift=DriftState(psi=0.31, status=DriftStatus.DRIFTING),
        )
        assert profile.drift.psi == pytest.approx(0.31)
        assert profile.drift.status is DriftStatus.DRIFTING

    def test_json_round_trip(self, sample_geo: GeoLocation) -> None:
        profile = EntityProfile(
            entity_id="u1",
            entity_type=EntityType.USER,
            cohort=3,
            typical_geo=[sample_geo],
            typical_resources={"/api/a": 0.7, "/api/b": 0.3},
            auth_method_dist={"password": 0.9, "mfa": 0.1},
        )
        restored = EntityProfile(**profile.model_dump(mode="json"))
        assert restored.cohort == 3
        assert restored.typical_geo[0].country == sample_geo.country


class TestExplanation:
    """Explainability payload attached to every detection."""

    def test_defaults_are_empty_not_none(self) -> None:
        explanation = Explanation()
        assert explanation.top_features == []
        assert explanation.mitre == []
        assert explanation.narrative_source == "template"

    def test_feature_attribution_direction_validated(self) -> None:
        with pytest.raises(ValidationError):
            FeatureAttribution(feature="hour", contribution=0.4, direction="sideways")

    def test_feature_attribution_direction_normalized(self) -> None:
        attribution = FeatureAttribution(
            feature="geo_velocity_kmh",
            contribution=0.62,
            direction="INCREASES_RISK",
        )
        assert attribution.direction == "increases_risk"

    def test_full_explanation_round_trip(self) -> None:
        explanation = Explanation(
            top_features=[
                FeatureAttribution(
                    feature="hour_of_day",
                    value=2,
                    contribution=0.41,
                    baseline_value=10,
                    description="Access at 02:14 versus a usual 09:00-18:00 window",
                )
            ],
            counterfactual=Counterfactual(
                changes=[
                    CounterfactualChange(
                        feature="geo_country", actual="Brazil", suggested="India"
                    )
                ],
                original_risk=91.0,
                resulting_risk=18.0,
                found=True,
                summary="Would be benign from India at 10:00",
            ),
            sequence_attribution=[
                SequenceStepAttribution(position=0, token="whoami", score=0.8)
            ],
            mitre=[
                MitreTechnique(
                    technique_id="T1078", name="Valid Accounts", tactic="Defense Evasion"
                )
            ],
            baseline_comparison=BaselineComparison(
                fields={"country": {"observed": "Brazil", "typical": "India"}}
            ),
            narrative="Off-hours access from an unseen country.",
        )
        restored = Explanation(**explanation.model_dump(mode="json"))
        assert restored.counterfactual is not None
        assert restored.counterfactual.found is True
        assert restored.mitre[0].technique_id == "T1078"
        assert restored.top_features[0].feature == "hour_of_day"


class TestDetection:
    """The scored output: risk, type, explanation, campaign link."""

    def test_defaults_are_benign(self) -> None:
        """An unscored detection must default to 'normal', never to an alert."""
        detection = Detection(
            entity_id="u1", entity_type=EntityType.USER, timestamp=utc_now()
        )
        assert detection.risk_score == 0.0
        assert detection.is_anomaly is False
        assert detection.in_alert_budget is False
        assert detection.anomaly_type is AnomalyType.NORMAL
        assert detection.status is DetectionStatus.NEW
        assert detection.detection_id.startswith("det_")

    def test_risk_score_upper_bound_enforced(self) -> None:
        with pytest.raises(ValidationError):
            Detection(
                entity_id="u1",
                entity_type=EntityType.USER,
                timestamp=utc_now(),
                risk_score=101.0,
            )

    @pytest.mark.parametrize(
        ("score", "band"),
        [(0.0, "low"), (39.9, "low"), (40.0, "medium"), (60.0, "high"), (95.0, "critical")],
    )
    def test_risk_band_thresholds(self, score: float, band: str) -> None:
        detection = Detection(
            entity_id="u1",
            entity_type=EntityType.USER,
            timestamp=utc_now(),
            risk_score=score,
        )
        assert detection.risk_band == band

    def test_scores_bounded_to_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            DetectionScores(baseline=1.4)

    def test_full_detection_round_trip(self, fixed_timestamp: datetime) -> None:
        detection = Detection(
            entity_id="user_0001",
            entity_type=EntityType.USER,
            timestamp=fixed_timestamp,
            event_ref="evt_1",
            scores=DetectionScores(
                baseline=0.82, sequence=0.74, classifier_confidence=0.91, fused_raw=0.83
            ),
            risk_score=88.5,
            risk_uncertainty=4.2,
            in_alert_budget=True,
            is_anomaly=True,
            anomaly_type=AnomalyType.IMPOSSIBLE_TRAVEL,
            anomaly_type_probs={"impossible_travel": 0.91, "normal": 0.04},
            detector_hits=["impossible_travel"],
            campaign_id="cmp_abc",
            cold_start=False,
            ground_truth_label=AnomalyType.IMPOSSIBLE_TRAVEL,
        )
        payload = detection.model_dump(mode="json")

        assert payload["anomaly_type"] == "impossible_travel"
        assert isinstance(payload["timestamp"], str)

        restored = Detection(**payload)
        assert restored.risk_score == pytest.approx(88.5)
        assert restored.risk_band == "critical"
        assert restored.detector_hits == ["impossible_travel"]

    def test_ground_truth_is_separate_from_prediction(self) -> None:
        """Eval-only truth must be an independent field from the model's own verdict."""
        detection = Detection(
            entity_id="u1",
            entity_type=EntityType.USER,
            timestamp=utc_now(),
            anomaly_type=AnomalyType.NORMAL,
            ground_truth_label=AnomalyType.LOW_AND_SLOW_EXFIL,
        )
        assert detection.anomaly_type is not detection.ground_truth_label


class TestCampaign:
    """Multi-stage attack storylines (D1)."""

    def test_defaults(self, fixed_timestamp: datetime) -> None:
        campaign = Campaign(
            entity_id="u1", started_at=fixed_timestamp, last_activity=fixed_timestamp
        )
        assert campaign.campaign_id.startswith("cmp_")
        assert campaign.status is CampaignStatus.OPEN
        assert campaign.stage_count == 0

    def test_stage_count_tracks_stages(self, fixed_timestamp: datetime) -> None:
        campaign = Campaign(
            entity_id="u1",
            started_at=fixed_timestamp,
            last_activity=fixed_timestamp,
            stages=[
                CampaignStage(
                    anomaly_type=AnomalyType.BRUTE_FORCE,
                    detection_id="det_1",
                    timestamp=fixed_timestamp,
                    risk_score=71.0,
                ),
                CampaignStage(
                    anomaly_type=AnomalyType.LATERAL_MOVEMENT,
                    detection_id="det_2",
                    timestamp=fixed_timestamp,
                    risk_score=84.0,
                ),
            ],
            kill_chain=["brute_force", "lateral_movement"],
            max_risk=84.0,
        )
        assert campaign.stage_count == 2
        assert Campaign(**campaign.model_dump(mode="json")).stage_count == 2


class TestFeedback:
    """Analyst feedback loop (D6)."""

    def test_verdict_enum(self) -> None:
        feedback = Feedback(
            detection_id="det_1",
            entity_id="u1",
            analyst_verdict=AnalystVerdict.FALSE_POSITIVE,
        )
        assert feedback.analyst_verdict is AnalystVerdict.FALSE_POSITIVE
        assert feedback.feedback_id.startswith("fbk_")

    def test_invalid_verdict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Feedback(detection_id="d", entity_id="u", analyst_verdict="maybe")

    def test_applied_adjustment_round_trip(self) -> None:
        feedback = Feedback(
            detection_id="det_1",
            entity_id="u1",
            analyst_verdict=AnalystVerdict.CONFIRMED,
            applied=FeedbackAdjustment(
                scope="entity",
                scope_id="u1",
                adjustment=-5.0,
                previous_value=0.0,
                new_value=-5.0,
            ),
        )
        restored = Feedback(**feedback.model_dump(mode="json"))
        assert restored.applied is not None
        assert restored.applied.adjustment == pytest.approx(-5.0)


class TestModelMetrics:
    """Evaluation payload -- the numbers that go in the final report."""

    def test_class_order_defaults_to_canonical_list(self) -> None:
        assert ModelMetrics().class_order == ANOMALY_CLASSES

    def test_metric_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetrics(pr_auc=1.2)

    def test_full_round_trip(self) -> None:
        metrics = ModelMetrics(
            dataset_summary=DatasetSummary(
                n_events=100_000,
                n_entities=500,
                anomaly_rate=0.02,
                per_class_counts={"normal": 98_000, "brute_force": 400},
                split="test",
            ),
            pr_auc=0.93,
            roc_auc=0.97,
            recall_at_1pct_budget=0.84,
            macro_f1=0.88,
            calibration_ece=0.03,
            confusion_matrix=[[1, 0], [0, 1]],
            per_class={"brute_force": {"precision": 0.95, "recall": 0.91}},
            coldstart=ColdStartMetrics(
                recall_with_priors=0.74, recall_without_priors=0.41, uplift=0.33
            ),
        )
        restored = ModelMetrics(**metrics.model_dump(mode="json"))
        assert restored.pr_auc == pytest.approx(0.93)
        assert restored.coldstart.uplift == pytest.approx(0.33)
        assert restored.dataset_summary.n_events == 100_000

    def test_metrics_are_optional_before_a_run(self) -> None:
        """A fresh metrics document must be constructible with nothing measured yet."""
        metrics = ModelMetrics()
        assert metrics.pr_auc is None
        assert metrics.run_id.startswith("run_")


class TestHealthPayloads:
    """Health responses consumed by the dashboard."""

    def test_service_health_defaults(self) -> None:
        health = ServiceHealth(service="api")
        assert health.status == "ok"
        assert health.artifacts_ready is False
        assert health.dependencies == {}

    def test_nested_dependency_round_trip(self) -> None:
        health = ServiceHealth(
            service="serving",
            dependencies={
                "mongodb": HealthStatus(status="ok", latency_ms=3.4),
                "redis": HealthStatus(status="disabled", detail="streaming disabled"),
            },
        )
        restored = ServiceHealth(**health.model_dump(mode="json"))
        assert restored.dependencies["mongodb"].latency_ms == pytest.approx(3.4)
        assert restored.dependencies["redis"].status == "disabled"


class TestSerializationContract:
    """Anything persisted must survive a JSON round trip unchanged."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DetectionScores(baseline=0.5),
            lambda: DriftState(psi=0.1),
            lambda: Counterfactual(found=True),
            lambda: MitreTechnique(technique_id="T1110", name="Brute Force"),
            lambda: DatasetSummary(n_events=10),
        ],
    )
    def test_models_round_trip_through_json_mode(self, factory) -> None:
        original = factory()
        assert type(original)(**original.model_dump(mode="json")) == original

    def test_datetimes_serialize_to_iso_strings(self) -> None:
        detection = Detection(
            entity_id="u1",
            entity_type=EntityType.USER,
            timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        payload = detection.model_dump(mode="json")
        assert payload["timestamp"].startswith("2026-01-01T12:00:00")
        assert isinstance(payload["created_at"], str)
