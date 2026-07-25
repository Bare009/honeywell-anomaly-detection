"""Correlated multi-stage attack campaigns (D1).

Real intrusions are not isolated events. An attacker guesses a password, logs in, looks
around, then takes something. Each step alone might be dismissed; together they are a
story.

This module emits those stories: several attack stages that **share one entity**, occur in
**causal time order**, and carry a shared ``campaign_id`` with a ``stage`` index as ground
truth. That ground truth is what lets Phase 7's campaign reconstruction be *measured*
rather than merely demonstrated -- we can check whether the system relinked the stages the
generator actually related.

A campaign is not a new attack type. It is the existing injectors composed in a realistic
order, so no extra label space is needed and every stage is still individually labeled with
its own anomaly class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.models import AnomalyType, Campaign, CampaignStage, CampaignStatus, EntityType
from data_generator.attacks import (
    INJECTORS,
    AttackIncident,
    run_injector,
    sample_split,
    split_windows,
)
from data_generator.profiles import EntityGenProfile, GeneratorConfig, World


@dataclass(frozen=True)
class CampaignTemplate:
    """An ordered kill chain plus the time gaps between its stages."""

    name: str
    stages: Tuple[str, ...]
    #: Inclusive range, in minutes, for the pause between consecutive stages.
    gap_minutes: Tuple[float, float]
    entity_types: Tuple[EntityType, ...]
    description: str


#: The three campaign shapes in this dataset. Each maps onto a recognizable intrusion
#: pattern, and each ends in an objective (exfiltration or persistence) rather than trailing
#: off, so the storyline view has a payoff to show.
CAMPAIGN_TEMPLATES: Tuple[CampaignTemplate, ...] = (
    CampaignTemplate(
        name="credential_compromise",
        stages=(
            AnomalyType.BRUTE_FORCE.value,
            AnomalyType.CREDENTIAL_MISUSE.value,
            AnomalyType.LATERAL_MOVEMENT.value,
            AnomalyType.LOW_AND_SLOW_EXFIL.value,
        ),
        gap_minutes=(25.0, 600.0),
        entity_types=(EntityType.USER, EntityType.SERVICE_ACCOUNT),
        description=(
            "Password guessed, credentials then used off-hours, attacker pivots across "
            "unrelated systems and finally drains data in small pieces."
        ),
    ),
    CampaignTemplate(
        name="stolen_session",
        stages=(
            AnomalyType.CREDENTIAL_STUFFING.value,
            AnomalyType.IMPOSSIBLE_TRAVEL.value,
            AnomalyType.CREDENTIAL_MISUSE.value,
        ),
        gap_minutes=(15.0, 240.0),
        entity_types=(EntityType.USER,),
        description=(
            "Credentials from a dump are sprayed, one lands, and the account is then used "
            "from the other side of the world while the real user is still working."
        ),
    ),
    CampaignTemplate(
        name="device_takeover",
        stages=(
            AnomalyType.DEVICE_SPOOFING.value,
            AnomalyType.LATERAL_MOVEMENT.value,
            AnomalyType.LOW_AND_SLOW_EXFIL.value,
        ),
        gap_minutes=(20.0, 420.0),
        entity_types=(EntityType.EDGE_DEVICE, EntityType.USER),
        description=(
            "A device identity is impersonated, used to reach systems it never touches, "
            "then to move data out slowly."
        ),
    ),
)


@dataclass
class GeneratedCampaign:
    """A campaign plus the incidents that make up its stages."""

    campaign: Campaign
    incidents: List[AttackIncident] = field(default_factory=list)
    template_name: str = ""

    @property
    def events(self) -> List:
        """Every event across every stage, in time order."""
        collected = [event for incident in self.incidents for event in incident.events]
        collected.sort(key=lambda event: event.timestamp)
        return collected

    def as_dict(self) -> Dict:
        """Ground-truth description written to ``campaigns.json``."""
        payload = self.campaign.model_dump(mode="json")
        payload["template"] = self.template_name
        payload["n_events"] = len(self.events)
        return payload


def _eligible_for_template(
    world: World, template: CampaignTemplate, config: GeneratorConfig
) -> List[EntityGenProfile]:
    """Entities that can host every stage of this template.

    A campaign is only coherent if one entity can plausibly carry the whole chain, so an
    entity must be allowed by *all* the stage injectors, not just the first.
    """
    required_types = set(template.entity_types)
    for stage in template.stages:
        _, allowed = INJECTORS[stage]
        required_types &= set(allowed)

    if not required_types:
        return []

    # Leave room for the full chain plus its final long-running stage.
    latest_start = config.end_date() - timedelta(days=2.5)
    return [
        entity
        for entity in world.entities
        if entity.entity_type in required_types and entity.active_from <= latest_start
    ]


def generate_campaign(
    world: World,
    rng: np.random.Generator,
    config: GeneratorConfig,
    campaign_index: int,
    template: Optional[CampaignTemplate] = None,
    split: Optional[str] = None,
    split_weights: Optional[Dict[str, float]] = None,
) -> Optional[GeneratedCampaign]:
    """Build one multi-stage campaign against a single entity.

    Stage ``n + 1`` always starts after stage ``n`` ends, so the ground-truth ``stage``
    index and the event timestamps never contradict each other.
    """
    template = template or CAMPAIGN_TEMPLATES[int(rng.integers(0, len(CAMPAIGN_TEMPLATES)))]
    candidates = _eligible_for_template(world, template, config)
    if not candidates:
        return None

    entity = candidates[int(rng.integers(0, len(candidates)))]
    campaign_id = f"cmp_{template.name[:6]}_{campaign_index:04d}"

    # Place the campaign's first stage inside a split chosen in proportion to the split
    # sizes, matching how standalone incidents are placed. Later stages run on from there and
    # may cross a split boundary, which is realistic -- an intrusion does not respect our
    # evaluation cut.
    target_split = split or sample_split(rng, config, split_weights)
    window_start, window_end = split_windows(config)[target_split]
    earliest = max(entity.active_from, window_start)
    latest = min(window_end, config.end_date() - timedelta(days=2.5))
    if latest <= earliest:
        # Fall back to anywhere the entity can host a full chain.
        earliest = max(entity.active_from, config.start_date)
        latest = config.end_date() - timedelta(days=2.5)
        if latest <= earliest:
            return None
    stage_start = earliest + timedelta(
        seconds=float(rng.uniform(0, (latest - earliest).total_seconds()))
    )

    incidents: List[AttackIncident] = []
    stages: List[CampaignStage] = []
    kill_chain: List[str] = []

    for stage_index, anomaly_type in enumerate(template.stages):
        incident = run_injector(
            anomaly_type,
            world,
            rng,
            config,
            incident_id=f"{template.name[:4]}{campaign_index:04d}s{stage_index}",
            entity=entity,
            start=stage_start,
        )
        if incident is None:
            continue

        # Some injectors legitimately touch peer entities (credential stuffing sprays a whole
        # cohort). Inside a campaign we keep only the target entity's events: a campaign is
        # defined as one entity's storyline, and tagging a bystander's events with this
        # campaign_id would create ground truth the reconstruction could never reproduce.
        incident.events = [
            event for event in incident.events if event.entity_id == entity.entity_id
        ]
        if not incident.events:
            continue

        incident.tag_campaign(campaign_id, stage_index)
        incidents.append(incident)
        kill_chain.append(anomaly_type)

        stages.append(
            CampaignStage(
                anomaly_type=AnomalyType(anomaly_type),
                # Ground truth points at the stage's first event; detection ids are assigned
                # later by the serving pipeline.
                detection_id=incident.events[0].event_id,
                timestamp=incident.started_at,
                risk_score=0.0,
            )
        )

        low, high = template.gap_minutes
        gap = float(rng.uniform(low, high))
        stage_start = incident.ended_at + timedelta(minutes=gap)

    # A chain runs for days, so a campaign starting late can extend past the declared end of
    # the timeline. Trim rather than allow it: a dataset that claims 45 days must not contain
    # events on day 47, or the split boundaries and metadata stop describing reality.
    timeline_end = config.end_date()
    trimmed_incidents = []
    trimmed_stages = []
    trimmed_chain = []
    for incident, stage, chain_entry in zip(incidents, stages, kill_chain):
        incident.events = [
            event for event in incident.events if event.timestamp < timeline_end
        ]
        if not incident.events:
            continue
        trimmed_incidents.append(incident)
        trimmed_stages.append(stage)
        trimmed_chain.append(chain_entry)

    incidents, stages, kill_chain = trimmed_incidents, trimmed_stages, trimmed_chain

    if len(incidents) < 2:
        # A one-stage "campaign" is just an incident; do not pollute the ground truth.
        return None

    all_events = [event for incident in incidents for event in incident.events]
    campaign = Campaign(
        campaign_id=campaign_id,
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        started_at=min(event.timestamp for event in all_events),
        last_activity=max(event.timestamp for event in all_events),
        stages=stages,
        detection_ids=[event.event_id for event in all_events],
        kill_chain=kill_chain,
        max_risk=0.0,
        status=CampaignStatus.CLOSED,
        ground_truth_campaign_id=campaign_id,
    )

    return GeneratedCampaign(campaign=campaign, incidents=incidents, template_name=template.name)


def generate_campaigns(
    world: World,
    rng: np.random.Generator,
    config: GeneratorConfig,
    target_events: int,
    split_weights: Optional[Dict[str, float]] = None,
) -> List[GeneratedCampaign]:
    """Generate campaigns until the event budget is spent.

    Templates are cycled rather than sampled so every campaign shape is guaranteed to appear
    -- the storyline demo needs at least one of each, and a random draw could omit one.
    """
    if target_events <= 0:
        return []

    windows = split_windows(config)
    if split_weights:
        shares = {name: max(0.0, float(split_weights.get(name, 0.0))) for name in windows}
    else:
        test_fraction = max(0.0, 1.0 - config.train_fraction - config.val_fraction)
        shares = {
            "train": config.train_fraction,
            "val": config.val_fraction,
            "test": test_fraction,
        }
    total_share = sum(shares.values()) or 1.0
    shares = {name: value / total_share for name, value in shares.items()}

    campaigns: List[GeneratedCampaign] = []
    index = 0

    # Budget per split by where the events actually land, with spillover debited from the
    # later split as well. A campaign runs for days, so one that begins near a boundary
    # deposits most of its stages in the following split. Without debiting both, each split
    # fills its own quota and then *also* receives spillover, pushing the overall anomaly rate
    # above target and inflating the later splits.
    remaining = {
        split: int(round(target_events * share)) for split, share in shares.items()
    }

    consecutive_failures = 0
    last_size = 0
    guard = 0

    while guard < 500 and consecutive_failures <= 3 * len(CAMPAIGN_TEMPLATES):
        guard += 1
        open_splits = [name for name in ("train", "val", "test") if remaining[name] > 0]
        if not open_splits:
            break

        # A campaign is a large, indivisible unit (~35 events). Adding one to cover a small
        # remainder would overshoot the target anomaly rate by more than skipping it.
        if last_size and max(remaining[name] for name in open_splits) < last_size * 0.5:
            break

        room = np.asarray([remaining[name] for name in open_splits], dtype=float)
        split = open_splits[int(rng.choice(len(open_splits), p=room / room.sum()))]

        # Cycle templates so every campaign shape is guaranteed to appear; a random draw could
        # omit one, and the storyline demo needs at least one of each.
        template = CAMPAIGN_TEMPLATES[index % len(CAMPAIGN_TEMPLATES)]
        generated = generate_campaign(
            world,
            rng,
            config,
            campaign_index=index,
            template=template,
            split=split,
        )
        index += 1

        if generated is None:
            consecutive_failures += 1
            continue

        consecutive_failures = 0
        campaigns.append(generated)
        last_size = len(generated.events)
        for landed_split, window in windows.items():
            remaining[landed_split] -= sum(
                1
                for event in generated.events
                if window[0] <= event.timestamp < window[1]
            )

    return campaigns


def campaign_summary(campaigns: Sequence[GeneratedCampaign]) -> Dict[str, object]:
    """Aggregate campaign statistics for the taxonomy and the console report."""
    by_template: Dict[str, int] = {}
    for generated in campaigns:
        by_template[generated.template_name] = by_template.get(generated.template_name, 0) + 1

    stage_counts = [len(generated.incidents) for generated in campaigns]
    return {
        "n_campaigns": len(campaigns),
        "n_events": sum(len(generated.events) for generated in campaigns),
        "by_template": by_template,
        "mean_stages": float(np.mean(stage_counts)) if stage_counts else 0.0,
        "templates_available": [template.name for template in CAMPAIGN_TEMPLATES],
    }


__all__ = [
    "CampaignTemplate",
    "CAMPAIGN_TEMPLATES",
    "GeneratedCampaign",
    "generate_campaign",
    "generate_campaigns",
    "campaign_summary",
]
