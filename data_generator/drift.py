"""Benign concept drift baked into the timeline (D3).

Real behavioral baselines go stale. People change shifts, move office, get a new laptop;
services get migrated. A detector that treats every such change as an intrusion is useless
in week three.

So drift is generated **in the data itself**, labeled ``normal``, rather than being
simulated later at evaluation time. A subset of entities gradually changes its schedule,
location or device partway through the timeline. That gives us something honest to prove
against in Phase 6/9: PSI should rise and then settle as the baseline re-profiles, and the
false-positive rate should return to near baseline -- on data the models never saw as
benign-by-construction.

The drift is **gradual** (ramped over days) on purpose. An abrupt benign shift is
indistinguishable from an attack in a single event; only the sustained, slow character of
the change makes adaptation the right response.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from data_generator.profiles import (
    HOME_CITIES,
    SHIFTED_HOURS,
    City,
    DeviceSpec,
    EntityGenProfile,
    GeneratorConfig,
    World,
    random_mac,
    user_agent_for,
)


class DriftKind(str, Enum):
    """The kind of benign change an entity undergoes."""

    SCHEDULE = "schedule"
    LOCATION = "location"
    DEVICE = "device"
    RESOURCE = "resource"


@dataclass
class DriftPlan:
    """How, when and how quickly one entity's normal behavior changes."""

    kind: DriftKind
    starts_at: datetime
    ramp_days: float
    new_hour_weights: Optional[Tuple[float, ...]] = None
    new_city: Optional[City] = None
    new_device: Optional[DeviceSpec] = None
    new_resources: Tuple[str, ...] = ()

    def progress(self, moment: datetime) -> float:
        """Fraction of the change in effect at ``moment``, in ``[0, 1]``.

        Linear ramp: 0 before the change begins, 1 once fully adopted.
        """
        if moment < self.starts_at:
            return 0.0
        if self.ramp_days <= 0:
            return 1.0
        elapsed_days = (moment - self.starts_at).total_seconds() / 86400.0
        return float(min(1.0, elapsed_days / self.ramp_days))

    def as_dict(self) -> dict:
        """JSON-serializable description, written to ``entities.json`` as ground truth."""
        return {
            "kind": self.kind.value,
            "starts_at": self.starts_at.isoformat(),
            "ramp_days": self.ramp_days,
            "new_city": self.new_city.name if self.new_city else None,
            "new_country": self.new_city.country if self.new_city else None,
            "new_device_os": self.new_device.os if self.new_device else None,
            "new_resources": list(self.new_resources),
            "shifts_schedule": self.new_hour_weights is not None,
        }


def assign_drift_plans(
    world: World, rng: np.random.Generator, config: Optional[GeneratorConfig] = None
) -> List[EntityGenProfile]:
    """Give a subset of entities a benign drift plan, in place.

    Cold-start entities are excluded: an entity that only appears near the end of the
    timeline has no established baseline to drift away from, and mixing the two would
    confound the cold-start and drift experiments.

    Returns
    -------
    list
        The entities that received a plan, so callers can report how many drifted.
    """
    config = config or world.config
    timeline_days = config.resolved_days()
    drift_start = config.start_date + timedelta(days=timeline_days * config.drift_start_point)

    eligible = [entity for entity in world.entities if not entity.is_coldstart]
    if not eligible:
        return []

    count = int(round(len(world.entities) * config.drift_entity_fraction))
    count = min(count, len(eligible))
    if count <= 0:
        return []

    chosen_indices = rng.choice(len(eligible), size=count, replace=False)
    kinds = list(DriftKind)
    drifted: List[EntityGenProfile] = []

    for raw_index in chosen_indices:
        entity = eligible[int(raw_index)]
        kind = kinds[int(rng.integers(0, len(kinds)))]

        # Stagger onset so the whole population does not change on the same day, which
        # would make drift trivially detectable as a global step.
        jitter_days = float(rng.uniform(-2.5, 4.0))
        starts_at = drift_start + timedelta(days=jitter_days)
        ramp_days = float(max(2.0, config.drift_ramp_days * rng.uniform(0.7, 1.3)))

        plan = DriftPlan(kind=kind, starts_at=starts_at, ramp_days=ramp_days)

        if kind is DriftKind.SCHEDULE:
            # Moved to a later shift: the same person, working different hours.
            plan.new_hour_weights = SHIFTED_HOURS
        elif kind is DriftKind.LOCATION:
            options = [city for city in HOME_CITIES if city.name != entity.home_city.name]
            plan.new_city = options[int(rng.integers(0, len(options)))]
        elif kind is DriftKind.DEVICE:
            cohort = entity.cohort
            os_name = cohort.os_pool[int(rng.integers(0, len(cohort.os_pool)))]
            plan.new_device = DeviceSpec(
                os=os_name,
                mac=random_mac(rng),
                protocol=cohort.protocols[int(rng.integers(0, len(cohort.protocols)))],
                user_agent=user_agent_for(os_name),
            )
        else:  # DriftKind.RESOURCE -- took on new responsibilities
            new_resources = world.foreign_resources(entity, rng, count=int(rng.integers(2, 4)))
            plan.new_resources = tuple(new_resources)
            if not plan.new_resources:
                plan.kind = DriftKind.SCHEDULE
                plan.new_hour_weights = SHIFTED_HOURS

        entity.drift_plan = plan
        drifted.append(entity)

    return drifted


def effective_hour_weights(entity: EntityGenProfile, moment: datetime) -> Tuple[float, ...]:
    """Hour-of-day distribution for this entity at this point in time.

    Blends the original and post-drift schedules by ramp progress, so the shift is gradual
    rather than a step change.
    """
    plan = entity.drift_plan
    if plan is None or plan.new_hour_weights is None:
        return entity.hour_weights

    share = plan.progress(moment)
    if share <= 0.0:
        return entity.hour_weights

    blended = [
        (1.0 - share) * original + share * shifted
        for original, shifted in zip(entity.hour_weights, plan.new_hour_weights)
    ]
    total = sum(blended)
    return tuple(value / total for value in blended)


def effective_city(entity: EntityGenProfile, moment: datetime, rng: np.random.Generator) -> City:
    """Home location for this entity at this point in time.

    During a location drift the entity connects from the new city with probability equal to
    the ramp progress -- a relocation with a tail of trips back to the old office.
    """
    plan = entity.drift_plan
    if plan is None or plan.new_city is None:
        return entity.home_city

    share = plan.progress(moment)
    if share > 0.0 and rng.random() < share:
        return plan.new_city
    return entity.home_city


def effective_device(
    entity: EntityGenProfile, moment: datetime, rng: np.random.Generator
) -> DeviceSpec:
    """Device for this entity at this point in time.

    A hardware refresh: the new laptop takes over progressively while the old one is still
    occasionally used.
    """
    plan = entity.drift_plan
    if plan is not None and plan.new_device is not None:
        share = plan.progress(moment)
        if share > 0.0 and rng.random() < share:
            return plan.new_device

    devices = entity.devices
    if len(devices) == 1:
        return devices[0]
    # Primary device dominates; secondary is a genuine but minority choice.
    return devices[0] if rng.random() < 0.82 else devices[int(rng.integers(1, len(devices)))]


def drifted_resource_pool(entity: EntityGenProfile, moment: datetime) -> Tuple[str, ...]:
    """Extra resources this entity has legitimately started using, if any."""
    plan = entity.drift_plan
    if plan is None or not plan.new_resources:
        return ()
    return plan.new_resources if plan.progress(moment) > 0.25 else ()


def drift_summary(world: World) -> dict:
    """Aggregate description of the drift baked into this dataset."""
    drifted = [entity for entity in world.entities if entity.drift_plan is not None]
    by_kind: dict = {}
    for entity in drifted:
        assert entity.drift_plan is not None
        by_kind[entity.drift_plan.kind.value] = by_kind.get(entity.drift_plan.kind.value, 0) + 1
    return {
        "n_drifted_entities": len(drifted),
        "fraction_of_population": (
            len(drifted) / len(world.entities) if world.entities else 0.0
        ),
        "by_kind": by_kind,
        "entity_ids": sorted(entity.entity_id for entity in drifted),
        "all_labeled_benign": True,
    }


__all__ = [
    "DriftKind",
    "DriftPlan",
    "assign_drift_plans",
    "effective_hour_weights",
    "effective_city",
    "effective_device",
    "drifted_resource_pool",
    "drift_summary",
]
