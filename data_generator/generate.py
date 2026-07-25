"""Dataset generation CLI -- the entry point for Deliverable #1.

Run it with::

    python -m data_generator.generate --seed 42

and it writes, under ``artifacts/dataset/``:

======================  ====================================================
``events.parquet``      Feature-bearing telemetry. **No labels.**
``labels.parquet``      Ground truth, joined by ``event_id``.
``entities.json``       Entity profiles, cohorts, cold-start and drift truth.
``campaigns.json``      Multi-stage campaign ground truth (D1).
``metadata.json``       Generation config, split boundaries, class counts.
======================  ====================================================

Two decisions here matter more than the code.

**Labels live in a separate file.** Not a separate column -- a separate file. Feature code
physically cannot read a label by accident, which is the cheapest possible insurance against
target leakage.

**The split is by time, not at random.** Training on events that happen after the test events
would let the model exploit information no production system could have, and would produce
metrics we could not honestly report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.artifacts import dataset_path
from common.config import settings
from common.models import ANOMALY_CLASSES, AnomalyType, Event
from common.seed import set_global_seed
from data_generator.attacks import AttackIncident, attack_summary, inject_attacks
from data_generator.campaigns import GeneratedCampaign, campaign_summary, generate_campaigns
from data_generator.drift import assign_drift_plans, drift_summary
from data_generator.normal import benign_summary, generate_benign_events
from data_generator.profiles import COHORTS, GeneratorConfig, World, build_world

logger = logging.getLogger(__name__)

EVENTS_FILE = "events.parquet"
LABELS_FILE = "labels.parquet"
ENTITIES_FILE = "entities.json"
CAMPAIGNS_FILE = "campaigns.json"
METADATA_FILE = "metadata.json"

#: Columns of ``events.parquet``. Flattened from the nested :class:`Event` model because
#: flat columns are far easier to featurize and to inspect; :func:`dataframe_to_events`
#: reverses it losslessly.
EVENT_COLUMNS: Tuple[str, ...] = (
    "event_id",
    "entity_id",
    "entity_type",
    "timestamp",
    "source_ip",
    "geo_country",
    "geo_city",
    "geo_lat",
    "geo_lon",
    "resource_accessed",
    "auth_method",
    "auth_success",
    "session_id",
    "session_duration",
    "command_sequence",
    "device_os",
    "device_mac",
    "device_protocol",
    "device_user_agent",
    "bytes_out",
    "bytes_in",
    "split",
)

LABEL_COLUMNS: Tuple[str, ...] = (
    "event_id",
    "label",
    "is_anomaly",
    "campaign_id",
    "stage",
    "split",
)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def assign_split(timestamp: datetime, config: GeneratorConfig) -> str:
    """Map a timestamp to ``train`` | ``val`` | ``test`` by time order."""
    train_end, val_end = config.split_boundaries()
    if timestamp < train_end:
        return "train"
    if timestamp < val_end:
        return "val"
    return "test"


def benign_split_weights(
    events: Sequence[Event], config: GeneratorConfig
) -> Dict[str, float]:
    """Share of benign events falling in each split.

    Used to place attacks so every split ends up with comparable anomaly density.
    """
    counts = {"train": 0, "val": 0, "test": 0}
    for event in events:
        counts[assign_split(event.timestamp, config)] += 1
    total = sum(counts.values()) or 1
    return {name: value / total for name, value in counts.items()}


def events_to_dataframe(events: Sequence[Event], config: GeneratorConfig) -> pd.DataFrame:
    """Flatten events into the feature-bearing table. Drops all ground truth."""
    rows: List[Dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "event_id": event.event_id,
                "entity_id": event.entity_id,
                "entity_type": event.entity_type.value,
                "timestamp": event.timestamp,
                "source_ip": event.source_ip,
                "geo_country": event.geo.country,
                "geo_city": event.geo.city,
                "geo_lat": event.geo.lat,
                "geo_lon": event.geo.lon,
                "resource_accessed": event.resource_accessed,
                "auth_method": event.auth_method.value,
                "auth_success": bool(event.auth_success),
                "session_id": event.session_id,
                "session_duration": float(event.session_duration),
                "command_sequence": list(event.command_sequence),
                "device_os": event.device_fingerprint.os,
                "device_mac": event.device_fingerprint.mac,
                "device_protocol": event.device_fingerprint.protocol,
                "device_user_agent": event.device_fingerprint.user_agent,
                "bytes_out": float(event.bytes_out),
                "bytes_in": float(event.bytes_in),
                "split": assign_split(event.timestamp, config),
            }
        )
    frame = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def labels_to_dataframe(events: Sequence[Event], config: GeneratorConfig) -> pd.DataFrame:
    """Build the ground-truth table, keyed by ``event_id``."""
    rows: List[Dict[str, Any]] = []
    for event in events:
        label = (event.label or AnomalyType.NORMAL).value
        rows.append(
            {
                "event_id": event.event_id,
                "label": label,
                "is_anomaly": label != AnomalyType.NORMAL.value,
                "campaign_id": event.campaign_id,
                "stage": event.stage,
                "split": assign_split(event.timestamp, config),
            }
        )
    frame = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    frame["stage"] = frame["stage"].astype("Int64")
    return frame


def dataframe_to_events(frame: pd.DataFrame) -> List[Event]:
    """Rebuild :class:`Event` objects from the flat table.

    Used by the training and replay paths so exactly one event shape flows through the
    system, and by the tests to prove the flattening is lossless.
    """
    events: List[Event] = []
    for record in frame.to_dict(orient="records"):
        timestamp = record["timestamp"]
        events.append(
            Event(
                event_id=record["event_id"],
                entity_id=record["entity_id"],
                entity_type=record["entity_type"],
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
                source_ip=record["source_ip"],
                geo={
                    "country": record["geo_country"],
                    "city": record["geo_city"],
                    "lat": record["geo_lat"],
                    "lon": record["geo_lon"],
                },
                resource_accessed=record["resource_accessed"],
                auth_method=record["auth_method"],
                auth_success=bool(record["auth_success"]),
                session_id=record["session_id"],
                session_duration=record["session_duration"],
                command_sequence=list(record["command_sequence"]),
                device_fingerprint={
                    "os": record["device_os"],
                    "mac": record["device_mac"],
                    "protocol": record["device_protocol"],
                    "user_agent": record["device_user_agent"],
                },
                bytes_out=record["bytes_out"],
                bytes_in=record["bytes_in"],
            )
        )
    return events


def entities_to_records(world: World) -> List[Dict[str, Any]]:
    """Serialize entity ground truth, including cohort, cold-start and drift plans."""
    records: List[Dict[str, Any]] = []
    for entity in world.entities:
        records.append(
            {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "cohort_id": entity.cohort_id,
                "cohort_name": entity.cohort.name,
                "home_city": entity.home_city.name,
                "home_country": entity.home_city.country,
                "home_lat": entity.home_city.lat,
                "home_lon": entity.home_city.lon,
                "secondary_city": entity.secondary_city.name,
                "ip_prefix": entity.ip_prefix,
                "devices": [device.as_dict() for device in entity.devices],
                "typical_resources": entity.resource_weights,
                "auth_distribution": {
                    method.value: weight for method, weight in entity.auth_weights.items()
                },
                "sessions_per_day": entity.sessions_per_day,
                "auth_fail_rate": entity.auth_fail_rate,
                "active_from": entity.active_from.isoformat(),
                "is_coldstart": entity.is_coldstart,
                "drift": entity.drift_plan.as_dict() if entity.drift_plan else None,
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_dataset(
    config: Optional[GeneratorConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, World, List[GeneratedCampaign], Dict[str, Any]]:
    """Generate the complete labeled dataset in memory.

    Returns
    -------
    events, labels, world, campaigns, metadata
        ``events`` carries no ground truth; ``labels`` is joined by ``event_id``.
    """
    config = config or GeneratorConfig()
    set_global_seed(config.seed)

    # Independent child streams per stage: changing the attack mix does not reshuffle the
    # benign traffic, which keeps datasets comparable across tuning iterations.
    root = np.random.default_rng(config.seed)
    world_rng, drift_rng, benign_rng, attack_rng, campaign_rng = root.spawn(5)

    logger.info("Building world: %d entities, %d days", config.n_entities, config.resolved_days())
    world = build_world(config, world_rng)
    assign_drift_plans(world, drift_rng, config)

    logger.info("Generating benign traffic")
    benign_events = generate_benign_events(world, benign_rng, config)
    benign_count = len(benign_events)
    if benign_count == 0:
        raise RuntimeError("Benign generation produced no events; check the configuration.")

    # Solve for the attack budget that lands on the target rate:
    #   rate = A / (A + B)  =>  A = rate * B / (1 - rate)
    rate = float(np.clip(config.target_anomaly_rate, 1e-6, 0.5))
    total_attack_events = int(round(rate * benign_count / (1.0 - rate)))
    campaign_budget = int(round(total_attack_events * config.campaign_budget_fraction))
    standalone_budget = max(0, total_attack_events - campaign_budget)

    logger.info(
        "Attack budget: %d events (%d standalone, %d campaign) against %d benign",
        total_attack_events,
        standalone_budget,
        campaign_budget,
        benign_count,
    )

    # Weight attack placement by the *actual* benign volume per split, not by elapsed time.
    # Cold-start entities are onboarded late, so the later splits hold more events than their
    # share of the calendar; weighting by time would leave validation thin on anomalies.
    split_weights = benign_split_weights(benign_events, config)
    logger.info(
        "Benign split shares: %s",
        {name: round(value, 3) for name, value in split_weights.items()},
    )

    campaigns = generate_campaigns(
        world, campaign_rng, config, campaign_budget, split_weights=split_weights
    )
    incidents: List[AttackIncident] = inject_attacks(
        world, attack_rng, config, standalone_budget, split_weights=split_weights
    )

    attack_events: List[Event] = [event for incident in incidents for event in incident.events]
    campaign_events: List[Event] = [
        event for generated in campaigns for event in generated.events
    ]

    all_events = benign_events + attack_events + campaign_events

    # Defensive: no event may fall outside the declared timeline. Long-running incidents are
    # already sized to fit, but if any path ever produced a stray event the split boundaries
    # and the metadata would silently stop describing the data.
    timeline_end = config.end_date()
    inside = [
        event
        for event in all_events
        if config.start_date <= event.timestamp < timeline_end
    ]
    if len(inside) != len(all_events):
        logger.warning(
            "Dropped %d event(s) outside the %s..%s timeline",
            len(all_events) - len(inside),
            config.start_date.date(),
            timeline_end.date(),
        )
    all_events = inside
    all_events.sort(key=lambda event: (event.timestamp, event.event_id))

    events_frame = events_to_dataframe(all_events, config)
    labels_frame = labels_to_dataframe(all_events, config)

    metadata = build_metadata(
        config=config,
        world=world,
        events=all_events,
        benign_events=benign_events,
        incidents=incidents,
        campaigns=campaigns,
        labels=labels_frame,
    )
    return events_frame, labels_frame, world, campaigns, metadata


def build_metadata(
    *,
    config: GeneratorConfig,
    world: World,
    events: Sequence[Event],
    benign_events: Sequence[Event],
    incidents: Sequence[AttackIncident],
    campaigns: Sequence[GeneratedCampaign],
    labels: pd.DataFrame,
) -> Dict[str, Any]:
    """Assemble the dataset description written to ``metadata.json``."""
    train_end, val_end = config.split_boundaries()
    per_class = labels["label"].value_counts().to_dict()
    anomaly_count = int(labels["is_anomaly"].sum())

    split_counts = labels.groupby("split")["is_anomaly"].agg(["count", "sum"]).to_dict("index")
    per_split = {
        split: {
            "n_events": int(values["count"]),
            "n_anomalies": int(values["sum"]),
            "anomaly_rate": float(values["sum"]) / float(values["count"]) if values["count"] else 0.0,
        }
        for split, values in split_counts.items()
    }

    return {
        "seed": config.seed,
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": {
            "n_entities": config.n_entities,
            "days": config.resolved_days(),
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date().isoformat(),
            "target_anomaly_rate": config.target_anomaly_rate,
            "subtlety": config.subtlety,
            "campaign_budget_fraction": config.campaign_budget_fraction,
            "coldstart_entity_fraction": config.coldstart_entity_fraction,
            "drift_entity_fraction": config.drift_entity_fraction,
            "attack_class_weights": config.attack_class_weights,
        },
        "splits": {
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
            "per_split": per_split,
        },
        "totals": {
            "n_events": len(events),
            "n_benign": len(benign_events),
            "n_anomalies": anomaly_count,
            "anomaly_rate": anomaly_count / len(events) if events else 0.0,
            "n_entities": len(world.entities),
        },
        "per_class_counts": {name: int(per_class.get(name, 0)) for name in ANOMALY_CLASSES},
        "cohorts": [
            {
                "cohort_id": cohort.cohort_id,
                "name": cohort.name,
                "entity_type": cohort.entity_type.value,
                "n_entities": len(world.cohort_members(cohort.cohort_id)),
            }
            for cohort in COHORTS
        ],
        "coldstart": {
            "n_entities": sum(1 for entity in world.entities if entity.is_coldstart),
            "appears_after": (
                min(
                    (entity.active_from for entity in world.entities if entity.is_coldstart),
                    default=config.start_date,
                ).isoformat()
            ),
        },
        "drift": drift_summary(world),
        "benign": benign_summary(benign_events),
        "attacks": attack_summary(incidents),
        "campaigns": campaign_summary(campaigns),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_dataset(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    world: World,
    campaigns: Sequence[GeneratedCampaign],
    metadata: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write every dataset artifact to disk and return the paths written."""
    if output_dir is None:
        target = dataset_path(EVENTS_FILE).parent
    else:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

    paths = {
        "events": target / EVENTS_FILE,
        "labels": target / LABELS_FILE,
        "entities": target / ENTITIES_FILE,
        "campaigns": target / CAMPAIGNS_FILE,
        "metadata": target / METADATA_FILE,
    }

    events.to_parquet(paths["events"], index=False)
    labels.to_parquet(paths["labels"], index=False)

    with paths["entities"].open("w", encoding="utf-8") as handle:
        json.dump(entities_to_records(world), handle, indent=2)
        handle.write("\n")

    with paths["campaigns"].open("w", encoding="utf-8") as handle:
        json.dump([generated.as_dict() for generated in campaigns], handle, indent=2)
        handle.write("\n")

    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
        handle.write("\n")

    return paths


def load_events(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read ``events.parquet``. Raises a clear error if the dataset is missing."""
    target = Path(output_dir) if output_dir else dataset_path(EVENTS_FILE).parent
    path = target / EVENTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. Run: python -m data_generator.generate --seed 42"
        )
    return pd.read_parquet(path)


def load_labels(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read ``labels.parquet``."""
    target = Path(output_dir) if output_dir else dataset_path(LABELS_FILE).parent
    path = target / LABELS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No labels at {path}. Run: python -m data_generator.generate --seed 42"
        )
    return pd.read_parquet(path)


def load_metadata(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``metadata.json``."""
    target = Path(output_dir) if output_dir else dataset_path(METADATA_FILE).parent
    with (target / METADATA_FILE).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_report(metadata: Dict[str, Any]) -> str:
    """Human-readable summary printed after generation."""
    totals = metadata["totals"]
    lines = [
        "",
        "=" * 74,
        " Synthetic behavioral dataset",
        "=" * 74,
        f" seed                 : {metadata['seed']}",
        f" entities             : {totals['n_entities']}",
        f" timeline             : {metadata['config']['days']} days "
        f"({metadata['config']['start_date'][:10]} -> {metadata['config']['end_date'][:10]})",
        f" events               : {totals['n_events']:,}",
        f" anomalies            : {totals['n_anomalies']:,} "
        f"({totals['anomaly_rate'] * 100:.2f}%)  target "
        f"{metadata['config']['target_anomaly_rate'] * 100:.2f}%",
        f" subtlety             : {metadata['config']['subtlety']}",
        "",
        " per-class counts",
        " " + "-" * 40,
    ]
    for name, count in metadata["per_class_counts"].items():
        share = count / totals["n_events"] * 100 if totals["n_events"] else 0.0
        lines.append(f"   {name:<24} {count:>8,}  {share:>6.2f}%")

    lines += [
        "",
        " splits (by time)",
        " " + "-" * 40,
    ]
    for split in ("train", "val", "test"):
        values = metadata["splits"]["per_split"].get(split)
        if values:
            lines.append(
                f"   {split:<8} {values['n_events']:>8,} events   "
                f"{values['n_anomalies']:>6,} anomalies  "
                f"({values['anomaly_rate'] * 100:.2f}%)"
            )

    campaigns = metadata["campaigns"]
    drift = metadata["drift"]
    coldstart = metadata["coldstart"]
    lines += [
        "",
        " campaigns (D1)",
        " " + "-" * 40,
        f"   campaigns            : {campaigns['n_campaigns']}",
        f"   events in campaigns  : {campaigns['n_events']:,}",
        f"   mean stages          : {campaigns['mean_stages']:.2f}",
        f"   by template          : {campaigns['by_template']}",
        "",
        " benign drift (D3)",
        " " + "-" * 40,
        f"   drifted entities     : {drift['n_drifted_entities']} "
        f"({drift['fraction_of_population'] * 100:.1f}% of population)",
        f"   by kind              : {drift['by_kind']}",
        "",
        " cold start",
        " " + "-" * 40,
        f"   late-arriving        : {coldstart['n_entities']} entities",
        "=" * 74,
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        prog="python -m data_generator.generate",
        description="Generate the synthetic behavioral anomaly detection dataset.",
    )
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="Global seed.")
    parser.add_argument("--entities", type=int, default=None, help="Number of entities.")
    parser.add_argument("--days", type=int, default=None, help="Simulated timeline length.")
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=None,
        help="Target anomaly rate, e.g. 0.02 (valid range 0.005-0.03).",
    )
    parser.add_argument(
        "--subtlety",
        type=float,
        default=None,
        help="0 = blatant attacks, 1 = nearly invisible. Default 0.55.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/dataset/.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    config = GeneratorConfig(seed=args.seed)
    if args.entities is not None:
        config.n_entities = args.entities
    if args.days is not None:
        config.days = args.days
    if args.anomaly_rate is not None:
        config.target_anomaly_rate = args.anomaly_rate
    if args.subtlety is not None:
        config.subtlety = args.subtlety

    events, labels, world, campaigns, metadata = generate_dataset(config)
    paths = write_dataset(events, labels, world, campaigns, metadata, args.out)

    if not args.quiet:
        print(format_report(metadata))
        print(" written:")
        for name, path in paths.items():
            print(f"   {name:<10} {path}")
        print()

    # Fail loudly in dev: an out-of-range anomaly rate would silently invalidate every
    # metric downstream, so the generator refuses to pretend it succeeded.
    rate = metadata["totals"]["anomaly_rate"]
    if not 0.005 <= rate <= 0.030:
        logger.error(
            "Anomaly rate %.4f is outside the required 0.5%%-3.0%% band. "
            "Adjust --anomaly-rate or the class weights.",
            rate,
        )
        return 1

    missing = [
        name
        for name, count in metadata["per_class_counts"].items()
        if count == 0 and name != AnomalyType.NORMAL.value
    ]
    if missing:
        logger.error("Attack classes absent from the dataset: %s", missing)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
