"""Campaign-reconstruction accuracy on the test split (D1).

Feeds the test split's flagged anomalies through the same campaign linker the serving plane uses,
then compares the reconstructed grouping to the ground-truth ``campaign_id`` the generator assigned.
The headline number is stage-linking accuracy: of the flagged events that truly belong to a
campaign, what fraction were grouped with their campaign-mates (the majority reconstructed campaign
for their true campaign).

Only detection-flagged events are linked, so this measures *linking* quality given detection -- a
stage the detector missed is a detection miss, not a linking error, and is not charged here.

Run it with::

    python -m evaluation.campaign_experiment
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.config import settings
from common.models import AnomalyType, CampaignMetrics, Detection, DetectionScores, EntityType
from common.seed import set_global_seed
from evaluation.scoring import load_models, score_split
from serving.campaign import CampaignLinker
from serving.store import InMemoryStore
from training.train_baseline import load_split

logger = logging.getLogger(__name__)

CAMPAIGN_FILE = "campaign_metrics.json"


async def _link_flagged(scored, type_by_entity, threshold: float) -> Dict[str, Optional[str]]:
    """Link every flagged event in time order; return event_id -> reconstructed campaign id."""
    store = InMemoryStore()
    linker = CampaignLinker(store)
    reconstructed: Dict[str, Optional[str]] = {}

    for index, event_id in enumerate(scored.event_ids):
        detector_fired = any(r.fired for r in scored.detector_results[index])
        is_anomaly = bool(scored.risk[index] >= threshold or detector_fired)
        if not is_anomaly:
            continue
        detection = Detection(
            entity_id=scored.entity_ids[index],
            entity_type=type_by_entity.get(scored.entity_ids[index], EntityType.USER),
            timestamp=scored.vectors[index].timestamp,
            event_ref=event_id,
            risk_score=float(scored.risk[index]),
            is_anomaly=True,
            anomaly_type=AnomalyType(scored.predicted_type[index]),
            scores=DetectionScores(),
        )
        reconstructed[event_id] = await linker.link(detection)

    return reconstructed


def run(
    split: str = "test",
    dataset_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the campaign-reconstruction experiment and write ``campaign_metrics.json``."""
    set_global_seed()
    target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    models = load_models(target)
    risk = models[4]

    scored = score_split(split, *models, dataset_dir=dataset_dir, limit=limit)
    events = load_split(split, dataset_dir)
    type_by_entity = {event.entity_id: event.entity_type for event in events}

    reconstructed = asyncio.run(_link_flagged(scored, type_by_entity, risk.alert_threshold))

    # Group flagged events by their ground-truth campaign, collecting the reconstructed id each got.
    truth_to_reconstructed: Dict[str, List[str]] = defaultdict(list)
    for event_id, gt_campaign in zip(scored.event_ids, scored.campaign_ids):
        if gt_campaign is None:
            continue
        recon = reconstructed.get(event_id)
        if recon is not None:
            truth_to_reconstructed[gt_campaign].append(recon)

    correct = 0
    total = 0
    for recon_ids in truth_to_reconstructed.values():
        if not recon_ids:
            continue
        majority = Counter(recon_ids).most_common(1)[0][1]
        correct += majority
        total += len(recon_ids)

    stages_linked = float(correct / total) if total else float("nan")
    campaigns_expected = len({c for c in scored.campaign_ids if c is not None})
    campaigns_reconstructed = len({c for c in reconstructed.values() if c is not None})

    result = CampaignMetrics(
        stages_linked_correctly=stages_linked,
        campaigns_reconstructed=campaigns_reconstructed,
        campaigns_expected=campaigns_expected,
    )
    payload = {
        **result.model_dump(mode="json"),
        "flagged_campaign_events": total,
        "split": split,
    }
    path = target / CAMPAIGN_FILE
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return {"result": result, "payload": payload, "path": str(path)}


def format_report(summary: Dict[str, Any]) -> str:
    p = summary["payload"]

    def fmt(value: Any) -> str:
        try:
            return format(float(value), ".4f")
        except (TypeError, ValueError):
            return str(value)

    return "\n".join(
        [
            "",
            "=" * 74,
            " Campaign reconstruction (D1)",
            "=" * 74,
            f" ground-truth campaigns (split) : {p['campaigns_expected']}",
            f" reconstructed campaigns         : {p['campaigns_reconstructed']}",
            f" flagged campaign events         : {p['flagged_campaign_events']}",
            f" stages linked correctly         : {fmt(p['stages_linked_correctly'])}   target >= 0.90",
            "=" * 74,
            "",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.campaign_experiment")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        summary = run(split=args.split, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(format_report(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
