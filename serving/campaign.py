"""Attack-campaign / kill-chain reconstruction (D1).

Real intrusions are not single events -- they are a sequence: a brute force, then a successful
login, then lateral movement, then exfiltration. Surfacing each alert in isolation buries that
story. This linker stitches an entity's related anomalies into one campaign so an analyst sees the
whole kill chain at once.

The linking rule is deliberately simple and explainable: a new anomaly joins the entity's most
recent still-open campaign if it falls within a time window, otherwise it opens a new one. Grouping
by entity and time-proximity is enough to reconstruct the injected multi-stage attacks, because
their stages share an entity and are close in time by construction. The campaign accumulates its
ordered ``kill_chain`` (collapsing repeats), its member detections and its peak risk.
"""

from __future__ import annotations

import logging
from typing import Optional

from common.models import Campaign, CampaignStage, CampaignStatus, Detection, utc_now
from serving.store import DetectionStore

logger = logging.getLogger(__name__)

#: How long a campaign stays open for new stages after its last activity. Wide enough to span a
#: multi-stage attack (which unfolds over hours to a couple of days), narrow enough that unrelated
#: activity weeks later starts a fresh campaign.
DEFAULT_WINDOW_SECONDS = 72 * 3600.0


class CampaignLinker:
    """Links anomalous detections into per-entity, time-windowed campaigns."""

    def __init__(self, store: DetectionStore, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> None:
        self.store = store
        self.window_seconds = window_seconds

    async def link(self, detection: Detection) -> Optional[str]:
        """Attach a detection to a campaign, opening one if needed. Returns the campaign id.

        Only anomalies are linked; a benign detection returns ``None`` and joins no campaign.
        """
        if not detection.is_anomaly:
            return None

        campaign = await self.store.open_campaign_for(
            detection.entity_id, detection.timestamp, self.window_seconds
        )
        if campaign is None:
            campaign = Campaign(
                entity_id=detection.entity_id,
                entity_type=detection.entity_type,
                started_at=detection.timestamp,
                last_activity=detection.timestamp,
            )

        campaign.stages.append(
            CampaignStage(
                anomaly_type=detection.anomaly_type,
                detection_id=detection.detection_id,
                timestamp=detection.timestamp,
                risk_score=detection.risk_score,
            )
        )
        campaign.detection_ids.append(detection.detection_id)

        # Kill chain: ordered technique/stage names, collapsing consecutive repeats so
        # "brute_force x5 -> login" reads as one brute-force step.
        stage_name = detection.anomaly_type.value
        if not campaign.kill_chain or campaign.kill_chain[-1] != stage_name:
            campaign.kill_chain.append(stage_name)

        campaign.max_risk = max(campaign.max_risk, detection.risk_score)
        campaign.last_activity = detection.timestamp
        campaign.updated_at = utc_now()
        campaign.status = CampaignStatus.OPEN

        await self.store.upsert_campaign(campaign)
        return campaign.campaign_id


__all__ = ["CampaignLinker", "DEFAULT_WINDOW_SECONDS"]
