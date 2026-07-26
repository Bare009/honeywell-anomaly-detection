"""Persistence for the serving and read planes.

The scoring pipeline, campaign linker and feedback loop all read and write through a small
``DetectionStore`` interface rather than talking to a database directly. That indirection buys two
things: the scoring logic is tested end-to-end with an in-memory store and no database running
(tests stay fast and CPU-only), and the same logic runs in production against MongoDB by swapping
the implementation. Both stores expose identical async methods, so nothing above them changes.

Documents are stored as ``model_dump(mode="json")`` so datetimes and enums serialize consistently,
which is exactly what the Mongo driver and the read API expect.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from common.database import Collections, get_collection
from common.models import Campaign, Detection, Feedback

logger = logging.getLogger(__name__)


class DetectionStore:
    """Async persistence interface used by the serving plane.

    Concrete stores implement every method. The base raises, so a half-implemented store fails
    loudly rather than silently dropping writes.
    """

    async def save_detection(self, detection: Detection) -> None:
        raise NotImplementedError

    async def get_detection(self, detection_id: str) -> Optional[Detection]:
        raise NotImplementedError

    async def list_detections(
        self,
        skip: int = 0,
        limit: int = 50,
        anomaly_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        cold_start: Optional[bool] = None,
        min_risk: Optional[float] = None,
        sort: str = "risk",
    ) -> List[Detection]:
        raise NotImplementedError

    async def count_detections(self) -> int:
        raise NotImplementedError

    async def entity_detections(self, entity_id: str, skip: int = 0, limit: int = 100) -> List[Detection]:
        raise NotImplementedError

    async def count_entity_detections(self, entity_id: str) -> int:
        raise NotImplementedError

    async def upsert_campaign(self, campaign: Campaign) -> None:
        raise NotImplementedError

    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        raise NotImplementedError

    async def open_campaign_for(
        self, entity_id: str, as_of: datetime, window_seconds: float
    ) -> Optional[Campaign]:
        raise NotImplementedError

    async def list_campaigns(
        self, entity_id: Optional[str] = None, skip: int = 0, limit: int = 50
    ) -> List[Campaign]:
        raise NotImplementedError

    async def save_feedback(self, feedback: Feedback) -> None:
        raise NotImplementedError

    async def list_feedback(self, limit: int = 100) -> List[Feedback]:
        raise NotImplementedError

    async def get_entity_offset(self, entity_id: str) -> float:
        raise NotImplementedError

    async def set_entity_offset(self, entity_id: str, offset: float) -> None:
        raise NotImplementedError

    async def save_drift_state(self, entity_id: str, state: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def get_drift_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def dashboard_summary(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def clear(self) -> None:
        """Remove all detections, campaigns, feedback and drift state (fresh repopulation)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# In-memory store (tests, replay, demos without a database)
# --------------------------------------------------------------------------- #


class InMemoryStore(DetectionStore):
    """A dict-backed store. Everything the pipeline needs, nothing persisted beyond the process."""

    def __init__(self) -> None:
        self._detections: Dict[str, Detection] = {}
        self._campaigns: Dict[str, Campaign] = {}
        self._feedback: List[Feedback] = []
        self._offsets: Dict[str, float] = {}
        self._drift: Dict[str, Dict[str, Any]] = {}

    async def save_detection(self, detection: Detection) -> None:
        self._detections[detection.detection_id] = detection

    async def get_detection(self, detection_id: str) -> Optional[Detection]:
        return self._detections.get(detection_id)

    async def list_detections(
        self,
        skip: int = 0,
        limit: int = 50,
        anomaly_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        cold_start: Optional[bool] = None,
        min_risk: Optional[float] = None,
        sort: str = "risk",
    ) -> List[Detection]:
        items = list(self._detections.values())
        if anomaly_type is not None:
            items = [d for d in items if d.anomaly_type.value == anomaly_type]
        if entity_type is not None:
            items = [d for d in items if d.entity_type.value == entity_type]
        if cold_start is not None:
            items = [d for d in items if bool(d.cold_start) == cold_start]
        if min_risk is not None:
            items = [d for d in items if d.risk_score >= min_risk]
        if sort == "risk":
            items.sort(key=lambda d: d.risk_score, reverse=True)
        else:
            items.sort(key=lambda d: d.timestamp, reverse=True)
        return items[skip : skip + limit]

    async def count_detections(self) -> int:
        return len(self._detections)

    async def entity_detections(self, entity_id: str, skip: int = 0, limit: int = 100) -> List[Detection]:
        items = [d for d in self._detections.values() if d.entity_id == entity_id]
        # Order by risk so an entity's actual alerts surface. Ordering by time would bury rare
        # anomalies under the entity's much larger tail of recent normal events and truncate them.
        items.sort(key=lambda d: (d.risk_score, d.timestamp), reverse=True)
        return items[skip : skip + limit]

    async def count_entity_detections(self, entity_id: str) -> int:
        return sum(1 for d in self._detections.values() if d.entity_id == entity_id)

    async def upsert_campaign(self, campaign: Campaign) -> None:
        self._campaigns[campaign.campaign_id] = campaign

    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        return self._campaigns.get(campaign_id)

    async def open_campaign_for(
        self, entity_id: str, as_of: datetime, window_seconds: float
    ) -> Optional[Campaign]:
        candidates = [
            c
            for c in self._campaigns.values()
            if c.entity_id == entity_id
            and c.status.value == "open"
            and 0 <= (as_of - c.last_activity).total_seconds() <= window_seconds
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.last_activity)

    async def list_campaigns(
        self, entity_id: Optional[str] = None, skip: int = 0, limit: int = 50
    ) -> List[Campaign]:
        items = list(self._campaigns.values())
        if entity_id is not None:
            items = [c for c in items if c.entity_id == entity_id]
        items.sort(key=lambda c: c.last_activity, reverse=True)
        return items[skip : skip + limit]

    async def save_feedback(self, feedback: Feedback) -> None:
        self._feedback.append(feedback)

    async def list_feedback(self, limit: int = 100) -> List[Feedback]:
        return list(reversed(self._feedback))[:limit]

    async def get_entity_offset(self, entity_id: str) -> float:
        return self._offsets.get(entity_id, 0.0)

    async def set_entity_offset(self, entity_id: str, offset: float) -> None:
        self._offsets[entity_id] = offset

    async def save_drift_state(self, entity_id: str, state: Dict[str, Any]) -> None:
        self._drift[entity_id] = state

    async def get_drift_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self._drift.get(entity_id)

    async def clear(self) -> None:
        self._detections.clear()
        self._campaigns.clear()
        self._feedback.clear()
        self._offsets.clear()
        self._drift.clear()

    async def dashboard_summary(self) -> Dict[str, Any]:
        detections = list(self._detections.values())
        anomalies = [d for d in detections if d.is_anomaly]
        by_type: Dict[str, int] = {}
        for det in anomalies:
            by_type[det.anomaly_type.value] = by_type.get(det.anomaly_type.value, 0) + 1
        return {
            "n_detections": len(detections),
            "n_anomalies": len(anomalies),
            "n_campaigns": len(self._campaigns),
            "n_feedback": len(self._feedback),
            "by_type": by_type,
        }


# --------------------------------------------------------------------------- #
# MongoDB store (production)
# --------------------------------------------------------------------------- #


class MongoStore(DetectionStore):
    """Motor-backed store. Documents are JSON-mode dumps of the Pydantic models."""

    def __init__(self) -> None:
        self.detections = get_collection(Collections.DETECTIONS)
        self.campaigns = get_collection(Collections.CAMPAIGNS)
        self.feedback = get_collection(Collections.FEEDBACK)
        self.profiles = get_collection(Collections.ENTITY_PROFILES)
        self.drift = get_collection(Collections.DRIFT_STATE)

    @staticmethod
    def _clean(document: Dict[str, Any]) -> Dict[str, Any]:
        document.pop("_id", None)
        return document

    async def save_detection(self, detection: Detection) -> None:
        doc = detection.model_dump(mode="json")
        await self.detections.replace_one({"detection_id": detection.detection_id}, doc, upsert=True)

    async def get_detection(self, detection_id: str) -> Optional[Detection]:
        doc = await self.detections.find_one({"detection_id": detection_id})
        return Detection.model_validate(self._clean(doc)) if doc else None

    async def list_detections(
        self,
        skip: int = 0,
        limit: int = 50,
        anomaly_type: Optional[str] = None,
        entity_type: Optional[str] = None,
        cold_start: Optional[bool] = None,
        min_risk: Optional[float] = None,
        sort: str = "risk",
    ) -> List[Detection]:
        query: Dict[str, Any] = {}
        if anomaly_type is not None:
            query["anomaly_type"] = anomaly_type
        if entity_type is not None:
            query["entity_type"] = entity_type
        if cold_start is not None:
            query["cold_start"] = cold_start
        if min_risk is not None:
            query["risk_score"] = {"$gte": min_risk}
        sort_key = "risk_score" if sort == "risk" else "timestamp"
        cursor = self.detections.find(query).sort(sort_key, -1).skip(skip).limit(limit)
        return [Detection.model_validate(self._clean(doc)) async for doc in cursor]

    async def count_detections(self) -> int:
        return int(await self.detections.count_documents({}))

    async def entity_detections(self, entity_id: str, skip: int = 0, limit: int = 100) -> List[Detection]:
        # Order by risk so an entity's actual alerts surface. Ordering by time would bury rare
        # anomalies under the entity's much larger tail of recent normal events and truncate them.
        cursor = (
            self.detections.find({"entity_id": entity_id})
            .sort("risk_score", -1)
            .skip(skip)
            .limit(limit)
        )
        return [Detection.model_validate(self._clean(doc)) async for doc in cursor]

    async def count_entity_detections(self, entity_id: str) -> int:
        return int(await self.detections.count_documents({"entity_id": entity_id}))

    async def upsert_campaign(self, campaign: Campaign) -> None:
        doc = campaign.model_dump(mode="json")
        await self.campaigns.replace_one({"campaign_id": campaign.campaign_id}, doc, upsert=True)

    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        doc = await self.campaigns.find_one({"campaign_id": campaign_id})
        return Campaign.model_validate(self._clean(doc)) if doc else None

    async def open_campaign_for(
        self, entity_id: str, as_of: datetime, window_seconds: float
    ) -> Optional[Campaign]:
        floor = as_of - timedelta(seconds=window_seconds)
        doc = await self.campaigns.find_one(
            {
                "entity_id": entity_id,
                "status": "open",
                "last_activity": {"$gte": floor.isoformat(), "$lte": as_of.isoformat()},
            },
            sort=[("last_activity", -1)],
        )
        return Campaign.model_validate(self._clean(doc)) if doc else None

    async def list_campaigns(
        self, entity_id: Optional[str] = None, skip: int = 0, limit: int = 50
    ) -> List[Campaign]:
        query = {"entity_id": entity_id} if entity_id else {}
        cursor = self.campaigns.find(query).sort("last_activity", -1).skip(skip).limit(limit)
        return [Campaign.model_validate(self._clean(doc)) async for doc in cursor]

    async def save_feedback(self, feedback: Feedback) -> None:
        await self.feedback.insert_one(feedback.model_dump(mode="json"))

    async def list_feedback(self, limit: int = 100) -> List[Feedback]:
        cursor = self.feedback.find({}).sort("created_at", -1).limit(limit)
        return [Feedback.model_validate(self._clean(doc)) async for doc in cursor]

    async def get_entity_offset(self, entity_id: str) -> float:
        doc = await self.profiles.find_one({"entity_id": entity_id}, {"feedback_threshold_adjust": 1})
        return float(doc.get("feedback_threshold_adjust", 0.0)) if doc else 0.0

    async def set_entity_offset(self, entity_id: str, offset: float) -> None:
        await self.profiles.update_one(
            {"entity_id": entity_id},
            {"$set": {"feedback_threshold_adjust": offset}},
            upsert=True,
        )

    async def save_drift_state(self, entity_id: str, state: Dict[str, Any]) -> None:
        await self.drift.replace_one({"entity_id": entity_id}, {"entity_id": entity_id, **state}, upsert=True)

    async def get_drift_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        doc = await self.drift.find_one({"entity_id": entity_id})
        return self._clean(doc) if doc else None

    async def dashboard_summary(self) -> Dict[str, Any]:
        n_detections = await self.detections.count_documents({})
        n_anomalies = await self.detections.count_documents({"is_anomaly": True})
        n_campaigns = await self.campaigns.count_documents({})
        n_feedback = await self.feedback.count_documents({})
        by_type: Dict[str, int] = {}
        pipeline = [
            {"$match": {"is_anomaly": True}},
            {"$group": {"_id": "$anomaly_type", "count": {"$sum": 1}}},
        ]
        async for row in self.detections.aggregate(pipeline):
            by_type[row["_id"]] = int(row["count"])
        return {
            "n_detections": int(n_detections),
            "n_anomalies": int(n_anomalies),
            "n_campaigns": int(n_campaigns),
            "n_feedback": int(n_feedback),
            "by_type": by_type,
        }

    async def clear(self) -> None:
        for collection in (self.detections, self.campaigns, self.feedback, self.drift):
            await collection.delete_many({})


__all__ = ["DetectionStore", "InMemoryStore", "MongoStore"]
