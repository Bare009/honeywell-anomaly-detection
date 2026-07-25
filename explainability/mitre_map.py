"""Map an anomaly type to MITRE ATT&CK technique(s).

The mapping is a small, static table -- deliberately. Tying each anomaly class to a recognized
ATT&CK technique lets an analyst pivot straight into the framework they already use, and it costs
nothing at runtime and cannot fail. A Qdrant-backed semantic lookup is supported by configuration
(`mitre_map_source = "qdrant"`) but is entirely optional: if it is not configured, or the store is
unreachable, the static table is used, so the explanation never depends on the network.

This is a supporting cross-reference, not a detection signal -- it annotates the predicted type, it
never influences the score.
"""

from __future__ import annotations

from typing import Dict, List

from common.config import settings
from common.models import AnomalyType, MitreTechnique

_URL = "https://attack.mitre.org/techniques/{tid}/"


def _technique(technique_id: str, name: str, tactic: str, confidence: float = 1.0) -> MitreTechnique:
    return MitreTechnique(
        technique_id=technique_id,
        name=name,
        tactic=tactic,
        url=_URL.format(tid=technique_id.replace(".", "/")),
        confidence=confidence,
    )


#: Static anomaly-type -> ATT&CK technique table. Multiple techniques are ordered most-relevant
#: first. ``normal`` maps to nothing.
STATIC_MAP: Dict[AnomalyType, List[MitreTechnique]] = {
    AnomalyType.NORMAL: [],
    AnomalyType.BRUTE_FORCE: [
        _technique("T1110", "Brute Force", "Credential Access"),
    ],
    AnomalyType.CREDENTIAL_STUFFING: [
        _technique("T1110.004", "Brute Force: Credential Stuffing", "Credential Access"),
    ],
    AnomalyType.CREDENTIAL_MISUSE: [
        _technique("T1078", "Valid Accounts", "Defense Evasion"),
    ],
    AnomalyType.IMPOSSIBLE_TRAVEL: [
        _technique("T1078", "Valid Accounts", "Initial Access"),
    ],
    AnomalyType.LATERAL_MOVEMENT: [
        _technique("T1021", "Remote Services", "Lateral Movement"),
        _technique("T1210", "Exploitation of Remote Services", "Lateral Movement"),
    ],
    AnomalyType.DEVICE_SPOOFING: [
        _technique("T1036", "Masquerading", "Defense Evasion"),
        _technique("T1200", "Hardware Additions", "Initial Access"),
    ],
    AnomalyType.LOW_AND_SLOW_EXFIL: [
        _technique("T1048", "Exfiltration Over Alternative Protocol", "Exfiltration"),
    ],
    AnomalyType.INSIDER_DRIFT: [
        _technique("T1078", "Valid Accounts", "Privilege Escalation"),
        _technique("T1098", "Account Manipulation", "Persistence"),
    ],
}


def map_static(anomaly_type: AnomalyType) -> List[MitreTechnique]:
    """Return the static technique list for a type (a fresh copy so callers can mutate safely)."""
    return [technique.model_copy() for technique in STATIC_MAP.get(anomaly_type, [])]


def map_anomaly(anomaly_type: AnomalyType) -> List[MitreTechnique]:
    """Map an anomaly type to techniques, honouring the configured source with a static fallback.

    Only ``static`` is implemented here; ``qdrant`` is an optional Phase-10 enhancement. Any source
    other than a working Qdrant lookup falls back to the static table, so this never raises and never
    needs the network.
    """
    if settings.mitre_map_source == "qdrant":
        try:  # pragma: no cover - optional dependency path, exercised only when configured
            return _map_qdrant(anomaly_type)
        except Exception:  # noqa: BLE001 - any failure degrades to the static table
            return map_static(anomaly_type)
    return map_static(anomaly_type)


def _map_qdrant(anomaly_type: AnomalyType) -> List[MitreTechnique]:  # pragma: no cover
    """Optional semantic lookup. Not implemented in the core build; falls back to static."""
    raise NotImplementedError("Qdrant MITRE lookup is an optional Phase 10 enhancement")


__all__ = ["STATIC_MAP", "map_static", "map_anomaly"]
