"""Plain-language narrative for a detection.

A one- or two-sentence summary an analyst reads first, before the charts. It is generated from a
**deterministic template** by default -- so the demo never depends on the network and the same
detection always reads the same way. An optional Groq/Llama call can phrase it more naturally when
enabled, but it is strictly cosmetic: any failure (no key, timeout, bad response) silently falls
back to the template, and the narrative **never affects the score or the verdict**. The model owns
correctness; the language model, if used at all, only rewords an explanation the system already
computed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from common.config import settings
from common.models import AnomalyType, FeatureAttribution, MitreTechnique

_RISK_BANDS = [(80.0, "critical"), (60.0, "high"), (40.0, "moderate"), (0.0, "low")]


def _band(risk_score: float) -> str:
    for floor, label in _RISK_BANDS:
        if risk_score >= floor:
            return label
    return "low"


def _readable_type(anomaly_type: AnomalyType) -> str:
    return anomaly_type.value.replace("_", " ")


def template_narrative(
    entity_id: str,
    anomaly_type: AnomalyType,
    risk_score: float,
    top_features: Sequence[FeatureAttribution],
    mitre: Sequence[MitreTechnique] = (),
    cold_start: bool = False,
    detector_hits: Sequence[str] = (),
) -> str:
    """Deterministic, information-dense summary of a detection."""
    band = _band(risk_score)
    if anomaly_type == AnomalyType.NORMAL:
        lead = (
            f"Entity {entity_id} looks normal (risk {risk_score:.0f}/100, {band})."
        )
    else:
        lead = (
            f"Entity {entity_id} shows behaviour consistent with "
            f"{_readable_type(anomaly_type)} (risk {risk_score:.0f}/100, {band})."
        )

    drivers = [f.feature for f in top_features if f.direction == "increases_risk"][:3]
    driver_text = f" Main drivers: {', '.join(drivers)}." if drivers else ""

    detail = ""
    if detector_hits:
        detail += f" Deterministic checks fired: {', '.join(detector_hits)}."
    if mitre:
        techniques = ", ".join(f"{t.technique_id} ({t.name})" for t in mitre[:2])
        detail += f" Maps to MITRE ATT&CK {techniques}."
    if cold_start:
        detail += " This entity has little history, so the score is less certain."

    return (lead + driver_text + detail).strip()


class NarrativeGenerator:
    """Produces a narrative, optionally via an LLM, always with a deterministic fallback."""

    def __init__(self, use_llm: Optional[bool] = None) -> None:
        self.use_llm = settings.llm_enabled if use_llm is None else use_llm

    def generate(
        self,
        entity_id: str,
        anomaly_type: AnomalyType,
        risk_score: float,
        top_features: Sequence[FeatureAttribution],
        mitre: Sequence[MitreTechnique] = (),
        cold_start: bool = False,
        detector_hits: Sequence[str] = (),
    ) -> Tuple[str, str]:
        """Return ``(text, source)`` where source is ``"llm"`` or ``"template"``."""
        template = template_narrative(
            entity_id, anomaly_type, risk_score, top_features, mitre, cold_start, detector_hits
        )
        if not self.use_llm or not settings.groq_api_key:
            return template, "template"
        try:  # pragma: no cover - network path, disabled by default
            text = self._llm(template)
            return (text, "llm") if text else (template, "template")
        except Exception:  # noqa: BLE001 - never let the narrator break scoring
            return template, "template"

    def _llm(self, grounding: str) -> Optional[str]:  # pragma: no cover - optional network path
        """Ask Groq to reword the grounded template. Returns None on any problem."""
        import httpx

        prompt = (
            "Rewrite the following security alert summary in two concise sentences for a SOC "
            "analyst. Do not add facts beyond what is given.\n\n" + grounding
        )
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip() or None


__all__ = ["template_narrative", "NarrativeGenerator"]
