"""Assemble REPORT.md from the evaluation artifacts.

Reads whatever the evaluation and experiment steps wrote (``metrics.json``, ``coldstart_metrics.json``,
``campaign_metrics.json``, ``drift_metrics.json``) and renders one honest report: headline metrics
against their targets, per-class breakdown, the deterministic-detector precision, the three
experiments, a limitations section that states the synthetic-data caveats plainly, and a
deliverable -> artifact -> test mapping so a reader can trace every claim to the code that backs it.

Run it (after the evaluation steps) with::

    python -m evaluation.report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config import settings
from common.models import ANOMALY_CLASSES

logger = logging.getLogger(__name__)

REPORT_FILE = "REPORT.md"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any, spec: str = ".4f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "—" if value is None else str(value)


def _verdict(value: Any, target: float, good_high: bool = True) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    ok = v >= target if good_high else v <= target
    return "PASS" if ok else "MISS"


DELIVERABLE_MAP = [
    ("#1 Baseline profiling model", "models/baseline.py (autoencoder)", "tests/test_baseline.py"),
    ("#2 Sequence-aware model", "models/sequence.py (GRU)", "tests/test_sequence.py"),
    ("#3 Anomaly-type classifier", "models/classifier.py (LightGBM)", "tests/test_classifier.py"),
    ("D1 Attack-story reconstruction", "serving/campaign.py, evaluation/campaign_experiment.py", "tests/test_campaign.py"),
    ("D2 Explainability + counterfactual", "explainability/*", "tests/test_explainability.py, tests/test_counterfactual.py"),
    ("D3 Adaptability / drift", "models/drift.py, evaluation/drift_experiment.py", "tests/test_drift.py"),
    ("D4 Precision + alert budget", "models/risk.py, evaluation/evaluate.py", "tests/test_risk.py"),
    ("D5 Risk score + uncertainty", "models/risk.py", "tests/test_risk.py"),
    ("D6 Analyst feedback loop", "serving/feedback.py", "tests/test_feedback.py"),
    ("D7 Evaluation + honesty", "evaluation/*, REPORT.md", "tests/test_evaluation.py"),
]


def assemble(artifacts_dir: Optional[Path] = None, output: Optional[Path] = None) -> Path:
    """Render REPORT.md from the evaluation artifacts and return its path."""
    target = Path(artifacts_dir) if artifacts_dir else Path(settings.artifacts_dir)
    metrics = _load(target / "metrics.json")
    coldstart = _load(target / "coldstart_metrics.json")
    campaign = _load(target / "campaign_metrics.json")
    drift = _load(target / "drift_metrics.json")

    lines: List[str] = []
    add = lines.append

    add("# Behavioral Anomaly Detection — Evaluation Report")
    add("")
    add("All numbers below are computed on the held-out **test** split (never seen by the models or "
        "the fusion tuning). Regenerate with `python -m evaluation.evaluate` and the experiment modules.")
    add("")

    if metrics:
        ds = metrics.get("dataset_summary", {})
        add(f"- Split: `{ds.get('split')}` · {ds.get('n_events', 0):,} events · "
            f"{ds.get('n_entities', 0):,} entities · anomaly rate {_fmt(ds.get('anomaly_rate'), '.4%')}")
        add(f"- Artifact schema: `{metrics.get('artifact_schema_version')}` · git `{metrics.get('git_sha')}` · seed {metrics.get('seed')}")
        add("")
        add("## Headline metrics")
        add("")
        add("| Metric | Result | Target | Verdict |")
        add("| --- | --- | --- | --- |")
        add(f"| PR-AUC | {_fmt(metrics.get('pr_auc'))} | ≥ 0.90 | {_verdict(metrics.get('pr_auc'), 0.90)} |")
        add(f"| Recall @ 1% budget | {_fmt(metrics.get('recall_at_1pct_budget'))} | ≥ 0.80 | {_verdict(metrics.get('recall_at_1pct_budget'), 0.80)} |")
        add(f"| Macro-F1 (9 classes) | {_fmt(metrics.get('macro_f1'))} | ≥ 0.85 | {_verdict(metrics.get('macro_f1'), 0.85)} |")
        add(f"| Calibration ECE | {_fmt(metrics.get('calibration_ece'))} | ≤ 0.05 | {_verdict(metrics.get('calibration_ece'), 0.05, good_high=False)} |")
        add(f"| ROC-AUC | {_fmt(metrics.get('roc_auc'))} | (context only) | — |")
        add("")
        if metrics.get("notes"):
            add(f"> {metrics['notes']}")
            add("")

        add("## Per-class classification (type)")
        add("")
        add("| Class | Precision | Recall | F1 | Support |")
        add("| --- | --- | --- | --- | --- |")
        per_class = metrics.get("per_class", {})
        for cls in ANOMALY_CLASSES:
            row = per_class.get(cls, {})
            add(f"| {cls} | {_fmt(row.get('precision'), '.3f')} | {_fmt(row.get('recall'), '.3f')} | "
                f"{_fmt(row.get('f1'), '.3f')} | {int(row.get('support', 0))} |")
        add("")

        add("## Deterministic detectors")
        add("")
        add("| Detector | Anomaly precision | Type precision | Fired |")
        add("| --- | --- | --- | --- |")
        det = metrics.get("detector_precision", {})
        for name in ("impossible_travel", "brute_force"):
            row = det.get(name, {})
            add(f"| {name} | {_fmt(row.get('anomaly_precision'), '.3f')} | "
                f"{_fmt(row.get('type_precision'), '.3f')} | {int(row.get('n_fired', 0))} |")
        add("")
    else:
        add("_No `metrics.json` found. Run `python -m evaluation.evaluate` first._")
        add("")

    add("## Cold-start ablation (D3 / cold-start target)")
    add("")
    if coldstart:
        add(f"- Cold-start entities: {coldstart.get('n_cold_entities', 0):,} · "
            f"cold-start anomalies: {coldstart.get('n_cold_anomalies', 0):,}")
        add(f"- Recall **with** cohort priors: {_fmt(coldstart.get('recall_with_priors'))} "
            f"(target ≥ 0.70, {_verdict(coldstart.get('recall_with_priors'), 0.70)})")
        add(f"- Recall **without** cohort priors: {_fmt(coldstart.get('recall_without_priors'))}")
        add(f"- Uplift from cohort priors: **{_fmt(coldstart.get('uplift'))}**")
    else:
        add("_Run `python -m evaluation.coldstart_experiment`._")
    add("")

    add("## Campaign reconstruction (D1)")
    add("")
    if campaign:
        add(f"- Stages linked correctly: **{_fmt(campaign.get('stages_linked_correctly'))}** "
            f"(target ≥ 0.90, {_verdict(campaign.get('stages_linked_correctly'), 0.90)})")
        add(f"- Reconstructed campaigns: {campaign.get('campaigns_reconstructed', 0)} · "
            f"ground-truth campaigns in split: {campaign.get('campaigns_expected', 0)}")
    else:
        add("_Run `python -m evaluation.campaign_experiment`._")
    add("")

    add("## Drift adaptation (D3)")
    add("")
    if drift:
        add(f"- Benign drift DRIFTING rate — no adaptation: {_fmt(drift.get('fp_rate_before'))}, "
            f"with adaptation: {_fmt(drift.get('fp_rate_after'))}")
        add(f"- Adaptation events: {drift.get('adaptation_events', 0)} · mean PSI: {_fmt(drift.get('mean_psi'))}")
        add(f"- Abrupt shift max PSI: {_fmt(drift.get('abrupt_max_psi'))} "
            f"(threshold {_fmt(drift.get('series', {}).get('threshold'))}, flagged: {drift.get('abrupt_flagged')})")
    else:
        add("_Run `python -m evaluation.drift_experiment`._")
    add("")

    add("## Honesty: limitations and failure modes")
    add("")
    for note in [
        "**ROC-AUC is not the headline.** At ~1% prevalence it looks impressive while saying little; "
        "PR-AUC and recall within the alert budget are the metrics that reflect analyst reality.",
        "**Per-class metrics rest on few independent incidents** (e.g. 3 for `credential_stuffing`, "
        "4 for `low_and_slow_exfil`). Event counts look comfortable, but a regeneration can move a "
        "class by ±0.2. Only the aggregate metrics are stable.",
        "**The 9-class macro-F1 is sensitive to split composition.** A class thin or absent in a split "
        "is forced toward F1 0 regardless of model quality; the per-class table shows where the "
        "support is thin.",
        "**Calibration ECE on validation is optimistic** (the isotonic map is fitted there); the number "
        "above is on the test split, which the calibration never saw, and is the honest one.",
        "**The data is synthetic.** It exercises the pipeline end-to-end and encodes realistic attack "
        "structure, but real telemetry is messier; these numbers are an upper bound on how clean the "
        "signal is, not a production guarantee.",
        "**The autoencoder tier carries no fusion weight for ranking** — the classifier and sequence "
        "tiers dominate detection. The autoencoder earns its place through per-feature reconstruction "
        "*explanations*, not through the score.",
    ]:
        add(f"- {note}")
    add("")

    add("## Deliverable → artifact → test mapping")
    add("")
    add("| Deliverable | Implementation | Tests |")
    add("| --- | --- | --- |")
    for name, impl, tests in DELIVERABLE_MAP:
        add(f"| {name} | `{impl}` | `{tests}` |")
    add("")

    destination = Path(output) if output else Path.cwd() / REPORT_FILE
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", destination)
    return destination


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.report")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    path = assemble(artifacts_dir=args.artifacts_dir, output=args.output)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
