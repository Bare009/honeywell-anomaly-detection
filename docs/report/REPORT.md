# Behavioral Anomaly Detection — Evaluation Report

All numbers below are computed on the held-out **test** split (never seen by the models or the fusion tuning). Regenerate with `python -m evaluation.evaluate` and the experiment modules.

- Split: `test` · 38,875 events · 260 entities · anomaly rate 0.9672%
- Artifact schema: `1.0.0` · git `20b421e` · seed 42

## Headline metrics

| Metric | Result | Target | Verdict |
| --- | --- | --- | --- |
| PR-AUC | 0.9380 | ≥ 0.90 | PASS |
| Recall @ 1% budget | 0.8963 | ≥ 0.80 | PASS |
| Macro-F1 (9 classes) | 0.8607 | ≥ 0.85 | PASS |
| Calibration ECE | 0.0009 | ≤ 0.05 | PASS |
| ROC-AUC | 0.9956 | (context only) | — |

> macro_f1 over classes present in test: 0.8607

## Per-class classification (type)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| normal | 1.000 | 0.996 | 0.998 | 38499 |
| credential_misuse | 1.000 | 0.974 | 0.987 | 39 |
| lateral_movement | 1.000 | 1.000 | 1.000 | 63 |
| brute_force | 1.000 | 1.000 | 1.000 | 45 |
| impossible_travel | 0.607 | 0.944 | 0.739 | 18 |
| credential_stuffing | 1.000 | 0.889 | 0.941 | 36 |
| device_spoofing | 0.196 | 0.889 | 0.322 | 36 |
| low_and_slow_exfil | 1.000 | 1.000 | 1.000 | 96 |
| insider_drift | 0.833 | 0.698 | 0.759 | 43 |

## Deterministic detectors

| Detector | Anomaly precision | Type precision | Fired |
| --- | --- | --- | --- |
| impossible_travel | 0.789 | 0.263 | 19 |
| brute_force | 1.000 | 1.000 | 33 |

## Cold-start ablation (D3 / cold-start target)

- Cold-start entities: 32 · cold-start anomalies: 21
- Recall **with** cohort priors: 0.8571 (target ≥ 0.70, PASS)
- Recall **without** cohort priors: 0.8571
- Uplift from cohort priors: **0.0000**

## Campaign reconstruction (D1)

- Stages linked correctly: **1.0000** (target ≥ 0.90, PASS)
- Reconstructed campaigns: 55 · ground-truth campaigns in split: 4

## Drift adaptation (D3)

- Benign drift DRIFTING rate — no adaptation: 0.5283, with adaptation: 0.1350
- Adaptation events: 39 · mean PSI: 0.1055
- Abrupt shift max PSI: 12.8604 (threshold 0.2000, flagged: True)

## Honesty: limitations and failure modes

- **ROC-AUC is not the headline.** At ~1% prevalence it looks impressive while saying little; PR-AUC and recall within the alert budget are the metrics that reflect analyst reality.
- **Per-class metrics rest on few independent incidents** (e.g. 3 for `credential_stuffing`, 4 for `low_and_slow_exfil`). Event counts look comfortable, but a regeneration can move a class by ±0.2. Only the aggregate metrics are stable.
- **The 9-class macro-F1 is sensitive to split composition.** A class thin or absent in a split is forced toward F1 0 regardless of model quality; the per-class table shows where the support is thin.
- **Calibration ECE on validation is optimistic** (the isotonic map is fitted there); the number above is on the test split, which the calibration never saw, and is the honest one.
- **The data is synthetic.** It exercises the pipeline end-to-end and encodes realistic attack structure, but real telemetry is messier; these numbers are an upper bound on how clean the signal is, not a production guarantee.
- **The autoencoder tier carries no fusion weight for ranking** — the classifier and sequence tiers dominate detection. The autoencoder earns its place through per-feature reconstruction *explanations*, not through the score.

## Deliverable → artifact → test mapping

| Deliverable | Implementation | Tests |
| --- | --- | --- |
| #1 Baseline profiling model | `models/baseline.py (autoencoder)` | `tests/test_baseline.py` |
| #2 Sequence-aware model | `models/sequence.py (GRU)` | `tests/test_sequence.py` |
| #3 Anomaly-type classifier | `models/classifier.py (LightGBM)` | `tests/test_classifier.py` |
| D1 Attack-story reconstruction | `serving/campaign.py, evaluation/campaign_experiment.py` | `tests/test_campaign.py` |
| D2 Explainability + counterfactual | `explainability/*` | `tests/test_explainability.py, tests/test_counterfactual.py` |
| D3 Adaptability / drift | `models/drift.py, evaluation/drift_experiment.py` | `tests/test_drift.py` |
| D4 Precision + alert budget | `models/risk.py, evaluation/evaluate.py` | `tests/test_risk.py` |
| D5 Risk score + uncertainty | `models/risk.py` | `tests/test_risk.py` |
| D6 Analyst feedback loop | `serving/feedback.py` | `tests/test_feedback.py` |
| D7 Evaluation + honesty | `evaluation/*, REPORT.md` | `tests/test_evaluation.py` |

