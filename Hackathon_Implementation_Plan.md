# Hackathon Implementation Plan — AI-Powered Behavioral Anomaly Detection

> **What this document is.** A complete, self-contained, phase-by-phase work order for building a
> machine-learning behavioral anomaly detection system from an **empty folder**. It is the single
> source of truth for the build — it does not depend on, reference, or reuse any prior codebase.
> Everything described here is created from scratch.
>
> **How to use it.** Start in a fresh, empty project directory containing only this file (and,
> optionally, the hackathon brief). Drive the build by saying **"do phase N"**, starting at
> **Phase 0**. Each phase is standalone: objective, exact files to create, step-by-step tasks,
> measurable acceptance criteria, and verification commands. A phase is not "done" until its
> Definition of Done passes (§3). Phases are sequential — do not skip forward.
>
> **Golden rule.** The hackathon deliverables and evaluation criteria win every tradeoff.

---

## Table of Contents
1. Executive Summary
2. Project Vision
3. Execution Protocol (read before "do phase 0")
4. Concrete Success Targets (the numbers we must hit)
5. Innovation & Differentiation Strategy (how we beat the field)
6. Build Scope (what we build, component by component)
7. Requirements → Features Mapping
8. Evaluation-Criteria Maximization
9. Final System Architecture
10. Technology Stack
11. Folder Structure (target)
12. Database Design
13. Data Flow
14. AI / Model Pipeline
15. Behavioral Analytics / Feature Pipeline
16. Backend Architecture
17. Frontend & Dashboard
18. Explainability Features
19. Security Considerations
20. APIs
21. Deployment Strategy
22. Testing Strategy
23. Demo Strategy
24. **Phase-wise Development Plan (the executable spec — Phases 0–10)**
25. Justification of Major Decisions
26. Assumptions & Conventions Recap

---

## 1. Executive Summary

Build an **AI/ML system that learns normal access and connection behavior, detects compromised
credentials or intrusions in near real-time, classifies the anomaly type, and produces an
explainable, calibrated risk score** — surfaced in a live analyst dashboard.

This is fundamentally an ML problem: sequence modeling, behavioral profiling, extreme class
imbalance, concept drift, cold start, and explainability. The system is built as three layered
detectors fused into one score:

1. **Baseline** (unsupervised statistical profile + Isolation Forest / One-Class SVM + tabular
   autoencoder) — catches cold-start and zero-day deviations.
2. **Sequence** (GRU/LSTM over command/resource sequences) — catches order-aware anomalies.
3. **Classifier** (calibrated LightGBM multi-class) — names the anomaly type.

Fused into a **calibrated 0–100 risk score with an uncertainty band**, mapped to **MITRE ATT&CK**,
reconstructed into **attack campaigns**, explained with **SHAP + counterfactual "nearest-normal"**
reasons, and improved by an **analyst feedback loop**.

**Pitch:** *"A layered behavioral anomaly engine — statistical + autoencoder baselines, a sequence
model over access patterns, and a supervised anomaly-type classifier — producing a calibrated,
explainable risk score per event, mapped to MITRE ATT&CK, reconstructed into attack campaigns, and
surfaced in a live analyst dashboard with feature attributions, counterfactual explanations,
cold-start handling, concept-drift adaptation, and an analyst feedback loop that measurably improves
the model."*

---

## 2. Project Vision

The first line of SOC triage: watch behavioral telemetry, learn "normal" per entity, and raise
**ranked, explained, classified** alerts — then stitch related alerts into **attack storylines** and
let analysts **teach the system**. Opening any alert answers, in seconds: how risky, what type,
**why** (the exact behaviors that deviated and what would have made it normal), what this entity
usually does, which MITRE technique, and what else is part of the same campaign.

Principles: **AI-first** (models are the product), **explainable by construction**, **honest about
imbalance** (PR-AUC and recall@budget, never raw accuracy), **demo-ready** (deterministic, seeded).

---

## 3. Execution Protocol (read before "do phase 0")

**Conventions enforced in every phase:**
- **Determinism:** a single global seed (`RANDOM_SEED = 42`, in `common/config.py`) is used by NumPy,
  Python `random`, PyTorch, and LightGBM (via a shared `common/seed.py`). Every phase is reproducible.
- **Artifacts contract:** all trained state (models, scalers, encoders, entity profiles, thresholds,
  fusion weights, calibration, SHAP background, `metrics.json`) lives under `artifacts/` and is
  **version-stamped** (`artifacts/manifest.json` with a schema/version + git SHA). Serving loads only
  from `artifacts/`; it never retrains.
- **Train/serve parity:** there is exactly one `featurize()` used both offline and online. No
  duplicated feature logic.
- **Graceful degradation:** any optional external dependency (LLM narrative, Qdrant, network) has a
  deterministic fallback. Nothing in the demo path depends on the internet.
- **Fail loud in dev, safe in serve:** training scripts assert and stop on bad data; the serving path
  validates input and returns 4xx, never 500, never crashes.
- **Testing is mandatory per phase (not optional):** every phase ends with fast, CPU-only tests that
  are **written and run before the phase is considered complete**.

**Definition of Done (applies to every phase — all five must hold):**
1. All "Files to Create" exist and are complete (no stubs, no `TODO`, no placeholders).
2. **Tests for the phase are written** (unit + integration as applicable) under `tests/`.
3. **All phase tests pass**, the phase's Verification commands run clean, and **previously passing
   tests still pass (no regressions)**.
4. The phase's Acceptance Criteria are met and demonstrated (including actual metric numbers where
   relevant).
5. A one-line status is recorded (what was built, test result, key numbers).

**Phase Completion Gate (hard stop):** a phase is NOT done — and you do NOT advance — until its tests
are written, run, and green, with no earlier-phase regressions. If tests fail, fix the code (or the
test if it is wrong) and re-run until green. This gate applies to Phases 0–10 individually.

**Working method when the user says "do phase N":**
1. Confirm prerequisites from earlier phases are in place.
2. Build the files in the listed order.
3. **Write the phase's tests** and run them (plus the phase's Verification commands).
4. Fix failures and re-run until everything is green and no earlier phase regressed.
5. Report results against the Acceptance Criteria (actual metric numbers + test summary).
6. Only then advance.

**Version-control workflow (per phase):** after a phase's Definition of Done passes, make one commit
for that phase and push. Keep `.venv/`, `__pycache__/`, `.pytest_cache/`, and trained artifact
binaries out of git (see the `.gitignore` in Phase 0); track `artifacts/.gitkeep` and
`artifacts/manifest.json`.

**Suggested order & pacing:** Phase 0 (setup) then 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Phase 10 is
optional/bonus. Phases 1, 5, 6, and 9 are the highest-leverage; protect time for Phase 1 data quality
and Phase 9 demo rehearsal.

**Environment note:** developed on Python 3.11+ (3.12 fine). PyTorch is the **CPU** build (models
train in minutes, score in milliseconds). On Windows/PowerShell, chain commands with `;` (not `&&`).

---

## 4. Concrete Success Targets (the numbers we must hit)

These make acceptance criteria measurable. They are ambitious-but-realistic on well-designed
synthetic data; if the generator is too easy we tighten it (Phase 1), if too hard we tune models.
Report *actual* numbers in the final report — never fabricate.

| Metric | Target | Where proven |
|---|---|---|
| Anomaly detection **PR-AUC** (held-out) | **≥ 0.90** | Phase 9 |
| Anomaly detection **ROC-AUC** | **≥ 0.95** | Phase 9 |
| **Recall @ 1% alert budget** (top 1% of events by risk) | **≥ 0.80** | Phase 5 & 9 |
| Anomaly-**type** classifier **macro-F1** | **≥ 0.85** | Phase 5 & 9 |
| Deterministic detectors (impossible travel, brute force) precision on injected cases | **≈ 1.0** | Phase 5 |
| **Cold-start recall** (entities with < min sessions) with peer priors | **≥ 0.70**, with clear uplift vs no-prior baseline | Phase 9 |
| Risk-score **calibration** (Expected Calibration Error) | **≤ 0.05** | Phase 5 & 9 |
| **Drift** experiment: false-positive rate after benign shift | returns to near-baseline within the adaptation window | Phase 9 |
| Per-event **scoring latency** (serving, CPU) | **< 50 ms** median | Phase 7 |
| **Attack-campaign** grouping: injected multi-stage attacks reconstructed | **≥ 90%** of stages linked to the correct campaign | Phase 7 & 9 |

---

## 5. Innovation & Differentiation Strategy (how we beat the field)

Meeting requirements is table stakes. These differentiators are engineered for maximum judge impact
per unit effort and are **built into the phases** (not bolted on).

**CORE differentiators (must build — cheap, high impact):**
- **D1 — Attack-campaign / kill-chain reconstruction ("Storyline view").** Link an entity's related
  detections across time into one narrative (`brute force → login → lateral movement → low-and-slow
  exfil`). *Built in Phase 1 (correlated attack generation) + Phase 7 (linking) + Phase 8 (timeline UI).*
- **D2 — Counterfactual "nearest-normal" explanations.** For each alert: *"Would have scored benign if
  location = India (not Brazil) and time = 10:00 (not 02:14)."* A perturbation search over top
  features. *Built in Phase 6 + Phase 8.*
- **D3 — Drift baked into the synthetic data + live adaptation demo.** Normal behavior *evolves* over
  simulated weeks; drift handling is demonstrated on real evolving data. *Built in Phase 1 (generator)
  + Phase 6 (detection/adaptation) + Phase 9 (experiment).*
- **D4 — Analyst alert-budget curve as a headline visual.** *"If your SOC reviews N alerts/day, here's
  your catch rate."* *Built in Phase 5 (budget logic) + Phase 8/9 (chart).*

**RECOMMENDED differentiators (build if core is on track):**
- **D5 — Calibrated risk + uncertainty bands.** Reliability-calibrated probabilities and a confidence
  band per score, widened for cold-start entities. *Built in Phase 5.*
- **D6 — Human-in-the-loop active learning feedback.** Analyst marks FP/confirmed → threshold/fusion
  weight for that entity/cohort visibly adjusts. *Built in Phase 7 (feedback store + adjustment) +
  Phase 8 (UI).*

**STRETCH differentiators (bonus, only after Phases 0–9 are stable — Phase 10):**
- **D7 — Graph-based lateral movement** (entity↔resource access graph).
- **D8 — Grounded LLM copilot** (analyst asks "why?" → answer grounded in the detection's own
  SHAP/baseline/MITRE; read-only, deterministic fallback).
- **D9 — Self-supervised sequence pretraining** (masked-event modeling before fine-tuning).

---

## 6. Build Scope (what we build, component by component)

Everything is built from scratch. There is no external code to adapt.

| Component | What it is | Phase |
|---|---|---|
| **Synthetic data generator** | Per-entity profiles, benign traffic, 7 attack injectors, correlated multi-stage campaigns, baked-in drift, held-out labels, documented taxonomy | 1 |
| **Feature pipeline** | One shared `featurize()`, persisted encoders/scalers, entity baselines, behavioral cohorts | 2 |
| **Baseline model** | Statistical deviation + Isolation Forest/OCSVM + tabular autoencoder | 3 |
| **Sequence model** | GRU/LSTM next-event surprise + sequence-autoencoder, per-step attribution | 4 |
| **Classifier + detectors + risk fusion** | Calibrated LightGBM multi-class, deterministic detectors, fusion → calibrated risk + uncertainty + alert budget | 5 |
| **Explainability** | SHAP, counterfactuals, sequence attribution, MITRE mapping, optional narrative | 6 |
| **Serving** | Stateless FastAPI scorer, persistence, campaign reconstruction, feedback loop, optional streaming | 7 |
| **Analyst dashboard** | React app: ranked alerts, explanation drawer, entity explorer, storyline, model performance, drift monitor | 8 |
| **Evaluation + report + demo** | Metrics harness, cold-start/drift/campaign experiments, figures, final report, demo script | 9 |
| **Bonus** | Graph lateral movement, LLM copilot, SSL pretraining | 10 (optional) |

Supporting infrastructure built in Phase 0: shared config, Pydantic models, global seed utility,
MongoDB (+ Redis) clients and index bootstrap, Docker Compose skeleton, a minimal read API and
scoring service (health-only, expanded later), the artifacts contract, and the test harness.

---

## 7. Requirements → Features Mapping

The hackathon mandates seven deliverables. Each maps to a phase.

| # | Mandatory Deliverable | Feature | Phase |
|---|---|---|---|
| 1 | Synthetic data generator + taxonomy | `data_generator/` — profiles, 7 attacks at 0.5–3%, correlated campaigns, baked-in drift, held-out labels, `TAXONOMY.md` | 1 |
| 2 | Baseline profiling model | `models/baseline.py` — statistical + Isolation Forest/OCSVM + tabular Autoencoder | 3 |
| 3 | Sequence-aware detection | `models/sequence.py` — GRU/LSTM (next-event surprise + seq-autoencoder) | 4 |
| 4 | Anomaly classification | `models/classifier.py` — LightGBM multi-class, calibrated | 5 |
| 5 | Explainability layer | `explainability/` — SHAP + counterfactuals + MITRE + narrative | 6 |
| 6 | Analyst dashboard | `frontend/` — Overview, Ranked Alerts + drawer, Entity Explorer, Model Performance, Drift, Storyline | 8 |
| 7 | Final report + presentation | `docs/FINAL_REPORT.md` + slides, backed by `evaluation/` | 9 |

### 7.1 Problem & Data Schema (self-contained recap)

Behavioral telemetry per access/connection event (the schema the generator produces and the scorer
consumes):

| Field | Description |
|---|---|
| `entity_id` | User ID or Device ID |
| `entity_type` | User / Service Account / Edge Device |
| `timestamp` | Access/connection time |
| `source_ip` / `geo` | Origin of access (ip, lat/lon, country) |
| `resource_accessed` | File, endpoint, port, or device function |
| `auth_method` | Password, Token, Certificate, Biometric |
| `session_duration` | Length of connection |
| `command_sequence` | Ordered list of commands/actions |
| `device_fingerprint` | OS/Firmware, MAC, protocol |
| `label` | Normal or anomaly type (training/eval only; hidden at inference, stored separately) |

**Anomaly classes (label space):** `normal, credential_misuse, lateral_movement, brute_force,
impossible_travel, credential_stuffing, device_spoofing, low_and_slow_exfil, insider_drift`.

**Challenges the system must handle:** sequential behavior, extreme class imbalance (attacks are
0.5–3% of events), concept drift, explainability, cold start.

---

## 8. Evaluation-Criteria Maximization

| Criterion | How we win | Evidence |
|---|---|---|
| Detection accuracy on imbalanced data | Layered fused detectors; imbalance-aware training + metrics | PR-AUC ≥ 0.90, ROC-AUC ≥ 0.95 |
| Correct anomaly-type classification | Supervised multi-class + deterministic detectors for geometric classes | Macro-F1 ≥ 0.85, confusion matrix |
| Low FP under top-1% budget | Explicit alert-budget ranking + recall@budget (D4) | Recall@1% ≥ 0.80, budget curve |
| Explainability | SHAP + counterfactuals (D2) + sequence attribution + MITRE + narrative | Explanation drawer, report plots |
| Cold-start | Peer-cohort priors + hierarchical fallback + uncertainty (D5) | Cold-start recall ≥ 0.70, ablation |
| Concept drift | Drift baked into data (D3) + PSI detection + rolling re-profiling + feedback (D6) | Drift experiment, live adaptation |
| Scalable/streaming design | Stateless scorer + optional Redis Streams; O(1) per-event scoring | <50 ms/event, live stream demo |
| Report quality | Structured report: assumptions, taxonomy, metrics, limitations | `docs/FINAL_REPORT.md` |
| (Bonus) Practical usefulness & demo | Campaign storyline (D1) + feedback loop (D6) | Live scripted demo |

---

## 9. Final System Architecture

Three planes: **offline (train)**, **online (serve)**, **presentation (dashboard)**.

```
   OFFLINE (train)
   data_generator/ ─► labeled dataset (+ campaigns, drift, held-out labels)
        │
        ▼
   features/ (fit encoders/scalers, entity baselines, cohorts, sequence vocab)
        │
        ▼
   training/build_artifacts.py ─► models/ (baseline · sequence · classifier)
                                   + risk fusion + calibration + budget threshold
        │                                   │
        ▼                                   ▼
   artifacts/ (models, scalers, profiles, thresholds, SHAP bg, manifest.json)
        │                       ▲
   evaluation/ ─────────────────┘ (PR-AUC, recall@budget, confusion, coldstart, drift)

   ONLINE (serve)
   events ─► serving/ (FastAPI + optional Redis Streams)
             1 featurize (shared featurize(); hierarchical cold-start)
             2 baseline + sequence scores
             3 if anomalous → classifier (type + probs) + deterministic detectors
             4 risk fusion → calibrated 0–100 + uncertainty band + alert-budget flag
             5 explainability (SHAP + counterfactual + MITRE + narrative)
             6 campaign linking (attach to / open a campaign)
             7 drift update; feedback-adjusted thresholds applied
             ▼
        MongoDB (detections · entity_profiles · campaigns · feedback · model_metrics · drift_state)
             ▼
   api/ (FastAPI read) ─► ranked detections, entity history, campaigns, metrics, drift, WS live
             ▼
   frontend/ (React) ─► Overview · Ranked Alerts + Explanation drawer · Entity Explorer ·
                        Storyline · Model Performance · Drift Monitor · Feedback
```

---

## 10. Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Rich ML ecosystem; CPU-friendly. |
| Data gen | NumPy, pandas, Faker | Standard synthetic-data tooling. |
| Classical ML | scikit-learn (IsolationForest, OneClassSVM, calibration, metrics), **LightGBM** | Strong on tabular + imbalance; SHAP-friendly. |
| Deep models | **PyTorch (CPU)** (Autoencoder, GRU/LSTM) | Lightweight, deterministic on CPU. |
| Explainability | **SHAP** (TreeExplainer for the classifier), counterfactual perturbation search, sequence step-error | Standard; TreeExplainer is exact + fast. |
| Scoring API | FastAPI + Uvicorn | Async, typed, minimal. |
| Read API | FastAPI | Powers the dashboard. |
| Persistence | MongoDB (Motor async) | Flexible nested detection documents. |
| Streaming (opt) | Redis Streams | Demonstrates the "real-time" criterion. |
| Frontend | React 18 + Vite + TypeScript + Tailwind + Recharts + axios + react-router + lucide | Fast, modern analyst UI. |
| Narrative (opt) | Groq + Llama 3.x | Optional narrator; deterministic template fallback. |
| MITRE map | Static JSON (default) or Qdrant (optional) | Zero-dependency fallback. |
| Packaging | Docker Compose | Reproducible local demo. |
| Report | Markdown → PDF | Deliverable #7. |

> Pin exact versions in each `requirements.txt`. Notable compatibility pin: with `motor==3.5.x`, pin
> `pymongo==4.8.0` (motor 3.5.x is incompatible with `pymongo>=4.9`). Install PyTorch CPU-only unless a
> GPU is guaranteed.

---

## 11. Folder Structure (target)

```
project-root/
├── data_generator/   profiles.py · normal.py · attacks.py · campaigns.py · drift.py · generate.py · TAXONOMY.md
├── features/         event_features.py · session_features.py · entity_window.py · sequences.py · geo.py · cohorts.py · encoders.py · featurize.py
├── models/           baseline.py · sequence.py · classifier.py · detectors.py · risk.py · drift.py
├── explainability/   shap_explainer.py · counterfactual.py · sequence_attribution.py · mitre_map.py · narrative.py
├── training/         build_baselines.py · train_baseline.py · train_sequence.py · train_classifier.py · build_artifacts.py
├── evaluation/       metrics.py · coldstart_experiment.py · drift_experiment.py · campaign_experiment.py · report_figures.py
├── serving/          app.py · pipeline.py · campaign.py · feedback.py · stream_consumer.py · replay.py · Dockerfile · requirements.txt
├── api/              main.py · routers/(detections,entities,campaigns,metrics,drift,feedback,system,ws) · services/ · Dockerfile · requirements.txt
├── common/           config.py · models.py · seed.py · database.py
├── frontend/         (Vite React app) src/pages/(Overview,Alerts,EntityExplorer,Storyline,ModelPerformance,Drift,SystemHealth) · Dockerfile · nginx.conf
├── artifacts/        models, scalers, profiles, thresholds, manifest.json, metrics.json   (binaries git-ignored)
├── notebooks/        eda.ipynb (optional)
├── docs/             FINAL_REPORT.md
├── tests/            unit + integration + demo_script.md
├── docker-compose.yml
├── requirements.txt · pytest.ini · conftest.py · .gitignore · .env.example · README.md
└── Hackathon_Implementation_Plan.md   (this file)
```

---

## 12. Database Design (MongoDB `anomaly_detection`)

- **`events`** (optional raw/replay): `{event_id, entity_id, entity_type, timestamp, source_ip, geo{lat,lon,country}, resource_accessed, auth_method, auth_success, session_id, session_duration, command_sequence[], device_fingerprint{os,mac,protocol}, ingested_at}`
- **`entity_profiles`**: `{entity_id, entity_type, cohort, session_count, first_seen, last_seen, typical_login_hours[], typical_geo[], typical_resources{}, auth_method_dist{}, feature_means[], feature_stds[], sequence_ngram_profile{}, cold_start, drift{psi,last_refresh,status}, feedback_threshold_adjust, updated_at}`
- **`detections`**: `{detection_id, entity_id, entity_type, timestamp, event_ref, session_id, scores{baseline,sequence,classifier_confidence}, risk_score, risk_uncertainty, in_alert_budget, is_anomaly, anomaly_type, anomaly_type_probs{}, explanation{top_features[],counterfactual{},sequence_attribution[],mitre[],narrative,narrative_source,baseline_comparison{}}, campaign_id, cold_start, drift_flag, status, ground_truth_label?, analyst_feedback?, created_at}`
- **`campaigns`** (D1): `{campaign_id, entity_id, started_at, last_activity, stages[{anomaly_type, detection_id, timestamp}], detection_ids[], kill_chain[], max_risk, status}`
- **`feedback`** (D6): `{feedback_id, detection_id, entity_id, analyst_verdict(confirmed|false_positive), created_at, applied{scope, adjustment}}`
- **`model_metrics`**: `{run_id, created_at, dataset_summary{n,anomaly_rate,per_class_counts}, pr_auc, roc_auc, recall_at_1pct_budget, precision_at_k[], confusion_matrix[[]], per_class{}, coldstart{}, drift{}, calibration_ece}`

**Indexes:** `detections`: `{detection_id:1}` unique, `{risk_score:-1}`, `{entity_id:1,timestamp:-1}`, `{anomaly_type:1}`, `{campaign_id:1}`, `{created_at:-1}`. `entity_profiles`: `{entity_id:1}` unique. `campaigns`: `{campaign_id:1}` unique, `{entity_id:1,last_activity:-1}`. `events`: `{event_id:1}` unique, `{entity_id:1,timestamp:-1}`. `feedback`: `{detection_id:1}`, `{created_at:-1}`. `model_metrics`: `{run_id:1}` unique, `{created_at:-1}`.

---

## 13. Data Flow

**Offline:** generate labeled dataset (with campaigns + drift) → fit features/baselines/cohorts →
train models + fuse + calibrate + set budget → evaluate → write `artifacts/` + `model_metrics`.

**Online (per event):** ingest → featurize (cold-start fallback) → baseline + sequence scores → (if
gated) classifier + detectors → risk fusion (calibrated, uncertainty, budget) → explainability (SHAP
+ counterfactual + MITRE + narrative) → campaign link → persist + drift update + apply feedback
adjustments → WS push → dashboard.

---

## 14. AI / Model Pipeline

**Tier 1 — Baseline (unsupervised; cold-start + zero-day):** per-entity statistical profile deviation
+ Isolation Forest/OCSVM + tabular Autoencoder reconstruction error → normalized `baseline_score`.

**Tier 2 — Sequence (self-supervised; order):** GRU/LSTM over command/resource sequences; score =
next-event surprise (mean NLL) and/or sequence-autoencoder error → normalized `sequence_score`;
per-step attribution for explainability.

**Tier 3 — Classifier (supervised; type):** LightGBM multi-class over engineered features + tier-1/2
scores; class-weighted; **calibrated** (isotonic/Platt). Classes as in §7.1.

**Deterministic detectors:** impossible travel (haversine/velocity), brute force (failed-auth burst)
— feed the classifier as features and override the type when firing with high confidence.

**Risk fusion (`models/risk.py`):** weighted fusion of scores → calibrated 0–100 + **uncertainty
band** (D5); **alert budget** flags top 1% by risk (D4). Weights/threshold tuned on validation to
maximize recall@1%.

**Cold start:** entity with < `entity_history_min_sessions` → score against **cohort** priors then
global (hierarchical), widen threshold, set `cold_start`, widen uncertainty.

**Drift (`models/drift.py`):** PSI between recent window and baseline; rolling re-profiling absorbs
benign change; abrupt shifts still spike; feedback (D6) nudges per-entity/cohort thresholds.

---

## 15. Behavioral Analytics / Feature Pipeline

Feature groups (fit offline, applied online via one `featurize()`): **temporal** (hour, day, off-hours
vs entity norm, inter-event delta), **geo/network** (country, geo-velocity, new-country, IP/ASN
novelty), **resource/access** (resource novelty, breadth, sensitive flag, access z-score), **auth**
(method vs usual dist, change flag, fail/success ratio), **sequence** (tokens, n-gram novelty, length,
rare transitions → Tier 2), **device** (fingerprint hash, OS/MAC/protocol consistency, change flag →
spoofing), **volume/rate** (events/window, failed-auth burst → brute force). Numeric standardized via
a persisted `StandardScaler`; categoricals via persisted encoders; unseen categories → novelty flag,
never a crash.

---

## 16. Backend Architecture

- **`serving/app.py`:** loads artifacts once (lifespan + manifest schema check); `/score`,
  `/score/batch`, `/health`; stateless; optional `stream_consumer.py` (Redis Streams); writes
  detections + updates profiles/drift/campaigns; applies feedback adjustments; WS notify.
- **`api/`:** FastAPI read API with CORS, lifespan, WebSocket; routers → `detections`, `entities`,
  `campaigns`, `metrics`, `drift`, `feedback`, `dashboard/summary`, `system`, `ws`.

Separating scoring from dashboard reads isolates scoring latency and lets us demo throughput.

---

## 17. Frontend & Dashboard

A React 18 + Vite + TypeScript app (Tailwind, Recharts, axios, react-router, lucide). Routes:
`/` Overview · `/alerts` Ranked Alerts (+ Explanation drawer) · `/entities` Entity Explorer ·
`/storyline` Campaigns (D1) · `/performance` Model Performance (+ alert-budget curve D4) ·
`/drift` Drift Monitor (D3) · `/system` Health.

**Explanation drawer (the money shot):** risk gauge + uncertainty band, anomaly type + calibrated
confidence, **SHAP contribution chart**, **counterfactual "make-it-normal" panel (D2)**,
sequence-step highlight, baseline comparison, MITRE chips, narrative, **feedback buttons (D6)**, and a
"part of campaign X" link (D1).

---

## 18. Explainability Features (Deliverable #5)

- **Global:** SHAP summary over the classifier (report + Model Performance page).
- **Local:** SHAP waterfall → top contributing features (value, direction, vs baseline).
- **Counterfactual (D2):** minimal feature changes that flip the verdict to benign, in plain terms.
- **Sequence:** per-step error/attention highlighting the anomalous span.
- **Baseline comparison:** structured diff vs the entity's learned profile.
- **MITRE mapping:** anomaly class → technique(s) — brute_force→T1110, impossible_travel/
  credential_misuse→T1078, lateral_movement→T1021/T1210, low_and_slow_exfil→T1048,
  device_spoofing→T1036/T1200, credential_stuffing→T1110.004; static map default, Qdrant optional.
- **Narrative (optional):** one Groq/Llama call → 2–3 analyst sentences; deterministic template
  fallback; never affects score/verdict.

---

## 19. Security Considerations

Supporting concern, not scored. The scoring endpoint requires a bearer token (`scoring_auth_token`);
the read API is read-only; Pydantic validates all input; secrets come from `.env` (git-ignored);
synthetic data only (no real PII). If exposing any service on a network, note that authentication is
required and never ship an unauthenticated write endpoint silently.

---

## 20. APIs

**Scoring (`serving/`):** `POST /score` (one event → detection), `POST /score/batch` (array),
`GET /health`.
**Read (`api/`):** `GET /api/v1/dashboard/summary`, `GET /api/v1/detections?sort=risk&type=&entity_type=&cold_start=&skip=&limit=`,
`GET /api/v1/detections/{id}`, `GET /api/v1/entities/{id}`, `GET /api/v1/campaigns` & `/campaigns/{id}`,
`GET /api/v1/metrics`, `GET /api/v1/drift`, `POST /api/v1/feedback`, `WS /api/v1/ws`,
`GET /api/v1/health`, `GET /api/v1/system/health`.

---

## 21. Deployment Strategy

Local-first Docker Compose with `mongodb`, `redis` (optional streaming), `serving`, `api`, and `ui`.
Trained artifacts are produced on the host by `training/build_artifacts.py` and mounted read-only into
`serving`. Training never runs in the live path. Quickstart:
`docker compose up` + a `make seed` (`python -m data_generator.generate && python -m
training.build_artifacts && python -m serving.replay`). CPU-only; models train in minutes, score in
milliseconds. The `.env` file is optional (all settings have sensible defaults).

---

## 22. Testing Strategy

**Testing is a mandatory gate after every phase (see §3 Phase Completion Gate), not a final step.**
Each phase writes and runs its own tests before it is considered complete; a phase does not advance
until its tests are green and no earlier phase has regressed. Tests are fast and CPU-only.

Per-phase coverage: generator (injection rate, schema, per-class presence, campaign integrity, drift
presence), features (determinism, encoder round-trip, cold-start path, geo math), models (training
smoke, scoring determinism, gate/budget boundaries), evaluation (metric thresholds fail the build if
unmet), serving (per-class scoring, offline/online parity, 4xx on bad input, WS emit, latency),
API/UI (contract tests, drawer render). End-to-end: generate → train → replay → dashboard shows each
attack class caught, ranked, explained, and grouped into campaigns.

`tests/` is scoped via `pytest.ini` (`testpaths = tests`); a root `conftest.py` puts the project root
on `sys.path`.

---

## 23. Demo Strategy

Deterministic (fixed seed), ~5 min: (1) calm Overview — "already learned normal"; (2) benign traffic
— dashboard stays quiet; (3) scripted multi-stage attack streamed live — alerts appear ranked; (4)
open one alert — SHAP + **counterfactual** + sequence + MITRE + narrative; (5) **Storyline** — the same
attack reconstructed as a campaign; (6) **cold-start** — new entity scored via cohort priors, no false
alarm; (7) **drift** — benign shift absorbed; (8) **feedback** — mark an FP, show the threshold adapt;
(9) **Model Performance** — PR-AUC, recall@1% budget curve, confusion matrix. Rehearse 3–4× against
the fixed seed; narrative uses fallback if offline.

---

## 24. Phase-wise Development Plan (Executable Spec)

> Each phase is a standalone work order. Complexity: **S** hours · **M** ½–1 day · **L** 1–2 days ·
> **XL** 2+ days. Every phase must satisfy the universal Definition of Done in §3 — including the
> **Phase Completion Gate: tests are written and run after building the phase, and must pass before
> advancing.** Each phase's `Verification` block lists the exact test/commands that gate it.

---

### Phase 0 — Project Setup & Foundations
**Objective:** From an empty folder, create the ML-first skeleton, shared contracts, and the test
harness that every later phase builds on.
**Delivers:** foundation for all phases.
**Prerequisites:** an empty project directory (initialize git here).

**Files to Create:**
- `common/__init__.py`.
- `common/config.py`: `pydantic-settings` `Settings` (env prefix e.g. `ADP_`, `extra="ignore"` so
  stray env vars never crash startup). Include: `random_seed=42`; Mongo (`mongo_url`,
  `mongo_db_name="anomaly_detection"`); Redis (`redis_url`, optional stream keys); artifact paths
  (`artifacts_dir`, `dataset_dir`, `manifest_filename`, `artifact_schema_version`);
  `alert_budget_pct=0.01`; `entity_history_min_sessions`; drift params (window, PSI threshold, min
  samples, refresh alpha); fusion weight defaults (sum to 1.0) + `anomaly_gate_threshold`; feature
  params (`sequence_max_len`, `sequence_ngram_n`, `cohort_count`, `impossible_travel_kmh`,
  `brute_force_threshold`, `brute_force_window_minutes`); `scoring_auth_token`; optional LLM
  (`llm_enabled`, `groq_api_key`, `groq_model`, temperature, max tokens); `mitre_map_source`
  ("static"|"qdrant"); `api_cors_origins`.
- `common/models.py`: Pydantic v2. Enums `EntityType`, `AuthMethod`, `AnomalyType` (9 classes; expose
  an ordered `ANOMALY_CLASSES` list), `DetectionStatus`, `CampaignStatus`, `AnalystVerdict`. Value
  objects `GeoLocation`, `DeviceFingerprint`. Models `Event`, `Session`, `EntityProfile` (+ `DriftState`),
  `Explanation` (+ `FeatureAttribution`, `Counterfactual`, `MitreTechnique`), `Detection`
  (+ `DetectionScores`), `Campaign` (+ `CampaignStage`), `Feedback`, `ModelMetrics`. Include lenient
  string→int/float/bool coercion helpers (`LenientInt/Float/Bool`) for the optional LLM narrator.
  `Event.label/campaign_id/stage` are optional (labels stored separately, never required for scoring).
  All datetimes serialize to ISO strings via `model_dump(mode="json")`.
- `common/seed.py`: `set_global_seed(seed=None)` seeding `random`, NumPy, and (guarded import) PyTorch
  + `PYTHONHASHSEED`; `lightgbm_params(seed=None)` returning deterministic LightGBM params
  (`deterministic=True`, single-thread, seeded).
- `common/database.py`: Motor + async Redis clients (lazy singletons), health checks, a `Collections`
  constants class, and `ensure_indexes()` creating the §12 indexes (idempotent).
- Package dirs with `__init__.py`: `data_generator/`, `features/`, `models/`, `explainability/`,
  `training/`, `evaluation/`, `serving/`.
- `artifacts/.gitkeep` and `artifacts/manifest.json` (initial schema: `schema_version`, `seed`,
  null placeholders; populated by training later).
- Root `requirements.txt` (pinned: numpy, pandas, pyarrow, scipy, Faker, scikit-learn, lightgbm,
  torch (cpu), shap, fastapi, uvicorn, httpx, motor, `pymongo==4.8.0`, redis, pydantic,
  pydantic-settings, python-dotenv, matplotlib, pytest).
- `pytest.ini` (`testpaths = tests`) and root `conftest.py` (put project root on `sys.path`).
- `.gitignore` (`.venv/`, `__pycache__/`, `.pytest_cache/`, `.env`, `artifacts/*` except `.gitkeep`
  and `manifest.json`, model binaries).
- `.env.example` (documented, all-optional `ADP_` vars).
- `README.md` (quickstart: venv → install → generate → train → compose up → replay).
- `docker-compose.yml`: services `mongodb`, `redis`, `serving`, `api` (and `ui`, added in Phase 8);
  `.env` optional (`env_file: [{path: .env, required: false}]`); named volumes; a bridge network.
- Minimal read API: `api/__init__.py`, `api/main.py` (FastAPI, CORS, lifespan calling
  `ensure_indexes()` best-effort, `GET /api/v1/health`), `api/routers/__init__.py`,
  `api/routers/system.py` (`GET /api/v1/system/health` → Mongo + Redis checks),
  `api/services/__init__.py`, `api/services/health.py`, `api/Dockerfile`, `api/requirements.txt`.
- Minimal scoring service: `serving/app.py` (FastAPI, `GET /health` + a manifest schema check;
  scoring endpoints arrive in Phase 7), `serving/Dockerfile`, `serving/requirements.txt`.
- `tests/__init__.py`, `tests/test_models.py`, `tests/test_seed.py`, `tests/test_config.py`.

**Build Tasks:** 1) config + seed + models; 2) database clients + `ensure_indexes()`; 3) package dirs
+ artifacts contract; 4) requirements + pytest/conftest + gitignore + env example + README; 5)
compose + minimal API + minimal serving; 6) write and run the unit tests.

**Acceptance Criteria:** the package imports cleanly; the Pydantic models validate sample payloads and
round-trip through `model_dump(mode="json")`; the seed utility is reproducible; `docker compose config`
is valid; `GET /api/v1/health` and the serving `/health` return 200 (verifiable via FastAPI TestClient
without a running DB); all Phase 0 tests pass.

**Verification:** `pip install` the light subset (numpy, pydantic, pydantic-settings, pytest — plus
fastapi/motor/pymongo/redis/httpx for the health check); `pytest`; `docker compose config`;
TestClient hits on the two health endpoints.

**Risks:** dependency version conflicts → pin versions (esp. `pymongo==4.8.0` with motor 3.5.x).
**Complexity: M.**

---

### Phase 1 — Synthetic Data Generator + Taxonomy (Deliverable #1; D1, D3)
**Objective:** Produce a realistic, labeled behavioral dataset with documented taxonomy, **correlated
multi-stage attack campaigns (D1)**, and **baked-in concept drift (D3)**.
**Prerequisites:** Phase 0.

**Files to Create:**
- `data_generator/profiles.py`: per-entity normal profiles (login hours, geo home, typical resources,
  auth dist, devices) for Users / Service Accounts / Edge Devices, grouped into latent **cohorts**.
- `data_generator/normal.py`: benign session/event generation from profiles with realistic noise.
- `data_generator/attacks.py`: 7 injectors — brute force, impossible travel, credential stuffing,
  lateral movement, device spoofing, low-and-slow exfil, insider drift — at a configurable 0.5–3%
  rate; each returns labeled events with the attack class.
- `data_generator/campaigns.py` (D1): emit **multi-stage** attacks where stages share an entity and
  are time-ordered (e.g., brute_force → login → lateral_movement → low_and_slow_exfil), tagged with a
  `campaign_id` and `stage` for ground truth.
- `data_generator/drift.py` (D3): evolve a subset of entities' *normal* behavior over the simulated
  timeline (schedule/device/location shift) — labeled benign, to prove drift adaptation later.
- `data_generator/generate.py`: CLI → writes `artifacts/dataset/{events.parquet, labels.parquet,
  entities.json, campaigns.json}`; train/val/test split by time; labels stored **separately**.
- `data_generator/TAXONOMY.md`: documented assumptions, schema, each behavior's signal, injection
  rates, campaign structure, drift design, limitations.
- `notebooks/eda.ipynb` (optional): distributions + separability sanity check.

**Build Tasks:** profiles → normal → attacks → campaigns → drift → generate/split → TAXONOMY → EDA.

**Acceptance Criteria:** schema matches §7.1; overall anomaly rate within 0.5–3%; all 7 classes present
and separable-but-not-trivial; ≥ 1 multi-stage campaign type present with correct `campaign_id`/`stage`
labels; a labeled benign drift cohort exists; labels stored separately from features; deterministic
under seed 42.

**Verification:** `python -m data_generator.generate --seed 42`; `pytest tests/test_generator.py`
(asserts rate, schema, per-class presence, campaign integrity, drift presence).

**Risks:** data too easy (inflated metrics) or too hard → tune noise/overlap; document in TAXONOMY.
**Complexity: L.**

---

### Phase 2 — Feature Engineering + Cohorts (enables cold-start)
**Objective:** Turn events into model-ready features with one shared `featurize()`, fitted persisted
encoders, entity baselines, and **behavioral cohorts** for cold-start.
**Prerequisites:** Phase 1.

**Files to Create:**
- `features/event_features.py`, `session_features.py`, `entity_window.py`, `sequences.py` (vocab +
  encoding), `geo.py` (haversine/velocity), `cohorts.py` (cluster entities into cohorts for priors),
  `encoders.py` (fit/persist scalers + categorical encoders), `features/featurize.py` (the single
  shared entry point used offline + online).
- `training/build_baselines.py`: compute `entity_profiles` (means/stds, typical hours/geo/resources,
  n-gram profile) + cohort priors → `artifacts/`.

**Build Tasks:** implement feature groups (§15) → cohorts → fit + persist encoders → build baselines →
`featurize()` → tests.

**Acceptance Criteria:** `featurize()` offline == online for identical input; encoders round-trip;
cold-start fallback returns valid vectors + cohort assignment for unseen entities; geo-velocity correct
on known cases.

**Verification:** `python -m training.build_baselines`; `pytest tests/test_features.py`.

**Risks:** train/serve skew → single `featurize()`; unseen categories → novelty flag. **Complexity: L.**

---

### Phase 3 — Baseline Profiling Model (Deliverable #2)
**Objective:** Unsupervised anomaly scoring + hierarchical cold-start.
**Prerequisites:** Phase 2.

**Files to Create:** `models/baseline.py` (statistical deviation + IsolationForest/OCSVM + PyTorch
tabular Autoencoder; normalized `baseline_score`; `score_baseline(features, profile)`; hierarchical
cold-start path), `training/train_baseline.py`.
**Files to Modify:** `training/build_artifacts.py` (create it here to orchestrate artifact building).

**Build Tasks:** implement statistical + IsoForest/OCSVM + AE → normalize/combine → cold-start branch →
train script → tests.

**Acceptance Criteria:** baseline alone gives clear PR-AUC uplift over random on held-out; cold-start
path produces calibrated scores for new entities via cohort/global priors; artifacts written with
manifest.

**Verification:** `python -m training.train_baseline`; `pytest tests/test_baseline.py`.

**Risks:** AE over/underfit → small net, early stop. **Complexity: L.**

---

### Phase 4 — Sequence Model (Deliverable #3)
**Objective:** Order-aware anomaly scoring + step attribution.
**Prerequisites:** Phase 2 (sequence encoding).

**Files to Create:** `models/sequence.py` (GRU/LSTM; next-event surprise + sequence-autoencoder;
normalized `sequence_score`; `score_sequence(sequence)`; `attribute_sequence(sequence)` for per-step
weights), `training/train_sequence.py`.
**Files to Modify:** `training/build_artifacts.py`.

**Build Tasks:** dataset of sequences → model → train (next-event and/or seq-AE) → scoring +
attribution → tests.

**Acceptance Criteria:** sequence model improves recall on lateral-movement / low-and-slow vs
baseline-only; handles variable length (`<pad>`/`<unk>`); attribution returns per-step scores;
deterministic.

**Verification:** `python -m training.train_sequence`; `pytest tests/test_sequence.py`.

**Risks:** sparse vocab → pad/truncate, `<unk>`. **Complexity: L.**

---

### Phase 5 — Classifier + Detectors + Risk Fusion (Deliverable #4; D4, D5)
**Objective:** Name the attack; add geometric certainties; produce the **calibrated risk score with
uncertainty (D5)** and the **alert budget (D4)**.
**Prerequisites:** Phases 3–4 (scores as features).

**Files to Create:**
- `models/classifier.py`: LightGBM multi-class over features + tier-1/2 scores; class-weighted;
  **isotonic/Platt calibration**; `classify(features)` → type + calibrated probs.
- `models/detectors.py`: impossible-travel (haversine/velocity) + brute-force (burst rate); return fire
  + confidence; feed classifier + override type when highly confident.
- `models/risk.py`: fuse baseline + sequence + classifier confidence → **calibrated 0–100 risk** +
  **uncertainty band (D5, widened for cold-start)**; **alert-budget** flag (top `alert_budget_pct`);
  tune weights/threshold on validation for recall@1%.
- `training/train_classifier.py`.
**Files to Modify:** `training/build_artifacts.py` (train classifier, fit fusion + calibration + budget
threshold; write all to `artifacts/`).

**Build Tasks:** build classifier → calibrate → detectors → fusion + uncertainty + budget → tune on
val → tests.

**Acceptance Criteria:** macro-F1 ≥ 0.85; calibration ECE ≤ 0.05 (reliability check); detectors ≈ 1.0
precision on injected impossible-travel/brute-force; **recall@1% budget ≥ 0.80** on val; budget flag
yields ~1% volume; uncertainty band wider for cold-start.

**Verification:** `python -m training.train_classifier`; `pytest tests/test_classifier.py
tests/test_detectors.py tests/test_risk.py`.

**Risks:** rare-class imbalance → class weights + threshold tuning. **Complexity: XL.**

---

### Phase 6 — Explainability + Counterfactuals + Drift (Deliverable #5; D2, D3)
**Objective:** Make every alert explainable, add **counterfactual "nearest-normal" (D2)**, MITRE
mapping, optional narrative, and **concept-drift detection/adaptation (D3)**.
**Prerequisites:** Phase 5 (classifier for SHAP), Phase 4 (seq attribution).

**Files to Create:**
- `explainability/shap_explainer.py`: TreeExplainer on the classifier; local top-features (value,
  direction, vs baseline) + global summary; sampled background from `artifacts/`.
- `explainability/counterfactual.py` (D2): minimal-change perturbation search over top mutable features
  → the smallest changes that flip risk below the alert threshold, in plain terms.
- `explainability/sequence_attribution.py`: surface Phase-4 per-step weights.
- `explainability/mitre_map.py`: static class→technique map (Qdrant optional).
- `explainability/narrative.py`: optional Groq/Llama narrative with **deterministic template fallback**
  (never affects score).
- `models/drift.py` (D3): PSI/distribution comparison recent-window vs baseline; rolling re-profiling;
  drift status per entity; `update_drift(entity, window)`.
**Files to Modify:** `models/risk.py` (attach `explanation`); `common/config.py` (LLM optional, drift
params — already present from Phase 0).

**Build Tasks:** SHAP → counterfactual search → seq attribution surfacing → MITRE map → narrative +
fallback → drift detection/adaptation → tests.

**Acceptance Criteria:** every detection has non-empty `top_features`, a valid `counterfactual`, and
MITRE mapping; narrative falls back cleanly with no API key; drift experiment shows PSI rising on
injected abrupt shift and staying low (adapting) on the benign drift cohort from Phase 1.

**Verification:** `pytest tests/test_explainability.py tests/test_counterfactual.py
tests/test_drift.py`.

**Risks:** SHAP/counterfactual latency → TreeExplainer + bounded search iterations + feature subset.
**Complexity: L.**

---

### Phase 7 — Serving + Persistence + Campaigns + Feedback (D1, D6)
**Objective:** Online scoring writing detections to Mongo, streaming-capable, with **campaign/kill-chain
reconstruction (D1)** and the **feedback/active-learning loop (D6)**.
**Prerequisites:** Phases 2–6 (artifacts).

**Files to Create / Expand:**
- `serving/app.py`: load artifacts once (lifespan + manifest schema check); `/score`, `/score/batch`,
  `/health`; bearer-token auth; 4xx on bad input.
- `serving/pipeline.py`: featurize → baseline+sequence → gate → classifier+detectors → risk
  (calibrated, uncertainty, budget) → explainability → persist → drift update → WS notify.
- `serving/campaign.py` (D1): link a new detection to an open campaign for the entity (time-window +
  entity + stage ordering) or open a new one; maintain `kill_chain` and `max_risk`.
- `serving/feedback.py` (D6): apply analyst verdict → adjust the entity/cohort threshold; persist to
  `feedback`; expose the applied adjustment.
- `serving/stream_consumer.py`: optional Redis Streams consumer.
- `serving/replay.py`: replay `artifacts/dataset` (or a demo subset) through `/score` for the demo.
**Files to Modify:** `docker-compose.yml` (`serving` wired to artifacts + mongo + redis);
`api/routers/` (add `detections`, `entities`, `campaigns`, `metrics`, `drift`, `feedback`,
`dashboard`, `ws`).

**Build Tasks:** scoring pipeline → persistence → campaign linking → feedback/active-learning → stream
consumer → replay → tests (incl. latency).

**Acceptance Criteria:** `POST /score` returns the correct type per canned attack event; batch ==
offline scoring; detections persisted with full explanation + `campaign_id`; injected multi-stage
attacks reconstructed (≥ 90% stages linked correctly); posting feedback visibly changes the entity
threshold; malformed input → 4xx; median latency < 50 ms.

**Verification:** `docker compose up -d serving`; `python -m serving.replay --seed 42`;
`pytest tests/test_serving.py tests/test_campaign.py tests/test_feedback.py`.

**Risks:** artifact/version mismatch → manifest schema check at startup. **Complexity: L.**

---

### Phase 8 — Analyst Dashboard (Deliverable #6; surfaces D1, D2, D4, D6)
**Objective:** Build the analyst UI from scratch (Vite + React + TS + Tailwind + Recharts).
**Prerequisites:** Phase 7 (read API).

**Files to Create:** the Vite React app under `frontend/` — `package.json`, Vite/TS/Tailwind config,
`index.html`, `src/main.tsx`, `src/App.tsx` (routes), `src/api/client.ts` + `src/api/types.ts`
(axios client + typed models mirroring the API), layout components, and pages: `Overview.tsx`,
`Alerts.tsx` (+ Explanation drawer), `EntityExplorer.tsx`, `Storyline.tsx` (D1 timeline),
`ModelPerformance.tsx` (incl. **alert-budget curve** D4), `Drift.tsx`, `SystemHealth.tsx`; components
`RiskGauge.tsx`, `ShapChart.tsx`, `CounterfactualPanel.tsx` (D2), `SequenceHighlight.tsx`,
`FeedbackButtons.tsx` (D6); `frontend/Dockerfile` + `nginx.conf` (serve build, proxy `/api`).
**Files to Modify:** `docker-compose.yml` (add the `ui` service).

**Build Tasks:** scaffold app → client/types → Overview → Ranked Alerts + drawer (SHAP, counterfactual,
sequence, MITRE, narrative, feedback, campaign link) → Entity Explorer → Storyline → Model Performance
(+ budget curve) → Drift → sidebar/routes → contract tests.

**Acceptance Criteria:** alerts sort by risk with severity colors + filters; drawer shows SHAP +
counterfactual + sequence highlight + baseline diff + MITRE + narrative + feedback buttons + campaign
link; Storyline renders a reconstructed campaign; Entity Explorer shows baseline + history + cold-start
badge; Model Performance shows PR-AUC/recall@1% budget curve/confusion; Drift page renders; feedback
POST round-trips and reflects the adjustment.

**Verification:** `cd frontend && npm install && npm run build`; run the stack; click through each
page; contract tests.

**Risks:** API shape drift → lock `types.ts` to the backend models. **Complexity: L.**

---

### Phase 9 — Evaluation, Report & Demo (Deliverable #7; proves every criterion)
**Objective:** Produce the metrics, the report, and a rehearsed, deterministic demo.
**Prerequisites:** Phases 5–8.

**Files to Create:** `evaluation/metrics.py` (PR-AUC, ROC-AUC, recall@budget, precision@k, confusion,
per-class, ECE), `coldstart_experiment.py` (ablation with/without priors), `drift_experiment.py` (FP
before/after adaptation), `campaign_experiment.py` (stage-linking accuracy), `report_figures.py` (all
plots); `docs/FINAL_REPORT.md`; `tests/demo_script.md`.
**Files to Modify:** `training/build_artifacts.py` (populate `model_metrics`).

**Build Tasks:** metrics harness → cold-start/drift/campaign experiments → figures → report
(assumptions, taxonomy, models, imbalance-aware metrics, cold-start, drift, campaigns, limitations,
future work) → slides → demo script → 3× dry-run.

**Acceptance Criteria:** all §4 targets met and reported with *actual* numbers (or, if any target is
missed, the report states the real number honestly and explains why); report covers all required
sections; demo runs identically under seed 42; metric regression thresholds enforced in tests.

**Verification:** `python -m evaluation.metrics`; `python -m evaluation.coldstart_experiment`;
`python -m evaluation.drift_experiment`; `python -m evaluation.campaign_experiment`;
`pytest tests/test_metrics.py`; full demo dry-run.

**Risks:** time crunch → this phase is protected; cut Phase 10 first, never this. **Complexity: L.**

---

### Phase 10 — Bonus Differentiators (Optional; D7, D8, D9)
**Objective:** Extra technical depth if Phases 0–9 are stable.
**Prerequisites:** Phases 0–9 complete and demo-ready.

- **D7 — Graph-based lateral movement:** entity↔resource bipartite graph; unusual-breadth/path
  detection (graph metrics, optional GNN) feeding the classifier/risk. Files: `models/graph.py`,
  training + eval hooks.
- **D8 — Grounded LLM copilot:** read-only Q&A grounded in a detection's SHAP/baseline/MITRE;
  deterministic fallback. Files: `explainability/copilot.py`, a dashboard chat panel.
- **D9 — Self-supervised sequence pretraining:** masked-event objective before fine-tuning in
  `models/sequence.py`.

**Acceptance Criteria:** each addition improves a metric or demo moment without destabilizing the core;
all remain optional and degradable.
**Risks:** scope creep → strictly time-boxed; never at the expense of Phase 9. **Complexity: M–XL.**

---

## 25. Justification of Major Decisions

- **Layered detectors + fusion.** Each tier targets specific criteria (cold-start/zero-day, order,
  type); fusion beats any single model and yields better explanations. *(Highest-impact decision.)*
- **Imbalance-aware from day one.** PR-AUC, recall@budget, and an explicit alert budget — never raw
  accuracy — because attacks are 0.5–3% of events.
- **Explainable by construction.** SHAP + counterfactuals + sequence attribution + MITRE are built into
  the pipeline, not bolted on; explainability is the strongest differentiator in this brief.
- **LLM as narrator, not decider.** ML owns correctness; the optional LLM only phrases an explanation
  and always has a deterministic fallback, so the demo never depends on the internet.
- **Innovations built into phases, not bolted on.** Campaigns (D1), counterfactuals (D2), drift-in-data
  (D3), and the budget curve (D4) map directly onto judging language and are cheap where placed.
- **Determinism + artifacts contract.** A single seed and a versioned `artifacts/` make the whole build
  reproducible and the demo repeatable.

---

## 26. Assumptions & Conventions Recap

- **Synthetic data only** — no real PII; the generator's assumptions and injection rates are documented
  in `data_generator/TAXONOMY.md`.
- **CPU-only** — PyTorch CPU build; everything trains in minutes and scores in milliseconds.
- **Deterministic** — global seed 42 across NumPy/`random`/PyTorch/LightGBM; reproducible datasets,
  training, and scoring.
- **Train/serve parity** — one `featurize()` offline and online; serving loads only from `artifacts/`
  and never retrains.
- **Graceful degradation** — LLM narrative, Qdrant MITRE store, and Redis streaming are all optional
  with deterministic fallbacks; nothing in the demo path requires the internet.
- **Testing gate** — every phase writes and runs tests before advancing; no earlier phase may regress.
- **Report honesty** — report *actual* metric numbers; if a target is missed, say so and explain why.

---

*End of plan. Start in an empty folder with this file, then drive the build by saying "do phase 0",
then "do phase 1", and so on. Each phase ends demo-able and tested. Update this file as scope evolves.*
