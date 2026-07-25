# Project State — Handoff

> **Read this first, then `AI_Powered_Behavioral_Anomaly_Detection_Hackathon_Details.md` (the
> official brief, authoritative) and `Hackathon_Implementation_Plan.md` (the executable spec).**
>
> Purpose: transfer full working context to a new session. Everything below is verified against the
> repository, not remembered.

---

## 1. What this is

An AI/ML behavioral anomaly detection system for a Honeywell hackathon. It learns normal access
behavior per entity, detects compromised credentials and intrusions in near real-time, classifies the
anomaly type, and produces an explainable calibrated risk score surfaced in an analyst dashboard.

Synthetic data only. CPU only. Fully deterministic under seed 42.

**Two source-of-truth documents:**

| File | Role |
|---|---|
| `AI_Powered_Behavioral_Anomaly_Detection_Hackathon_Details.md` | The official brief. 7 deliverables, 8 evaluation criteria. **Authoritative — the plan serves this.** |
| `Hackathon_Implementation_Plan.md` | The phase-by-phase executable spec (Phases 0–10). Already reviewed against the brief and updated. |

---

## 2. Environment

Windows / PowerShell. Chain commands with `;` not `&&`.

```powershell
# Python 3.12.0, venv at repo root with all deps installed
.\.venv\Scripts\python.exe -m pytest                       # 463 tests, ~28s
.\.venv\Scripts\python.exe -m data_generator.generate --seed 42   # ~21s
.\.venv\Scripts\python.exe -m training.build_baselines            # ~15s
```

Key pinned deps: `torch==2.4.1` (CPU), `lightgbm==4.4.0`, `shap==0.45.1`, `scikit-learn==1.5.1`,
`pandas==2.2.2`, `fastapi==0.111.1`, `motor==3.5.1` + `pymongo==4.8.0` (must stay pinned together),
`pydantic==2.8.2`, `matplotlib==3.9.2` (3.9.1 has no wheel — do not "fix" this back).

**PowerShell quirk:** `git push` returns a non-zero exit code because PowerShell wraps Git's stderr
progress output in a `NativeCommandError`. The push succeeds. Verify with
`git rev-parse HEAD origin/main`.

---

## 3. Status: Phases 0, 1, 2 complete. Next is Phase 3.

**463 tests passing, no regressions.** Every phase is gated on tests being written and green before
advancing (plan §3).

### Git

Remote: `https://github.com/Bare009/honeywell-anomaly-detection.git`, branch `main`.

| Commit | Contents |
|---|---|
| `e0d5056` | Feature pipeline, entity baselines, behavioral cohorts (Phase 2) |
| `d2aa832` | Synthetic data generator, attack injectors, taxonomy (Phase 1) |
| `a1ed84a` | Project foundations: config, models, DB clients, health services (Phase 0) |

**Uncommitted right now:** `Hackathon_Implementation_Plan.md`, `README.md`, `models/__init__.py`
(the one-technique-per-tier revision, section 6 below), plus untracked
`AI_Powered_Behavioral_Anomaly_Detection_Hackathon_Details.md` and this file.

**Commit message convention (important):** precise, concise, describing the actual change. **Never
mention "Phase 0/1/2"** or hackathon phasing. Example: `Add feature pipeline with entity baselines
and behavioral cohorts for cold start`.

### Built and committed

```
common/       config.py (Settings, ADP_ prefix), models.py (Pydantic contracts),
              seed.py, database.py, artifacts.py (manifest contract)
data_generator/ profiles.py, normal.py, attacks.py, campaigns.py, drift.py,
              generate.py, TAXONOMY.md
features/     geo.py, encoders.py, sequences.py, entity_window.py,
              event_features.py, session_features.py, cohorts.py, featurize.py
training/     build_baselines.py
api/          main.py, routers/system.py, services/health.py  (health only so far)
serving/      app.py  (health + manifest + bearer-protected /ready only so far)
tests/        test_config, test_seed, test_models, test_artifacts, test_database,
              test_health, test_generator, test_features
```

### Not built yet

`models/` (baseline, sequence, classifier, detectors, risk, drift), `explainability/`,
`evaluation/`, `frontend/`, `training/train_*.py` + `build_artifacts.py`, `docs/FINAL_REPORT.md`,
scoring endpoints in `serving/`, read-API routers beyond `system`.

---

## 4. The dataset (regenerate with `--seed 42`, byte-identical)

`artifacts/dataset/` — git-ignored, ~11.7 MB. **168,968 events, 260 entities, 45 days.**

| Metric | Value |
|---|---|
| Anomalies | 1,502 = **0.89% of events / 0.88% of sessions** (brief requires 0.5–3% of sessions) |
| Splits (by time) | train 99,047 (0.86%), val 31,046 (0.89%), test 38,875 (0.97%) |
| Campaigns | 18, across 3 templates, 3.33 mean stages |
| Benign drift | 36 entities (13.8%) |
| Cold start | 31 entities onboarded at 82% of the timeline |

Per-class event counts: `low_and_slow_exfil` 384, `lateral_movement` 261, `credential_misuse` 193,
`brute_force` 189, `device_spoofing` 138, `insider_drift` 132, `credential_stuffing` 114,
`impossible_travel` 91.

**Labels live in a separate file** (`labels.parquet`), never a column in `events.parquet`. This is
deliberate anti-leakage insurance — feature code physically cannot read a label.

---

## 5. The feature pipeline

**66 features = 59 numeric + 7 categorical.** One shared entry point:
`FeaturePipeline.featurize(event)` — used identically by training and serving. Batch replay calls it
in a loop rather than reimplementing it vectorized, deliberately trading offline speed for zero
train/serve skew.

6 cohorts fitted unsupervised; they rediscovered the generator's latent archetypes without being told
(*"service_accounts active around the clock, including weekends"*, *"users peaking at 11:00, on
weekdays"*, *"edge_devices around the clock"*).

Artifacts written to `artifacts/` (git-ignored except `manifest.json`): `encoders.json`,
`sequence_vocab.json`, `entity_profiles.json` (2.2 MB), `cohorts.json`, `corpus_stats.json`,
`feature_space.json`.

### Measured value over the Phase 1 naive floor

| Metric | Naive floor (raw columns) | With the 66 features | Phase 9 target |
|---|---|---|---|
| PR-AUC | 0.706 | **0.979** | ≥ 0.90 |
| ROC-AUC | 0.982 | 0.999 | ≥ 0.95 |
| Recall @ 1% budget | 0.630 | **0.955** | ≥ 0.80 |

Single LightGBM on the features, no baseline/sequence/fusion tiers yet. Ablation: removing the four
sequence-novelty features drops PR-AUC to 0.958; those features alone score 0.651. So the signal is
spread across the feature set, **not** a leak. See open item 8.2.

---

## 6. Architectural decisions — do not silently reverse these

Each was made for a stated reason. If a decision needs revisiting, do it explicitly.

### 6.1 One technique per deliverable (latest change, uncommitted)

The brief offers a *choice* for deliverables 2 and 3. We build exactly one of each:

- **Deliverable 2 → tabular Autoencoder.** Rejected One-Class SVM (~O(n²) training, unusable at
  ~100k events; one opaque distance with nothing per-feature to explain). Rejected a standalone
  statistical scorer as redundant — it already exists in the feature layer, and univariate z-scores
  cannot express "this *combination* never occurs", which is the zero-day case.
- **Deliverable 3 → GRU next-event model.** Rejected Transformer (54-token vocabulary, 20-step
  sequences — self-attention has nothing to exploit and would overfit at higher cost). Rejected LSTM
  (GRU has 3 gates vs 4: fewer parameters, faster, equal or better on short sequences with limited
  data). Rejected Graph (models adjacency, not order — kept as optional Phase 10).

Consequence: Phase 3 and Phase 4 complexity both drop from L to M. Per-step GRU NLL serves as both
the score and the attribution, so explanation and score cannot disagree.

### 6.2 Anomaly rate is 0.89%, not 2%

`recall@1% budget ≥ 0.80` is **arithmetically impossible** when anomalies exceed ~1.25% of events —
the top 1% cannot physically contain 80% of them. Confirmed empirically: at a 2% rate a naive
baseline scored exactly 0.4012, which is the ceiling of 197/491. The brief makes the top-1% budget an
explicit judging criterion, so this is grounded in the brief, not convenience. **A test fails the
build if `target_anomaly_rate` is raised past the feasible point.**

### 6.3 Split anomaly density is deliberately equalized

The alert-budget threshold is tuned on val and applied to test, so the two must carry comparable
density. Took four corrections: volume-weighted per-(class, split) budgets, spending budget by where
events *land*, debiting spillover from the receiving split, and per-incident split choice instead of
a grid. Before: 1.95% / 1.28% / 3.06%. Now: 0.86% / 0.89% / 0.97%.

### 6.4 No leakage in the offline fit

`build_baselines.py` replays training events in time order and featurizes each against the profile
built from **strictly earlier** events, then folds it in. A batch build (all profiles first, then
featurize) would be much faster and quietly wrong — every event compared to a baseline containing
itself.

### 6.5 Cold start is shrinkage, not a switch

An entity with thin history is scored against a **blend** of its own history and its cohort prior,
weight `n/(n+12)`, exposed as the `profile_confidence` feature. Not a hard fallback: 5 sessions carry
*some* signal. Hierarchy is entity → entity+cohort → cohort → global.

### 6.6 JSON everywhere, never pickle

Encoders, scaler, cohort centroids and profiles all persist as JSON. A pickled sklearn object is tied
to its library version and would break the serving container after any dependency bump.

### 6.7 Profiles persist at full float precision

Rounding to 6 decimals shrank the artifact but broke exact train/serve parity — training used
full-precision profiles while serving loaded rounded ones. The parity test now asserts **exact**
equality.

### 6.8 No hardcoded domain lists in features

Resource sensitivity is *learned* (`CorpusStats`: global frequency + share of entities that touch it).
A hardcoded list of `/vault/` paths would be the generator's knowledge leaking into the detector and
would not transfer to real telemetry.

### 6.9 Anti-leak fixes already applied to the generator

Found by probing difficulty, not by assumption:
- Attacker devices use ordinary cohort OS strings with an unfamiliar MAC. An OS field reading
  `Kali Linux` was detectable from one event with no profiling.
- `credential_misuse` uses a legitimately-used city and late evening (not hostile country + 03:00),
  with only 35% sensitive resources. Was 100% detectable naively.
- `device_spoofing` stays within the cohort's OS/protocol pool. Cross-cohort protocol
  (`entity_type` + `protocol`) gave 100% naive recall.
- Jakarta removed from hostile cities — 905 km from the Singapore office.
- `low_and_slow_exfil` transfers are only ~1.7× median volume with 65% ordinary resources.

### 6.10 Performance: profile before optimizing

The streaming fit took 165 s. Profiling showed `bounding_distance_km` was 65% of runtime (1.35M
haversine calls from rebuilding the live profile per event). Replaced with O(n) centroid-radius spread
plus a profile cache refreshed every 20 events. Now **11.6 s, 0.12 ms/event**. The cache is applied
identically offline and online, and a cached profile is only ever *older* than reality, never newer,
so it cannot leak future information.

### 6.11 `artifacts_ready()` checks the filesystem, not just the manifest

`manifest.json` is tracked in git while the artifact binaries are not, so a fresh clone has a manifest
naming files it does not have. Trusting the manifest alone let the serving readiness gate report
success and then fail on the first request.

---

## 7. Bugs found by tests (illustrates the value of the gate)

| Bug | Detail |
|---|---|
| Window cap off by one | `EntityState.prune` ran before the append, so the deque exceeded `max_events` |
| Sequence reconstruction | Only suppressed *adjacent* duplicates: `[login,view]` + `[login,view,logout]` → `login, view, login, view, logout`. Now merges at longest overlap |
| Profile rounding | Broke exact train/serve parity (6.7) |
| Anomaly rate overshoot | A `(class, split)` grid forced ≥1 full incident per cell — 24 incidents regardless of budget, 67% overshoot at small scale |
| Campaigns past the timeline | Chains could extend beyond the declared end date; now trimmed, and dropped if <2 stages survive |
| CSV env parsing | pydantic-settings JSON-decodes list fields inside the env source before validators run, so `ADP_API_CORS_ORIGINS=http://a,http://b` was a hard startup crash in Docker |

---

## 8. Open items

### 8.1 Confirmed gaps vs the brief (from my review — actions 1–4 are mine, 5 is yours)

1. **`low_and_slow_exfil` does not match the brief.** Brief says *"Gradual after-hours resource
   access over days/weeks"*. Measured: **median span 0.80 days, 34.6% off-hours** (benign reference is
   23.2%). `insider_drift` shows the right shape at 5.27 days / 70.5%. Fix: extend span to 2–10 days,
   concentrate after-hours, and make it detectable from *access pattern* not just `bytes_out`
   (`bytes_out`/`bytes_in` are additions of mine, not in the brief's schema). Requires regenerate +
   refit + `TAXONOMY.md` update. **Not yet applied — awaiting go-ahead, since it revises Phase 1.**
2. **Deliverable packaging missing.** Brief requires *"PDF or ZIP deliverables"*. Phase 9 produces
   `docs/FINAL_REPORT.md` but has no render-to-PDF or bundle step. Add to Phase 9.
3. **Streaming is optional but "preferred".** Brief criterion: *"Scalable system design (real-time
   streaming preferred)"*. `redis_enabled` defaults to `False`. Phase 7 should make streaming a
   *demonstrated* capability with the offline fallback kept as a safety net.
4. **Report the per-session injection rate** alongside per-event; the brief specifies sessions.
5. **Presentation template — needed from the user.** Brief requires *"Presentation using the provided
   template"*. The file is not in the repo. Final slides cannot be built without it.

### 8.2 Undecided: the data may be too easy

Metrics already exceed the Phase 9 targets using only a LightGBM on features (section 5), so Phases
3–5 have little headroom to show the layered architecture earns its place — which is the project's
central claim. Verified this is *not* a leak (ablation in section 5); attack-only command tokens
appear in only 20.6% of attack events, though they are perfectly disjoint from benign traffic.

**My recommendation:** keep the default dataset (the brief rewards accuracy and low false positives)
and additionally report a harder variant (`--subtlety 0.75`) as a robustness ablation in the final
report. Rigour without sacrificing headline numbers, ~1 minute of compute.

### 8.3 Smaller notes for the report

- The brief classes **insider drift as an "Edge Case"**, not an anomaly. We treat it as a full 9th
  label. Defensible, but the report should acknowledge the deviation rather than quietly reclassify.
- **MITRE ATT&CK mapping is not in the brief at all.** It is in our Phase 6. Keep it (reads well to a
  security audience, cheap) but it must not absorb time belonging to the 7 mandated deliverables.
- Per-class metrics rest on very few independent incidents (3 for `credential_stuffing`, 4 for
  `low_and_slow_exfil`). Event counts look comfortable but a regeneration moves a class by ±0.2. Only
  aggregate metrics are stable. Documented in `TAXONOMY.md` §14; must be stated in the report.
- ROC-AUC is near-uninformative at 99.1% negatives (naive floor already 0.982). Lead with PR-AUC and
  recall@budget.

---

## 9. Next: Phase 3 — Baseline Profiling Model (Deliverable #2)

Spec: `Hackathon_Implementation_Plan.md` → Phase 3. Summary:

- **Create** `models/baseline.py` — PyTorch tabular Autoencoder (input → 32 → 16 → 32 → input, ReLU,
  MSE), `score_baseline(features)` → normalized `baseline_score`, `reconstruction_errors(features)` →
  per-feature error for explanations, score normalization fitted on held-out train residuals.
- **Create** `training/train_baseline.py`; **create** `training/build_artifacts.py` here to
  orchestrate artifact building.
- **Acceptance:** clear PR-AUC uplift over random on held-out; cold-start entities get finite
  comparable scores via cohort-blended inputs; per-feature error available; artifacts + manifest
  written.
- **Verify:** `python -m training.train_baseline`; `pytest tests/test_baseline.py`; full suite green
  with no regressions.

Input is ready: `FeaturePipeline.load()` gives 66 features, `categorical_indices`, and
`FeaturePipeline.to_matrix(vectors)`.

---

## 10. Working protocol

1. **Phase completion gate is hard.** A phase is not done until its tests are written, run and green,
   with no earlier-phase regressions. Do not advance otherwise.
2. **Verify, do not assume.** Every metric quoted in this file was measured. Temporary probe scripts
   are fine but delete them afterwards.
3. **Report actual numbers**, including when a target is missed, and explain why.
4. Use dedicated file tools, not shell `cat`/`sed`. Run the venv Python explicitly:
   `.\.venv\Scripts\python.exe`.
5. Commit only when asked. Messages must not mention phases.
