# AI-Powered Behavioral Anomaly Detection

Detect compromised credentials and intrusions from behavioral telemetry — logins, geolocation,
resource access, command sequences and device fingerprints — in near real time, with a
**calibrated, explainable risk score** for every event.

The system learns what *normal* looks like per entity (user, service account, edge device), then
flags deviations regardless of whether it has seen the attack before. It is trained and evaluated
entirely on **synthetic** data, is **CPU-only**, and is **deterministic** under a single seed
(42) so every number reproduces.

---

## 1. What it does, briefly

Three complementary detectors fuse into one risk score, because no single model catches everything:

| Tier | Model | Catches |
|---|---|---|
| **1 — Baseline** | Tabular **autoencoder** over entity-relative features | Zero-day deviation, cold-start entities (via cohort priors) |
| **2 — Sequence** | **GRU** next-command language model | Unusual *order/breadth* of actions (lateral movement, recon) |
| **3 — Classifier** | Calibrated **LightGBM** multi-class | Names the anomaly *type* |

Two **deterministic detectors** (impossible-travel geometry, brute-force bursts) act as
high-precision overrides. A **risk-fusion** layer blends the tiers into a calibrated 0–100 score
with an uncertainty band and an analyst **alert budget** (top ~1% of events). Every alert ships
with **SHAP** feature attributions, a **counterfactual** ("this would have scored benign if…"),
**sequence-step** attribution, a **MITRE ATT&CK** mapping, and a plain-language narrative. Related
alerts for an entity are stitched into **attack campaigns** (kill chains), and analyst **feedback**
tunes the entity's threshold. A live **React dashboard** surfaces all of it.

### How it flows

```
OFFLINE  (train once)                     ONLINE  (score every event)
─────────────────────                     ───────────────────────────

data_generator/   labeled dataset         event
        │                                   │
        ▼                                   ▼
features/         encoders, entity        featurize()   66 features
                  baselines, cohorts      the same function, both planes
        │                                   │
        ▼                                   ▼
training/         autoencoder → GRU →     Tier 1   Autoencoder    unsupervised
                  classifier → fusion     Tier 2   GRU            sequence
                  → calibration           Tier 3   Classifier     anomaly type
        │                                 Detectors (deterministic rules)
        ▼                                   │
artifacts/  ───── loaded read-only ─────►   ▼
  version-stamped manifest                risk fusion → 0–100 risk + uncertainty
        │                                   │
        ▼                                   ▼
evaluation/       metrics + report        explainability + campaign linking
                                            │
                                            ▼
                                          MongoDB → api/ → frontend/ dashboard
```

The **same `featurize()`** runs offline and online, so an online score equals the offline score
for the same event (no train/serve skew). The serving plane only *loads* trained state from
`artifacts/` and refuses a schema mismatch — it never retrains.

### Deliverables & challenges — where each lives

| Brief requirement | Implementation | Tests |
|---|---|---|
| **D1** Synthetic data generator + taxonomy | `data_generator/`, `data_generator/TAXONOMY.md` | `tests/test_generator.py` |
| **D2** Baseline profiling model | `models/baseline.py` (autoencoder) | `tests/test_baseline.py` |
| **D3** Sequence-aware model | `models/sequence.py` (GRU) | `tests/test_sequence.py` |
| **D4** Anomaly-type classifier | `models/classifier.py` (LightGBM) | `tests/test_classifier.py` |
| **D5** Explainability layer | `explainability/` (SHAP, counterfactual, MITRE, narrative) | `tests/test_explainability.py`, `tests/test_counterfactual.py` |
| **D6** Analyst dashboard | `frontend/` (React + Vite + TS) | — |
| **D7** Final report | `DETAILED_REPORT.md`, `docs/`, `evaluation/` | `tests/test_evaluation.py` |
| Extreme class imbalance | PR-AUC + recall@1% budget everywhere; never raw accuracy | `tests/test_risk.py` |
| Cold start | Cohort-prior blending in the feature layer | `evaluation/coldstart_experiment.py` |
| Concept drift | PSI monitor with adaptive re-profiling | `models/drift.py`, `tests/test_drift.py` |
| Low false-positive rate | Calibrated fusion + tuned alert budget | `models/risk.py` |
| Scalable / streaming | FastAPI scorer + optional Redis Streams consumer | `tests/test_serving.py` |
| Attack-story reconstruction | Per-entity campaign / kill-chain linking | `serving/campaign.py`, `tests/test_campaign.py` |

Headline results on the held-out **test** split (full numbers, assumptions and honest limitations
in [`DETAILED_REPORT.md`](DETAILED_REPORT.md)): PR-AUC 0.975, recall @ 1% budget 0.965, macro-F1 0.943,
calibration ECE 0.0004, brute-force detector precision 1.000.

---

## 2. Setting up on your machine

### 2.1 Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| **Python** | 3.11 or 3.12 | data generation, training, evaluation, tests |
| **Docker Desktop** (with Compose v2) | recent | running the live stack (MongoDB, scorer, API, dashboard) |
| **Git** | any | cloning |
| Node.js + npm | 20+ | *optional* — only to run/build the dashboard outside Docker |

CPU-only — **no GPU required**. Everything runs offline; no API keys are needed.

> **Windows note:** run the commands in **PowerShell**. If virtual-env activation is blocked, run
> once: `Set-ExecutionPolicy -Scope Process RemoteSigned`. Chain commands with `;` (not `&&`).

### 2.2 Clone and enter the project

```bash
git clone <repository-url>
cd honeywell-anomaly-detection
```

### 2.3 Create the Python environment and install dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Confirm the venv is really active before installing -- this must print a path
# ending in \honeywell-anomaly-detection\.venv
python -c "import sys; print(sys.prefix)"

python -m pip install --only-binary :all: -r requirements.txt
```

**macOS / Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate

# Confirm the venv is really active -- must print a path ending in /.venv
python -c "import sys; print(sys.prefix)"

python -m pip install --only-binary :all: -r requirements.txt
```

Two details in that command matter:

- **`python -m pip`, not bare `pip`.** A bare `pip` can resolve to a different Python's pip that
  happens to sit earlier on `PATH`, which installs the dependencies somewhere the virtual
  environment cannot see. `python -m pip` always installs into the interpreter you are running.
- **`--only-binary :all:`** forces pre-built wheels. Every pinned version has one for Python
  3.11/3.12. Without it, pip may try to compile a package from source and fail if a C toolchain
  (for example a MinGW `gcc`) is on your `PATH`.

> If pip resolves a CUDA build of PyTorch, install the CPU wheel explicitly first:
> `python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu`

<details>
<summary><b>Troubleshooting: <code>ModuleNotFoundError: No module named 'numpy'</code>, or pip says "Defaulting to user installation"</b></summary>

Both symptoms have the same cause: the install went somewhere other than the virtual environment.
The giveaway is pip reporting a path outside your project, such as
`Requirement already satisfied: pip in c:\program files\python312\lib\site-packages`, together with
`Defaulting to user installation because normal site-packages is not writeable`.

Fix it by calling the environment's interpreter **by full path**, which cannot be shadowed by `PATH`:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install --only-binary :all: -r requirements.txt
.\.venv\Scripts\python.exe -m data_generator.generate --seed 42
```

```bash
# macOS / Linux
./.venv/bin/python -m pip install --only-binary :all: -r requirements.txt
./.venv/bin/python -m data_generator.generate --seed 42
```

Using the full interpreter path for every command in this guide works regardless of whether
activation succeeded, and is the most reliable option if you hit any of this.

Other things worth checking:

- `Activate.ps1` blocked by execution policy: run `Set-ExecutionPolicy -Scope Process RemoteSigned`
  once, then activate again.
- Do **not** run `pip install --upgrade pip`. If bare `pip` is the system one, pip refuses to
  upgrade itself and errors out. It is not needed here.
- If the environment looks broken, delete and recreate it: `Remove-Item -Recurse -Force .venv`
  then repeat the steps above.
- Cloud-synced folders (OneDrive, Dropbox) occasionally interfere with virtual-env creation. If
  problems persist, clone to a local path such as `C:\dev\honeywell-anomaly-detection`.

</details>

### 2.4 The dataset and trained models are already in the repo

`artifacts/` is committed — the labeled dataset (`artifacts/dataset/`) and every trained model
(autoencoder, GRU, classifier, fusion, cohorts, encoders) are already on disk after cloning.
**There is nothing to generate or train to run the demo.** Skip straight to section 2.5.

Sanity-check that everything is present:
```bash
python -m pytest -q
```

> **Why commit trained artifacts at all?** The usual advice is to keep binaries out of git. We
> break that rule deliberately here so an evaluator can run the full stack immediately with no
> data generation or training step. Every committed artifact is still fully reproducible: the
> dataset regenerates byte-identically under seed 42 in ~20 seconds, and retraining takes a few
> minutes on CPU. See **section 2.9** if you want to regenerate them yourself, e.g. to confirm
> the determinism claim or to try a different `--subtlety`.

### 2.5 Start the live stack (Docker)

Make sure **Docker Desktop is running**, then:
```bash
docker compose up -d --build
```
This starts MongoDB, Redis, the scoring service, the read API, and the dashboard (the dashboard
image is built inside Docker, so Node is not required). First build pulls images and compiles the
UI — give it a few minutes. Check status with `docker compose ps`.

### 2.6 Populate the dashboard with scored detections

With the stack up (MongoDB reachable), replay the test split through the scorer into MongoDB:
```bash
python -m serving.replay --split test --mongo --fresh
```
`--fresh` clears previous runs so counts don't double. For a quick demo instead of the full split,
add `--limit 5000`.

### 2.7 (Recommended) Run the evaluation and generate the report

With the stack up (so metrics land in MongoDB for the dashboard's *Model Performance* page):
```bash
python -m evaluation.evaluate               # test-set metrics → artifacts/metrics.json + MongoDB
python -m evaluation.coldstart_experiment   # cohort-prior uplift
python -m evaluation.campaign_experiment    # kill-chain reconstruction accuracy
python -m evaluation.drift_experiment       # PSI adaptation curves
python -m evaluation.report                 # writes a local REPORT.md of the measured numbers
```

### 2.8 Open the dashboard

Visit **http://localhost:3000**.

You should see ranked alerts, the explanation drawer (SHAP, counterfactual, sequence highlight,
MITRE, narrative, feedback), entity explorer, attack storylines, model performance, drift monitor,
and system health.

### 2.9 (Optional) Regenerate the dataset and retrain from scratch

Not required to run the demo — the committed `artifacts/` already contains everything. Do this
only if you want to verify the determinism claim yourself, or experiment with different generator
settings (e.g. `--subtlety`, `--entities`).

```bash
python -m data_generator.generate --seed 42   # ~20s, overwrites artifacts/dataset/
python -m training.build_artifacts            # ~5-8 min CPU, overwrites the model artifacts
```

The dataset write is byte-identical every time under the same seed; the trained models are
deterministic under seed 42 but will differ from the committed copies if you change any generator
or training parameter. Re-run `python -m evaluation.evaluate` (section 2.7) afterwards to refresh
`artifacts/metrics.json` against whatever you just trained, and `git diff --stat artifacts/` to see
exactly what changed before committing.

---

## 3. Services and ports

| Service | URL | Purpose |
|---|---|---|
| **Dashboard** | http://localhost:3000 | Analyst UI |
| **Read API** | http://localhost:8000/api/v1/health | Powers the dashboard (read-only) |
| **Scoring API** | http://localhost:8100/health | Scores events (bearer-token protected) |
| **MongoDB** | localhost:27017 | Detections, profiles, campaigns, feedback, metrics |

The scoring service requires a bearer token on its write endpoints
(`ADP_SCORING_AUTH_TOKEN`, default `dev-scoring-token`). Change it before exposing the service on
any network.

---

## 4. Running the dashboard without Docker (optional)

If you prefer a live dev server (hot reload) instead of the Docker UI, and have Node 20+:

```bash
cd frontend
npm install
npm run build     # type-check + production build, or:
npm run dev       # dev server on http://localhost:5173 (proxies /api to :8000)
```
The read API must be running (either via Docker, or `python -m uvicorn api.main:app --port 8000`).

---

## 5. Configuration

Every setting has a working default, so **no configuration is required**. To override, copy the
example env file and edit it:

**Windows:** `Copy-Item .env.example .env`  **macOS/Linux:** `cp .env.example .env`

All variables use the `ADP_` prefix and are documented inline in `.env.example`; the authoritative
definitions live in [`common/config.py`](common/config.py). Notable knobs: `ADP_ALERT_BUDGET_PCT`
(alert budget), `ADP_RISK_ALERT_THRESHOLD`, the fusion weights, cold-start and drift parameters,
and the optional LLM narrator (`ADP_LLM_ENABLED`, off by default — a deterministic template is used
instead).

---

## 6. Verifying and testing

```bash
python -m pytest                       # full suite (unit + integration), CPU-only
python -m pytest -m "not integration"  # skip tests that need trained artifacts
docker compose config                  # validate the compose file
```
647 tests, fast and deterministic; integration tests auto-skip if artifacts aren't built yet.
`python -m pytest` rather than bare `pytest` for the same reason as `python -m pip` — it guarantees
the test run uses this environment's interpreter and packages.

---

## 7. Project layout

```
common/          config, Pydantic contracts, seed utility, DB clients, artifacts manifest
data_generator/  synthetic telemetry: profiles, benign traffic, attacks, campaigns, drift
features/        one shared featurize(), encoders, entity baselines, cohorts
models/          baseline, sequence, classifier, deterministic detectors, risk fusion, drift
explainability/  SHAP, counterfactuals, sequence attribution, MITRE map, narrative
training/        offline training scripts → artifacts/
evaluation/      imbalance-aware metrics, cold-start/drift/campaign experiments, report
serving/         FastAPI scorer, campaign linking, feedback loop, stream consumer, replay
api/             FastAPI read API for the dashboard
frontend/        React 18 + Vite + TypeScript analyst dashboard
artifacts/       trained models + dataset (committed) -- see section 2
docs/            report PDF, dashboard screenshots, 100-event sample of the dataset
tests/           unit and integration tests
```

Submission artifacts live in `docs/`:

| Path | Contents |
|---|---|
| [`DETAILED_REPORT.md`](DETAILED_REPORT.md) | The final report: architecture, metrics, assumptions, limitations |
| `docs/report/report_honeywell.pdf` | The same report as a PDF |
| `docs/frontend/` | Dashboard screenshots (overview, alerts, entity explorer, storyline, model performance) |
| [`docs/sample_data/`](docs/sample_data/) | 100-event excerpt of the generated dataset, with labels in a separate file as in the real thing |

---

## 8. Notes for evaluators

- **Determinism:** one seed (42) drives `random`, NumPy, PyTorch and LightGBM
  ([`common/seed.py`](common/seed.py)). Trained state is version-stamped in
  `artifacts/manifest.json` with the git SHA that produced it.
- **Honesty:** [`DETAILED_REPORT.md`](DETAILED_REPORT.md) states every headline metric against its target
  *and* the known limitations (why ROC-AUC is not the headline at ~1% prevalence, thin per-class
  incident counts, the zero cold-start uplift, `device_spoofing` precision, synthetic-data caveats).
  Metrics are computed on the held-out **test** split and regenerate via section 2.7.
- **Data:** synthetic only, no real PII. Assumptions, per-class signals, injection rates, campaign
  structure and drift design are documented in `data_generator/TAXONOMY.md`.
- **Fastest path to a running demo:** sections 2.3 → 2.8 — no data generation or training needed,
  the dataset and every model are already committed. If Docker is unavailable, the read-only
  evaluation and test suite (2.3, 2.4, 2.7, 6) still run fully without it.
