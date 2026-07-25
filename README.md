# AI-Powered Behavioral Anomaly Detection

A layered behavioral anomaly engine for detecting compromised credentials and intrusions in
near real-time. An autoencoder baseline over entity-relative features, a GRU sequence model
over command patterns, and a supervised anomaly-type classifier fuse into a **calibrated,
explainable risk score** per event — mapped to MITRE ATT&CK, reconstructed into campaigns, and
surfaced in a live analyst dashboard.

The system is trained and evaluated entirely on synthetic behavioral telemetry. It is
CPU-only, deterministic under a single seed, and runs offline end to end.

## Why it is built this way

| Decision | Reason |
|---|---|
| Three fused detector tiers | Each tier targets a different failure mode: the unsupervised autoencoder catches cold-start and zero-day deviation, the GRU catches order-aware behavior, the classifier names the attack type. Fusion beats any single model. |
| One technique per tier, not several | The brief allows a choice for the baseline and sequence models. One-Class SVM was rejected on scalability (~O(n²)) and on having nothing per-feature to explain; a Transformer was rejected because 54 tokens over 20 steps gives self-attention nothing to exploit. Picking the best fit beats stacking alternatives that each need tuning and justification. |
| PR-AUC and recall@budget, never raw accuracy | Attacks are 0.5–3% of events. A model that predicts "normal" always would score 97%+ accuracy and catch nothing. |
| Explainable by construction | SHAP attributions, counterfactual "nearest-normal" reasons, sequence-step attribution and MITRE mapping are produced inside the scoring pipeline, not bolted on afterwards. |
| One `featurize()` for train and serve | Train/serve skew is the most common silent failure in ML systems. There is exactly one feature function, and it is used by both planes. |
| LLM as narrator, not decider | The optional language model only phrases an explanation. It never influences a score, and it always falls back to a deterministic template. |
| Versioned `artifacts/` contract | The serving plane loads trained state only from `artifacts/` and refuses a schema mismatch at startup. It never retrains. |

## Architecture

```
OFFLINE (train)                          ONLINE (serve)
data_generator/  labeled dataset         events
      |                                    |
features/        encoders, baselines     serving/  featurize -> baseline + sequence
      |                                    |         -> classifier + detectors
training/        models + fusion           |         -> risk fusion (calibrated 0-100)
      |          + calibration             |         -> explainability
      v                                    |         -> campaign linking
artifacts/  <-- loaded read-only ----------+         -> persist
      |                                    v
evaluation/      metrics + report        MongoDB -> api/ -> frontend/ (dashboard)
```

## Quickstart

Windows PowerShell (chain with `;`, not `&&`):

```powershell
# 1. Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate the synthetic dataset (deterministic under seed 42)
python -m data_generator.generate --seed 42

# 3. Train every model and write artifacts/
python -m training.build_artifacts

# 4. Start MongoDB, the scoring service, the read API and the dashboard
docker compose up -d

# 5. Replay the dataset through the scorer to populate the dashboard
python -m serving.replay --seed 42
```

macOS/Linux is identical apart from `source .venv/bin/activate` and `&&` chaining.

Steps 2–5 depend on later build phases; step 1 and the test suite work today.

## Verify the install

```powershell
pytest
docker compose config
```

## Services

| Service | URL | Purpose |
|---|---|---|
| Dashboard | http://localhost:8080 | Analyst UI: ranked alerts, explanations, storylines |
| Read API | http://localhost:8000/api/v1/health | Powers the dashboard (read-only) |
| Scoring API | http://localhost:8100/health | Scores events (bearer-token protected) |
| MongoDB | localhost:27017 | Detections, profiles, campaigns, feedback, metrics |

The scoring service requires a bearer token on its write endpoints (`ADP_SCORING_AUTH_TOKEN`).
Change it from the development default before exposing the service on any network.

## Configuration

Every setting has a working default, so no configuration is required. To override, copy
`.env.example` to `.env` and edit. All variables use the `ADP_` prefix and are documented
inline in that file; the authoritative definitions live in
[`common/config.py`](common/config.py).

## Project layout

```
common/           config, Pydantic contracts, seed utility, DB clients, artifacts manifest
data_generator/   synthetic telemetry: profiles, benign traffic, attacks, campaigns, drift
features/         one shared featurize(), encoders, entity baselines, cohorts
models/           baseline, sequence, classifier, deterministic detectors, risk fusion, drift
explainability/   SHAP, counterfactuals, sequence attribution, MITRE map, narrative
training/         offline training scripts -> artifacts/
evaluation/        imbalance-aware metrics, cold-start/drift/campaign experiments, figures
serving/          FastAPI scorer, campaign linking, feedback loop, stream consumer, replay
api/              FastAPI read API for the dashboard
frontend/         React 18 + Vite + TypeScript analyst dashboard
artifacts/        trained state (git-ignored) + tracked manifest.json
tests/            unit and integration tests, run after every phase
docs/             final report
```

## Determinism

One seed (42) drives Python `random`, NumPy, PyTorch and LightGBM via
[`common/seed.py`](common/seed.py). The dataset, the trained models, the metrics and the
demo are all reproducible. Trained state is version-stamped in `artifacts/manifest.json`
alongside the git SHA that produced it.

## Data

Synthetic only — no real PII. The generator's assumptions, per-class signals, injection
rates, campaign structure and drift design are documented in
`data_generator/TAXONOMY.md`.

## Tests

```powershell
pytest                       # everything
pytest -m "not slow"         # skip training smoke tests
pytest tests/test_models.py  # one module
```

Tests are fast, CPU-only, and run after each build phase — no phase is complete until its
tests pass and no earlier phase has regressed.
