# Behavioral Anomaly Taxonomy & Dataset Design

Documentation for Deliverable #1: what the synthetic dataset contains, what each behavior
means, how it was generated, and where it falls short of reality.

Everything here describes the dataset produced by:

```powershell
python -m data_generator.generate --seed 42
```

All numbers in this document are **measured from that dataset**, not estimated.

---

## 1. Why synthetic data

Real access telemetry containing genuine credential compromise is not publicly available,
and would carry PII if it were. Synthetic generation is not a compromise here — it is what
makes the rest of the project possible:

- **Ground truth is exact.** Every event has a known label, so precision and recall mean
  something. Real-world labels come from analyst triage and are themselves noisy.
- **Rare classes can be guaranteed.** Eight attack behaviors are all present in every split.
  A real capture might contain two of them.
- **Difficulty is tunable.** One `subtlety` parameter controls how far attacks deviate from
  normal, so the dataset can be made harder if the models look too good.
- **Concept drift can be labeled benign.** In real data, distinguishing "the user changed
  jobs" from "the account was stolen" is guesswork. Here it is known.

The cost is that the models learn *our* notion of normal. Section 9 lists what that misses.

---

## 2. Event schema

One row per access/connection event. This matches section 7.1 of the implementation plan.

| Field | Type | Description |
|---|---|---|
| `event_id` | string | Unique event identifier |
| `entity_id` | string | User, service account or device identifier |
| `entity_type` | enum | `user` \| `service_account` \| `edge_device` |
| `timestamp` | datetime (UTC) | When the access occurred |
| `source_ip` | string | Origin address |
| `geo_country`, `geo_city`, `geo_lat`, `geo_lon` | string/float | Geolocation of the origin |
| `resource_accessed` | string | File, endpoint, port or device function |
| `auth_method` | enum | `password` \| `token` \| `certificate` \| `biometric` \| `mfa` |
| `auth_success` | bool | Whether authentication succeeded |
| `session_id` | string | Groups events belonging to one connection |
| `session_duration` | float | Length of the connection, in seconds |
| `command_sequence` | list[string] | Ordered commands/actions |
| `device_os`, `device_mac`, `device_protocol`, `device_user_agent` | string | Device fingerprint |
| `bytes_out`, `bytes_in` | float | Transfer volume |
| `split` | string | `train` \| `val` \| `test`, assigned by time |

`geo` and `device_fingerprint` are flattened into columns for convenience. The round trip
back to the nested `Event` model is lossless and tested
(`dataframe_to_events` in `generate.py`).

### Label space

Nine classes, in this fixed order (`common.models.ANOMALY_CLASSES`):

```
normal, credential_misuse, lateral_movement, brute_force, impossible_travel,
credential_stuffing, device_spoofing, low_and_slow_exfil, insider_drift
```

The order is a contract. Model output columns, confusion-matrix axes and the dashboard
legend all index into it.

### Labels are stored in a separate file

`events.parquet` contains **no** label, `campaign_id` or `stage` column. Ground truth lives
in `labels.parquet`, joined on `event_id`.

This is deliberate. Keeping the label in a separate *file* rather than a separate column
means feature code cannot read it by accident — the cheapest possible insurance against
target leakage, which is the most common way a security ML result turns out to be fiction.

---

## 3. Population and cohorts

260 entities across six **latent cohorts**. Cohorts are groups that behave alike; they are
never given to the models, which rediscover them by clustering in Phase 2. They exist
because they exist in real organizations, and because they are the priors the cold-start
path falls back on when an entity has no history of its own.

| Cohort | Entity type | Schedule | Typical activity |
|---|---|---|---|
| `office_staff` | user | Business hours | Portal, HR, expenses, documents |
| `engineering` | user | Extended hours, weekends | Git, CI, Kubernetes, staging DB, SSH |
| `business_analytics` | user | Business hours | BI dashboards, warehouse queries, exports |
| `batch_services` | service account | Nightly batch windows | ETL extract/load, warehouse writes, backups |
| `integration_services` | service account | 24/7 | API gateways, partner sync, queues |
| `plant_devices` | edge device | 24/7 | Telemetry push, sensor reads, heartbeats, Modbus/MQTT |

Within a cohort, each entity gets its **own** parameters: resource preferences drawn from a
Dirichlet, its own session rate, its own home city, its own devices, its own auth mix, its
own failure rate. Without that per-entity variation, entity-level baselines would be
redundant and the whole premise of behavioral profiling would collapse.

**Geography.** Eight home cities across four countries (India ×5, Singapore, United Kingdom,
United States). Attacks originate from seven cities chosen to be far from every home
location, so geo-velocity and new-country signals carry information.

---

## 4. Benign traffic

167,466 benign events in 39,202 sessions.

A session is a coherent unit: one location, one device, one auth method, a shared duration,
and an ordered command sequence spread across its events. Sessions per day are Poisson
around the entity's own rate; start hours are drawn from a 24-slot activity distribution;
weekends are damped per cohort.

**Benign traffic is deliberately messy.** If normal behavior were clean, every attack would
stand out and the reported metrics would be meaningless. Measured rates in the generated
dataset:

| Behavior | Rate | Why it is there |
|---|---|---|
| Off-hours activity (before 06:00 or after 22:00) | **23.2%** of events | Includes 24/7 services and night batch windows. "Off-hours" alone must never imply an attack. |
| Legitimate off-schedule access for schedule-bound entities | 3.5% of sessions | People occasionally work late. |
| First-time access to a new resource | 3.0% of sessions | Novelty must be suspicious, not conclusive. |
| Legitimate sensitive-resource access | 6.0% of sessions (cohorts that own one) | Keeps sensitive resource names from being near-labels. |
| Failed authentication | **0.34%** of events | Mistyped passwords, 1–2 failures before success. |
| Travel to a second office | 0.8–1.6% of sessions (users) | Makes plausible-travel geo changes normal. |

Command sequences start from a per-cohort template, then get perturbed: a middle step
repeated, dropped, or two adjacent steps swapped. Templates give the sequence model
learnable n-gram structure; the perturbations stop it from memorizing a handful of exact
strings.

---

## 5. Attack behaviors

1,502 anomalous events, **0.89%** of the dataset — inside the mandated 0.5–3% band.

### 5.1 Why 0.89% and not 2%

The headline operating metric is recall inside a **1%-of-events analyst alert budget**. If
anomalies were 2% of events, the top 1% by risk could not physically contain more than half
of them, and the ≥0.80 target would be unreachable regardless of model quality. This was
confirmed empirically before the rate was chosen: at a 2% rate a naive baseline scored
recall@1% = 0.4012, which is exactly the arithmetic ceiling of 197/491.

At 0.89% the budget can hold essentially every anomaly (the test split's ceiling is 1.0), so
recall@budget measures the *ranking* rather than the arithmetic. The rate is also more
imbalanced, which makes the problem harder, not easier.

This is asserted in `tests/test_generator.py`, so raising `target_anomaly_rate` past the point
where the target becomes unreachable fails the build rather than silently producing an
impossible goal.

### 5.2 The eight classes

The plan's prose lists seven injectors; the label space in section 7.1 has eight attack
classes. All eight are implemented, so every class the classifier must predict actually
exists in the data.

Each class has a **distinct primary signal**. Without that, the classifier would be guessing
between overlapping classes and macro-F1 would be unreachable no matter how good the model
is.

| Class | Events | Incidents | Primary signal | Which tier should catch it |
|---|---|---|---|---|
| `brute_force` | 189 | 7 | 7–22 failed auths against one entity inside a ~4–9 min window | Deterministic detector |
| `credential_stuffing` | 114 | 3 | One source IP against 9–26 entities, 1–3 attempts each | Classifier (breadth vs depth) |
| `impossible_travel` | 91 | 20 | Implied geo velocity far above 900 km/h | Deterministic detector |
| `credential_misuse` | 193 | 20 | Valid auth; several behavioral facets deviate at once | Baseline profile + fusion |
| `lateral_movement` | 261 | 17 | Fan-out across 5–13 resources outside the entity's cohort, with recon commands | Sequence model |
| `device_spoofing` | 138 | 23 | Device fingerprint inconsistent with the entity's own history | Baseline profile |
| `low_and_slow_exfil` | 384 | 4 | Sustained mildly-elevated transfers over 6–30 hours | Windowed / sequence features |
| `insider_drift` | 132 | 6 | Gradual self-directed escalation toward sensitive resources | Drift-aware baseline |

Incident counts vary widely per class because incident *sizes* differ: one credential
stuffing spray produces ~34 events across many victims, while one impossible-travel event
pair produces ~4.

### 5.3 Notes on individual classes

**`brute_force`** is sized to stay above the deterministic detector's threshold even at
maximum subtlety. This is not making the task easy: a brute force that does not burst is not
a brute force. The class exists to prove the deterministic tier reaches ~1.0 precision.

**`impossible_travel`** always exceeds the velocity threshold by a wide margin. Subtlety only
widens the time gap, never enough to make the travel physically possible. Same reasoning: the
class is defined by the physics.

**`credential_misuse`** is the class most carefully tuned. Early versions were detectable
from a single event at 100% recall by a naive model, because they combined a hostile country,
deep small-hours access and an all-sensitive resource list. All three were global giveaways
requiring no behavioral profiling at all. Now, at default subtlety, the attacker connects
from a location the entity legitimately uses, in the late evening rather than at 03:00, and
touches sensitive resources in only ~35% of accesses. What remains is a genuine multi-facet
deviation from the entity's own baseline.

**`low_and_slow_exfil`** transfers are only ~1.7× the entity's median volume, and 65% of
accesses are to the entity's ordinary resources. An individual event is meant to be
unremarkable — if a single event were large enough to alert on, this would be a threshold
rule and there would be nothing "low and slow" about it.

**`device_spoofing`** presents a fingerprint that is plausible for the cohort but never used
by this entity: an unseen MAC and an OS build its peers legitimately run. An earlier version
borrowed an OS and protocol from a *different* cohort — a plant sensor reporting Windows over
gRPC — which a naive model caught at 100% recall from `entity_type` plus `protocol` alone, with
no device history involved. The class then proved nothing about behavioral profiling. 75% of
these attacks also come from inside the network, so geolocation is not an alternative shortcut.

**Attacker devices** use ordinary corporate OS strings with an unfamiliar MAC, not obviously
hostile ones. An OS field reading `Kali Linux` would be learnable from one event with no
profiling, which is precisely the shortcut we do not want the models to take.

**`insider_drift` vs benign drift** is the sharpest distinction in the dataset. Both are
gradual behavioral change; the difference is *direction*. Benign drift moves toward a new but
ordinary routine. Insider drift converges on sensitive resources and off-hours access. A
system that flags all change fails here, and so does one that adapts to all change.

---

## 6. Multi-stage campaigns (D1)

**18 campaigns, 616 events, 3.33 stages on average.**

A campaign is several attack stages that share one entity, occur in causal time order, and
carry a shared `campaign_id` with a `stage` index. It is not a new label — each stage keeps
its own anomaly class, so no extra label space is needed.

| Template | Count | Kill chain |
|---|---|---|
| `credential_compromise` | 6 | brute_force → credential_misuse → lateral_movement → low_and_slow_exfil |
| `stolen_session` | 6 | credential_stuffing → impossible_travel → credential_misuse |
| `device_takeover` | 6 | device_spoofing → lateral_movement → low_and_slow_exfil |

Templates are cycled rather than sampled, so every shape is guaranteed to appear — a random
draw could omit one, and the storyline demo needs at least one of each.

Stage gaps range from 15 minutes to 10 hours depending on the template. An entity must be
allowed by *all* of a template's stage injectors, not just the first, or the chain would not
be coherent.

**Campaigns are restricted to the target entity's own events.** Credential stuffing normally
sprays a whole cohort, but inside a campaign only the target's events are kept: a campaign is
defined as one entity's storyline, and tagging a bystander's events with the same
`campaign_id` would create ground truth that reconstruction could never reproduce.

**Campaigns are trimmed to the timeline.** A chain runs for days, so one starting late would
otherwise extend past the declared end date, and the split boundaries and metadata would stop
describing the data. A campaign trimmed down to fewer than two surviving stages is discarded —
that is an incident, not a campaign.

This ground truth is what makes Phase 7's campaign reconstruction *measurable* rather than
merely demonstrable — we can check whether the system relinked the stages the generator
actually related.

---

## 7. Concept drift (D3)

**36 entities (13.8% of the population) drift benignly**: 11 device, 11 resource, 8 schedule,
6 location.

Drift is generated **in the data**, labeled `normal`, rather than simulated later at
evaluation time. A subset of entities gradually changes partway through the timeline:

| Kind | Change | Real-world equivalent |
|---|---|---|
| `schedule` | Activity distribution shifts to a later window | Moved to a later shift |
| `location` | New home city adopted progressively | Relocation, with trips back |
| `device` | New device takes over progressively | Hardware refresh |
| `resource` | 2–3 new resources become routine | New responsibilities |

Drift begins around 55% through the timeline, staggered per entity by −2.5 to +4 days so the
whole population does not change on the same day, and ramps **linearly over 6–12 days**.

The gradualness is the point. An abrupt benign change is indistinguishable from an attack in
a single event; only the sustained, slow character of the change makes adaptation the right
response rather than an alert.

**Cold-start entities are excluded from drift.** An entity that appears near the end of the
timeline has no established baseline to drift away from, and mixing the two would confound
the cold-start and drift experiments.

---

## 8. Cold start

**31 entities (11.9%) are onboarded late**, appearing only after ~82% of the timeline. They
accumulate few sessions before evaluation, so they must be scored against cohort priors
rather than their own history.

Attacks against them are only placed after their onboarding date — an entity cannot be
attacked before it exists.

---

## 9. Splits

Split by **time**, never at random. Training on events that occur after the test events
would let a model exploit information no production system could have, and would produce
metrics we could not honestly report.

| Split | Share of timeline | Events | Anomalies | Rate |
|---|---|---|---|---|
| `train` | first 60% | 99,047 | 850 | 0.86% |
| `val` | next 20% | 31,046 | 276 | 0.89% |
| `test` | final 20% | 38,875 | 376 | 0.97% |

The later splits hold more events than their share of the calendar because cold-start
entities are onboarded late.

**Anomaly density is deliberately equalized across splits.** The alert-budget threshold is
tuned on validation and then applied to test, so if the two carried different anomaly density
the tuned threshold would be calibrated for the wrong world. Achieving this required more
than weighting by elapsed time:

1. Per-class split budgets are weighted by each split's actual **event volume**, not its share
   of the calendar. The later splits hold more events than their share of the timeline because
   cold-start entities are onboarded late.
2. Budgets are spent by where events **actually land**, not where the incident was aimed.
   Long-running incidents (low-and-slow spans ~19 h, insider drift ~6 days) cross boundaries.
3. Spillover is debited from the receiving split's budget too. Without this, each split filled
   its own quota and *then* received spillover on top, pushing the overall rate above target.
4. Splits are chosen per incident, weighted by remaining room, rather than iterated as a
   `(class, split)` grid. The grid forced at least one full incident into every cell — 24
   incidents regardless of budget — which overshot the rate by 67% at small scale.
5. Filling stops once no split has room for half a typical incident, since incidents are chunky
   (a credential-stuffing spray is ~34 events) and one more would overshoot.

Before these corrections, measured densities were 1.95% / 1.28% / 3.06% — validation at less
than half of test. They are now within 0.11 percentage points of each other, and the overall
rate is 0.89% against a 0.80% target.

---

## 10. Difficulty calibration

`subtlety` ∈ [0, 1] is a single dial interpolating **every** injector between blatant and
nearly invisible. Default: **0.55**. If the dataset turns out too easy or too hard, that one
number is tuned rather than eight separate injectors.

Difficulty was verified rather than assumed, by fitting a deliberately naive LightGBM model
on raw per-event columns only — no entity baselines, no sequence features, no time windows.
That is the floor the real pipeline must beat.

| Naive per-event floor (test split) | Value | Phase 9 target |
|---|---|---|
| PR-AUC | 0.706 | ≥ 0.90 |
| ROC-AUC | 0.982 | ≥ 0.95 |
| Recall @ 1% alert budget | 0.630 | ≥ 0.80 |

Per-class recall at the 1% budget for that naive floor:

| Class | Naive floor recall | Reading |
|---|---|---|
| `brute_force` | 1.000 | Correctly easy — an auth-failure burst is visible in one event |
| `credential_stuffing` | 1.000 | Same; the deterministic and classifier tiers should own these |
| `device_spoofing` | 0.722 | Remainder needs per-entity device history |
| `credential_misuse` | 0.718 | Remainder needs the entity's own baseline |
| `impossible_travel` | 0.667 | Partly reachable via country; true velocity needs the previous event |
| `lateral_movement` | 0.492 | Needs sequence and breadth features |
| `low_and_slow_exfil` | 0.448 | Needs windowed aggregation |
| `insider_drift` | 0.372 | Needs drift-aware baselines |

This is the profile we want: the classes that need the layered detectors are precisely the
ones a naive model fails on, and there is real headroom (0.19 PR-AUC, 0.17 recall@budget)
for the pipeline to earn.

**These per-class figures are noisy.** Each class contributes only 3–23 independent incidents,
and the test split sees a subset of those, so a regeneration can move an individual class by
±0.2 while the aggregate figures stay stable. Treat the ordering as meaningful and the exact
values as indicative. Only the aggregate PR-AUC and recall@budget are stable enough to compare
against across runs.

**Note on ROC-AUC.** The naive floor already reaches 0.982 against a ≥0.95 target. ROC-AUC is
close to uninformative at this level of imbalance — with 99.11% negatives, ranking most of them
below most positives is easy. PR-AUC and recall@budget are the metrics that discriminate, and
the final report will lead with those.

---

## 11. Determinism

Everything is reproducible from seed 42. The seed feeds Python `random`, NumPy and PyTorch
through `common.seed.set_global_seed`.

Generation uses **independent child streams** per stage (world, drift, benign, attacks,
campaigns) via `numpy.random.Generator.spawn`. Changing the attack mix therefore does not
reshuffle the benign traffic, which keeps datasets comparable across tuning iterations.

Regenerating with the same seed produces byte-identical parquet output. This is asserted in
`tests/test_generator.py`.

---

## 12. Output files

Written to `artifacts/dataset/`:

| File | Contents |
|---|---|
| `events.parquet` | 168,968 feature-bearing events. **No labels.** |
| `labels.parquet` | `event_id`, `label`, `is_anomaly`, `campaign_id`, `stage`, `split` |
| `entities.json` | Per-entity ground truth: cohort, home, devices, resource weights, cold-start flag, drift plan |
| `campaigns.json` | Campaign ground truth: stages, kill chain, timings, template |
| `metadata.json` | Config, split boundaries, per-class counts, drift/cold-start/campaign summaries |

Generation takes ~21 seconds and the dataset is ~30 MB. Binaries are git-ignored; the
dataset is rebuilt from source rather than committed.

---

## 13. Assumptions

1. **One organization, four countries.** Cross-country access is plausible but notable. A
   genuinely global company would weaken every geo signal.
2. **Sessions are well-formed.** Every event belongs to a session with a known duration. Real
   telemetry has orphaned and truncated sessions.
3. **Device fingerprints are stable.** MAC and OS change only through drift or spoofing. Real
   fingerprints churn from VPNs, browser updates and NAT.
4. **No adversarial adaptation.** Attackers do not learn from being detected or deliberately
   mimic a specific victim's baseline.
5. **Attacks are independent of benign traffic.** An injected incident does not suppress the
   entity's normal activity, so an entity can appear to be in two places at once outside the
   impossible-travel class.
6. **Commands are a small closed vocabulary.** Roughly 40 benign tokens plus 12 hostile ones.
   Real command telemetry has a long tail of thousands.
7. **Volume is uniform in the small.** No holidays, quarter-end spikes, incident response
   surges or maintenance windows.
8. **Geolocation is accurate.** Only city-level jitter is applied. Real IP geolocation is
   frequently wrong, especially through VPNs and mobile carriers.

---

## 14. Limitations

- **The models learn our notion of normal.** Metrics on this dataset are an upper bound on
  real-world performance. The generator and the detector share assumptions, which is the
  fundamental limitation of any synthetic evaluation.
- **Incident counts are small for some classes.** `low_and_slow_exfil` has 384 events across
  only 4 incidents; `credential_stuffing` 114 across 3. Per-class metrics on the test split
  rest on a handful of independent incidents, so confidence intervals are wide even where
  event counts look comfortable. This is the single most important caveat when reading any
  per-class number from this dataset.
- **Campaign granularity.** 18 campaigns and 60 stages means campaign-reconstruction accuracy
  can only be measured to roughly ±2 percentage points.
- **Only three campaign templates.** Real intrusions vary far more.
- **Drift is linear.** Real behavioral change is bursty and often reverts.
- **No label noise.** Ground truth is perfect, which is never true in practice. A real
  deployment would need to tolerate mislabeled analyst verdicts.
- **Class boundaries are cleaner than reality.** A real intrusion is often several of these
  behaviors at once, and analysts would disagree about which label applies.

---

## 15. Tuning the dataset

```powershell
# Harder: attacks closer to normal
python -m data_generator.generate --seed 42 --subtlety 0.8

# Easier, for debugging a model end to end
python -m data_generator.generate --seed 42 --subtlety 0.2

# Different scale
python -m data_generator.generate --seed 42 --entities 400 --days 60

# Different imbalance (stays within the mandated 0.5-3% band)
python -m data_generator.generate --seed 42 --anomaly-rate 0.015
```

The generator **exits non-zero** if the resulting anomaly rate falls outside 0.5–3% or if any
attack class is missing. A silently invalid dataset would invalidate every metric downstream,
so it refuses to pretend it succeeded.
