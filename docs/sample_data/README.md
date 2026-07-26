# Sample generated data

A 100-event excerpt of the synthetic dataset, included so the data can be inspected without running
the generator. The full dataset is 168,968 events and is not committed — it rebuilds byte-identically
in about 21 seconds:

```powershell
python -m data_generator.generate --seed 42
```

## Files

| File | Contents |
|---|---|
| `sample_events.csv` | 100 events, 22 columns — **exactly the schema the models consume** |
| `sample_labels.csv` | Ground truth for those events, joined on `event_id` |

**Labels are in a separate file, as they are in the real dataset.** `sample_events.csv` contains no
`label`, `campaign_id` or `stage` column. Keeping ground truth in a separate *file* rather than a
separate column means feature code cannot read a label by accident — the cheapest insurance against
target leakage. Join on `event_id` to inspect both together.

## What is in the excerpt

Curated, not random. A uniform sample of a 0.89%-anomaly dataset would contain roughly one attack and
show nothing useful. This excerpt takes **contiguous runs**, so session grouping and command-sequence
progression stay intact.

| Class | Events |
|---|---|
| `normal` | 67 |
| `brute_force` | 5 |
| `credential_misuse` | 5 |
| `insider_drift` | 5 |
| `lateral_movement` | 5 |
| `low_and_slow_exfil` | 5 |
| `device_spoofing` | 3 |
| `impossible_travel` | 3 |
| `credential_stuffing` | 2 |

Also spread across all three time-based splits (train 58, val 23, test 19) and all three entity types
(user 66, service account 20, edge device 14).

The class balance here is **deliberately not representative** — attacks are over-sampled roughly 40×
so every behaviour is visible. Do not compute metrics from this file; the real prevalence is 0.89%.

## Reading it

`command_sequence` is a list in the real dataset (Parquet); it is flattened to `a > b > c` here so the
CSV opens cleanly in a spreadsheet.

Worth looking at, to see how the attacks differ from the baseline rather than from a global rule:

- **`brute_force`** — a run of `auth_success = False` against one `entity_id` from one `source_ip`
  inside a few minutes.
- **`impossible_travel`** — two events for the same entity whose `geo_lat`/`geo_lon` and `timestamp`
  imply a speed no aircraft could manage.
- **`device_spoofing`** — a `device_mac` this entity has never used, on an OS its peers legitimately
  run. There is no globally suspicious value; it is only detectable against that entity's history.
- **`low_and_slow_exfil`** — `bytes_out` only modestly above the entity's normal. No single event is
  alarming; the sustained repetition is the signal.

Full documentation of assumptions, per-class signals, injection rates and limitations is in
[`data_generator/TAXONOMY.md`](../../data_generator/TAXONOMY.md).
