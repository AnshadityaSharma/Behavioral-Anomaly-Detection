# Behavioral Anomaly Detection

AI/ML system for detecting and classifying anomalous access behavior (credential
misuse, lateral movement, brute force, impossible travel, device spoofing) from
entity access logs, with an explainable risk score for each alert.

Since real access-log/intrusion data is scarce and privacy-restricted, this
project generates its own synthetic dataset with injected attack patterns and
ground-truth labels, then trains detection and classification models on top of
it.

## What's inside

- `src/generator/` - synthetic access-log generator: per-entity behavioral
  profiles (users, service accounts, edge devices) plus six injected attack
  patterns (including a low-and-slow exfiltration pattern that builds up
  gradually in off-hours over days to weeks) plus an ambiguous "insider
  drift" edge case - seven injected labels in total.
- `src/features/` - turns raw session rows into per-session features
  (geo-velocity, resource novelty, device fingerprint mismatch, rolling
  failure/breadth windows) with a drift-tolerant rolling baseline instead of a
  fixed all-time one.
- `src/models/` - three complementary models:
  - `baseline.py` - per-entity-type isolation forest plus a per-entity
    statistical profile
  - `classifier.py` - random forest over tabular features (multi-class
    anomaly type, SHAP-friendly)
  - `sequence_model.py` - small GRU over each entity's recent session window
  - `cold_start.py` - blends a new entity's score toward its population
    baseline until it has enough history of its own
- `src/explain/` - SHAP attribution composed into a one-sentence explanation
  per alert, with the raw values kept alongside it.
- `src/pipeline.py` - runs the whole thing end to end and writes scored
  sessions + metrics to `data/processed/`.
- `src/demo/coldstart_drift.py` - reproduces the cold-start and drift evidence
  in the report as two charts.
- `src/dashboard/app.py` - Streamlit analyst view: ranked alert queue, alert
  detail with explanation and entity history, confirm/dismiss triage, and a
  system-behaviour tab.
- `tests/` - pytest coverage for the generator, features, models, cold-start
  blending, and the feedback loop.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Running it

Generate the synthetic dataset and run the full pipeline (this also generates
the data on first run if `data/raw/` is empty):

```bash
python -m src.pipeline
```

This writes `data/raw/access_log.csv`, `data/raw/labels.csv`,
`data/raw/entities.csv`, and `data/processed/scored_sessions.csv` +
`data/processed/metrics.json`.

To regenerate the dataset on its own (e.g. with a different seed):

```bash
python -m src.generator.generate_dataset --seed 7
```

Reproduce the cold-start and concept-drift charts used in the report (needs the
pipeline to have run once, since it reuses the trained models):

```bash
python -m src.demo.coldstart_drift
```

Launch the dashboard once `data/processed/` exists:

```bash
streamlit run src/dashboard/app.py
```

Run tests:

```bash
pytest
```

## Approach

Full assumptions, metrics, and known limitations are in
[reports/report.md](reports/report.md).

Short version: three models score every session (an isolation forest baseline,
a random forest over tabular features, and a GRU over each entity's recent
session window), blended into one risk score. New entities get their score
pulled toward the population baseline until they build up history - without
that, a device's first session scores 0.83 purely for being unfamiliar. Two
z-scored features (session duration, login hour) use a trailing 30-day window
per entity rather than all-time history, so a permanent behavior shift stops
being flagged after it's been the norm for a while. A separate 7-day resource-
breadth window catches slow, gradual resource accumulation that a 24-hour
window would miss - the signature of the low-and-slow exfiltration pattern.

Each alert carries a generated one-line explanation ("Flagged as brute force
due to 47 failed logins for this entity within 10 minutes, combined with a 100%
authentication failure rate from this source IP"), with the SHAP values behind
it available in the dashboard. Analysts can confirm or dismiss alerts;
dismissals push that entity down the queue by a capped, documented offset.

The report is explicit about where this falls short - resource-footprint drift
is not forgiven, the feedback loop trusts the analyst blindly, and precision
degrades sharply past a top-2% alert budget.
