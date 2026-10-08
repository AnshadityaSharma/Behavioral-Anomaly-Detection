# Behavioral Anomaly Detection

**ML-powered behavioral analytics for detecting anomalous access activity.**

## Live demo

Streamlit Community Cloud: **[Add deployment URL here]**

Open **Detect Anomalies** and click **Load Demo Dataset · Run Detection**. The bundled demo contains 8,000 deterministically sampled sessions from the project's synthetic access-log pipeline, with saved predictions and explanations. It opens without a training step.

## What it does

The app helps an analyst move from a large access log to a ranked alert queue, investigate a session, and inspect the model evidence. It covers synthetic users, service accounts, and edge devices. Injected patterns include brute force, credential stuffing, impossible travel, lateral movement, device spoofing, insider drift, and low-and-slow exfiltration. These are synthetic research patterns, not claims of real-world detection performance.

The public demo includes:

- An overview of events, model-predicted anomalies, users/IPs, and activity over time.
- A ranked, filterable alert queue and event investigation with recent entity history.
- The existing deterministic explanation and SHAP attribution for high-risk sessions.
- Model insights using saved held-out metrics from the full synthetic dataset.
- Optional CSV upload for live inference using the existing trained ensemble.
- Analyst confirm/dismiss feedback that reorders the local queue without retraining.

## Architecture

```text
Synthetic access log or uploaded CSV
             ↓
Feature engineering: timing, geo, auth, devices, resources, rolling baselines
             ↓
Isolation Forest + Random Forest + GRU sequence model
             ↓
Weighted risk score → cold-start blending → ranked alerts
             ↓
Random Forest SHAP attribution → analyst investigation and feedback
```

The original pipeline lives in [`src/pipeline.py`](src/pipeline.py). It trains the three models on a time-based split, scores sessions, and saves full results under `data/processed/`. The Streamlit entrypoint is [`src/dashboard/app.py`](src/dashboard/app.py). Upload inference and validation live in [`src/dashboard/detection.py`](src/dashboard/detection.py). The app loads the existing [`models/trained_bundle.joblib`](models/trained_bundle.joblib) only when a CSV is scored, so the saved demo starts quickly. The bundle is around 44 MB; PyTorch is the largest deployment dependency.

## Models and evidence

| Component | Contribution |
| --- | --- |
| Isolation Forest | Scores unusual behavior within each entity type. |
| Random Forest | Classifies attack types from engineered tabular features. |
| GRU | Adds recent session sequence context. |
| SHAP | Attributes the Random Forest's predicted type to contributing features for high-risk sessions. |

Risk weights are 25% baseline, 35% tabular, and 40% sequence before cold-start blending. The app shows predicted anomaly types, not verified incidents. Risk bands are percentile bands **within the current dataset**; they are not calibrated probabilities. The Model Insights metrics come from the saved time-split evaluation of the full synthetic data and are not calculated from the 8,000-row demo excerpt or an upload. See [`reports/report.md`](reports/report.md) for the full evaluation and limitations.

## Run locally

Use Python 3.12 (the version used for local validation).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

For tests, install `requirements-dev.txt` and run `python -m pytest`. To regenerate the full synthetic dataset, trained bundle, scores, and metrics, run `python -m src.pipeline`. This is a research/training command and is **not** part of the app's startup. `python -m src.generator.generate_dataset --seed 42` generates raw data separately.

To try upload inference, download the sample access log from the app. The CSV must use the generator's 15 access-log columns (`src/generator/generate_dataset.py`) and contain at most 10,000 rows. Inference builds behavioral history from the rows in that upload. Include enough history per entity for useful rolling and sequence features. There is no ground-truth evaluation for uploaded data.

## Deploy on Streamlit Community Cloud

1. Push this repository, including `data/demo/` and `models/trained_bundle.joblib`, to GitHub.
2. Create a Streamlit app from the repository. Set the main file path to `src/dashboard/app.py` and choose Python 3.12.
3. Let Cloud install `requirements.txt`, then open the app and run the bundled demo.
4. Replace the Live demo placeholder above with the app URL.

No secrets or external services are needed. Paths resolve from the repository root. The trained bundle is loaded lazily and cached for uploads. Community Cloud's local filesystem is ephemeral: analyst feedback can disappear when the app restarts or moves to another instance. Deploying PyTorch can take longer than a typical small Streamlit app.

## Limitations and next steps

The data is synthetic, so real-world false positive rates are unknown. Scores and type predictions can disagree; some types have substantially weaker precision than others. SHAP here explains the Random Forest class prediction, not the whole weighted ensemble. Uploads with incomplete entity history may produce different scores than full-history scoring. Future work could test on privacy-safe real logs, calibrate alert thresholds, and persist analyst feedback in an approved store.
