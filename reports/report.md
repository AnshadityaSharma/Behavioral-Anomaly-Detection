# Behavioral Anomaly Detection - Report

## Problem

Model "normal" access behavior per entity (user, service account, edge
device), detect intrusions or compromised-credential activity from access
logs, classify the anomaly type, and produce a risk score an analyst can
actually act on - all under extreme class imbalance and with no real intrusion
data to train on.

## Synthetic data

Real access logs for this kind of problem are scarce and privacy-restricted,
so `src/generator/` builds a synthetic one instead: 250 users, 40 service
accounts, and 80 edge devices, each with its own habitual login hours, home
location, typical resource set, and (for privileged sessions) command
sequence, sampled with noise over a 45-day window. A handful of entities join
partway through the window to exercise the cold-start path.

Six attack patterns are injected on top of the normal traffic, plus one
ambiguous edge case:

| Pattern | How it's simulated |
|---|---|
| Brute force | 12-60 rapid failed logins from one attacker IP against one entity |
| Impossible travel | A session placed a few minutes to a few hours after a real one, from a location that implies an unreasonable travel speed |
| Credential stuffing | Many entities (20-70), 1-3 attacker IPs, high failure rate, all within a short window |
| Lateral movement | A burst of sessions touching resources the entity has never accessed, often with privileged commands |
| Device spoofing | Same entity_id, mismatched OS/MAC fingerprint |
| Insider drift | A legitimate entity gradually expanding its resource footprint over 1-3 weeks - not treated as a hard positive, used to check the false-positive rate on slow, ambiguous behavior change |

Injection rates land in the 0.3-0.6% range per attack type (about 3% combined),
matching the brief's suggested 0.5-3% band. Ground truth (`label`) is kept in
a separate `labels.csv`, joined back in only for training and evaluation - the
access log itself carries no label column, matching how this would actually
show up at inference time.

Known simplifications: geo coordinates are jittered around a home point rather
than drawn from a real road/flight network, so "impossible travel" distances
are geometrically but not always geographically realistic. Command sequences
are drawn from a fixed action vocabulary rather than modeling real
shell/API semantics.

## Features

`src/features/build_features.py` turns each raw session into ~20 features:
time-of-day/day-of-week, hours since the entity's last session, geo distance
and implied velocity from the previous session and from home, whether the
resource or device fingerprint has ever been seen before for this entity,
rolling failure counts and distinct-entity counts per source IP (for brute
force / credential stuffing), and rolling resource breadth per entity (for
lateral movement).

Two of these - session duration and login hour - are z-scored against a
**trailing 30-day window per entity** rather than all-time history
(`src/features/drift.py`). That's the concept-drift handling: if a user's
schedule shifts and stays shifted for a few weeks, the model stops treating
the new schedule as anomalous instead of flagging it forever.

## Models

Three models score every session, combined into one risk score:

- **Baseline profiler** (`models/baseline.py`) - an isolation forest per
  entity_type, plus a per-entity statistical profile used for the dashboard's
  entity history view.
- **Tabular classifier** (`models/classifier.py`) - a random forest over the
  same feature set, multi-class over `normal` + the six attack types,
  `class_weight="balanced"` to deal with the imbalance. Chosen specifically
  because it's SHAP-friendly (`TreeExplainer`), so it's also the model behind
  the explainability layer.
- **Sequence model** (`models/sequence_model.py`) - a single-layer GRU over
  each entity's last 10 sessions, also multi-class, trained with inverse-
  frequency class weights. This is the "sequence-aware" piece: it can catch
  patterns that only show up across a run of sessions (e.g. lateral movement)
  rather than in any single row's features.

Final risk score = `0.25 * isolation_forest + 0.35 * random_forest_anomaly_prob
+ 0.40 * gru_anomaly_prob`, then blended toward the entity_type's average risk
for cold-start entities (`models/cold_start.py`): an entity with fewer than 5
prior sessions has its score pulled toward the population baseline in
proportion to how little history it has, since neither its own profile nor
the GRU's context window means much yet.

Predicted anomaly type is the argmax of the average of the random forest's and
GRU's class probabilities.

## Explainability

`src/explain/attribution.py` runs SHAP's `TreeExplainer` on the random forest
for the top 5% of sessions by risk score (computing it for the full,
overwhelmingly-normal log would be expensive and pointless - nobody needs a
reason for a session nobody's going to look at). The top 3 SHAP features per
alert are mapped through a small template dictionary into short, readable
reasons (e.g. "implausible travel speed (2400 km/h since last session); first-
ever access to this resource") instead of raw SHAP values. Cold-start alerts
get an extra note about limited history.

## Evaluation

Split by time, not randomly: the first 70% of the simulation window is train,
the last 30% is test, so the sequence model and rolling features are never
evaluated on data that leaked into their own history.

Results on the held-out (last 30%) window, seed 42, 113,245 total sessions
(35,724 in the test window, 911 of them anomalous):

| Metric | Value |
|---|---|
| PR-AUC, binary anomaly vs. normal (risk score) | 0.909 |
| Precision @ top 1% alert budget | 1.000 |
| False positive rate @ top 1% alert budget | 0.000 |
| Overall multi-class accuracy | 0.895 |

Per-class precision/recall (multi-class, combined RF + GRU prediction):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brute_force | 0.965 | 1.000 | 0.982 | 165 |
| credential_stuffing | 0.986 | 0.973 | 0.979 | 221 |
| impossible_travel | 0.973 | 1.000 | 0.986 | 142 |
| device_spoofing | 0.713 | 0.800 | 0.754 | 115 |
| lateral_movement | 0.356 | 0.948 | 0.518 | 77 |
| insider_drift | 0.045 | 0.874 | 0.085 | 191 |
| normal | 0.999 | 0.893 | 0.943 | 34,813 |

The headline numbers are strong precisely because the attack patterns with the
sharpest behavioral signal (rapid failed auths, geo-velocity, many-entities-
one-IP) are exactly what the rolling/geo features were built to catch, and the
top-1%-by-risk-score alert queue is dominated by those. The per-class table
tells the more honest story:

- **Brute force / credential stuffing / impossible travel** are essentially
  solved by the feature set - the signal is close to definitional.
- **Device spoofing and lateral movement** are harder - both require the
  model to recognize "never seen before" patterns from a single or a few
  sessions, and precision suffers (0.71 and 0.36) even though recall stays
  high, meaning the models over-flag some normal sessions as these types.
- **Insider drift** is the class the brief calls out as ambiguous, and the
  numbers show it: 0.045 precision means the model tags a lot of ordinary
  resource-footprint growth as drift. That's the expected failure mode for an
  edge case defined by *not* being clearly anomalous, and is why it's kept out
  of the binary anomaly target used for the PR-AUC number above - it's useful
  for false-positive-rate tuning, not as a hard detection target.

## Known limitations

- Single synthetic seed evaluated here; real deployments would need to check
  sensitivity to different attack-rate assumptions.
- SHAP attribution runs against the random forest, not the GRU - the two
  models can disagree on the predicted type, and the reason shown is only
  strictly faithful to the random forest's decision.
- The GRU uses a fixed-length, zero-padded window (10 sessions) rather than a
  packed/masked sequence, so very new entities are working with a mostly-zero
  input until they build up history.
- Insider drift is generated and labeled but deliberately not folded into the
  "is_anomaly" binary target for the PR-AUC number above, since the brief
  frames it as an ambiguous edge case for false-positive tuning, not a hard
  detection target.
- Isolation forest contamination and GRU class weights are fixed constants,
  not tuned per deployment; a real system would want these calibrated against
  an analyst's actual alert budget.

## Scalability notes

Feature computation is dominated by per-entity and per-source-IP rolling
windows, which is naturally incremental - a streaming implementation would
maintain per-entity/per-IP state (last session, running counts) instead of
recomputing over the full log, which is how this would need to work for
near-real-time scoring anyway. Inference for all three models is cheap per
session (tree lookups and one small GRU forward pass), so the scoring path
itself is not the bottleneck - the state store for rolling per-entity/per-IP
windows is.
