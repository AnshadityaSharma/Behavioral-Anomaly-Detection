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
the GRU's context window means much yet. Measured effect of that blend is in
the cold-start section below.

Predicted anomaly type is the argmax of the average of the random forest's and
GRU's class probabilities.

## Explainability

`src/explain/attribution.py` runs SHAP's `TreeExplainer` on the random forest
for the top 5% of sessions by risk score (computing it for the full,
overwhelmingly-normal log would be expensive and pointless - nobody needs a
reason for a session nobody's going to look at). The top 3 SHAP features are
then composed into a single sentence by `src/explain/narrative.py`, which is
what the analyst sees first:

> Flagged as brute force due to 47 failed logins for this entity within 10
> minutes, combined with a 100% authentication failure rate from this source IP.

> Flagged as credential stuffing due to a 100% authentication failure rate from
> this source IP, combined with a failed authentication on this session and a
> device fingerprint that does not match this entity's prior sessions (now
> reporting Windows11).

This is deterministic templating, not a generation call. Two reasons: the same
alert has to read the same way every time an analyst opens it, and it runs for
every alert in the queue rather than once per demo. The raw SHAP values stay
available behind an expander in the dashboard, so the sentence can be checked
rather than taken on faith.

One thing that needed handling: SHAP will attribute positively to a feature
whose *value* is unremarkable, because trees split on low values too. Early
versions produced text like "a 0% authentication failure rate" and "an implied
travel speed of 1 km/h" as grounds for an alert, which is worse than saying
nothing. Each phrase now only renders above a value worth reading (geo-velocity
over 100 km/h, IP failure rate over 20%, |z| over 1.5, and so on). These are
display thresholds only - they do not affect the score, and if every phrase is
suppressed the alert says so and points to the SHAP breakdown.

## Evaluation

Split by time, not randomly: the first 70% of the simulation window is train,
the last 30% is test, so the sequence model and rolling features are never
evaluated on data that leaked into their own history.

Results on the held-out (last 30%) window, seed 42, 113,245 total sessions
(35,724 in the test window, 911 of them anomalous):

| Metric | Value |
|---|---|
| PR-AUC, binary anomaly vs. normal (risk score) | 0.915 |
| Precision @ top 1% alert budget | 1.000 |
| False positive rate @ top 1% alert budget | 0.000 |
| Overall multi-class accuracy | 0.852 |

Per-class precision/recall (multi-class, combined RF + GRU prediction):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| impossible_travel | 0.973 | 1.000 | 0.986 | 142 |
| credential_stuffing | 0.981 | 0.959 | 0.970 | 221 |
| brute_force | 0.922 | 1.000 | 0.959 | 165 |
| device_spoofing | 0.696 | 0.817 | 0.752 | 115 |
| lateral_movement | 0.500 | 0.909 | 0.645 | 77 |
| insider_drift | 0.034 | 0.937 | 0.065 | 191 |
| normal | 0.999 | 0.849 | 0.918 | 34,813 |

### What the alert budget actually buys

Precision at the top 1% is the number that matters for an analyst, and it holds
up - but it degrades fast if the budget is widened, which is worth stating
plainly:

| Alert budget | Sessions surfaced | Genuine attacks | insider_drift | normal | Precision |
|---|---|---|---|---|---|
| top 1% | 1,132 | 1,132 | 0 | 0 | 1.000 |
| top 2% | 2,264 | 2,195 | 69 | 0 | 0.970 |
| top 5% | 5,662 | 2,614 | 523 | 2,525 | 0.462 |
| top 10% | 11,324 | 2,645 | 661 | 8,018 | 0.234 |

There are only ~2,650 genuine attack sessions in the whole log, so past roughly
the top 2% the queue runs out of real attacks and starts filling with normal
traffic. The system is well-calibrated for a tight budget and should not be
sold as usable at a loose one.

The headline numbers are strong precisely because the attack patterns with the
sharpest behavioral signal (rapid failed auths, geo-velocity, many-entities-
one-IP) are exactly what the rolling/geo features were built to catch, and the
top-1%-by-risk-score alert queue is dominated by those. The per-class table
tells the more honest story:

- **Brute force / credential stuffing / impossible travel** are essentially
  solved by the feature set - the signal is close to definitional.
- **Device spoofing and lateral movement** are harder - both require the
  model to recognize "never seen before" patterns from a single or a few
  sessions, and precision suffers (0.70 and 0.50) even though recall stays
  high, meaning the models over-flag some normal sessions as these types.
- **Insider drift** is the class the brief calls out as ambiguous, and the
  numbers show it: 0.034 precision means the model tags a lot of ordinary
  resource-footprint growth as drift. That's the expected failure mode for an
  edge case defined by *not* being clearly anomalous, and is why it's kept out
  of the binary anomaly target used for the PR-AUC number above - it's useful
  for false-positive-rate tuning, not as a hard detection target.

## Cold start and concept drift

Both are named evaluation criteria, so rather than assert they work,
`python -m src.demo.coldstart_drift` reproduces the evidence below and writes
the two charts in `reports/figures/`. The same charts and numbers appear in the
dashboard's "System behaviour" tab.

### Cold start

![cold start](figures/cold_start_progression.png)

Twenty brand-new edge devices, none of them present when the models were
trained, scored at increasing history depth:

| Session | Reported risk | Raw model risk | sd across devices | Weight on own history |
|---|---|---|---|---|
| 1 | 0.193 | 0.693 | 0.000 | 0.00 |
| 5 | 0.275 | 0.295 | 0.076 | 0.80 |
| 20 | 0.248 | 0.248 | 0.088 | 1.00 |
| 50 | 0.224 | 0.224 | 0.055 | 1.00 |

The number that matters is the first row. A device on its first session scores
**0.693** from the models alone, because it is novel on every feature that
exists - unseen resource, unseen fingerprint, no prior session to compare
against. That is a false positive waiting to happen. Blending against the
population baseline for that entity type reports **0.193** instead, comfortably
below the 0.90 alert threshold.

The mechanism is a linear ramp in `models/cold_start.py`: an entity's score is
`w * own_score + (1 - w) * population_prior`, where `w = min(1, history / 5)`.
So an entity stops being "cold" after **5 sessions**, and the prior is fixed at
training time so that scoring a new entity does not depend on whatever else is
in the batch.

Two honest caveats:

- **The band does not narrow the way you might expect.** Spread across devices
  is 0.000 at session 1 - not because the system is confident, but because
  every new device is given the same prior. Real disagreement between devices
  only appears once their own history takes over (sd 0.088 at session 20),
  settling to 0.055 by session 50. The first-session score is the *least*
  informative one, despite looking the most certain.
- **Five sessions is a count, not a duration.** An edge device chatting every
  few minutes clears cold start in under an hour; a quiet service account that
  authenticates weekly stays in cold start for over a month, and a
  genuinely compromised new account gets its score suppressed for its first
  five sessions. The threshold should really be a function of expected session
  rate per entity type, which it currently is not.

### Concept drift

![drift](figures/drift_vs_attack.png)

**What works (left panel).** Session duration and login hour are z-scored
against a trailing 30-day per-entity window (`features/drift.py`). On a series
where behaviour changes permanently on day 15, the trailing baseline spikes to
|z| **50.7** when the change happens - correctly flagging it - then settles to
|z| **0.58** once the new behaviour is simply the norm. A baseline profiled once
and never refreshed sits at |z| **47.8** indefinitely, i.e. it would flag that
entity forever. That is the failure mode the rolling window exists to avoid.

**What does not work (right panel).** The rolling window only covers timing and
duration. Resource-footprint growth is scored by `is_new_resource`, which is
computed against all-time first-seen, and there is no mechanism that ever
forgives a permanently expanded resource set. Measured across the 27
insider-drift entities with enough history to compare:

- median risk rose from **0.477** before the shift to **0.687** during it
- it rose for **27 of 27** entities - there is no favourable case to point at
- the best case still rose, 0.728 to 0.814

So the honest position is: drift handling is real for *when* and *how long* an
entity works, and absent for *what it touches*.

The one thing that keeps this from being a practical problem at the current
operating point is that drifting entities do not clear the alert bar: **0 of
719** insider-drift sessions land in the top 1%, against a cutoff of 0.897.
They raise the score without raising an alert. That margin is thin, though -
at a top-2% budget, 69 of them appear, and the fix (decaying resource novelty,
or scoring novelty against a trailing window like the timing features) is not
implemented.

## Analyst feedback loop

The dashboard records confirm/dismiss decisions per alert in a local SQLite
file and turns them into a per-entity score offset: **-0.02 per dismissal,
+0.02 per confirmation, capped at +/-0.10**. `effective_risk = risk_score +
offset`, clipped to [0, 1], and the queue ranks on that.

This is deliberately not learning, and should not be described as such. It does
not retrain anything, does not touch stored risk scores, does not generalise
from one entity to another, and forgets nothing over time. It is a manual
re-ranking knob with an audit trail.

Measured on the current run: dismissing five alerts from the entity with the
largest queue footprint moved its best rank from **29 to 1,129** and cut its
presence inside the top-1% budget from **54 alerts to 1**.

Its limitations are worth being blunt about:

- **It trusts the analyst completely.** In the run above, the entity whose
  alerts were dismissed had 55 genuine brute-force sessions. Five wrong clicks
  suppressed real attacks. The +/-0.10 cap bounds the damage but does not
  prevent it, and nothing detects that a dismissal contradicted the label.
- **It starts empty.** With zero decisions the offset is zero for every entity,
  so the loop contributes nothing on day one and only becomes useful after an
  analyst has worked the queue for a while.
- **It is per-entity, so it does not transfer.** Dismissing a noisy scanner
  teaches the system nothing about the next identical scanner.
- **Decisions never expire.** An entity dismissed five times in January is
  still suppressed in June, which is its own small concept-drift problem.

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
- The narrative layer's display thresholds (when a feature is "worth
  mentioning") are hand-set constants. They stop nonsense phrasing but were
  chosen by inspection, not derived from the data.
- Everything above is one seed. The cold-start and drift figures come from a
  single run; the direction of each result is clear-cut, but the exact numbers
  would move on a reseed.

## Scalability notes

Feature computation is dominated by per-entity and per-source-IP rolling
windows, which is naturally incremental - a streaming implementation would
maintain per-entity/per-IP state (last session, running counts) instead of
recomputing over the full log, which is how this would need to work for
near-real-time scoring anyway. Inference for all three models is cheap per
session (tree lookups and one small GRU forward pass), so the scoring path
itself is not the bottleneck - the state store for rolling per-entity/per-IP
windows is.
