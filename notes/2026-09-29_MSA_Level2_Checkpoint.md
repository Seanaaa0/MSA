# MSA Research Checkpoint — 2026-09-29

## 0. Purpose

This note records the current state of **MSA (Model Shock Absorber)** at the current Level-2 checkpoint: what is implemented, what the experiments currently support, the present limitations, and the next steps before Level 3.

This is a research checkpoint, not a final thesis claim.

---

# 1. Core research question

MSA does not primarily ask:

> Is the model / belief perfectly correct?

It asks:

> Is the current imperfect model or belief **adequate for the current decision**?

Core distinction:

\[
\|\delta\| \not\Rightarrow \text{Decision Adequacy}
\]

The same raw error or uncertainty can be harmless in one decision context and harmful in another.

Current intended architecture:

```text
Reality / Sensor
      ↓
 Observation
      ↓
 Belief / Estimator
      ↓
 Planner
      ↓
 Candidate Action
      ↓
 Adequacy Monitor
      ↓
 Trust / QueryReality
      ↓
 Executor
```

The Adequacy Monitor is therefore not merely an uncertainty detector. Its role is to determine whether currently available information is sufficient to support the current decision.

---

# 2. Research levels

## Level 0 — deterministic baseline

Purpose:

- establish a simple deterministic GridWorld,
- verify environment and planner behavior,
- create an exact ground-truth setting.

Status: **completed**.

## Level 1 — controlled belief/state error

Purpose:

- inject controlled localization / belief perturbation,
- compare belief-conditioned actions against ground-truth consequence,
- establish that raw error magnitude and action divergence are insufficient adequacy criteria.

Main findings:

- the same error magnitude can produce both adequate and inadequate decisions,
- action divergence from the oracle does not necessarily imply worse task consequence,
- inadequacy is not identical to collision.

Status: **completed as principle evidence**.

## Level 2 — partial world knowledge + selective Reality Query

Purpose:

- move from artificial coordinate perturbation to genuine partial knowledge,
- build an occupancy belief from local observations,
- allow active information acquisition through `QueryReality`,
- determine when uncertainty actually deserves intervention.

Status: **core pipeline implemented and working**.

Current Level-2 focus remains symbolic and deterministic.

## Level 3 — learned / latent representation

Planned transition:

```text
pixels / sensors
    ↓
encoder
    ↓
latent state z_t
    ↓
world model / learned representation
    ↓
latent uncertainty / support
    ↓
adequacy monitor
```

Status: **not started**.

---

# 3. Current repository structure

Important files:

```text
src/
├── env.py
├── maps.py
├── planner.py
├── belief.py
├── perturbation.py
├── evaluator.py
├── observation.py
├── adequacy.py
├── query.py
├── level2_episode.py
└── episode.py

experiments/
├── run_sweep.py
└── run_level2.py
```

Important separation:

## Agent side

The agent may use:

- local observation,
- occupancy belief,
- candidate action,
- current partial map,
- monitor-internal uncertainty features.

The agent must **not** use:

- true state,
- full true map,
- oracle action,
- actual regret,
- counterfactual future benefit.

## Experiment side

The evaluator may use ground truth to create labels and metrics.

This separation is central to the research design.

---

# 4. Level-1 / V2 regression checkpoint

The old controlled-perturbation experiment still works after the Level-2 additions.

Current regression result:

```text
Total cases:                  10086
Valid beliefs:                4067
Invalid beliefs:              6019
Adequate valid decisions:     1991
Inadequate valid decisions:   2076
Non-collision inadequacies:   445
Collisions:                   1582
False terminals:              49
Mean regret:                  0.615
Maximum regret:               2
```

Interpretation:

- Level-2 changes did not break the original V2 experiment.
- `DecisionEvaluator.evaluate()` remains available for the old belief-position experiment.
- `DecisionEvaluator.evaluate_action()` is available for externally generated Level-2 actions.

---

# 5. Level-2 world-knowledge uncertainty

Current Level-2 uncertainty is intentionally restricted to:

> **unknown map geometry**

The agent currently knows:

- its own current position,
- the goal,
- map dimensions.

The agent does not initially know:

- obstacle layout outside observed regions.

The current experiment deliberately does **not** yet combine:

- localization uncertainty,
- image/classification uncertainty,
- semantic uncertainty,
- dynamic obstacles,
- learned world models,
- sensor noise.

This isolates uncertainty about the world map itself.

---

# 6. Observation and QueryReality

Current sensing setup:

```text
normal observation radius = 1
QueryReality radius        = 4
```

Normal observation is passive and cheap.

`QueryReality` reveals a larger local region and incurs a bookkeeping query cost.

Current nominal value:

```text
query_cost = 1.0
```

This value does **not** yet have physical meaning.

---

# 7. Level-2 baseline policies

Three policies are compared:

```text
always_trust
always_query
monitor
```

- **Always Trust:** never actively acquires extra information.
- **Always Query:** queries at every decision point.
- **Monitor:** uses the MSA Adequacy Monitor to decide whether to trust or query.

---

# 8. Adequacy Monitor progression

## V0 — local action stability

Question:

> If one nearby unknown cell were actually blocked, would the immediate candidate action change?

Main limitation:

\[
\text{Local Action Stability}
\neq
\text{Decision Adequacy}
\]

A route can depend strongly on unknown geometry while the first action stays unchanged.

## V1 — path uncertainty

V1 added:

- immediate action stability,
- whole-path unknown ratio,
- near-term unknown ratio.

Result:

```text
Total monitor steps   = 77
Total monitor queries = 40

TP = 3
FP = 37
FN = 0
```

V1 found useful query opportunities but queried too aggressively.

Lesson:

\[
\text{Uncertainty}
\neq
\text{Need for intervention}
\]

## V2 — temporal persistence / passive absorption

Current monitor version.

Main idea:

> Do not query immediately just because uncertainty exists.

Instead:

1. detect high path uncertainty,
2. continue and receive passive observations,
3. see whether uncertainty resolves naturally,
4. query only when uncertainty persists.

Conceptual behavior:

```text
high uncertainty once
        ↓
continue + passive sensing
        ↓
uncertainty resolves
        ↓
TRUST
```

or:

```text
high uncertainty persists
        ↓
QUERY REALITY
```

This is currently the closest implementation to the “shock absorber” intuition: absorb disturbances that naturally disappear and intervene when they persist.

Current frozen parameters:

```text
persistent_path_threshold = 0.78
persistence_steps         = 2
action_path_gate          = 0.50
sensitivity_radius        = 4
stability_threshold       = 1.0
```

These parameters have not yet been systematically swept.

---

# 9. Counterfactual query-value evaluator

The experiment side now evaluates the value of querying at each decision point.

At a given state:

```text
              current belief
                    │
          ┌─────────┴─────────┐
          │                   │
      TRUST NOW           QUERY NOW
          │                   │
   no active query       one QueryReality
          │                   │
   passive sensing       passive sensing
          │                   │
      rollout               rollout
```

The agent never sees this counterfactual result.

The evaluator asks:

> Would querying **now** improve future task consequence?

Current comparison priority:

1. task success,
2. collisions,
3. episode steps.

This creates labels:

```text
TP = queried and querying was beneficial
FP = queried but querying was not beneficial
TN = trusted and querying had no benefit
FN = trusted although querying would have helped
```

Important distinction:

\[
\text{Immediate Action Adequacy}
\neq
\text{Horizon / Plan Adequacy}
\]

A one-step action may be locally adequate even though additional information would prevent later route inefficiency.

---

# 10. Current Level-2 V2 results

Maps:

```text
simple
branch
detour
dead_end
corridor
```

Aggregate result:

```text
policy         success  steps  queries  benefit-opps  TP  FP  FN  TN
------------------------------------------------------------------------------
always_trust         5     85        0             7   0   0   7  78
always_query         5     77       77             3   3  74   0   0
monitor              5     77       14             3   3  11   0  63
```

MSA Monitor query-value analysis:

```text
True positives:          3
False positives:         11
False negatives:         0
True negatives:          63

Query precision:         0.214
Query recall:            1.000

Monitor total steps:     77
Monitor total queries:   14

Potential positive step gain encountered: 8
```

---

# 11. Main current result

On the current five deterministic maps:

```text
Always Query:
77 steps
77 queries

MSA Monitor V2:
77 steps
14 queries
```

Therefore, in this setting, MSA matched the aggregate task-step performance of Always Query while using approximately:

\[
1 - \frac{14}{77} \approx 81.8\%
\]

fewer active queries.

This is **preliminary Level-2 evidence**, not a general result.

---

# 12. Corridor case

The corridor map currently gives the clearest example.

```text
Always Trust:
26 steps
0 queries

Always Query:
18 steps
18 queries

MSA Monitor:
18 steps
5 queries
```

Monitor labels on corridor:

```text
TP = 3
FP = 2
FN = 0
```

Interpretation:

- passive-only behavior takes an 8-step longer route,
- querying every step avoids that loss,
- MSA reaches the same 18-step result with only 5 active queries.

This is currently the strongest single Level-2 example.

---

# 13. V1 → V2 improvement

V1:

```text
steps   = 77
queries = 40
TP      = 3
FP      = 37
FN      = 0
```

V2:

```text
steps   = 77
queries = 14
TP      = 3
FP      = 11
FN      = 0
```

So V2 reduced queries:

```text
40 → 14
```

which is a **65% reduction relative to V1**, while preserving:

- the same aggregate task steps,
- all three useful query events,
- zero false negatives on the current trajectories.

---

# 14. Current interpretation

The current experiments support the limited claim:

> In the current deterministic partial-observation GridWorld setting, a monitor that distinguishes persistent decision-relevant uncertainty from transient uncertainty can preserve the performance benefit of frequent information acquisition while substantially reducing Reality Queries.

The current evidence also supports the design intuition:

\[
\text{Raw uncertainty}
\neq
\text{intervention necessity}
\]

and suggests that persistent decision-relevant uncertainty is a more useful trigger than raw unknown-cell quantity.

---

# 15. What is NOT yet demonstrated

The current results do **not** establish:

- generalization to unseen map distributions,
- robustness to planner changes,
- robustness to heuristic/tie-breaking changes,
- learned adequacy prediction,
- localization uncertainty,
- vision or semantic uncertainty,
- noisy sensors,
- dynamic worlds,
- continuous control,
- robot manipulation,
- physical robot effectiveness,
- general Physical-AI performance.

Current wording should remain:

> “demonstrated in this setting”

rather than:

> “proven generally.”

---

# 16. Important current limitations

## Small fixed map set

Only five hand-designed maps are currently used.

Risk:

> the monitor parameters may accidentally fit these maps.

Therefore **Adequacy V2 should now be frozen** before further testing.

## Planner dependence

Current planning uses:

```text
A*
+ Manhattan heuristic
+ fixed neighbor ordering
```

Candidate actions can therefore depend on heuristic and tie-breaking details.

## Ground-truth oracle shares planner machinery

The evaluator currently uses exact-map A* for ground-truth consequence.

Before learned planners are introduced, it would be cleaner to separate the experiment oracle using an exact BFS/Dijkstra-style solver.

## Query cost is not physically calibrated

Current:

```text
query_cost = 1.0
```

A physical query may actually cost:

- latency,
- energy,
- stopping time,
- camera reacquisition,
- relocalization,
- active viewpoint change,
- extra computation.

---

# 17. Query-cost tradeoff

A future objective can be:

\[
J = \text{TaskCost} + \lambda \cdot \text{QueryCost}
\]

Current aggregate values:

```text
Always Trust:
Task steps = 85
Queries    = 0

MSA:
Task steps = 77
Queries    = 14
```

Using the simplified cost:

\[
J = Steps + \lambda Q
\]

MSA is better than Always Trust when:

\[
77 + 14\lambda < 85
\]

so:

\[
\lambda < 0.571
\]

This is only a bookkeeping sensitivity result, not yet a physical conclusion.

---

# 18. Current decision: freeze Adequacy V2

Do **not** continue tuning:

```text
0.78 → 0.81
2 persistence steps → 3
...
```

using the same five maps.

That would risk overfitting the monitor to the current test cases.

Current V2 is now the checkpoint baseline.

---

# 19. Next steps before Level 3

## Step 1 — unseen / generated maps

Highest priority.

Generate maps that were not used while designing V2.

Goal:

> Test whether query reduction and low-FN behavior survive on unseen environments.

Rules:

- keep V2 parameters frozen,
- do not tune using the unseen test set before recording the first result.

## Step 2 — planner robustness

Test the frozen monitor under multiple planning configurations.

Suggested:

```text
Planner A:
A* + Manhattan

Planner B:
A* with heuristic = 0
(Dijkstra/BFS-like in unit-cost grid)

Planner C:
A* + alternate neighbor ordering / tie-breaking
```

Main question:

> Does the MSA result survive changes in candidate-action generation?

## Step 3 — independent exact oracle

Separate:

```text
Agent Planner
```

from:

```text
Experiment Ground-Truth Oracle
```

Prefer a simple exact BFS/Dijkstra solver for shortest-path ground truth.

## Step 4 — query-cost sweep

Evaluate multiple values of:

\[
\lambda
\]

in:

\[
TaskCost + \lambda QueryCost
\]

Goal:

> characterize when selective querying is worthwhile.

## Step 5 — decide if Level 2 is sufficient

After unseen-map testing, planner robustness, oracle separation, and cost analysis, decide whether Level 2 needs one more iteration or can be frozen before Level 3.

---

# 20. Planned Level-3 transition

Level 3 should change the **representation**, not the core research question.

Level 2:

```text
symbolic occupancy belief
+
explicit unknown cells
```

Level 3:

```text
learned latent representation
+
latent / learned uncertainty
```

Core question remains:

> Is the current representation adequate for the current action or plan?

Possible Level-3 architecture:

```text
image / sensor
      ↓
encoder
      ↓
z_t
      ↓
world model / latent predictor
      ↓
candidate action
      ↓
learned or hybrid adequacy monitor
      ↓
Trust / Query / Reobserve
```

The Level-2 symbolic system then becomes:

- a conceptual baseline,
- an interpretable baseline,
- an ablation reference.

---

# 21. Current research narrative

```text
Level 1:
Raw model error is insufficient.
Same error magnitude can have different decision consequences.

↓

Level 2 V0:
Immediate action stability is insufficient.
A stable first action may still rely on unsupported future assumptions.

↓

Level 2 V1:
Path uncertainty finds useful query opportunities,
but raw uncertainty produces too many interventions.

↓

Level 2 V2:
Allow passive sensing to absorb transient uncertainty.
Intervene when uncertainty persists.

↓

Current result:
Same aggregate task steps as Always Query
with 14 instead of 77 active queries
on the current five-map experiment.
```

---

# 22. Concise current thesis intuition

> A useful intelligent system does not need perfect knowledge at every moment. It needs enough knowledge to support the current decision. MSA attempts to distinguish uncertainty that can safely be absorbed through continued observation from uncertainty that is persistent and decision-relevant enough to justify intervention.

Short version:

> **Do not eliminate all uncertainty. Intervene only when uncertainty matters.**

---

# 23. Immediate next coding task

Recommended next implementation:

```text
Generated / unseen Level-2 maps
```

Keep these frozen during the first unseen-map test:

```text
persistent_path_threshold = 0.78
persistence_steps         = 2
action_path_gate          = 0.50
sensitivity_radius        = 4
stability_threshold       = 1.0
```

Do not tune against the unseen test maps before recording the first result.

---

# 24. Checkpoint status

```text
Level 0 baseline                         DONE
Level 1 controlled-error evidence        DONE
Level 2 partial observation              DONE
Level 2 QueryReality                     DONE
Level 2 counterfactual query labeling    DONE
Level 2 monitor V0                       DONE
Level 2 monitor V1                       DONE
Level 2 monitor V2                       CURRENT CHECKPOINT

Unseen-map robustness                    NEXT
Planner robustness                       TODO
Independent evaluator oracle             TODO
Query-cost sweep                         TODO

Level 3 learned latent representation    NOT STARTED
Level 4 physical robot validation        NOT STARTED
```

---

**Checkpoint date:** 2026-09-29  
**Project:** MSA — Model Shock Absorber
