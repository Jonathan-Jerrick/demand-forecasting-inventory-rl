# Improvement Plan — Hand-off Spec for Claude Code

## DECISIONS LOCKED (do not re-litigate)
1. **RL ambition: rework the environment so RL genuinely wins.** §4.1 (Task 4) is
   committed, not optional. Build correlated demand shocks + stochastic lead times +
   batch/fixed ordering cost, then a residual PPO/SAC agent. All of §4 is in scope.
2. **Primary forecaster: LightGBM-quantile.** It is the production distributional
   model downstream and the one wired into the twin (T6). The neural net (T2) is the
   sequence-model showcase and must only stay *competitive*; it does not need to win.

---

This is an implementation spec. Each task has a **goal**, **files to touch**,
**approach**, and **acceptance criteria** so the work can be verified without a
human in the loop. Execute in the order given; respect the **kill criteria** in §6
so effort is not wasted on structurally-capped models.

Context: a 300-series M5 subset run exposed two problems. (1) The GRU quantile
forecaster collapsed to the per-series mean (WRMSSE 1.45, worse than naive 1.03;
LightGBM was best at 0.46). (2) The Double-DQN under-orders to cut holding cost,
blowing up stockouts (+27% total cost vs Newsvendor, fill 0.90). Calibration of the
neural net is fine (80% PI coverage ≈ 0.815); only the *location* is broken.

---

## 0. Two strategic facts that constrain the work

**Fact A — Newsvendor is near-optimal for the current inventory problem.**
Single-DC, lost-sales, linear holding/stockout cost, daily ordering ⇒ a base-stock
/ newsvendor level is provably near-optimal. **No amount of DQN training beats it.**
RL only wins when the environment has structure a per-DC heuristic cannot exploit.
Therefore RL work is gated on the environment changes in §4.1.

**Fact B — Gradient-boosted trees dominate M5.** The neural forecaster's success
criterion is *calibrated + competitive*, **not** "beat LightGBM." The production
distributional forecaster will be LightGBM-quantile (§3.1). The neural model (§3.2)
is the sequence-model showcase and must merely come within ~10% WRMSSE of LightGBM
without mean-collapse.

Both facts are honest results in their own right. Present them; do not hide them.

---

## 1. Recommended target architecture (end state)

```
Forecast layer
  • LightGBM-quantile  → 5 quantiles, the production distributional forecaster
  • Neural (improved)  → sequence-model showcase, competitive + calibrated
Decision layer
  • Digital twin reworked so multi-period / cross-DC structure MATTERS
  • RL = residual policy on top of base-stock, with a service constraint
  • The twin's state is fed the REAL forecast quantiles (rolling), not a fallback
```

The headline narrative becomes: *"a calibrated probabilistic forecast drives a
learned replenishment policy that beats strong heuristics specifically where
cross-DC risk-pooling and lost-sales-with-lead-time make the problem non-myopic."*

---

## 2. Execution order (do these in sequence)

1. §3.1 LightGBM-quantile forecaster — fast, high-certainty win.
2. §3.2 Fix the neural forecaster — convergence + features + target transform.
3. §5 Evaluation rigor — multi-seed, validation curves, experiment log (needed to
   trust everything after).
4. §4.1 Rework the twin so RL has structural room — **gate for all RL work.**
5. §4.2 Residual RL policy + service constraint + warm start.
6. §4.3 Wire the real forecast into the twin state.
7. §4.4 Ablations that prove where RL's edge comes from.

---

## 3. Forecasting fixes

### T1 — LightGBM-quantile forecaster (do first) `[high ROI, low risk]`
**Goal:** a calibrated 5-quantile forecast inheriting LightGBM's 0.46 WRMSSE.
**Files:** new `notebooks/02b_lightgbm_quantile.ipynb`; reuse `src/metrics.py`.
**Approach:** train one LightGBM per quantile α ∈ {0.1, 0.25, 0.5, 0.75, 0.9} with
`objective="quantile", alpha=α`, identical features to notebook 01. After
prediction, **sort the 5 quantile outputs per row** to guarantee non-crossing.
Save to `data/lgbm_forecast.parquet` in the same schema as `neural_forecast.parquet`
(`series_id, date, units, q0.1…q0.9, ids`).
**Acceptance:**
- median WRMSSE ≤ 0.55 (i.e. close to the point-LightGBM 0.46).
- mean pinball **lower** than the GRU's current value.
- 80% PI coverage in [0.78, 0.86]; no crossing after the sort.

### T2 — Fix the neural quantile forecaster `[medium ROI, medium risk]`
**Goal:** kill the mean-collapse; land within ~10% WRMSSE of LightGBM.
**Files:** `notebooks/02_probabilistic_neural_forecasting.ipynb`.
**Root causes to fix (all of them):**
- **Undertraining.** Add a **validation split** (hold out the last 28 days of the
  *training* region as an inner val), track **val pinball every epoch**, add
  **early stopping** (patience 8) and a **cosine or ReduceLROnPlateau** schedule.
  Train up to 60 epochs (cheap on MPS). The current notebook tracks *train* loss
  only — that is why nobody saw it never converged.
- **Target transform.** Stop normalizing by per-series *mean alone* (collapses the
  signal). Model in **log1p(units)** space, or standardize by **mean+std**. For
  intermittent series prefer a **Tweedie or zero-inflated/negative-binomial head**
  over raw-count pinball.
- **Starved inputs.** Feed the **engineered lag/rolling features** from notebook 00
  into the encoder (hybrid: GRU over history **+** explicit `lag_{1,7,14,28}`,
  `rmean_{7,28}` concatenated). Lengthen the **encoder window to 56 days** so the
  net sees weekly + monthly structure. Add explicit day-of-week / month signals.
- **Quantile crossing.** Make the head emit the lowest quantile then **non-negative
  increments via softplus**, cumulatively summed, so q0.1 ≤ … ≤ q0.9 by construction.
**Acceptance:**
- a **validation pinball curve** is plotted and flattens before training stops.
- **no mean-collapse**: per-series std of the q0.5 forecast across the 28 days is
  ≥ 50% of the actual per-series demand std (add this as an automated check/assert).
- median WRMSSE ≤ 1.1 × (LightGBM-quantile WRMSSE).
- quantiles non-crossing; 80% coverage in [0.78, 0.86].

### T3 — Pick the production forecaster
After T1/T2, set the forecaster consumed downstream to whichever wins on WRMSSE +
pinball (expected: LightGBM-quantile). Record the comparison in `RESULTS.md`.

---

## 4. Inventory / RL fixes

### T4 (== §4.1) — Rework the twin so RL has structural advantage `[GATE]`
**Goal:** change the problem so a per-DC heuristic is *no longer near-optimal*.
Without this, skip §4.2 entirely. **Files:** `src/simulator.py`, `notebooks/04`.
Add the levers that create non-myopic, cross-DC value (implement at least the first
two; all four is better):
1. **Correlated demand shocks + stochastic, correlated lead times.** Make CA and TX
   demand share a common shock and lead times random and occasionally long. Now
   **transshipment / risk-pooling** between DCs has real value — and per-DC
   newsvendor *cannot* pool. This is the primary lever.
2. **Fixed ordering cost + minimum/batch order quantity.** A non-trivial
   `order_fixed` and an order multiple make the optimal policy an **(s,S)** /
   non-myopic policy; "order to a level every day" becomes suboptimal.
3. **Lost-sales WITH lead time L>0** (already lost-sales — keep it, and *raise* L
   and its variance). This problem has **no simple optimal policy**; base-stock is a
   heuristic, so there is genuine room above it.
4. **Nonlinear holding** (e.g. capacity overflow penalty / perishability/expiry)
   so over-ordering is punished super-linearly.
**Acceptance:** re-run the classical policies; the gap between the best heuristic
and a *clairvoyant* lower bound (an oracle that sees true demand) must widen
materially vs today — that gap is the room RL can capture. Document the gap.

### T5 (== §4.2) — Residual RL policy + service constraint + warm start `[gated on T4]`
**Goal:** an agent that starts no worse than base-stock and improves on it.
**Files:** `notebooks/05_rl_inventory_agent.ipynb`.
- **Residual action.** Action = base-stock order-up-to **+ learned adjustment**
  (small bounded delta per DC + transship). Floor ≈ newsvendor; RL only learns the
  correction. Collapses the 108-action search and removes the "start worse" risk.
- **Warm start.** Pre-fill replay with base-stock trajectories and/or
  behavior-clone the base-stock policy for a few hundred steps before RL.
- **Service in the objective, not just the state.** Add a **fill-rate penalty /
  Lagrangian** term so the agent is punished for dropping below the segment service
  target. This kills the "exploit the cheap low-inventory corner" failure.
- **Reward scaling.** Normalize holding and stockout to comparable magnitudes so
  stockout spikes are not washed out by smooth daily holding.
- **Algorithm.** Order quantity is continuous → prefer **PPO or SAC** with a
  continuous residual action over coarse 6-level DQN. Keep DQN only as a comparison.
- **Train to convergence** (thousands of episodes on MPS), not 400/20s.
**Acceptance:**
- DQN/PPO total cost ≤ best classical policy at **fill ≥ segment service target**,
  on a **multi-seed** eval (§5). If it only ties, that is acceptable *only* if §4.4
  shows it wins in the correlated-shock regime.

### T6 (== §4.3) — Wire the real forecast into the twin state `[narrative payoff]`
**Goal:** the agent acts on the *deep forecast's* distribution, not the twin's own
fallback. **Files:** `src/simulator.py` (already accepts `forecast_q`),
`notebooks/04` & `05`.
Replace the seasonal-quantile fallback used for training/eval with a **rolling**
forecast from the chosen model (T3): at each step the state's `forecast_q` window
comes from the model's quantile prediction for the upcoming lead-time days. For
tractability over many episodes, precompute a rolling/backtested quantile forecast
for the whole replay horizon and index into it.
**Acceptance:** an **ablation** (§4.4) shows the agent with real forecast quantiles
in-state beats the same agent with a flat/mean forecast in-state.

### T7 (== §4.4) — Ablations that prove the edge
Run and tabulate: (a) RL with vs without forecast-in-state; (b) RL with vs without
transshipment enabled; (c) RL in the correlated-shock regime vs the i.i.d. regime.
**Acceptance:** at least one ablation isolates a regime where RL clearly beats the
best heuristic, with the mechanism named (pooling / non-myopic ordering).

---

## 5. Evaluation rigor (applies to everything after T1)

- **Multi-seed evaluation** with mean ± std over ≥ 20 seeds; never report a
  single-seed number. The current RL eval is single-seed and noisy.
- **Validation curves** for every trained model (forecaster and RL). No model is
  "done" without a convergence plot.
- **Experiment log**: append every run's config + headline metrics to
  `experiments.csv` so results are reproducible and comparable.
- **Clairvoyant lower bound** for the inventory problem (oracle that sees true
  demand) so "how much room is left" is always quantified.

---

## 6. Kill criteria (stop wasting effort)

- **Forecaster:** if after T2 the neural net still loses to LightGBM-quantile by
  > 10% WRMSSE, **ship LightGBM-quantile as primary** and present the neural model
  as "explored; GBM won on this tabular problem" — an honest, defensible outcome.
  Do **not** keep tuning the net past two serious attempts.
- **RL:** if after T4 + T5 the agent still cannot beat the best heuristic at the
  service target on multi-seed eval, **stop**. Present RL as "matches a strong
  analytical baseline, and wins specifically under correlated-shock / transshipment
  regimes (§4.4)." Do not fabricate a win.

---

## 7. Guidelines for Claude Code

- Work **task by task in the §2 order**; commit after each with the acceptance
  criteria checked. Do not start RL (§4.2) before the env rework (§4.1) passes.
- Keep all shared logic in `src/`; notebooks stay thin and runnable top-to-bottom.
- **MPS:** device selection is `cuda → mps → cpu`; verify `torch.backends.mps`.
- Every claim in `RESULTS.md` must be reproducible from a notebook run — **no
  invented numbers, no business-dollar figures.** WRMSSE / pinball / coverage /
  cost / fill only.
- Add the automated **assert checks** named in the acceptance criteria (mean-collapse
  guard, non-crossing guard, fill ≥ target) so regressions fail loudly.
- After implementation, write `RESULTS.md`: final metrics table, the two honest
  findings (GBM vs neural; where RL wins), and the convergence + frontier plots.
- If a model hits its kill criterion, follow §6 and report it — a clear-eyed
  negative result is a feature of this project, not a bug.
