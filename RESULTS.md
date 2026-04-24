# Results

End-to-end results for the demand-forecasting + inventory-RL pipeline on a
**10,000-series M5 subset** (2016 holdout, 28-day horizon starting 2016-04-25). Every
number here is reproducible from the notebook / committed artifact it cites — only
WRMSSE / pinball / coverage / cost / fill, no invented figures. Each run also appends to
`data/experiments.csv`.

This document records the outcome of the improvement plan (`IMPROVEMENT_PLAN.md`),
which was written after a first run exposed two failures:

1. the neural quantile forecaster **collapsed to the per-series mean** (worse than a
   naive forecast), and
2. the from-scratch Double-DQN **under-ordered to cut holding**, landing costlier than
   Newsvendor at lower fill.

Both are fixed below, and two results are reported honestly as the plan's kill
criteria anticipated.

> **Scale note.** Numbers are reported on the full 10k-series subset (the long
> intermittent tail included), not a hand-picked top-volume slice. This makes the
> forecasting problem genuinely hard — over half of series-days are zero — and the
> headline numbers should be read in that light.

---

## 1. Forecasting

### The accuracy ladder — point forecasts, scored by WRMSSE (notebook 01)
WRMSSE is a **squared-error** metric, so it rewards an estimate of the **conditional
mean**. These rows are point forecasts; the distributional models are scored separately
below.

| Tier | WRMSSE | WAPE |
|---|---|---|
| Naive (last value) | 1.298 | 0.812 |
| Seasonal-naive (lag-7) | 0.805 | 0.733 |
| **LightGBM (Tweedie, point)** | **0.493** | 0.567 |

Gradient-boosted trees dominate this tabular problem — WRMSSE 0.493 vs a 1.0 naive
random-walk scale — and that is the context for everything that follows. (Recomputed at
10k scale from `data/baseline_forecast.parquet` + the panel; WRMSSE via
`src.metrics.WRMSSEEvaluator` over all 12 M5 levels.)

### Probabilistic forecasters — scored by pinball + coverage (notebooks 02b, 02)
For the **distributional** models the right score is **pinball loss + interval
coverage**, not WRMSSE: more than half of series-days are zero, so the cost-optimal
*median* of an intermittent series is 0, and judging a quantile model's median with a
squared-error metric would punish it for being correctly calibrated. We therefore rank
the distributional models by pinball/coverage and keep WRMSSE for the point/mean tier
above.

| Model | Mean pinball | 80% PI coverage | 50% PI coverage | Crossing | Scope | Notebook |
|---|---|---|---|---|---|---|
| **LightGBM-quantile** (production) | **0.648** | 0.857 | 0.523 | none | full 10k | `02b` |
| Neural GRU-quantile (showcase) | 0.644 | 0.833 | — | none | 3k subset | `02` |

**Head-to-head on the same 3,000-series subset** (apples-to-apples):

| Model | Mean pinball | 80% PI coverage |
|---|---|---|
| **LightGBM-quantile** | **0.631** | 0.860 |
| Neural GRU-quantile | 0.644 | 0.833 |

**T1 — LightGBM-quantile** (one model per quantile, sorted to prevent crossing) is a
calibrated probabilistic forecaster: 80% interval coverage 0.857 (nominal 0.80, mildly
conservative), no quantile crossing. It is the **production distributional forecaster**
consumed downstream. *Acceptance: coverage ∈ [0.78, 0.88] ✓, monotone quantiles ✓.*

**T2 — Neural redesign.** The mean-collapse is gone — the GRU now produces a genuine
spread (non-degenerate, non-crossing quantiles), is calibrated (80% coverage 0.83), and
trains to a proper **validation-pinball** optimum with early stopping, the diagnostic
the original lacked. But on this tabular problem it lands a hair behind LightGBM-quantile
on the same series (pinball 0.644 vs 0.631, coverage 0.833 vs 0.860).

**T3 / kill criterion §6.** Two serious attempts left the neural net behind the tree on
the proper distributional score, so per the plan we **ship LightGBM-quantile as
primary** and keep the neural model as the calibrated sequence-model showcase. *Honest
outcome: GBM wins this tabular problem; the neural net is fixed and calibrated but not
the production model.*

---

## 2. Inventory & RL

### T4 — Reworked twin (the gate)
A single-DC, lost-sales, linear-cost, daily-ordering problem is solved near-optimally
by a base-stock/newsvendor level, so RL has no room (Fact A). The twin
(`src/simulator.py`) was reworked to add **correlated demand shocks, stochastic lead
times, fixed + batch ordering cost, and convex (overflow) holding**, creating
non-myopic, cross-DC structure.

Measured as **room above a clairvoyant oracle** (an order policy that sees realised
demand — a cost lower bound), over 30 paired seeds (notebook 04):

| Quantity | Value |
|---|---|
| Best heuristic (Newsvendor) cost, 90-day | 1,168,024 |
| Oracle (clairvoyant floor) | 972,011 |
| **Room above oracle** | **16.8%** |

**Gate PASS** — the reworked regime leaves a clairvoyant oracle ~17% below the best
per-DC heuristic, i.e. real structure for a learned policy to exploit.

### T5 — Residual PPO agent
The agent acts as a **bounded residual around the Newsvendor order-up-to level** plus a
transshipment fraction (so at residual 0 it *is* Newsvendor — no "start worse" risk),
trained with **PPO** and a **service-penalty (Lagrangian)** term so it cannot exploit
the cheap low-inventory corner that sank the old DQN. State includes the **real
out-of-sample forecast** (rolling LightGBM-quantile at DC level, `src/dc_forecast.py`).

Frontier over 40 paired seeds (new regime), notebook 05 — see
`figures/cost_fill_frontier.png`:

| Policy | Cost (90-day) | Fill | Note |
|---|---|---|---|
| Oracle | 1,000,368 | 0.993 | clairvoyant floor |
| **PPO-Residual (ours)** | **1,147,975** | **0.950** | beats best heuristic |
| Newsvendor | 1,190,385 | 0.968 | best classical |
| Base-Stock (forecast) | 1,314,022 | 0.976 | |
| Fixed-Order | 2,069,040 | 0.928 | |

**T5 PASS** — PPO is **−3.6% cost vs Newsvendor** (1,147,975 vs 1,190,385) on paired
multi-seed evaluation, holding fill at **0.950 — above the ~0.94 DC service target**
(CA 0.94, TX 0.939, WI 0.938; notebooks 03–04). It trades a little of Newsvendor's
0.968 fill for lower total cost, landing closest to the oracle floor of any deployable
policy. (Contrast the old from-scratch DQN, which lost to Newsvendor by 27%.)

### T7 — Ablations: where the edge comes from
Each row retrains the agent under one changed condition (notebook 05,
`data/ablations.parquet`). Edge = how much cheaper the agent is than Newsvendor under
that condition (positive = agent wins):

| Condition | Agent edge vs Newsvendor |
|---|---|
| Full agent (real forecast, pooling on) | **+7.8%** |
| Flat (mean) forecast in-state | **−7.2%** |
| Pooling (transshipment) OFF | **−17.2%** |
| High shock correlation (ρ = 0.85) | +2.1% |

**Honest finding (this flips the earlier subset's conclusion).** On the full 10k twin
the edge is driven by **both** levers, and removing **either** erases it:

- **Forecast information is necessary** — replacing the real calibrated forecast with a
  flat mean forecast turns the win into a loss (+7.8% → −7.2%).
- **Cross-DC pooling is necessary too** — turning transshipment off is even more
  damaging (+7.8% → −17.2%).

So the agent's advantage comes from *acting on a calibrated probabilistic forecast
across a pooled network*, not from either ingredient alone. We report this rather than
claim the single-driver story the smaller top-volume subset suggested.

> Reproducibility note: the ablation study retrains the agent per condition, so its
> full-agent reference (+7.8%) is a separately seeded instance and differs in absolute
> magnitude from the 40-seed frontier headline (−3.6%); read the ablation column as the
> **direction and relative size** of each lever, not as a second point estimate of the
> headline.

---

## 3. Headline

> A calibrated LightGBM-quantile forecast drives a residual PPO replenishment policy
> that **beats a strong analytical baseline (Newsvendor) by 3.6%** on a reworked,
> non-myopic multi-DC twin grounded in real M5 demand, while holding fill above the
> ~0.94 DC service target — landing closest to a clairvoyant oracle that marks
> ~17% of remaining theoretical headroom. Ablations show the edge needs **both** the
> forecast information and cross-DC pooling.

Two results are reported as honest negatives, by design: gradient-boosted trees beat the
neural forecaster on this tabular data, and on the full 10k twin the RL agent's
advantage depends on cross-DC pooling as well as forecast information (the smaller
top-volume subset had suggested pooling didn't matter — the larger run corrected that).

## Reproduce
Run notebooks in order: `00 → 01 → 02b → 02 → 03 → 04 → 05 → 06`.
Regenerate the figures from the committed result table: `python -m src.make_figures`.
Key artifacts: `data/lgbm_forecast.parquet` (production forecast, gitignored — rebuilt
by 02b), `data/dc_forecast_q.npz` (twin forecast), `data/rl_results.parquet` (frontier),
`data/ablations.parquet`, `data/results_summary.json`, `data/experiments.csv`.

> Note: train LightGBM and PyTorch in **separate processes** on macOS (duplicate
> OpenMP runtimes segfault). The pipeline already does this — notebook 04 builds and
> caches the DC forecast, notebook 05 loads the cache and only uses PyTorch.
