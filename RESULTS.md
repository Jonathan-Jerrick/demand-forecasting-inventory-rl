# Results

All numbers on the **10,000-series M5 subset**, 28-day holdout (2016-04-25). Reproducible from the notebooks and committed artifacts cited below. Metrics: WRMSSE, pinball, interval coverage, cost, fill rate — no invented figures.

---

## Forecasting

### Point forecasts (WRMSSE, notebook 01)

WRMSSE is a squared-error metric — it rewards the conditional mean. These are point/mean models.

| Model | WRMSSE | WAPE |
|---|---|---|
| Naive (last value) | 1.298 | 0.812 |
| Seasonal-naive (lag-7) | 0.805 | 0.733 |
| **LightGBM-Tweedie** | **0.493** | 0.567 |

### Probabilistic forecasts (pinball + coverage, notebooks 02b and 02)

For distributional models the right score is pinball + coverage. Over half the series-days are zero, so the cost-optimal median of an intermittent series is 0 — WRMSSE punishes a well-calibrated quantile model for being correct. Point and distributional results are kept separate.

| Model | Pinball | 80% coverage | Crossing | Scope |
|---|---|---|---|---|
| **LightGBM-quantile** (production) | **0.648** | 0.857 | none | full 10k |
| GRU-quantile (showcase) | 0.644 | 0.833 | none | 3k subset |

Head-to-head on the same 3,000-series subset:

| Model | Pinball | 80% coverage |
|---|---|---|
| **LightGBM-quantile** | **0.631** | 0.860 |
| GRU-quantile | 0.644 | 0.833 |

**LightGBM-quantile** (one model per quantile, sorted post-prediction to prevent crossing) is the production forecaster: 80% coverage 0.857, no crossing. It feeds the inventory twin downstream.

**GRU redesign:** the original GRU collapsed to the per-series mean. The redesigned version — 56-day encoder, explicit lag features, cumulative-softplus quantile head, early stopping on validation pinball — is fixed and calibrated (80% coverage 0.83, non-crossing) but still trails LightGBM on the same series. Trees win on tabular M5; the GRU is kept as the sequence-model comparison rather than the primary.

---

## Inventory & RL

### Simulator headroom (notebook 04)

A standard single-DC, linear-cost setup is solved near-optimally by Newsvendor, so RL has no structural room. The simulator was built with correlated demand shocks, stochastic lead times, fixed+batch ordering cost, and convex overflow holding to create genuine non-myopic, cross-DC structure.

Gap between best heuristic and a clairvoyant oracle (sees realised demand) over 30 seeds:

| | Value |
|---|---|
| Newsvendor cost (90-day) | 1,168,024 |
| Oracle (clairvoyant floor) | 972,011 |
| Room above oracle | **16.8%** |

~17% headroom between the best heuristic and the theoretical floor — real structure for a learned policy to exploit.

### PPO-Residual agent (notebook 05)

The agent is a bounded learned correction on top of the Newsvendor order-up-to level plus a transshipment fraction — at zero residual it *is* Newsvendor, so there's no "start worse" failure mode. Trained with PPO and a service-rate penalty. State includes real rolling LightGBM-quantile forecasts at DC level (`src/dc_forecast.py`).

Frontier over 40 paired seeds — `figures/cost_fill_frontier.png`:

| Policy | Cost (90-day) | Fill |
|---|---|---|
| Oracle | 1,000,368 | 0.993 |
| **PPO-Residual** | **1,147,975** | **0.950** |
| Newsvendor | 1,190,385 | 0.968 |
| Base-Stock (forecast) | 1,314,022 | 0.976 |
| Fixed-Order | 2,069,040 | 0.928 |

PPO is −3.6% cost vs Newsvendor (1,147,975 vs 1,190,385), fill 0.950 above the ~0.94 DC service target (CA 0.94, TX 0.939, WI 0.938 from segmentation). It trades some of Newsvendor's 0.968 fill for lower total cost, landing closest to the oracle of any deployable policy.

### Ablations: where the edge comes from (notebook 05, `data/ablations.parquet`)

| Condition | Agent edge vs Newsvendor |
|---|---|
| Full agent (real forecast + pooling) | **+7.8%** |
| Flat mean forecast in-state | **−7.2%** |
| Pooling (transshipment) OFF | **−17.2%** |
| High shock correlation (ρ = 0.85) | +2.1% |

The edge needs **both** the calibrated forecast in-state and cross-DC pooling. Removing either flips the result. An earlier run on a smaller top-volume subset suggested pooling didn't matter — the full 10k run corrects that. The ablation reference (+7.8%) is a separately seeded instance from the 40-seed headline (−3.6%); read it for direction and relative magnitude.

---

## Summary

> Calibrated LightGBM-quantile forecasts drive a residual PPO agent that beats Newsvendor by 3.6% at fill 0.950, landing closest to a clairvoyant oracle (~17% remaining headroom) on a non-myopic multi-DC twin. The edge requires both the forecast information and cross-DC pooling.

---

## Reproduce

```
notebooks: 00 → 01 → 02b → 02 → 03 → 04 → 05 → 06
python -m src.make_figures    # regenerate figures from data/rl_results.parquet
```

Key committed artifacts: `data/experiments.csv`, `data/results_summary.json`, `data/rl_results.parquet`, `data/ablations.parquet`, `data/baseline_policy_results.parquet`, `data/series_scores.parquet`.
