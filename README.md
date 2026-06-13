# Demand Forecasting & Inventory Optimization on M5

I built this to understand how a probabilistic forecast can actually drive an inventory decision, not just improve a metric. It runs on the public [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — 10,000 item×store daily sales series from Walmart (2011–2016) — and takes the forecast all the way to an order quantity on a multi-DC simulator, with a clairvoyant oracle to mark where theoretical optimality sits.

**Headline:** a residual PPO agent beats the analytical Newsvendor baseline by 3.6% at fill rate 0.950, on a 40-seed evaluation. The clairvoyant oracle sits ~17% below the best heuristic — that's what's left on the table.

---

## Why these choices

**LightGBM over a neural net as the primary forecaster.** I tried both. The GRU I built is calibrated and non-collapsing (after fixing a mean-collapse bug in v1), but LightGBM-quantile still wins on pinball loss (0.631 vs 0.644) and coverage (0.860 vs 0.833) on the same 3k-series subset. This isn't surprising — most top M5 Kaggle solutions are GBM-based; the series are too sparse and tabular for a recurrent net to have a structural advantage. I kept the GRU in the repo because it shows the sequence-modelling work, but it's not the production forecaster.

**Residual PPO instead of from-scratch DQN.** My first RL attempt was a from-scratch Double-DQN. It lost to Newsvendor by 27% — it learned to under-order to cut holding cost, which torpedoed fill rate. The problem was the environment: a standard single-DC, linear-cost daily-ordering setup is near-optimally solved by the Newsvendor formula analytically, so RL had no room. I reworked the twin with correlated shocks, stochastic lead times, batch ordering cost, and lateral transshipment between DCs before touching RL again. The residual PPO formulation (action = correction on top of Newsvendor, not an order from scratch) was the other key fix — it starts at the analytical baseline and only learns to improve from there.

**Pinball loss + coverage instead of WRMSSE for distributional models.** Over half the series-days are zero. The cost-optimal median of an intermittent series is zero, so a well-calibrated quantile model gets penalised by WRMSSE for being correct. I use WRMSSE only for the point-forecast tier and pinball+coverage for the distributional models — they're different questions.

---

## Results

| Policy | Cost (90-day) | Fill rate |
|---|---|---|
| Oracle (clairvoyant floor) | 1,000,368 | 0.993 |
| **PPO-Residual** | **1,147,975** | **0.950** |
| Newsvendor | 1,190,385 | 0.968 |
| Base-Stock (forecast) | 1,314,022 | 0.976 |
| Fixed-Order | 2,069,040 | 0.928 |

Forecasting head-to-head on the same 3,000-series subset:

| Model | Pinball | 80% coverage | WRMSSE (median) |
|---|---|---|---|
| LightGBM-Tweedie (point) | — | — | **0.493** |
| LightGBM-quantile | **0.631** | **0.860** | — |
| GRU-quantile | 0.644 | 0.833 | — |

Two findings I'd call honest negatives: GBM beats the neural net on tabular M5 (reported as-is), and the RL edge on the full 10k twin needs *both* the calibrated forecast in-state *and* cross-DC pooling — removing either flips it to a loss. An earlier run on a smaller top-volume subset had suggested pooling didn't matter; the full-scale run corrected that.

| Cost/fill frontier | Cost breakdown |
|---|---|
| ![frontier](figures/cost_fill_frontier.png) | ![decomposition](figures/cost_decomposition.png) |

Full numbers in [`RESULTS.md`](RESULTS.md).

---

## One thing specific to this dataset

The M5 data has pronounced sales spikes on the 1st–3rd and 28th–31st of each month — SNAP (food stamp) disbursement and payday patterns. I added an `is_payday_window` binary feature (flag those 6 days per month) after noticing the pattern in lag residuals. It's a small thing but it's the kind of signal a rolling mean would wash out.

A weirder one: `sell_prices.csv` in M5 is at item×store×week granularity, not daily. Joining it to the daily panel without forward-filling the weekly prices creates a lot of accidental NaNs that make price-ratio features silently degenerate. That bit me in preprocessing.

---

## Structure

```
notebooks/
  00_data_preprocessing.ipynb
  01_baseline_forecasting.ipynb
  02b_lightgbm_quantile.ipynb        production quantile forecaster
  02_probabilistic_neural_forecasting.ipynb   GRU quantile (calibrated, for comparison)
  03_inventory_segmentation.ipynb
  04_inventory_digital_twin.ipynb
  05_rl_inventory_agent.ipynb
  06_order_policy_and_results.ipynb
src/
  metrics.py       WRMSSE (all 12 M5 levels), pinball, coverage, inventory metrics
  simulator.py     multi-DC digital twin + clairvoyant oracle
  policies.py      Newsvendor, base-stock, min/max, fixed-order, oracle
  dc_forecast.py   rolling out-of-sample DC-level quantile forecast for the twin state
  features.py      lag, rolling, calendar, price features (shared by preprocessing + notebooks)
  tracking.py      experiment logger → data/experiments.csv
scratch/           early DQN exploration (abandoned; kept for reference)
```

---

## Setup

Download the M5 dataset from Kaggle (`m5-forecasting-accuracy`) into `data/raw_m5/` — you need `sales_train_evaluation.csv`, `sell_prices.csv`, and `calendar.csv`. Notebook 00 builds everything downstream. A `SUBSET` flag controls scale (a few hundred series on CPU up to the full panel).

```bash
pip install -r requirements.txt
# notebooks in order: 00 → 01 → 02b → 02 → 03 → 04 → 05 → 06
pytest -q
```

PyTorch notebooks (02, 05) auto-detect MPS/CUDA/CPU. Everything else is CPU only.

Note: train LightGBM and PyTorch in separate kernel processes — both ship OpenMP runtimes that conflict on macOS. Notebooks 04 and 05 already do this by design (04 caches the DC forecast, 05 loads it).
