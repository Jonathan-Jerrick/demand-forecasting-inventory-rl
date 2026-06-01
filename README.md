# Demand Forecasting & Inventory Optimization on M5

End-to-end pipeline on the public [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Walmart hierarchical daily sales, 2011–2016): probabilistic demand forecasting → inventory segmentation → RL-based replenishment on a multi-DC simulator.

10,000-series subset, 28-day holdout starting 2016-04-25.

---

## What it does

**Forecast:** Trains a calibrated quantile forecast (LightGBM and a GRU) over item-store series. Scored by pinball loss + interval coverage, not just point error — because on intermittent demand the cost-optimal median is often zero, making WRMSSE misleading for distributional models.

**Simulate:** Builds a multi-DC inventory simulator that replays real M5 demand with correlated shocks, stochastic lead times, fixed+batch ordering costs, and overflow holding. A clairvoyant oracle (sees future demand) sets the theoretical cost floor.

**Optimize:** Trains a residual PPO agent — a learned correction on top of the Newsvendor order-up-to level — with a service-rate penalty so it can't win by cutting fill. The agent's state includes the real rolling forecast quantiles.

---

## Results

| Policy | Cost (90-day) | Fill rate |
|---|---|---|
| Oracle (clairvoyant floor) | 1,000,368 | 0.993 |
| **PPO-Residual** | **1,147,975** | **0.950** |
| Newsvendor | 1,190,385 | 0.968 |
| Base-Stock (forecast) | 1,314,022 | 0.976 |
| Fixed-Order | 2,069,040 | 0.928 |

PPO beats Newsvendor by 3.6% at fill 0.950 (above the ~0.94 DC service target). The oracle marks ~17% remaining headroom.

Forecasting (probabilistic, head-to-head on same 3k subset):

| Model | Pinball | 80% coverage |
|---|---|---|
| LightGBM-quantile | 0.631 | 0.860 |
| GRU-quantile | 0.644 | 0.833 |

Trees win on this tabular problem. The GRU is fixed and calibrated but kept as the sequence-model comparison. Point forecast: LightGBM-Tweedie at WRMSSE 0.493.

Two findings worth flagging: (1) gradient-boosted trees beat the neural net on tabular M5 — reported as-is rather than hidden. (2) On the full 10k twin, the RL edge needs *both* the calibrated forecast in-state *and* cross-DC pooling — removing either flips it to a loss (ablations in notebook 05).

Full numbers in [`RESULTS.md`](RESULTS.md).

| Cost/fill frontier | Cost breakdown |
|---|---|
| ![frontier](figures/cost_fill_frontier.png) | ![decomposition](figures/cost_decomposition.png) |

---

## Structure

```
notebooks/
  00_data_preprocessing.ipynb
  01_baseline_forecasting.ipynb
  02b_lightgbm_quantile.ipynb       production quantile forecaster
  02_probabilistic_neural_forecasting.ipynb   GRU quantile forecaster
  03_inventory_segmentation.ipynb
  04_inventory_digital_twin.ipynb
  05_rl_inventory_agent.ipynb
  06_order_policy_and_results.ipynb
src/
  metrics.py      WRMSSE, pinball, coverage, inventory metrics
  simulator.py    multi-DC digital twin + oracle
  policies.py     classical baselines + clairvoyant oracle
  dc_forecast.py  rolling out-of-sample DC-level quantile forecast
  features.py     shared feature engineering
  tracking.py     experiment logger
```

---

## Setup

Download the M5 dataset from Kaggle into `data/raw_m5/` (`sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv`). Everything downstream is built by notebook 00.

```bash
pip install -r requirements.txt
# run notebooks in order: 00 → 01 → 02b → 02 → 03 → 04 → 05 → 06
pytest -q
```

Notebooks 02 and 05 use PyTorch (MPS/CUDA/CPU auto-detected). Everything else is CPU.
