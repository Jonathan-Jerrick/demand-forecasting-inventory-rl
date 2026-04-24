"""Fast, data-independent sanity tests for the shared library.

These guard the invariants the project depends on so a regression in `src/` fails
loudly. They use synthetic demand (no M5 download needed) and run in well under a
second:  `pytest -q` from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as M
from src.simulator import InventoryDigitalTwin, TwinConfig, CostParams
from src import policies as P


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_perfect_point_forecast_scores_zero():
    y = np.array([3.0, 0.0, 5.0, 2.0])
    assert M.wape(y, y) == 0.0
    assert M.rmse(y, y) == 0.0


def test_pinball_nonnegative_and_zero_at_truth():
    y = np.array([2.0, 4.0, 1.0])
    assert M.pinball_loss(y, y, 0.5) == 0.0
    assert M.pinball_loss(y, y - 1.0, 0.9) >= 0.0


def test_coverage_in_unit_interval():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    cov = M.coverage(y, y - 1, y + 1)
    assert 0.0 <= cov <= 1.0 and cov == 1.0


def test_wrmsse_perfect_forecast_is_zero():
    # tiny synthetic panel: 2 items x 2 stores, 40 train days + 7 holdout
    dates = pd.date_range("2020-01-01", periods=47, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for item in ("A", "B"):
        for store in ("S1", "S2"):
            units = rng.integers(0, 10, size=len(dates))
            for d, u in zip(dates, units):
                rows.append(dict(item_id=item, dept_id="D", cat_id="C",
                                 store_id=store, state_id="X", date=d,
                                 units=float(u), revenue=float(u) * 2.0))
    panel = pd.DataFrame(rows)
    train = panel[panel.date < dates[40]]
    valid = panel[panel.date >= dates[40]].copy()
    ev = M.WRMSSEEvaluator(train, valid)
    pred = valid.rename(columns={"units": "pred"})[
        ["item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "pred"]]
    out = ev.score(pred)
    assert out["WRMSSE"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #
def _toy_twin(**cfg_kw):
    rng = np.random.default_rng(1)
    demand = {n: np.abs(rng.normal(100, 20, size=400)) for n in ("CA", "TX")}
    cfg = TwinConfig(episode_len=60, **cfg_kw)
    return InventoryDigitalTwin(demand, config=cfg, seed=0)


def test_obs_shape_and_finite():
    tw = _toy_twin()
    obs = tw.reset(seed=3)
    assert obs.shape == (tw.obs_dim,)
    assert np.isfinite(obs).all()


def test_lost_sales_inventory_never_negative_and_costs_nonneg():
    tw = _toy_twin(lost_sales=True)
    tw.reset(seed=5)
    for _ in range(tw.cfg.episode_len):
        _, _, done, info = tw.step({n: 50.0 for n in tw.nodes})
        for v in info["inventory"].values():
            assert v >= -1e-9
        for k in ("holding", "stockout", "ordering", "transship", "cost"):
            assert info[k] >= -1e-9
        if done:
            break


def test_episode_is_reproducible_from_seed():
    tw = _toy_twin()
    def roll(seed):
        tw.reset(seed=seed); costs = []
        for _ in range(tw.cfg.episode_len):
            _, _, done, info = tw.step({n: 80.0 for n in tw.nodes})
            costs.append(info["cost"])
            if done:
                break
        return costs
    assert roll(7) == roll(7)          # same seed -> identical
    assert roll(7) != roll(8)          # different seed -> different stochastics


def test_shock_off_recovers_deterministic_demand():
    tw = _toy_twin(demand_shock_std=0.0)
    tw.reset(seed=0)
    # with shocks off, realised demand equals the raw replayed series
    assert tw.realized_demand("CA", 0) == pytest.approx(tw.demand["CA"][tw.start])


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def test_orders_nonnegative():
    tw = _toy_twin()
    obs = tw.reset(seed=2)
    for fn in P.CLASSICAL_POLICIES.values():
        orders, _ = fn(obs, tw)
        assert all(q >= 0 for q in orders.values())


def test_oracle_is_a_lower_bound_vs_newsvendor():
    # the clairvoyant oracle must not cost more than the best deployable heuristic
    tw = _toy_twin()
    oracle = tw.run_policy(P.oracle_policy, n_episodes=8, seed_base=100)
    news = tw.run_policy(P.newsvendor_policy, n_episodes=8, seed_base=100)
    assert oracle["cost"] <= news["cost"]
    assert oracle["fill_rate"] >= 0.9


def test_emergency_pooling_reduces_stockouts():
    # with transshipment enabled, a sister DC's surplus should cover a DC's shortfall,
    # so unmet demand is no higher than with pooling off (and strictly lower on average).
    rng = np.random.default_rng(2)
    demand = {"CA": np.full(200, 100.0), "TX": np.full(200, 100.0)}
    def run(allow):
        cfg = TwinConfig(nodes=("CA", "TX"), episode_len=40, init_inventory=100.0,
                         demand_shock_std=0.4, shock_corr=0.0, allow_transship=allow)
        tw = InventoryDigitalTwin(demand, config=cfg, seed=7)
        tw.reset(seed=7); unmet = 0.0
        for _ in range(cfg.episode_len):
            # order each DC's mean only -> shortfalls happen, pooling must rescue them
            _, _, done, info = tw.step({"CA": 100.0, "TX": 100.0})
            unmet += sum(info["unmet"].values())
            if done:
                break
        return unmet
    assert run(allow=True) < run(allow=False)


def test_nnode_transship_conserves_stock():
    # an N-node net-flow redistribution must not create or destroy inventory
    tw = _toy_twin()
    tw.reset(seed=1)
    before = sum(tw.inv.values())
    moved = tw._apply_transship(np.array([-50.0, 30.0, 20.0][: len(tw.nodes)]))
    after = sum(tw.inv.values())
    assert moved >= 0
    assert after == pytest.approx(before, abs=1e-6)
