"""Rolling, out-of-sample DC-level quantile forecast for the inventory twin.

The digital twin replays demand at the distribution-centre (state) level and puts a
**real, out-of-sample** forecast distribution in the agent's state. The forecast
layer of this project (notebooks 01/02b) showed LightGBM-quantile is the production
distributional model, so here we apply the *same* model to the DC-aggregated series
in an **expanding-window backtest**: refit periodically on all data seen so far and
predict the next block, so every forecast value is genuinely out-of-sample. This
produces a `(T, n_quantiles)` array spanning the whole replay horizon that the twin
can index into at any episode start -- decoupled from realised demand.

Used by notebooks 04 (twin validation) and 05 (RL), so the agent and the classical
baselines all see the identical forecast signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:                       # pragma: no cover
    _HAS_LGB = False

_FEAT = ["dow", "dom", "month", "woy", "is_weekend",
         "lag1", "lag7", "lag14", "lag28", "rmean7", "rmean28", "rstd7", "rstd28"]


def build_dc_demand(panel: pd.DataFrame, nodes) -> tuple[dict, pd.DatetimeIndex]:
    """Aggregate the SKU panel to daily demand per distribution centre (state)."""
    demand, dates = {}, None
    for n in nodes:
        s = panel[panel["state_id"] == n].groupby("date")["units"].sum().sort_index()
        demand[n] = s.values.astype(float)
        dates = s.index
    return demand, dates


def _features(arr, dates) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(arr, dtype=float)}, index=dates)
    d = df.index
    df["dow"] = d.dayofweek; df["dom"] = d.day; df["month"] = d.month
    df["woy"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    for L in (1, 7, 14, 28):
        df[f"lag{L}"] = df["y"].shift(L)
    for W in (7, 28):
        df[f"rmean{W}"] = df["y"].shift(1).rolling(W).mean()
        df[f"rstd{W}"] = df["y"].shift(1).rolling(W).std()
    return df


def rolling_quantile_forecast(arr, dates, q_levels=(0.5, 0.9),
                              block: int = 56, start_frac: float = 0.4,
                              num_boost_round: int = 200) -> np.ndarray:
    """Expanding-window LightGBM-quantile forecast over the whole series.

    The first ``start_frac`` of the series is seeded with a seasonal day-of-week
    quantile (so the array is full and usable), after which the model refits every
    ``block`` days on all prior data and predicts the next block out-of-sample.
    Returns an array of shape ``(len(arr), len(q_levels))`` with sorted (non-crossing)
    quantiles. Falls back to the seasonal estimator throughout if LightGBM is absent.
    """
    q_levels = list(q_levels)
    df = _features(arr, dates)
    n = len(df)
    out = np.full((n, len(q_levels)), np.nan)
    t0 = int(n * start_frac)

    # seasonal seed for the warm-up region
    dow_mean = df.iloc[:t0].groupby("dow")["y"].mean()
    sd0 = float(df.iloc[:t0]["y"].std())
    glob = float(df.iloc[:t0]["y"].mean())
    for i in range(t0):
        mu = float(dow_mean.get(df["dow"].iloc[i], glob))
        out[i] = [max(mu + (0.0 if q <= 0.5 else 0.84) * sd0, 0.0) for q in q_levels]

    if not _HAS_LGB:
        for i in range(t0, n):
            mu = float(dow_mean.get(df["dow"].iloc[i], glob))
            out[i] = [max(mu + (0.0 if q <= 0.5 else 0.84) * sd0, 0.0) for q in q_levels]
        return np.sort(out, axis=1)

    b0 = t0
    while b0 < n:
        b1 = min(b0 + block, n)
        tr = df.iloc[:b0].dropna(subset=_FEAT + ["y"])
        blk = df.iloc[b0:b1][_FEAT].fillna(0.0)
        for qi, q in enumerate(q_levels):
            m = lgb.train(dict(objective="quantile", alpha=q, learning_rate=0.05,
                               num_leaves=31, min_data_in_leaf=20, verbose=-1),
                          lgb.Dataset(tr[_FEAT], tr["y"]), num_boost_round=num_boost_round)
            out[b0:b1, qi] = np.clip(m.predict(blk), 0, None)
        b0 = b1
    return np.sort(out, axis=1)
