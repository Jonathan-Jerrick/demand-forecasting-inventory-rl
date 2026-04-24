"""Forecasting and inventory evaluation metrics.

This module is intentionally dependency-light (numpy + pandas only) so it can be
imported in any notebook and unit-tested without a GPU.

It provides three families of metrics:

1. Point-accuracy metrics (WAPE, RMSE, BIAS, MAPE) -- quick sanity checks.
2. The official M5 accuracy metric, WRMSSE, implemented over the 12 standard
   aggregation levels with dollar-sales weighting.
3. Probabilistic-forecast metrics (pinball / quantile loss, weighted scaled
   pinball loss) used to score the neural quantile forecaster.
4. Inventory metrics (fill rate, total cost decomposition) used to score the
   replenishment policies and the RL agent on the digital twin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Point-accuracy metrics
# --------------------------------------------------------------------------- #
def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error = sum|e| / sum|y|."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred) -> float:
    """Forecast bias as a fraction of total actual demand (positive = over-forecast)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    s = np.sum(y_true)
    return float(np.sum(y_pred - y_true) / s) if s > 0 else np.nan


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))))


def point_metrics(y_true, y_pred) -> dict:
    return {
        "WAPE": round(wape(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "BIAS": round(bias(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
    }


# --------------------------------------------------------------------------- #
# 2. WRMSSE -- the official M5 accuracy metric
# --------------------------------------------------------------------------- #
# RMSSE for one bottom-level series:
#
#         sqrt( mean_h (y_h - yhat_h)^2 )
#   RMSSE = ----------------------------------------------------
#         sqrt( (1/(n-1)) * sum_{t=2..n} (y_t - y_{t-1})^2 )
#
# i.e. the forecast RMSE on the horizon divided by the in-sample RMSE of a naive
# (random-walk) forecast on the training history. WRMSSE aggregates RMSSE across
# the 12 standard M5 levels; within each level series are weighted by their share
# of cumulative dollar sales over the last 28 training days, and the 12 levels are
# averaged with equal weight 1/12.

# The 12 standard M5 aggregation levels, expressed as the id columns to group by.
# `[]` means the grand total (one series for the whole dataset).
M5_LEVELS = [
    [],                                   # 1  total
    ["state_id"],                         # 2
    ["store_id"],                         # 3
    ["cat_id"],                           # 4
    ["dept_id"],                          # 5
    ["state_id", "cat_id"],               # 6
    ["state_id", "dept_id"],              # 7
    ["store_id", "cat_id"],               # 8
    ["store_id", "dept_id"],              # 9
    ["item_id"],                          # 10
    ["item_id", "state_id"],              # 11
    ["item_id", "store_id"],              # 12  bottom level
]


class WRMSSEEvaluator:
    """Compute WRMSSE over the 12 M5 levels from a long-format panel.

    Parameters
    ----------
    train : DataFrame
        Long panel with columns: date, the id columns
        (item_id, dept_id, cat_id, store_id, state_id), `units`, and `revenue`
        (= units * sell_price). Covers the training period only.
    valid : DataFrame
        Same schema, covering the forecast horizon (the held-out window) with the
        true `units`.
    id_cols : list[str]
        The bottom-level identifier columns. Default matches M5.
    """

    def __init__(self, train: pd.DataFrame, valid: pd.DataFrame,
                 id_cols=("item_id", "dept_id", "cat_id", "store_id", "state_id")):
        self.id_cols = list(id_cols)
        self.train = train
        self.valid = valid
        self.horizon = valid["date"].nunique()
        self._build_level_cache()

    def _series_key(self, df, group):
        if len(group) == 0:
            return pd.Series(["Total"] * len(df), index=df.index)
        return df[group].astype(str).agg("--".join, axis=1)

    def _build_level_cache(self):
        self.level_cache = []
        # M5 weight denominator: total dollar sales over the LAST 28 training days,
        # summed across the bottom-level series. The same denominator is reused for
        # every aggregation level, so each level's weights sum to 1.
        last28_cut = self.train["date"].max() - pd.Timedelta(days=27)
        total_dollar = self.train.loc[self.train["date"] >= last28_cut, "revenue"].sum()
        for group in M5_LEVELS:
            tr = self.train.copy()
            tr["__key"] = self._series_key(tr, group)
            # aggregate units to this level, ordered by date
            agg = (tr.groupby(["__key", "date"])["units"].sum()
                     .unstack(fill_value=0).sort_index(axis=1)
                     .astype(np.float64))  # float: avoid int overflow in diff**2
            # scaling denominator: in-sample mean squared 1-step diff per series
            diffs = agg.diff(axis=1).iloc[:, 1:].values
            denom = np.sqrt(np.nanmean(diffs ** 2, axis=1))
            denom = np.where(denom == 0, np.nan, denom)
            # weight per series = its share of total dollar sales (last 28 train days)
            last28 = (tr[tr["date"] >= tr["date"].max() - pd.Timedelta(days=27)]
                      .groupby("__key")["revenue"].sum())
            last28 = last28.reindex(agg.index).fillna(0.0)
            weight = (last28 / total_dollar).values
            self.level_cache.append({
                "group": group, "keys": agg.index.values,
                "denom": denom, "weight": weight,
            })

    def score(self, pred: pd.DataFrame) -> dict:
        """Score a long-format prediction frame (same id cols + date + `pred`).

        Returns a dict with per-level RMSSE-weighted error and the final WRMSSE.
        """
        out = {}
        level_scores = []
        for level in self.level_cache:
            group = level["group"]
            v = self.valid.copy()
            p = pred.copy()
            v["__key"] = self._series_key(v, group)
            p["__key"] = self._series_key(p, group)
            v["units"] = v["units"].astype(np.float64)
            p["pred"] = p["pred"].astype(np.float64)
            y = v.groupby(["__key", "date"])["units"].sum().unstack(fill_value=0)
            yhat = p.groupby(["__key", "date"])["pred"].sum().unstack(fill_value=0)
            y = y.reindex(level["keys"]).fillna(0.0)
            yhat = yhat.reindex(index=level["keys"], columns=y.columns).fillna(0.0)
            mse_h = np.mean((y.values - yhat.values) ** 2, axis=1)
            rmsse = np.sqrt(mse_h) / level["denom"]
            w = level["weight"]
            mask = ~np.isnan(rmsse) & ~np.isnan(w)
            lvl = float(np.sum(rmsse[mask] * w[mask]))
            level_scores.append(lvl)
            key = "L" + str(len(group)) + "_" + ("+".join(group) if group else "total")
            out[key] = round(lvl, 5)
        out["WRMSSE"] = round(float(np.mean(level_scores)), 5)
        return out


def rmsse(train_y, y_true, y_pred) -> float:
    """Single-series RMSSE (convenience helper)."""
    train_y = np.asarray(train_y, dtype=float)
    diffs = np.diff(train_y)
    denom = np.sqrt(np.mean(diffs ** 2)) if len(diffs) else np.nan
    if not denom or np.isnan(denom):
        return np.nan
    num = np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))
    return float(num / denom)


# --------------------------------------------------------------------------- #
# 3. Probabilistic metrics
# --------------------------------------------------------------------------- #
def pinball_loss(y_true, y_pred_q, q: float) -> float:
    """Pinball (quantile) loss for a single quantile level q in (0, 1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    e = y_true - y_pred_q
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def multi_quantile_pinball(y_true, preds_by_q: dict) -> dict:
    """preds_by_q: {q: array_of_predictions}. Returns per-q loss and the mean."""
    out = {f"pinball@{q:.3f}": round(pinball_loss(y_true, p, q), 5)
           for q, p in preds_by_q.items()}
    out["mean_pinball"] = round(float(np.mean(list(out.values()))), 5)
    return out


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage of a prediction interval [lower, upper]."""
    y_true = np.asarray(y_true, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# --------------------------------------------------------------------------- #
# 4. Inventory / policy metrics
# --------------------------------------------------------------------------- #
def fill_rate(demand, fulfilled) -> float:
    """Units-based fill rate = fulfilled / demanded."""
    d = np.sum(np.asarray(demand, float))
    return float(np.sum(np.asarray(fulfilled, float)) / d) if d > 0 else np.nan


def cost_breakdown(holding, stockout, ordering, transship=0.0) -> dict:
    total = holding + stockout + ordering + transship
    return {
        "holding": round(float(holding), 2),
        "stockout": round(float(stockout), 2),
        "ordering": round(float(ordering), 2),
        "transship": round(float(transship), 2),
        "total": round(float(total), 2),
    }
