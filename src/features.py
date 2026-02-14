"""Feature engineering shared by the preprocessing notebook.

Builds the lag / rolling / calendar / price features used by both the
gradient-boosted baseline and the neural forecaster. Pure pandas/numpy so it is
testable on a subset without a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 21, 28, 35]
ROLL_WINDOWS = [7, 14, 28]


def add_lag_features(df: pd.DataFrame, group_col: str = "series_id",
                     target: str = "units") -> pd.DataFrame:
    """Per-series demand lags and rolling mean/std (computed on lag-1 to avoid leakage)."""
    df = df.sort_values([group_col, "date"]).copy()
    g = df.groupby(group_col)[target]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    base = g.shift(1)
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = base.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"rstd_{w}"] = base.rolling(w).std().reset_index(level=0, drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar signals derived from the `date` column."""
    d = df["date"].dt
    df["day_of_week"] = d.dayofweek
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["day_of_month"] = d.day
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["is_month_end"] = d.is_month_end.astype(int)
    # SNAP (food stamp) disbursements and payday cycles cluster on the 1st-3rd and
    # 28th-31st of each month in Walmart POS data — visible in lag-1 residuals for
    # FOODS series. A rolling mean washes this out; the binary flag preserves it.
    df["is_payday_window"] = ((d.day <= 3) | (d.day >= 28)).astype(int)
    return df


def add_price_features(df: pd.DataFrame, group_col: str = "series_id") -> pd.DataFrame:
    """Price dynamics: relative price, change, trailing mean, promo flag."""
    df = df.sort_values([group_col, "date"]).copy()
    g = df.groupby(group_col)["sell_price"]
    df["price_lag_7"] = g.shift(7)
    df["price_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    df["price_ratio"] = df["sell_price"] / df["price_mean_28"].replace(0, np.nan)
    df["price_change"] = df["sell_price"] / g.shift(1).replace(0, np.nan) - 1
    df["on_promo"] = (df["price_ratio"] < 0.98).astype(int)
    df["revenue"] = df["units"] * df["sell_price"].fillna(0)
    return df


def encode_ids(df: pd.DataFrame, cols=("item_id", "dept_id", "cat_id",
                                       "store_id", "state_id")) -> pd.DataFrame:
    """Integer-encode categorical id columns for tree / embedding models."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_enc"] = df[c].astype("category").cat.codes
    return df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes to keep the full-scale panel in memory."""
    for c in df.select_dtypes("float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df
