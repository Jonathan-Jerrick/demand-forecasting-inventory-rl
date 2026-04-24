"""Tiny experiment logger -- appends one row per run to data/experiments.csv.

Keeps results reproducible and comparable across notebook runs (forecasters and
inventory policies alike). No dependency beyond pandas.
"""

from __future__ import annotations

from pathlib import Path
import datetime as _dt

import pandas as pd


def log_experiment(path, run: str, **metrics) -> pd.DataFrame:
    """Append a run's metrics to the CSV at `path` and return the full table.

    `run` is a short label (e.g. "T1_lgbm_quantile"); `metrics` are flat scalars.
    A UTC timestamp is added automatically. Existing columns are unioned so runs
    with different metric sets coexist.
    """
    path = Path(path)
    row = {"ts": _dt.datetime.utcnow().isoformat(timespec="seconds"), "run": run, **metrics}
    df_new = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    else:
        df = df_new
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
