"""Regenerate the README/RESULTS figures from committed result tables.

Reads only `data/rl_results.parquet` (committed, ~8 KB) so anyone who clones the
repo can reproduce the figures without the raw M5 data or a GPU:

    python -m src.make_figures        # writes figures/*.png

Two figures:
  1. figures/cost_fill_frontier.png   cost vs fill-rate per policy, oracle floor
  2. figures/cost_decomposition.png   holding vs stockout cost per policy
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Policies to show, in a stable order, with display labels.
ORDER = [
    ("Oracle", "Oracle\n(clairvoyant floor)"),
    ("PPO-Residual(ours)", "PPO-Residual\n(ours)"),
    ("Newsvendor", "Newsvendor"),
    ("Base-Stock(forecast)", "Base-Stock"),
    ("Fixed-Order", "Fixed-Order"),
]
COLORS = {
    "Oracle": "#6c757d",
    "PPO-Residual(ours)": "#1b6ec2",
    "Newsvendor": "#e8590c",
    "Base-Stock(forecast)": "#2f9e44",
    "Fixed-Order": "#adb5bd",
}


def _load() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "rl_results.parquet")
    return df.loc[[p for p, _ in ORDER if p in df.index]]


# Label offsets (points) to keep the near-optimal cluster legible.
_LABEL_OFFSET = {
    "Oracle": (8, -4),
    "PPO-Residual(ours)": (-2, 16),
    "Newsvendor": (8, -20),
    "Base-Stock(forecast)": (6, 8),
}


def frontier(df: pd.DataFrame) -> None:
    # Fixed-Order's variance dwarfs the rest; show the near-optimal frontier only.
    keep = [p for p in df.index if p != "Fixed-Order"]
    sub = df.loc[keep]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for pol, label in ORDER:
        if pol not in sub.index:
            continue
        r = sub.loc[pol]
        ax.errorbar(
            r["fill_rate"], r["cost"] / 1e6,
            xerr=r["fill_std"], yerr=r["cost_std"] / 1e6,
            fmt="o", ms=12, capsize=4, lw=1.3,
            color=COLORS[pol], ecolor=COLORS[pol], alpha=0.9, zorder=3,
        )
        dx, dy = _LABEL_OFFSET.get(pol, (10, 6))
        ax.annotate(
            label.replace("\n", " "), (r["fill_rate"], r["cost"] / 1e6),
            textcoords="offset points", xytext=(dx, dy), fontsize=9.5,
            color=COLORS[pol], fontweight="bold",
        )
    floor = sub.loc["Oracle", "cost"] / 1e6
    ax.axhline(floor, ls="--", lw=1, color="#6c757d", alpha=0.6, zorder=1)
    ax.text(sub["fill_rate"].min(), floor, " oracle cost floor",
            va="bottom", ha="left", fontsize=8, color="#6c757d")
    ax.set_xlabel("Fill rate  (higher is better service)")
    ax.set_ylabel("Total 90-day cost  (millions, lower is better)")
    ax.set_title("Cost / fill-rate frontier — paired multi-seed (10k-series M5 twin)")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.18, y=0.18)
    fig.tight_layout()
    fig.savefig(FIG / "cost_fill_frontier.png", dpi=150)
    plt.close(fig)


def decomposition(df: pd.DataFrame) -> None:
    labels = [lbl.replace("\n", " ") for p, lbl in ORDER if p in df.index]
    holding = df["holding"].values / 1e6
    stockout = df["stockout"].values / 1e6
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = range(len(df))
    ax.bar(x, holding, label="Holding cost", color="#1b6ec2", alpha=0.85)
    ax.bar(x, stockout, bottom=holding, label="Stockout cost",
           color="#e8590c", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cost  (millions)")
    ax.set_title("Where the cost goes — holding vs stockout (10k-series M5 twin)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cost_decomposition.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = _load()
    frontier(df)
    decomposition(df)
    print(f"wrote {FIG/'cost_fill_frontier.png'} and {FIG/'cost_decomposition.png'}")


if __name__ == "__main__":
    main()
