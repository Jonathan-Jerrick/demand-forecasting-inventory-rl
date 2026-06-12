"""Regenerate all README/RESULTS figures from committed result tables.

Reads only files in data/ (committed) so anyone who clones the repo can
reproduce all figures without the raw M5 data or a GPU:

    python -m src.make_figures        # writes figures/*.png

Figures produced:
  1. figures/cost_fill_frontier.png     cost vs fill-rate per policy, oracle floor
  2. figures/cost_decomposition.png     holding vs stockout cost per policy
  3. figures/ablation_edge.png          PPO edge % under feature/pooling ablations
  4. figures/rmsse_distribution.png     per-series RMSSE histogram across 10k series
  5. figures/demand_segments.png        intermittency map (CV vs zero-fraction by segment)
  6. figures/quantile_coverage.png      empirical vs nominal coverage — LightGBM & GRU
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

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

_LABEL_OFFSET = {
    "Oracle": (8, -4),
    "PPO-Residual(ours)": (-2, 16),
    "Newsvendor": (8, -20),
    "Base-Stock(forecast)": (6, 8),
}


def _load_rl() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "rl_results.parquet")
    return df.loc[[p for p, _ in ORDER if p in df.index]]


# ── 1. Cost / fill-rate frontier ─────────────────────────────────────────────

def frontier(df: pd.DataFrame) -> None:
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


# ── 2. Cost decomposition ─────────────────────────────────────────────────────

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


# ── 3. Ablation study ────────────────────────────────────────────────────────

def ablation_edge() -> None:
    df = pd.read_parquet(ROOT / "data" / "ablations.parquet")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["#d9534f" if v < 0 else "#1b6ec2" for v in df["edge_pct"]]
    bars = ax.barh(df.index[::-1], df["edge_pct"][::-1], color=colors[::-1], alpha=0.88)
    ax.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, df["edge_pct"][::-1]):
        xpos = val + (0.3 if val >= 0 else -0.3)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", ha=ha, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("PPO edge over Newsvendor  (% cost reduction, higher = better)")
    ax.set_title("Ablation study — what drives the RL agent's edge")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(df["edge_pct"].min() - 5, df["edge_pct"].max() + 5)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_edge.png", dpi=150)
    plt.close(fig)


# ── 4. RMSSE distribution ─────────────────────────────────────────────────────

def rmsse_distribution() -> None:
    df = pd.read_parquet(ROOT / "data" / "series_scores.parquet")
    rmsse = df["rmsse"].clip(upper=3.0)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(rmsse, bins=80, color="#1b6ec2", alpha=0.82, edgecolor="white", lw=0.3)
    med = df["rmsse"].median()
    ax.axvline(med, color="#e8590c", lw=1.8, ls="--", label=f"Median {med:.3f}")
    ax.axvline(1.0, color="#6c757d", lw=1.2, ls=":", label="RMSSE = 1 (naïve baseline)")
    pct_below = (df["rmsse"] < 1.0).mean() * 100
    ax.text(0.98, 0.95, f"{pct_below:.0f}% of series\nbeat naïve baseline",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))
    ax.set_xlabel("RMSSE  (lower is better; 1.0 = seasonal naïve)")
    ax.set_ylabel("Number of series")
    ax.set_title("Forecast accuracy distribution across 10,000 M5 series (LightGBM-Tweedie)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "rmsse_distribution.png", dpi=150)
    plt.close(fig)


# ── 5. Demand intermittency map ───────────────────────────────────────────────

def demand_segments() -> None:
    df = pd.read_parquet(ROOT / "data" / "segments.parquet")
    seg_colors = {
        "fast-moving": "#1b6ec2",
        "intermittent": "#e8590c",
        "lumpy": "#f08c00",
        "slow-moving": "#2f9e44",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for seg, grp in df.groupby("segment"):
        ax.scatter(grp["zero_frac"], grp["cv"],
                   c=seg_colors.get(seg, "#aaa"), alpha=0.18, s=8,
                   label=f"{seg}  (n={len(grp):,})", rasterized=True)
    ax.axvline(0.5, color="#888", lw=0.9, ls="--", alpha=0.6)
    ax.axhline(0.5, color="#888", lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel("Zero-demand fraction  (intermittency)")
    ax.set_ylabel("Coefficient of variation  (volatility)")
    ax.set_title("Demand characterisation — 10,000 M5 series")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, min(df["cv"].quantile(0.99) * 1.1, 6))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, markerscale=2.5, fontsize=9)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG / "demand_segments.png", dpi=150)
    plt.close(fig)


# ── 6. Quantile calibration (coverage) ───────────────────────────────────────

def quantile_coverage() -> None:
    nominal = [0.1, 0.25, 0.5, 0.75, 0.9]
    results: dict[str, list[float]] = {}

    for label, path in [("LightGBM-Quantile", ROOT / "data" / "lgbm_forecast.parquet"),
                         ("GRU-Quantile", ROOT / "data" / "neural_forecast.parquet")]:
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            continue
        actual = df["units"].values
        coverage = []
        for q in nominal:
            pred = df[f"q{q}"].values
            if q <= 0.5:
                coverage.append((actual <= pred).mean())
            else:
                coverage.append((actual <= pred).mean())
        results[label] = coverage

    if not results:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    markers = ["o", "s"]
    model_colors = {"LightGBM-Quantile": "#1b6ec2", "GRU-Quantile": "#e8590c"}
    for (label, cov), mk in zip(results.items(), markers):
        ax.plot(nominal, cov, mk + "-", ms=8, lw=1.8,
                color=model_colors[label], label=label, alpha=0.9)
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage (fraction of actuals ≤ predicted)")
    ax.set_title("Quantile calibration — LightGBM vs GRU (28-day holdout)")
    ax.set_xlim(-0.02, 0.98)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "quantile_coverage.png", dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = _load_rl()
    frontier(df)
    decomposition(df)
    ablation_edge()
    rmsse_distribution()
    demand_segments()
    quantile_coverage()
    figs = sorted(FIG.glob("*.png"))
    print(f"wrote {len(figs)} figures to {FIG}:")
    for f in figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
