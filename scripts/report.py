"""Portfolio report — metrics and figures from saved test results.

Loads the four ETHUSDT test runs available under result/high_level/ETHUSDT/:
the MacroHFT baseline and three Dynamic Hybrid variants (default / aggressive
/ conservative). Computes the 19-metric performance table reported in the
team paper (Table 1) and writes four publication-style figures to
docs/figures/.

Run with `uv run scripts/report.py` — PEP 723 metadata below pins deps.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
# ]
# ///

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "result" / "high_level" / "ETHUSDT"
FIGURES_DIR = REPO / "docs" / "figures"
METRICS_CSV = REPO / "docs" / "metrics.csv"

MINUTES_PER_YEAR = 60 * 24 * 365

RUNS: dict[str, Path] = {
    "MacroHFT (baseline)": RESULTS / "exp1" / "seed_12345",
    "Dynamic Hybrid (default)": (
        RESULTS / "dynamic_mixing_default" / "seed_12345" / "best_epoch_12_test_results"
    ),
    "Dynamic Hybrid (aggressive)": (
        RESULTS / "dynamic_mixing_aggressive" / "seed_12345" / "best_epoch_4_test_results"
    ),
    "Dynamic Hybrid (conservative)": (
        RESULTS / "dynamic_mixing_conservative" / "seed_12345" / "best_epoch_5_test_results"
    ),
}

# Numbers from the team report (Table 1) — used as a sanity check on stdout.
PAPER_REFERENCE = {
    "MacroHFT (baseline)": {"Total Return": 0.06, "Sharpe Ratio": 0.867, "Max Drawdown": 0.089},
    "Dynamic Hybrid (default)": {"Total Return": 0.38662, "Sharpe Ratio": 3.643, "Max Drawdown": 0.0977},
}


@dataclass
class Run:
    name: str
    action: np.ndarray
    reward: np.ndarray
    profit: float
    capital: float
    fees: float


def _scalar(path: Path) -> float:
    return float(np.load(path).reshape(-1)[0])


def load_run(name: str, path: Path) -> Run:
    return Run(
        name=name,
        action=np.load(path / "action.npy").squeeze(),
        reward=np.load(path / "reward.npy").squeeze(),
        profit=_scalar(path / "final_balance.npy"),
        capital=_scalar(path / "require_money.npy"),
        fees=_scalar(path / "commission_fee_history.npy"),
    )


def trade_pnls(action: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Per-trade PnL — sum of rewards during each contiguous long position."""
    in_pos = (action == 1).astype(np.int8)
    padded = np.concatenate([[0], in_pos, [0]])
    edges = np.diff(padded)
    entries = np.where(edges == 1)[0]
    exits = np.where(edges == -1)[0]
    return np.array([reward[e:x].sum() for e, x in zip(entries, exits)])


def _max_run_length(flags: np.ndarray) -> int:
    if not flags.any():
        return 0
    padded = np.concatenate([[0], flags.astype(int), [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int((ends - starts).max())


def compute_metrics(r: Run) -> dict[str, float]:
    n = len(r.reward)
    pnls = trade_pnls(r.action, r.reward)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_return = r.profit / r.capital
    ann_return = (1 + total_return) ** (MINUTES_PER_YEAR / n) - 1

    mu = float(r.reward.mean())
    sigma = float(r.reward.std(ddof=1))
    downside = r.reward[r.reward < 0]
    sigma_down = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sqrt_year = math.sqrt(MINUTES_PER_YEAR)
    sharpe = (mu / sigma) * sqrt_year if sigma > 0 else float("nan")
    sortino = (mu / sigma_down) * sqrt_year if sigma_down and sigma_down > 0 else float("nan")

    equity = r.capital + np.cumsum(r.reward)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = float(-dd.min())
    max_dd_dur = _max_run_length(dd < 0)
    calmar = ann_return / max_dd if max_dd > 0 else float("nan")

    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() != 0 else float("nan")
    expectancy = float(pnls.mean()) if len(pnls) else float("nan")
    sqn_std = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    sqn = math.sqrt(len(pnls)) * expectancy / sqn_std if sqn_std > 0 else float("nan")

    return {
        "Total Trades": int(len(pnls)),
        "Profitable Trades": int(len(wins)),
        "Losing Trades": int(len(losses)),
        "Win Rate": float(len(wins) / len(pnls)) if len(pnls) else float("nan"),
        "Avg Profit/Trade": expectancy,
        "Avg Win": float(wins.mean()) if len(wins) else float("nan"),
        "Avg Loss": float(losses.mean()) if len(losses) else float("nan"),
        "Profit Factor": pf,
        "Expectancy": expectancy,
        "SQN": sqn,
        "Total Profit": r.profit,
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Max Drawdown Duration": max_dd_dur,
        "Calmar Ratio": calmar,
        "Total Fees": r.fees,
    }


def _palette(runs: Iterable[Run]) -> dict[str, str]:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    return {r.name: colors[i % len(colors)] for i, r in enumerate(runs)}


def plot_cumulative_returns(runs: list[Run], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = _palette(runs)
    for r in runs:
        equity = r.capital + np.cumsum(r.reward)
        pct = (equity - r.capital) / r.capital * 100
        ax.plot(pct, label=r.name, linewidth=1.4, color=colors[r.name])
    ax.set_title("Cumulative return — ETHUSDT test set (minute-level)")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Return (%)")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_drawdown(runs: list[Run], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = _palette(runs)
    for r in runs:
        equity = r.capital + np.cumsum(r.reward)
        dd = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        ax.plot(dd, label=r.name, linewidth=1.2, color=colors[r.name])
    ax.set_title("Drawdown — ETHUSDT test set")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_trade_pnl(runs: list[Run], out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        pnls = trade_pnls(r.action, r.reward)
        if len(pnls):
            colors_per_bar = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors_per_bar, edgecolor="black", linewidth=0.3)
            ax.set_xticks(range(0, len(pnls), max(1, len(pnls) // 8)))
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(r.name, fontsize=10)
        ax.set_xlabel("Trade #")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Trade PnL")
    fig.suptitle("Per-trade PnL", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_minute_returns(runs: list[Run], out: Path) -> None:
    """Per-minute reward distribution while in position (log y to reveal tails)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = _palette(runs)
    active = [r.reward[r.action == 1] for r in runs]
    # Clip range to 1st–99th percentile across all runs so tails dominate the plot.
    flat = np.concatenate(active)
    lo, hi = np.percentile(flat, [1, 99])
    bins = np.linspace(lo, hi, 81)
    for r, a in zip(runs, active):
        ax.hist(
            a,
            bins=bins,
            histtype="step",
            label=f"{r.name} (n={len(a):,})",
            color=colors[r.name],
            linewidth=1.5,
            density=True,
        )
    ax.set_yscale("log")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.6)
    ax.set_title("Per-minute reward distribution — in-position only (log y)")
    ax.set_xlabel("Per-minute reward")
    ax.set_ylabel("Density (log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fmt(name: str, value: float) -> str:
    if name in {"Total Trades", "Profitable Trades", "Losing Trades", "Max Drawdown Duration"}:
        return f"{int(value)}"
    if name in {"Win Rate", "Total Return", "Annualized Return", "Max Drawdown"}:
        return f"{value:.4f}"
    return f"{value:,.4f}"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for name, path in RUNS.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        runs.append(load_run(name, path))

    if not runs:
        raise SystemExit("no runs found")

    metrics = {r.name: compute_metrics(r) for r in runs}
    df = pd.DataFrame(metrics)

    display = pd.DataFrame(
        {col: [_fmt(idx, df.at[idx, col]) for idx in df.index] for col in df.columns},
        index=df.index,
    )
    print(display.to_string())
    print()

    df.to_csv(METRICS_CSV)
    print(f"wrote {METRICS_CSV.relative_to(REPO)}")

    plot_cumulative_returns(runs, FIGURES_DIR / "cumulative_returns.png")
    plot_drawdown(runs, FIGURES_DIR / "drawdown.png")
    plot_trade_pnl(runs, FIGURES_DIR / "trade_pnl.png")
    plot_minute_returns(runs, FIGURES_DIR / "minute_returns.png")
    print(f"wrote 4 figures to {FIGURES_DIR.relative_to(REPO)}/")

    print("\nSanity check vs. team paper Table 1:")
    print(f"{'strategy':<32} {'metric':<14} {'computed':>10} {'paper':>10}  diff")
    for strategy, expected in PAPER_REFERENCE.items():
        if strategy not in metrics:
            continue
        for k, v in expected.items():
            got = metrics[strategy][k]
            diff = got - v
            marker = "OK" if abs(diff) < 0.01 else "!!"
            print(f"  {strategy:<30} {k:<14} {got:>10.4f} {v:>10.4f}  {diff:+.4f} {marker}")


if __name__ == "__main__":
    main()
