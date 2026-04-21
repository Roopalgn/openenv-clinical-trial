"""plot_rewards.py — Visualise per-episode reward curves from reward_log.csv.

Usage:
    python plot_rewards.py --csv logs/reward_log.csv --out reward_curve.png

Outputs a single PNG with:
  • Scatter of per-episode total reward
  • Rolling average (window = min(20, n//5))
  • Linear trend line with slope annotation
  • Stats box: best / mean / final reward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-episode reward curve from reward_log.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("logs/reward_log.csv"),
        help="Path to reward_log.csv (default: logs/reward_log.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reward_curve.png"),
        help="Output PNG path (default: reward_curve.png)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[plot_rewards] ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if "total_reward" not in df.columns:
        print(
            f"[plot_rewards] ERROR: 'total_reward' column missing in {csv_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    df = df.reset_index(drop=True)
    df["episode_num"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(df: pd.DataFrame, out_path: Path) -> None:
    rewards = df["total_reward"].to_numpy(dtype=float)
    episodes = df["episode_num"].to_numpy(dtype=float)
    n = len(rewards)

    window = max(2, min(20, n // 5))
    rolling = (
        pd.Series(rewards)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )

    # Linear trend
    coeffs = np.polyfit(episodes, rewards, 1)
    trend = np.polyval(coeffs, episodes)
    slope = coeffs[0]

    # Stats
    best_r = float(np.max(rewards))
    mean_r = float(np.mean(rewards))
    final_r = float(rewards[-1])

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Scatter
    ax.scatter(
        episodes,
        rewards,
        s=18,
        alpha=0.55,
        color="#4c9be8",
        zorder=2,
        label="Episode reward",
    )

    # Rolling average
    ax.plot(
        episodes,
        rolling,
        color="#f0a500",
        linewidth=2.0,
        zorder=3,
        label=f"Rolling avg (w={window})",
    )

    # Trend line
    ax.plot(
        episodes,
        trend,
        color="#e05c5c",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
        label=f"Trend (slope={slope:+.4f}/ep)",
    )

    # Stats box
    stats_text = (
        f"  Best : {best_r:+.3f}\n"
        f"  Mean : {mean_r:+.3f}\n"
        f"  Final: {final_r:+.3f}"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1e2130",
            edgecolor="#4c9be8",
            alpha=0.85,
        ),
        family="monospace",
    )

    # Axes styling
    ax.set_xlabel("Episode", color="white", fontsize=11)
    ax.set_ylabel("Total Reward", color="white", fontsize=11)
    ax.set_title("Training Reward Curve", color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333a4a")
    ax.grid(True, color="#1e2130", linewidth=0.8, zorder=0)

    ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor="#1e2130",
        edgecolor="#4c9be8",
        labelcolor="white",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot_rewards] Saved → {out_path}  ({n} episodes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    df = _load(args.csv)
    _plot(df, args.out)


if __name__ == "__main__":
    main()
