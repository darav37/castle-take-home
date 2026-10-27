"""Charts saved as PNGs (the chart tool returns the path)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "out" / "charts"


def payoff(strategy, dist=None, name: str = "payoff") -> str:
    """P&L at first expiry, with the user's view density shaded underneath."""
    S = strategy.S
    g = np.linspace(S * 0.7, S * 1.3, 301)
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    ax.plot(g, strategy.payoff_at(g), lw=2, label=strategy.name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(S, color="grey", lw=0.8, ls=":")
    if dist is not None:
        mu = np.log(1 + dist.mean_move) - 0.5 * dist.sd ** 2
        dens = norm.pdf((np.log(g / S) - mu) / dist.sd) / (g * dist.sd)
        ax2 = ax.twinx()
        ax2.fill_between(g, dens, alpha=0.15, color="tab:orange", label="your view")
        ax2.set_yticks([])
    ax.set_xlabel("price at expiry")
    ax.set_ylabel("P&L ($)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def scenario_bars(rows: list[dict], name: str = "scan") -> str:
    """P&L if YES by asset class, one bar group per event."""
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    classes = sorted({c for r in rows for c in r["pnl_yes_by_class"]})
    w = 0.8 / max(len(classes), 1)
    for j, c in enumerate(classes):
        ax.bar([i + j * w for i in range(len(rows))], [r["pnl_yes_by_class"].get(c, 0) for r in rows], w, label=c)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(rows))], [r["event_id"] for r in rows], rotation=20, fontsize=7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("P&L if YES ($)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)
