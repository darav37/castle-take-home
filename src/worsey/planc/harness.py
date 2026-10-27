"""Plan C's frozen test harness (C0; docs/research/plan_c.md §7a, DECISIONS #187). Every rung is scored here, the same way.

A prediction is, per event and target, the 19 quantiles at TAUS (5%, 10%, ..., 95%) of the move over the horizon, in the
target's own units. Scores are standardized by the target's normal noise over the same window (its daily sd over the 504
sessions before the event x sqrt(window sessions)), so oil doesn't drown out credit and factors compare with ETFs.

Frozen splits (by first-report date): train < 2023-01-01; validation 2023-01-01 .. 2025-12-31; test >= 2026-01-01
(after the LLM's training cutoff; check against the text embedder's cutoff too), plus the prospective log as it grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)          # 19 levels
SPLITS = {"train": ("1900-01-01", "2023-01-01"), "validation": ("2023-01-01", "2026-01-01"), "test": ("2026-01-01", "2100-01-01")}
LO90, HI90 = 0, len(TAUS) - 1                              # 5% and 95%
LO50, HI50 = 4, 14                                         # 25% and 75%
MID = 9                                                    # 50%


def split_of(date: str) -> str:
    for k, (a, b) in SPLITS.items():
        if a <= date < b:
            return k
    raise ValueError(date)


def quantiles_from_samples(x: np.ndarray) -> np.ndarray:
    return np.quantile(x, TAUS)


def quantiles_normal(mean: float, sd: float) -> np.ndarray:
    from scipy.stats import norm
    return mean + sd * norm.ppf(TAUS)


def crps_q(q: np.ndarray, y: float) -> float:
    """CRPS approximated by the average pinball loss over the 19 quantile levels (x2)."""
    q = np.sort(np.asarray(q, float))
    d = y - q
    return float(2 * np.mean(np.maximum(TAUS * d, (TAUS - 1) * d)))


def score_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """rows: one per (event, target) with columns q (19 quantiles), y (real move), noise_sd. Adds standardized CRPS,
    90%/50% coverage, and the median."""
    out = rows.copy()
    out["crps_std"] = [crps_q(q, y) / s for q, y, s in zip(out.q, out.y, out.noise_sd)]
    out["in90"] = [q[LO90] <= y <= q[HI90] for q, y in zip(out.q, out.y)]
    out["in50"] = [q[LO50] <= y <= q[HI50] for q, y in zip(out.q, out.y)]
    out["below5"] = [y < q[LO90] for q, y in zip(out.q, out.y)]
    out["median"] = [q[MID] for q in out.q]
    return out


def summary(scored: pd.DataFrame) -> dict:
    y, m = scored.y.to_numpy(), scored["median"].to_numpy()
    return {"events": int(scored.event.nunique()), "rows": int(len(scored)), "crps_std": float(scored.crps_std.mean()),
            "coverage_90": float(scored.in90.mean()), "coverage_50": float(scored.in50.mean()), "below_5": float(scored.below5.mean()),
            "r2_median_vs_zero": float(1 - ((y - m) ** 2).sum() / (y ** 2).sum()) if (y ** 2).sum() > 0 else None}


def paired_better(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 1729) -> dict:
    """Is rung a better than rung b (lower standardized CRPS)? Paired bootstrap over events on the rows both scored.
    Returns the mean difference (b - a; > 0 means a is better), its 90% range and the share of resamples where a wins."""
    k = ["event", "target"]
    m = a[k + ["crps_std"]].merge(b[k + ["crps_std"]], on=k, suffixes=("_a", "_b"))
    per_event = (m.crps_std_b - m.crps_std_a).groupby(m.event).agg(["sum", "count"])
    s, c = per_event["sum"].to_numpy(), per_event["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return {"b_minus_a": float(s.sum() / c.sum()), "range_90": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))],
            "a_better_support": float(np.mean(boots > 0)), "events": int(len(s)), "rows": int(len(m))}
