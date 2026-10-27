"""Probabilities from markets (SPEC §3.2 steps 3–4).

All the arithmetic for 'derived' matches lives here: strike-ladder interpolation and
constant-hazard time rescaling. Claude decides *that* a match is derived; code computes p.
"""

from __future__ import annotations

import math

import numpy as np


def market_prob(bid: float | None, ask: float | None, last: float | None, wide: float | None = None) -> dict:
    """Mid if the spread is tight, last trade if wide (METHODOLOGY §7; wide = config.SPREAD_USE_LAST)."""
    from .. import config
    wide = config.SPREAD_USE_LAST if wide is None else wide
    if bid is not None and ask is not None and ask >= bid:
        spread = ask - bid
        if spread <= wide or last is None:
            return {"p": (bid + ask) / 2, "source": "mid", "spread": spread, "confidence": "high" if spread <= 0.04 else "med"}
        return {"p": last, "source": "last", "spread": spread, "confidence": "low"}
    if last is not None:
        return {"p": last, "source": "last", "spread": None, "confidence": "low"}
    raise ValueError("no quote")


def combine(quotes: list[dict]) -> float:
    """Liquidity-weighted average of several venues' p (each dict: p, liquidity)."""
    w = np.array([max(q.get("liquidity", 0.0), 0.0) for q in quotes])
    p = np.array([q["p"] for q in quotes])
    if w.sum() == 0:
        return float(p.mean())
    return float((w * p).sum() / w.sum())


def combine_venues(quotes: list[dict], gap_flag: float | None = None) -> dict:
    """Both venues' p, the liquidity-weighted p, and a 'venues_disagree' flag past 5pp (METHODOLOGY §7; C16).

    quotes: [{venue, p, liquidity}]. A big cross-venue gap is itself information, so it is shown, not averaged away.
    """
    from .. import config
    gap_flag = config.VENUE_GAP_FLAG if gap_flag is None else gap_flag
    ps = [q["p"] for q in quotes]
    gap = max(ps) - min(ps) if len(ps) > 1 else 0.0
    return {"p": combine(quotes), "by_venue": {q["venue"]: q["p"] for q in quotes}, "gap": gap,
            "flags": ["venues_disagree"] if gap > gap_flag else []}


def rescale_meetings(p: float, n: int, n_new: int) -> float:
    """Release-driven events (METHODOLOGY §6): per-decision probability q = 1 − (1−p)^(1/n), then
    p′ = 1 − (1−q)^(n′). Counts decision dates, not calendar days ("Fed cuts by December")."""
    if n <= 0:
        raise ValueError("need at least one decision date in the market's window")
    q = 1 - (1 - p) ** (1 / n)
    return 1 - (1 - q) ** n_new


def apply_time_rule(p: float, rule: dict) -> float:
    """p for the wanted deadline, from a matching result's time_rule."""
    if rule["kind"] == "meetings":
        return rescale_meetings(p, rule["n"], rule["n_new"])
    return rescale_time(p, rule["T"], rule["T_new"])


def rescale_time(p: float, T: float, T_new: float) -> float:
    """p′ = 1 − (1−p)^(T′/T): 'happens by date' under a constant hazard."""
    if T <= 0:
        raise ValueError("T must be positive")
    return 1 - (1 - p) ** (T_new / T)


def enforce_ladder(strikes: list[float], probs: list[float], comparator: str = ">=") -> list[float]:
    """Make a strike ladder coherent: P(X ≥ k) is non-increasing in k (non-decreasing for ≤).

    Pool-adjacent-violators (isotonic regression), so incoherent quotes are averaged, not dropped.
    """
    order = np.argsort(strikes)
    y = np.asarray(probs, float)[order]
    sign = -1 if comparator in (">=", ">") else 1
    y = sign * y
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = np.concatenate([[v] * n for v, n in blocks]) * sign
    out = np.empty_like(fitted)
    out[order] = fitted
    return out.tolist()


def ladder_interp(strikes: list[float], probs: list[float], x: float, comparator: str = ">=") -> dict:
    """p for threshold x from a strike ladder; linear between strikes, no extrapolation."""
    probs = enforce_ladder(strikes, probs, comparator)
    s = np.asarray(strikes, float)
    order = np.argsort(s)
    s, pr = s[order], np.asarray(probs)[order]
    if x < s[0] or x > s[-1]:
        return {"p": None, "reason": f"threshold {x} outside ladder [{s[0]}, {s[-1]}]: no extrapolation"}
    return {"p": float(np.interp(x, s, pr)), "reason": f"interpolated between strikes {s.tolist()}"}


def longshot_flag(p: float) -> bool:
    """Favourite-longshot bias: tails under ~10% tend to be overpriced. A caveat, not a correction."""
    return p < 0.10 or p > 0.90
