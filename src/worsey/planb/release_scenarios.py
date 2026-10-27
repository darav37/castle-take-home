"""Probability-weighted release damage from Kalshi ladders (DECISIONS #174). No LLM.

For an upcoming release with a Kalshi ladder:
  1. the ladder gives a probability for every outcome bucket (planb_b5_ladders.implied's distribution)
  2. each outcome -> surprise vs the implied mean (in implied sd)
  3. surprise -> release-day 2y/10y yield changes (stage 1, fitted on 2021-26 Kalshi releases of that type, with its
     residual spread kept so the rates reaction is a range, not a point)
  4. yield changes -> 49-industry / factor moves through the long-history regime kernel at TODAY's regime (stage 2)
  5. -> the book's P&L distribution: probability-weighted mean, and the tails (5% / 95%), plus "if it prints X" rows.
Every number is code; the LLM only explains the result.
"""

from __future__ import annotations

import numpy as np


def outcome_distribution(strikes: np.ndarray, p_above: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outcome points and probabilities from a "greater than k" ladder (monotone; tails half a step out)."""
    order = np.argsort(strikes)
    k, p = strikes[order], np.minimum.accumulate(np.clip(p_above[order], 0, 1))
    step = np.median(np.diff(k)) if len(k) > 1 else 1.0
    pts = np.concatenate([[k[0] - step / 2], (k[:-1] + k[1:]) / 2, [k[-1] + step / 2]])
    mass = np.clip(np.concatenate([[1 - p[0]], p[:-1] - p[1:], [p[-1]]]), 0, None)
    return pts, mass / mass.sum()


def book_pnl_distribution(pts, mass, stage1_coef, stage1_resid_sd, stage2_fn, book_weights, n_draws: int = 4000, seed: int = 1729):
    """Simulate: outcome ~ ladder; rates = stage1(surprise) + N(0, resid); industries = stage2(rates); P&L = weights · moves.
    stage1_coef: (intercept, slope) per yield as a 2x2 array [[a2, b2], [a10, b10]]; surprise in implied-sd units.
    stage2_fn(rates_2y, rates_10y) -> vector of industry moves (decimal returns). Returns summary and per-outcome rows."""
    rng = np.random.default_rng(seed)
    mean = float(mass @ pts)
    sd = float(np.sqrt(mass @ (pts - mean) ** 2)) or 1.0
    draws = rng.choice(len(pts), n_draws, p=mass)
    z = (pts[draws] - mean) / sd
    r2 = stage1_coef[0, 0] + stage1_coef[0, 1] * z + rng.normal(0, stage1_resid_sd[0], n_draws)
    r10 = stage1_coef[1, 0] + stage1_coef[1, 1] * z + rng.normal(0, stage1_resid_sd[1], n_draws)
    pnl = np.array([book_weights @ stage2_fn(a, b) for a, b in zip(r2, r10)])
    rows = []
    for j, (x, m) in enumerate(zip(pts, mass)):
        zz = (x - mean) / sd
        a = stage1_coef[0, 0] + stage1_coef[0, 1] * zz
        b = stage1_coef[1, 0] + stage1_coef[1, 1] * zz
        rows.append({"outcome": float(x), "probability": float(m), "rates_2y_bp": float(a * 100), "book_pnl": float(book_weights @ stage2_fn(a, b))})
    return {"implied_mean": mean, "implied_sd": sd, "expected_pnl": float(pnl.mean()), "p05": float(np.quantile(pnl, 0.05)),
            "p95": float(np.quantile(pnl, 0.95)), "by_outcome": rows}
