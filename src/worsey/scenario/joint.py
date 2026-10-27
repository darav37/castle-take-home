"""Dependence and joint scenarios (METHODOLOGY §8; C7, C8).

Per-event P&Ls are never summed in the UI; the joint distribution comes from here.
- Dependence is measured on daily Δlogit(p) (logit stabilizes variance near 0 and 1), clipped to
  p ∈ [0.01, 0.99], and shrunk toward the identity (Ledoit-Wolf).
- Latent Gaussian copula: marginals match market p exactly.
- Logical constraints are structural, not estimated: a strike ladder on one variable shares a
  single latent Z (so nested probabilities are monotone by construction), and mutually exclusive
  outcomes share a single latent Z cut into intervals (a categorical draw).
- Reported: VaR and expected shortfall at 5%, each event's Euler contribution to ES, and the most
  likely bad combination.
Stated assumption: Δlogit correlation proxies the correlation of outcomes.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import config


# ---------------------------------------------------------------- dependence


def logit(p, clip=None):
    lo, hi = clip or config.LOGIT_CLIP
    p = np.clip(np.asarray(p, float), lo, hi)
    return np.log(p / (1 - p))


def dlogit(p_frame: pd.DataFrame) -> pd.DataFrame:
    """Daily Δlogit per market from daily p (NaN where p is stale/missing)."""
    x = pd.DataFrame(logit(p_frame.to_numpy()), index=p_frame.index, columns=p_frame.columns)
    x[p_frame.isna()] = np.nan
    return x.diff()


def ledoit_wolf_corr(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of a sample correlation toward the identity.

    X: n × k complete rows. Standardize, S = XᵀX/n; δ* = min(1, Σₜ‖xₜxₜᵀ − S‖²_F / n² ÷ ‖S − I‖²_F).
    Returns (R, δ).
    """
    n, k = X.shape
    Z = (X - X.mean(0)) / X.std(0, ddof=0)
    S = Z.T @ Z / n
    num = sum(np.sum((np.outer(z, z) - S) ** 2) for z in Z) / n ** 2
    den = np.sum((S - np.eye(k)) ** 2)
    delta = 1.0 if den == 0 else float(min(1.0, max(0.0, num / den)))
    return delta * np.eye(k) + (1 - delta) * S, delta


def nearest_psd(c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Pairwise correlations on uneven overlaps need not be PSD; clip eigenvalues and rescale."""
    c = np.nan_to_num((c + c.T) / 2, nan=0.0)
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def correlation(p_frame: pd.DataFrame, min_overlap: int | None = None) -> dict:
    """Latent correlation R for the copula from daily p histories (METHODOLOGY §8.2 v1).

    Ledoit-Wolf on the days where every market has a Δlogit, if there are ≥ min_overlap of them.
    Otherwise pairwise correlations where a pair overlaps ≥ min_overlap days, and NaN where it
    doesn't: `fill_missing` then applies the tail-conservative fallback. Short-lived contracts
    rarely overlap, so the fallback is common and always labelled.
    """
    min_overlap = config.MIN_OVERLAP_DAYS if min_overlap is None else min_overlap
    d = dlogit(p_frame)
    full = d.dropna()
    ids = list(p_frame.columns)
    cov = d.cov(min_periods=min_overlap)
    if len(ids) == 1:
        return {"R": pd.DataFrame([[1.0]], index=ids, columns=ids), "method": "single", "delta": None, "n": len(full), "cov_dlogit": cov}
    if len(full) >= min_overlap:
        R, delta = ledoit_wolf_corr(full.to_numpy())
        return {"R": pd.DataFrame(R, index=ids, columns=ids), "method": "ledoit_wolf", "delta": delta, "n": len(full), "cov_dlogit": cov}
    R = d.corr(min_periods=min_overlap)
    for i in ids:
        R.at[i, i] = 1.0
    return {"R": R, "method": "pairwise", "delta": None, "n": len(full), "cov_dlogit": cov}


def fill_missing(R: pd.DataFrame, B: pd.Series, value: float | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Fill pairs with too little overlap, conservatively for tail risk.

    ρᵢⱼ = value·sign(Bᵢ·Bⱼ): same-sign exposures get +value, opposite-sign −value. Either way the pair
    adds to the variance of portfolio P&L instead of cancelling, so the tail is never understated.
    Then projects to the nearest PSD matrix. Returns (R, filled pairs for labelling).
    """
    value = config.FALLBACK_CORR if value is None else value
    R = R.copy()
    filled = []
    ids = list(R.index)
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            if pd.isna(R.at[a, b]):
                rho = value * float(np.sign(B.get(a, 0.0) * B.get(b, 0.0)) or 1.0)
                R.at[a, b] = R.at[b, a] = rho
                filled.append({"pair": [a, b], "rho": rho, "why": "too little overlapping history: tail-conservative fallback"})
    psd = nearest_psd(R.to_numpy(dtype=float))
    return pd.DataFrame(psd, index=ids, columns=ids), filled


def themes(corr: pd.DataFrame, threshold: float | None = None) -> list[list[str]]:
    """Cluster events whose |correlation| ≥ threshold (connected components)."""
    threshold = config.THEME_CORR if threshold is None else threshold
    names = list(corr.index)
    seen, out = set(), []
    for n in names:
        if n in seen:
            continue
        stack, comp = [n], []
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            comp.append(a)
            for b in names:
                c = corr.at[a, b]
                if b not in seen and pd.notna(c) and abs(c) >= threshold:
                    stack.append(b)
        out.append(sorted(comp))
    return out


# ---------------------------------------------------------------- joint simulation


def _latent_layout(ids: list[str], ladders: list[list[str]], exclusive: list[list[str]]) -> tuple[list[str], dict]:
    """One latent per ladder / exclusive group / standalone event. Returns (latent reps, id → (latent idx, kind))."""
    owner, reps = {}, []
    for kind, groups in (("ladder", ladders), ("exclusive", exclusive)):
        for g in groups:
            g = [i for i in g if i in ids]
            if not g:
                continue
            for i in g:
                if i in owner:
                    raise ValueError(f"{i} is in two structural groups")
                owner[i] = (len(reps), kind, g)
            reps.append(g[0])
    for i in ids:
        if i not in owner:
            owner[i] = (len(reps), "single", [i])
            reps.append(i)
    return reps, owner


def simulate(p: pd.Series, corr: pd.DataFrame, beta_usd: pd.Series, n: int | None = None, seed: int = 0,
             condition: dict[str, int] | None = None, ladders: list[list[str]] | None = None,
             exclusive: list[list[str]] | None = None, tail: float | None = None) -> dict:
    """Joint Monte Carlo over event outcomes; P&L = Σ Bₘ·(Oₘ − pₘ).

    ladders: groups of markets on the same variable (e.g. tariff ≥ 20%, ≥ 25%, ≥ 30%): one shared Z,
      outcome = Z < Φ⁻¹(pᵢ), so P(≥30) ≤ P(≥25) holds in every draw.
    exclusive: groups of mutually exclusive outcomes: one shared Z cut at cumulative p's.
    condition: e.g. {"tariffs": 1} keeps only matching draws ("what if X and Y?").
    """
    n = config.COPULA_DRAWS if n is None else n
    tail = config.TAIL if tail is None else tail
    ids = list(p.index)
    reps, owner = _latent_layout(ids, ladders or [], exclusive or [])
    R = nearest_psd(corr.reindex(index=reps, columns=reps).fillna(0.0).to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(np.zeros(len(reps)), R, size=n)
    out = np.zeros((n, len(ids)), dtype=int)
    for j, i in enumerate(ids):
        li, kind, group = owner[i]
        z = Z[:, li]
        if kind == "exclusive":
            cum = np.cumsum([p[g] for g in group])
            if cum[-1] > 1 + 1e-9:
                cum = cum / cum[-1]  # incoherent quotes summing > 1: scale, never invent extra mass
            k = group.index(i)
            lo = norm.ppf(cum[k - 1]) if k > 0 else -np.inf
            out[:, j] = (z >= lo) & (z < norm.ppf(cum[k]))
        else:
            out[:, j] = z < norm.ppf(p[i])
    if condition:
        mask = np.ones(n, bool)
        for k, v in condition.items():
            mask &= out[:, ids.index(k)] == v
        out = out[mask]
    if len(out) == 0:
        return {"n": 0}
    contrib = (out - p.to_numpy()) * beta_usd.reindex(ids).to_numpy()
    pnl = contrib.sum(axis=1)
    var = float(np.quantile(pnl, tail))
    in_tail = pnl <= var
    es = float(pnl[in_tail].mean())
    euler = contrib[in_tail].mean(axis=0)  # Σ euler = ES: each event's share of the bad tail
    combo, cnt = Counter(map(tuple, out[in_tail])).most_common(1)[0]
    return {
        "n": int(len(pnl)), "mean": float(pnl.mean()), "sd": float(pnl.std()),
        "var": var, "es": es, "tail": tail,
        "euler_es": {i: float(e) for i, e in zip(ids, euler)},
        "worst_combo": {i: int(o) for i, o in zip(ids, combo)},
        "worst_combo_share_of_tail": cnt / int(in_tail.sum()),
        "joint_prob": {i: float(out[:, j].mean()) for j, i in enumerate(ids)},
    }


def odds_move(p: pd.Series, cov_dlogit: pd.DataFrame, moved: str, new_p: float, beta_usd: pd.Series) -> dict:
    """"What if A's odds jump to new_p?" (METHODOLOGY §8.3): Gaussian conditioning in logit space.

    E[Δx_B | Δx_A = a] = Σ_BA / Σ_AA · a, mapped back to p and then to P&L via B (portfolio
    sensitivity per +1.00 Δp). Uses the daily Δlogit covariance as the dependence estimate.
    """
    x0 = pd.Series(logit(p.to_numpy()), index=p.index)
    a = float(logit([new_p])[0] - x0[moved])
    cov = cov_dlogit.reindex(index=p.index, columns=p.index)
    s_aa = float(cov.at[moved, moved])
    dx = (cov[moved] / s_aa * a) if s_aa > 0 else pd.Series(0.0, index=p.index)
    dx[moved] = a
    p_new = 1 / (1 + np.exp(-(x0 + dx.fillna(0.0))))
    dp = p_new - p
    pnl = float((beta_usd.reindex(p.index).fillna(0.0) * dp).sum())
    return {"p_new": p_new.to_dict(), "dp": dp.to_dict(), "pnl": pnl}
