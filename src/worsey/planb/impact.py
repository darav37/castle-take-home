"""Plan B stage 2: event -> factor impact (PLAN_B §2, §5.3; BUILD_PLAN B3).

- Scheduled releases: factor responses regressed on each release type's surprise, with a shared low-rank structure
  (reduced-rank ridge regression), so thin types borrow strength from rich ones.
- Unscheduled events: analog prediction by code (§5.3 steps 2 and 4 without the LLM comparison, which is B4):
  the predicted response is a similarity-weighted mean of the nearest past events' realized responses.
Every number here is computed in code; nothing calls an LLM.
"""

from __future__ import annotations

import numpy as np


def ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Coefficients (p×q) of Y on X (X already includes any intercept columns), ridge on all columns."""
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)


def reduced_rank(X: np.ndarray, Y: np.ndarray, rank: int, lam: float = 1.0) -> np.ndarray:
    """Reduced-rank ridge regression: project the ridge fit onto its top `rank` response directions
    (Izenman's RRR with a ridge first stage). rank >= q returns the full ridge fit."""
    B = ridge(X, Y, lam)
    if rank >= Y.shape[1]:
        return B
    Yhat = X @ B
    _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
    V = Vt[:rank].T
    return B @ V @ V.T


def design(types: list[str], S: np.ndarray, levels: list[str]) -> np.ndarray:
    """Type-specific intercepts and type-specific surprise slopes: one block per release type."""
    n, k = S.shape
    cols = []
    for lv in levels:
        m = np.array([t == lv for t in types], dtype=float)
        cols.append(m[:, None])
        cols.append(np.nan_to_num(S) * m[:, None])
    return np.hstack(cols)


def r2_vs(Y: np.ndarray, P: np.ndarray, base: np.ndarray) -> float:
    """1 - SSE(model)/SSE(baseline), pooled over all entries."""
    m = ~np.isnan(Y)
    return float(1 - ((Y - P)[m] ** 2).sum() / ((Y - base)[m] ** 2).sum())


# ---------------------------------------------------------------- analogs

SEV = {"minor": 0, "moderate": 1, "major": 2, "extreme": 3}


def similarity(a: dict, b: dict) -> float:
    """Schema-field similarity in [0, 1] (PLAN_B §5.3 step 2, code only): category, targeted sectors, severity,
    region, anticipation. Fixed weights, set before scoring."""
    s = 0.40 * (a.get("category") == b.get("category"))
    sa, sb = set(a.get("targeted_sectors") or []), set(b.get("targeted_sectors") or [])
    s += 0.25 * (len(sa & sb) / len(sa | sb) if sa | sb else 0.0)
    ea, eb = SEV.get(a.get("severity") or "", 1), SEV.get(b.get("severity") or "", 1)
    s += 0.15 * (1 - abs(ea - eb) / 3)
    ra, rb = (a.get("region") or "").lower(), (b.get("region") or "").lower()
    us = lambda r: ("united states" in r) or r.startswith("us") or "u.s." in r
    s += 0.10 * (us(ra) == us(rb))
    s += 0.10 * (a.get("anticipation") == b.get("anticipation"))
    return float(s)


def analog_predict(target: dict, pool: list[dict], Y: np.ndarray, k: int = 5, sim=None) -> tuple[np.ndarray, float]:
    """Similarity-weighted mean response of the k most similar events in `pool` (rows of Y), and the spread of those
    analogs (the uncertainty band grows when analogs disagree)."""
    sims = np.array([(sim or similarity)(target, p) for p in pool])
    idx = np.argsort(-sims)[:k]
    w = np.maximum(sims[idx], 1e-6)
    w = w / w.sum()
    pred = w @ Y[idx]
    spread = float(np.sqrt((w[:, None] * (Y[idx] - pred) ** 2).sum(0).mean()))
    return pred, spread


# ---------------------------------------------------------------- direction

DIRECTION_FIELDS = ("escalation", "growth_shock", "inflation_shock", "oil_supply", "policy_stance", "risk_sentiment")
NEUTRAL = {"none", "neutral", "not_applicable", None}


def direction_similarity(a: dict, b: dict) -> float:
    """Schema similarity (above) blended with agreement of the event-direction labels: share of the six channels
    where both events point the same way (neutral counts as agreement only when both are neutral)."""
    da, db = a.get("direction") or {}, b.get("direction") or {}
    agree = np.mean([da.get(f) == db.get(f) for f in DIRECTION_FIELDS]) if da and db else 0.5
    return 0.5 * similarity(a, b) + 0.5 * float(agree)


def direction_features(e: dict) -> np.ndarray:
    """One-hot of the non-neutral direction values (code learns what each direction does to each factor)."""
    d = e.get("direction") or {}
    vals = [("escalation", "escalation"), ("escalation", "de-escalation"), ("growth_shock", "negative"),
            ("growth_shock", "positive"), ("inflation_shock", "up"), ("inflation_shock", "down"),
            ("oil_supply", "disruption"), ("oil_supply", "increase"), ("policy_stance", "tighter"),
            ("policy_stance", "looser"), ("risk_sentiment", "risk_off"), ("risk_sentiment", "risk_on")]
    return np.array([1.0 if d.get(f) == v else 0.0 for f, v in vals])


def ridge_loo_predictions(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Exact leave-one-out predictions of ridge regression in one pass (hat-matrix identity):
    ŷ_(-i) = y_i - (y_i - ŷ_i) / (1 - h_ii), with H = X (X'X + λI)⁻¹ X'. X includes the intercept column."""
    G = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T)       # p×n
    Yhat = X @ (G @ Y)
    h = np.einsum("ij,ji->i", X, G)
    return Y - (Y - Yhat) / (1 - h)[:, None]
