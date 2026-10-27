"""Plan B stage 1: the broad factor model (PLAN_B §2, §5.2; BUILD_PLAN B1).

Three versions, compared on held-out days and held-out assets:
- latent:   principal factors of standardized daily returns; each day's factor values are estimated by GLS from the
            assets that day (r_t ≈ L f_t + e_t, Var(e) = diag(psi)).
- anchored: observed factor returns (market, 2y and 10y yield changes, dollar, oil, gold, credit, size, value); each
            asset's loadings by time-series regression in the training window.
- hybrid:   the anchored core plus latent factors fitted to what the core leaves unexplained.

Scoring (no factor version may grade itself): assets are split into two halves; on every test day the latent
factors are estimated from one half and used to predict the other, and the reverse. Anchored factors are observed
returns of proxies that are excluded from the scored set. Score = pooled out-of-sample R² of standardized returns.
Every number here is computed in code; nothing here calls an LLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- latent

def standardize(R: pd.DataFrame, mu: pd.Series | None = None, sd: pd.Series | None = None):
    mu = R.mean() if mu is None else mu
    sd = R.std(ddof=0) if sd is None else sd
    return (R - mu) / sd.replace(0, np.nan), mu, sd


def pca_loadings(Z: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Loadings (d×k) and idiosyncratic variances (d) from the correlation matrix of standardized returns.
    Pairwise-complete moments so late or gappy series still contribute; the matrix is made PSD."""
    C = Z.cov(min_periods=60).to_numpy()
    C = np.nan_to_num((C + C.T) / 2)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    L = V[:, :k] * np.sqrt(np.maximum(w[:k], 1e-9))
    psi = np.maximum(np.diag(C) - (L ** 2).sum(1), 0.02)
    return L, psi


def gls_factors(Z: np.ndarray, L: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Per-day factor estimates from the assets observed that day: f = (L'Ψ⁻¹L)⁻¹ L'Ψ⁻¹ z (missing assets dropped)."""
    T, k = Z.shape[0], L.shape[1]
    F = np.full((T, k), np.nan)
    for t in range(T):
        o = ~np.isnan(Z[t])
        if o.sum() <= k + 2:
            continue
        Lo, po = L[o], psi[o]
        A = (Lo.T / po) @ Lo
        F[t] = np.linalg.solve(A + 1e-9 * np.eye(k), (Lo.T / po) @ Z[t, o])
    return F


def r2(Z: np.ndarray, P: np.ndarray) -> float:
    m = ~np.isnan(Z) & ~np.isnan(P)
    return float(1 - ((Z[m] - P[m]) ** 2).sum() / (Z[m] ** 2).sum()) if m.any() else float("nan")


def latent_cv_r2(Ztr: pd.DataFrame, Zte: pd.DataFrame, k: int, seed: int = 0) -> float:
    """Cross-asset, cross-time R²: loadings from training days; on test days the factors come from one half of the
    assets and predict the other half (both directions)."""
    if k == 0:
        return 0.0
    cols = np.array(Ztr.columns)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cols))
    halves = [cols[perm[: len(cols) // 2]], cols[perm[len(cols) // 2:]]]
    L, psi = pca_loadings(Ztr, k)
    idx = {c: i for i, c in enumerate(cols)}
    num = den = 0.0
    for a, b in ((0, 1), (1, 0)):
        ia, ib = [idx[c] for c in halves[a]], [idx[c] for c in halves[b]]
        F = gls_factors(Zte[halves[a]].to_numpy(), L[ia], psi[ia])
        P = F @ L[ib].T
        Zb = Zte[halves[b]].to_numpy()
        m = ~np.isnan(Zb) & ~np.isnan(P)
        num += ((Zb[m] - P[m]) ** 2).sum()
        den += (Zb[m] ** 2).sum()
    return float(1 - num / den)


# ---------------------------------------------------------------- anchored and hybrid

def regress_loadings(Z: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Each asset's loadings on observed factors by OLS on the training days it has (≥ 120 days)."""
    out = {}
    for c in Z.columns:
        d = pd.concat([Z[c], X], axis=1).dropna()
        if len(d) < 120:
            continue
        y, A = d.iloc[:, 0].to_numpy(), d.iloc[:, 1:].to_numpy()
        A1 = np.column_stack([np.ones(len(A)), A])
        out[c] = np.linalg.lstsq(A1, y, rcond=None)[0][1:]
    return pd.DataFrame(out, index=X.columns).T


def anchored_predict(B: pd.DataFrame, X: pd.DataFrame, cols) -> np.ndarray:
    return X.to_numpy() @ B.loc[cols].to_numpy().T


def hybrid_cv_r2(Ztr, Zte, Xtr, Xte, k2: int, seed: int = 0) -> float:
    """Anchored core on observed factors, plus k2 latent factors of the training residuals, scored cross-asset on the
    test residuals (latent part estimated from one half, scored on the other)."""
    cols = [c for c in Ztr.columns]
    B = regress_loadings(Ztr[cols], Xtr)
    cols = list(B.index)
    Etr = Ztr[cols] - pd.DataFrame(anchored_predict(B, Xtr, cols), index=Ztr.index, columns=cols)
    Ete = Zte[cols] - pd.DataFrame(anchored_predict(B, Xte, cols), index=Zte.index, columns=cols)
    base_num = (np.nan_to_num(Ete.to_numpy()) ** 2).sum()
    den = (np.nan_to_num(Zte[cols].to_numpy() * ~np.isnan(Ete.to_numpy())) ** 2).sum()
    if k2 == 0:
        return float(1 - base_num / den)
    rest = 1 - latent_cv_r2(Etr, Ete, k2, seed)        # share of residual variance left after the latent part
    return float(1 - base_num * rest / den)


# ---------------------------------------------------------------- stability

def subspace_similarity(L1: np.ndarray, L2: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between two loading subspaces (1 = identical, rotation-free)."""
    Q1, _ = np.linalg.qr(L1)
    Q2, _ = np.linalg.qr(L2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))
