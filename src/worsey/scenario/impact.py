"""Scenario impact: event betas by tier, priced-in P&L, ranking (SPEC §3.1, §3.1b).

Units, used everywhere in this module:
- An asset β is the asset's return per +1.00 change in the event probability
  (Δp from 0 to 1 = the event going from impossible to certain).
- A dollar β is Σ Vᵢ·βᵢ: portfolio dollars per +1.00 Δp.
- P&L if YES = β$·(1−p); if NO = −β$·p. Expected resolution P&L ≈ 0 because p is priced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------- alignment


def align_dp(p: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Daily Δp on equity trading days.

    Prediction markets trade 24/7; equities don't. Sample p at each trading day (last
    value on or before it) so a weekend's Δp lands on Monday with Monday's return.
    Live Kalshi series come from data.kalshi.sample_at_close (hourly candles, last print
    ≤ 16:00 ET); NaN marks a stale day, whose Δp is masked.
    """
    p = p.sort_index()
    if p.isna().any():
        # live Kalshi: p is sampled daily at 16:00 ET with stale days already NaN (CP-L F2).
        # Δp runs from the last good print; a stale day itself is masked, not forward-filled.
        sampled = p.reindex(trading_days)
        dp = sampled.ffill().diff()
        return dp.where(sampled.notna())
    sampled = p.reindex(p.index.union(trading_days)).ffill().reindex(trading_days)
    return sampled.diff()


# ---------------------------------------------------------------- tier 1: empirical


@dataclass(frozen=True)
class Beta:
    beta: float
    se: float
    n: int
    tier: int
    shrunk: float | None = None

    @property
    def t(self) -> float:
        return self.beta / self.se if self.se > 0 else 0.0


def ols_beta(y: pd.Series, x: pd.Series, controls: pd.DataFrame | None = None,
             cov: str = "HAC") -> tuple[float, float, int]:
    """Slope of y on x (with intercept and optional controls). Returns (beta, se, n).

    Daily time-series regressions use HAC (Newey-West, config.HAC_LAGS lags): residuals are
    heteroskedastic (information arrives on jump days) and can be serially correlated
    (ruling C2). cov="HC1" is for independent observations (pooled event windows).
    """
    from .. import config
    X = x.rename("x").to_frame()
    if controls is not None:
        X = X.join(controls, how="inner")
    df = X.join(y.rename("y"), how="inner").dropna()
    if len(df) < 10 or df["x"].abs().sum() == 0:
        return 0.0, math.inf, len(df)
    model = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y")))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": config.HAC_LAGS}) if cov == "HAC" else model.fit(cov_type=cov)
    return float(res.params["x"]), float(res.bse["x"]), int(res.nobs)


def shrink(beta: float, se: float, k: float | None = None) -> float:
    """β·t²/(t²+k): a t-stat of 2 keeps half the estimate (k = config.SHRINK_K)."""
    from .. import config
    k = config.SHRINK_K if k is None else k
    if not math.isfinite(se) or se <= 0:
        return 0.0
    t2 = (beta / se) ** 2
    return beta * t2 / (t2 + k)


def empirical_betas(returns: pd.DataFrame, dp: pd.Series, controls: pd.DataFrame | None = None,
                    k: float | None = None) -> dict[str, Beta]:
    """Tier 1: per-position β of daily returns on Δp."""
    from .. import config
    k = config.SHRINK_K if k is None else k
    out = {}
    for col in returns.columns:
        b, se, n = ols_beta(returns[col], dp, controls)
        out[col] = Beta(b, se, n, tier=1, shrunk=shrink(b, se, k))
    return out


def pooled_series_beta(surprise: pd.Series, reaction: pd.Series) -> Beta:
    """Tier 1(b) for recurring series (Fed, CPI): pool past resolutions.

    surprise = outcome − p the day before; reaction = the asset's return over the
    resolution window. Each resolution is one independent observation, so HC1 (not HAC)
    is the right error (ruling C2). n is small but cleanly identified.
    """
    df = pd.concat({"s": surprise, "r": reaction}, axis=1).dropna()
    if len(df) < 3 or df["s"].abs().sum() == 0:
        return Beta(0.0, math.inf, len(df), tier=1)
    model = sm.OLS(df["r"], df[["s"]])
    res = model.fit(cov_type="HC1") if len(df) > 5 else model.fit()
    b, se = float(res.params["s"]), float(res.bse["s"])
    return Beta(b, se, int(res.nobs), tier=1, shrunk=shrink(b, se))


# ---------------------------------------------------------------- tier 2: factor mapping

# Factor set (SPEC §3.1b). All shocks are expressed as the proxy's % return, so a
# "10y +10bp" proposal is entered as the matching TLT return. CREDIT = HYG − IEF return.
FACTORS = ["SPY", "TLT", "UUP", "USO", "GLD", "CREDIT", "VIXY"]


def factor_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily factor returns from proxy closes (needs HYG and IEF for CREDIT)."""
    r = closes.pct_change()
    out = r[[f for f in FACTORS if f in r.columns]].copy()
    if {"HYG", "IEF"} <= set(r.columns):
        out["CREDIT"] = r["HYG"] - r["IEF"]
    return out


def factor_loadings(asset_returns: pd.DataFrame, factors: pd.DataFrame, window: int | None = None,
                    alpha: float | None = None) -> pd.DataFrame:
    """Stage B (METHODOLOGY §4.2; C9): each asset on the factor set over the last `window`
    trading days (default ~2 years), ridge-regularized. Rows assets, cols factors.

    Ridge on standardized factors with penalty alpha·n (alpha = config.RIDGE_ALPHA), then mapped
    back to raw units: it tames the collinearity between SPY, credit and vol without changing
    well-identified loadings much. HAC standard errors from the same-window OLS are attached as
    `.attrs["se"]` for reporting.
    """
    from .. import config
    window = config.LOADING_WINDOW_DAYS if window is None else window
    alpha = config.RIDGE_ALPHA if alpha is None else alpha
    rows, ses = {}, {}
    for col in asset_returns.columns:
        df = factors.join(asset_returns[col].rename("_y"), how="inner").dropna().tail(window)
        X, y = df.drop(columns="_y"), df["_y"]
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
        Z = ((X - mu) / sd).to_numpy()
        yc = (y - y.mean()).to_numpy()
        n, k = Z.shape
        coef_z = np.linalg.solve(Z.T @ Z + alpha * n * np.eye(k), Z.T @ yc)
        rows[col] = pd.Series(coef_z / sd.to_numpy(), index=X.columns)
        ols = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": config.HAC_LAGS})
        ses[col] = ols.bse.drop("const")
    out = pd.DataFrame(rows).T
    out.attrs["se"] = pd.DataFrame(ses).T
    return out


def shock_impact(loadings: pd.DataFrame, shocks: dict[str, float]) -> pd.DataFrame:
    """Stage A × B: each asset's return by channel for a factor shock map.

    Returns assets × channels (one column per shocked factor) plus 'total'. An asset that
    IS a factor proxy takes the shock directly (its loading on itself is 1 by construction).
    """
    cols = [f for f in shocks if f in loadings.columns]
    by_channel = loadings[cols].mul(pd.Series({f: shocks[f] for f in cols}), axis=1)
    for a in by_channel.index:
        if a in shocks:  # a proxy itself: exact, not regression-estimated
            by_channel.loc[a] = 0.0
            if a in by_channel.columns:
                by_channel.loc[a, a] = shocks[a]
    by_channel["total"] = by_channel.sum(axis=1)
    return by_channel


def window_moves(closes: pd.DataFrame, date, before: int = 1, after: int = 1) -> pd.Series:
    """Realized move over an analog window: close `before` trading days before → `after` days after.

    Used for worked examples and analogs (SPEC §3.1c, §3.2 step 5). Code computes these;
    the LLM never types a realized move. Adds CREDIT if HYG and IEF are present.
    """
    idx = closes.index
    d = pd.Timestamp(date)
    i = idx.searchsorted(d)  # first trading day on/after the event date
    i0, i1 = max(i - before, 0), min(i + after - 1, len(idx) - 1)
    mv = closes.iloc[i1] / closes.iloc[i0] - 1
    if {"HYG", "IEF"} <= set(mv.index):
        mv["CREDIT"] = mv["HYG"] - mv["IEF"]
    return mv


def fit_template(surprise: pd.Series, moves: pd.DataFrame, prior_b: pd.Series | None = None,
                 prior_sd: float | pd.Series = 0.05) -> pd.DataFrame:
    """Fit an event type's shock template: Δf_k = s·u_k + ε_k (METHODOLOGY §4.4; C10).

    Ridge regression toward the prior: s = (Σ u·Δf + λ·s₀) / (Σ u² + λ), with λ = σ̂²_ε / sd²_prior
    (the Bayesian ridge, so a tight prior pulls hard and a vague one barely). No intercept: zero
    surprise → zero move. Without a prior it is plain least squares. Units: factor move per unit
    surprise, the same units as a tier-1 β. Returns factors × [b, se, n].
    """
    s = surprise.astype(float)
    out = {}
    for f in moves.columns:
        df = pd.concat({"s": s, "m": moves[f]}, axis=1).dropna()
        n = len(df)
        ss = float((df["s"] ** 2).sum())
        sm_ = float((df["s"] * df["m"]).sum())
        if n == 0 or ss == 0:
            out[f] = {"b": float(prior_b[f]) if prior_b is not None and f in prior_b.index else 0.0, "se": math.inf, "n": n}
            continue
        b_ols = sm_ / ss
        resid_var = float(((df["m"] - b_ols * df["s"]) ** 2).sum()) / max(n - 1, 1) if n > 1 else 0.0
        if n < 3:  # noise is unidentified with 1–2 instances: assume it's as large as the moves themselves
            resid_var = max(resid_var, float((df["m"] ** 2).mean()))
        if prior_b is not None and f in prior_b.index:
            psd = float(prior_sd[f] if isinstance(prior_sd, pd.Series) else prior_sd)
            lam = resid_var / psd ** 2 if psd > 0 else math.inf
            b = (sm_ + lam * float(prior_b[f])) / (ss + lam) if math.isfinite(lam) else float(prior_b[f])
            se = math.sqrt(resid_var / (ss + lam)) if math.isfinite(lam) and resid_var > 0 else psd
        else:
            b = b_ols
            se = math.sqrt(resid_var / ss) if n > 1 else math.inf
        conf = "low" if n < 3 else ("med" if n < 6 else "high")
        out[f] = {"b": b, "se": se, "n": n, "confidence": conf, "band_lo": b - 2 * se, "band_hi": b + 2 * se}
    return pd.DataFrame(out).T


def template_loo(surprise: pd.Series, moves: pd.DataFrame, **kw) -> pd.DataFrame:
    """Leave-one-out: predict each instance's factor moves from a template fit on the others."""
    rows = {}
    for i in moves.index:
        t = fit_template(surprise.drop(i), moves.drop(i), **kw)
        rows[i] = moves.loc[i] - t["b"] * surprise[i]
    return pd.DataFrame(rows).T  # errors: instances × factors


# ---------------------------------------------------------------- tier 3 + blending

# Claude gives only a sign and a size bucket; the numbers live in config (METHODOLOGY §4.3; C1).
# Value = asset return per +1.00 Δp. Prior sd = PRIOR_SD_FRACTION × bucket: deliberately wide.


def prior_beta(sign: int, bucket: str) -> Beta:
    from .. import config
    m = config.PRIOR_BUCKETS[bucket]
    floor = config.PRIOR_BUCKETS["small"] * config.PRIOR_SD_FRACTION / 2  # "none" still gets a finite, wide-ish sd
    return Beta(sign * m, max(m * config.PRIOR_SD_FRACTION, floor), n=0, tier=3)


def inverse_variance_blend(b_prior: float, s_prior: float, b_hat: float, se_hat: float) -> tuple[float, float]:
    """β_post = (β_prior/s²_prior + β̂/se²) / (1/s²_prior + 1/se²)."""
    if not math.isfinite(se_hat) or se_hat <= 0:
        return b_prior, s_prior
    w0, w1 = 1 / s_prior ** 2, 1 / se_hat ** 2
    return (b_prior * w0 + b_hat * w1) / (w0 + w1), math.sqrt(1 / (w0 + w1))


def blend(prior: Beta, est: Beta) -> Beta:
    b, se = inverse_variance_blend(prior.beta, prior.se, est.beta, est.se)
    return Beta(b, se, est.n, tier=est.tier)


# ---------------------------------------------------------------- P&L and ranking


def dollar_beta(values: pd.Series, betas: dict[str, float]) -> pd.Series:
    """Per-position dollar β (Vᵢ·βᵢ). Positions without a β contribute 0."""
    return pd.Series({t: v * betas.get(t, 0.0) for t, v in values.items()})


def resolution_pnl(beta_usd: float, p: float) -> dict:
    """Priced-in P&L for one event: if YES, if NO, and σ of resolution P&L."""
    return {
        "pnl_yes": beta_usd * (1 - p),
        "pnl_no": -beta_usd * p,
        "sigma": abs(beta_usd) * math.sqrt(p * (1 - p)),
    }


def event_row(event_id: str, p: float, values: pd.Series, betas: dict[str, float], asset_classes: pd.Series,
              beta_tier: str, market: dict | None = None, beta_se: float | None = None, p_confidence: str | None = None,
              resolves_in_horizon: bool = True, flags: list[str] | None = None, title: str | None = None,
              schema: dict | None = None, **extra) -> dict:
    """One event row in the METHODOLOGY §13 output contract (C15).

    B = Σ Vᵢβᵢ (portfolio $ per +1.00 Δp); P&L if YES = B(1−p), if NO = −Bp; σ = |B|·√(p(1−p));
    P&L if YES split by asset class; drivers = the positions contributing most to P&L if YES.
    Extra keys (event_id, close_date, p_source, …) ride along for references and the CLI.
    """
    dv = dollar_beta(values, betas)
    B = float(dv.sum())
    by_ac = (dv * (1 - p)).groupby(asset_classes.reindex(dv.index).fillna("equity")).sum()
    top = dv.abs().sort_values(ascending=False).head(3).index
    row = {
        "event_id": event_id, "event": title or event_id, "schema": schema, "market": market,
        "p": p, "p_confidence": p_confidence, "B": B, "beta_tier": beta_tier, "beta_se": beta_se,
        **resolution_pnl(B, p),
        "pnl_yes_by_class": {k: float(v) for k, v in by_ac.items() if abs(v) > 0},
        "resolves_in_horizon": resolves_in_horizon,
        "drivers": [{"position": k, "contrib": float(dv[k] * (1 - p))} for k in top],
        "theme": None, "flags": list(flags or []),
    }
    row.update(extra)
    return row


def tier2_betas(loadings: pd.DataFrame, shock: dict[str, float], shock_se: dict[str, float] | None = None) -> dict:
    """Tier 2 per asset (METHODOLOGY §4.2): βᵢ = Lᵢ·s, Var(βᵢ) ≈ Lᵢ Σ_s Lᵢᵀ + sᵀ Σ_L s (diagonal Σs).

    loadings.attrs["se"] (from factor_loadings) supplies Σ_L; shock_se supplies Σ_s.
    Returns {asset: (beta, se)}.
    """
    f = [k for k in shock if k in loadings.columns]
    s_vec = pd.Series({k: shock[k] for k in f})
    s_se = pd.Series({k: (shock_se or {}).get(k, 0.0) for k in f})
    L_se = loadings.attrs.get("se", pd.DataFrame(0.0, index=loadings.index, columns=loadings.columns))
    out = {}
    for a in loadings.index:
        L = loadings.loc[a, f]
        var = float((L ** 2 * s_se ** 2).sum() + (s_vec ** 2 * L_se.loc[a, f].fillna(0.0) ** 2).sum())
        out[a] = (float((L * s_vec).sum()), math.sqrt(var))
    return out


def rank_events(rows: list[dict]) -> pd.DataFrame:
    """Order by σ of resolution P&L: the definition of 'what could hit me' (SPEC §3.1a)."""
    return pd.DataFrame(rows).sort_values("sigma", ascending=False).reset_index(drop=True)


def portfolio_regression_check(returns: pd.DataFrame, values: pd.Series, dp: pd.Series) -> dict:
    """Regress the book's dollar P&L on Δp directly: a check on Σ Vᵢβᵢ, and the source of B's
    standard error (it captures the cross-position covariance a per-position sum would miss)."""
    pnl = (returns[values.index] * values).sum(axis=1, min_count=1)
    b, se, n = ols_beta(pnl, dp)
    return {"B": b, "se": se, "n": n}
