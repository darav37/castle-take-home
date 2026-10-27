"""Market regime state, daily, known before each day (DECISIONS #165).

Every variable is observable the day BEFORE it is used (lagged one session), so attaching it to an event never leaks
the event's own outcome. Sources: FRED (DGS10, T5YIE, DFF, VIXCLS, BAA10Y, T10Y2Y) and the B1 factor series.

Continuous state (for models):
  sb_corr60      trailing 60-session correlation of the S&P (market factor) with 10y yield changes
  infl_be5y      5-year breakeven inflation (level, %)
  ff_change_6m   change in effective fed funds over ~126 sessions (pp): > 0 hiking, < 0 cutting
  vix            VIX level
  credit_baa     Moody's Baa corporate yield minus 10y Treasury (pp; long history, unlike the HY OAS on FRED)
  curve_10y2y    10y minus 2y Treasury slope (pp)
Labels (for the library and for reading results): stock_bond (together | apart), inflation (high >= 2.5 | normal),
policy (hiking | cutting | hold), volatility (stressed VIX >= 25 | calm), credit (stressed Baa spread >= 3 | normal),
curve (inverted | normal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import library as L

STATE = ["sb_corr60", "infl_be5y", "ff_change_6m", "vix", "credit_baa", "curve_10y2y"]


def daily_state(sessions: pd.DatetimeIndex, market: pd.Series, d10y: pd.Series) -> pd.DataFrame:
    """Regime state per session, lagged one session (known before the session opens)."""
    def lvl(sid):
        return L.fred_daily(sid).reindex(sessions, method="ffill")

    s = pd.DataFrame(index=sessions)
    s["sb_corr60"] = market.rolling(60, min_periods=40).corr(d10y)
    s["infl_be5y"] = lvl("T5YIE")
    s["ff_change_6m"] = lvl("DFF").diff(126)
    s["vix"] = lvl("VIXCLS")
    s["credit_baa"] = lvl("BAA10Y")   # FRED keeps only ~3 years of the ICE HY OAS (licensing), so Moody's Baa - 10y
    s["curve_10y2y"] = lvl("T10Y2Y")
    return s.shift(1)


def labels(row: pd.Series) -> dict:
    g = lambda k: row.get(k)
    f = lambda x: x is not None and not (isinstance(x, float) and np.isnan(x))
    return {
        "stock_bond": None if not f(g("sb_corr60")) else ("together" if g("sb_corr60") > 0 else "apart"),
        "inflation": None if not f(g("infl_be5y")) else ("high" if g("infl_be5y") >= 2.5 else "normal"),
        "policy": None if not f(g("ff_change_6m")) else ("hiking" if g("ff_change_6m") > 0.25 else "cutting" if g("ff_change_6m") < -0.25 else "hold"),
        "volatility": None if not f(g("vix")) else ("stressed" if g("vix") >= 25 else "calm"),
        "credit": None if not f(g("credit_baa")) else ("stressed" if g("credit_baa") >= 3 else "normal"),
        "curve": None if not f(g("curve_10y2y")) else ("inverted" if g("curve_10y2y") < 0 else "normal"),
    }


def recency_weights(dates: pd.Series | np.ndarray, ref: pd.Timestamp, half_life_years: float | None) -> np.ndarray:
    """Exponential decay by age at `ref`: weight = 0.5 ** (age_years / half_life). None = equal weights."""
    d = pd.to_datetime(pd.Series(dates))
    if half_life_years is None:
        return np.ones(len(d))
    age = (ref - d).dt.days.clip(lower=0).to_numpy() / 365.25
    return 0.5 ** (age / half_life_years)


def weighted_ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float = 1.0) -> np.ndarray:
    W = w[:, None]
    return np.linalg.solve(X.T @ (X * W) + lam * np.eye(X.shape[1]), X.T @ (Y * W))
