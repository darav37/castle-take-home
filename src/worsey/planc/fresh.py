"""Market data past the research freeze (Oct 16, 2026) for the shadow run (DECISIONS #201).

The research caches end at pmrm_audit.AUDIT_END so reruns hit the cache. The shadow run needs the same series up to
today: the 13 anchor factors (verified identical to B1's definitions on the overlap) and FRED levels. Days up to the
freeze come from the frozen research series; later days are appended from fresh pulls, cached per day.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..data import massive
from ..planb import library as LB

ANCHOR_ETFS = ["SPY", "UUP", "USO", "GLD", "HYG", "IEF", "IWM", "VTV", "VUG", "EEM", "XLV", "DBA", "XLU", "XLE"]
FREEZE = pd.Timestamp("2026-10-16")


def fred(sid: str, end: date) -> pd.Series:
    """A FRED series through `end` (a new cache key per day, unlike LB.fred_daily, which is frozen)."""
    obs = LB.fred_api("/series/observations", series_id=sid, observation_end=end.isoformat())["observations"]
    return pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")}).sort_index()


def _total_returns(tickers, start: str, end: date) -> pd.DataFrame:
    out = {}
    for t in tickers:
        px = massive.daily_bars(t, start, end.isoformat(), trim_to_listing=False)["close"]
        px = px[~px.index.duplicated()]
        div = pd.Series(0.0, index=px.index)
        try:
            for d in massive.dividends(t):
                ex = pd.Timestamp(d.get("ex_dividend_date") or "1900-01-01")
                if ex in div.index and d.get("dividend_type") in (None, "CD", "SC"):
                    div[ex] += float(d.get("cash_amount") or 0)
        except Exception:  # noqa: BLE001
            pass
        out[t] = (px + div) / px.shift(1) - 1
    return pd.DataFrame(out)


def anchors(ext: dict, end: date) -> pd.DataFrame:
    """Standardized anchors (as research uses them) on research sessions through the freeze, plus fresh sessions to `end`."""
    A = ext["anchors"]
    R = _total_returns(ANCHOR_ETFS, "2026-06-01", end)
    new = R.index[R.index > FREEZE]
    if len(new):
        d2, d10 = fred("DGS2", end), fred("DGS10", end)
        full = pd.DatetimeIndex(sorted(set(A.index[A.index >= "2026-06-01"]) | set(new)))
        ch2 = d2.reindex(full).ffill(limit=2).diff().reindex(new)          # pp changes, as planb_b1.fred_change
        ch10 = d10.reindex(full).ffill(limit=2).diff().reindex(new)
        Rn = R.loc[new]
        add = pd.DataFrame({"market": Rn.SPY, "rates_2y": ch2, "rates_10y": ch10, "dollar": Rn.UUP, "oil": Rn.USO, "gold": Rn.GLD,
                            "credit": Rn.HYG - Rn.IEF, "size": Rn.IWM - Rn.SPY, "value": Rn.VTV - Rn.VUG, "asia_em": Rn.EEM - Rn.SPY,
                            "healthcare": Rn.XLV, "agriculture": Rn.DBA, "defensive_vs_producers": Rn.XLU - Rn.XLE}, index=new)
        A = pd.concat([A, add[A.columns]])
    return (A - ext["anchor_mean"]) / ext["anchor_sd"]


def check_rates_units(ext: dict) -> float:
    """The research rates anchors vs FRED percentage-point changes (should be 1)."""
    d10 = LB.fred_daily("DGS10")
    a = ext["anchors"]["rates_10y"].loc["2026-01-01":"2026-10-16"]
    f = d10.reindex(ext["anchors"].index).ffill(limit=2).diff().loc[a.index]
    ok = a.notna() & f.notna() & (f != 0)
    return float(np.median(a[ok] / f[ok]))
