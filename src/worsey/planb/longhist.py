"""Long-history data for regime-robust release models (DECISIONS #166).

- Kenneth R. French Data Library (free for research, cited): daily value-weighted returns of 49 US industry
  portfolios (1926+), the Fama-French 5 factors (1963+) and momentum. Files cached in cache/pmrm/ff/ (not committed).
- FRED: 2y (1976+) and 10y (1962+) yields, fed funds (1954+), exchange rates (1971+), WTI (1986+), Baa - 10y (1986+),
  10y - 2y slope (1976+), CPI (1947+).
- ALFRED: release dates (first publication) for CPI, payrolls, GDP, PCE.
All returns in decimal; yield changes in percentage points.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.cache import CACHE_ROOT
from . import library as L

FF = CACHE_ROOT / "pmrm" / "ff"


def _ff_csv(name: str) -> list[str]:
    z = zipfile.ZipFile(FF / name)
    return z.read(z.namelist()[0]).decode("latin-1").splitlines()


def _parse_block(lines: list[str], start: int) -> pd.DataFrame:
    """One table from a Ken French CSV: header line starting with ',' then rows 'YYYYMMDD, v1, v2...' until a blank."""
    hdr = [h.strip() for h in lines[start].split(",")]
    rows = []
    for l in lines[start + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) < 2 or not p[0].isdigit() or len(p[0]) != 8:
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["date"] + hdr[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    return df.where(df > -99) / 100.0


def industries() -> pd.DataFrame:
    """49 industries, value-weighted daily returns (the first table in the file)."""
    lines = _ff_csv("49_Industry_Portfolios_daily_CSV.zip")
    i = next(k for k, l in enumerate(lines) if l.startswith(",Agric"))
    return _parse_block(lines, i)


def ff_factors() -> pd.DataFrame:
    lines = _ff_csv("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    f = _parse_block(lines, next(k for k, l in enumerate(lines) if l.startswith(",Mkt-RF")))
    ml = _ff_csv("F-F_Momentum_Factor_daily_CSV.zip")
    mom = _parse_block(ml, next(k for k, l in enumerate(ml) if l.strip().startswith(",Mom") or l.strip().startswith(",MOM")))
    f["Mom"] = mom.iloc[:, 0]
    return f


def long_factors(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Named factors with long history (rows = sessions). NaN where a series hasn't started."""
    f = ff_factors().reindex(sessions)
    lv = lambda sid: L.fred_daily(sid).reindex(sessions, method="ffill", limit=3)
    X = pd.DataFrame(index=sessions)
    X["market"] = f["Mkt-RF"] + f["RF"]
    X["size"], X["value"], X["profitability"], X["investment"], X["momentum"] = f["SMB"], f["HML"], f["RMW"], f["CMA"], f["Mom"]
    X["rates_2y"] = lv("DGS2").diff()
    X["rates_10y"] = lv("DGS10").diff()
    # dollar: average log change of USD vs JPY, CHF and GBP (all from 1971)
    jpy, chf, gbp = lv("DEXJPUS"), lv("DEXSZUS"), lv("DEXUSUK")
    X["dollar"] = (np.log(jpy).diff() + np.log(chf).diff() - np.log(gbp).diff()) / 3
    X["oil"] = np.log(lv("DCOILWTICO").where(lambda s: s > 0)).diff()
    return X


def long_state(sessions: pd.DatetimeIndex, X: pd.DataFrame) -> pd.DataFrame:
    """Regime state known the day before (lagged one session).
    infl_yoy uses the latest CPI vintage with a 45-day publication lag (a close point-in-time approximation)."""
    lv = lambda sid: L.fred_daily(sid)
    s = pd.DataFrame(index=sessions)
    s["sb_corr60"] = X["market"].rolling(60, min_periods=40).corr(X["rates_10y"])
    cpi = lv("CPIAUCSL")
    yoy = (cpi / cpi.shift(12) - 1) * 100
    yoy.index = yoy.index + pd.Timedelta(days=45 + 31)      # observation month end + publication lag
    s["infl_yoy"] = yoy.reindex(sessions, method="ffill")
    s["ff_change_6m"] = lv("DFF").reindex(sessions, method="ffill").diff(126)
    s["realized_vol"] = X["market"].rolling(20, min_periods=15).std() * np.sqrt(252) * 100
    s["credit_baa"] = lv("BAA10Y").reindex(sessions, method="ffill")
    s["curve_10y2y"] = lv("T10Y2Y").reindex(sessions, method="ffill")
    return s.shift(1)


def release_dates(start: str = "1976-01-01") -> pd.DataFrame:
    """CPI, jobs, PCE first-publication dates and GDP vintages (ALFRED), FOMC from Feb 1994 (announced decisions)."""
    rows = []
    for rel, sid in (("cpi", "CPIAUCSL"), ("jobs", "PAYEMS"), ("pce", "PCEPI")):
        rows += [{"date": p.normalize(), "release": rel} for p in L.first_releases(sid)["published"]]
    rows += [{"date": pd.Timestamp(d), "release": "gdp"} for d in L.vintage_dates("GDPC1")]
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import pmrm_p0b as P
    rows += [{"date": pd.Timestamp(r["date"]), "release": "fomc"} for r in L.fomc_dates(1994, P.fomc())
             if r["scheduled"] and r["date"] >= pd.Timestamp("1994-02-04").date()]
    df = pd.DataFrame(rows).drop_duplicates().sort_values("date")
    return df[df["date"] >= start].reset_index(drop=True)
