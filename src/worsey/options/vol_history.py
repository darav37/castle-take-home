"""Daily implied-vol history per underlying. Feeds the playbook signals in options/signals.py (skew, term structure and IV
z-scores, spot–vol correlation).

One row per trading day, constant maturity:
  atm_front, put25_front, call25_front   (~30d)
  atm_back,  put25_back,  call25_back    (~91d)
  spot, source ('backfill' | 'snapshot')

Backfill: for each standard monthly expiry in the window, list its contracts (reference endpoint), pick a
log-spaced strike ladder around the spot range it lived through, fetch each contract's daily closes, back
out IV from the close, and build a per-day smile. A day's constant-maturity point interpolates total
variance between the bracketing expiries. Out-of-the-money contracts only (puts below spot, calls above),
since in-the-money closes are mostly intrinsic and stale.

Pure functions take a `fetch` object (contracts(u, expiry, expired) / closes(ticker, start, end)) so the
tests run on synthetic Black–Scholes prices and the build runs on Massive. Every call is disk-cached by the
HTTP layer, so a re-run is free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..data.cache import CACHE_ROOT
from .pricer import bs_greeks, implied_vol

HIST_DIR = CACHE_ROOT / "vol_history"
TENORS = {"front": 30, "back": 91}
COLS = [f"{m}_{t}" for t in TENORS for m in ("atm", "put25", "call25")]
R, Q = 0.04, 0.0


class Fetch(Protocol):
    def contracts(self, underlying: str, expiry: date, expired: bool) -> list[dict]: ...
    def closes(self, contract: str, start: date, end: date) -> pd.Series: ...


class MassiveFetch:
    def contracts(self, underlying, expiry, expired):
        from ..data import massive
        return massive.option_contracts(underlying, expiry.isoformat(), expired)

    def closes(self, contract, start, end):
        from ..data import massive
        return massive.option_closes(contract, start.isoformat(), end.isoformat())


# ---------------------------------------------------------------- smile → three numbers


def smile_points(S: float, T: float, quotes: pd.DataFrame) -> dict | None:
    """One expiry's smile on one day → ATM, 25Δ put and 25Δ call IV.
    quotes: strike, kind, iv (OTM only). ATM interpolates IV linearly in log-moneyness at k=0; the 25Δ
    points interpolate IV linearly in delta. None if the smile doesn't straddle spot or a 25Δ point."""
    q = quotes.dropna(subset=["iv"])
    q = q[((q.kind == "put") & (q.strike <= S)) | ((q.kind == "call") & (q.strike >= S))]
    if len(q) < 3:
        return None
    q = q.assign(k=np.log(q.strike / S)).sort_values("k")
    below, above = q[q.k <= 0], q[q.k >= 0]
    if below.empty or above.empty:
        return None
    a, b = below.iloc[-1], above.iloc[0]
    atm = a.iv if b.k == a.k else a.iv + (b.iv - a.iv) * (0 - a.k) / (b.k - a.k)

    def at_delta(kind, target):
        s = q[q.kind == kind]
        if s.empty:
            return None
        # anchor the OTM wing at the ATM point (Δ ≈ ±0.5), so a sparse ladder still brackets 25Δ
        pts = [(bs_greeks(S, x.strike, T, R, Q, x.iv, kind).delta, x.iv) for x in s.itertuples()]
        pts.append((bs_greeks(S, S, T, R, Q, atm, kind).delta, atm))
        d, iv = map(np.array, zip(*sorted(pts)))
        if not (d.min() <= target <= d.max()):
            return None
        return float(np.interp(target, d, iv))

    p, c = at_delta("put", -0.25), at_delta("call", 0.25)
    if p is None or c is None:
        return None
    return {"atm": float(atm), "put25": p, "call25": c}


def constant_maturity(points: list[tuple[float, dict]], tenor_days: int) -> dict | None:
    """points: [(T_years, smile_points)] for one day, any order. Interpolate total variance (iv²T) linearly
    in T between the two expiries bracketing the tenor; flat within 15 days if one side is missing."""
    pts = sorted((T, p) for T, p in points if p)
    tau = tenor_days / 365
    lo = [x for x in pts if x[0] <= tau]
    hi = [x for x in pts if x[0] >= tau]
    if not lo or not hi:
        # monthlies only: around a roll there's no expiry on one side. Use the nearest one flat if it is
        # within 15 days of the tenor; otherwise no point (never a long extrapolation).
        near = min(pts, key=lambda x: abs(x[0] - tau)) if pts else None
        return dict(near[1]) if near and abs(near[0] - tau) <= 15 / 365 else None
    (T1, p1), (T2, p2) = lo[-1], hi[0]
    if T2 == T1:
        return dict(p1)
    w = (tau - T1) / (T2 - T1)
    return {m: math.sqrt(max(((1 - w) * p1[m] ** 2 * T1 + w * p2[m] ** 2 * T2) / tau, 0.0)) for m in p1}


def chain_row(chain: pd.DataFrame, S: float) -> dict:
    """Today's row from a live chain with IV filled (market_state.fill_iv): the daily snapshot."""
    pts = []
    for exp, g in chain.groupby("expiry"):
        T = float(g["T"].iloc[0])
        if T <= 2 / 365:
            continue
        pts.append((T, smile_points(S, T, g[["strike", "kind", "iv"]])))
    row = {"spot": S}
    for t, days in TENORS.items():
        cm = constant_maturity(pts, days) or {}
        for m in ("atm", "put25", "call25"):
            row[f"{m}_{t}"] = cm.get(m, np.nan)
    return row


# ---------------------------------------------------------------- backfill


def third_fridays(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        d = date(y, m, 15)
        d += timedelta(days=(4 - d.weekday()) % 7)
        if start <= d <= end:
            out.append(d)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _ladder(listed: list[dict], lo: float, hi: float, n: int) -> list[dict]:
    """~n contracts per kind, log-spaced over [lo, hi], snapped to listed strikes (OTM side of the range)."""
    out = []
    for kind in ("put", "call"):
        ks = sorted({c["strike"] for c in listed if c["kind"] == kind and lo <= c["strike"] <= hi})
        if not ks:
            continue
        targets = np.exp(np.linspace(math.log(lo), math.log(hi), n))
        pick = sorted({min(ks, key=lambda k: abs(math.log(k / t))) for t in targets})
        by_k = {c["strike"]: c for c in listed if c["kind"] == kind}
        out += [by_k[k] for k in pick]
    return out


def backfill(underlying: str, closes: pd.Series, today: date, fetch: Fetch, days: int = 365,
             strikes_per_kind: int = 12, band: float = 0.30, log=print) -> pd.DataFrame:
    """Daily constant-maturity IV history over the last `days`, from historical option closes.
    closes: the underlying's daily closes (date index), covering the window."""
    start = today - timedelta(days=days)
    closes = closes.sort_index()
    closes.index = pd.to_datetime(closes.index).normalize()
    life = TENORS["back"] + 45                      # how long before expiry a contract must be priced
    expiries = third_fridays(start + timedelta(days=5), today + timedelta(days=TENORS["back"] + 40))
    quotes = []                                      # date, expiry, strike, kind, close
    for i, E in enumerate(expiries):
        w0, w1 = max(start, E - timedelta(days=life)), min(today, E - timedelta(days=1))
        if w1 <= w0:
            continue
        s = closes[(closes.index >= pd.Timestamp(w0)) & (closes.index <= pd.Timestamp(w1))]
        if s.empty:
            continue
        lo, hi = float(s.min()) * (1 - band), float(s.max()) * (1 + band)
        listed = fetch.contracts(underlying, E, expired=E < today)
        for c in _ladder(listed, lo, hi, strikes_per_kind):
            px = fetch.closes(c["ticker"], w0, w1)
            for d, v in px.items():
                quotes.append((pd.Timestamp(d).normalize(), E, c["strike"], c["kind"], float(v)))
        log(f"[vol_history] {underlying}: expiry {E} ({i + 1}/{len(expiries)}), {len(quotes)} quotes so far")
    if not quotes:
        return pd.DataFrame(columns=["spot", *COLS, "source"])
    qf = pd.DataFrame(quotes, columns=["date", "expiry", "strike", "kind", "close"])
    rows = {}
    for d, g in qf.groupby("date"):
        if d not in closes.index:
            continue
        S = float(closes.loc[d])
        pts = []
        for E, ge in g.groupby("expiry"):
            T = (pd.Timestamp(E) - d).days / 365
            if T <= 2 / 365:
                continue
            ivs = [implied_vol(x.close, S, x.strike, T, R, Q, x.kind) if x.close > 0 else np.nan for x in ge.itertuples()]
            pts.append((T, smile_points(S, T, ge.assign(iv=ivs)[["strike", "kind", "iv"]])))
        row = {"spot": S}
        for t, dd in TENORS.items():
            cm = constant_maturity(pts, dd) or {}
            for m in ("atm", "put25", "call25"):
                row[f"{m}_{t}"] = cm.get(m, np.nan)
        rows[d] = row
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out["source"] = "backfill"
    out.index.name = "date"
    return out


# ---------------------------------------------------------------- storage


def path(underlying: str, root: Path | None = None) -> Path:
    return (root or HIST_DIR) / f"{underlying.upper()}.csv"


def load(underlying: str, root: Path | None = None) -> pd.DataFrame:
    p = path(underlying, root)
    if not p.exists():
        return pd.DataFrame(columns=["spot", *COLS, "source"])
    return pd.read_csv(p, index_col=0, parse_dates=True)


def save(underlying: str, df: pd.DataFrame, root: Path | None = None) -> Path:
    p = path(underlying, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(p)
    return p


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New rows win on the same date; a snapshot never overwrites a backfilled day with NaNs."""
    if old.empty:
        return new.sort_index()
    both = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    return both[~both.index.duplicated(keep="last")]


def append_snapshot(underlying: str, chain: pd.DataFrame, S: float, today: date, root: Path | None = None) -> pd.DataFrame:
    row = chain_row(chain, S)
    if all(pd.isna(row[c]) for c in COLS):
        return load(underlying, root)
    new = pd.DataFrame([{**row, "source": "snapshot"}], index=pd.DatetimeIndex([pd.Timestamp(today)], name="date"))
    df = merge(load(underlying, root), new)
    save(underlying, df, root)
    return df


def update(tickers: list[str], data, today: date, fetch: Fetch | None = None, days: int = 365, log=print,
           root: Path | None = None) -> dict:
    """Build step: backfill any ticker with no backfilled history, then append today's snapshot for all.
    data: a MarketData (closes, chain). One bad ticker never kills the build."""
    from .market_state import fill_iv
    fetch = fetch or MassiveFetch()
    report = {}
    for u in tickers:
        try:
            df = load(u, root)
            has_backfill = len(df) > 0 and bool((df["source"] == "backfill").any())
            if not has_backfill:
                cl = data.closes([u], days=days + 40)[u].dropna()
                df = merge(df, backfill(u, cl, today, fetch, days=days, log=log))
                save(u, df, root)
            S = float(data.closes([u], days=10)[u].dropna().iloc[-1])
            ch = data.chain(u, today, today + timedelta(days=TENORS["back"] + 40), 0.35)
            df = append_snapshot(u, fill_iv(ch, S, today), S, today, root)
            report[u] = {"rows": int(len(df)), "first": str(df.index.min().date()) if len(df) else None}
        except Exception as e:  # noqa: BLE001
            report[u] = {"error": str(e)[:120]}
            log(f"[vol_history] {u}: {e}")
    return report
