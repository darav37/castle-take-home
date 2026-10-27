"""PMRM stage 0: data and power audit (docs/research/pmrm_spec.md §0.1, §0.1a, §2, §3, §6; BUILD_PLAN P0). No LLM calls.

  python scripts/pmrm_audit.py [--skip-pools] [--reps 300]

What it does
1. Resolves the Study A seed contracts and audits each one over its whole life: listed/close/settled dates, eligible
   16:00 ET endpoints, valid quotes under the §6 rules, intervals, jump days, staleness, independent episodes.
2. Builds the proposed Study B pools by category (recurring series first) as rolling "front contract" variables and
   audits them the same way; stocks (incl. placebos) get total-return coverage.
3. Pairwise overlap of clean intervals, and the Study A complete-case row count.
4. Simulates the planned primary comparison (M2c vs matched-k M2) at the real row counts (§6 power gate).
Writes docs/pmrm/results/audit.md and audit.json. Every external call goes through the disk cache; the audit end
date is fixed so reruns hit the cache.

Quote rules (§6): midpoint; 0 <= bid <= ask <= 1; spread <= 0.04; quote at or before the endpoint, never after.
Kalshi hourly candles exist only for hours with a trade or a quote change, so the 16:00 ET book is either a candle
ending exactly at 16:00 ("strict": age 0) or the last earlier candle carried forward ("carried": a standing quote
that has not changed; its last-update age is reported). §6's 900 s age limit can't be checked on hourly candles under
the carried reading; both readings are reported and the choice is the trainer's (audit.md, DECISIONS).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey.data import kalshi, massive  # noqa: E402

OUT_DIR = ROOT / "docs" / "pmrm" / "results"
ET = "America/New_York"
AUDIT_END = date(2026, 10, 16)            # last full session before the audit (Fri); fixed so reruns hit the cache
POOL_START = date(2022, 1, 3)             # Study B window: recurring series from 2022 (two years was too short)
SPEC = {"max_spread": 0.04, "p_lo": 0.05, "p_hi": 0.95, "dp_cap": 0.05, "elapsed": (23.0, 25.0),
        "jump_flag": 0.03, "train": 120, "val": 40, "test_min": 40, "refit": 20, "seed": 1729}
# NYSE shortened sessions in range (13:00 close). No exchange-calendar package in the env; listed by hand, checked
# against NYSE holiday notices. Intervals touching them are excluded (§2 interval policy).
EARLY_CLOSES = {date(2023, 7, 3), date(2023, 11, 24), date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
                date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24), date(2026, 11, 27), date(2026, 12, 24)}

SEEDS = {   # Study A (§2). Resolved Oct 18; see audit.md for the rule text and the threshold caveat on event_A.
    "event_A": {"series": "KXWTIMAX", "ticker": "KXWTIMAX-26DEC31-T150", "theme": "oil record high"},
    "event_B": {"series": "KXRECSSNBER", "ticker": "KXRECSSNBER-26", "theme": "US recession"},
    "event_C": {"series": "KXFEDDECISION", "ticker": "KXFEDDECISION-26DEC-C25", "theme": "exactly 25 bp Dec cut"},
}
SEED_STOCKS = ["XOM", "DAL", "MSFT"]

# Study B pools (§0.1a): recurring series first. Each family is a rolling variable: the front unresolved event, and
# within it the market whose mid is nearest 0.5 on the first session it becomes front (chosen with information at
# that time only; never re-picked). An interval counts only if both endpoints are on the same market.
POOLS = {
    "fed": [("KXFEDDECISION", None)],
    "cpi": [("KXCPI", None)],
    "payrolls": [("KXPAYROLLS", None)],
    "growth_recession": [("KXRECSSNBER", None), ("KXGDP", None)],
    "energy_commodities": [("KXWTIMAX", "DEC31"), ("KXWTIMAX", "monthly")],
    "trade_geopolitics": [("KXHORMUZNORM", None), ("KXUSAIRANAGREEMENT", None), ("KXEFFTARIFF", None)],
    "elections": [("CONTROLH", None), ("CONTROLS", None)],
}
STOCK_POOL = {
    "rate_sensitive": ["JPM", "BAC", "DHI", "LEN", "NEE", "DUK"],
    "energy": ["XOM", "CVX", "COP", "OXY"],
    "travel_consumer": ["DAL", "UAL", "MAR", "HD"],
    "megacap_tech": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"],
    "defensives": ["PG", "KO", "JNJ", "WMT"],
    "placebo": ["ROL", "WM", "CHD", "VRSK", "ORLY"],
}


# ---------------------------------------------------------------- calendar and endpoints

def sessions(start: date, end: date) -> pd.DataFrame:
    """Full XNYS sessions from SPY's daily bars (one bar per session), with the 16:00 ET endpoint in UTC."""
    spy = massive.daily_bars("SPY", start.isoformat(), end.isoformat(), trim_to_listing=False)
    days = [d.date() for d in spy.index]
    ep = [pd.Timestamp(datetime(d.year, d.month, d.day, 16, 0), tz=ET).tz_convert("UTC") for d in days]
    return pd.DataFrame({"day": days, "endpoint": ep, "early": [d in EARLY_CLOSES for d in days]})


def intervals(cal: pd.DataFrame) -> pd.DataFrame:
    """Eligible core daily intervals: consecutive full sessions 23-25 h apart (no weekends/holidays, no early closes)."""
    c = cal.reset_index(drop=True)
    rows = []
    for i in range(1, len(c)):
        a, b = c.iloc[i - 1], c.iloc[i]
        hrs = (b.endpoint - a.endpoint).total_seconds() / 3600
        ok = SPEC["elapsed"][0] <= hrs <= SPEC["elapsed"][1] and not a.early and not b.early
        rows.append({"start": a.endpoint, "end": b.endpoint, "day": b.day, "elapsed_h": hrs, "eligible": ok})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Kalshi

def _t(s) -> pd.Timestamp | None:
    return pd.Timestamp(s).tz_convert("UTC") if s else None


def all_markets(series: str) -> list[dict]:
    """Every market in a series, live and historical, with raw times and rules."""
    out, seen = [], set()
    for path, hist in (("/historical/markets", True), ("/markets", False)):
        try:
            rows = kalshi._paged(path, {"series_ticker": series, "limit": 1000}, "markets")
        except Exception as e:  # noqa: BLE001  (a missing series is reported, not fatal)
            print(f"  {series} {path}: {type(e).__name__}", file=sys.stderr)
            rows = []
        for m in rows:
            if m["ticker"] in seen:
                continue
            seen.add(m["ticker"])
            out.append({"ticker": m["ticker"], "event": m.get("event_ticker"), "series": series, "historical": hist,
                        "open": _t(m.get("open_time")), "close": _t(m.get("close_time")),
                        "settled": _t(m.get("settlement_ts")), "status": m.get("status"), "result": m.get("result"),
                        "sub": m.get("yes_sub_title"), "rules": m.get("rules_primary"), "title": m.get("title"),
                        "volume": kalshi._f(m.get("volume_fp")) or 0.0})
    return out


def candles(m: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    a = max(start, m["open"]) if m["open"] is not None else start
    b = min(end, m["settled"]) if m["settled"] is not None else end
    if b <= a:
        return kalshi.parse_candles([])
    return kalshi.hourly(m["series"], m["ticker"], a.to_pydatetime(), b.to_pydatetime(), m["historical"])


def endpoint_quotes(h: pd.DataFrame, endpoints: pd.Series) -> pd.DataFrame:
    """Book at each endpoint: last candle ending at or before it (never after). strict = that candle ends exactly at
    the endpoint; carried = any earlier candle. Volume traded since the previous endpoint, for staleness."""
    out = []
    idx = h.index
    cumvol = h["volume"].cumsum().to_numpy() if len(h) else np.array([])
    prev_cv = 0.0
    for ep in endpoints:
        j = idx.searchsorted(ep, side="right") - 1 if len(h) else -1
        if j < 0:
            out.append({"bid": np.nan, "ask": np.nan, "age_h": np.nan, "strict": False, "vol": 0.0})
            continue
        r = h.iloc[j]
        cv = float(cumvol[j])
        age = (ep - idx[j]).total_seconds() / 3600
        out.append({"bid": r["bid"], "ask": r["ask"], "age_h": age, "strict": age == 0.0, "vol": cv - prev_cv})
        prev_cv = cv
    q = pd.DataFrame(out, index=endpoints.index)
    q["mid"] = (q["bid"] + q["ask"]) / 2
    q["spread"] = q["ask"] - q["bid"]
    q["quote_ok"] = (q["bid"].notna() & q["ask"].notna() & (q["bid"] >= 0) & (q["ask"] <= 1) & (q["bid"] <= q["ask"])
                     & (q["spread"] <= SPEC["max_spread"] + 1e-9))
    return q


def variable_rows(iv: pd.DataFrame, q: pd.DataFrame, market: pd.Series, life_end: pd.Series) -> pd.DataFrame:
    """Per eligible interval: dp, validity under both readings, interior start, staleness. `q` is indexed by endpoint
    time; `market` gives the market id used at each endpoint (same market at both ends required)."""
    r = iv.copy()
    qs, qe = q.reindex(r["start"]).reset_index(drop=True), q.reindex(r["end"]).reset_index(drop=True)
    ms, me = market.reindex(r["start"]).reset_index(drop=True), market.reindex(r["end"]).reset_index(drop=True)
    le = life_end.reindex(r["end"]).reset_index(drop=True)
    r = r.reset_index(drop=True)
    same = ms.notna() & (ms == me)
    alive = le.notna() & (r["end"] < le)           # excludes the interval containing resolution/close and after
    base = r["eligible"] & same & alive & qs["quote_ok"] & qe["quote_ok"]
    r["market"] = me.where(same)
    r["p_start"], r["p_end"] = qs["mid"], qe["mid"]
    r["dp"] = (qe["mid"] - qs["mid"]).where(base)
    r["valid_carried"] = base
    r["valid_strict"] = base & qs["strict"] & qe["strict"]
    r["interior"] = (qs["mid"] >= SPEC["p_lo"]) & (qs["mid"] <= SPEC["p_hi"])
    r["stale"] = base & (qe["vol"] <= 0) & (qe["mid"] == qs["mid"])
    r["spread_end"], r["age_end_h"] = qe["spread"], qe["age_h"]
    r["in_life"] = r["eligible"] & alive & ms.notna()
    return r


def episodes(days: list, gap: int = 5) -> int:
    """Independent jump episodes: jump days more than `gap` sessions apart count separately."""
    n, last = 0, None
    for d in days:
        if last is None or d - last > gap:
            n += 1
        last = d
    return n


def summarize(rows: pd.DataFrame, cal_pos: dict) -> dict:
    out = {"intervals_in_life": int(rows["in_life"].sum())}
    for rd in ("carried", "strict"):
        v = rows[rows[f"valid_{rd}"]]
        clean = v[v["interior"]]
        big3 = clean[clean["dp"].abs() >= SPEC["jump_flag"]]
        big5 = clean[clean["dp"].abs() >= SPEC["dp_cap"]]
        out[rd] = {"valid": int(len(v)), "clean": int(len(clean)), "stale": int(v["stale"].sum()),
                   "stale_share": round(float(v["stale"].mean()), 3) if len(v) else None,
                   "jumps_3pp": int(len(big3)), "jumps_5pp": int(len(big5)),
                   "jump_episodes_3pp": episodes(sorted(cal_pos[d] for d in big3["day"])),
                   "within_cap": int((clean["dp"].abs() <= SPEC["dp_cap"]).sum()),
                   "first_clean": str(clean["day"].min()) if len(clean) else None,
                   "last_clean": str(clean["day"].max()) if len(clean) else None,
                   "median_spread": round(float(v["spread_end"].median()), 4) if len(v) else None,
                   "median_age_h": round(float(v["age_end_h"].median()), 1) if len(v) else None,
                   "p95_age_h": round(float(v["age_end_h"].quantile(0.95)), 1) if len(v) else None,
                   "sd_dp": round(float(clean["dp"].std()), 4) if len(clean) > 2 else None,
                   "excess_kurtosis_dp": round(float(clean["dp"].kurt()), 1) if len(clean) > 10 else None}
    return out


# ---------------------------------------------------------------- seeds (fixed contract) and pools (rolling front)

def fixed_variable(m: dict, cal: pd.DataFrame, iv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    start = cal["endpoint"].iloc[0] - pd.Timedelta(days=10)
    h = candles(m, start, cal["endpoint"].iloc[-1] + pd.Timedelta(hours=1))
    eps = cal.set_index("endpoint", drop=False)["endpoint"]
    q = endpoint_quotes(h, eps)
    ix = pd.DatetimeIndex(eps)
    q.index = ix
    life = [m["open"] is not None and ep >= m["open"] for ep in eps]
    market = pd.Series([m["ticker"] if ok else None for ok in life], index=ix)
    end = min(x for x in (m["close"], m["settled"]) if x is not None) if (m["close"] or m["settled"]) else pd.Timestamp.max.tz_localize("UTC")
    life_end = pd.Series([end] * len(eps), index=ix)
    return variable_rows(iv, q, market, life_end), {"candles": int(len(h))}


def rolling_variable(series: str, flt: str | None, cal: pd.DataFrame, iv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ms = all_markets(series)
    if flt == "DEC31":
        ms = [m for m in ms if (m["event"] or "").endswith("DEC31") or (m["event"] or "").endswith("DEC29")]
    elif flt == "monthly":
        ms = [m for m in ms if not ((m["event"] or "").endswith("DEC31") or (m["event"] or "").endswith("DEC29"))]
    ms = [m for m in ms if m["close"] is not None]
    events = {}
    for m in ms:
        events.setdefault(m["event"], []).append(m)
    # the event's deadline is its LAST market close: touch-threshold strikes close early when hit (Oct 18 fix)
    ev_close = {e: max(x["close"] for x in v) for e, v in events.items()}
    eps = list(cal["endpoint"])
    lo, hi = eps[0], eps[-1]
    market = pd.Series([None] * len(eps), index=eps, dtype=object)
    life_end = pd.Series([pd.NaT] * len(eps), index=eps, dtype=object)
    qall = pd.DataFrame(index=eps, columns=["bid", "ask", "age_h", "strict", "vol", "mid", "spread", "quote_ok"])
    n_events, n_candle_calls, n_picks = 0, 0, 0
    order = sorted(ev_close, key=ev_close.get)
    for i, e in enumerate(order):
        c = ev_close[e]
        prev_c = ev_close[order[i - 1]] if i else None
        win = [ep for ep in eps if ep < c and (prev_c is None or ep >= prev_c) and lo <= ep <= hi]
        if not win:
            continue
        mkts = {m["ticker"]: m for m in events[e] if m["open"] is not None and m["open"] < c}
        if not mkts:
            continue
        w0 = win[0] - pd.Timedelta(days=10)
        qs = {}
        for t, m in mkts.items():
            h = candles(m, w0, win[-1] + pd.Timedelta(hours=1))
            n_candle_calls += 1
            qs[t] = endpoint_quotes(h, pd.Series(win, index=win))
        own_end = {t: min(x for x in (m["close"], m["settled"]) if x is not None) for t, m in mkts.items()}
        chosen, used = None, False
        for ep in win:
            if chosen is not None and ep >= own_end[chosen]:
                chosen = None                       # the chosen market resolved (e.g. a strike was touched): re-pick
            if chosen is None:   # nearest 0.5 among live markets with a valid quote now; information at ep only
                cands = [(abs(qs[t].at[ep, "mid"] - 0.5), t) for t in qs if ep < own_end[t] and qs[t].at[ep, "quote_ok"]]
                if not cands:
                    continue
                chosen = min(cands)[1]
                n_picks += 1
            used = True
            market[ep] = chosen
            life_end[ep] = own_end[chosen]
            qall.loc[ep] = qs[chosen].loc[ep]
        n_events += used
    for col in ("bid", "ask", "age_h", "vol", "mid", "spread"):
        qall[col] = pd.to_numeric(qall[col])
    qall["strict"] = qall["strict"].fillna(False).astype(bool)
    qall["quote_ok"] = qall["quote_ok"].fillna(False).astype(bool)
    rows = variable_rows(iv, qall, market, pd.Series(pd.to_datetime(life_end.values, utc=True), index=eps))
    return rows, {"markets": len(ms), "events_front": n_events, "picks": n_picks, "candle_calls": n_candle_calls}


# ---------------------------------------------------------------- stocks (total return)

def stock_returns(t: str, cal: pd.DataFrame, iv: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Interval total returns: split-adjusted closes (Massive adjusted=true adjusts splits, not dividends) plus cash
    dividends on the ex-date. Returns NaN where an endpoint close is missing."""
    s0, s1 = cal["day"].iloc[0], cal["day"].iloc[-1]
    px = massive.daily_bars(t, (s0 - timedelta(days=7)).isoformat(), s1.isoformat(), trim_to_listing=False)["close"]
    px.index = [d.date() for d in px.index]
    divs = pd.Series(0.0, index=px.index)
    n_div = 0
    for d in massive.dividends(t):
        ex = d.get("ex_dividend_date")
        if ex and date.fromisoformat(ex) in divs.index and (d.get("dividend_type") in (None, "CD", "SC")):
            divs[date.fromisoformat(ex)] += float(d.get("cash_amount") or 0)
            n_div += 1
    by_ep = {e: d for e, d in zip(cal["endpoint"], cal["day"])}
    r, pr = [], []
    for a, b in zip(iv["start"], iv["end"]):
        da, db = by_ep[a], by_ep[b]
        if da in px.index and db in px.index:
            r.append((px[db] + divs[db]) / px[da] - 1)
            pr.append(px[db] / px[da] - 1)
        else:
            r.append(np.nan)
            pr.append(np.nan)
    tr = pd.Series(r, index=iv.index)
    gap = (tr - pd.Series(pr, index=iv.index)).abs()
    ok = iv["eligible"] & tr.notna()
    return tr.where(iv["eligible"]), {"valid": int(ok.sum()), "eligible": int(iv["eligible"].sum()),
                                      "dividends_in_window": n_div, "max_div_return": round(float(gap.max()), 4) if gap.notna().any() else 0.0,
                                      "excess_kurtosis": round(float(tr[ok].kurt()), 1) if ok.sum() > 10 else None,
                                      "ac1": round(float(tr[ok].autocorr(1)), 3) if ok.sum() > 10 else None}


# ---------------------------------------------------------------- power gate: simulate M2c vs matched-k M2

def fa_fit(S: np.ndarray, k: int, iters: int = 300, psi_floor: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """ML factor analysis on a correlation-scale matrix by EM (Rubin & Thayer); deterministic start from PCA."""
    d = S.shape[0]
    w, V = np.linalg.eigh(S)
    L = V[:, ::-1][:, :k] * np.sqrt(np.maximum(w[::-1][:k], 1e-6))
    psi = np.maximum(np.diag(S) - (L ** 2).sum(1), psi_floor)
    for _ in range(iters):
        Pinv = 1 / psi
        M = np.eye(k) + (L.T * Pinv) @ L
        Minv = np.linalg.inv(M)
        beta = Minv @ (L.T * Pinv)                         # k x d
        Ezz = Minv + beta @ S @ beta.T
        L_new = S @ beta.T @ np.linalg.inv(Ezz)
        psi = np.maximum(np.diag(S - L_new @ beta @ S), psi_floor)
        L = L_new
    return L, psi


def psd(S: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.maximum(w, floor)) @ V.T


def fit_m2(U: np.ndarray, k: int, mask: np.ndarray | None = None, topup: bool = True) -> np.ndarray:
    """Covariance (original units) of a k-factor model. With `mask` (True = masked in the loading step): moments and
    pairwise covariances from unmasked entries, factor fit on that, then the §4.2 variance top-up to raw variances.
    (The real build fits the masked likelihood directly; pairwise moments are the simulation's approximation.)"""
    if mask is None:
        mu, s = U.mean(0), U.std(0)
        Z = (U - mu) / s
        L, psi = fa_fit(Z.T @ Z / len(Z), k)
        return (L @ L.T + np.diag(psi)) * np.outer(s, s), mu
    X = np.where(mask, np.nan, U)
    mu, s = np.nanmean(X, 0), np.nanstd(X, 0)
    Z = (X - mu) / s
    d = U.shape[1]
    C = np.empty((d, d))
    for a in range(d):
        for b in range(a, d):
            ok = ~np.isnan(Z[:, a]) & ~np.isnan(Z[:, b])
            C[a, b] = C[b, a] = (Z[ok, a] * Z[ok, b]).mean()
    L, psi = fa_fit(psd(C), k)
    Sig = (L @ L.T + np.diag(psi)) * np.outer(s, s)
    if topup:
        v_raw = U.var(0)
        np.fill_diagonal(Sig, np.maximum(np.diag(Sig), v_raw))
    return Sig, mu


def cond_portfolio(Sig: np.ndarray, mu: np.ndarray, h: np.ndarray, xi: int, stock_idx: list[int], x: np.ndarray):
    """Gaussian conditional of portfolio P&L h'r given event change x (same formulas as §5)."""
    hs = np.zeros(len(mu))
    hs[stock_idx] = h
    m0 = hs @ mu
    cov_px = hs @ Sig[:, xi]
    vxx = Sig[xi, xi]
    v = hs @ Sig @ hs - cov_px ** 2 / vxx
    return m0 + cov_px / vxx * (x - mu[xi]), max(v, 1e-12)


def nls(y, m, v):
    return 0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)


def simulate(n_rows: int, pi: float, jump_sd: float, rng, nu: float = 6.0, phi: float = 0.0, recall: float = 0.85,
             false_flag: float = 0.15, k: int = 1, block: int = 5, n_boot: int = 500, shared: bool = False):
    """One replication of the planned primary at n_rows: rolling 120-row training, refit every 20, first 160 rows are
    training + validation (validation unused here: k is matched and fixed), the rest scored raw; conditioning event =
    event_B, portfolio = equal $ in the three stocks; scored rows need |x| <= 3 training sd (§6 support gate)."""
    lam_e, lam_s = np.array([0.30, 0.45, 0.35]), np.array([0.60, 0.70, 0.65])
    lam = np.concatenate([lam_e, lam_s])
    d = 6
    f = np.zeros(n_rows)
    tdraw = lambda size: rng.standard_t(nu, size) / math.sqrt(nu / (nu - 2))
    eps_f = tdraw(n_rows)
    for t in range(n_rows):
        f[t] = phi * (f[t - 1] if t else 0.0) + math.sqrt(1 - phi ** 2) * eps_f[t]
    E = tdraw((n_rows, d)) * np.sqrt(1 - lam ** 2)
    Uc = f[:, None] * lam + E                                   # clean data, unit variances
    if shared:   # venue-wide bad prints: the same spurious move on every event that day (a fake event factor)
        J = ((rng.random(n_rows) < pi) * rng.normal(0, jump_sd, n_rows))[:, None] * np.ones(3)
    else:        # idiosyncratic event jumps (one contract's bad print or own news)
        J = (rng.random((n_rows, 3)) < pi) * rng.normal(0, jump_sd, (n_rows, 3))
    U = Uc.copy()
    U[:, :3] += J                                               # idiosyncratic event jumps (bad prints, own news)
    flagged = (np.abs((U - U.mean(0)) / U.std(0)) >= 2.5)[:, :3]   # J2 flag rule (§6 flag_abs_standardized), events
    is_jump = J != 0
    label_idio = flagged & np.where(is_jump, rng.random((n_rows, 3)) < recall, rng.random((n_rows, 3)) < false_flag)
    mask = np.zeros_like(U, dtype=bool)
    mask[:, :3] = label_idio
    xi, stocks, h = 1, [3, 4, 5], np.ones(3)
    start = SPEC["train"] + SPEC["val"]
    diffs, days = [], []
    for s0 in range(start, n_rows, SPEC["refit"]):
        tr = slice(s0 - SPEC["train"], s0)
        S2, mu2 = fit_m2(U[tr], k)
        Sc, muc = fit_m2(U[tr], k, mask=mask[tr])
        sdx = U[tr, xi].std()
        for t in range(s0, min(s0 + SPEC["refit"], n_rows)):
            x = U[t, xi]
            if abs(x - U[tr, xi].mean()) > 3 * sdx:
                continue
            y = h @ U[t, stocks]
            m2, v2 = cond_portfolio(S2, mu2, h, xi, stocks, x)
            mc, vc = cond_portfolio(Sc, muc, h, xi, stocks, x)
            diffs.append(nls(y, m2, v2) - nls(y, mc, vc))       # > 0: M2c better
            days.append(t)
    diffs = np.array(diffs)
    if len(diffs) < 10:
        return np.nan, False, len(diffs), np.nan
    # day-block percentile-t bootstrap (Oct 18: the plain percentile bootstrap over-rejected, ~10% size at 250 rows)
    nb = len(diffs) // block
    b = diffs[: nb * block].reshape(nb, block).mean(1)          # non-overlapping day blocks
    se = b.std(ddof=1) / math.sqrt(nb)
    t_obs = b.mean() / se
    bs = b[rng.integers(0, nb, (n_boot, nb))]
    t_star = (bs.mean(1) - b.mean()) / (bs.std(1, ddof=1) / math.sqrt(nb) + 1e-12)
    return float(diffs.mean()), bool(t_obs > np.quantile(t_star, 0.95)), len(diffs), float(se)


def power_gate(row_counts: dict, reps: int, nu: float, phi: float) -> dict:
    rng = np.random.default_rng(SPEC["seed"])
    grid = [(0.0, 0.0, False), (0.05, 4.0, False), (0.10, 6.0, False), (0.20, 6.0, False),
            (0.03, 4.0, True), (0.05, 6.0, True), (0.10, 6.0, True)]
    out = {"assumptions": {"k_true": 1, "t_df": nu, "factor_ac1": phi, "recall": 0.85, "false_flag_on_real_moves": 0.15,
                           "block": 5, "bootstrap": 500, "test": "one-sided 5%, day-block percentile-t bootstrap of the mean paired NLS difference",
                           "reps": reps, "loadings_events": [0.30, 0.45, 0.35], "loadings_stocks": [0.60, 0.70, 0.65]},
           "cells": []}
    for label, n in row_counts.items():
        for pi, js, shared in grid:
            res = [simulate(n, pi, js, rng, nu=nu, phi=phi, shared=shared) for _ in range(reps)]
            eff = [r[0] for r in res if not math.isnan(r[0])]
            sig = [r[1] for r in res if not math.isnan(r[0])]
            out["cells"].append({"universe": label, "rows": n, "test_rows": max(n - SPEC["train"] - SPEC["val"], 0),
                                 "contamination": pi, "jump_sd": js, "kind": "shared" if shared else "idiosyncratic",
                                 "mean_gain_nats": round(float(np.mean(eff)), 4) if eff else None,
                                 "se_nats": round(float(np.median([r[3] for r in res if not math.isnan(r[0])])), 4) if eff else None,
                                 "power": round(float(np.mean(sig)), 3) if sig else None})
            c = out["cells"][-1]
            print(f"  power {label:20} n={n:4d} {c['kind'][:5]} pi={pi:.2f} J={js:.0f}: gain={c['mean_gain_nats']} "
                  f"se={c['se_nats']} power={c['power']}", flush=True)
    return out


def mde(cells: list[dict], universe: str) -> float | None:
    """Smallest mean gain (nats/row) the one-sided 5% block-bootstrap test detects with 80% power:
    (z_0.95 + z_0.80) x the median bootstrap SE of the mean paired difference at this row count."""
    se = [c["se_nats"] for c in cells if c["universe"] == universe and c["se_nats"] is not None]
    return round(2.486 * float(np.median(se)), 4) if se else None


# ---------------------------------------------------------------- main

def complete_case(vars_rows: dict, stocks: dict, cond: str, rd: str) -> pd.Series:
    """Study-universe rows: every event valid and interior at the start, conditioning |dp| <= cap, every stock valid."""
    ok = None
    for n, r in vars_rows.items():
        o = r[f"valid_{rd}"] & r["interior"]
        if n == cond:
            o &= r["dp"].abs() <= SPEC["dp_cap"]
        ok = o if ok is None else ok & o
    for t, s in stocks.items():
        ok &= s.notna()
    return ok


def overlap(vars_rows: dict, rd: str) -> pd.DataFrame:
    names = list(vars_rows)
    clean = {n: vars_rows[n][f"valid_{rd}"] & vars_rows[n]["interior"] for n in names}
    return pd.DataFrame([[int((clean[a] & clean[b]).sum()) for b in names] for a in names], index=names, columns=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pools", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"audit_end": AUDIT_END.isoformat(), "spec": SPEC}

    # Study A
    first_open = date(2025, 7, 1)
    cal = sessions(first_open, AUDIT_END)
    iv = intervals(cal)
    pos = {d: i for i, d in enumerate(cal["day"])}
    seeds, seed_meta = {}, {}
    for name, sd in SEEDS.items():
        m = next((x for x in all_markets(sd["series"]) if x["ticker"] == sd["ticker"]), None)
        if m is None:
            seed_meta[name] = {"error": "not found"}
            continue
        rows, info = fixed_variable(m, cal, iv)
        seeds[name] = rows
        seed_meta[name] = {**sd, "title": m["title"], "sub": m["sub"], "rules": m["rules"], "open": str(m["open"]),
                           "close": str(m["close"]), "settled": str(m["settled"]), "status": m["status"],
                           "volume": m["volume"], **info, **summarize(rows, pos)}
        print(name, sd["ticker"], seed_meta[name]["carried"]["clean"], "clean (carried)", flush=True)
    stocks, stock_meta = {}, {}
    for t in SEED_STOCKS:
        stocks[t], stock_meta[t] = stock_returns(t, cal, iv)
    report["study_a"] = {"seeds": seed_meta, "stocks": stock_meta,
                         "overlap_carried": overlap(seeds, "carried").to_dict(),
                         "overlap_strict": overlap(seeds, "strict").to_dict()}
    rows_a = {rd: complete_case(seeds, stocks, "event_B", rd) for rd in ("carried", "strict")}
    report["study_a"]["rows"] = {rd: int(v.sum()) for rd, v in rows_a.items()}
    report["study_a"]["rows_first_last"] = {rd: [str(iv.loc[v, "day"].min()), str(iv.loc[v, "day"].max())] if v.any() else None
                                            for rd, v in rows_a.items()}
    print("Study A rows:", report["study_a"]["rows"], flush=True)

    # Study B pools
    pool_rows, pool_meta = {}, {}
    if not args.skip_pools:
        calb = sessions(POOL_START, AUDIT_END)
        ivb = intervals(calb)
        posb = {d: i for i, d in enumerate(calb["day"])}
        for cat, fams in POOLS.items():
            for series, flt in fams:
                key = f"{cat}:{series}" + (f":{flt}" if flt else "")
                try:
                    rows, info = rolling_variable(series, flt, calb, ivb)
                except Exception as e:  # noqa: BLE001
                    pool_meta[key] = {"error": f"{type(e).__name__}: {e}"[:200]}
                    print(key, "ERROR", pool_meta[key]["error"], flush=True)
                    continue
                pool_rows[key] = rows
                pool_meta[key] = {**info, **summarize(rows, posb)}
                print(key, info, pool_meta[key]["carried"]["clean"], "clean (carried)", flush=True)
        sb, sbm = {}, {}
        for cat, ts in STOCK_POOL.items():
            for t in ts:
                try:
                    sb[t], sbm[t] = stock_returns(t, calb, ivb)
                    sbm[t]["category"] = cat
                except Exception as e:  # noqa: BLE001
                    sbm[t] = {"category": cat, "error": f"{type(e).__name__}"}
        report["study_b"] = {"window": [POOL_START.isoformat(), AUDIT_END.isoformat()], "families": pool_meta,
                             "stocks": sbm, "overlap_carried": overlap(pool_rows, "carried").to_dict() if pool_rows else {},
                             "overlap_strict": overlap(pool_rows, "strict").to_dict() if pool_rows else {}}
        # candidate substitutes for Study A (§0.1a: same themes, recurring series first). Oil has no long-lived Kalshi
        # series, so the last two keep the oil theme as a context variable (USO total return), not an event.
        cands = {"oil+recession+fed (events)": ["energy_commodities:KXWTIMAX:DEC31", "growth_recession:KXRECSSNBER", "fed:KXFEDDECISION"],
                 "brent_weekly+recession+fed": ["energy_commodities:KXBRENTW", "growth_recession:KXRECSSNBER", "fed:KXFEDDECISION"],
                 "recession+fed": ["growth_recession:KXRECSSNBER", "fed:KXFEDDECISION"],
                 "recession+fed+cpi": ["growth_recession:KXRECSSNBER", "fed:KXFEDDECISION", "cpi:KXCPI"],
                 "recession+fed+payrolls": ["growth_recession:KXRECSSNBER", "fed:KXFEDDECISION", "payrolls:KXPAYROLLS"]}
        if "energy_commodities:KXBRENTW" not in pool_rows:
            try:
                pool_rows["energy_commodities:KXBRENTW"], info = rolling_variable("KXBRENTW", None, calb, ivb)
                pool_meta["energy_commodities:KXBRENTW"] = {**info, **summarize(pool_rows["energy_commodities:KXBRENTW"], posb)}
            except Exception:  # noqa: BLE001
                pass
        st_sub = {t: sb[t] for t in SEED_STOCKS if t in sb}
        uso, _ = stock_returns("USO", calb, ivb)
        subs = {}
        for name, keys in cands.items():
            if not all(k in pool_rows for k in keys):
                continue
            ev = {k: pool_rows[k] for k in keys}
            cond = "growth_recession:KXRECSSNBER"
            subs[name] = {rd: int(complete_case(ev, st_sub, cond, rd).sum()) for rd in ("carried", "strict")}
            if "KXWTIMAX" not in " ".join(keys) and "BRENTW" not in " ".join(keys):
                subs[name + " + USO context"] = {rd: int(complete_case(ev, {**st_sub, "USO": uso}, cond, rd).sum())
                                                 for rd in ("carried", "strict")}
        report["substitutes"] = subs
        print("substitutes:", subs, flush=True)

    # power gate at the real row counts
    kurts = [m["excess_kurtosis"] for m in stock_meta.values() if m.get("excess_kurtosis") is not None]
    k_ex = float(np.median(kurts)) if kurts else 3.0
    nu = float(min(max(4 + 6 / max(k_ex, 0.2), 4.5), 30))
    acs = [m["ac1"] for m in stock_meta.values() if m.get("ac1") is not None]
    phi = float(np.clip(np.median(acs) if acs else 0.0, -0.3, 0.3))
    counts = {"study_a_carried": max(report["study_a"]["rows"]["carried"], 0)}
    best = max((v["carried"], k) for k, v in report.get("substitutes", {"none": {"carried": 0}}).items())
    counts[f"best_substitute ({best[1]})"] = best[0]
    counts.update({"reference_500": 500, "reference_1000": 1000})
    counts = {k: v for k, v in counts.items() if v >= SPEC["train"] + SPEC["val"] + 10}
    report["power"] = power_gate(counts, args.reps, nu, phi)
    report["power"]["mde_80"] = {u: mde(report["power"]["cells"], u) for u in counts}
    report["power"]["size_at_null"] = {u: next((c["power"] for c in report["power"]["cells"]
                                                if c["universe"] == u and c["contamination"] == 0.0), None) for u in counts}

    (OUT_DIR / "audit.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT_DIR / "audit.json")


if __name__ == "__main__":
    main()
