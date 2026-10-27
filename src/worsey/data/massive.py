"""Massive client: daily bars, option chains, ETF holdings, reference data.

Verified live in CP-L (SPEC §10b): base https://api.massive.com, `Authorization: Bearer`.
- Option chain snapshot is paginated (≤ 250/page, `next_url`), nearest expiries first (F6),
  and carries vendor IV/greeks on only ~70% of contracts (F7).
- ETF holdings come from ETF Global constituents with many effective dates per fund (F8).
- Ticker history can predate the listing (F10), so bars are trimmed to `list_date`.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlparse

import numpy as np
import pandas as pd

from .http import get_json

BASE = "https://api.massive.com"


def _headers() -> dict:
    key = os.environ.get("MASSIVE_API_KEY", "")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY not set (put it in .env)")
    return {"Authorization": f"Bearer {key}"}


def _get(path: str, params: dict | None = None) -> dict:
    return get_json("massive", BASE, path, params, _headers)  # headers built only if the call isn't cached


def _paged(path: str, params: dict, max_pages: int = 100) -> list[dict]:
    """Follow next_url. The cursor becomes a normal param, so every page is disk-cached."""
    out, p = [], dict(params)
    for _ in range(max_pages):
        data = _get(path, p)
        out += data.get("results") or []
        nxt = data.get("next_url")
        if not nxt:
            break
        u = urlparse(nxt)
        path, p = u.path, dict(parse_qsl(u.query))
        p.pop("apiKey", None)
    return out


# ---------------------------------------------------------------- reference + bars


def overview(ticker: str) -> dict:
    """Ticker overview (cached). {} if Massive has none: callers degrade, never crash."""
    try:
        return _get(f"/v3/reference/tickers/{ticker}").get("results") or {}
    except Exception:  # noqa: BLE001
        return {}


def _get_live(path: str, params: dict | None = None) -> dict:
    """A live call, made even when market data is offline: cached per minute, so it is at most a minute old."""
    from datetime import datetime, timezone
    minute = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    return get_json("massive", BASE, path, params, _headers, cache_tag=minute, force_network=True)


def live_price(ticker: str) -> dict:
    """The latest minute bar (stocks, ETFs; crypto as X:BTCUSD). Massive's timeframe depends on the plan."""
    from datetime import date, timedelta
    end, start = date.today(), date.today() - timedelta(days=5)   # covers weekends and holidays
    rows = _get_live(f"/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}",
                     {"adjusted": "true", "sort": "desc", "limit": 1}).get("results") or []
    if not rows:
        return {}
    r = rows[0]
    return {"price": r.get("c"), "bar_open": r.get("o"), "bar_high": r.get("h"), "bar_low": r.get("l"),
            "as_of_utc": pd.Timestamp(r["t"], unit="ms", tz="UTC").isoformat()}


def opening_price_today(ticker: str) -> float | None:
    """Today's regular-session opening price (the 09:30 ET minute bar's open), or None before the open or on a day
    without a session. A live call, cached per minute (DECISIONS #200)."""
    from datetime import date
    d = date.today().isoformat()
    rows = _get_live(f"/v2/aggs/ticker/{ticker}/range/1/minute/{d}/{d}", {"adjusted": "true", "sort": "asc", "limit": 50000}).get("results") or []
    for r in rows:
        ts = pd.Timestamp(r["t"], unit="ms", tz="UTC").tz_convert("America/New_York")
        if (ts.hour, ts.minute) >= (9, 30) and ts.hour < 16:
            return float(r["o"])
    return None


def live_option(underlying: str, expiry: str, strike: float, kind: str, near: bool = False) -> list[dict]:
    """Live snapshot rows for one option contract, or (near=True) for contracts close to it, to suggest listed ones."""
    if near:
        lo, hi = pd.Timestamp(expiry) - pd.Timedelta(days=10), pd.Timestamp(expiry) + pd.Timedelta(days=10)
        params = {"contract_type": kind, "expiration_date.gte": str(lo.date()), "expiration_date.lte": str(hi.date()),
                  "strike_price.gte": round(strike * 0.93, 2), "strike_price.lte": round(strike * 1.07, 2), "limit": 50}
    else:
        params = {"contract_type": kind, "expiration_date": expiry, "strike_price": strike, "limit": 5}
    return _get_live(f"/v3/snapshot/options/{underlying}", params).get("results") or []


def grouped_daily(day: str) -> list[dict]:
    """Every US stock/ETF bar for one day in one call (~12.6k rows). Not disk-cached: etfmatch stores closes compactly."""
    return get_json("massive", BASE, f"/v2/aggs/grouped/locale/us/market/stocks/{day}", {"adjusted": "true"}, _headers,
                    cache=False).get("results") or []


def etf_list() -> list[dict]:
    """Active US-listed ETFs (ticker, name, exchange)."""
    return _paged("/v3/reference/tickers", {"type": "ETF", "market": "stocks", "active": "true", "limit": 1000})


def dividends(ticker: str) -> list[dict]:
    """Cash dividends (ex-date, amount), for total returns."""
    return _paged("/v3/reference/dividends", {"ticker": ticker, "limit": 1000})


def futures_snapshot(product: str, max_pages: int = 12) -> list[dict]:
    """Every contract of a futures product (outrights, spreads, minis) with its settlement date, latest settlement and
    real-time quote. Cached per hour (not forever: it's live)."""
    from datetime import datetime, timezone
    from urllib.parse import parse_qsl, urlparse
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    out, path, p = [], "/futures/v1/snapshot", {"product_code": product, "limit": 250}
    for _ in range(max_pages):
        d = get_json("massive", BASE, path, p, _headers, cache_tag=tag)
        out += d.get("results") or []
        if not d.get("next_url"):
            break
        u = urlparse(d["next_url"])
        path, p = u.path, dict(parse_qsl(u.query))
        p.pop("apiKey", None)
    return out


def news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent news items for a ticker (Benzinga and other publishers via Massive), newest first."""
    return _get("/v2/reference/news", {"ticker": ticker, "limit": limit, "order": "desc", "sort": "published_utc"}).get("results") or []


def income_statements(ticker: str, n: int = 4) -> list[dict]:
    """The last n quarterly income statements."""
    return _get("/stocks/financials/v1/income-statements",
                {"tickers": ticker, "timeframe": "quarterly", "limit": n, "sort": "period_end.desc"}).get("results") or []


def list_date(ticker: str) -> pd.Timestamp | None:
    """First listing date from ticker overview (F10: IBIT bars start before its 2024 launch)."""
    d = overview(ticker).get("list_date")
    return pd.Timestamp(d) if d else None


def sector(ticker: str) -> tuple[str, str | None]:
    """GICS-style (sector, industry) from the overview's SIC code."""
    from ..sectors import sic_to_sector
    return sic_to_sector(overview(ticker).get("sic_code"))


def daily_bars(ticker: str, start: str, end: str, trim_to_listing: bool = True) -> pd.DataFrame:
    """Adjusted daily bars → DataFrame indexed by date with column 'close'."""
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    rows = _get(path, {"adjusted": "true", "sort": "asc", "limit": 50000}).get("results") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["close"])
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    out = df.set_index("date")[["c"]].rename(columns={"c": "close"})
    if trim_to_listing:
        ld = list_date(ticker)
        if ld is not None:
            out = out[out.index >= ld]
    return out


def closes(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wide frame of closes, one column per ticker."""
    return pd.concat({t: daily_bars(t, start, end)["close"] for t in tickers}, axis=1)


# ---------------------------------------------------------------- earnings (Benzinga, CP-L F9)


def earnings(ticker: str, today, closes: pd.Series, n_past: int = 8) -> dict:
    """{next: {date, time, status} | None, past_moves: [{date, move}]}.

    A report after the close (time ≥ 16:00) moves the next session; one before the open moves that
    day's session. The move is close-before → close-after, from adjusted bars.
    """
    rows = _get("/benzinga/v1/earnings", {"ticker": ticker, "limit": 100, "sort": "date.desc"}).get("results") or []
    t0 = pd.Timestamp(today)
    nxt = [r for r in rows if r.get("date") and pd.Timestamp(r["date"]) >= t0]
    nxt = min(nxt, key=lambda r: r["date"]) if nxt else None
    past, idx = [], closes.dropna().index
    for r in sorted((r for r in rows if r.get("date") and pd.Timestamp(r["date"]) < t0), key=lambda r: r["date"], reverse=True):
        d = pd.Timestamp(r["date"])
        after_close = str(r.get("time") or "00:00") >= "16:00"
        i = idx.searchsorted(d)  # first session on/after the report date
        if i >= len(idx):
            continue
        i0, i1 = (i, i + 1) if after_close and idx[i] == d else (i - 1, i)
        if i0 < 0 or i1 >= len(idx):
            continue
        past.append({"date": r["date"], "move": float(closes.loc[idx[i1]] / closes.loc[idx[i0]] - 1),
                     "from": str(idx[i0].date()), "to": str(idx[i1].date())})   # the sessions, for capture tests
        if len(past) >= n_past:
            break
    return {"next": {"date": nxt["date"], "time": nxt.get("time"), "status": nxt.get("date_status")} if nxt else None,
            "past_moves": past}


# ---------------------------------------------------------------- options


def normalize_chain(rows: list[dict]) -> pd.DataFrame:
    """Snapshot rows → expiry, strike, kind, bid, ask, mid, iv (vendor or NaN), greeks, oi, volume, contract.

    `iv_source` is per contract: 'vendor' where Massive supplied IV, NaN otherwise; the
    market_state layer backs out the rest from the mid and marks them 'computed' (F7).
    """
    out = []
    for r in rows:
        d, q, g = r.get("details") or {}, r.get("last_quote") or {}, r.get("greeks") or {}
        bid, ask = q.get("bid"), q.get("ask")
        mid = q.get("midpoint")
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2
        iv = r.get("implied_volatility")
        out.append({
            "contract": d.get("ticker"), "expiry": d.get("expiration_date"), "strike": d.get("strike_price"),
            "kind": d.get("contract_type"), "bid": bid, "ask": ask, "mid": mid,
            "iv": iv if iv else np.nan, "iv_source": "vendor" if iv else None,
            "delta": g.get("delta"), "gamma": g.get("gamma"), "theta": g.get("theta"), "vega": g.get("vega"),
            "oi": r.get("open_interest"), "volume": (r.get("day") or {}).get("volume"),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def option_contracts(underlying: str, expiry: str, expired: bool) -> list[dict]:
    """Listed contracts for one expiry (reference endpoint; `expired=true` for past expiries) →
    [{ticker, strike, kind}]. Used by the IV-history backfill (options/vol_history.py)."""
    rows = _paged("/v3/reference/options/contracts",
                  {"underlying_ticker": underlying, "expiration_date": expiry, "expired": str(expired).lower(), "limit": 1000})
    return [{"ticker": r["ticker"], "strike": float(r["strike_price"]), "kind": r["contract_type"]}
            for r in rows if r.get("ticker") and r.get("strike_price") is not None and r.get("contract_type") in ("call", "put")]


def option_closes(contract: str, start: str, end: str) -> pd.Series:
    """Daily closes of one option contract (same aggs endpoint as stocks, ticker 'O:...')."""
    return daily_bars(contract, start, end, trim_to_listing=False)["close"]


def option_chain_snapshot(underlying: str, expiry_from: str | None = None, expiry_to: str | None = None,
                          strike_lo: float | None = None, strike_hi: float | None = None) -> pd.DataFrame:
    """Every page of the chain, optionally restricted to an expiry window and a strike band (F6)."""
    params = {"limit": 250}
    if strike_lo is not None:
        params["strike_price.gte"] = round(strike_lo, 2)
    if strike_hi is not None:
        params["strike_price.lte"] = round(strike_hi, 2)
    if expiry_from:
        params["expiration_date.gte"] = expiry_from
    if expiry_to:
        params["expiration_date.lte"] = expiry_to
    return normalize_chain(_paged(f"/v3/snapshot/options/{underlying}", params))


# ---------------------------------------------------------------- ETF holdings


def etf_holdings(etf: str) -> pd.DataFrame:
    """Latest holdings snapshot only (F8) → ticker, weight, name, asset_class, security_type.

    Weights are as published (not renormalized): look-through truncates and buckets the rest
    (methodology §3, ruling on C11).
    """
    probe = _get("/etf-global/v1/constituents", {"composite_ticker": etf, "limit": 50, "sort": "processed_date.desc"})
    rows = probe.get("results") or []
    if not rows:
        return pd.DataFrame(columns=["ticker", "weight"])
    latest = max(r["effective_date"] for r in rows if r.get("effective_date"))
    rows = _paged("/etf-global/v1/constituents", {"composite_ticker": etf, "effective_date": latest, "limit": 5000})
    df = pd.DataFrame([{"ticker": r.get("constituent_ticker") or r.get("constituent_name"), "weight": r.get("weight"),
                        "name": r.get("constituent_name"), "asset_class": r.get("asset_class"),
                        "security_type": r.get("security_type"), "effective_date": r.get("effective_date")} for r in rows])
    return df.dropna(subset=["weight"]).sort_values("weight", ascending=False).reset_index(drop=True)
