"""Kalshi public market data (no auth), verified live in CP-L (SPEC §10b).

What the real API looks like (findings F1–F5, F14):
- Prices are dollar strings: markets `yes_bid_dollars`, `last_price_dollars`; live candles
  `price.close_dollars`; *historical* candles `price.close` (no suffix). Both → float in [0, 1].
- Daily candles close at midnight ET, so Δp is built from hourly candles sampled at the last
  print ≤ 16:00 ET (methodology §4.1). Hourly requests are capped (~200 days), so we chunk.
- Markets settled before the historical cutoff live only under /historical/...
- Category is on the event (and series), not the market; events also carry `mutually_exclusive`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .http import get_json

BASE = "https://api.elections.kalshi.com/trade-api/v2"
ET = "America/New_York"
CHUNK_DAYS = 150  # hourly candles: ~200 days works, 400 is rejected


def _f(x) -> float | None:
    """Kalshi dollar string → float. None/'' stay None."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ts(s) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def _paged(path: str, params: dict, key: str, max_pages: int = 50) -> list[dict]:
    out, cursor = [], None
    for _ in range(max_pages):
        p = dict(params, **({"cursor": cursor} if cursor else {}))
        data = get_json("kalshi", BASE, path, p)
        out += data.get(key, [])
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


# ---------------------------------------------------------------- metadata


def cutoff() -> datetime:
    """Markets settled before this are served only by /historical/... endpoints."""
    return _ts(get_json("kalshi", BASE, "/historical/cutoff", None)["market_settled_ts"])


def series(category: str | None = None) -> list[dict]:
    return get_json("kalshi", BASE, "/series", {"category": category} if category else None).get("series", [])


def open_events() -> list[dict]:
    # >10k open events (a third of them sports): this belongs in `worsey build`, cached
    return _paged("/events", {"status": "open", "limit": 200}, "events", max_pages=400)


def normalize_market(m: dict, event: dict | None = None) -> dict:
    """One flat dict per market, in the shape discovery/probability expect."""
    ev = event or {}
    return {
        "id": m["ticker"],
        "venue": "kalshi",
        "title": m.get("title") or ev.get("title"),
        "subtitle": m.get("yes_sub_title") or m.get("subtitle"),
        "rules_primary": m.get("rules_primary"),
        "event_ticker": m.get("event_ticker"),
        "series_ticker": ev.get("series_ticker"),
        "category": ev.get("category"),
        "mutually_exclusive": ev.get("mutually_exclusive"),
        "bid": _f(m.get("yes_bid_dollars")),
        "ask": _f(m.get("yes_ask_dollars")),
        "last": _f(m.get("last_price_dollars")),
        "volume": _f(m.get("volume_fp")) or 0.0,
        "open_interest": _f(m.get("open_interest_fp")),
        "open_date": (m.get("open_time") or "")[:10] or None,
        "close_date": (m.get("close_time") or "")[:10] or None,
        "strike_type": m.get("strike_type"),
        "floor_strike": m.get("floor_strike"),
        "cap_strike": m.get("cap_strike"),
        "status": m.get("status"),
        "result": m.get("result"),
        "settlement_ts": m.get("settlement_ts"),
    }


def open_markets(categories: set[str] | None = None) -> list[dict]:
    """All open (non-combo) markets with their event's category attached (F5)."""
    events = {e["event_ticker"]: e for e in open_events()}
    out = []
    for m in _paged("/markets", {"status": "open", "limit": 1000, "mve_filter": "exclude"}, "markets", max_pages=400):
        ev = events.get(m.get("event_ticker"))
        if ev is None:
            continue
        if categories and ev.get("category") not in categories:
            continue
        out.append(normalize_market(m, ev))
    return out


def settled_markets(series_ticker: str) -> list[dict]:
    """Every settled market in a series, live and historical (F3), oldest first."""
    live = _paged("/markets", {"series_ticker": series_ticker, "status": "settled", "limit": 1000}, "markets")
    hist = _paged("/historical/markets", {"series_ticker": series_ticker, "limit": 1000}, "markets")
    seen, out = set(), []
    for m in hist + live:
        if m["ticker"] not in seen:
            seen.add(m["ticker"])
            out.append(normalize_market(m, {"series_ticker": series_ticker}))
    return sorted(out, key=lambda x: x.get("settlement_ts") or "")


# ---------------------------------------------------------------- candles


def parse_candles(candles: list[dict]) -> pd.DataFrame:
    """Hourly candles → frame indexed by end time (UTC) with trade close, bid, ask, volume.

    Accepts both live (`close_dollars`) and historical (`close`) key styles (F1, F14).
    """
    rows = []
    for c in candles:
        def g(obj, name):
            o = c.get(obj) or {}
            return _f(o.get(f"{name}_dollars", o.get(name)))
        rows.append({"ts": pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC"),
                     "trade": g("price", "close"), "bid": g("yes_bid", "close"), "ask": g("yes_ask", "close"),
                     "volume": _f(c.get("volume_fp", c.get("volume"))) or 0.0})
    if not rows:
        return pd.DataFrame(columns=["trade", "bid", "ask", "volume"])
    return pd.DataFrame(rows).set_index("ts").sort_index()


def hourly(series_ticker: str | None, ticker: str, start: datetime, end: datetime, historical: bool) -> pd.DataFrame:
    """Hourly candles over [start, end], chunked under the API's range cap."""
    path = f"/historical/markets/{ticker}/candlesticks" if historical else f"/series/{series_ticker}/markets/{ticker}/candlesticks"
    frames, t0 = [], int(start.timestamp())
    t_end = int(end.timestamp())
    while t0 < t_end:
        t1 = min(t0 + CHUNK_DAYS * 86400, t_end)
        data = get_json("kalshi", BASE, path, {"start_ts": t0, "end_ts": t1, "period_interval": 60})
        frames.append(parse_candles(data.get("candlesticks", [])))
        t0 = t1
    df = pd.concat(frames) if frames else parse_candles([])
    return df[~df.index.duplicated(keep="last")]


def sample_at_close(h: pd.DataFrame, hour_et: int = 16) -> pd.DataFrame:
    """Daily p at the equity close: the last hourly print ending ≤ 16:00 ET each day.

    p = last trade if one printed in that hour, else the bid/ask mid. A day is *stale* when
    no trade printed since the previous sample AND the quote mid didn't move (methodology §4.1:
    stale prices are masked).
    """
    if h.empty:
        return pd.DataFrame(columns=["p", "stale", "source"])
    x = h.copy()
    x["mid"] = (x["bid"] + x["ask"]) / 2
    local = x.index.tz_convert(ET)
    # a candle ending exactly at 16:00 ET covers 15:00–16:00, so it counts for that day
    eligible = (local.hour < hour_et) | ((local.hour == hour_et) & (local.minute == 0))
    x["day"] = local.normalize().tz_localize(None)
    x = x[eligible]
    rows = []
    prev_mid = None
    traded_since = False
    for day, g in x.groupby("day"):
        traded = g["trade"].notna() & (g["volume"] > 0)
        last = g.iloc[-1]
        last_trade = g.loc[traded, "trade"]
        p = float(last_trade.iloc[-1]) if len(last_trade) else (float(last["mid"]) if pd.notna(last["mid"]) else np.nan)
        src = "trade" if len(last_trade) else "mid"
        mid = float(last["mid"]) if pd.notna(last["mid"]) else None
        stale = (not traded.any()) and (prev_mid is not None and mid == prev_mid)
        rows.append({"day": day, "p": p, "stale": bool(stale), "source": src})
        prev_mid = mid
    if not rows:   # every candle fell outside the sampling window: no daily history, same shape as no candles
        return pd.DataFrame(columns=["p", "stale", "source"])
    return pd.DataFrame(rows).set_index("day")


def daily_history(market: dict, start: datetime, end: datetime | None = None) -> pd.DataFrame:
    """Daily p sampled at 16:00 ET (+ stale flag) for a normalized market, routing by cutoff (F3)."""
    end = end or datetime.now(timezone.utc)
    st = _ts(market.get("settlement_ts"))
    historical = st is not None and st < cutoff()
    if st is not None:
        end = min(end, st)
    return sample_at_close(hourly(market.get("series_ticker"), market["id"], start, end, historical))


def p_series(market: dict, start: datetime, end: datetime | None = None) -> pd.Series:
    """Convenience: daily p with stale days set to NaN (so Δp on them is masked)."""
    d = daily_history(market, start, end)
    return d["p"].where(~d["stale"])


def quotes(tickers: list[str], tag: str | None = None) -> dict[str, dict]:
    """Live bid/ask/last for many markets (100 per call), cached per `tag` (default: the current UTC hour).
    The only Kalshi call scan_risks makes at query time (SPEC §11b rule 1)."""
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        data = get_json("kalshi", BASE, "/markets", {"tickers": ",".join(chunk), "limit": 1000}, cache_tag=tag)
        for m in data.get("markets", []):
            out[m["ticker"]] = {"bid": _f(m.get("yes_bid_dollars")), "ask": _f(m.get("yes_ask_dollars")),
                                "last": _f(m.get("last_price_dollars")), "volume": _f(m.get("volume_fp")) or 0.0,
                                "status": m.get("status")}
    return out
