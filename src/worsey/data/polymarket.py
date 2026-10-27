"""Polymarket public data (no auth).

topical(): the event universe's second venue (DECISIONS #224). Polymarket's open list is mostly sports, so it is read
through topic tags (politics, geopolitics, economy, finance, world, business, tech, Fed rates, trade war), top events by
24-hour volume, normalized to the same market dict as Kalshi (ids prefixed "PM-"). Cached per hour, so the odds refresh
hourly while replays stay offline.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .http import get_json

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
TAGS = ["politics", "geopolitics", "economy", "finance", "world", "business", "tech", "fed-rates", "trade-war"]
MIN_VOLUME = 50_000
PER_EVENT = 3          # ladders ("by June / by Sept / by Dec") keep their three most traded rungs


def markets(active: bool = True, limit: int = 500) -> list[dict]:
    return get_json("polymarket", GAMMA, "/markets", {"active": str(active).lower(), "closed": "false", "limit": limit})


def daily_history(token_id: str) -> pd.Series:
    data = get_json("polymarket", CLOB, "/prices-history", {"market": token_id, "interval": "max", "fidelity": 1440})
    rows = [(pd.to_datetime(p["t"], unit="s").normalize(), float(p["p"])) for p in data.get("history", [])]
    return pd.Series(dict(rows), name=token_id, dtype=float)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize(m: dict, ev: dict) -> dict:
    last = _f(m.get("lastTradePrice"))
    if last is None:
        try:
            last = _f(json.loads(m.get("outcomePrices") or "[]")[0])
        except (ValueError, IndexError, TypeError):
            last = None
    return {"id": f"PM-{m['id']}", "venue": "polymarket", "title": m.get("question"), "subtitle": m.get("groupItemTitle"),
            "rules_primary": (m.get("description") or "")[:600], "event_ticker": f"PM-E{ev.get('id')}", "series_ticker": None,
            "category": "polymarket:" + ",".join(t.get("slug", "") for t in (ev.get("tags") or [])[:3]),
            "bid": _f(m.get("bestBid")), "ask": _f(m.get("bestAsk")), "last": last, "volume": _f(m.get("volumeNum")) or 0.0,
            "open_date": (m.get("startDate") or "")[:10] or None, "close_date": (m.get("endDate") or "")[:10] or None,
            "status": "active"}


def topical(tag: str | None = None, horizon_days: int = 365) -> list[dict]:
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    end = (date.today() + timedelta(days=horizon_days)).isoformat()
    seen, out = set(), []
    for t in TAGS:
        evs = get_json("polymarket", GAMMA, "/events", {"active": "true", "closed": "false", "tag_slug": t, "order": "volume24hr",
                                                        "ascending": "false", "limit": 50}, cache_tag=tag)
        for ev in evs or []:
            ms = [m for m in ev.get("markets") or [] if m.get("active") and not m.get("closed") and (_f(m.get("volumeNum")) or 0) >= MIN_VOLUME
                  and (m.get("endDate") or "9999")[:10] <= end]
            for m in sorted(ms, key=lambda m: -(_f(m.get("volumeNum")) or 0))[:PER_EVENT]:
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                out.append(normalize(m, ev))
    return out
