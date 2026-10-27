"""What the tape cares about (SPEC §1b; DECISIONS #115, #121): the recent_markets_analyst's code-side sources.

Source priority (#121): (1) a tape summary computed by code (biggest moves vs their own vol, IV/skew changes,
prediction-market odds shifts) > (2) Massive/Benzinga news and web search > (3) curated RSS (config.RSS_FEEDS, the trainer's
list) > (4) Bluesky (needs a key) > (5) X (needs the trainer's developer account). No Reddit or StockTwits. Prices are the
primary sentiment source; posts are color, cited, weighted by the curated list, never by virality.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .data.cache import CACHE_ROOT

DIGEST = CACHE_ROOT / "digest" / "latest.json"
WATCH = ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "HYG", "GLD", "USO", "UUP", "VIXY", "XLK", "XLF", "XLE", "XLV", "XLI",
         "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "EEM", "FXI"]


def tape_summary(data, holdings: list[str] | None = None, top: int = 8) -> dict:
    """Code-computed: the week's biggest moves (z = 5-day move / its own 20-day vol), IV and skew changes, odds shifts."""
    tick = list(dict.fromkeys(WATCH + list(holdings or [])))
    out = {"as_of": str(data.today())}
    try:
        c = data.closes(tick, days=90).dropna(how="all", axis=1)
        r = c.pct_change()
        vol = r.tail(21).std()
        m5 = c.iloc[-1] / c.iloc[-6] - 1
        m1 = c.iloc[-1] / c.iloc[-2] - 1
        z = (m5 / (vol * np.sqrt(5))).replace([np.inf, -np.inf], np.nan).dropna()
        rows = [{"ticker": t, "move_5d": float(m5[t]), "move_1d": float(m1[t]), "z_5d": float(z[t])}
                for t in z.abs().sort_values(ascending=False).index[:top]]
        out["biggest_moves"] = rows
    except Exception as e:  # noqa: BLE001
        out["biggest_moves_error"] = type(e).__name__
    vol_rows = []
    try:
        from .options import vol_history
        for t in ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "SMH", "TLT", "GLD"]:
            h = vol_history.load(t)
            if h is None or len(h) < 10:
                continue
            h = h.dropna(subset=["atm_front"])
            if len(h) < 6:
                continue
            a, b = h.iloc[-1], h.iloc[-6]
            rr = lambda x: (x["put25_front"] - x["call25_front"]) if pd.notna(x["put25_front"]) and pd.notna(x["call25_front"]) else None
            vol_rows.append({"ticker": t, "atm_iv": float(a["atm_front"]), "atm_iv_change_5d": float(a["atm_front"] - b["atm_front"]),
                             "skew_rr": rr(a), "skew_rr_change_5d": (rr(a) - rr(b)) if rr(a) is not None and rr(b) is not None else None})
        out["vol"] = sorted(vol_rows, key=lambda x: -abs(x["atm_iv_change_5d"]))
    except Exception as e:  # noqa: BLE001
        out["vol_error"] = type(e).__name__
    try:
        shifts = []
        names = ((data.structure() or {}).get("names") or {}) if hasattr(data, "structure") else {}
        for m in data.event_markets():
            hist = data.market_history(m["id"])
            series = hist.iloc[:, 0] if isinstance(hist, pd.DataFrame) else hist
            series = series.dropna()
            b, a = m.get("bid"), m.get("ask")
            p_now = (b + a) / 2 if b is not None and a is not None else m.get("last")
            if p_now is None or len(series) < 6:
                continue
            p_then = float(series.iloc[-6])
            shifts.append({"market": m["id"], "name": names.get(m["id"]) or m.get("title"), "p_then": p_then, "p_now": float(p_now),
                           "shift": float(p_now) - p_then})
        out["odds_shifts"] = sorted(shifts, key=lambda x: -abs(x["shift"]))[:top]
    except Exception as e:  # noqa: BLE001
        out["odds_error"] = type(e).__name__
    return out


def headlines(tickers=("SPY", "QQQ"), per: int = 6) -> list[dict]:
    from .data import massive
    out = []
    for t in tickers:
        try:
            for n in massive.news(t, limit=per):
                out.append({"title": n.get("title"), "publisher": (n.get("publisher") or {}).get("name"),
                            "published": n.get("published_utc"), "url": n.get("article_url")})
        except Exception:  # noqa: BLE001
            continue
    seen, uniq = set(), []
    for h in out:
        if h["url"] not in seen:
            seen.add(h["url"])
            uniq.append(h)
    return uniq


def rss(feeds: list[str], per: int = 5) -> list[dict]:
    """Curated newsletter/blog RSS."""
    import requests
    out = []
    for url in feeds:
        try:
            root = ET.fromstring(requests.get(url, timeout=10).content)
            for it in root.iter("item"):
                out.append({"feed": url, "title": (it.findtext("title") or "").strip(), "url": (it.findtext("link") or "").strip(),
                            "published": it.findtext("pubDate")})
                if sum(1 for o in out if o["feed"] == url) >= per:
                    break
        except Exception:  # noqa: BLE001
            continue
    return out


DIGEST_PROMPT = """Write this week's "what the tape cares about" digest for the other Worsey experts (not for users).
At most 8 short lines. Lead with what PRICES say (the tape summary below: biggest moves vs their own vol, vol and skew
changes, prediction-market odds shifts), then the stories that explain them, citing a headline or a web source for each
story. Posts and headlines are color, not facts. Market commentary only: no advice. Use only numbers from the tape
summary, written as given (fractions as percentages).

Tape summary: {tape}
Headlines: {heads}
RSS: {rss}"""


def write_digest(llm, data, model: str) -> dict:
    """One call (with web search) at build time; the grounding check strips any number not in the tape summary."""
    from . import config
    from .agent import grounding
    tape = tape_summary(data)
    heads, feeds = headlines(), rss(getattr(config, "RSS_FEEDS", []))
    resp = llm.create(model=model, max_tokens=1500, output_config={"effort": "medium"},
                      tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                      messages=[{"role": "user", "content": DIGEST_PROMPT.format(tape=json.dumps(tape, default=str),
                                                                                  heads=json.dumps(heads[:12], default=str),
                                                                                  rss=json.dumps(feeds[:10], default=str))}])
    text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text").strip()
    cites = [{"url": c.get("url"), "title": c.get("title"), "cited_text": c.get("cited_text")}
             for b in resp["content"] if b.get("type") == "text" for c in (b.get("citations") or [])]
    g = grounding.check(text, [tape, cites])
    if not g["ok"]:
        text = grounding.strip(text, g["ungrounded"])
    out = {"text": text, "created": time.time(), "as_of": tape["as_of"], "tape": tape, "sources": heads[:12] + cites,
           "stripped": g.get("ungrounded", [])}
    DIGEST.parent.mkdir(parents=True, exist_ok=True)
    DIGEST.write_text(json.dumps(out, default=str, indent=1), encoding="utf-8")
    return out


def digest() -> dict | None:
    if not DIGEST.exists():
        return None
    d = json.loads(DIGEST.read_text(encoding="utf-8"))
    d["age_days"] = round((time.time() - d.get("created", 0)) / 86400, 1)
    return d
