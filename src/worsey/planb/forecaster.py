"""Tier 3, in the chat: Worsey's forecaster for events no market lists (docs/event_pipeline/probabilities.md; DECISIONS #233).

The pilot's arm C (scripts/part2_pilot.py, #230): three agents, each given the event, the library base rate for its
kind, the 8 most similar past library events (local bge-small embeddings), the Wikipedia news from the last 14 days, and
in the chat up to 3 related (not identical) prediction markets with corrected odds (PART2 tier 3: related markets are
evidence; untested in the pilot, where markets were hidden), in different orders; the median in log-odds, clipped to [2%, 98%] and to any tier-2b bounds. No calibration step yet
(the backtest would fit one). Runs only for the top few unpriced events per scan (cost: 3 calls each).
"""

from __future__ import annotations

import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EMB = ROOT / "cache/pmrm/part2/lib_emb.pkl"
STOP = {"the", "a", "an", "of", "by", "in", "on", "to", "will", "be", "or", "and", "for", "at", "than", "more", "less",
        "before", "after", "is", "2026", "2027", "this", "that", "with", "any", "from", "as"}
ASK = """Today is {T}. Forecast the probability that this happens within the next {h} days (by {end}).

Event: {q}
{evidence}
Think briefly, then answer with ONLY a JSON object: {{"p": <probability between 0 and 1>, "why": "<at most 3 short lines>"}}"""
ORDERS = [("base", "mkts", "sim", "news"), ("news", "sim", "mkts", "base"), ("sim", "news", "base", "mkts")]


def _tok(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


@lru_cache(maxsize=1)
def _library():
    from . import live
    ev = live.pack()["events"]
    if EMB.exists():
        E = pickle.load(open(EMB, "rb"))
        if len(E["ids"]) == len(ev):
            return ev, E["V"]
    V = _model().encode([f"{e.get('title') or ''}. {e.get('description') or ''}"[:600] for e in ev], batch_size=64,
                        normalize_embeddings=True).astype(np.float32)
    EMB.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"ids": [e["id"] for e in ev], "V": V}, open(EMB, "wb"))
    return ev, V


def base_rate(category: str | None, horizon: int) -> float | None:
    ev, _ = _library()
    if not category:
        return None
    n = sum(1 for e in ev if e["category"] == category and e.get("severity") in ("major", "extreme") and "2016" <= e["date"][:4] <= "2025")
    return float(1 - np.exp(-n / 10 * horizon / 365)) if n else None


def similar(text: str, T: str, k: int = 8) -> list[dict]:
    ev, V = _library()
    s = V @ _model().encode([text], normalize_embeddings=True)[0]
    idx = [i for i in np.argsort(-s) if ev[i]["date"] < T][:k]
    return [{"date": ev[i]["date"], "category": ev[i]["category"], "severity": ev[i].get("severity"), "event": (ev[i].get("title") or "")[:140]} for i in idx]


def news(text: str, T: str, n: int = 15) -> list[str]:
    from . import wikinews as W
    t = date.fromisoformat(T)
    lo = (t - timedelta(days=14)).isoformat()
    items = []
    for y, m in sorted({(d.year, d.month) for d in (t - timedelta(days=14), t)}):
        try:
            items += [it for it in W.month_items(y, m) if lo <= it["day"] <= T]
        except Exception:  # noqa: BLE001 — no news is evidence too
            pass
    q = _tok(text)
    items.sort(key=lambda it: -len(q & _tok(it["text"])))
    return [f"{it['day']}: {it['text'][:220]}" for it in items[:n] if q & _tok(it["text"])]


def _evidence(category, br, horizon, sim, nw, order, related=None) -> str:
    words = {"correlated": "related", "E_implies_M": "broader: if this event happens, that market resolves YES",
             "M_implies_E": "narrower: if that market resolves YES, this event happens", "exclusive": "can't both happen"}
    parts = {
        "mkts": ("Related prediction markets (NOT the same event; context only), with odds corrected for longshot bias:\n"
                 + "\n".join(f"- {m['title']}: {m['p']:.0%} ({words.get(m['relation'], 'related')})" for m in related)) if related else "",
        "base": (f"Base rate: in 2016-2025, at least one major or extreme {category.replace('_', ' ')} event happened in about "
                 f"{br:.0%} of {horizon}-day windows (how often events of that KIND happen, not this event's odds)." if br is not None else ""),
        "sim": "Similar past events (Worsey's library):\n" + "\n".join(f"- {s['date']} [{s['category']}, {s['severity']}] {s['event']}" for s in sim) if sim else "",
        "news": "News from the last 14 days (Wikipedia current events):\n" + "\n".join(f"- {x}" for x in nw) if nw else "News from the last 14 days: nothing matching found."}
    return "\n\n".join(parts[k] for k in order if parts[k])


def _p(txt: str) -> float | None:
    hits = re.findall(r'"p"\s*:\s*"?([0-9]*\.?[0-9]+)', txt)
    p = float(hits[-1]) if hits else None
    return None if p is None or not (0 <= p <= 1) else p


def forecast(llm, event: str, category: str | None, horizon_days: int, today: str, bounds: list | None = None,
             model: str | None = None, related: list | None = None) -> dict:
    from .. import config
    model = model or config.EXPERT_MODEL
    br = base_rate(category, horizon_days)
    sim, nw = similar(event, today), news(event, today)
    end = (date.fromisoformat(today) + timedelta(days=horizon_days)).isoformat()

    def one(order):
        pr = ASK.format(T=today, h=horizon_days, end=end, q=event,
                        evidence="\n" + _evidence(category or "", br, horizon_days, sim, nw, order, related) + "\n")
        r = llm.create(model=model, max_tokens=600, thinking={"type": "between_tools"},
                       messages=[{"role": "user", "content": pr}], _source="part2_forecast")
        return _p("".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text"))
    with ThreadPoolExecutor(3) as ex:
        ps = [p for p in ex.map(one, ORDERS) if p is not None]
    if not ps:
        return {"p": None, "agents": []}
    lg = np.median([np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99))) for p in ps])
    p = float(np.clip(1 / (1 + np.exp(-lg)), 0.02, 0.98))
    clipped = False
    if bounds and (bounds[0] > 0 or bounds[1] < 1):
        q = float(np.clip(p, bounds[0], bounds[1]))
        clipped, p = q != p, q
    return {"p": p, "agents": ps, "spread": [min(ps), max(ps)], "clipped_to_bounds": clipped, "base_rate": br,
            "similar_events": len(sim), "news_items": len(nw)}
