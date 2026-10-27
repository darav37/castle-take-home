"""A probability on every event, in tiers (docs/event_pipeline/probabilities.md; DECISIONS #227-229).

Tier 1   E is a listed market: its price, with the measured longshot correction where the bias is significant
         (scripts/part2_longshot.py -> planb/longshot.json). Raw and corrected both returned.
Tier 2a  same variable, different strike or deadline: derived from the market's siblings in code.
           strike ladders ("above k" markets, or exact buckets summed into a survival curve): made monotone
           (pool-adjacent-violators), interpolated linearly, never extrapolated (scenario/probability.ladder_interp)
           deadline rungs ("by Nov 1 / Dec 1 / Jan 1"): P(by date) made non-decreasing and interpolated between rungs;
           past the last rung only by a constant hazard within 3x the time (scenario/probability.rescale_time)
Tier 2b  a different but related event: the LLM classifies E's logical relation to each candidate market (one call per
         event); code turns relations into bounds and intersects them:
           E implies M  -> P(E) <= P(M);  M implies E -> P(E) >= P(M);  mutually exclusive -> P(E) <= 1 - P(M)
         narrow (hi - lo <= 0.10): the midpoint, labelled "derived from markets ..."; otherwise tier 3 inside [lo, hi]
Tier 3   nothing listed is close: the forecaster (scripts/part2_pilot.py for the test; the chat hook comes later).
The LLM never produces a probability here: it labels relations and reads thresholds and deadlines from the event's words.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..scenario import probability as P1

LONGSHOT = Path(__file__).resolve().parent / "longshot.json"
RELATIONS = ("same", "other_strike", "other_deadline", "E_implies_M", "M_implies_E", "exclusive", "correlated", "unrelated")
NARROW = 0.10


# ---------------------------------------------------------------- tier 1

@lru_cache(maxsize=1)
def _longshot() -> dict | None:
    return json.loads(LONGSHOT.read_text(encoding="utf-8")) if LONGSHOT.exists() else None


def corrected(p: float, venue: str = "kalshi") -> dict:
    """The market price and the longshot-corrected probability (only inside the price bins where the bias was
    significant in the full sample and again out of sample; raw elsewhere)."""
    L = _longshot()
    out = {"raw": float(p), "corrected": float(p), "applied": False}
    if not L or p is None:
        return out
    fit = L.get("by_venue", {}).get(venue) or L["all"]
    lo_hi = [tuple(b) for b in L["apply_in"]]
    if any(lo <= p < hi for lo, hi in lo_hi):
        x = np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))
        out.update({"corrected": float(1 / (1 + np.exp(-(fit["a"] + fit["b"] * x)))), "applied": True})
    return out


# ---------------------------------------------------------------- tier 2a

def _p(m: dict) -> float | None:
    from .live import _p as live_p
    return live_p(m)


def siblings(m: dict, universe: list[dict]) -> list[dict]:
    return [x for x in universe if x.get("event_ticker") and x.get("event_ticker") == m.get("event_ticker")]


def strike_ladder(m: dict, universe: list[dict]) -> dict | None:
    """P(X > k) at each listed k, from 'greater' markets or from exact buckets summed into a survival curve."""
    sib = siblings(m, universe)
    g = [(x["floor_strike"], _p(x)) for x in sib if x.get("strike_type") == "greater" and x.get("floor_strike") is not None and _p(x) is not None]
    if len(g) >= 2:
        k, p = zip(*sorted(g))
        return {"strikes": list(k), "p_above": list(p), "from": "above-k markets"}
    b = [(x["floor_strike"], _p(x)) for x in sib if x.get("strike_type") == "between" and x.get("floor_strike") is not None and _p(x) is not None]
    if len(b) >= 3:
        b.sort()
        tot = sum(p for _, p in b) or 1.0
        surv = [sum(p for f, p in b if f >= k) / tot for k, _ in b]       # P(X >= k), buckets normalized to sum to 1
        return {"strikes": [k - 0.5 for k, _ in b], "p_above": surv, "from": "exact buckets, normalized and summed"}
    return None


def derive_strike(m: dict, universe: list[dict], threshold: float, above: bool = True) -> dict:
    lad = strike_ladder(m, universe)
    if not lad:
        return {"p": None, "reason": "no strike ladder among this market's siblings"}
    r = P1.ladder_interp(lad["strikes"], lad["p_above"], threshold, ">=")
    if r["p"] is None:
        return r
    p = r["p"] if above else 1 - r["p"]
    return {"p": p, "reason": f"{r['reason']} ({lad['from']})" + ("" if above else "; below = 1 - above")}


def deadline_rungs(m: dict, universe: list[dict]) -> list[tuple[str, float]]:
    sib = [x for x in siblings(m, universe) if not x.get("floor_strike") and x.get("close_date") and _p(x) is not None]
    by = {}
    for x in sib:
        by.setdefault(x["close_date"], _p(x))
    return sorted(by.items())


def derive_deadline(m: dict, universe: list[dict], deadline: str, today: date | None = None) -> dict:
    """P(E by `deadline`) from the market's deadline rungs; a single rung: constant hazard within 3x the time."""
    today = today or date.today()
    rungs = deadline_rungs(m, universe) or ([(m["close_date"], _p(m))] if _p(m) is not None and m.get("close_date") else [])
    if not rungs:
        return {"p": None, "reason": "no deadline rungs"}
    days = np.array([(date.fromisoformat(d) - today).days for d, _ in rungs], float)
    ps = np.maximum.accumulate(np.array([p for _, p in rungs], float))     # P(by date) can't fall with a later date
    t = (date.fromisoformat(deadline) - today).days
    if t <= 0:
        return {"p": None, "reason": "deadline already passed"}
    if days[0] <= t <= days[-1] and len(rungs) >= 2:
        return {"p": float(np.interp(t, days, ps)), "reason": f"interpolated between deadline rungs {[d for d, _ in rungs]}"}
    near = 0 if t < days[0] else len(days) - 1
    T = days[near]
    if T <= 0 or max(T, t) / min(T, t) > 3:
        return {"p": None, "reason": f"deadline {deadline} too far from the listed rungs to rescale (more than 3x the time)"}
    return {"p": float(P1.rescale_time(ps[near], T, t)), "reason": f"constant hazard from the {rungs[near][0]} rung ({int(T)} -> {t} days)"}


# ---------------------------------------------------------------- tier 2b

CLASSIFY = """An event E and some listed prediction markets. For each market, classify how E relates to the market's YES
outcome. Logic only: do NOT estimate any probability.

  "same"            E is exactly this market's YES (same variable, threshold, deadline, resolution)
  "other_strike"    the same variable but a different threshold (give E's threshold, in the market's units, and
                    whether E is "above" or "below" it)
  "other_deadline"  the same event but a different deadline (give E's deadline, YYYY-MM-DD)
  "E_implies_M"     if E happens, this market's YES must happen (E is a special case of it)
  "M_implies_E"     if this market's YES happens, E must happen (it is a special case of E)
  "exclusive"       E and this market's YES can't both happen
  "correlated"      related (makes E more or less likely) but no logical implication
  "unrelated"

Today is {today}. Return ONLY a JSON array: [{{"id": "...", "relation": "...", "threshold": <number or null>,
"above": <true|false|null>, "deadline": "<YYYY-MM-DD or null>", "why": "<at most 12 words>"}}]

E: {event}

Markets:
{markets}"""


def classify(llm, event: str, candidates: list[dict], today: date | None = None, model: str | None = None) -> list[dict]:
    from .. import config
    today = today or date.today()
    items = [{"id": m["id"], "title": m.get("title"), "subtitle": m.get("subtitle"), "closes": m.get("close_date"),
              "rules": (m.get("rules_primary") or "")[:300]} for m in candidates]
    prompt = CLASSIFY.format(today=today.isoformat(), event=event, markets=json.dumps(items, ensure_ascii=False, indent=0))
    resp = llm.create(model=model or config.EXPERT_MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}],
                      _source="part2_classify")
    txt = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    try:
        arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    ids = {m["id"] for m in candidates}
    return [r for r in arr if isinstance(r, dict) and r.get("id") in ids and r.get("relation") in RELATIONS]


def bounds(relations: list[dict], prices: dict) -> dict:
    """Intersect the bounds implied by logical relations. prices: market id -> probability (corrected)."""
    lo, hi, used = 0.0, 1.0, []
    for r in relations:
        p = prices.get(r["id"])
        if p is None:
            continue
        if r["relation"] == "E_implies_M":
            hi = min(hi, p)
        elif r["relation"] == "M_implies_E":
            lo = max(lo, p)
        elif r["relation"] == "exclusive":
            hi = min(hi, 1 - p)
        else:
            continue
        used.append(r["id"])
    if lo > hi:            # incoherent quotes across markets: report it, keep the overlap's midpoint
        lo, hi = hi, lo
    return {"lo": lo, "hi": hi, "markets": used, "narrow": bool(used) and hi - lo <= NARROW}


# ---------------------------------------------------------------- the pipeline for one event

def probability(llm, event: str, candidates: list[dict], universe: list[dict], today: date | None = None) -> dict:
    """Tier 1 -> 2a -> 2b for one event; returns the tier it stopped at. Tier 3 is signalled with the bounds."""
    rel = classify(llm, event, candidates, today) if candidates else []
    by = {m["id"]: m for m in candidates}
    for r in rel:            # tier 1
        if r["relation"] == "same":
            m = by[r["id"]]
            c = corrected(_p(m), m.get("venue", "kalshi"))
            return {"tier": 1, "p": c["corrected"], "raw": c["raw"], "longshot_corrected": c["applied"], "market": m["id"],
                    "label": f"market: {m.get('title')}", "relations": rel}
    for r in rel:            # tier 2a
        m = by[r["id"]]
        if r["relation"] == "other_strike" and r.get("threshold") is not None:
            d = derive_strike(m, universe, float(r["threshold"]), r.get("above") is not False)
        elif r["relation"] == "other_deadline" and r.get("deadline"):
            d = derive_deadline(m, universe, r["deadline"], today)
        else:
            continue
        if d["p"] is not None:
            c = corrected(d["p"], m.get("venue", "kalshi"))
            return {"tier": "2a", "p": c["corrected"], "raw": c["raw"], "longshot_corrected": c["applied"], "market": m["id"],
                    "label": f"derived from {m.get('title')}: {d['reason']}", "relations": rel}
    prices = {m["id"]: corrected(_p(m), m.get("venue", "kalshi"))["corrected"] for m in candidates if _p(m) is not None}
    b = bounds(rel, prices)
    if b["narrow"]:          # tier 2b
        return {"tier": "2b", "p": (b["lo"] + b["hi"]) / 2, "bounds": [b["lo"], b["hi"]], "markets": b["markets"],
                "label": "derived from markets " + ", ".join(b["markets"]), "relations": rel}
    return {"tier": 3, "p": None, "bounds": [b["lo"], b["hi"]], "markets": b["markets"],
            "evidence_markets": [r["id"] for r in rel if r["relation"] == "correlated"],
            "label": "no market lists this: forecaster", "relations": rel}


# ---------------------------------------------------------------- the chat: one batched relation call, tiers per event

CLASSIFY_MANY = """Several events, each with some listed prediction markets. For each pair, classify how the event relates to
the market's YES outcome. Logic only: do NOT estimate any probability.

  "same" | "other_strike" (give the event's threshold in the market's units and "above": true/false) |
  "other_deadline" (give the event's deadline YYYY-MM-DD) | "E_implies_M" | "M_implies_E" | "exclusive" |
  "correlated" | "unrelated"
(E = the event, M = the market's YES; definitions as usual: "same" only if variable, threshold, deadline and resolution
all match.)

Today is {today}. Return ONLY a JSON array: [{{"event": <event index>, "id": "<market id>", "relation": "...",
"threshold": <number or null>, "above": <true|false|null>, "deadline": "<YYYY-MM-DD or null>"}}]

{blocks}"""


_TITLE_EMB: dict = {}


def candidates_for(event: str, universe: list[dict], k: int = 6) -> list[dict]:
    """The markets most likely to be (or relate to) this event: the top k-2 by meaning (the forecaster's local embedder on
    market titles, cached per market) plus the top 2 by words in the title. Word overlap over titles + rules alone missed
    obvious matches ("AI bubble bursts" vs "AI bubble burst in 2026?", #234)."""
    from ..scenario.matching import keyword_candidates
    priced = [m for m in universe if _p(m) is not None]
    out = []
    try:
        from .forecaster import _model
        todo = [m for m in priced if m["id"] not in _TITLE_EMB]
        if todo:
            V = _model().encode([f"{m.get('title') or ''} {m.get('subtitle') or ''}" for m in todo], batch_size=64, normalize_embeddings=True)
            _TITLE_EMB.update({m["id"]: v for m, v in zip(todo, V)})
        q = _model().encode([event], normalize_embeddings=True)[0]
        out = sorted(priced, key=lambda m: -float(_TITLE_EMB[m["id"]] @ q))[:max(k - 2, 1)]
    except Exception:  # noqa: BLE001 — no embedder: words only
        out = []
    for m in keyword_candidates(event, priced, n=k, text_keys=("title", "subtitle")):
        m = next((x for x in priced if x["id"] == m["id"]), None)
        if m and m not in out and len(out) < k:
            out.append(m)
    return out


def classify_many(llm, items: list[tuple[str, list[dict]]], today: date | None = None, model: str | None = None) -> dict:
    """One call for many events: {event index: [relations]}."""
    from .. import config
    today = today or date.today()
    blocks = []
    for i, (ev, cands) in enumerate(items):
        ms = "\n".join(f"  - {m['id']} | {m.get('title')} {m.get('subtitle') or ''} | closes {m.get('close_date')}" for m in cands)
        blocks.append(f"Event {i}: {ev}\n{ms or '  (no markets)'}")
    prompt = CLASSIFY_MANY.format(today=today.isoformat(), blocks="\n\n".join(blocks))
    resp = llm.create(model=model or config.EXPERT_MODEL, max_tokens=3000, messages=[{"role": "user", "content": prompt}],
                      _source="part2_classify")
    txt = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    try:
        arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return {}
    out = {}
    for r in arr if isinstance(arr, list) else []:
        if isinstance(r, dict) and isinstance(r.get("event"), int) and r.get("relation") in RELATIONS:
            ids = {m["id"] for m in items[r["event"]][1]} if 0 <= r["event"] < len(items) else set()
            if r.get("id") in ids:
                out.setdefault(r["event"], []).append(r)
    return out


def tier_from(rel: list[dict], candidates: list[dict], universe: list[dict], today: date | None = None) -> dict:
    """Tiers 1 -> 2a -> 2b from already-classified relations (no LLM call); tier 3 is signalled with the bounds."""
    by = {m["id"]: m for m in candidates}
    for r in rel:
        if r["relation"] == "same" and r["id"] in by:
            m = by[r["id"]]
            c = corrected(_p(m), m.get("venue", "kalshi"))
            return {"tier": 1, "p": c["corrected"], "raw": c["raw"], "market": m["id"], "label": f"market: {m.get('title')}"}
    for r in rel:
        m = by.get(r["id"])
        if not m:
            continue
        if r["relation"] == "other_strike" and r.get("threshold") is not None:
            d = derive_strike(m, universe, float(r["threshold"]), r.get("above") is not False)
        elif r["relation"] == "other_deadline" and r.get("deadline"):
            d = derive_deadline(m, universe, r["deadline"], today)
        else:
            continue
        if d["p"] is not None:
            c = corrected(d["p"], m.get("venue", "kalshi"))
            return {"tier": "2a", "p": c["corrected"], "raw": c["raw"], "market": m["id"], "label": f"derived from {m.get('title')}: {d['reason']}"}
    prices = {m["id"]: corrected(_p(m), m.get("venue", "kalshi"))["corrected"] for m in candidates if _p(m) is not None}
    b = bounds(rel, prices)
    if b["narrow"]:
        return {"tier": "2b", "p": (b["lo"] + b["hi"]) / 2, "bounds": [b["lo"], b["hi"]], "label": "bounded by markets " + ", ".join(b["markets"])}
    return {"tier": 3, "p": None, "bounds": [b["lo"], b["hi"]], "markets": b["markets"]}
