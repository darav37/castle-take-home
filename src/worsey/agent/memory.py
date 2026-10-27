"""Accumulated knowledge (SPEC §1b; DECISIONS #114): Worsey remembers what it recommended and what happened, so
replies can read as if it has traded with these people for years, without ever changing its principles.

Four layers, highest precedence first:
1. Principles (library, playbook, prompts): fixed; only the trainer changes them.
2. Episodic log (this module): append-only, written by CODE after each turn. Outcomes are computed by code from prices
   when the entry is recalled (an expired option's payoff, an event's odds then vs now, a replica's realized tracking).
3. Lessons: proposed by `worsey reflect`; code keeps only those citing ≥ MIN_CITES logged entries with computed
   outcomes; advisory, ranked below principles, expiring (general 30d, market-regime 7d). Only the trainer promotes a lesson
   into the library.
4. Member profiles and warm sessions (profiles.py, loop.py).
Numbers reach replies only through the `recall` / `book_history` tools, so grounding covers them.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..data.cache import CACHE_ROOT

LOG = CACHE_ROOT / "memory" / "episodic.jsonl"
BOOKS = CACHE_ROOT / "memory" / "books.jsonl"
LESSONS = CACHE_ROOT / "memory" / "lessons.json"
SEED = Path(__file__).resolve().parents[3] / "data" / "demo" / "episodic_seed.jsonl"
MIN_CITES, TTL_GENERAL_D, TTL_REGIME_D = 3, 30, 7
SAID_DID = re.compile(r"\bi (just )?(bought|sold|put on|opened|closed|took|did|went with|passed on|skipped)\b[^.?!]{0,80}", re.I)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def entries(member: str | None = None, include_seed: bool = True) -> list[dict]:
    rows = _rows(LOG) + (_rows(SEED) if include_seed else [])
    return [r for r in rows if member is None or r.get("member") == member]


# ---------------------------------------------------------------- writing (code, after each turn)

def record_turn(s, user_text: str) -> list[dict]:
    """What this turn recommended, as code saw it. Called by loop.run_turn; never by a model."""
    today = s.data.today().isoformat()
    base = {"member": s.member, "chat": s.__dict__.get("chat_id"), "date": today, "ts": time.time(), "demo": False}
    new = []
    names = {x["tool"] for x in s.log[-12:] if x.get("ok")}
    if "view_and_strategies" in names and s.last_strategies and s.view is not None:
        v = s.view
        cands = []
        for c in s.last_strategies.get("candidates", [])[:3]:
            legs = [{"kind": l["kind"], "qty": l["qty"], "strike": l["strike"],
                     "expiry": (s.data.today() + timedelta(days=int(l["days"]))).isoformat()} for l in c.metrics.get("legs", [])]
            cands.append({"id": c.id, "legs": legs, "cost": c.metrics.get("cost")})
        spot = s.last_strategies.get("_spot")
        new.append(dict(base, kind="strategy", underlying=v.underlying, view={"direction": v.direction, "move_pct": v.target_move_pct,
                                                                               "horizon": v.horizon.isoformat()},
                        spot=spot, candidates=cands))
    if "compare_hedges" in names and s.last_hedges:
        h = s.last_hedges
        new.append(dict(base, kind="hedge", event_id=h.get("event_id"), p=h.get("p"), loss_if_yes=h.get("loss_if_yes"),
                        options=[{"name": r.get("name"), "cost": r.get("cost")} for r in (h.get("rows") or [])[:4]]))
    if "analyze_event" in names and s.active_event and s.last_scan:
        row = next((e for e in s.last_scan["events"] + s.last_scan.get("drift", []) if e["event_id"] == s.active_event), None)
        if row:
            new.append(dict(base, kind="event", event_id=row["event_id"], title=row.get("short") or row.get("event"), p=row.get("p"),
                            pnl_yes=row.get("pnl_yes")))
    m = s.__dict__.get("last_etf_match")
    if "replicate_with_etfs" in names and m:
        new.append(dict(base, kind="etf_match", target=m["target"], weights=m["replica"]["weights"], cash=m["replica"]["cash"]))
    for x in SAID_DID.finditer(user_text or ""):   # only what the member said they did, in their words
        new.append(dict(base, kind="member_said", text=x.group(0)[:120]))
    for e in new:
        e["id"] = uuid.uuid4().hex[:8]
        _append(LOG, e)
    return new


def snapshot_book(s) -> None:
    """Each portfolio load is a dated snapshot, so book history is code state, not memory."""
    if s.priced_real is None:
        return
    _append(BOOKS, {"member": s.member, "date": s.data.today().isoformat(), "ts": time.time(),
                    "positions": {t: float(v) for t, v in s.priced_real["value"].items()}})


# ---------------------------------------------------------------- outcomes (code, at recall time)

def _close_on(data, ticker: str, day: str) -> float | None:
    try:
        c = data.closes([ticker], days=900)[ticker].dropna()
    except Exception:  # noqa: BLE001
        return None
    c = c[c.index <= pd.Timestamp(day)]
    return float(c.iloc[-1]) if len(c) else None


def outcome(e: dict, data) -> dict:
    today = data.today()
    if e["kind"] == "strategy":
        now = _close_on(data, e["underlying"], today.isoformat())
        out = {"underlying_then": e.get("spot"), "underlying_now": now, "now_is": "today's price, not the expiry close",
               "underlying_move": (now / e["spot"] - 1) if now and e.get("spot") else None, "candidates": []}
        for c in e.get("candidates", []):
            exps = [l["expiry"] for l in c["legs"]]
            if exps and max(exps) <= today.isoformat():
                pay = 0.0
                for l in c["legs"]:
                    px = _close_on(data, e["underlying"], l["expiry"])
                    if px is None:
                        pay = None
                        break
                    intrinsic = max(px - l["strike"], 0) if l["kind"] == "call" else max(l["strike"] - px, 0) if l["kind"] == "put" else 0
                    pay += l["qty"] * 100 * intrinsic
                exp = max(exps)
                out["candidates"].append({"id": c["id"], "status": "expired", "expiry": exp, "cost": c.get("cost"),
                                          "underlying_close_at_expiry": _close_on(data, e["underlying"], exp),
                                          "payoff_at_expiry": pay,
                                          "pnl_at_expiry": (pay - (c.get("cost") or 0)) if pay is not None else None})
            else:
                out["candidates"].append({"id": c["id"], "status": "open", "cost": c.get("cost")})
        return out
    if e["kind"] in ("event", "hedge"):
        cur = next((m for m in data.event_markets() if m["id"] == e.get("event_id")), None) if hasattr(data, "event_markets") else None
        p_now = None
        if cur:
            b, a = cur.get("bid"), cur.get("ask")
            p_now = (b + a) / 2 if b is not None and a is not None else cur.get("last")
        return {"p_then": e.get("p"), "p_now": p_now, "status": (cur or {}).get("status", "not in current universe (may have resolved)")}
    if e["kind"] == "etf_match":
        tgt = list((e.get("target") or {}).keys())
        try:
            c = data.closes(tgt + list(e["weights"]), days=400)
            c = c[c.index >= pd.Timestamp(e["date"])].dropna()
            r = c.pct_change().dropna()
            y = sum(r[t] * w for t, w in e["target"].items()) / sum(e["target"].values())
            p = sum(r[t] * w for t, w in e["weights"].items()) + (e.get("cash") or 0) * 0.04 / 252
            gap = y - p
            return {"days": int(len(r)), "realized_tracking_error": float(gap.std() * (252 ** 0.5)) if len(r) > 5 else None,
                    "target_return": float((1 + y).prod() - 1), "replica_return": float((1 + p).prod() - 1)}
        except Exception:  # noqa: BLE001
            return {"days": 0}
    return {}


def recall(member: str, data, query: str = "", k: int = 5) -> list[dict]:
    """The member's relevant past entries (by ticker/event in the query, else most recent), each with its outcome."""
    toks = {t.upper() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-\.]{1,24}", query or "")}
    rows = entries(member)

    def score(e):
        keys = {str(e.get("underlying", "")).upper(), str(e.get("event_id", "")).upper()} | {t.upper() for t in (e.get("target") or {})}
        return (len(keys & toks), e.get("ts", 0) or 0)
    rows = sorted(rows, key=score, reverse=True)[:k]
    return [dict(e, outcome=outcome(e, data)) for e in rows]


def book_history(member: str, data) -> list[dict]:
    """Book snapshots with their value then and now (code), newest first."""
    out = []
    for b in sorted(_rows(BOOKS), key=lambda r: r["ts"], reverse=True):
        if b["member"] != member:
            continue
        now = 0.0
        for t, v in b["positions"].items():
            p0, p1 = _close_on(data, t, b["date"]), _close_on(data, t, data.today().isoformat())
            now += v * (p1 / p0) if p0 and p1 else v
        then = sum(b["positions"].values())
        out.append({"date": b["date"], "positions": b["positions"], "value_then": then, "value_now": now,
                    "change": now / then - 1 if then else None})
    return out[:10]


# ---------------------------------------------------------------- lessons (gated, advisory, expiring)

def lessons(active_only: bool = True) -> list[dict]:
    rows = json.loads(LESSONS.read_text(encoding="utf-8")) if LESSONS.exists() else []
    if not active_only:
        return rows
    now = time.time()
    return [l for l in rows if l.get("expires", 0) > now]


def accept_lessons(proposed: list[dict], log: list[dict]) -> list[dict]:
    """Code gates on reflection proposals: ≥ MIN_CITES cited entries that exist and have computed outcomes."""
    have = {e["id"]: e for e in log if e.get("id") and e.get("outcome")}
    out, now = [], time.time()
    for p in proposed or []:
        cites = [c for c in p.get("cites", []) if c in have]
        text = str(p.get("text", "")).strip()
        if len(cites) < MIN_CITES or not text or len(text) > 200:
            continue
        regime = p.get("kind") == "regime"
        out.append({"id": uuid.uuid4().hex[:6], "text": text, "cites": cites, "kind": "regime" if regime else "general",
                    "created": now, "expires": now + 86400 * (TTL_REGIME_D if regime else TTL_GENERAL_D),
                    "status": "advisory (only the trainer promotes a lesson into the library)"})
    return out


REFLECT_PROMPT = """You review Worsey's log of past recommendations and their code-computed outcomes. Propose at most 3
LESSONS that would make future recommendations better for these members. Each lesson must cite at least {n} entry ids
whose outcomes support it, and must never contradict trading principles (it is advisory). Mark market-regime lessons
(about current conditions) as kind "regime"; durable ones as "general". Reply with JSON only:
{{"lessons": [{{"text": "...", "kind": "general"|"regime", "cites": ["id", ...]}}]}}

Log:
{log}"""


def reflect(llm, data, model: str) -> list[dict]:
    """The periodic reflection pass (`worsey reflect`): one call proposes, code gates, survivors join the lessons file."""
    log = [dict(e, outcome=outcome(e, data)) for e in entries()]
    brief = [{k: e.get(k) for k in ("id", "member", "date", "kind", "underlying", "event_id", "title", "candidates", "outcome")}
             for e in log][-60:]
    resp = llm.create(model=model, max_tokens=1500, output_config={"effort": "medium"},
                      messages=[{"role": "user", "content": REFLECT_PROMPT.format(n=MIN_CITES, log=json.dumps(brief, default=str))}])
    raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        proposed = json.loads(m.group(0)).get("lessons", []) if m else []
    except ValueError:
        proposed = []
    kept = accept_lessons(proposed, log)
    if kept:
        LESSONS.parent.mkdir(parents=True, exist_ok=True)
        LESSONS.write_text(json.dumps(lessons(active_only=False) + kept, indent=1), encoding="utf-8")
    return kept
