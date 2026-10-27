"""Hard limits on live LLM calls: no runaway spending on Castle's key, whatever the loop does.

Every live Anthropic call goes through `gate()` before it is sent and `record()` after. Limits (env-overridable):

  WORSEY_LLM_MAX_PER_TURN    25         calls in one user turn (router + experts + rewrites; a hedge turn runs 3 experts)
  WORSEY_LLM_MAX_PER_HOUR    300        calls in any rolling hour   (raised Oct 17: ~2¢/call, cost isn't the constraint;
  WORSEY_LLM_MAX_PER_DAY     1500       calls since UTC midnight     the per-turn cap is what stops a runaway loop)
  WORSEY_LLM_MAX_USD_TOTAL   90         estimated USD for the whole project, all time; checked before every call, whatever the daily cap
  WORSEY_LLM_MAX_USD_DAY     25         estimated USD since UTC midnight (Oct 18: replaces the 3M-token cap, which counted
                                        cheap cache reads at full weight and tripped at ~$4.60 of real spend)
  WORSEY_LLM_MAX_TOKENS_DAY  0 (off)    optional raw token cap
  WORSEY_LLM_DISABLED=1                 kill switch (or create cache/usage/STOP)

The ledger (cache/usage/llm_calls.jsonl, one line per call: time, model, tokens) persists across restarts, so a
restarted server can't reset the daily limits. Replays and tests never reach this module.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..data.cache import CACHE_ROOT

USAGE = CACHE_ROOT / "usage"
LEDGER = USAGE / "llm_calls.jsonl"
STOP = USAGE / "STOP"
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """A live LLM call was refused by a limit. The message is safe to show the user."""


def limits() -> dict:
    env = os.environ.get
    return {"per_turn": int(env("WORSEY_LLM_MAX_PER_TURN", 25)), "per_hour": int(env("WORSEY_LLM_MAX_PER_HOUR", 300)),
            "per_day": int(env("WORSEY_LLM_MAX_PER_DAY", 1500)), "tokens_day": int(env("WORSEY_LLM_MAX_TOKENS_DAY", 0)),
            "usd_day": float(env("WORSEY_LLM_MAX_USD_DAY", 25)), "usd_total": float(env("WORSEY_LLM_MAX_USD_TOTAL", 90))}


def _entries(since: float) -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("t", 0) >= since:
            out.append(e)
    return out


def _midnight_utc(now: float) -> float:
    d = datetime.fromtimestamp(now, timezone.utc)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


# Estimated USD per million tokens and per web search. ASSUMED list prices: the ledger records
# tokens, not dollars, and these rates are not confirmed for the 5.x models. Override with WORSEY_PRICE_<FAMILY>_<KIND>,
# e.g. WORSEY_PRICE_SONNET_IN=3. Cache writes bill at 1.25x input, cache reads at 0.1x input.
PRICES = {"sonnet": {"in": 3.0, "out": 15.0}, "haiku": {"in": 1.0, "out": 5.0}, "opus": {"in": 5.0, "out": 25.0}}
WEB_SEARCH_USD = 0.01   # $10 per 1,000 searches


def _rate(model: str, kind: str) -> float:
    fam = next((f for f in PRICES if f in (model or "")), "sonnet")
    return float(os.environ.get(f"WORSEY_PRICE_{fam.upper()}_{kind.upper()}", PRICES[fam][kind]))


def cost(e: dict) -> float:
    """Estimated USD for one ledger entry. Entries from before the split (only `in`) are priced as fresh input, an
    upper bound since much of that was cached."""
    m = e.get("model", "")
    if "fresh" in e:
        tok_in = e["fresh"] * _rate(m, "in") + e.get("cache_write", 0) * 1.25 * _rate(m, "in") + e.get("cache_read", 0) * 0.1 * _rate(m, "in")
    else:
        tok_in = e.get("in", 0) * _rate(m, "in")
    return (tok_in + e.get("out", 0) * _rate(m, "out")) / 1e6 + e.get("web_searches", 0) * WEB_SEARCH_USD


def usage(now: float | None = None) -> dict:
    now = now or time.time()
    day, hour, every = _entries(_midnight_utc(now)), _entries(now - 3600), _entries(0)
    return {"calls_hour": len(hour), "calls_day": len(day), "calls_all": len(every),
            "tokens_day": sum(e.get("in", 0) + e.get("out", 0) for e in day), "limits": limits(),
            "spend_usd": {"hour": round(sum(map(cost, hour)), 2), "day": round(sum(map(cost, day)), 2),
                          "all": round(sum(map(cost, every)), 2),
                          "since": datetime.fromtimestamp(every[0]["t"], timezone.utc).strftime("%b %d") if every else None,
                          "estimate": "assumed list prices; see agent/budget.py PRICES"}}


def gate(turn_calls: int) -> None:
    """Raise BudgetExceeded if the next call would break a limit. turn_calls = calls already made this turn."""
    if os.environ.get("WORSEY_LLM_DISABLED") == "1" or STOP.exists():
        raise BudgetExceeded("Live answers are switched off right now (kill switch).")
    lim, u = limits(), usage()
    if turn_calls >= lim["per_turn"]:
        raise BudgetExceeded(f"That question needed more than {lim['per_turn']} model calls, so I stopped it there.")
    if u["calls_hour"] >= lim["per_hour"]:
        raise BudgetExceeded("I've hit my hourly usage limit. Try again in a bit.")
    over_tokens = lim["tokens_day"] and u["tokens_day"] >= lim["tokens_day"]   # off by default (cache reads made it misleading)
    spent_all = sum(map(cost, _entries(0)))
    if spent_all >= lim["usd_total"]:
        raise BudgetExceeded(f"The project's total LLM budget (${lim['usd_total']:.0f}) is used up.")
    spent_today = sum(map(cost, _entries(_midnight_utc(time.time()))))   # unrounded (usage() rounds for display)
    if u["calls_day"] >= lim["per_day"] or over_tokens or spent_today >= lim["usd_day"]:
        raise BudgetExceeded("I've hit today's usage limit. It resets at midnight UTC.")


def record(model: str, resp: dict | None, ok: bool, source: str | None = None) -> None:
    u = (resp or {}).get("usage") or {}
    fresh, cw, cr = (int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    e = {"t": time.time(), "model": model, "ok": ok, "in": fresh + cw + cr, "out": int(u.get("output_tokens") or 0),
         "fresh": fresh, "cache_write": cw, "cache_read": cr,   # split, so the spend estimate prices each at its rate
         "web_searches": int((u.get("server_tool_use") or {}).get("web_search_requests") or 0)}
    if source:
        e["source"] = source   # e.g. "pmrm" (SPEC §6.1: experiment calls are tagged in the ledger)
    with _lock:
        USAGE.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e) + "\n")
