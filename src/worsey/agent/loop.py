"""The agent loop on the raw Messages API (SPEC §1, §1b, §1c). No framework.

Router turn: user msg → Claude (router) → tool calls → results → … → reply.
Experts (event_analyst, options_desk, equity_analyst) are the same loop with their own prompt, tools and a fresh
message list (context isolation).

Who writes the reply (DECISIONS #100): on judgment turns the EXPERT writes the user-facing reply;
code checks it (grounding + format, one rewrite by that expert) and it becomes the turn's reply. Hedge requests run
event_analyst and options_desk in parallel and the reply is their two options, side by side. The router writes the
reply only on simple turns (portfolio, exposures, chit-chat), with the same checks.
History is append-only: whole response content goes back unchanged.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config, lookthrough

from . import grounding, tools
from .llm import EXPERT_MODEL, ROUTER_MODEL
from .session import Session, ToolError
from .validate import validate

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
MAX_STEPS = 8
EXPERT_DONE = "[expert]"   # the router's whole final message when an expert wrote the reply

ROLES = {
    "router": {"model": ROUTER_MODEL, "prompt": "router.md", "max_tokens": config.ROUTER_MAX_TOKENS, "effort": "low"},
    # quality over latency (Oct 17, DECISIONS #98): experts at medium effort with room to think and research
    "event_analyst": {"model": EXPERT_MODEL, "prompt": "event_analyst.md", "max_tokens": 8000, "effort": "medium"},
    "options_desk": {"model": EXPERT_MODEL, "prompt": "options_desk.md", "max_tokens": 8000, "effort": "medium"},
    "equity_analyst": {"model": EXPERT_MODEL, "prompt": "equity_analyst.md", "max_tokens": 8000, "effort": "medium"},
    "etf_desk": {"model": EXPERT_MODEL, "prompt": "etf_desk.md", "max_tokens": 8000, "effort": "medium"},
    "recent_markets_analyst": {"model": EXPERT_MODEL, "prompt": "recent_markets_analyst.md", "max_tokens": 8000, "effort": "medium"},
    "natural_world_analyst": {"model": EXPERT_MODEL, "prompt": "natural_world_analyst.md", "max_tokens": 8000, "effort": "medium"},
}
WRITERS = ("event_analyst", "options_desk", "equity_analyst", "etf_desk", "recent_markets_analyst", "natural_world_analyst")   # experts that write user-facing replies (prompts/voice.md)

EXPERT_TOOL = {"type": "object", "properties": {"request": {"type": "string", "minLength": 1}},
               "required": ["request"], "additionalProperties": False}
EXPERT_DESC = {
    "event_analyst": "Follow-ups on an event already shown (e.g. 'why is TLT a hedge here?'). For discovery use scan_risks; "
                     "for one event use analyze_event. The expert writes the reply.",
    "options_desk": "Any view to trade or change to it ('NVDA rips 10% in 6 weeks', 'make it 2 weeks', 'show me the put spread'), "
                    "vol questions, options explanations. Pass the user's words plus resolved references. The expert writes the reply.",
    "equity_analyst": "Questions about a company or stock ('why is NVDA down?', 'what does AAPL make in China?'): researches public "
                      "sources and Massive data. Pass the user's words and the ticker(s). The expert writes the reply.",
    "natural_world_analyst": "Commodities and weather: oil, gasoline, diesel, natural gas, power, corn, soybeans, wheat; futures "
                             "curves, carry and roll, cracks, weather forecasts (GFS / ECMWF / AI) and their market impact, WASDE "
                             "and EIA reports, pipelines and basis, how to trade futures. The expert writes the reply.",
    "recent_markets_analyst": "Market commentary: 'what do you think of the market?', 'what's going on with rates / oil / "
                              "tech?', 'why is everything down today?'. Reads the tape (prices, vol, odds) first, then cited "
                              "news. Not for one company (equity_analyst) or one event (analyze_event). The expert writes the reply.",
    "etf_desk": "Someone who can't trade a stock (employer restrictions, 'I can only buy ETFs') and wants that exposure through "
                "ETFs; and every follow-up to an ETF match in progress ('what if I can't use SMH?', 'make it 2 ETFs'). It runs "
                "the intake about what they may trade, then the match. Pass the user's words. The expert writes the reply.",
}
# the basic version on purpose: web_search_20260209's dynamic filtering answers from code output WITHOUT citations, and every
# company fact must trace to a cited source (DECISIONS #101). Measured Oct 17: 20260209 → 0 citations; 20250305 → cited text.
CONSULT_TOOL = {"name": "consult",
                "description": "Ask another expert a question in its lane (DECISIONS #112): event_analyst (events, markets, "
                               "scenarios), options_desk (options, vol, structures), equity_analyst (a company's business and "
                               "news). One hop only: the expert you ask can't consult further. Several consults in one step run "
                               "in parallel. Their answer is notes for you, not for the user.",
                "input_schema": {"type": "object", "properties": {
                    "expert": {"type": "string", "enum": ["event_analyst", "options_desk", "equity_analyst", "etf_desk",
                                                          "recent_markets_analyst", "natural_world_analyst"]},
                    "question": {"type": "string", "minLength": 3}}, "required": ["expert", "question"],
                    "additionalProperties": False}}
_local = threading.local()     # consult depth per thread: experts in parallel each run in their own thread

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": config.WEB_SEARCH_MAX_USES}


def _approved_corrections(role: str) -> str:
    """The trainer's approved /fixes that live in the library (playbook, strategy files) rather than in the prompt files."""
    if role != "options_desk" or os.environ.get("WORSEY_LIBRARY_DIR"):   # replays/tests pin the fixture library
        return ""
    from .train_fix import SECTION
    lib = PROMPTS.parent / "library"
    parts = []
    for f in [lib / "playbook" / "corrections.md"] + sorted((lib / "strategies").glob("*.md")):
        if f.exists():
            t = f.read_text(encoding="utf-8")
            if SECTION in t:
                parts.append(t.split(SECTION, 1)[1].strip())
    return ("\n\n## The trainer's approved corrections (they outrank your defaults)\n" + "\n".join(parts)) if parts else ""


def _prompt_file(role: str) -> str:
    """event_analyst has its own prompt per scenario engine (DECISIONS #179); the v1 prompt stays byte-identical so the
    recorded demos (which pin v1) still replay."""
    if role == "event_analyst" and config.SCENARIO_ENGINE == "planb":
        return "event_analyst_planb.md"
    return ROLES[role]["prompt"]


def system_blocks(role: str) -> list[dict]:
    text = (PROMPTS / _prompt_file(role)).read_text(encoding="utf-8") + _approved_corrections(role)
    if role in WRITERS:
        text += "\n\n" + (PROMPTS / "voice.md").read_text(encoding="utf-8")
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# ---------------------------------------------------------------- coverage: which experts run (code, testable)


def _single_stocks(s: Session, tickers: list[str]) -> list[str]:
    """Held equities that are single companies (not ETFs: those have holdings)."""
    out = []
    for t in tickers:
        if s.priced is None or t not in s.priced.index or s.priced.at[t, "asset_class"] != "equity":
            continue
        h = s.data.holdings(t, sectors=False)
        if h is None or h.empty:
            out.append(t)
    return out


def coverage(s: Session, kind: str, event: dict | None = None, names: list[str] | None = None) -> list[dict]:
    """Which experts an entry tool runs, and why. Unbuilt experts are logged as skipped, never silently dropped."""
    plan = [{"expert": "event_analyst", "run": True, "why": "always for events (writes the reply)"}]
    if kind == "scan" and s.priced is not None:
        w = float(s.priced.loc[[t for t in s.priced.index if t in config.WEATHER_SENSITIVE], "weight"].sum())
        plan.append({"expert": "natural_world_analyst", "run": w >= config.WEATHER_MATERIALITY,
                     "why": f"weather-sensitive share of book {w:.0%} vs threshold {config.WEATHER_MATERIALITY:.0%}"})
        hold = {t: h for t in s.priced.index[s.priced["asset_class"] == "equity"]
                if (h := s.data.holdings(t, sectors=False)) is not None and not h.empty}
        eff = lookthrough.effective_exposure(s.priced["value"], hold)
        big = [n for n in eff.index if not lookthrough.is_bucket(n) and eff.at[n, "weight"] >= config.EQUITY_ANALYST_NAME_WEIGHT
               and s.priced.get("asset_class", {}).get(n, "equity") == "equity"]
        plan.append({"expert": "equity_analyst", "run": bool(big), "names": big,
                     "why": f"names ≥ {config.EQUITY_ANALYST_NAME_WEIGHT:.0%} look-through: {big}"})
    if kind == "event" and event is not None:
        weather = event.get("category") == "Climate and Weather"
        plan[0] = {"expert": "natural_world_analyst" if weather else "event_analyst", "run": True,
                   "why": f"event category {event.get('category')!r} (writes the reply)"}
        plan.append({"expert": "equity_analyst", "run": bool(names), "names": names or [],
                     "why": f"single stocks among the top affected holdings: {names or 'none'}"})
    for p in plan:
        if p["run"] and p["expert"] not in config.BUILT_EXPERTS:
            p["run"], p["why"] = False, p["why"] + " → would run, not built yet (Tier B)"
    return plan


def _equity_notes(llm, s: Session, plan: list[dict], question: str) -> dict | None:
    """equity_analyst runs FIRST when covered, so its cited findings feed the expert who writes the reply."""
    p = next((p for p in plan if p["expert"] == "equity_analyst" and p["run"]), None)
    if p is None:
        return None
    return run_expert(llm, s, "equity_analyst", f"Notes for event_analyst (not the user): for {', '.join(p['names'])}, what in "
                      f"the business makes it sensitive to this: {question}. ≤ 6 short lines, every company fact cited.",
                      mode="notes")


# ---------------------------------------------------------------- entry tools (SPEC §3.4h)
# The router picks *what kind of question* this is; these deterministic orchestrators compute the card in code,
# decide which experts run (coverage rules), run them, and hand back the expert-written reply.


def _scan_risks_planb(llm, s: Session, horizon_days: int, n: int, question: str, user_events: list | None = None) -> dict:
    """Plan B discovery (DECISIONS #179): every kind of big event and release replayed through this book's factor
    exposures, in code (the card); then event_analyst writes the reply."""
    t0 = time.perf_counter()
    table = tools.compact(tools.scan_risks_planb(s, horizon_days=horizon_days, n=n))
    s.turn_results.append({"scan_risks.table": table})
    s.timings.append({"kind": "card", "name": "scan_risks.table", "start": t0 - s.turn_t0, "s": time.perf_counter() - t0})
    s.send("card", {"type": "risks", "payload": table})
    upcoming = None
    try:   # the event universe matched to this book (#218, #222-234), in code, always; proposals need a live LLM
        from ..agent.llm import AnthropicLLM, RecordingLLM
        from .. import discovery, scan_messages
        s.send("status", {"text": "checking what's coming up…"})
        live_llm = llm if isinstance(llm, (AnthropicLLM, RecordingLLM)) else None
        s.__dict__["scan_messages"] = []

        def progress(stage, data):   # the scan's messages, sent as each stage finishes (#234)
            msg = None
            if stage == "listed":
                msg = scan_messages.book(table, data)
            elif stage == "researching":
                msg = scan_messages.researching(data["forecasting"])
                s.send("status", {"text": "researching odds…"})
            if msg:
                s.send("reply", {"text": msg})
                s.__dict__.setdefault("scan_messages", []).append(msg)
        upcoming = tools.compact(discovery.watchlist(s, live_llm, horizon_days=horizon_days, user_events=user_events,
                                                     replay=table["risks"], progress=progress if live_llm else None))
        if upcoming.get("picks") or upcoming.get("yours"):
            s.turn_results.append({"watchlist": upcoming})
        for r in (upcoming.get("yours") or []) + (upcoming.get("picks") or []):   # one message per event, written by code
            msg = scan_messages.event(r, upcoming.get("normal_2_day_swing_usd") or 0.0)
            s.send("reply", {"text": msg})
            s.__dict__.setdefault("scan_messages", []).append(msg)
    except Exception:  # noqa: BLE001 — discovery must never break the scan
        upcoming = None
    s.send("status", {"text": "thinking it through…"})
    has = upcoming and (upcoming.get("picks") or upcoming.get("yours"))
    sent = s.__dict__.get("scan_messages") or []
    final = {k: (upcoming or {}).get(k) for k in ("yours", "picks", "more_count", "noise_for_you", "normal_2_day_swing_usd", "how")}
    reply = run_expert(llm, s, "event_analyst", f"User asked: {question or 'what could hit my portfolio?'}\nWrite the reply. "
                       "Risks (the past kinds of events replayed on this book): " + json.dumps(table, default=str)
                       + (("\nAlready sent to the user, in order, one message per event with its probability, each holding's "
                           "average move and 9-in-10 range, and the book's (don't repeat them): " + json.dumps(sent) if sent else "")
                          + "\nThe scan's result: " + json.dumps(final, default=str)
                          + "\nWrite the last message: two to four short lines in plain words. Which event matters most for "
                          "this book and why, what is noise for it, and that you looked at more_count other events and can show "
                          "them. Quote a number only if it appears in the messages above; never compute or adjust one. "
                          "No tables, no lists of holdings." if has else ""),
                       mode="reply")
    s.turn_replies = [reply]
    plan = [{"expert": "event_analyst", "run": True, "why": "always for events (writes the reply)"}]
    s.routing.append({"entry": "scan_risks", "engine": "planb", "plan": plan, "seconds": round(time.perf_counter() - t0, 2)})
    return {"table_shown_to_user": True, "risks": [r["risk_id"] for r in table["risks"]], "expert_reply": reply["text"]}


def _analyze_event_planb(llm, s: Session, event_id: str | None, query: str | None, question: str) -> dict:
    """Plan B, one event (DECISIONS #179): the expert structures the event, reads code's evidence (event_evidence),
    judges each factor's move, and code prices it (price_scenario, the card); then the expert writes the reply."""
    t0 = time.perf_counter()
    s.require_portfolio()
    row = None
    if event_id:
        row = next((r for r in (s.__dict__.get("planb_scan") or {}).get("risks", []) if r["risk_id"] == event_id), None)
    ctx = {"user_words": query or event_id, "scan_row": tools.compact(row) if row else None}
    s.send("status", {"text": "checking past events…"})
    reply = run_expert(llm, s, "event_analyst", f"User asked: {question or 'tell me more'}\nEvent: " + json.dumps(ctx, default=str)
                       + "\nStructure the event, call event_evidence, judge the factor buckets, call price_scenario, then write "
                       "the reply. The user sees the scenario card when you price it.", mode="reply")
    s.turn_replies = [reply]
    sc = s.__dict__.get("planb_scenario") or {}
    s.routing.append({"entry": "analyze_event", "engine": "planb", "event": sc.get("id"), "query": query,
                      "plan": [{"expert": "event_analyst", "run": True, "why": "Plan B assisted loop"}],
                      "seconds": round(time.perf_counter() - t0, 2)})
    return {"event": sc.get("id"), "expert_reply": reply["text"]}


def scan_risks(llm, s: Session, horizon_days: int = 91, n: int = 6, question: str = "", user_events: list | None = None) -> dict:
    """Discovery entry tool: the ranked table in pure code first (the card), then event_analyst writes the reply."""
    if config.SCENARIO_ENGINE == "planb":
        return _scan_risks_planb(llm, s, horizon_days, n, question, user_events)
    t0 = time.perf_counter()
    table = tools.compact(tools.scan_events(s, horizon_days=horizon_days, n=n))
    s.turn_results.append({"scan_risks.table": table})
    s.timings.append({"kind": "card", "name": "scan_risks.table", "start": t0 - s.turn_t0, "s": time.perf_counter() - t0})
    s.send("card", {"type": "scan", "payload": table})
    plan = coverage(s, "scan")
    notes = ((s.data.structure() or {}).get("notes") or {}) if hasattr(s.data, "structure") else {}
    shown = [e["event_id"] for e in table["events"] + table.get("drift", [])]
    equity = _equity_notes(llm, s, plan, "the events in this scan")
    nat = next((p for p in plan if p["expert"] == "natural_world_analyst" and p["run"]), None)
    if nat:
        n_ = run_expert(llm, s, "natural_world_analyst", "Notes for event_analyst (not the user): which events in this scan run "
                        "through weather or commodities for this book, and what the curves and forecasts say. ≤ 6 lines.", mode="notes")
        equity = {"text": ((equity or {}).get("text", "") + "\nNatural-world notes: " + n_["text"]).strip()}
    brief = {"events": [{k: e.get(k) for k in ("event_id", "short", "event", "p", "pnl_yes", "pnl_no", "sigma", "pnl_yes_by_class",
                                               "drivers", "flags", "above_divider", "evidence", "close_date")} for e in table["events"]],
             "drift": [{k: e.get(k) for k in ("event_id", "short", "p", "sigma_drift", "close_date")} for e in table.get("drift", [])],
             "joint": table.get("joint"), "horizon_days": horizon_days,
             "event_notes": {i: notes[i] for i in shown if i in notes}}   # build-time notes are input, not a substitute (#98)
    s.send("status", {"text": "thinking it through…"})
    reply = run_expert(llm, s, "event_analyst", f"User asked: {question or 'what could hit my portfolio?'}\nWrite the reply. The "
                       "user sees the scan card already. Scan: " + json.dumps(brief, default=str)
                       + ("\nEquity analyst notes: " + equity["text"] if equity else ""), mode="reply")
    s.turn_replies = [reply]
    s.routing.append({"entry": "scan_risks", "plan": plan, "seconds": round(time.perf_counter() - t0, 2)})
    return {"table_shown_to_user": True, "events": [e["event_id"] for e in table["events"]], "expert_reply": reply["text"]}


def analyze_event(llm, s: Session, event_id: str | None = None, query: str | None = None, question: str = "") -> dict:
    """Pointed-question entry tool: drivers and priced-in math in code, then the expert routed by event type writes the
    reply (equity_analyst first, when a top affected holding is a single stock).
    Pass event_id for a scanned row, or query ("tariffs") to match against the whole universe (Case 2)."""
    if config.SCENARIO_ENGINE == "planb":
        return _analyze_event_planb(llm, s, event_id, query, question)
    t0 = time.perf_counter()
    match = None
    rows_now = {e["event_id"] for e in (s.last_scan or {}).get("events", []) + (s.last_scan or {}).get("drift", [])}
    template = None
    if query and (not event_id or event_id not in rows_now):
        try:
            match = tools.compact(tools.find_event(s, query))
        except ToolError:
            match = {"query": query, "candidates": []}
        priced = [c for c in match["candidates"] if c["priced"]]
        strong = [c for c in priced if (c.get("keyword_score") or 0) >= config.STRONG_MATCH_SCORE]
        kind = tools.event_type(query)
        if not strong and kind:  # no strong market: say so, then the type's template
            template = tools.compact(tools.event_template(s, kind))
            match["strong_market"] = False
        if not priced and template is None:
            raise ToolError(f"markets matched {query!r} but none has a usable history: unpriced")
        event_id = priced[0]["event_id"] if priced else None
    if not event_id and template is None:
        raise ToolError("give event_id or query")
    detail = tools.compact(tools.explain_event(s, event_id)) if event_id else {"event": None}
    if match:
        detail["match"] = match
    if template is not None:
        detail["template"] = template
        detail["no_strong_market"] = True
        s.active_event = None                     # the scenario now under discussion is the template, not the weak match
        s.__dict__["template_weak_match"] = event_id
    s.turn_results.append({"analyze_event.detail": detail})
    s.timings.append({"kind": "card", "name": "analyze_event.detail", "start": t0 - s.turn_t0, "s": time.perf_counter() - t0})
    ev = detail.get("event") if isinstance(detail.get("event"), dict) else {}
    if ev.get("event_id"):
        s.send("card", {"type": "meta", "payload": {"event_id": ev["event_id"], "event": ev.get("event"), "short": ev.get("short")}})
    if template is not None:
        s.send("card", {"type": "template", "payload": template})
    row = next((e for e in s.last_scan["events"] + s.last_scan.get("drift", []) if e["event_id"] == event_id), {})
    if template is not None:
        top = sorted(template.get("by_position", {}), key=lambda t: template["by_position"][t])[:3]
    else:
        top = [p["ticker"] for p in detail.get("positions", [])[:3]]
    plan = coverage(s, "event", row, names=_single_stocks(s, top))
    strong_match = match is None or any(c.get("priced") and (c.get("keyword_score") or 0) >= config.STRONG_MATCH_SCORE
                                        for c in match.get("candidates", []))
    if match is not None:
        match["strong_match"] = strong_match  # code's keyword verdict; the expert judges the actual fit
    notes = ((s.data.structure() or {}).get("notes") or {}) if hasattr(s.data, "structure") else {}
    if event_id in notes:
        detail["event_note"] = notes[event_id]
    equity = _equity_notes(llm, s, plan, question or (ev.get("short") or ev.get("event") or "this event"))
    s.send("status", {"text": "thinking it through…"})
    writer = plan[0]["expert"] if plan[0]["run"] else "event_analyst"
    reply = run_expert(llm, s, writer, f"User asked: {question or 'tell me more'}\nWrite the reply. If a keyword match is shown, "
                       "judge whether the market really is the event the user means (variable, threshold, deadline, resolution "
                       "source); say so plainly if it isn't. If `no_strong_market` is set, say first and plainly that no strong "
                       "market prices this event, then use `template` (unpriced, conditional P&L from past episodes, unreviewed). "
                       "Event detail: " + json.dumps(detail, default=str)
                       + ("\nEquity analyst notes: " + equity["text"] if equity else ""), mode="reply")
    s.turn_replies = [reply]
    s.routing.append({"entry": "analyze_event", "event": event_id, "query": query, "plan": plan,
                      "seconds": round(time.perf_counter() - t0, 2)})
    return {"event": ev.get("event_id"), "template": template is not None, "expert_reply": reply["text"]}


def compare_hedges(llm, s: Session, event_id: str | None = None, kind: str = "both", question: str = "",
                   bad_case_weight: float = 1.0) -> dict:
    """Hedge entry tool: the hedge table in code (the card), then event_analyst (the event hedge) and
    options_desk (the options hedge) IN PARALLEL, each writing its option; the reply shows both, unless the user asked
    for one kind."""
    t0 = time.perf_counter()
    table = tools.compact(tools.compare_hedges_tool(s, event_id=event_id, bad_case_weight=bad_case_weight))
    s.turn_results.append({"compare_hedges": table})
    s.timings.append({"kind": "card", "name": "compare_hedges", "start": t0 - s.turn_t0, "s": time.perf_counter() - t0})
    s.send("card", {"type": "hedge", "payload": table})
    ask = (f"User asked: {question or 'how would I hedge that?'}\nThe user sees the hedge card (your book after each hedge, "
           "if YES / if NO, cost, coverage). Hedge table: " + json.dumps(table, default=str)
           + ("\nLead with code's `should_you_hedge` verdict and its numbers (the normal bad case, the cheapest hedge's cost, "
              "how often such protection has paid). If it says 'skip by default', say plainly that hedging this probably isn't "
              "worth the premium and why, then describe the hedges only as what they'd cost if the user still wants one."
              if table.get("should_you_hedge") else ""))
    jobs = []
    if kind in ("both", "event"):
        jobs.append(("event hedge", "event_analyst", ask + "\nWrite the EVENT-HEDGE option: hedging the event itself (the "
                     "prediction contract) or its direct channel. Is it the right event, what basis risk, worth the cost?"))
    if kind in ("both", "options"):
        jobs.append(("options hedge", "options_desk", ask + "\nWrite the OPTIONS-HEDGE option: which option hedge in the table "
                     "(if any) suits this user and why, what it actually protects against, and its basis risk."))
    s.send("status", {"text": "thinking it through…"})
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:   # the two experts run in parallel (DECISIONS #100)
        futs = [(label, ex.submit(run_expert, llm, s, role, req, "reply")) for label, role, req in jobs]
        s.turn_replies = [dict(f.result(), label=label) for label, f in futs]
    s.routing.append({"entry": "compare_hedges", "event": table.get("event_id"), "kind": kind,
                      "plan": [{"expert": r, "run": True, "why": l} for l, r, _ in jobs], "seconds": round(time.perf_counter() - t0, 2)})
    return {"event": table.get("event_id"), "expert_reply": [r["text"] for r in s.turn_replies]}


ENTRY = {
    "scan_risks": (scan_risks, "Discovery: 'what could hit me?'. Ranks prediction-market events by σ of resolution P&L (code), "
                   "events after the horizon as drift rows; event_analyst writes the reply.",
                   tools._obj({"horizon_days": {"type": "integer", "minimum": 7, "maximum": 365},
                               "n": {"type": "integer", "minimum": 1, "maximum": 10}, "question": {"type": "string"}}, [])),
    "analyze_event": (analyze_event, "Pointed question about one event: which holdings drive it, priced-in math, correlations; "
                      "the right expert writes the reply. Use event_id for a row already shown, or query (plain words, e.g. "
                      "'tariffs') to find it among all markets.",
                      tools._obj({"event_id": {"type": "string", "minLength": 1}, "query": {"type": "string", "minLength": 2},
                                  "question": {"type": "string"}}, [])),
    "compare_hedges": (compare_hedges, "Hedging a risk or the portfolio: the hedge table for one event (default: the active one) "
                       "in code, then the event hedge (event_analyst) and the options hedge (options_desk) in parallel, shown as "
                       "two options. kind='event' or 'options' only if the user asked for that kind specifically.",
                       tools._obj({"event_id": {"type": "string"}, "kind": {"type": "string", "enum": ["both", "event", "options"]},
                                   "question": {"type": "string"}}, [])),
}


ENTRY_PLANB = {   # DECISIONS #179: same entry tools, Plan B wording (v1 wording above stays for the recorded demos)
    "scan_risks": "Discovery: 'what could hit me?'. Replays every kind of big past event and each scheduled release through "
                  "this book's factor exposures (code), and matches what's coming (markets, calendar, news) with events "
                  "proposed from the book's exposures; event_analyst writes the reply. If the user names their own worries "
                  "AND asks for anything else ('I'm worried about an oil spike, and anything else my book cares about'), "
                  "use this with user_events = their events in their words: they are priced first and the rest of the "
                  "universe excludes them. One named event and nothing else: analyze_event.",
    "analyze_event": "Pointed question about one event, on a market or not ('what if Iran escalates?', 'the CPI one'): "
                     "event_analyst structures it, reads code's evidence (past events, base rates, the book's exposures) and "
                     "code prices first-hours and end-of-day ranges. Use event_id for a row already shown (e.g. "
                     "'category:trade_tariffs'), else query = the event in plain words.",
    "compare_hedges": "Hedging a risk or the portfolio: the hedge table for the scenario under discussion (default: the "
                      "active one) in code, priced on its middle and bad case, then the event hedge (event_analyst) and the "
                      "options hedge (options_desk) in parallel, shown as two options. kind='event' or 'options' only if the "
                      "user asked for that kind specifically.",
}


def _library_catalog() -> str:
    """The strategies the builder can price right now, so `include` ids come from the library,
    never from a hardcoded list (a stale list sent "short the risky" to the wrong structures, Oct 17)."""
    try:
        lib = tools._library()
    except Exception:  # noqa: BLE001
        return ""
    rows = [f"{e['id']} = {e.get('name', e['id'])}" + (f" (aka {', '.join(e['aka'])})" if e.get("aka") else "")
            + (" [stand-in]" if e.get("stand_in") else "") for e in sorted(lib, key=lambda e: e["id"])]
    return " Strategy ids for `include`: " + "; ".join(rows) + "."


# Plan B only (DECISIONS #200): the user's weight on a loss beyond their normal bad case; v1's schema stays byte-identical
# so the recorded demos replay
PLANB_ENTRY_SCHEMA = {"scan_risks": tools._obj({
    "horizon_days": {"type": "integer", "minimum": 7, "maximum": 365}, "n": {"type": "integer", "minimum": 1, "maximum": 10},
    "question": {"type": "string"},
    "user_events": {"type": "array", "maxItems": 4, "items": {"type": "string", "minLength": 3, "maxLength": 200},
                    "description": "events the user named themselves, in their words (e.g. 'an oil spike'); empty if none"}}, []),
    "compare_hedges": tools._obj({
    "event_id": {"type": "string"}, "kind": {"type": "string", "enum": ["both", "event", "options"]}, "question": {"type": "string"},
    "bad_case_weight": {"type": "number", "minimum": 1, "maximum": 10,
                        "description": "how much more a loss beyond the user's normal bad case hurts them than everyday swings; "
                                       "1 unless they said so (e.g. 'I can't afford a big drawdown this month' -> 4)"}}, [])}


def tool_defs(role: str) -> list[dict]:
    defs = tools.schemas(role)
    if role == "options_desk":
        cat = _library_catalog()
        defs = [dict(d, description=d["description"] + cat) if d["name"] == "view_and_strategies" else d for d in defs]
    if role in ("router", "options_desk", "equity_analyst") and os.environ.get("WORSEY_LIVE_QUOTES") == "1":
        # live chat only (set by `worsey serve`): adding it everywhere would change every recorded request
        defs = defs + [{"name": "live_quote", "description": tools.TOOLS["live_quote"][1], "input_schema": tools.TOOLS["live_quote"][2]}]
    if role == "router":
        planb = config.SCENARIO_ENGINE == "planb"
        defs += [{"name": n, "description": ENTRY_PLANB[n] if planb else ENTRY[n][1],
                  "input_schema": PLANB_ENTRY_SCHEMA.get(n, ENTRY[n][2]) if planb else ENTRY[n][2]} for n in ENTRY]
        defs += [{"name": n, "description": EXPERT_DESC[n], "input_schema": EXPERT_TOOL} for n in EXPERT_DESC]
    if role in ("equity_analyst", "recent_markets_analyst", "natural_world_analyst"):
        defs = defs + [WEB_SEARCH]
    if role in WRITERS:
        defs = defs + [CONSULT_TOOL]
    return defs


STATUS = {  # SPEC §1d-0: lowercase, ≤ ~4 words, present tense, only from real tool calls
    "scan_risks": "scanning prediction markets…", "scan_events": "scanning prediction markets…",
    "compare_hedges": "pricing hedges…", "earnings_calendar": "checking earnings dates…", "stress_scenario": "stress-testing your book…", "upcoming_events": "checking what's coming up…", "view_and_strategies": "checking the options chain…",
    "propose_strategies": "checking the options chain…", "market_state": "checking the options chain…",
    "load_portfolio": "pulling prices…", "exposures": "pulling prices…", "explain_event": "pulling prices…",
    "event_analyst": "thinking it through…", "options_desk": "thinking it through…", "chart": "drawing the chart…",
    "equity_analyst": "reading up on the company…", "company_facts": "reading up on the company…",
    "live_quote": "getting a live quote…", "price_structure": "pricing that structure…", "show_structure": "drawing the payoff…",
    "etf_desk": "matching with ETFs…", "replicate_with_etfs": "testing ETF matches…", "set_restrictions": "noting your rules…",
    "reference": "checking the reference…", "recall": "checking our history…", "book_history": "checking your book history…",
    "natural_world_analyst": "checking the curves and weather…", "futures_curve": "reading the futures curve…",
    "weather_gas": "checking the weather models…", "weather_crops": "checking crop weather…", "crack_spread": "pricing the crack…",
    "recent_markets_analyst": "reading the tape…", "tape_summary": "reading the tape…", "market_digest": "reading the tape…",
    "event_evidence": "checking past events…", "price_scenario": "pricing the scenario…",
}


def _status_text(s: Session, name: str, inp: dict) -> str:
    if name == "scan_risks" and config.SCENARIO_ENGINE == "planb":
        return "replaying past events on your book…"
    if name == "analyze_event":
        label = inp.get("query") or inp.get("event_id") or "that event"
        rows = {e["event_id"]: e for e in ((s.last_scan or {}).get("events", []) + (s.last_scan or {}).get("drift", []))}
        if inp.get("event_id") in rows:
            label = rows[inp["event_id"]].get("short") or rows[inp["event_id"]].get("event") or label
        label = str(label)
        return f"digging into {label[:28] + '…' if len(label) > 28 else label}…"
    return STATUS.get(name, "working on it…")


def _cards(name: str, inp: dict, out: dict) -> list[tuple[str, dict]]:
    """Card events straight from plain tool results (SPEC §1d-0). Entry tools send their own cards as soon as the code
    part is done, before the experts run."""
    if not isinstance(out, dict) or "error" in out:
        return []
    if name == "scan_events":
        return [("scan", out)]
    if name == "price_scenario":
        return [("scenario", out)]
    if name == "stress_scenario":
        return [("stress", out)]
    if name == "upcoming_events" and out.get("rows"):
        return [("upcoming", out)]
    if name == "earnings_calendar" and out.get("rows"):
        return [("earnings", out)]
    # pricing shows a compact summary; the full diagram card only for an agreed structure, on request
    if name in ("view_and_strategies", "propose_strategies"):
        return [("strategy", dict(out, compact=True))]
    if name == "price_structure":
        st = out.get("structure") or {}
        return [("strategy", {"view": out.get("view"), "market": {"spot": out.get("spot")}, "candidates": [st], "compact": True})]
    if name == "show_structure":
        return [("strategy_full", out)]
    if name == "chart":
        return [("chart", out)]
    if name == "replicate_with_etfs":
        return [("etf_match", out)]
    return []


def _consult(llm, s: Session, caller: str, inp: dict) -> tuple[str, bool]:
    """Brokered expert-to-expert question (DECISIONS #112): code decides whether it happens. One hop; the consulted
    expert answers from the asker's warm session for that lane; it can't consult further."""
    if getattr(_local, "depth", 0) > 0:
        return json.dumps({"error": "consult is one hop: answer from what you have"}), True
    if inp["expert"] == caller:
        return json.dumps({"error": "ask a different expert"}), True
    s.send("status", {"text": STATUS.get(inp["expert"], "thinking it through…")})
    _local.depth = 1
    try:
        r = run_expert(llm, s, inp["expert"], f"{caller} asks: {inp['question']}", mode="consult")
    finally:
        _local.depth = 0
    s.routing.append({"entry": "consult", "plan": [{"expert": inp["expert"], "run": True, "why": f"asked by {caller}"}],
                      "question": inp["question"]})
    return json.dumps({"from": inp["expert"], "notes": r["text"]}), False


def _execute(llm, s: Session, name: str, inp: dict, strikes: dict, caller: str = "router") -> tuple[str, bool]:
    """Run one tool. Returns (content JSON, is_error). Validation errors get one retry per tool per turn."""
    if name == "consult":
        errs = validate(inp, CONSULT_TOOL["input_schema"])
        return (json.dumps({"validation_error": "; ".join(errs)}), True) if errs else _consult(llm, s, caller, inp)
    schema = (EXPERT_TOOL if name in EXPERT_DESC else
              (PLANB_ENTRY_SCHEMA.get(name, ENTRY[name][2]) if config.SCENARIO_ENGINE == "planb" else ENTRY[name][2]) if name in ENTRY
              else tools.TOOLS.get(name, (None, None, None))[2])
    if schema is None:
        return json.dumps({"error": f"unknown tool {name}"}), True
    errs = validate(inp, schema)
    if errs:
        strikes[name] = strikes.get(name, 0) + 1
        msg = "; ".join(errs) + (" (second failure: stop and tell the user this couldn't be computed)" if strikes[name] > 1 else "")
        return json.dumps({"validation_error": msg}), True
    t_tool = time.perf_counter()
    s.send("status", {"text": _status_text(s, name, inp)})
    s.replies_fresh = False   # set again below only if THIS tool produced the expert-written reply
    try:
        if name in EXPERT_DESC:   # the router handed the whole question to an expert: it writes the reply
            r = run_expert(llm, s, name, inp["request"], mode="reply")
            s.turn_replies = [r]
            s.routing.append({"entry": name, "plan": [{"expert": name, "run": True, "why": "router: the expert's question"}]})
            out = {"expert_reply": r["text"]}
        elif name in ENTRY:
            out = ENTRY[name][0](llm, s, **inp)
        else:
            out = tools.compact(tools.TOOLS[name][0](s, **inp))
            s.turn_results.append({name: out})
            s.timings.append({"kind": "card", "name": name, "start": t_tool - s.turn_t0, "s": time.perf_counter() - t_tool})
        s.timings.append({"kind": "tool", "name": name, "start": t_tool - s.turn_t0, "s": time.perf_counter() - t_tool})
        s.log.append({"tool": name, "input": inp, "ok": True})
        s.replies_fresh = isinstance(out, dict) and "expert_reply" in out
        for kind, payload in _cards(name, inp, out):
            s.send("card", {"type": kind, "payload": payload})
        return json.dumps(out, default=str), False
    except ToolError as e:
        s.log.append({"tool": name, "input": inp, "ok": False, "error": str(e)})
        return json.dumps({"error": str(e)}), True
    except Exception as e:  # noqa: BLE001 — a tool bug must surface to the model as an error, never crash the chat
        from ..data.cache import OfflineCacheMiss
        from .budget import BudgetExceeded
        if isinstance(e, BudgetExceeded):
            raise          # a spending limit ends the turn; it is never retried by the model
        if isinstance(e, OfflineCacheMiss):
            # live chat on cached data (no Massive/Kalshi calls): say what's missing instead of "internal error"
            s.log.append({"tool": name, "input": inp, "ok": False, "error": f"offline cache miss: {e}"})
            return json.dumps({"error": "this needs market data that isn't in the cached snapshot (live market data is off "
                               "for this chat); say so plainly"}), True
        s.log.append({"tool": name, "input": inp, "ok": False, "error": f"internal: {type(e).__name__}: {e}"})
        return json.dumps({"error": "internal tool error; say this couldn't be computed"}), True  # type only in the log: replays stay identical


def _web_citations(s: Session, content: list) -> None:
    """Web-search citations become grounding sources (a quoted figure must appear in its cited text, DECISIONS #101) and
    a sources card for the user."""
    cites = []
    for b in content:
        for c in (b.get("citations") or []) if b.get("type") == "text" else []:
            if c.get("type") == "web_search_result_location":
                cites.append({"url": c.get("url"), "title": c.get("title"), "cited_text": c.get("cited_text")})
    if cites:
        s.turn_results.append({"web_citations": cites})
        every = [c for r in list(s.turn_results) if "web_citations" in r for c in r["web_citations"]]   # the turn so far
        s.send("card", {"type": "sources", "payload": {"items": [{"url": c["url"], "title": c["title"]} for c in every]}})


def _step_loop(llm, s: Session, role: str, messages: list) -> str:
    """Tool loop. Experts are capped at config.EXPERT_MAX_ROUNDS tool rounds and EXPERT_TIMEOUT_S seconds
    (SPEC §11b, C14): past either, they return a partial answer instead of stalling."""
    cfg = ROLES[role]
    strikes: dict = {}
    expert = role != "router" and not role.startswith("train")
    rounds = cfg.get("max_rounds", config.EXPERT_MAX_ROUNDS) if expert else MAX_STEPS
    t0 = time.time()
    for i in range(rounds + (1 if expert else 0)):  # experts: N tool rounds + one final answer
        timed_out = expert and time.time() - t0 > config.EXPERT_TIMEOUT_S
        if expert and hasattr(llm, "timeout_check"):
            timed_out = llm.timeout_check(role, timed_out)
        if timed_out:
            return _final_answer(llm, s, role, messages, f"time's up after {config.EXPERT_TIMEOUT_S:.0f}s")
        t_call = time.perf_counter()
        resp = llm.create(model=cfg["model"], max_tokens=cfg["max_tokens"], system=system_blocks(role),
                          tools=tool_defs(role), messages=_cached(messages) if expert else messages,
                          output_config={"effort": cfg["effort"]})
        s.timings.append({"kind": "llm", "name": role, "start": t_call - s.turn_t0, "s": time.perf_counter() - t_call})
        messages.append({"role": "assistant", "content": resp["content"]})
        _web_citations(s, resp["content"])
        if resp.get("stop_reason") == "refusal":
            return "Sorry, I can't help with that one."
        if resp.get("stop_reason") == "pause_turn":   # a long server-tool turn (web search): resend to let it continue
            continue
        has_calls = any(b.get("type") == "tool_use" for b in resp["content"])
        if resp.get("stop_reason") != "tool_use" or not has_calls:  # seen live: stop_reason tool_use with empty content
            text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text").strip()
            if not text and expert:
                return _final_answer(llm, s, role, messages, "empty response")
            return text
        if expert and i >= rounds:
            pending = [b for b in resp["content"] if b.get("type") == "tool_use"]
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": b["id"], "is_error": True,
                                                          "content": "not run: tool budget used"} for b in pending]})
            return _final_answer(llm, s, role, messages, f"hit the {rounds}-round tool cap")
        results = []
        calls = [b for b in resp["content"] if b.get("type") == "tool_use"]
        # loading a portfolio must happen before anything that analyzes it; identical calls run once
        calls.sort(key=lambda b: b["name"] != "load_portfolio")
        done: dict[str, tuple] = {}
        consults = [b for b in calls if b["name"] == "consult"]
        if len(consults) > 1:   # several lanes asked at once: in parallel (DECISIONS #112)
            with ThreadPoolExecutor(max_workers=len(consults)) as ex:
                futs = {json.dumps([b["name"], b.get("input") or {}], sort_keys=True, default=str):
                        ex.submit(_execute, llm, s, b["name"], b.get("input") or {}, strikes, role) for b in consults}
                done.update({k: f.result() for k, f in futs.items()})
        for b in calls:
            sig = json.dumps([b["name"], b.get("input") or {}], sort_keys=True, default=str)
            if sig not in done:
                done[sig] = _execute(llm, s, b["name"], b.get("input") or {}, strikes, role)
            content, err = done[sig]
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": content, "is_error": err})
        messages.append({"role": "user", "content": results})  # all results in one message
    return "I ran out of steps on that one. Try asking a narrower question?"


def _final_answer(llm, s: Session, role: str, messages: list, why: str) -> str:
    """Past the round/time cap an expert still answers, from what it already has: one last call with
    tools disabled (tool_choice none), instead of returning nothing (SPEC §11b rule 3)."""
    cfg = ROLES[role]
    messages.append({"role": "user", "content": f"[system: {why}] Answer now from the tool results you already have. "
                     "No more tools. Mark anything you couldn't check as unchecked."})
    t_call = time.perf_counter()
    resp = llm.create(model=cfg["model"], max_tokens=cfg["max_tokens"], system=system_blocks(role), tools=tool_defs(role),
                      tool_choice={"type": "none"}, messages=messages, output_config={"effort": cfg["effort"]})
    s.timings.append({"kind": "llm", "name": role, "start": t_call - s.turn_t0, "s": time.perf_counter() - t_call})
    s.log.append({"tool": f"{role}:capped", "input": {"why": why}, "ok": True})
    _web_citations(s, resp["content"])
    return "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text").strip()


def _said(s: Session) -> str:
    return " ".join(m["content"] for m in s.history if m["role"] == "user" and isinstance(m["content"], str))


def _sources(s: Session) -> list:
    return s.__dict__.setdefault("_all_results", []) + list(s.turn_results)


def _checked(llm, s: Session, role: str, messages: list, text: str) -> tuple[str, dict]:
    """The reply checks (grounding: every number from a tool result or a cited source; format: short, no tables), with
    ONE rewrite by whoever wrote it, then fail closed (strip unsupported claims, enforce the shape)."""
    sources, said = _sources(s), _said(s)
    g, f = grounding.check(text, sources, said), grounding.format_check(text)
    if g["ok"] and f["ok"]:
        return text, dict(g, format=f)
    why = []
    if not g["ok"]:
        why.append("These numbers aren't in any tool result or cited source: " + ", ".join(g["ungrounded"]) + ". Use only "
                   "numbers that appear in tool results or, for company facts, in a cited source: no ratios, multiples "
                   "('9x', 'twice') or percentages of your own; drop a figure rather than compute it.")
    if not f["ok"]:
        why.append("Too long for the chat (" + "; ".join(f["problems"]) + "): the card already shows the numbers. Write a "
                   "bold one-line headline and at most 3 short lines (the takeaway and one question). No tables, no bullets, "
                   "under 600 characters in all.")
    messages.append({"role": "user", "content": "[system: reply check] " + " ".join(why) + " Rewrite your previous reply "
                     "for the user. Output only the rewritten reply. Never mention this check."})
    text = _step_loop(llm, s, role, messages)
    sources = _sources(s)
    g, f = grounding.check(text, sources, said), grounding.format_check(text)
    if not f["ok"]:  # fail closed on shape: no tables, at most MAX_LINES lines
        text = grounding.enforce_format(text)
        f = dict(grounding.format_check(text), enforced=True)
        g = grounding.check(text, sources, said)
    if not g["ok"]:  # fail closed: strip the claims that still aren't supported
        text = grounding.strip(text, g["ungrounded"])
        g = {"ok": grounding.check(text, sources, said)["ok"], "ungrounded": [], "stripped": g["ungrounded"]}
    return text, dict(g, format=f)


def _cached(messages: list) -> list:
    """The API copy of a transcript with a cache breakpoint on its last block, so a warm session's growing history bills
    as cache reads (~0.1x input). The stored transcript itself is never modified."""
    if not messages:
        return messages
    last = dict(messages[-1])
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) for b in content]
    if blocks:
        blocks[-1] = dict(blocks[-1], cache_control={"type": "ephemeral"})
    last["content"] = blocks
    return messages[:-1] + [last]


def _alternate(messages: list) -> list:
    """A transcript ready for the next user message: it must end with the assistant."""
    if messages and messages[-1]["role"] == "user":
        messages.append({"role": "assistant", "content": "(no answer)"})
    return messages


HANDOFF_ASK = ("[system: this conversation is going cold] Write your handoff note for the next time this member asks you "
               "something, as JSON only: {\"member_goal\": str, \"current_view_or_event\": str, \"what_you_recommended\": str, "
               "\"open_questions\": [str], \"rerun\": [tool calls to repeat for fresh numbers]}. No numbers: the next "
               "session re-runs the tools.")


def _handoff(llm, s: Session, role: str, messages: list) -> dict:
    """The expert writes its own handoff note: it knows what mattered (DECISIONS #108)."""
    cfg = ROLES[role]
    msgs = _alternate(list(messages)) + [{"role": "user", "content": HANDOFF_ASK}]
    t_call = time.perf_counter()
    resp = llm.create(model=cfg["model"], max_tokens=2000, system=system_blocks(role), tools=tool_defs(role),
                      tool_choice={"type": "none"}, messages=_cached(msgs), output_config={"effort": "low"})
    s.timings.append({"kind": "llm", "name": role, "start": t_call - s.turn_t0, "s": time.perf_counter() - t_call})
    text = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    import re as _re
    m = _re.search(r"\{.*\}", text, _re.S)
    try:
        return json.loads(m.group(0)) if m else {"summary": text.strip()}
    except ValueError:
        return {"summary": text.strip()}


def _warm(llm, s: Session, role: str) -> dict:
    """The (expert, member) session, made cold (handoff note, fresh transcript) if idle past the TTL or too long."""
    key = f"{role}|{s.member}"
    ws = s.expert_sessions.get(key)
    if ws and ws["messages"]:
        cold = time.time() - ws["last"] > config.WARM_TTL_S or len(ws["messages"]) > config.WARM_MAX_MESSAGES
        if hasattr(llm, "timeout_check"):     # wall-clock decision recorded, so replays take the same path
            cold = llm.timeout_check(f"{role}:cold", cold)
        if cold:
            ws = {"messages": [], "last": time.time(), "handoff": _handoff(llm, s, role, ws["messages"])}
            s.routing.append({"entry": "handoff", "plan": [{"expert": role, "run": True, "why": "session went cold"}]})
    ws = ws or {"messages": [], "last": time.time(), "handoff": None}
    s.expert_sessions[key] = ws
    return ws


def _session_rules(s: Session, role: str) -> list[str]:
    """In-chat train mode: the trainer's /fix rules, active for this session (they outrank your defaults here)."""
    from .train_fix import session_rules
    return session_rules(s, role)


def _digest_text() -> str | None:
    """This week's shared digest for every expert (DECISIONS #115); live chat only, so replays stay identical."""
    if os.environ.get("WORSEY_LIVE_QUOTES") != "1":
        return None
    try:
        from .. import markets
        d = markets.digest()
        return f"{d['text']} (as of {d['as_of']})" if d and d.get("age_days", 99) <= 8 else None
    except Exception:  # noqa: BLE001
        return None


def run_expert(llm, s: Session, role: str, request: str, mode: str = "reply") -> dict:
    """An expert call. mode="reply": writes the user-facing reply (checked here); "consult": answers another expert;
    both continue the member's warm session for this expert (DECISIONS #108). mode="notes": stateless cited notes.
    A cold start gets the session slice it needs (and its own handoff note, if it wrote one)."""
    ctx = {"portfolio": s.portfolio.name if s.portfolio else None,
           "positions": s.priced["value"].round(0).to_dict() if s.priced is not None else None,
           "active_event": s.active_event, "view": s.view.to_dict() if s.view is not None else None,
           "scanned_events": [e["event_id"] for e in s.last_scan["events"]] if s.last_scan else None,
           "conversation": [m["content"] for m in s.history if m["role"] == "user" and isinstance(m["content"], str)][-4:],
           **({"member_profile": s.__dict__["profile_view"]} if s.__dict__.get("profile_view") else {}),
           "lingo": s.__dict__.get("lingo", "mixed"),
           **({"this_week_on_the_tape": _digest_text()} if _digest_text() else {}),
           **({"darin_corrections_this_session": _rules} if (_rules := _session_rules(s, role)) else {})}
    how = {"reply": "You write the reply the user sees in the group chat.",
           "consult": "Another expert is asking you. Answer for them, not the user: at most 6 short lines, cite company facts.",
           "notes": "You write notes for another expert, not for the user."}[mode]
    if mode == "notes":
        msgs = [{"role": "user", "content": f"{how}\n\nSession: {json.dumps(ctx, default=str)}\n\nRequest: {request}"}]
        return {"expert": role, "text": _step_loop(llm, s, role, msgs)}
    ws = _warm(llm, s, role)
    if ws["messages"]:   # warm: one more message in the same conversation
        msgs = _alternate(ws["messages"])
        msgs.append({"role": "user", "content": f"{how}\n\nSession now: {json.dumps(ctx, default=str)}\n\nRequest: {request}"})
    else:
        note = f"\n\nYour handoff note from earlier: {json.dumps(ws['handoff'])}" if ws.get("handoff") else ""
        msgs = [{"role": "user", "content": f"{how}\n\nSession: {json.dumps(ctx, default=str)}{note}\n\nRequest: {request}"}]
    text = _step_loop(llm, s, role, msgs)
    out = {"expert": role, "text": text}
    if mode == "reply":
        out["text"], out["grounding"] = _checked(llm, s, role, msgs, text)
    s.expert_sessions[f"{role}|{s.member}"] = {"messages": msgs, "last": time.time(), "handoff": None}
    return out


def compose(replies: list[dict]) -> tuple[str, dict]:
    """The turn's reply from expert-written parts: one part as is; a hedge turn's two options, each labelled."""
    if len(replies) == 1:
        r = replies[0]
        return r["text"], r["grounding"]
    parts = [f"*{r.get('label', r['expert']).capitalize()}*\n{r['text']}" for r in replies]
    gs = [r["grounding"] for r in replies]
    fmt = {"ok": all(g["format"]["ok"] for g in gs), "problems": [p for g in gs for p in g["format"]["problems"]],
           **({"enforced": True} if any(g["format"].get("enforced") for g in gs) else {})}
    g = {"ok": all(g["ok"] for g in gs), "ungrounded": [u for g in gs for u in g.get("ungrounded", [])],
         **({"stripped": [x for g in gs for x in g.get("stripped", [])]} if any(g.get("stripped") for g in gs) else {}),
         "format": fmt}
    return "\n\n".join(parts), g


def latency(s: Session, total: float) -> dict:
    """Budget view of one turn: first card = when the first code result (table/card) was ready."""
    cards = [t["start"] + t["s"] for t in s.timings if t["kind"] == "card"]
    llm = [t for t in s.timings if t["kind"] == "llm"]
    return {"first_card_s": round(min(cards), 2) if cards else None, "total_s": round(total, 2),
            "first_router_call_s": round(llm[0]["s"], 2) if llm else None,
            "router_llm_s": round(sum(t["s"] for t in llm if t["name"] == "router"), 2),
            "expert_llm_s": round(sum(t["s"] for t in llm if t["name"] != "router"), 2),
            "llm_calls": len(llm)}


def run_turn(llm, s: Session, user_text: str) -> dict:
    """One user turn through the router. Returns {reply, grounding, tools, latency}."""
    s.turn_results, start = [], len(s.log)
    s.turn_replies, s.replies_fresh = [], False
    s.timings, s.turn_t0 = [], time.perf_counter()
    getattr(llm, "start_turn", lambda: None)()   # live client: per-turn call limit (agent/budget.py)
    note = ""
    if config.SCENARIO_ENGINE == "planb" and s.portfolio is not None and s.priced is not None:
        # the router infers state from the conversation; a book loaded outside it (the web app's demo buttons, scripts)
        # was sometimes missed (seen Oct 23: "I need your holdings first"). Plan B only, so v1 replays stay identical.
        opts = s.__dict__.get("options") or []
        note = (f"\n\n[session: a portfolio is loaded ({', '.join(s.priced.index)}"
                + (f" + {len(opts)} option position{'s' if len(opts) > 1 else ''}" if opts else "") + "); use it]")
    s.history.append({"role": "user", "content": user_text + note})
    reply = _step_loop(llm, s, "router", s.history)
    # the expert-written reply stands whenever the turn's last tool produced it (code-enforced: the router sometimes
    # "helpfully" re-summarized both hedge options into one, seen live Oct 17); a later tool or error falls back to the router
    if s.turn_replies and (s.replies_fresh or reply.strip().startswith(EXPERT_DONE)):
        reply, g = compose(s.turn_replies)
    else:
        reply, g = _checked(llm, s, "router", s.history, reply)
    s.__dict__.setdefault("_all_results", []).extend(s.turn_results)
    if s.__dict__.get("remember"):   # live chat: code writes the episodic log (DECISIONS #114)
        try:
            from . import memory
            memory.record_turn(s, user_text.rsplit("to Worsey]:", 1)[-1])
        except Exception:  # noqa: BLE001 — memory must never break a turn
            pass
    return {"reply": reply, "grounding": g, "tools": [x["tool"] for x in s.log[start:]],
            "latency": latency(s, time.perf_counter() - s.turn_t0)}
