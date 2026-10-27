"""Agent loop, tools, validation and grounding, offline (scripted fake LLM, synthetic data)."""

import json
from pathlib import Path

import pytest

from worsey.agent import grounding, tools
from worsey.agent.llm import FakeLLM, RecordingLLM, ReplayLLM, response, text_block, tool_use
from worsey.agent.loop import run_turn
from worsey.agent.session import Session, ToolError
from worsey.agent.validate import validate
from worsey.data.provider import SyntheticMarketData

FIX = Path(__file__).parent / "fixtures" / "strategies"


def sess():
    return Session(SyntheticMarketData())


def test_grounding_accepts_formatted_tool_numbers_and_flags_invented_ones():
    res = [{"pnl_yes": -4132.7, "p": 0.352, "cost": 812.0}]
    ok = grounding.check("Tariffs YES: -$4.1k. Market says 35%. Hedge costs $812. Here are 3 ideas.", res)
    assert ok["ok"], ok
    bad = grounding.check("That would cost you about $9.5k", res)
    assert not bad["ok"] and bad["ungrounded"] == ["$9.5k"]
    assert grounding.check("make it 2 weeks instead of 6", [], user_text="")["ok"]  # small counts exempt
    assert grounding.check("you said 25%", [], user_text="tariffs of 25% by March")["ok"]  # user's own number


def test_validator_strict_schema():
    schema = tools.TOOLS["set_view"][2]
    assert validate({"direction": "up", "target_move_pct": 8}, schema) == []
    errs = validate({"direction": "sideways", "foo": 1}, schema)
    assert any("not one of" in e for e in errs) and any("unexpected field 'foo'" in e for e in errs)
    assert validate({"horizon_days": True}, schema)  # bool is not an integer


def test_plain_tools_on_synthetic_world():
    s = sess()
    with pytest.raises(ToolError):
        tools.scan_events(s)
    lp = tools.load_portfolio(s, name="balanced")
    assert lp["synthetic_data"] and abs(lp["total_value"] - 100_000) < 1
    ex = tools.exposures(s)
    nvda = next(r for r in ex["top_names"] if r["name"] == "NVDA")
    assert nvda["weight"] > nvda["direct_weight"]  # look-through adds NVDA via SPY/QQQ
    sc = tools.scan_events(s, horizon_days=200)
    assert sc["events"][0]["event_id"] == "TARIFF-25-MAR"
    ee = tools.explain_event(s, "TARIFF-25-MAR")
    assert ee["positions"][0]["pnl_if_yes"] < 0
    h = tools.compare_hedges_tool(s)
    kinds = [r["hedge"] for r in h["rows"]]
    assert kinds[:2] == ["do_nothing", "prediction_market"] and "index_put_spread" in kinds


def test_view_tools_and_update_diff(monkeypatch):
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(FIX))
    s = sess()
    v = tools.set_view(s, underlying="NVDA", direction="up", target_move_pct=8, horizon_days=42)
    assert v["diff"] is None
    p1 = tools.propose_strategies(s)
    assert p1["candidates"] and p1["changes_vs_last"] is None
    v2 = tools.set_view(s, horizon_days=14)
    assert set(v2["diff"]) == {"horizon"}
    p2 = tools.propose_strategies(s)
    assert p2["changes_vs_last"] is not None
    flipped = tools.set_view(s, direction="down")  # direction flips without a size → 1σ default, said out loud
    assert flipped["view"]["target_move_pct"] < 0 and "assumed" in flipped
    with pytest.raises(ToolError):
        tools.set_view(s, direction="up", target_move_pct=-5)  # contradictory size still rejected


def _role(kw) -> str:
    text = kw["system"][0]["text"]
    if "Worsey's front desk" in text:
        return "router"
    return next(r for r in ("event_analyst", "options_desk", "equity_analyst") if f"You are {r}" in text)


def _req(kw) -> str:
    """The latest user message's text, whether plain or block-list (warm transcripts carry a cache marker)."""
    m = next(m for m in reversed(kw["messages"]) if m["role"] == "user" and not (
        isinstance(m["content"], list) and m["content"] and m["content"][0].get("type") == "tool_result"))
    c = m["content"]
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c)


def _script(seen=None):
    """Router: load + scan_risks → [expert]. Inside scan_risks: equity_analyst notes (NVDA ≥ 5%), then event_analyst
    writes the reply, citing the top row's σ from the scan it was given."""
    state = {"router": 0}

    def f(kw):
        role = _role(kw)
        if seen is not None:
            seen.append(role)
        if role == "router":
            state["router"] += 1
            if state["router"] == 1:
                return response(tool_use("t1", "load_portfolio", {"name": "balanced"}))
            if state["router"] == 2:
                return response(tool_use("t2", "scan_risks", {"horizon_days": 200, "n": 3}))
            return response(text_block("[expert]"))
        if role == "equity_analyst":
            return response(text_block("NVDA: data-center revenue is the exposure (inference from principles)."))
        req = _req(kw)
        scan = json.loads(req.split("Scan: ", 1)[1].split("\nEquity analyst notes:")[0])
        top = scan["events"][0]
        return response(text_block(f"**Biggest swing: {top['event_id']}.** σ ${top['sigma']:,.0f}. Test numbers, not live."))
    return f


def test_full_turn_routes_through_entry_tool_and_passes_grounding():
    s = sess()
    seen = []
    out = run_turn(FakeLLM(_script(seen)), s, "what could hit my portfolio?")
    assert out["tools"] == ["load_portfolio", "scan_risks"]
    assert out["reply"].startswith("**Biggest swing: TARIFF-25-MAR") and out["grounding"]["ok"], out  # the expert's words
    plan = {p["expert"]: p for p in s.routing[-1]["plan"]}
    assert s.routing[-1]["entry"] == "scan_risks"
    assert plan["event_analyst"]["run"] and not plan["natural_world_analyst"]["run"]  # balanced: no weather exposure
    assert plan["equity_analyst"]["run"] and "NVDA" in plan["equity_analyst"]["names"]  # Tier A since Oct 17
    assert seen.index("equity_analyst") < seen.index("event_analyst")  # its notes feed the writer
    assert s.history[0]["role"] == "user" and s.history[-1]["role"] == "assistant"


def test_coverage_rules_are_code_and_weather_triggers():
    from worsey.agent.loop import coverage
    s = sess()
    s.portfolio = None
    import pandas as pd
    s.priced = pd.DataFrame({"value": [60_000.0, 40_000.0], "weight": [0.6, 0.4], "asset_class": ["commodities", "equity"]},
                            index=["UNG", "VST"])
    plan = {p["expert"]: p for p in coverage(s, "scan")}
    assert plan["natural_world_analyst"]["run"] and "100%" in plan["natural_world_analyst"]["why"]   # built Oct 18
    ev = coverage(s, "event", {"category": "Climate and Weather"})
    assert ev[0]["expert"] == "natural_world_analyst" and ev[0]["run"]  # weather events: natural_world_analyst writes


def test_analyze_event_entry_tool():
    from worsey.agent.loop import analyze_event
    s = sess()
    tools.load_portfolio(s, name="balanced")
    tools.scan_events(s, horizon_days=200)
    out = analyze_event(FakeLLM(lambda kw: response(text_block("**QQQ and NVDA drive it.**"))), s, "TARIFF-25-MAR")
    detail = next(r["analyze_event.detail"] for r in s.turn_results if "analyze_event.detail" in r)
    assert detail["positions"] and out["expert_reply"].startswith("**QQQ") and s.turn_replies[0]["expert"] == "event_analyst"
    assert s.routing[-1]["entry"] == "analyze_event"


def test_expert_round_cap_and_timeout_still_answer(monkeypatch):
    from worsey import config
    from worsey.agent.loop import run_expert
    s = sess()
    tools.load_portfolio(s, name="balanced")
    tools.scan_events(s, horizon_days=200)

    def keep_digging(kw):  # always wants another tool, unless tools are switched off
        if kw.get("tool_choice") == {"type": "none"}:
            return response(text_block("from what I have"))
        return response(tool_use("x", "explain_event", {"event_id": "TARIFF-25-MAR"}))
    out = run_expert(FakeLLM(keep_digging), s, "event_analyst", "keep digging", mode="notes")
    assert out["text"] == "from what I have" and s.log[-1]["tool"] == "event_analyst:capped"
    assert sum(1 for x in s.log if x["tool"] == "explain_event") == config.EXPERT_MAX_ROUNDS  # the next call was refused
    monkeypatch.setattr(config, "EXPERT_TIMEOUT_S", -1.0)
    out2 = run_expert(FakeLLM(keep_digging), s, "event_analyst", "keep digging", mode="notes")
    assert out2["text"] == "from what I have" and "time's up" in s.log[-1]["input"]["why"]


def test_directional_view_without_size_gets_a_1sigma_default():
    s = sess()
    v = tools.set_view(s, underlying="SPY", direction="down", horizon_days=90)
    assert v["assumed"]["target_move_pct"] < 0 and "1σ" in v["assumed"]["why"]


def test_validation_error_goes_back_to_model():
    calls = []

    def f(kw):
        calls.append(kw)
        if len(calls) == 1:
            return response(tool_use("t1", "load_portfolio", {"name": "nope"}))
        r = kw["messages"][-1]["content"][0]
        assert r["is_error"] and "not one of" in r["content"]
        return response(text_block("Couldn't load that one."))
    out = run_turn(FakeLLM(f), sess(), "load nope")
    assert out["reply"] == "Couldn't load that one."


def test_grounding_fails_closed_after_one_rewrite():
    replies = [response(text_block("You'd lose $9.9k. That's 9x worse.")), response(text_block("Still $9.9k. Ask me more."))]
    out = run_turn(FakeLLM(replies), sess(), "how bad?")
    assert "$9.9k" not in out["reply"] and out["reply"] == "Ask me more." and out["grounding"]["stripped"]


def test_grounding_retry_rewrites_invented_number():
    replies = [response(text_block("You'd lose $9.9k.")), response(text_block("Can't size that yet: no scan."))]
    out = run_turn(FakeLLM(replies), sess(), "how bad?")
    assert out["reply"].startswith("Can't size") and out["grounding"]["ok"]


def test_record_then_replay(tmp_path):
    rec = tmp_path / "r.jsonl"
    run_turn(RecordingLLM(FakeLLM(_script()), rec), sess(), "what could hit my portfolio?")
    out = run_turn(ReplayLLM(rec), sess(), "what could hit my portfolio?")
    assert "TARIFF-25-MAR" in out["reply"]


def test_analyze_event_by_query_finds_markets_outside_the_top_rows():
    from worsey.agent.loop import analyze_event
    s = sess()
    tools.load_portfolio(s, name="balanced")
    tools.scan_events(s, horizon_days=200, n=1)  # only the top row is in the scan
    assert [e["event_id"] for e in s.last_scan["events"]] == ["TARIFF-25-MAR"]
    out = analyze_event(FakeLLM(lambda kw: response(text_block("**Fed market, fits.**"))), s, query="Fed cuts rates December")
    detail = next(r["analyze_event.detail"] for r in s.turn_results if "analyze_event.detail" in r)
    assert out["event"] == "FED-CUT-DEC" and detail["event"]["event_id"] == "FED-CUT-DEC"
    assert detail["match"]["candidates"][0]["event_id"] == "FED-CUT-DEC"
    oil = tools.find_event(s, "WTI oil above $100")["candidates"][0]
    assert not oil["priced"] and "price-level" in oil["why_not"]  # reflexive: excluded, and says why


def test_load_runs_first_and_duplicate_calls_run_once():
    calls = []

    def f(kw):
        calls.append(kw)
        if len(calls) == 1:  # model emits the analysis before the load, and twice
            return response(tool_use("a", "exposures", {}), tool_use("b", "load_portfolio", {"name": "balanced"}),
                            tool_use("c", "exposures", {}))
        res = {r["tool_use_id"]: r for r in kw["messages"][-1]["content"]}
        assert not res["a"]["is_error"] and res["a"]["content"] == res["c"]["content"]
        return response(text_block("done"))
    s = sess()
    out = run_turn(FakeLLM(f), s, "load balanced and show exposures")
    assert out["tools"] == ["load_portfolio", "exposures"]  # exposures ran once, after the load


def test_new_view_without_horizon_is_a_tool_error_not_a_crash():
    s = sess()
    with pytest.raises(ToolError, match="Ask the user ONE"):
        tools.set_view(s, underlying="NVDA", direction="up")
    with pytest.raises(ToolError, match="missing direction"):
        tools.set_view(s, underlying="NVDA", horizon_days=30)


def test_internal_tool_errors_never_crash_the_turn(monkeypatch):
    def boom(s, **kw):
        raise ZeroDivisionError("x")
    monkeypatch.setitem(tools.TOOLS, "exposures", (boom, tools.TOOLS["exposures"][1], tools.TOOLS["exposures"][2]))
    script = [response(tool_use("t", "exposures", {})), response(text_block("Couldn't compute that."))]
    out = run_turn(FakeLLM(script), sess(), "exposures?")
    assert out["reply"] == "Couldn't compute that."


def test_empty_tool_use_response_is_final_not_an_empty_tool_round():
    from worsey.agent.loop import run_expert
    s = sess()
    script = [{"role": "assistant", "content": [], "stop_reason": "tool_use"},  # seen live from the API
              response(text_block("recovered"))]
    fake = FakeLLM(script)
    out = run_expert(fake, s, "options_desk", "anything", mode="notes")
    assert out["text"] == "recovered" and fake.calls[-1].get("tool_choice") == {"type": "none"}


def test_weak_match_falls_back_to_unpriced_template(monkeypatch):
    from worsey import config
    from worsey.agent.loop import analyze_event
    s = sess()
    tools.load_portfolio(s, name="balanced")
    tools.scan_events(s, horizon_days=200)
    days = s.data.closes(["SPY"]).index
    monkeypatch.setitem(config.TEMPLATE_ANALOGS, "trade_policy",
                        [{"date": str(days[100].date()), "what": "analog A", "after": 1},
                         {"date": str(days[300].date()), "what": "analog B", "after": 2}])
    out = analyze_event(FakeLLM(lambda kw: response(text_block("**No strong tariff market; past episodes instead.**"))), s,
                        query="what would a big tariff escalation do to me")
    detail = next(r["analyze_event.detail"] for r in s.turn_results if "analyze_event.detail" in r)
    out["detail"] = detail
    t = out["detail"]["template"]
    assert out["template"] and out["detail"]["no_strong_market"] and "unpriced" in t["status"] and len(t["analogs"]) == 2
    assert t["range"][0] <= t["pnl_if_it_happens"] <= t["range"][1]
    assert s.active_event is None
    h = tools.compare_hedges_tool(s, out["detail"]["match"]["candidates"][0]["event_id"] if out["detail"]["match"]["candidates"] else None)
    assert h["unpriced"] and h["event_id"] == "trade_policy"  # "how would I hedge that?" hedges the template scenario
    if h["rows"][1:]:  # stress case: same hedges against the worst analog, which loses at least as much as the mean
        assert h["stress"]["loss_if_it_happens"] <= t["pnl_if_it_happens"] + 1e-6


def test_scan_risks_passes_build_notes_to_the_expert_as_input():
    # Oct 17 (DECISIONS #98): the build's notes are the expert's input, no longer a substitute for it
    from worsey.agent.loop import scan_risks
    s = sess()
    tools.load_portfolio(s, name="balanced")
    ids = [m["id"] for m in s.data.event_markets()]
    s.data.structure = lambda: {"notes": {i: {"what": "w", "channels": "c", "caveats": "k"} for i in ids}}
    reqs = []

    def f(kw):
        reqs.append((_role(kw), _req(kw)))
        return response(text_block("**Tariffs lead.**"))
    out = scan_risks(FakeLLM(f), s, horizon_days=200, n=3)
    writer = [r for role, r in reqs if role == "event_analyst"]
    assert writer and '"event_notes"' in writer[0] and out["expert_reply"] == "**Tariffs lead.**"


def test_emit_hook_streams_status_and_cards_from_real_tool_calls():
    events = []
    s = sess()
    s.emit = lambda kind, payload: events.append((kind, payload))
    run_turn(FakeLLM(_script()), s, "what could hit my portfolio?")
    kinds = [k for k, _ in events]
    assert ("status", {"text": "pulling prices…"}) in events and ("status", {"text": "scanning prediction markets…"}) in events
    assert ("status", {"text": "thinking it through…"}) in events
    card = next(p for k, p in events if k == "card")
    assert card["type"] == "scan" and card["payload"]["events"]
    assert kinds.index("card") > kinds.index("status")


def _hedge_script(seen):
    state = {"router": 0}

    def f(kw):
        role = _role(kw)
        seen.append(role)
        if role == "router":
            state["router"] += 1
            if state["router"] == 1:
                return response(tool_use("t1", "load_portfolio", {"name": "balanced"}))
            if state["router"] == 2:
                return response(tool_use("t2", "scan_risks", {"horizon_days": 200, "n": 3}))
            if state["router"] == 3:
                return response(tool_use("t3", "compare_hedges", {"event_id": "TARIFF-25-MAR", "kind": kind[0]}))
            return response(text_block("[expert]"))
        if role == "equity_analyst":
            return response(text_block("notes"))
        if "EVENT-HEDGE" in _req(kw):
            return response(text_block("**Buy the YES contract.** It covers the event itself."))
        if "OPTIONS-HEDGE" in _req(kw):
            return response(text_block("**The index put spread is the options route.** It pays on a selloff, not the event."))
        return response(text_block("**Tariffs lead.**"))
    kind = ["both"]
    return f, kind


def test_hedge_turn_runs_both_experts_and_shows_two_options():
    seen = []
    f, kind = _hedge_script(seen)
    s = sess()
    out = run_turn(FakeLLM(f), s, "what could hit me and how would I hedge it?")
    assert "*Event hedge*" in out["reply"] and "*Options hedge*" in out["reply"]
    assert out["reply"].index("YES contract") < out["reply"].index("index put spread")   # event first, then options
    assert out["grounding"]["ok"] and out["grounding"]["format"]["ok"]
    assert s.routing[-1]["entry"] == "compare_hedges" and {p["expert"] for p in s.routing[-1]["plan"]} == {"event_analyst", "options_desk"}


def test_hedge_turn_runs_one_expert_when_the_user_names_the_kind():
    seen = []
    f, kind = _hedge_script(seen)
    kind[0] = "options"
    s = sess()
    out = run_turn(FakeLLM(f), s, "how would I hedge it with options?")
    assert out["reply"].startswith("**The index put spread") and "*Event hedge*" not in out["reply"]
    assert [p["expert"] for p in s.routing[-1]["plan"]] == ["options_desk"]


def test_router_hands_a_view_to_options_desk_which_writes_the_reply(monkeypatch):
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(FIX))
    state = {"router": 0, "desk": 0}

    def f(kw):
        role = _role(kw)
        if role == "router":
            state["router"] += 1
            if state["router"] == 1:
                return response(tool_use("r1", "options_desk", {"request": "NVDA up 8% in 6 weeks"}))
            return response(text_block("[expert]"))
        state["desk"] += 1
        if state["desk"] == 1:
            return response(tool_use("d1", "view_and_strategies", {"underlying": "NVDA", "direction": "up",
                                                                   "target_move_pct": 8, "horizon_days": 42, "confirmed": True}))
        cands = json.loads(kw["messages"][-1]["content"][0]["content"])["candidates"]
        return response(text_block(f"**The {cands[0]['id'].replace('_', ' ')} fits.** It costs ${cands[0]['cost']:,.0f}."))
    s = sess()
    out = run_turn(FakeLLM(f), s, "NVDA rips 8% in 6 weeks")
    assert out["reply"].startswith("**The ") and out["grounding"]["ok"] and s.view is not None
    assert "view_and_strategies" not in [t["name"] for t in __import__("worsey.agent.loop", fromlist=["tool_defs"]).tool_defs("router")]


def test_an_experts_own_reply_is_checked_and_rewritten_by_that_expert():
    from worsey.agent.loop import run_expert
    s = sess()
    tools.load_portfolio(s, name="balanced")
    tools.scan_events(s, horizon_days=200)
    calls = []

    def f(kw):
        calls.append((str(kw["messages"][-1]["content"]), kw["system"][0]["text"]))   # snapshot: the list keeps growing
        if len(calls) == 1:
            return response(text_block("**Bad.** You'd lose $9.9k."))   # invented number
        return response(text_block("**Tariffs lead the scan.**"))
    out = run_expert(FakeLLM(f), s, "event_analyst", "explain", mode="reply")
    assert out["text"] == "**Tariffs lead the scan.**" and out["grounding"]["ok"]
    assert "reply check" in calls[1][0] and "You are event_analyst" in calls[1][1]


def test_a_failed_last_tool_shows_the_routers_explanation_not_a_stale_expert_reply():
    state = {"router": 0}

    def f(kw):
        if _role(kw) != "router":
            return response(text_block("**Tariffs lead.**"))
        state["router"] += 1
        if state["router"] == 1:
            return response(tool_use("t1", "load_portfolio", {"name": "balanced"}))
        if state["router"] == 2:
            return response(tool_use("t2", "scan_risks", {"horizon_days": 200, "n": 3}))
        if state["router"] == 3:
            return response(tool_use("t3", "compare_hedges", {"event_id": "NOPE"}))
        return response(text_block("**I couldn't price hedges for that event.**"))
    out = run_turn(FakeLLM(f), sess(), "scan and hedge")
    assert out["reply"].startswith("**I couldn't price")


def test_router_cannot_rewrite_the_experts_hedge_options():
    seen = []
    f0, kind = _hedge_script(seen)

    def f(kw):
        r = f0(kw)
        if _role(kw) == "router" and r["content"][0].get("text") == "[expert]":
            return response(text_block("**My own merged summary.**"))   # the router ignores the [expert] rule
        return r
    out = run_turn(FakeLLM(f), sess(), "what could hit me and how would I hedge it?")
    assert "*Event hedge*" in out["reply"] and "*Options hedge*" in out["reply"] and "merged" not in out["reply"]


def test_live_quote_is_offered_only_in_live_chat(monkeypatch):
    from worsey.agent.loop import tool_defs
    monkeypatch.delenv("WORSEY_LIVE_QUOTES", raising=False)
    assert "live_quote" not in [t["name"] for t in tool_defs("router")]      # replays keep their recorded tool lists
    monkeypatch.setenv("WORSEY_LIVE_QUOTES", "1")
    for role in ("router", "options_desk", "equity_analyst"):
        assert "live_quote" in [t.get("name") for t in tool_defs(role)]
    with pytest.raises(ToolError, match="needs expiry"):
        tools.live_quote(sess(), "NVDA", strike=200.0)


def test_live_calls_go_online_even_when_market_data_is_offline(tmp_path, monkeypatch):
    from worsey.data import cache
    monkeypatch.setenv("WORSEY_OFFLINE", "1")
    calls = []
    with pytest.raises(cache.OfflineCacheMiss):
        cache.cached_call("x", "/p", {"a": 1}, lambda: calls.append(1), root=tmp_path)
    assert cache.cached_call("x", "/p", {"a": 1}, lambda: calls.append(1) or {"ok": 1}, root=tmp_path, force_network=True) == {"ok": 1}
    assert calls == [1]


def test_name_exposure_finds_small_names_and_bonds_through_funds():
    import pandas as pd
    s = sess()
    tools.load_portfolio(s, name="balanced")
    real = s.data.holdings

    def holdings(etf, sectors=True):   # a small stock line in QQQ and two bond lines in HYG, beyond any top-N
        if etf == "QQQ":
            return pd.DataFrame({"ticker": ["NVDA", "CRWV"], "weight": [0.08, 0.00128], "name": ["NVIDIA CORP", "COREWEAVE INC"],
                                 "asset_class": ["Equity", "Equity"], "effective_date": ["2026-10-14"] * 2})
        if etf == "HYG":
            return pd.DataFrame({"ticker": ["CRWV 9.75 10/01/31 144A", "CRWV 9 02/01/31 144A"], "weight": [0.0019, 0.0012],
                                 "name": ["COREWEAVE 9.75% 10/01/31", "COREWEAVE INC 9 2/1/2031"],
                                 "asset_class": ["Fixed Income"] * 2, "effective_date": ["2026-10-14"] * 2})
        return real(etf, sectors=sectors)
    s.data.holdings = holdings
    r = tools.exposures(s, names=["CoreWeave"])["names"][0]
    qqq, hyg = float(s.priced.at["QQQ", "value"]), float(s.priced.at["HYG", "value"])
    assert r["stock_usd"] == pytest.approx(qqq * 0.00128) and r["debt_usd"] == pytest.approx(hyg * 0.0031)
    assert {p["via"] for p in r["through"]} == {"QQQ", "HYG"} and r["holdings_as_of"] == "2026-10-14"
    assert tools.exposures(s, names=["CRWV"])["names"][0]["total_usd"] == pytest.approx(r["total_usd"])   # ticker works too
    none = tools.exposures(s, names=["Zzzz Corp"])["names"][0]
    assert none["total_usd"] == 0 and "not held" in none["note"]


def test_warm_sessions_continue_per_member_and_go_cold_with_a_handoff(monkeypatch):
    from worsey import config
    from worsey.agent.loop import run_expert
    s = sess()
    tools.load_portfolio(s, name="balanced")
    seen = []

    def f(kw):
        seen.append((len(kw["messages"]), _req(kw), kw.get("tool_choice")))
        if kw.get("tool_choice") == {"type": "none"}:
            return response(text_block('{"member_goal": "play NVDA up", "open_questions": ["size?"]}'))
        return response(text_block("**Noted.**"))
    llm = FakeLLM(f)
    run_expert(llm, s, "options_desk", "NVDA up 10% in 6 weeks")
    run_expert(llm, s, "options_desk", "make it 2 weeks")
    assert seen[1][0] == 3 and "make it 2 weeks" in seen[1][1]          # same conversation: request, answer, follow-up
    s.member = "sam"
    run_expert(llm, s, "options_desk", "SPY down 5%")
    assert seen[2][0] == 1                                              # another member: their own session
    s.member = "me"
    monkeypatch.setattr(config, "WARM_TTL_S", -1.0)
    run_expert(llm, s, "options_desk", "what about TSLA?")
    handoff, fresh = seen[3], seen[4]
    assert handoff[2] == {"type": "none"} and "going cold" in handoff[1]  # the expert writes its own handoff note
    assert fresh[0] == 1 and "handoff note" in fresh[1] and "play NVDA up" in fresh[1]
    assert any(r["entry"] == "handoff" for r in s.routing)


def test_consult_is_one_hop_parallel_and_logged():
    from worsey.agent.loop import run_expert
    s = sess()
    tools.load_portfolio(s, name="balanced")
    asked = []

    def f(kw):
        role, req = _role(kw), _req(kw)
        if role == "options_desk":
            if "Another expert" in req or kw["messages"][-1]["role"] == "user" and isinstance(kw["messages"][-1]["content"], list) \
                    and kw["messages"][-1]["content"][0].get("type") == "tool_result":
                return response(text_block("**Calls look fine.**"))
            return response(tool_use("c1", "consult", {"expert": "equity_analyst", "question": "why does NVDA move on earnings?"}),
                            tool_use("c2", "consult", {"expert": "event_analyst", "question": "any macro event before Nov 20?"}))
        asked.append(role)
        if role == "equity_analyst" and "may consult" not in req:   # a consulted expert trying to consult again
            return response(tool_use("c3", "consult", {"expert": "event_analyst", "question": "nested?"})) if len(asked) == 1 \
                else response(text_block("data-center demand (cited)"))
        return response(text_block("CPI on Nov 13"))
    out = run_expert(FakeLLM(f), s, "options_desk", "NVDA up 10%")
    assert out["text"] == "**Calls look fine.**"
    consults = [r for r in s.routing if r["entry"] == "consult"]
    assert {c["plan"][0]["expert"] for c in consults} == {"equity_analyst", "event_analyst"}
    assert "nested?" not in json.dumps([r for r in s.routing if r["entry"] == "consult"])   # one hop: the nested one was refused


def test_a_new_view_needs_the_intake_before_pricing_and_sizing_comes_from_the_budget(monkeypatch):
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(FIX))
    s = sess()
    with pytest.raises(ToolError, match="intake first"):          #: ask before pricing
        tools.view_and_strategies(s, underlying="NVDA", direction="up", target_move_pct=8, horizon_days=42)
    out = tools.view_and_strategies(s, underlying="NVDA", direction="up", target_move_pct=8, horizon_days=42, confirmed=True,
                                    constraints={"risk_budget_usd": 2000})
    for c in out["candidates"]:
        z = c["sized_to_budget"]
        if z["contracts"] is not None:
            assert z["contracts"] == int(2000 // abs(c["max_loss"])) and z["total_max_loss"] == pytest.approx(z["contracts"] * c["max_loss"])
    again = tools.view_and_strategies(s, horizon_days=14)          # a follow-up on the same view: no new intake
    assert again["view_diff"]


def test_price_any_structure_by_delta_and_show_only_a_priced_one(monkeypatch):
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(FIX))
    s = sess()
    tools.view_and_strategies(s, underlying="NVDA", direction="up", target_move_pct=8, horizon_days=42, confirmed=True)
    out = tools.price_structure(s, legs=[{"kind": "call", "qty": 1, "delta": 0.30}, {"kind": "call", "qty": -1, "delta": 0.15}],
                                name="30/15 call spread")["structure"]
    d_long, d_short = (l["delta"] for l in out["legs"])
    assert 0.2 < d_long < 0.4 and 0.08 < d_short < 0.25 and out["legs"][0]["strike"] < out["legs"][1]["strike"]
    assert out["max_loss"] == pytest.approx(-out["cost"]) and "expected_pnl_market" in out
    full = tools.show_structure(s, "30/15 call spread")
    assert full["full"] and full["candidates"][0]["id"] == "30/15 call spread"
    with pytest.raises(ToolError, match="no priced structure"):
        tools.show_structure(s, "iron_butterfly")
    from worsey.agent.loop import _cards
    assert _cards("view_and_strategies", {}, {"candidates": []})[0][1]["compact"]      # summaries while choosing
    assert _cards("show_structure", {}, full)[0][0] == "strategy_full"                 # the diagram only on request
