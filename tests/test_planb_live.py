"""Plan B as Worsey's scenario engine (DECISIONS #179): scan, evidence, pricing, hedges and a full turn, offline
(synthetic prices, scripted LLM, the committed pack)."""

import json

import numpy as np
import pytest

from worsey import config
from worsey.agent import tools
from worsey.agent.llm import FakeLLM, response, text_block, tool_use
from worsey.agent.loop import run_turn, tool_defs
from worsey.agent.session import Session, ToolError
from worsey.data.provider import SyntheticMarketData
from worsey.planb import live

SPEC = {"title": "Oil tankers attacked in the Gulf", "category": "conflict_war", "severity": "major", "region": "Middle East",
        "direction": {"escalation": "escalation", "oil_supply": "disruption", "risk_sentiment": "risk_off"},
        "targeted_sectors": ["energy"]}


@pytest.fixture(autouse=True)
def _planb(monkeypatch):
    monkeypatch.setattr(config, "SCENARIO_ENGINE", "planb")


def book():
    s = Session(SyntheticMarketData())
    tools.load_portfolio(s, name="balanced")
    return s


def test_pack_loads_and_the_model_is_reproducible():
    p = live.pack()
    assert len(p["events"]) > 3000 and p["factors"] == live.FACTORS
    assert p["_U"].shape == (len(p["events"]), len(live.FACTORS))
    assert set(p["_resid"]) == set(live.SEVERITIES) and len(p["_resid"]["extreme"]) >= 20
    assert p["model"]["severity_scale"]["extreme"] > p["model"]["severity_scale"]["moderate"]   # #175: big events are bigger


def test_scan_ranks_risks_for_this_book_by_bad_case():
    s = book()
    out = live.scan(s, n=6)
    bad = [r["bad_case_1_in_10_usd"] for r in out["risks"]]
    assert len(out["risks"]) == 6 and bad == sorted(bad)
    assert all(r["typical_usd"] >= r["bad_case_1_in_10_usd"] for r in out["risks"])
    assert out["coverage"]["share_explained"] is not None and s.__dict__["planb_scan"] is out


def test_evidence_then_price_gives_ordered_ranges_that_add_up():
    s = book()
    ev = live.evidence(s, SPEC)
    assert len(ev["closest_past_events"]) == 5 and "oil" in ev["code_estimate"]
    assert ev["history_says"].startswith("weak")          # wars: say plainly that history shows little broad effect
    out = live.price(s)
    for w in ("first_hours", "end_of_day"):
        q = out["if_it_happens"][w]
        assert q["wide_range"][0] <= q["likely_range"][0] <= q["middle"] <= q["likely_range"][1] <= q["wide_range"][1]
    pb = s.__dict__["planb_scenario"]
    assert abs(sum(r["pnl_if_it_happens"] for r in out["by_position"]) - pb["pnl_center"]) < 1e-6
    assert abs(out["if_it_happens"]["end_of_day"]["middle"] - pb["pnl_center"]) < 1e-6   # one middle case everywhere
    assert pb["pnl_bad"] < pb["pnl_center"]
    # #182: numbers only from code — the middle is the engine's estimate through the book's exposures
    assert abs(pb["pnl_center"] - out["reply_facts"]["middle_usd"]) < 1e-6
    near = out["nearest_past_events_on_this_book"]
    assert near["worst"]["your_book_usd"] <= near["best"]["your_book_usd"]
    for t in out["tail_probabilities_end_of_day"]:
        assert 0.02 <= t["probability"] <= 0.6 and t["loss_over_usd"] > 0
    assert isinstance(out["probability"], str)            # no market linked: conditional, no odds


def test_price_needs_evidence_and_bad_specs_are_tool_errors():
    s = book()
    with pytest.raises(ToolError):
        live.price(s)
    with pytest.raises(ToolError):
        live.evidence(s, dict(SPEC, category="weather"))


def test_hedges_price_the_planb_scenario_middle_and_bad_case():
    s = book()
    live.evidence(s, SPEC)
    live.price(s)
    out = tools.compare_hedges_tool(s)
    assert out["scenario"] == SPEC["title"] and out["rows"] and out["stress"]["rows"]
    assert out["stress"]["loss_if_it_happens"] < out["loss_if_yes"]


def test_hedge_verdict_defaults_to_skip_and_flips_for_extreme_or_a_high_bad_case_weight():
    s = book()
    live.evidence(s, SPEC)
    live.price(s)
    v = tools.compare_hedges_tool(s)["should_you_hedge"]
    assert v["verdict"] == "skip by default"                       # #199: protection held through news rarely pays
    assert v["your_normal_2_day_bad_case_usd"] < 0 and 0 < v["history"]["share_that_paid_for_themselves"] < 0.5
    assert tools.compare_hedges_tool(s, bad_case_weight=8)["should_you_hedge"]["verdict"] == "worth considering"
    live.evidence(s, dict(SPEC, severity="extreme"))
    live.price(s)
    assert tools.compare_hedges_tool(s)["should_you_hedge"]["verdict"] == "worth considering"


def test_after_the_open_range_only_for_events_reported_before_the_open():
    s = book()
    live.evidence(s, SPEC)
    assert live.price(s)["after_the_open"] is None
    a = live.price(s, already_reported_before_open=True)["after_the_open"]
    assert a["range_9_in_10_usd"][0] < a["middle_to_tomorrows_close_usd"] < a["range_9_in_10_usd"][1]
    assert abs(a["middle_to_tomorrows_close_usd"] - live.OPEN_SLOPE * a["book_opening_move_usd"]) < 1e-6


def test_event_analyst_gets_planb_tools_only_under_planb(monkeypatch):
    names = [d["name"] for d in tool_defs("event_analyst")]
    assert "event_evidence" in names and "price_scenario" in names and "explain_event" not in names
    ps = next(d for d in tool_defs("event_analyst") if d["name"] == "price_scenario")
    props = ps["input_schema"]["properties"]                            # #182: no way for the LLM to pass a number
    assert set(props) == {"market_id", "already_reported_before_open", "company_news", "sector_news", "country_news"} and props["already_reported_before_open"]["type"] == "boolean"
    sec = props["sector_news"]["items"]["properties"]                       # fixed choices only
    assert all("enum" in sec[k] for k in ("sector", "direction", "size"))
    leg = props["company_news"]["items"]["properties"]                     # a ticker and two fixed choices: no numbers
    assert leg["ticker"]["type"] == "string" and "enum" in leg["news_type"] and "enum" in leg["direction"]
    monkeypatch.setattr(config, "SCENARIO_ENGINE", "v1")
    names = [d["name"] for d in tool_defs("event_analyst")]
    assert "explain_event" in names and "event_evidence" not in names


def _role(kw):
    sys_text = " ".join(b.get("text", "") for b in kw.get("system", []) if isinstance(b, dict))
    for r in ("router", "event_analyst"):
        if f"You are {r}" in sys_text or (r == "router" and "front desk" in sys_text):
            return r
    return "other"


def _last_result(kw):
    for b in reversed(kw["messages"][-1]["content"] if isinstance(kw["messages"][-1]["content"], list) else []):
        if b.get("type") == "tool_result":
            c = b["content"]
            return json.loads(c if isinstance(c, str) else c[0]["text"])
    return None


def test_full_turn_one_event_runs_the_assisted_loop_and_passes_grounding():
    state = {"router": 0, "expert": 0}
    cards = []

    def f(kw):
        role = _role(kw)
        if role == "router":
            state["router"] += 1
            if state["router"] == 1:
                return response(tool_use("r1", "load_portfolio", {"name": "balanced"}))
            if state["router"] == 2:
                return response(tool_use("r2", "analyze_event", {"query": "tankers attacked in the Gulf",
                                                                  "question": "what if tankers get hit in the Gulf?"}))
            return response(text_block("[expert]"))
        state["expert"] += 1
        if state["expert"] == 1:
            return response(tool_use("e1", "event_evidence", SPEC))
        if state["expert"] == 2:
            return response(tool_use("e2", "price_scenario", {}))
        res = _last_result(kw)
        mid = res["if_it_happens"]["end_of_day"]["middle"]
        return response(text_block(f"**If it happens: about ${abs(mid):,.0f} by the close.** Oil is the channel. Test numbers."))

    s = Session(SyntheticMarketData())
    s.emit = lambda kind, payload: cards.append(payload.get("type")) if kind == "card" else None
    out = run_turn(FakeLLM(f), s, "what if tankers get hit in the Gulf?")
    assert {"load_portfolio", "analyze_event", "event_evidence", "price_scenario"} <= set(out["tools"]) and out["grounding"]["ok"], out
    assert "scenario" not in cards and s.routing[-1]["engine"] == "planb"   # the live Plan B chat is text only for events
    assert s.__dict__["planb_scenario"]["title"] == SPEC["title"]
    assert np.isfinite(s.__dict__["planb_scenario"]["pnl_center"])


def test_earnings_calendar_counts_look_through_and_prices_the_move(monkeypatch):
    s = book()
    today = s.data.today()
    nxt = {"date": str(today.replace(day=min(today.day + 10, 28))), "time": "16:05", "status": "confirmed"}
    monkeypatch.setattr(type(s.data), "earnings", lambda self, t: {"next": nxt if t == "NVDA" else None,
                                                                    "past_moves": [{"move": m} for m in (0.08, -0.05, 0.11, -0.03)]},
                        raising=False)
    out = tools.earnings_calendar_tool(s)
    nv = next(r for r in out["rows"] if r["ticker"] == "NVDA")
    assert nv["your_exposure_usd"] >= float(s.priced.at["NVDA", "value"])      # direct + through QQQ/SPY
    assert nv["past_median_move"] == 0.065 and "options_pricing" in out["evidence"]
    names = [d["name"] for d in tool_defs("router")]
    assert "earnings_calendar" in names


def test_news_base_rates_scale_to_the_stock_and_the_book():
    s = book()
    out = tools.news_base_rates_tool(s, "Downgrades", "NVDA")
    assert 0.5 < out["headline_direction_held"] < 1 and out["typical_move_x_normal_day"] > 0
    assert out["your_exposure_usd"] >= float(s.priced.at["NVDA", "value"]) and out["your_dollars_at_typical_move"] > 0
    assert "error" in tools.news_base_rates_tool(s, "Weather")
    ao = out["after_the_open"]                                  # synthetic data has today's opening gaps
    assert ao["range_9_in_10_to_close"][0] < ao["middle_to_close"] < ao["range_9_in_10_to_close"][1]


def test_scan_splits_every_risk_by_position_and_the_parts_add_up():
    s = book()
    out = live.scan(s)
    r = out["risks"][0]
    bp = r["by_position"]
    assert {x["ticker"] for x in bp["positions"]} <= set(s.priced.index) and len(bp["positions"]) >= 5
    assert abs(sum(x["in_the_books_bad_case_usd"] for x in bp["positions"]) - bp["books_worst_1_in_10_average_usd"]) < 1e-6
    assert all(x["own_bad_case_1_in_10_usd"] <= x["own_good_case_1_in_10_usd"] for x in bp["positions"])
    assert set(r["worst_past_example"]["by_position_usd"]) <= set(s.priced.index) if "worst_past_example" in r else True


def test_stress_scenario_maps_a_plain_shock_through_every_holding_and_flags_beyond_history():
    s = book()
    out = tools.stress_scenario_tool(s, "Hormuz closed for weeks", {"oil": 60.0, "market": -6.0}, {"oil": "20% of world supply"})
    assert out["shock"]["oil"]["set_by"] == "analyst" and out["shock"]["oil"]["move"] == "+60.0%"
    assert any("oil" in f for f in out["beyond_history"])                  # +60% is past any 2-session event window
    tot = sum(r["pnl_usd"] for r in out["by_position"])
    assert abs(out["book_usd"]["as_written"] - tot) < 1e-6 and abs(out["book_usd"]["if_it_is_1_5x"] - 1.5 * tot) < 1e-6
    xle = next(r for r in out["by_position"] if r["ticker"] == "XLE")
    assert xle["pnl_usd"] > 0                                                # energy gains when oil spikes
    assert tools.compare_hedges_tool(s)["should_you_hedge"]["verdict"] == "worth considering"   # a black swan
    assert abs(live.to_plain("rates_10y", live.to_factor("rates_10y", 25.0)) - 25.0) < 1e-9


def _book_with_puts():
    s = Session(SyntheticMarketData())
    S = s.data.prices(["SPY"])["SPY"]
    exp = str(s.data.chain("SPY")["expiry"].iloc[-1])[:10]
    tools.load_portfolio(s, holdings=[{"ticker": "QQQ", "value_usd": 50000}, {"ticker": "TLT", "value_usd": 30000},
                                      {"ticker": "SPY", "option": {"kind": "put", "strike": round(S * 0.95), "expiry": exp, "contracts": 2}}])
    return s


def test_option_positions_are_repriced_in_every_engine():
    s = _book_with_puts()
    assert s.__dict__["options"] and "SPY" not in s.priced.index            # options sit beside the linear book
    out = tools.stress_scenario_tool(s, "crash", {"market": -8.0})
    put = next(r for r in out["by_position"] if "put" in r["ticker"])
    assert put["pnl_usd"] > 0 and put["pnl_1_5x_shock_usd"] > put["pnl_usd"]          # puts gain more in a bigger crash
    g = out["options"][0]["greeks_at_the_bad_case"]
    assert g["by_greek_usd"]["delta"] > 0 and abs(g["full_repricing_usd"] - g["sum_of_greeks_usd"] - g["remainder_usd"]) < 1e-6
    sc = live.scan(s)
    labels = [x["ticker"] for x in sc["risks"][0]["by_position"]["positions"]]
    assert any("put" in t for t in labels)
    live.evidence(s, SPEC)
    pr = live.price(s)
    assert pr["options"] and pr["options"][0]["max_loss_usd"] < 0


def test_full_repricing_caps_a_long_put_at_its_premium():
    from worsey.options import scenario as OS
    pos = OS.OptionPosition("SPY", "put", 100.0, __import__("datetime").date(2027, 1, 15), 1)
    resp = {"down": -0.8, "up": -0.4}
    pnl, _ = OS.scenario_pnl(pos, 100.0, 0.2, 0.25, __import__("numpy").array([0.5, 1.0]), resp)
    prem = float(OS.bs_vec(100.0, 100.0, 0.25, 0.2, "put")) * 100
    assert pnl.min() >= -prem - 1e-6                                       # a +50%/+100% rally loses at most the premium


def _uso_nvda():
    s = Session(SyntheticMarketData())
    tools.load_portfolio(s, holdings=[{"ticker": "USO", "value_usd": 50000}, {"ticker": "NVDA", "value_usd": 50000}])
    return s


def test_a_what_if_with_directions_gets_a_sensible_middle_and_keeps_historys_average():
    s = _uso_nvda()
    spec = dict(SPEC, severity="major", factor_signs={"oil": "+", "market": "-", "gold": "+"})
    tools.event_evidence(s, **{k: spec[k] for k in ("title", "category", "severity", "direction", "region", "targeted_sectors", "factor_signs")})
    out = live.price(s)
    uso = next(r for r in out["by_position"] if r["ticker"] == "USO")
    assert uso["move_if_it_happens"] > 0.01                                  # oil up on a real supply threat
    assert out["mode"].startswith("if it's a real market mover") and "history_average_for_events_like_this" in out
    assert out["factor_middle_moves"]["oil"].startswith("+")


def test_company_news_leg_adds_to_the_stock_and_shows_separately():
    s = _uso_nvda()
    tools.event_evidence(s, title="Iran attack", category="conflict_war", severity="major", factor_signs={"oil": "+", "market": "-"})
    base = next(r for r in live.price(s)["by_position"] if r["ticker"] == "NVDA")
    out = live.price(s, company_news=[{"ticker": "NVDA", "news_type": "Contracts", "direction": "positive"}])
    nv = next(r for r in out["by_position"] if r["ticker"] == "NVDA")
    assert nv["pnl_if_it_happens"] > base["pnl_if_it_happens"] and nv["of_which_from_the_headline"] > 0
    assert out["company_news"][0]["pnl_middle_usd"] > 0


def test_upcoming_events_price_in_the_odds(monkeypatch):
    from worsey import discovery
    s = book()
    mk = [m for m in s.data.event_markets() if live._p(m) is not None][:2]
    fake = {m["id"]: {"moves_markets": True, "severity": "major", "factor_signs": {"oil": "+", "market": "-"},
                      "factor_sizes": {"oil": "large", "market": "medium"}, "sectors": [], "companies": [], "why": "test"} for m in mk}
    monkeypatch.setattr(discovery, "labels", lambda: fake)
    out = tools.upcoming_events_tool(s, horizon_days=3650)
    r = out["rows"][0]
    p_ = r["odds_yes"]
    assert abs(r["if_yes_usd"] - r["full_impact_usd"] * (1 - p_)) < 1e-6 and abs(r["if_no_usd"] + r["full_impact_usd"] * p_) < 1e-6
    assert abs(r["risk_left_usd"] - abs(r["full_impact_usd"]) * (p_ * (1 - p_)) ** 0.5) < 1e-6
    assert r["by_position"] and r["title"]


def test_noise_is_an_answer():
    s = _uso_nvda()
    tools.event_evidence(s, title="A celebrity tweets about chips", category="regulatory_court", severity="minor", market_moving=False)
    out = live.price(s)
    assert out["verdict"]["call"] == "noise" and out["mode"].startswith("not a market event")
    assert abs(out["if_it_happens"]["end_of_day"]["middle"]) < 0.25 * out["verdict"]["normal_2_day_swing_usd"]
    tools.event_evidence(s, title="Hormuz closed", category="conflict_war", severity="extreme",
                         factor_signs={"oil": "+", "market": "-"}, factor_sizes={"oil": "large", "market": "large"})
    assert live.price(s)["verdict"]["call"] in ("small", "meaningful")


def _book_with_qqq_and_puts():
    s = Session(SyntheticMarketData())
    S = s.data.prices(["QQQ"])["QQQ"]
    exp = str(s.data.chain("QQQ")["expiry"].iloc[-1])[:10]
    tools.load_portfolio(s, holdings=[{"ticker": "QQQ", "value_usd": 50000}, {"ticker": "TLT", "value_usd": 30000},
                                      {"ticker": "QQQ", "option": {"kind": "put", "strike": round(S * 0.95), "expiry": exp, "contracts": 2}}])
    return s


def test_a_held_name_and_its_option_share_one_own_noise_draw():
    # #235: drawn jointly, the book keeps its own-noise spread and the name its own; the two are correlated as measured
    s = _book_with_qqq_and_puts()
    vals = s.priced["value"]
    L = live.loadings(s.data, list(vals.index))
    sn = live.shared_noise(L, vals, ["QQQ"])
    cov = live.book_coverage(L, vals)
    assert abs(sn["idio"].std() / (cov["daily_noise_usd"] * np.sqrt(1.5)) - 1) < 0.05
    assert abs(sn["name"]["QQQ"].std() / (float(L["resid_sd"]["QQQ"]) * np.sqrt(1.5)) - 1) < 0.08
    assert np.corrcoef(sn["idio"], sn["name"]["QQQ"])[0, 1] > 0.3          # QQQ is most of the book's own noise
    assert live.shared_noise(L, vals, ["SPY"]) is None                      # an underlier not held: independent, as before


def test_an_option_underlier_gets_the_stocks_company_leg_and_shared_noise():
    from worsey import stocks
    s = _book_with_qqq_and_puts()
    live.evidence(s, SPEC)
    typ = sorted(stocks._news_stats())[0]
    pr = live.price(s, company_news=[{"ticker": "QQQ", "news_type": typ, "direction": "negative"}])
    opt = pr["options"][0]
    stock = next(p for p in pr["by_position"] if p["ticker"] == "QQQ")
    assert opt["underlier_legs"]["company"] < 0 and opt["underlier_legs"]["noise_shared_with_the_stock"]
    assert abs(opt["underlying_move_middle"] - stock["move_if_it_happens"]) < 1e-9   # the option's underlier = the stock


def test_an_option_on_a_name_not_held_is_unchanged():
    s = _book_with_puts()                                                   # SPY puts, no SPY stock (like the trainer's USO puts)
    live.evidence(s, SPEC)
    pr = live.price(s)
    assert pr["options"][0]["underlier_legs"] == {"sector": 0.0, "company": 0, "noise_shared_with_the_stock": False}


def test_every_holding_gets_an_average_and_range_that_add_up_to_the_book():
    # text-only event messages: each holding's average move and 9-in-10 range from the same draws as the book, options
    # repriced in every draw; the holdings' averages add up to the book's
    from worsey import discovery, scan_messages
    s = _book_with_qqq_and_puts()
    ctx = discovery.context(s)
    r = discovery.impact(ctx, {"severity": "major", "factor_signs": {"market": "-"}, "factor_sizes": {"market": "large"}})
    pos, book = r["by_position"], r["book_if_yes"]
    assert len(pos) == len(ctx["held"]) + len(ctx["opts"]) == 3
    assert sum(p["mean_usd"] for p in pos) == pytest.approx(book["mean_usd"], rel=1e-6, abs=1e-6)
    for p in pos + [book]:
        lo, hi = p["range_90_usd"]
        assert lo <= p["mean_usd"] <= hi and hi > lo
    put = next(p for p in pos if p["ticker"] not in ctx["held"])
    assert put["mean_usd"] > 0 and put["range_90_usd"][1] > put["range_90_usd"][0]   # a put gains when US stocks fall
    row = dict(r, title="US stocks fall hard", odds=0.12, tier=3)
    msg = scan_messages.event(row, ctx["swing"])
    assert "Probability: 12% (Worsey forecast" in msg and "9 in 10:" in msg and all(f"- {p['ticker']}:" in msg for p in pos)


def test_event_cards_are_dropped_on_the_planb_path_and_kept_for_v1(monkeypatch):
    from worsey import config
    sent = []
    s = Session(SyntheticMarketData())
    s.emit = lambda kind, payload: sent.append(payload.get("type"))
    monkeypatch.setattr(config, "SCENARIO_ENGINE", "planb")
    for kind in ("risks", "scenario", "strategy", "etf_match"):
        s.send("card", {"type": kind, "payload": {}})
    assert sent == ["strategy", "etf_match"]
    monkeypatch.setattr(config, "SCENARIO_ENGINE", "v1")
    s.send("card", {"type": "scenario", "payload": {}})
    assert sent[-1] == "scenario"
