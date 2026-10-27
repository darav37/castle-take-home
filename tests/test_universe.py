"""The event universe's pure pieces (DECISIONS #223, #224)."""

from datetime import date

import pytest

from worsey import discovery as D


def test_market_odds_split_the_move_and_base_rates_stay_context():
    r = D._with_odds({"full_impact_usd": -1000.0}, 0.2, "market")
    # the priced-in split uses the market's raw price (what asset prices carry); the probability and the risk left use the
    # longshot-corrected one (tier 1, #233)
    assert r["if_yes_usd"] == pytest.approx(-800.0) and r["if_no_usd"] == pytest.approx(200.0)
    assert r["odds_raw"] == 0.2 and r["tier"] == 1 and r["odds"] < 0.2
    q = r["odds"]
    assert r["risk_left_usd"] == pytest.approx(1000 * (q * (1 - q)) ** 0.5)
    b = D._with_odds({"full_impact_usd": -1000.0}, 0.6, "base rate")
    assert "if_yes_usd" not in b and "risk_left_usd" not in b and "not the odds" in b["odds_note"]


def test_same_question_on_two_venues_is_kept_once():
    k = {"venue": "kalshi", "title": "Fed decision in December: hike 25bps", "when": "2026-12-09"}
    p = {"venue": "polymarket", "title": "Fed decision in December? (hike 25bps)", "when": "2026-12-10"}
    assert D._same(k, p) and not D._same(k, dict(p, venue="kalshi"))
    assert not D._same(k, dict(p, title="Will Brazil's Lula win the second round?"))


def test_us_election_is_the_tuesday_after_the_first_monday():
    assert D._us_election(date(2026, 10, 24), 91) == date(2026, 11, 3)
    assert D._us_election(date(2027, 1, 1), 91) is None


def test_other_routes_to_the_users_move_are_folded_but_opposites_and_plain_risk_off_are_not():
    spike = {"factor_signs": {"oil": "+", "market": "-"}, "factor_sizes": {"oil": "large", "market": "medium"}}
    gas = {"factor_signs": {"oil": "+", "market": "-", "credit": "-"}, "factor_sizes": {"oil": "large", "market": "medium", "credit": "small"}}
    deal = {"factor_signs": {"oil": "-", "market": "+"}, "factor_sizes": {"oil": "large", "market": "small"}}
    ai_bust = {"factor_signs": {"market": "-", "credit": "-"}, "factor_sizes": {"market": "large", "credit": "medium"}}
    assert D.same_move(spike, gas)
    assert not D.same_move(spike, deal) and not D.same_move(spike, ai_bust)
    assert D.same_move(ai_bust, {"factor_signs": {"market": "-"}, "factor_sizes": {"market": "large"}})


def _replay_rows():
    pos = lambda *xs: {"positions": [{"ticker": t, "in_the_books_bad_case_usd": x} for t, x in xs]}  # noqa: E731
    one = lambda c, bad, *xs: {"risk_id": f"category:{c}", "kind": "one-off event", "risk": c.replace("_", " "), "bad_case_1_in_10_usd": bad,  # noqa: E731
                               "how_often": "1.2 big ones a year (2016-2025)", "history_says": "learnable",
                               "worst_past_example": {"date": "2020-03-16", "what": "Covid crash", "your_book_usd": -11500.4}, "by_position": pos(*xs)}
    return [one("financial_stress", -3832.0, ("SPY", -3275.0), ("QQQ", -1223.0), ("AAPL", -1170.0)),
            {"risk_id": "release:cpi", "kind": "scheduled release", "risk": "CPI", "bad_case_1_in_10_usd": -9000.0},
            one("supply_shock", -2340.0, ("QQQ", -770.0), ("SPY", -1955.0)),
            one("disaster", -500.0), one("sanctions", -400.0), one("conflict_war", -2000.0)]


def _fake_ctx():
    import pandas as pd
    return {"SP": {"software": {}}, "today": pd.Timestamp("2026-10-24"), "vals": pd.Series({"SPY": 1.0}), "swing": 1000.0}


def _stub_pricing(monkeypatch):
    monkeypatch.setattr(D, "exposures", lambda ctx: {"book_value": 1.0})
    monkeypatch.setattr(D, "impact", lambda ctx, lb: {"_lb": lb, "full_impact_usd": -2500.0, "by_position": []})
    monkeypatch.setattr(D, "_big_event_rates", lambda: {})


def test_history_block_has_the_top_four_one_off_categories_and_their_holdings():
    h = D.history(_replay_rows())
    assert [r["category"] for r in h] == ["financial_stress", "supply_shock", "conflict_war", "disaster"]   # releases left out
    assert h[0]["holdings_carrying_the_bad_case"] == ["SPY", "QQQ"] and h[1]["holdings_carrying_the_bad_case"] == ["SPY", "QQQ"]
    assert h[0]["worst_past_example"] == {"date": "2020-03-16", "what": "Covid crash", "this_book_usd": -11500}
    assert D.history(None) == [] and D.history([_replay_rows()[1]]) == []


def test_the_prompt_carries_the_history_and_from_history_is_validated(monkeypatch):
    import json
    from worsey.agent.llm import FakeLLM, response, text_block
    _stub_pricing(monkeypatch)
    f = "-M 0S 0S 0S +L 0S -S"
    arr = [{"title": "Private-credit fund gates", "category": "financial_stress", "severity": "major", "f": f, "from_history": "financial_stress", "links": {}},
           {"title": "Made-up kind", "category": "supply_shock", "severity": "major", "f": f, "from_history": "aliens", "links": {}},
           {"title": "No history", "category": "disaster", "severity": "major", "f": f, "links": {}}]
    llm = FakeLLM(lambda kw: response(text_block(json.dumps(arr))))
    out = D.propose(llm, _fake_ctx(), [], [], [], replay=_replay_rows())
    pr = llm.calls[0]["messages"][0]["content"]
    assert "The book's history (code)" in pr and '"category": "financial_stress"' in pr and '"from_history"' in pr
    assert "release:cpi" not in pr and "-3832" in pr
    assert out[0]["from_history"] == "financial_stress" and out[0]["from_history_bad_case_usd"] == -3832.0
    assert out[1]["from_history"] is None and "from_history_bad_case_usd" not in out[1]   # not a category: dropped
    assert out[2]["from_history"] is None
    llm2 = FakeLLM(lambda kw: response(text_block(json.dumps(arr))))
    out2 = D.propose(llm2, _fake_ctx(), [], [], [])            # no replay: the old prompt, nothing claimed from history
    pr2 = llm2.calls[0]["messages"][0]["content"]
    assert "history" not in pr2 and '"why_this_book": "<at most 15 words>", "links"' in pr2
    assert all(r["from_history"] is None for r in out2)


def test_watchlist_without_replay_computes_it_and_leaves_the_card_scan_alone(monkeypatch):
    from worsey.planb import live
    seen, scans = {}, []
    monkeypatch.setattr(D, "context", lambda s: _fake_ctx())
    monkeypatch.setattr(D, "exposures", lambda ctx: {})
    for f in ("market_candidates", "calendar"):
        monkeypatch.setattr(D, f, lambda ctx, h=91: [])
    monkeypatch.setattr(D, "news_candidates", lambda ctx, days=7: [])

    def scan(s, horizon_days=91, n=6):
        scans.append(n)
        s.__dict__["planb_scan"] = {"risks": "overwritten"}
        return {"risks": _replay_rows()}
    monkeypatch.setattr(live, "scan", scan)
    monkeypatch.setattr(D, "propose", lambda *a, replay=None, **k: seen.setdefault("replay", replay) and [])

    class S:
        def require_portfolio(self):
            pass
    s = S()
    s.__dict__["planb_scan"] = {"risks": "the card's"}
    D.watchlist(s, llm=object())
    assert seen["replay"] == _replay_rows() and scans and scans[0] >= len(live.CATEGORIES)
    assert s.__dict__["planb_scan"] == {"risks": "the card's"}
    seen.clear()
    D.watchlist(s, llm=object(), replay=["given"])
    assert seen["replay"] == ["given"] and len(scans) == 1           # given: not recomputed
    D.watchlist(s, llm=None)
    assert len(scans) == 1                                           # no LLM, no proposals: no replay needed


def test_company_and_sector_legs_take_the_larger_never_the_sum():
    # a broad event (AI bust) that also lists NVDA: the semis leg is bigger, so the company leg is set aside
    legs, skipped, over = D.combine_legs({"NVDA": -0.030, "AMD": -0.025}, {"NVDA": -0.018})
    assert legs == {"NVDA": -0.030, "AMD": -0.025} and skipped == ["NVDA"] and over == []
    # an event aimed at NVDA (an export ban naming it) with a semis leg: its own move is bigger and is kept, not stacked
    legs, skipped, over = D.combine_legs({"NVDA": -0.012}, {"NVDA": -0.040})
    assert legs == {"NVDA": -0.040} and over == ["NVDA"] and skipped == []


def test_a_company_specific_label_keeps_its_company_leg():
    # an FDA decision on a held drug maker, no sector leg: the company leg stands alone
    legs, skipped, over = D.combine_legs({}, {"CMPS": 0.06})
    assert legs == {"CMPS": 0.06} and not skipped and not over


def test_both_prompts_carry_the_company_rule():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("dl", Path(__file__).resolve().parents[1] / "scripts/discovery_label.py")
    src = spec.loader.get_source("dl")
    for text in (D.PROPOSE, src):
        assert "only when the event is about that specific company" in text and "Usually empty" in text


def test_tiers_from_relations_without_an_llm_call():
    from worsey.planb import probability as PR
    u = [{"id": "g5", "event_ticker": "G", "last": 0.28, "bid": 0.28, "ask": 0.28, "strike_type": "greater", "floor_strike": 5.0, "title": "gas > $5"},
         {"id": "g6", "event_ticker": "G", "last": 0.08, "bid": 0.08, "ask": 0.08, "strike_type": "greater", "floor_strike": 6.0, "title": "gas > $6"}]
    assert PR.tier_from([{"id": "g5", "relation": "same"}], u, u)["tier"] == 1
    t2a = PR.tier_from([{"id": "g5", "relation": "other_strike", "threshold": 5.5, "above": True}], u, u)
    assert t2a["tier"] == "2a" and 0.08 < t2a["raw"] < 0.28
    assert PR.tier_from([{"id": "g5", "relation": "correlated"}], u, u)["tier"] == 3


def _lbl(f, d, size="large"):
    return {"factor_signs": {f: d}, "factor_sizes": {f: size}}


def test_three_picks_by_risk_left_need_a_real_probability_and_differ_in_move():
    pool = [{"title": "oil spike A", "tier": 1, "risk_left_usd": 900, "_lb": _lbl("oil", "+")},
            {"title": "oil spike B", "tier": 3, "risk_left_usd": 800, "_lb": _lbl("oil", "+")},      # same move as A: skipped
            {"title": "base-rate only", "tier": None, "risk_left_usd": None, "_lb": _lbl("gold", "+")},  # no real probability
            {"title": "rates shock", "tier": "2a", "risk_left_usd": 700, "_lb": _lbl("rates_2y", "+")},
            {"title": "credit event", "tier": 3, "risk_left_usd": 300, "_lb": _lbl("credit", "-")},
            {"title": "small one", "tier": 1, "risk_left_usd": 100, "_lb": _lbl("dollar", "+")}]
    assert [r["title"] for r in D.choose_three(pool)] == ["oil spike A", "rates shock", "credit event"]


def test_forecaster_runs_on_all_unpriced_user_events_and_the_top_unpriced_big_proposals(monkeypatch):
    import pandas as pd
    from worsey.planb import forecaster as FC
    asked = []
    monkeypatch.setattr(FC, "forecast", lambda llm, title, *a, **k: asked.append(title) or {"p": 0.2, "agents": [0.2] * 3,
                                                                                           "spread": [0.2, 0.2], "clipped_to_bounds": False})
    ctx = {"today": pd.Timestamp("2026-10-25")}
    yours = [{"title": "my small worry", "source": "user", "tier": None, "full_impact_usd": -50.0},
             {"title": "my listed worry", "source": "user", "tier": 1, "full_impact_usd": -900.0}]
    props = [{"title": f"p{k}", "source": "proposal", "tier": None, "full_impact_usd": -1000.0 * k} for k in range(1, 6)]
    props.append({"title": "priced", "source": "proposal", "tier": "2a", "full_impact_usd": -9000.0})
    props.append({"title": "too small", "source": "proposal", "tier": None, "full_impact_usd": -100.0})
    st = D.forecast_probabilities(object(), ctx, yours, props, thr=500.0)
    assert asked == ["my small worry", "p5", "p4", "p3"] and st["forecast_runs"] == 4
    assert yours[0]["tier"] == 3 and props[0]["tier"] is None      # p1 is not in the top 3: keeps its base rate


def test_three_legs_take_the_largest_never_the_sum():
    # #238: sector, company and country legs on one holding: the largest move, signed
    legs, skipped, over = D.combine_legs({"EWZ": -0.004, "VALE": 0.010}, {"VALE": -0.006}, {"EWZ": -0.012, "VALE": -0.015})
    assert legs == {"EWZ": -0.012, "VALE": -0.015} and skipped == ["VALE"] and over == []
    legs, _, _ = D.combine_legs({}, {}, {"EPU": 0.02})
    assert legs == {"EPU": 0.02}


def test_country_moves_and_packs():
    from worsey.planb import live
    cp = live.country_pack()["countries"]
    assert cp["Brazil"]["etf"] == "EWZ" and cp["Peru"]["etf"] == "EPU" and "EEM" not in {v["etf"] for v in cp.values()}
    m = live.country_moves([{"country": "Brazil", "direction": "negative"}, {"country": "Peru", "direction": "negative", "size": "big"}])
    assert m["Brazil"] < 0 and m["Peru"] < m["Brazil"]                       # big = the 90th percentile, beyond the median
    dom = live.domicile()
    assert dom["VALE"] == "Brazil" and dom["UBS"] == "Switzerland" and dom["BHP"] == "Australia" and "AAPL" not in dom
