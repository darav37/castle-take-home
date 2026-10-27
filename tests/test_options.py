"""View, market state, builder on a synthetic surface."""

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worsey.options import builder, market_state
from worsey.options.pricer import bs_greeks, bs_price
from worsey.options.view import View, ViewError

TODAY = date(2026, 10, 14)
FIX = Path(__file__).parent / "fixtures" / "strategies"
LIB = builder.load_library(FIX)
EXPS = [TODAY + timedelta(days=d) for d in (7, 14, 21, 35, 63, 91)]


def quoter(iv=0.5, S=100.0):
    return builder.FlatQuoter(S, TODAY, EXPS, lambda T, K: iv, step=1.0)


def test_view_validation_and_diff():
    with pytest.raises(ViewError):
        View("NVDA", "up", date(2026, 11, 20), target_move_pct=-5)
    with pytest.raises(ViewError):
        View("NVDA", "sideways", date(2026, 11, 20))
    v = View("nvda", "up", "2026-11-20", target_move_pct=8)
    v2, d = v.update(horizon=date(2026, 10, 28))
    assert d == {"horizon": {"from": "2026-11-20", "to": "2026-10-28"}}
    dist = v.distribution(TODAY, realized_vol=0.4)
    assert dist.sd == pytest.approx(0.4 * math.sqrt(37 / 365))


def test_library_loader_reads_front_matter_only_fixtures():
    ids = {e["id"] for e in LIB}
    assert {"long_call", "call_debit_spread", "iron_condor"} <= ids
    assert LIB[0]["direction"] and isinstance(LIB[0]["direction"], list)


def test_propose_up_view_builds_priced_candidates():
    v = View("NVDA", "up", TODAY + timedelta(days=21), target_move_pct=10)
    out = builder.propose(v, 100.0, TODAY, quoter(), realized_vol=0.45, library=LIB)
    ids = [c.id for c in out["candidates"]]
    assert set(ids) <= {"long_call", "call_debit_spread"}
    cds = next(c for c in out["candidates"] + out["also_ran"] if c.id == "call_debit_spread")
    ks = sorted(l["strike"] for l in cds.metrics["legs"])
    assert ks == [102.0, 110.0]  # slightly OTM long, short at the target
    assert cds.metrics["max_loss"] == pytest.approx(-cds.metrics["cost"])
    assert any(d["id"] == "short_put" for d in out["dropped"])  # vol view 'none' not in [contract]


def test_teaching_moment_view_inside_breakeven():
    # implied ±9% over 21d; +3% view, high conviction → long call loses on the user's own view
    T = 21 / 365
    iv = 0.09 / (0.8 * math.sqrt(T))
    v = View("NVDA", "up", TODAY + timedelta(days=21), target_move_pct=3, conviction="high")
    out = builder.propose(v, 100.0, TODAY, quoter(iv), realized_vol=0.35, library=LIB)
    assert out["teaching"] == "long_premium_loses_under_view"
    lc = next(c for c in out["candidates"] + out["also_ran"] if c.id == "long_call")
    assert "negative E[P&L] under your own view" in lc.notes


def test_constraints_prune_and_playbook_rule_demotes():
    v = View("NVDA", "up", TODAY + timedelta(days=21), target_move_pct=10, constraints={"max_premium_usd": 300})
    out = builder.propose(v, 100.0, TODAY, quoter(), realized_vol=0.45, library=LIB)
    assert any(d["id"] == "long_call" and "max premium" in d["reason"] for d in out["dropped"])

    rule = {"id": "r1", "when": [{"metric": "view_over_implied", "op": "<", "value": 0.8}],
            "demote": ["long_call"], "promote": ["call_debit_spread"], "say": "view inside implied"}
    v2 = View("NVDA", "up", TODAY + timedelta(days=21), target_move_pct=5)
    out2 = builder.propose(v2, 100.0, TODAY, quoter(), realized_vol=0.45, library=LIB, playbook=[rule], implied_move=0.1)
    assert out2["candidates"][0].id == "call_debit_spread"
    assert any("demoted by r1" in n for c in out2["candidates"] + out2["also_ran"] for n in c.notes)


def test_neutral_views_and_calendar_and_diff():
    v = View("SPY", "neutral", TODAY + timedelta(days=35), vol_view="contract")
    out = builder.propose(v, 100.0, TODAY, quoter(0.2), realized_vol=0.15, library=LIB)
    ic = next(c for c in out["candidates"] + out["also_ran"] if c.id == "iron_condor")
    assert len(ic.metrics["legs"]) == 4 and ic.metrics["cost"] < 0  # credit
    v2, _ = v.update(vol_view="expand")
    out2 = builder.propose(v2, 100.0, TODAY, quoter(0.2), realized_vol=0.15, library=LIB)
    cal = next(c for c in out2["candidates"] + out2["also_ran"] if c.id == "call_calendar")
    days = sorted(l["days"] for l in cal.metrics["legs"])
    assert days[1] - days[0] >= 21
    d = builder.diff_proposals(out, out2)
    assert "iron_condor" in [x["id"] for x in d["out"]]


def test_vol_crush_metric():
    v = View("NVDA", "neutral", TODAY + timedelta(days=7), vol_view="expand", event="earnings 2026-10-16")
    out = builder.propose(v, 100.0, TODAY, quoter(0.9), realized_vol=0.4, library=LIB,
                          event_date=date(2026, 10, 16), iv_after_event={7 / 365: 0.4})
    st = next(c for c in out["candidates"] + out["also_ran"] if c.id == "long_straddle")
    assert st.metrics["pnl_vol_crush_flat_spot"] < -0.3 * st.metrics["cost"]


def test_market_state_backs_out_iv_and_measures_skew():
    S, r = 100.0, 0.04
    rows = []
    for e in EXPS[:3]:
        T = (e - TODAY).days / 365
        for K in range(80, 121, 5):
            for kind in ("call", "put"):
                iv = 0.40 + 0.004 * (100 - K)  # put skew: lower strikes richer
                m = bs_price(S, K, T, r, 0, iv, kind)
                rows.append({"expiry": pd.Timestamp(e), "strike": float(K), "kind": kind, "mid": m, "bid": m, "ask": m})
    chain = pd.DataFrame(rows)
    closes = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 100))))
    closes.iloc[-1] = S
    out = market_state.summarize(chain, closes, TODAY, r=r, earnings_expiry=pd.Timestamp(EXPS[0]), past_earnings_moves=[0.05, -0.07, 0.03])
    assert out["iv_source"] == {"computed": len(chain[chain.expiry == pd.Timestamp(EXPS[0])]) * 3}
    assert out["term"][0]["atm_iv"] == pytest.approx(0.40, abs=1e-4)
    assert out["skew_25d_front"] > 0
    assert out["earnings_hist_mean_abs"] == pytest.approx(0.05)


def test_event_move_from_term_and_crush():
    sd = market_state.event_move_from_term(iv_pre=0.40, iv_post=0.60, T_post=10 / 365)
    assert sd == pytest.approx(math.sqrt(10 / 365 * (0.36 - 0.16)))
    assert market_state.vol_after_event(0.60, 10 / 365, sd) == pytest.approx(0.40)


def test_live_chain_fixture_normalizes_and_fills_iv_per_contract():
    import json
    from worsey.data import massive
    raw = json.loads((Path(__file__).parent / "fixtures" / "live" / "massive_chain_nvda.json").read_text(encoding="utf-8"))
    ch = massive.normalize_chain([r for r in raw["results"] if isinstance(r, dict)])
    assert {"contract", "expiry", "strike", "kind", "bid", "ask", "mid", "iv", "iv_source"} <= set(ch.columns)
    # knock out one vendor IV: it must come back 'computed', with BSM greeks
    ch.loc[0, ["iv", "delta"]] = [float("nan"), None]
    ch.loc[0, "iv_source"] = None
    S = 180.0
    filled = market_state.fill_iv(ch, S, ch["expiry"].min().date() - timedelta(days=1))
    if 0 in filled.index:
        row = filled.loc[0]
        assert row["iv_source"] == "computed" and row["iv"] == pytest.approx(row["iv_mid"]) and pd.notna(row["delta"])
    else:  # deep ITM mid at/below intrinsic → no IV can be backed out; the contract is dropped, not guessed
        assert ch.loc[0, "strike"] < S


def test_pricing_iv_makes_entry_value_equal_cost():
    S, today = 100.0, TODAY
    e = pd.Timestamp(EXPS[2])
    m = bs_price(S, 100, (e.date() - today).days / 365, 0.04, 0, 0.45, "call")
    ch = pd.DataFrame([{"expiry": e, "strike": 100.0, "kind": "call", "mid": m, "bid": m, "ask": m, "iv": 0.50}])  # vendor says 0.50
    f = market_state.fill_iv(ch, S, today)
    assert f.iloc[0]["iv_vendor"] == pytest.approx(0.50) and f.iloc[0]["iv_source"] == "vendor"  # display only
    assert f.iloc[0]["iv"] == pytest.approx(0.45, abs=1e-6)  # mid-implied: pricing, greeks, everything
    g = bs_greeks(S, 100, (e.date() - today).days / 365, 0.04, 0, 0.45, "call")
    assert f.iloc[0]["delta"] == pytest.approx(g.delta)
    q = builder.ChainQuoter(f)
    iv, mid = q.quote("call", 100.0, EXPS[2])
    assert bs_price(S, 100, (e.date() - today).days / 365, 0.04, 0, iv, "call") == pytest.approx(mid, rel=1e-6)


def test_named_structure_bypasses_vol_filter():
    v = View("IWM", "down", TODAY + timedelta(days=63), target_move_pct=-8, vol_view="expand")
    out = builder.propose(v, 100.0, TODAY, quoter(), realized_vol=0.25, library=LIB)
    assert "put_debit_spread" not in [c.id for c in out["candidates"] + out["also_ran"]]  # filtered by vol view
    out2 = builder.propose(v, 100.0, TODAY, quoter(), realized_vol=0.25, library=LIB, include=["put_debit_spread"])
    assert out2["candidates"][0].id == "put_debit_spread" and "asked for by name" in out2["candidates"][0].notes


def test_market_distribution_prices_fair_options_near_zero_and_edge_adds_up():
    # horizon = an expiry, flat model-priced vols: under the market's own distribution a long option is ~fair
    v = View("NVDA", "up", EXPS[2], target_move_pct=10)
    out = builder.propose(v, 100.0, TODAY, quoter(0.5), realized_vol=0.30, library=LIB)
    md = out["market_distribution"]
    assert md["sd"] == pytest.approx(0.5 * math.sqrt(21 / 365), rel=1e-6)
    for c in out["candidates"]:
        m = c.metrics
        assert abs(m["expected_pnl_market"]) < 0.02 * m["cost"]      # fair under the market (forward drift, BSM mids)
        assert m["expected_pnl_market"] + m["edge_direction"] + m["edge_vol"] == pytest.approx(m["expected_pnl"])
        assert m["edge_direction"] > 0    # +10% view is the edge
        if c.id == "long_call":           # RV 30% < IV 50%: a long call pays for vol it doesn't expect
            assert m["edge_vol"] < 0      # (a call spread capped at the target can gain from lower vol instead)
        assert 0 < m["prob_profit_market"] < m["prob_profit"]


def test_vol_effect_words_match_the_sign():
    v = View("NVDA", "up", EXPS[2], target_move_pct=10)
    for c in builder.propose(v, 100.0, TODAY, quoter(0.5), realized_vol=0.30, library=LIB)["candidates"]:
        assert c.metrics["vol_effect"] == ("vol works for you" if c.metrics["edge_vol"] >= 0 else "you pay for vol")


def test_real_library_loads_and_gates():
    lib = builder.load_library(Path(__file__).parents[1] / "library" / "strategies")
    ids = {e["id"] for e in lib}
    assert {"call_debit_spread", "long_straddle", "call_calendar", "reverse_calendar", "long_risky", "short_risky"} <= ids
    v = View("NVDA", "up", TODAY + timedelta(days=21), target_move_pct=10)
    out = builder.propose(v, 100.0, TODAY, quoter(), realized_vol=0.45, library=lib)
    assert "short_risky" not in {c.id for c in out["candidates"]}
    assert any(d["id"] == "short_risky" and d["reason"].startswith("gated") for d in out["dropped"])
    out2 = builder.propose(v, 100.0, TODAY, quoter(), realized_vol=0.45, library=lib, include=["short_risky"])
    assert "short_risky" in {c.id for c in out2["candidates"]}
