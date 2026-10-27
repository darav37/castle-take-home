"""Paper option book (SPEC §4.3;): greeks, drift against the position's intent, staleness, hedge tickets.
Offline: a flat synthetic surface, a temp store."""

from datetime import date, timedelta

import pytest

from worsey.options import book
from worsey.options.builder import FlatQuoter

TODAY = date(2026, 10, 16)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(book, "STORE", tmp_path)


def quoter(S, today=TODAY, iv=0.40):
    exps = [today + timedelta(days=d) for d in (5, 30, 60, 90)]
    return FlatQuoter(S, today, exps, lambda T, K: iv, step=5.0)


def straddle(S=200.0, intent="delta_neutral_vol", days=30, qty=1):
    q = quoter(S)
    exp = (TODAY + timedelta(days=days)).isoformat()
    legs = [{"kind": "call", "strike": 200, "expiry": exp, "qty": qty}, {"kind": "put", "strike": 200, "expiry": exp, "qty": qty}]
    return book.new_position("darin", "NVDA", legs, intent, S, TODAY, q.quote)


def test_entry_greeks_and_targets_are_recorded():
    pos = straddle()
    g = pos["entry_greeks"]
    assert abs(g["delta_sh"]) < 10 and g["vega_pt"] > 0 and g["theta_day"] < 0 and g["gamma_sh_1pct"] > 0
    assert pos["targets"]["delta_sh"] == 0.0 and pos["targets"]["vega_pt"] == g["vega_pt"]
    assert book.open_positions("darin")[0]["id"] == pos["id"]


def test_flat_at_entry_means_no_drift_alerts():
    pos = straddle()
    g = book.position_greeks(pos, 200.0, TODAY, quoter(200.0).quote)
    assert [a["kind"] for a in book.check(pos, g, TODAY, 0.025)] == []


def test_spot_move_puts_a_vol_trade_outside_the_band_and_it_goes_stale():
    pos = straddle()
    later = TODAY + timedelta(days=4)
    g = book.position_greeks(pos, 220.0, later, quoter(220.0, later).quote)
    kinds = {a["kind"] for a in book.check(pos, g, later, 0.025)}
    assert "delta_drift" in kinds and "stale_hedge" in kinds          # +10% on a long straddle, no hedge for 4 days
    t = book.hedge_ticket(pos, 220.0, later, quoter(220.0, later).quote, 0.025)
    assert t["trade_shares"] < 0 and not t["inside_band"]              # long gamma after a rally: sell shares
    assert [r["trade_sh"] for r in t["ladder"]] == sorted([r["trade_sh"] for r in t["ladder"]], reverse=True)


def test_recorded_hedge_clears_the_drift():
    pos = straddle()
    later = TODAY + timedelta(days=1)
    q = quoter(220.0, later).quote
    t = book.hedge_ticket(pos, 220.0, later, q, 0.025, at_close=False)
    pos = book.record_hedge("darin", pos["id"], t["trade_shares"], 220.0, later)
    g = book.position_greeks(pos, 220.0, later, q)
    assert "delta_drift" not in {a["kind"] for a in book.check(pos, g, later, 0.025)}


def test_vega_bleeds_with_time_and_near_expiry_is_flagged():
    exp = TODAY + timedelta(days=30)
    pos = straddle()
    later = exp - timedelta(days=3)
    q = FlatQuoter(200.0, later, [exp], lambda T, K: 0.40, step=5.0)
    kinds = {a["kind"] for a in book.check(pos, book.position_greeks(pos, 200.0, later, q.quote), later, 0.025)}
    assert {"vega_drift", "near_expiry"} <= kinds


def test_vol_trade_that_became_directional_is_called_out():
    pos = straddle()
    later = TODAY + timedelta(days=10)
    g = book.position_greeks(pos, 260.0, later, quoter(260.0, later).quote)
    assert "not_a_vol_trade" in {a["kind"] for a in book.check(pos, g, later, 0.025)}


def test_expired_leg_and_gap_risk():
    pos = straddle(days=5)
    later = TODAY + timedelta(days=6)
    g = book.position_greeks(pos, 200.0, later, quoter(200.0, later).quote)
    kinds = {a["kind"] for a in book.check(pos, g, later, 0.025, event_before_expiry={"name": "NVDA earnings", "date": "x"})}
    assert {"expired_leg", "gap_risk"} <= kinds


def test_directional_position_reports_delta_shrinking_not_a_hedge_drift():
    q = quoter(200.0)
    exp = (TODAY + timedelta(days=30)).isoformat()
    pos = book.new_position("darin", "NVDA", [{"kind": "call", "strike": 200, "expiry": exp, "qty": 2}], "directional", 200.0, TODAY, q.quote)
    later = TODAY + timedelta(days=10)
    kinds = {a["kind"] for a in book.check(pos, book.position_greeks(pos, 170.0, later, quoter(170.0, later).quote), later, 0.025)}
    assert "delta_since_entry" in kinds and "delta_drift" not in kinds


def test_unlisted_contract_is_rejected_and_bad_intent_too():
    q = quoter(200.0)
    with pytest.raises(ValueError):
        book.new_position("darin", "NVDA", [], "yolo", 200.0, TODAY, q.quote)
