"""natural_world_analyst data (DECISIONS #116): futures curves from the snapshot, carry, crack, calendar."""

from datetime import date

import pytest

from worsey import natural as nw

TODAY = date(2026, 10, 18)


def snap(rows):
    return lambda product: [{"details": {"ticker": t, "settlement_date": sd}, "session": {"settlement_price": px}} for t, sd, px in rows]


def test_curve_keeps_live_outrights_only_and_computes_carry():
    rows = [("CLV6", "2026-09-22", 94.0),          # expired: dropped
            ("CLX6-CLF7", "2026-10-20", 4.5),       # a spread: dropped
            ("MCLX6", "2026-10-19", 101.0),         # a micro: dropped
            ("CLX6", "2026-10-20", 101.84), ("CLZ6", "2026-11-19", 99.04), ("CLF7", "2026-12-17", 97.4)]
    c = nw.futures_curve("CL", TODAY, snapshot=snap(rows), history=lambda t: [])
    assert [p["ticker"] for p in c["curve"]] == ["CLX6", "CLZ6", "CLF7"]
    assert c["shape"] == "backwardation" and c["front_minus_second"] == pytest.approx(2.80)
    assert c["roll_yield_annualized"] == pytest.approx((101.84 / 99.04 - 1) * 12)


def test_two_digit_years_and_contango():
    rows = [("NGX26", "2026-10-28", 3.117), ("NGZ26", "2026-11-25", 3.374)]
    c = nw.futures_curve("NG", TODAY, snapshot=snap(rows), history=lambda t: [])
    assert c["front_month"] == "Nov 2026" and c["shape"] == "contango" and c["roll_yield_annualized"] < 0


def test_release_calendar_has_the_weekly_reports():
    names = [r["release"] for r in nw.release_calendar(TODAY)]
    assert any("Natural Gas Storage" in n for n in names) and any("WASDE" in n for n in names)


def test_keyed_hooks_say_how_to_turn_on(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    assert "add EIA_API_KEY" in nw.keyed_hooks()["EIA (inventories, grid load)"]
    assert nw.eia_series("NG.NW2_EPG0_SWO_R48_BCF.W")["available"] is False
