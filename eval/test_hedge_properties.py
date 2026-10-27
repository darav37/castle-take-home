"""Property tests for hedge legs: a long option position held to the event's resolution must not
make money when the underlying moves the wrong way (or not at all), net of premium. Random grid over spot, vol,
skew, time and move; legs built exactly as hedge.option_hedge builds them."""

from datetime import date, timedelta

import numpy as np
import pytest

from worsey.hedge import Scenario, Underlying, option_hedge
from worsey.options.builder import FlatQuoter

TODAY = date(2026, 10, 16)
rng = np.random.default_rng(20261016)


def cases(n=150):
    for _ in range(n):
        S = float(rng.uniform(20, 900))
        iv0 = float(rng.uniform(0.10, 0.80))
        skew = float(rng.uniform(0.0, 0.6))            # puts richer below spot
        res_days = int(rng.integers(7, 200))
        exps = [TODAY + timedelta(days=d) for d in (7, 14, 30, 45, 60, 91, 120, 182, 270)]
        step = 1.0 if S < 100 else 5.0
        q = FlatQuoter(S, TODAY, exps, (lambda iv0, skew, S: lambda T, K: iv0 * (1 + skew * max(0.0, 1 - K / S)))(iv0, skew, S), step=step)
        yield S, q, res_days, float(rng.uniform(0.0005, 0.25))


def _sc(ticker, res_days, mv_yes, mv_no):
    return Scenario("e", 0.3, TODAY + timedelta(days=res_days), -10_000.0, 4_000.0, {ticker: mv_yes}, {ticker: mv_no})


@pytest.mark.parametrize("case", list(cases()))
def test_long_puts_never_gain_when_the_underlying_rises_or_holds(case):
    S, q, res_days, m = case
    for opt in ("put_spread", "put"):
        for mv_no in (0.0, m / 3, m):          # NO side: underlying flat or up
            row = option_hedge("index_put_spread", Underlying("X", S, q), opt, _sc("X", res_days, -m, mv_no), TODAY,
                               exposure_usd=S * 100)
            if row is None or "skipped" in row:
                continue
            assert row["pnl_if_no"] <= 1e-6, (opt, S, res_days, m, mv_no, row["name"], row["pnl_if_no"])
            assert "pays_if_no" not in row["flags"]
            assert all(l["strike"] <= S for l in row["legs"] if l["qty"] > 0)  # long leg never starts ITM
            assert row["contracts"] == pytest.approx(1.0)                      # notional / (spot·100), never inflated


@pytest.mark.parametrize("case", list(cases(80)))
def test_long_calls_never_gain_when_the_underlying_falls_or_holds(case):
    S, q, res_days, m = case
    for mv_no in (0.0, -m / 3, -m):
        row = option_hedge("cross_asset_call", Underlying("X", S, q), "call", _sc("X", res_days, m, mv_no), TODAY,
                           exposure_usd=S * 100)
        if row is None or "skipped" in row:
            continue
        assert row["pnl_if_no"] <= 1e-6, (S, res_days, m, mv_no, row["name"], row["pnl_if_no"])
        assert row["legs"][0]["strike"] >= S


def test_spread_is_struck_from_horizon_vol_never_a_sliver():
    S = 728.0
    q = FlatQuoter(S, TODAY, [TODAY + timedelta(days=91)], lambda T, K: 0.2, step=5.0)
    # the Oct 16 CPI case (QQQ −0.24% if YES): strikes now come from σ_h, not the move, so no 730/725 sliver
    row = option_hedge("index_put_spread", Underlying("QQQ", S, q), "put_spread", _sc("QQQ", 77, -0.0024, 0.001), TODAY,
                       exposure_usd=60_000.0)
    k1, k2 = sorted((l["strike"] for l in row["legs"]), reverse=True)
    assert k1 < S and (k1 - k2) / S >= 0.05 and row["pnl_if_no"] <= 0
    assert "cost_exceeds_loss" in row["flags"] or row["cost"] <= 10_000   # flagged when the premium dwarfs the event
