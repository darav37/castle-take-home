import math
from datetime import date, timedelta

import pytest

from worsey.hedge import Scenario, Underlying, compare_hedges, pick_cross_asset, prediction_market_hedge
from worsey.options.builder import FlatQuoter

TODAY = date(2026, 10, 14)
EXPS = [TODAY + timedelta(days=d) for d in (7, 14, 30, 45, 60, 90)]


def q(S, iv):
    return FlatQuoter(S, TODAY, EXPS, lambda T, K: iv, step=1.0 if S < 200 else 5.0)


def scen():
    # tariffs: book β$ = −20k, p = 0.4 ⇒ −12k if YES, +8k if NO
    return Scenario("tariffs", 0.4, TODAY + timedelta(days=40), loss_if_yes=-12_000, pnl_if_no=8_000,
                    asset_moves_yes={"SPY": -0.06, "XLK": -0.09, "TLT": 0.012, "GLD": 0.02},
                    asset_moves_no={"SPY": 0.04, "XLK": 0.06, "TLT": -0.008, "GLD": -0.013})


def test_prediction_market_hedge_exact_coverage():
    r = prediction_market_hedge(scen())
    assert r["coverage"] == pytest.approx(1.0)
    assert r["cost"] == pytest.approx(12_000 / 0.6 * 0.4)
    assert r["pnl_if_no"] == pytest.approx(-r["cost"])
    assert r["residual_if_yes"] == pytest.approx(0)
    # paying p for a p-probability payout: expected hedge P&L is zero (before fees)
    assert 0.4 * r["pnl_if_yes"] + 0.6 * r["pnl_if_no"] == pytest.approx(0)


def test_compare_table_has_all_kinds_and_sane_metrics():
    sc = scen()
    sc.asset_moves_yes["GLD"] = 0.06
    expo = {"SPY": 60_000.0, "XLK": 40_000.0, "GLD": 20_000.0}
    out = compare_hedges(sc, TODAY, index=Underlying("SPY", 600.0, q(600.0, 0.18)),
                         sector=Underlying("XLK", 250.0, q(250.0, 0.25)),
                         cross=Underlying("GLD", 300.0, q(300.0, 0.16)), exposures=expo)
    kinds = [r["hedge"] for r in out["rows"]]
    assert kinds == ["do_nothing", "prediction_market", "index_put_spread", "sector_puts", "cross_asset_call"]
    for r in out["rows"][2:]:
        # sized to the β-weighted exposure, not stretched to cover the loss: contracts = notional / (spot·100)
        und = r["underlying"]
        spot = {"SPY": 600.0, "XLK": 250.0, "GLD": 300.0}[und]
        assert r["contracts"] == pytest.approx(expo[und] / (spot * 100))
        assert r["cost"] > 0 and r["pnl_if_no"] <= 0 and r["basis_risk"]
    ps = out["rows"][2]
    k1, k2 = sorted((l["strike"] for l in ps["legs"]), reverse=True)
    sig = ps["horizon_sigma"]
    assert k1 < 600 and abs(math.log(k1 / 600) + 0.35 * sig) < 0.01 + 5 / 600   # long leg ~0.35σ OTM (strike grid 5)
    assert abs(math.log(k2 / 600) + 1.25 * sig) < 0.01 + 5 / 600                 # short leg ~1.25σ OTM


def test_coverage_is_honest_and_cost_above_loss_is_flagged():
    # typical case: a -5% index move on a $100k equity book → the spread costs less than the loss it hedges
    sc = Scenario("crash", 0.3, TODAY + timedelta(days=60), -5_000.0, 2_000.0, {"SPY": -0.05}, {"SPY": 0.02})
    row = compare_hedges(sc, TODAY, index=Underlying("SPY", 600.0, q(600.0, 0.18)), exposures={"SPY": 100_000.0},
                         include_market=False)["rows"][1]
    assert row["cost"] <= 5_000 and "cost_exceeds_loss" not in row["flags"]
    assert 0 < row["coverage"] < 1.0          # partial, reported as is
    # a tiny move (the Oct 16 CPI case): the index spread barely pays, so its cost now exceeds the loss → flagged
    tiny = Scenario("cpi", 0.12, TODAY + timedelta(days=30), -250.0, 35.0, {"SPY": -0.0025}, {"SPY": 0.0004})
    r2 = compare_hedges(tiny, TODAY, index=Underlying("SPY", 600.0, q(600.0, 0.18)), exposures={"SPY": 100_000.0},
                        include_market=False)["rows"][1]
    assert "cost_exceeds_loss" in r2["flags"] and r2["coverage"] < 0.5


def test_skips_and_nothing_to_hedge():
    sc = scen()
    out = compare_hedges(sc, TODAY, cross=Underlying("SPY", 600.0, q(600.0, 0.18)), exposures={"SPY": 50_000.0})
    assert out["rows"][-1]["hedge"] == "cross_asset_put"  # SPY falls if YES ⇒ a put, not a call
    gain = Scenario("x", 0.5, sc.resolution, 5_000, -5_000, {}, {})
    assert [r["hedge"] for r in compare_hedges(gain, TODAY)["rows"]] == ["do_nothing"]
    assert pick_cross_asset(sc.asset_moves_yes) == ["GLD", "TLT"]  # biggest |move| first
    small = compare_hedges(sc, TODAY, cross=Underlying("GLD", 300.0, q(300.0, 0.16)), include_market=False,
                           exposures={"GLD": 20_000.0})
    assert "does_not_pay_if_yes" in small["rows"][-1]["flags"]  # +2% GLD can't beat the call's premium


def test_cross_asset_put_when_asset_falls_if_yes():
    sc = scen()
    sc.asset_moves_yes["TLT"] = -0.06   # bonds sell off on an inflation scare
    sc.asset_moves_no["TLT"] = 0.02
    out = compare_hedges(sc, TODAY, cross=[Underlying("TLT", 90.0, q(90.0, 0.15))], include_market=False,
                         exposures={"TLT": 30_000.0})
    row = out["rows"][-1]
    assert row["hedge"] == "cross_asset_put" and row["legs"][0]["kind"] == "put" and row["legs"][0]["strike"] < 90
    assert row["contracts"] == pytest.approx(30_000 / 9_000)


def test_unpriced_template_hedge_has_no_contract_and_no_invented_probability():
    sc = Scenario("trade_policy", None, TODAY + timedelta(days=91), -6_000.0, 0.0,
                  {"SPY": -0.05, "GLD": 0.03}, {"SPY": 0.0, "GLD": 0.0})
    out = compare_hedges(sc, TODAY, index=Underlying("SPY", 600.0, q(600.0, 0.18)), book_value=100_000,
                         exposures={"SPY": 100_000.0})
    kinds = [r["hedge"] for r in out["rows"]]
    assert "prediction_market" not in kinds and out["unpriced"]
    dn = out["rows"][0]
    assert dn["expected_resolution_pnl"] is None and dn["sigma_resolution"] is None
    ps = next(r for r in out["rows"] if r["hedge"] == "index_put_spread")
    assert 0 < ps["coverage"] <= 1.0 and ps["pnl_if_no"] <= 0
