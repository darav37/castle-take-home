"""Pricer tests (SPEC §8 item 4): parity, greeks vs finite differences, spread payoffs vs closed form.
Put-call parity on *fetched* chains is added once chains are cached (checkpoint 1/4).
"""

import math

import numpy as np
import pytest

from worsey.options.pricer import (
    MULT, Leg, MoveDistribution, Strategy, bs_greeks, bs_price, implied_vol, straddle_implied_move,
)

S, K, T, r, q, sig = 100.0, 105.0, 0.5, 0.04, 0.01, 0.30


def test_put_call_parity():
    c = bs_price(S, K, T, r, q, sig, "call")
    p = bs_price(S, K, T, r, q, sig, "put")
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_known_value():
    # Hull-style textbook check: S=K=100, T=1, r=5%, σ=20%, q=0 → call ≈ 10.4506
    assert bs_price(100, 100, 1, 0.05, 0, 0.2, "call") == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_vs_finite_differences(kind):
    g = bs_greeks(S, K, T, r, q, sig, kind)
    h = 1e-3
    f = lambda **kw: bs_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("sig", sig), kind)
    assert g.delta == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
    assert g.vega == pytest.approx((f(sig=sig + h) - f(sig=sig - h)) / (2 * h), rel=1e-6)
    assert g.rho == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    # theta is dV/dt = -dV/dT
    assert g.theta == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_roundtrip(kind):
    px = bs_price(S, K, T, r, q, 0.42, kind)
    assert implied_vol(px, S, K, T, r, q, kind) == pytest.approx(0.42, abs=1e-7)


def test_implied_vol_out_of_bounds_is_nan():
    assert math.isnan(implied_vol(0.0001, S, 50, T, r, q, "call"))  # below intrinsic


def test_call_spread_payoff_closed_form():
    K1, K2 = 100.0, 110.0
    st = Strategy("cds", [Leg("call", 1, K1, T, 0.3), Leg("call", -1, K2, T, 0.28)], S, r, q)
    cost = st.cost()
    grid = np.array([80.0, 100.0, 104.0, 110.0, 130.0])
    expect = (np.clip(grid - K1, 0, None) - np.clip(grid - K2, 0, None)) * MULT - cost
    assert np.allclose(st.payoff_at(grid), expect, atol=1e-8)
    mg, ml = st.max_gain_loss()
    assert mg == pytest.approx((K2 - K1) * MULT - cost, abs=1e-6)
    assert ml == pytest.approx(-cost, abs=1e-6)
    (be,) = st.breakevens()
    assert be == pytest.approx(K1 + cost / MULT, abs=1e-6)


def test_unbounded_and_straddle_breakevens():
    long_call = Strategy("lc", [Leg("call", 1, 100, T, 0.3)], S, r, q)
    assert long_call.max_gain_loss()[0] == math.inf
    short_call = Strategy("sc", [Leg("call", -1, 100, T, 0.3)], S, r, q)
    assert short_call.max_gain_loss()[1] == -math.inf
    strad = Strategy("straddle", [Leg("call", 1, 100, T, 0.3), Leg("put", 1, 100, T, 0.3)], S, r, q)
    lo, hi = strad.breakevens()
    prem = strad.cost() / MULT
    assert lo == pytest.approx(100 - prem, abs=1e-6) and hi == pytest.approx(100 + prem, abs=1e-6)


def test_straddle_implied_move_rule_of_thumb():
    # ATM straddle ≈ 0.8·σ·√T of spot for small r, q
    m = straddle_implied_move(100, 100, 0.1, 0, 0, 0.5, 0.5)
    assert m == pytest.approx(0.8 * 0.5 * math.sqrt(0.1), rel=0.01)


def test_expected_pnl_under_risk_neutral_view_is_carry():
    # If the user's distribution IS the risk-neutral one, E[payoff] = price·e^{rT}
    st = Strategy("lc", [Leg("call", 1, 105, T, sig)], S, r, 0.0)
    rn = MoveDistribution(mean_move=math.exp(r * T) - 1, sd=sig * math.sqrt(T), T=T)
    assert st.expected_pnl(rn) == pytest.approx(st.cost() * (math.exp(r * T) - 1), rel=2e-3)


def test_view_inside_call_breakeven_long_call_loses():
    # The teaching moment (§4.1). Implied move ≈ ±9% ⇒ ATM call costs ≈ 4.5% of spot (half the straddle).
    # Finding: the bar for a long call is that ~4.5% breakeven, not the full 9% implied move.
    T1 = 30 / 365
    iv = 0.09 / (0.8 * math.sqrt(T1))  # implied move ≈ 9%
    st = Strategy("lc", [Leg("call", 1, 100, T1, iv)], 100, 0.0, 0.0)
    assert st.expected_pnl(MoveDistribution(mean_move=0.03, sd=0.04, T=T1)) < 0  # +3%: right, still loses
    assert st.expected_pnl(MoveDistribution(mean_move=0.07, sd=0.04, T=T1)) > 0  # +7%: inside ±9%, still wins


def test_iv_crush_lowers_value():
    st = Strategy("strad", [Leg("call", 1, 100, 0.05, 0.8), Leg("put", 1, 100, 0.05, 0.8)], 100, 0.0, 0.0)
    crushed = st.with_iv_crush({0.05: 0.4})
    assert crushed.value(100, t_elapsed=0.003) < st.value(100, t_elapsed=0.003)
