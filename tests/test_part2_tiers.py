"""Part 2 probability tiers: the code-only pieces (DECISIONS #227-228)."""

from datetime import date

import pytest

from worsey.planb import probability as PR


def _m(i, last, **kw):
    return dict({"id": i, "event_ticker": "E", "last": last, "bid": last, "ask": last}, **kw)


def test_strike_ladder_interpolates_and_never_extrapolates():
    u = [_m("a", 0.62, strike_type="greater", floor_strike=0.1), _m("b", 0.48, strike_type="greater", floor_strike=0.2),
         _m("c", 0.24, strike_type="greater", floor_strike=0.4)]
    assert PR.derive_strike(u[0], u, 0.3)["p"] == pytest.approx(0.36)
    assert PR.derive_strike(u[0], u, 0.3, above=False)["p"] == pytest.approx(0.64)
    assert PR.derive_strike(u[0], u, 0.6)["p"] is None


def test_exact_buckets_become_a_survival_curve():
    u = [_m(str(k), p, strike_type="between", floor_strike=k) for k, p in ((0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05))]
    lad = PR.strike_ladder(u[0], u)
    assert lad["p_above"] == pytest.approx([1.0, 0.5, 0.2, 0.05])


def test_deadline_rungs_interpolate_and_stay_monotone():
    today = date(2026, 10, 1)
    u = [_m("n", 0.01, close_date="2026-11-01"), _m("d", 0.07, close_date="2026-12-01"), _m("j", 0.05, close_date="2027-01-01")]
    r = PR.derive_deadline(u[0], u, "2026-11-16", today)
    assert r["p"] == pytest.approx(0.04, abs=0.002)
    assert PR.derive_deadline(u[0], u, "2026-12-20", today)["p"] == pytest.approx(0.07)   # Jan rung lifted to 0.07
    assert PR.derive_deadline(u[0], u, "2028-06-01", today)["p"] is None                   # beyond 3x: no rescale


def test_bounds_intersect_and_flag_narrow():
    rel = [{"id": "any_strike_iran", "relation": "E_implies_M"}, {"id": "tehran_or_bigger", "relation": "M_implies_E"},
           {"id": "peace_deal", "relation": "exclusive"}, {"id": "oil", "relation": "correlated"}]
    b = PR.bounds(rel, {"any_strike_iran": 0.30, "tehran_or_bigger": 0.22, "peace_deal": 0.75, "oil": 0.5})
    assert (b["lo"], b["hi"]) == pytest.approx((0.22, 0.25)) and b["narrow"] and "oil" not in b["markets"]


def test_longshot_correction_only_inside_its_bins(monkeypatch):
    monkeypatch.setattr(PR, "_longshot", lambda: {"all": {"a": -0.17, "b": 1.13}, "by_venue": {}, "apply_in": [[0.02, 0.10]]})
    assert PR.corrected(0.05)["applied"] and PR.corrected(0.05)["corrected"] < 0.05
    assert PR.corrected(0.5) == {"raw": 0.5, "corrected": 0.5, "applied": False}
