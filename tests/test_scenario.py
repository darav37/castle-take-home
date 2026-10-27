"""Scenario engine on synthetic data with known answers."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from worsey.scenario import discovery, impact, joint, matching, probability

rng = np.random.default_rng(42)
DAYS = pd.bdate_range("2024-01-01", periods=400)


def _synthetic(true_beta=0.05, noise=0.01):
    # p random-walks with a few jump days, like real event markets
    steps = rng.normal(0, 0.005, len(DAYS))
    steps[rng.choice(len(DAYS), 8, replace=False)] += rng.choice([-0.15, 0.15], 8)
    p = pd.Series(np.clip(0.4 + np.cumsum(steps), 0.01, 0.99), index=DAYS)
    dp = p.diff()
    rets = pd.DataFrame({
        "SPY": true_beta * dp + rng.normal(0, noise, len(DAYS)),
        "TLT": -0.5 * true_beta * dp + rng.normal(0, noise, len(DAYS)),
        "GLD": rng.normal(0, noise, len(DAYS)),  # no exposure
    }, index=DAYS)
    return p, dp, rets


def test_align_dp_rolls_weekends_into_monday():
    p = pd.Series([0.30, 0.40, 0.50, 0.55], index=pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]))
    dp = impact.align_dp(p, pd.DatetimeIndex(pd.to_datetime(["2025-01-03", "2025-01-06"])))
    assert dp.iloc[1] == pytest.approx(0.25)  # Fri→Mon includes the weekend moves


def test_empirical_betas_recover_truth_and_shrink_noise():
    # Finding: at β = 0.05 with 1% daily noise, 400 days gives t ≈ 1.5. Tier 1 only pins down
    # large betas (Liberation-Day-sized, ~0.2), which is why tiers 2–3 exist.
    p, dp, rets = _synthetic(true_beta=0.2)
    b = impact.empirical_betas(rets, dp)
    assert b["SPY"].beta == pytest.approx(0.2, abs=0.05)
    assert b["TLT"].beta == pytest.approx(-0.1, abs=0.05)
    assert abs(b["GLD"].shrunk) < abs(b["GLD"].beta) + 1e-12
    assert abs(b["SPY"].shrunk / b["SPY"].beta) > 0.8  # strong signal barely shrunk


def test_resolution_pnl_is_priced_in():
    r = impact.resolution_pnl(-10_000, 0.3)
    assert r["pnl_yes"] == pytest.approx(-7_000) and r["pnl_no"] == pytest.approx(3_000)
    assert 0.3 * r["pnl_yes"] + 0.7 * r["pnl_no"] == pytest.approx(0)  # expected resolution P&L ≈ 0
    assert r["sigma"] == pytest.approx(10_000 * math.sqrt(0.21))


def test_event_rows_rank_by_sigma_and_split_asset_classes():
    values = pd.Series({"SPY": 60_000.0, "TLT": 40_000.0})
    ac = pd.Series({"SPY": "equity", "TLT": "rates"})
    r1 = impact.event_row("tariffs", 0.5, values, {"SPY": -0.05, "TLT": 0.02}, ac, beta_tier="1a")
    r2 = impact.event_row("fed_cut", 0.95, values, {"SPY": 0.05}, ac, beta_tier="1a")
    assert r1["pnl_yes_by_class"]["equity"] == pytest.approx(-1500) and r1["pnl_yes_by_class"]["rates"] == pytest.approx(400)
    # METHODOLOGY §13 contract
    for k in ("event", "schema", "market", "p", "p_confidence", "B", "beta_tier", "beta_se", "pnl_yes", "pnl_no",
              "pnl_yes_by_class", "sigma", "resolves_in_horizon", "drivers", "theme", "flags"):
        assert k in r1
    assert r1["B"] == pytest.approx(-2200) and r1["drivers"][0] == {"position": "SPY", "contrib": pytest.approx(-1500)}
    ranked = impact.rank_events([r2, r1])
    assert ranked.loc[0, "event_id"] == "tariffs"  # p=0.95 is mostly priced in


def test_portfolio_regression_matches_sum():
    p, dp, rets = _synthetic()
    values = pd.Series({"SPY": 50_000.0, "TLT": 50_000.0})
    b = impact.empirical_betas(rets, dp)
    chk = impact.portfolio_regression_check(rets, values, dp)
    assert chk["B"] == pytest.approx(sum(values[k] * b[k].beta for k in values.index), rel=1e-6)


def test_factor_loadings_and_shock_channels():
    f = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=["SPY", "TLT", "USO"])
    asset = 1.2 * f["SPY"] - 0.3 * f["TLT"] + rng.normal(0, 0.002, 500)
    L = impact.factor_loadings(pd.DataFrame({"XLE": asset}), f)
    assert L.at["XLE", "SPY"] == pytest.approx(1.2, abs=0.05)
    L.loc["SPY"] = {"SPY": 1.0, "TLT": 0.0, "USO": 0.0}
    out = impact.shock_impact(L, {"SPY": -0.02, "TLT": 0.01})
    assert out.at["XLE", "total"] == pytest.approx(1.2 * -0.02 - 0.3 * 0.01, abs=0.002)
    assert out.at["SPY", "total"] == pytest.approx(-0.02)


def test_window_moves_with_credit():
    idx = pd.bdate_range("2025-03-31", periods=5)
    closes = pd.DataFrame({"SPY": [100, 100, 90, 88, 89], "HYG": [80, 80, 78, 78, 78], "IEF": [95, 95, 96, 96, 96]}, index=idx)
    mv = impact.window_moves(closes, "2025-04-02", before=1, after=2)  # 04-01 close → 04-03 close
    assert mv["SPY"] == pytest.approx(-0.12)
    assert mv["CREDIT"] == pytest.approx((78 / 80 - 1) - (96 / 95 - 1))


def test_template_fit_and_loo():
    s = pd.Series([0.6, 0.3, 0.9, 0.5], index=list("abcd"))
    moves = pd.DataFrame({"SPY": -0.04 * s + rng.normal(0, 0.002, 4), "GLD": 0.01 * s}, index=s.index)
    t = impact.fit_template(s, moves)
    assert t.at["SPY", "b"] == pytest.approx(-0.04, abs=0.01)
    loo = impact.template_loo(s, moves)
    assert loo["GLD"].abs().max() < 1e-12  # noiseless factor predicted exactly
    shrunk = impact.fit_template(s.iloc[:1], moves.iloc[:1], prior_b=pd.Series({"SPY": 0.0, "GLD": 0.0}), prior_sd=0.001)
    assert abs(shrunk.at["SPY", "b"]) < 0.005  # one instance + tight prior → stays near prior


def test_prior_and_blend():
    pr = impact.prior_beta(-1, "medium")
    assert pr.beta == -0.03 and pr.se == pytest.approx(0.03) and pr.tier == 3  # ruling C1: 0.01/0.03/0.08, sd = 100%
    assert impact.prior_beta(1, "large").beta == 0.08 and impact.prior_beta(0, "none").se > 0
    b, se = impact.inverse_variance_blend(-0.03, 0.03, -0.06, 0.01)
    assert -0.06 < b < -0.03 and se < 0.01


def test_probability_helpers():
    assert probability.market_prob(0.40, 0.44, 0.5)["p"] == pytest.approx(0.42)
    assert probability.market_prob(0.20, 0.50, 0.33)["source"] == "last"
    assert probability.rescale_time(0.5, 90, 45) == pytest.approx(1 - 0.5 ** 0.5)
    fixed = probability.enforce_ladder([10, 15, 20], [0.6, 0.65, 0.2])
    assert fixed[0] >= fixed[1] >= fixed[2]
    r = probability.ladder_interp([10, 20], [0.6, 0.2], 15)
    assert r["p"] == pytest.approx(0.4)
    assert probability.ladder_interp([10, 20], [0.6, 0.2], 25)["p"] is None
    assert probability.combine([{"p": 0.4, "liquidity": 3}, {"p": 0.6, "liquidity": 1}]) == pytest.approx(0.45)


def test_matching_rules():
    today = date(2026, 10, 14)
    want = matching.EventSchema("us_avg_tariff_rate", ">=", 25, "%", date(2027, 3, 31), "US")
    exact = matching.EventSchema("us_avg_tariff_rate", ">=", 25, "%", date(2027, 3, 31), "US")
    assert matching.score_match(want, exact, today=today)["decision"] == "exact"
    ladder = matching.EventSchema("us_avg_tariff_rate", ">=", 20, "%", date(2027, 3, 31), "US",
                                  ladder=[{"threshold": 20, "p": .5}, {"threshold": 30, "p": .2}])
    assert matching.score_match(want, ladder, today=today)["decision"] == "derived_strike"
    sooner = matching.EventSchema("us_avg_tariff_rate", ">=", 25, "%", date(2027, 1, 15), "US")
    assert matching.score_match(want, sooner, today=today)["decision"] == "derived_time"
    far = matching.EventSchema("us_avg_tariff_rate", ">=", 25, "%", date(2026, 10, 31), "US")
    assert matching.score_match(want, far, today=today)["decision"] == "no_fit"
    other = matching.EventSchema("eu_avg_tariff_rate", ">=", 25, "%", date(2027, 3, 31), "EU")
    assert matching.score_match(want, other, today=today)["decision"] == "no_fit"


def test_keyword_candidates():
    mk = [{"id": 1, "title": "Average US tariff rate above 25% by March"}, {"id": 2, "title": "Fed cuts in December"}]
    c = matching.keyword_candidates("tariffs of 25% or more by March 31 tariff rate", mk)
    assert c[0]["id"] == 1 and all(x["id"] != 2 for x in c)


def test_discovery_filter_category_horizon_and_merge():
    today = date(2026, 10, 14)
    mk = [
        {"id": "A", "close_date": "2027-01-01", "open_date": "2026-01-01", "volume": 5000, "category": "Economics"},
        {"id": "B", "close_date": "2026-10-20", "open_date": "2026-10-01", "volume": 5000, "category": "Economics"},  # too little history
        {"id": "C", "close_date": "2028-06-01", "open_date": "2026-01-01", "volume": 5000, "category": "Economics"},  # > 12 months
        {"id": "D", "close_date": "2026-12-01", "open_date": "2026-01-01", "volume": 5000},                         # no category → out
        {"id": "E", "close_date": "2026-12-01", "open_date": "2026-01-01", "volume": 5000, "category": "Sports"},  # out of scope
        {"id": "F", "close_date": "2026-12-01", "open_date": "2026-01-01", "volume": 5000, "category": "Politics"},
    ]
    kept = {m["id"]: m for m in discovery.bottom_up_filter(mk, today, horizon_days=60)}
    assert set(kept) == {"A", "F"}
    assert kept["F"]["resolves_in_horizon"] and not kept["A"]["resolves_in_horizon"]  # A becomes a drift row
    merged = discovery.merge([kept["A"]], [{"name": "x", "matched_market_id": "A"}, {"name": "quake", "matched_market_id": None}])
    assert merged["priced"][0]["sources"] == ["bottom_up", "top_down"] and merged["unpriced"][0]["status"] == "unpriced"


def test_tier1_eligibility_needs_history_and_jumps():
    idx = pd.bdate_range("2025-01-01", periods=80)
    calm = pd.Series(0.001, index=idx)
    assert not discovery.tier1_eligible(calm)["eligible"]
    jumpy = calm.copy()
    jumpy.iloc[[5, 15, 25, 35, 45]] = 0.06
    r = discovery.tier1_eligible(jumpy)
    assert r["eligible"] and r["jump_days"] == 5
    short = jumpy.iloc[:40]
    assert "usable days" in discovery.tier1_eligible(short)["reason"]


def test_drift_sigma_scales_with_horizon():
    from worsey.scenario.pipeline import drift_sigma
    dp = pd.Series(np.random.default_rng(3).normal(0, 0.01, 300))
    assert drift_sigma(dp, 365) == pytest.approx(2 * drift_sigma(dp, 91.25), rel=1e-6)


def test_copula_marginals_tail_euler_and_conditioning():
    p = pd.Series({"a": 0.3, "b": 0.6})
    corr = pd.DataFrame([[1, 0.8], [0.8, 1]], index=p.index, columns=p.index)
    beta = pd.Series({"a": -10_000.0, "b": -5_000.0})
    sim = joint.simulate(p, corr, beta, n=40_000)
    assert sim["joint_prob"]["a"] == pytest.approx(0.3, abs=0.01)
    assert sim["mean"] == pytest.approx(0, abs=150)
    assert sim["worst_combo"] == {"a": 1, "b": 1}
    assert sum(sim["euler_es"].values()) == pytest.approx(sim["es"], rel=1e-9)  # Euler contributions add up to ES
    assert sim["es"] <= sim["var"]
    cond = joint.simulate(p, corr, beta, n=40_000, condition={"a": 1})
    assert cond["joint_prob"]["b"] > 0.8  # strongly correlated: b likely given a
    assert joint.themes(corr) == [["a", "b"]]


def test_structural_ladders_and_exclusive_groups():
    # ladder: tariff ≥20 / ≥25 / ≥30 share one latent → nested in every draw
    p = pd.Series({"t20": 0.6, "t25": 0.4, "t30": 0.2, "hold": 0.5, "cut": 0.4, "hike": 0.1})
    corr = pd.DataFrame(np.eye(6), index=p.index, columns=p.index)
    beta = pd.Series(-1000.0, index=p.index)
    sim = joint.simulate(p, corr, beta, n=20_000, ladders=[["t20", "t25", "t30"]], exclusive=[["hold", "cut", "hike"]])
    jp = sim["joint_prob"]
    for k in p.index:
        assert jp[k] == pytest.approx(p[k], abs=0.015)  # marginals still match the market
    cond = joint.simulate(p, corr, beta, n=20_000, ladders=[["t20", "t25", "t30"]], exclusive=[["hold", "cut", "hike"]],
                          condition={"t30": 1})
    assert cond["joint_prob"]["t25"] == 1.0 and cond["joint_prob"]["t20"] == 1.0  # ≥30 ⇒ ≥25 ⇒ ≥20
    c2 = joint.simulate(p, corr, beta, n=20_000, exclusive=[["hold", "cut", "hike"]], condition={"cut": 1})
    assert c2["joint_prob"]["hold"] == 0.0 and c2["joint_prob"]["hike"] == 0.0  # exactly one outcome


def test_logit_correlation_ledoit_wolf_and_odds_move():
    rng3 = np.random.default_rng(5)
    n = 300
    common = rng3.normal(0, 0.15, n)
    x = pd.DataFrame({"a": np.cumsum(common + rng3.normal(0, 0.05, n)), "b": np.cumsum(common + rng3.normal(0, 0.05, n)),
                      "c": np.cumsum(rng3.normal(0, 0.15, n))})
    pf = 1 / (1 + np.exp(-x))
    dep = joint.correlation(pf)
    assert dep["method"] == "ledoit_wolf" and 0 <= dep["delta"] < 0.5
    assert dep["R"].at["a", "b"] > 0.7 and abs(dep["R"].at["a", "c"]) < 0.3
    p_now = pf.iloc[-1]
    mv = joint.odds_move(p_now, dep["cov_dlogit"], "a", min(0.95, p_now["a"] + 0.2), pd.Series({"a": -1000.0, "b": -1000.0, "c": -1000.0}))
    assert mv["dp"]["b"] > 0 and abs(mv["dp"]["c"]) < mv["dp"]["b"]  # b moves with a, c barely


def test_align_dp_masks_stale_days_instead_of_filling():
    days = pd.DatetimeIndex(pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]))
    p = pd.Series([0.30, float("nan"), 0.50, 0.55], index=days)  # the 7th is stale
    dp = impact.align_dp(p, days)
    assert pd.isna(dp.iloc[1])                       # stale day masked
    assert dp.iloc[2] == pytest.approx(0.20)         # 8th measured from the last good print (6th)
    assert dp.iloc[3] == pytest.approx(0.05)


def test_hac_vs_hc1_and_ridge_loadings():
    # AR(1) errors: HAC se should exceed HC1 se for a persistent regressor
    rng2 = np.random.default_rng(11)
    n = 600
    x = pd.Series(np.cumsum(rng2.normal(0, 1, n)) * 0.01)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng2.normal(0, 0.01)
    y = 0.5 * x + e
    _, se_hac, _ = impact.ols_beta(y, x, cov="HAC")
    _, se_hc1, _ = impact.ols_beta(y, x, cov="HC1")
    assert se_hac > se_hc1
    # ridge loadings: close to OLS on clean data, window respected, se attached
    f = pd.DataFrame(rng2.normal(0, 0.01, (900, 2)), columns=["SPY", "TLT"])
    a = 1.1 * f["SPY"] - 0.4 * f["TLT"] + rng2.normal(0, 0.002, 900)
    L = impact.factor_loadings(pd.DataFrame({"A": a}), f, window=504)
    assert L.at["A", "SPY"] == pytest.approx(1.1, abs=0.05) and L.at["A", "TLT"] == pytest.approx(-0.4, abs=0.05)
    assert set(L.attrs["se"].columns) == {"SPY", "TLT"}


def test_meeting_count_rescaling_for_release_events():
    today = date(2026, 10, 14)
    fomc = [date(2026, 10, 28), date(2026, 12, 9), date(2027, 1, 27), date(2027, 3, 17)]
    want = matching.EventSchema("fed_cut", ">=", 1, "", date(2026, 12, 31), "US", timing="release", decision_dates=fomc)
    cand = matching.EventSchema("fed_cut", ">=", 1, "", date(2027, 3, 31), "US", timing="release", decision_dates=fomc)
    r = matching.score_match(want, cand, today=today)
    assert r["decision"] == "derived_time" and r["time_rule"] == {"kind": "meetings", "n": 4, "n_new": 2}
    p2 = probability.apply_time_rule(0.6, r["time_rule"])
    assert p2 == pytest.approx(1 - (1 - 0.6) ** (2 / 4))
    # calendar-hazard would have used 78/168 days instead: a different answer
    assert p2 != pytest.approx(probability.rescale_time(0.6, 168, 78))


def test_cross_venue_flag():
    out = probability.combine_venues([{"venue": "kalshi", "p": 0.40, "liquidity": 3}, {"venue": "polymarket", "p": 0.48, "liquidity": 1}])
    assert out["flags"] == ["venues_disagree"] and out["gap"] == pytest.approx(0.08) and out["p"] == pytest.approx(0.42)
    assert probability.combine_venues([{"venue": "kalshi", "p": 0.40, "liquidity": 1}, {"venue": "polymarket", "p": 0.43, "liquidity": 1}])["flags"] == []


def test_tier2_betas_and_pipeline_blend_hook():
    L = pd.DataFrame({"SPY": [1.0, 1.2], "TLT": [0.0, -0.3]}, index=["SPY", "QQQ"])
    L.attrs["se"] = pd.DataFrame({"SPY": [0.0, 0.05], "TLT": [0.0, 0.05]}, index=["SPY", "QQQ"])
    t2 = impact.tier2_betas(L, {"SPY": -0.05, "TLT": 0.02}, {"SPY": 0.02, "TLT": 0.01})
    assert t2["QQQ"][0] == pytest.approx(1.2 * -0.05 - 0.3 * 0.02)
    assert t2["QQQ"][1] == pytest.approx(((1.2 * 0.02) ** 2 + (0.3 * 0.01) ** 2 + (0.05 * 0.05) ** 2 + (0.02 * 0.05) ** 2) ** 0.5)
    from worsey.data.provider import SyntheticMarketData
    from worsey.portfolio import Portfolio
    from worsey.scenario.pipeline import REFERENCE, scan
    md = SyntheticMarketData()
    pr = Portfolio.load("balanced").priced(md.prices(Portfolio.load("balanced").tickers))
    mk = md.event_markets()
    hist = {m["id"]: md.market_history(m["id"]) for m in mk}
    cl = md.closes(list(pr.index) + REFERENCE)
    Lall = pd.DataFrame(0.0, index=cl.columns, columns=["SPY"]); Lall["SPY"] = 1.0
    Lall.attrs["se"] = pd.DataFrame(0.0, index=cl.columns, columns=["SPY"])
    out = scan(pr["value"], pr["asset_class"], cl, mk, hist, md.today(), horizon_days=200,
               templates={"TARIFF-25-MAR": {"shock": {"SPY": -0.2}, "shock_se": {"SPY": 0.05}}}, loadings=Lall)
    rows = {e["event_id"]: e for e in out["events"]}
    # a template is the event's structural prior, not a blend label
    assert rows["TARIFF-25-MAR"]["prior_tier"] == "2" and rows["FED-CUT-DEC"]["prior_tier"] is None
    held_proxies_unshocked = float(pr.loc[["TLT", "GLD"], "value"].sum())  # factor proxies take their own (zero) shock
    assert rows["TARIFF-25-MAR"]["B_prior"] == pytest.approx(-0.2 * (float(pr["value"].sum()) - held_proxies_unshocked), rel=1e-6)
    assert rows["TARIFF-25-MAR"]["above_divider"]  # tier-2 mechanism with material exposure (or strong evidence)


def _mk(i, ev, st, floor=None, cap=None, bid=0.4, ask=0.42, excl=False):
    return {"id": i, "event_ticker": ev, "strike_type": st, "floor_strike": floor, "cap_strike": cap,
            "bid": bid, "ask": ask, "last": None, "mutually_exclusive": excl}


def test_ladder_only_for_one_numeric_variable_with_monotone_prices():
    from worsey.scenario.pipeline import structural_groups
    ladder = [_mk("t20", "TAR", "greater", 20, bid=0.60, ask=0.62), _mk("t30", "TAR", "greater", 30, bid=0.20, ask=0.22),
              _mk("t25", "TAR", "greater", 25, bid=0.40, ask=0.42)]
    countries = [_mk("FR", "WHO", "custom"), _mk("DE", "WHO", "custom")]               # "which countries…"
    either = [_mk("A", "AB", None), _mk("B", "AB", None)]                                # "will A / will B"
    broken = [_mk("c3", "CPI", "greater", 3.0, bid=0.30, ask=0.32), _mk("c4", "CPI", "greater", 4.0, bid=0.50, ask=0.52)]
    fed = [_mk("cut", "FED", "custom", excl=True), _mk("hold", "FED", "custom", excl=True)]
    ladders, exclusive, log = structural_groups(ladder + countries + either + broken + fed)
    assert ladders == [["t20", "t25", "t30"]]  # ordered by threshold
    assert exclusive == [["cut", "hold"]]
    how = {x["event"]: x["as"] for x in log}
    assert how == {"TAR": "ladder", "WHO": "separate", "AB": "separate", "CPI": "separate", "FED": "exclusive"}
    assert "not monotone" in next(x["why"] for x in log if x["event"] == "CPI")


def test_fallback_correlation_is_tail_conservative():
    R = pd.DataFrame([[1.0, np.nan, np.nan], [np.nan, 1.0, 0.2], [np.nan, 0.2, 1.0]], index=list("abc"), columns=list("abc"))
    B = pd.Series({"a": -5000.0, "b": -3000.0, "c": 2000.0})
    R2, filled = joint.fill_missing(R, B)
    assert R2.at["a", "b"] == pytest.approx(0.5, abs=0.02)   # same-sign exposures: positive, never 0
    assert R2.at["a", "c"] == pytest.approx(-0.5, abs=0.02)  # opposite-sign: also adds variance
    assert {tuple(f["pair"]) for f in filled} == {("a", "b"), ("a", "c")}
    assert np.all(np.linalg.eigvalsh(R2.to_numpy()) > -1e-9)


def test_small_template_is_low_confidence_with_wide_bands():
    s1 = pd.Series([0.8, 0.5], index=["x", "y"])
    mv = pd.DataFrame({"SPY": [-0.04, -0.02]}, index=s1.index)
    t = impact.fit_template(s1, mv)
    assert t.at["SPY", "confidence"] == "low" and t.at["SPY", "band_hi"] - t.at["SPY", "band_lo"] > abs(t.at["SPY", "b"])


def test_reflexive_price_markets_and_diversified_cap():
    assert discovery.is_reflexive({"title": "Will the S&P 500 be above 6845.5 on Dec 31, 2026 at 4pm EST?"})
    assert discovery.is_reflexive({"title": "Will Bitcoin be below $55000.00 by Jan 1, 2027?"})
    assert discovery.is_reflexive({"title": "Will the maximum WTI front month settle price reach $115.01?"})
    assert discovery.is_reflexive({"title": "Will the maximum SP500 value reach 7999.99 by Jan 1, 2027?"})
    assert not discovery.is_reflexive({"title": "Will the Federal Reserve cut rates before 2027?"})
    assert not discovery.is_reflexive({"title": "Will the CIT strike down Trump's Section 301 tariffs in 2026?"})
    mk = [{"id": f"c{i}", "category": "Crypto", "volume": 1e6 - i} for i in range(10)] + \
         [{"id": "t1", "category": "Economics", "volume": 5e3}, {"id": "p1", "category": "Politics", "volume": 7e3}]
    cap = [m["id"] for m in discovery.diversified_cap(mk, 4)]
    assert cap[:3] == ["c0", "p1", "t1"] or set(cap[:3]) == {"c0", "p1", "t1"}
    assert "t1" in cap and "p1" in cap


def test_empirical_bayes_shrinks_toward_the_structural_prior():
    from worsey.scenario import structural
    rng4 = np.random.default_rng(9)
    rows = []
    for i in range(200):  # true B = prior; estimates are pure noise around it
        prior = float(rng4.normal(0, 1000))
        rows.append({"B_hat": prior + float(rng4.normal(0, 1500)), "se": 1500.0, "B_prior": prior, "s_prior": 0.0, "p": 0.5})
    eb = structural.empirical_bayes(rows)
    assert eb["tau2"] < 0.15 * 1500 ** 2               # no signal beyond the prior (up to sampling noise)
    assert all(abs(r["B_post"] - r["B_prior"]) < 0.2 * abs(r["B_hat"] - r["B_prior"]) + 1e-9 for r in rows)
    # with prior uncertainty, a weak estimate stays weak evidence (no degenerate sd = 0)
    r = {"B_hat": 3000.0, "se": 2500.0, "B_prior": 0.0, "s_prior": 0.0, "p": 0.06}
    structural.empirical_bayes([r])
    assert not structural.above_divider(r, 100_000)[0]
    strong = [{"B_hat": 8000.0 + k, "se": 1000.0, "B_prior": 0.0, "s_prior": 500.0, "p": 0.4} for k in range(5)] + \
             [{"B_hat": -8000.0, "se": 1000.0, "B_prior": 0.0, "s_prior": 500.0, "p": 0.4}]
    structural.empirical_bayes(strong)
    assert structural.above_divider(strong[0], 100_000)[0]


def test_book_prior_from_shock_map_and_loadings():
    from worsey.scenario import structural
    L = pd.DataFrame({f: [0.0, 0.0] for f in structural.FACTORS}, index=["QQQ", "XLE"])
    L.loc["QQQ", "SPY"], L.loc["XLE", "USO"] = 1.2, 0.5
    shock, sd = structural.shock_from_buckets({"SPY": [-1, "medium"], "USO": [1, "large"]})
    B, s_p, beta = structural.book_prior(pd.Series({"QQQ": 10_000.0, "XLE": 10_000.0, "SPY": 5_000.0}), L, shock, sd)
    assert beta["QQQ"] == pytest.approx(1.2 * -0.03) and beta["XLE"] == pytest.approx(0.5 * 0.08)
    assert B == pytest.approx(10_000 * -0.036 + 10_000 * 0.04 + 5_000 * -0.03) and s_p > 0


def test_pooled_key_and_book_beta():
    from worsey.scenario import structural
    assert structural.pooled_key({"id": "KXFEDDECISION-26DEC-H0", "series_ticker": "KXFEDDECISION"}) == "KXFEDDECISION:H0"
    assert structural.pooled_key({"id": "KXCPI-26OCT-T0.2", "series_ticker": "KXCPI"}) == "KXCPI:above"
    assert structural.pooled_key({"id": "KXMAYORLA-26-KBAS", "series_ticker": "KXMAYORLA"}) is None
    rng5 = np.random.default_rng(4)
    rel = np.repeat(np.arange(20), 3)
    surp = rng5.uniform(-0.6, 0.6, 60)
    obs = pd.DataFrame({"key": "KXCPI:above", "release": rel, "surprise": surp,
                        "SPY": -0.02 * surp + rng5.normal(0, 0.003, 60), "TLT": 0.01 * surp + rng5.normal(0, 0.003, 60)})
    out = structural.pooled_book_beta(obs, pd.Series({"SPY": 50_000.0, "TLT": 50_000.0}), "KXCPI:above")
    assert out["B"] == pytest.approx(50_000 * -0.02 + 50_000 * 0.01, abs=250) and out["releases"] == 20
