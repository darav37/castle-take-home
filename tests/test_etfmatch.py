"""ETF matching engine (SPEC §5a), offline: a synthetic market where the right answer is known."""

import numpy as np
import pandas as pd
import pytest

from worsey import etfmatch as em


def world(n=em.TRAIN_DAYS + em.TEST_DAYS, seed=3):
    """Target = 0.6·SEMI + 0.4·TECH + small noise. Plus look-alikes, a broad fund, a bond fund, and a decoy."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.010, n)
    semi = mkt * 1.4 + rng.normal(0, 0.012, n)
    tech = mkt * 1.1 + rng.normal(0, 0.006, n)
    R = pd.DataFrame({
        "SEMI": semi, "TECH": tech, "SPY": mkt, "BOND": rng.normal(0.0001, 0.004, n),
        "SEMI2": semi + rng.normal(0, 0.004, n),      # a noisier copy of SEMI
        "OIL": rng.normal(0, 0.02, n),
    }, index=idx)
    target = 0.6 * R["SEMI"] + 0.4 * R["TECH"] + rng.normal(0, 0.002, n)
    cash = pd.Series(em.FLAT_RF, index=idx)
    return target, R, cash


def test_classifier_flags_leverage_inverse_single_stock_and_kinds():
    c = em.classify_name
    assert c("Direxion Daily Semiconductor Bull 3X Shares")["leveraged"]
    us = c("ProShares UltraShort Semiconductors")
    assert us["leveraged"] and us["inverse"]                       # seen live Oct 17: "UltraShort" slipped through
    assert c("ProShares Ultra QQQ")["leveraged"] and not c("iShares Ultra Short-Term Bond ETF")["leveraged"]
    assert c("ProShares Short S&P500")["inverse"] and not c("iShares Short Treasury Bond ETF")["inverse"]
    assert c("YieldMax NVDA Option Income Strategy ETF")["kind"] == "single_stock"
    assert c("VanEck Semiconductor ETF")["kind"] == "sector"
    assert c("iShares MSCI Japan ETF")["kind"] == "country"
    assert c("SPDR Gold Shares")["kind"] == "commodity" and c("VanEck Gold Miners ETF")["kind"] != "commodity"
    assert c("Vanguard Total Stock Market ETF")["kind"] == "broad"
    assert c("iShares 20+ Year Treasury Bond ETF")["kind"] == "bond"


def test_holdings_refine_kind():
    base = em.classify_name("Some Fund ETF")
    one = pd.DataFrame({"ticker": ["NVDA", "CASH"], "weight": [0.95, 0.05], "asset_class": ["Equity", "Equity"]})
    assert em.classify_holdings(base, one)["kind"] == "single_stock"
    bonds = pd.DataFrame({"ticker": ["A 5% 2030", "B 4% 2031"], "weight": [0.5, 0.5], "asset_class": ["Fixed Income"] * 2})
    assert em.classify_holdings(base, bonds)["kind"] == "bond"
    swaps = pd.DataFrame({"ticker": ["X", "Y"], "weight": [0.8, -2.0], "name": ["DJ US SEMICONDUCTORS INDEX SWAP", "CASH"],
                          "asset_class": ["Other", "Cash"]})
    info = em.classify_holdings(base, swaps)
    assert info["leveraged"] and info["inverse"] and not em.permitted(info, "SSG", em.Restrictions(allow_short=True))[0]


def test_restrictions_permit_and_never_allow_leverage():
    r = em.Restrictions(allowed=["broad"])
    assert em.permitted({"kind": "broad"}, "SPY", r)[0]
    assert not em.permitted({"kind": "sector"}, "SMH", r)[0]
    assert not em.permitted({"kind": "broad", "leveraged": True}, "SSO", em.Restrictions())[0]
    assert not em.permitted({"kind": "broad"}, "SPY", em.Restrictions(approved_list=["VTI"]))[0]
    with pytest.raises(ValueError):
        em.Restrictions(allowed=["sector", "crypto"])


def test_fit_long_only_sums_to_one_with_cash():
    y, R, cash = world()
    w, c = em.fit(y.values, R[["SEMI", "TECH"]].values, cash.values, em.Restrictions())
    assert (w >= 0).all() and abs(w.sum() + c - 1) < 1e-6
    assert w[0] == pytest.approx(0.6, abs=0.05) and w[1] == pytest.approx(0.4, abs=0.05)


def test_fit_with_shorts_respects_the_gross_cap():
    y, R, cash = world()
    y2 = R["SEMI"] - 0.5 * R["SPY"]          # needs a short to match
    r = em.Restrictions(allow_short=True, gross_cap=1.5)
    w, c = em.fit(y2.values, R[["SEMI", "SPY"]].values, cash.values, r)
    assert w[1] < -0.2 and np.abs(w).sum() <= 1.5 + 1e-6
    w_lo, _ = em.fit(y2.values, R[["SEMI", "SPY"]].values, cash.values, em.Restrictions())
    assert (w_lo >= 0).all()


def test_select_finds_the_true_components_and_stops_small():
    y, R, cash = world()
    sel = em.select(y, R, cash, em.Restrictions(), k=5)
    assert set(sel["tickers"]) >= {"SEMI", "TECH"} and len(sel["tickers"]) <= 3   # extra ETFs barely help → stops
    assert sel["train_te"] < 0.06


def test_evaluate_reports_oos_with_baselines_and_beats_spy():
    y, R, cash = world()
    out = em.evaluate(y, R, cash, em.Restrictions(), k=3, holdings_weights={"SEMI": 0.3, "SPY": 0.1})
    m = out["test_metrics"]
    assert {"replica_buy_and_hold", "replica_daily_rebalanced", "best_single_etf", "spy_alone", "holdings_based"} <= set(m)
    assert m["replica_buy_and_hold"]["tracking_error"] < m["spy_alone"]["tracking_error"]
    assert m["replica_buy_and_hold"]["r2"] > 0.9
    assert out["train"][1] < out["test"][0]          # the test window is unseen


def test_stability_and_lookthrough_and_earnings_capture():
    y, R, cash = world()
    sel = em.select(y, R, cash, em.Restrictions(), k=3)
    st = em.stability(y, R, cash, em.Restrictions(), 3, sel["tickers"])
    assert st["stable"]
    holdings = {"SEMI": pd.DataFrame({"ticker": ["NVDA", "AMD"], "weight": [0.2, 0.1]}),
                "TECH": pd.DataFrame({"ticker": ["AAPL", "NVDA"], "weight": [0.15, 0.1]})}
    lt = em.lookthrough({"SEMI": 0.6, "TECH": 0.4}, {"NVDA": 1.0}, holdings)
    assert lt["replica_weight_in_target"] == pytest.approx(0.6 * 0.2 + 0.4 * 0.1)
    assert lt["top_other_names"][0]["name"] == "AMD" or lt["top_other_names"][0]["name"] == "AAPL"
    d = y.index
    past = [{"date": str(d[100].date()), "move": 0.08, "from": str(d[99].date()), "to": str(d[100].date())}]
    cap = em.earnings_capture(y, y * 0.5, past)
    assert cap["reports"] and cap["median_captured"] == pytest.approx(0.5 * float(y.loc[d[100]]) / 0.08)


def test_total_returns_add_dividends_on_ex_dates():
    c = pd.Series([100.0, 100.0, 100.0], index=pd.bdate_range("2026-01-05", periods=3))
    r = em.total_returns(c, [{"ex_dividend_date": "2026-01-06", "cash_amount": 1.0}])
    assert r.iloc[1] == pytest.approx(0.01) and r.iloc[2] == pytest.approx(0.0)


def test_panel_builds_incrementally_and_remembers_holidays(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "PANEL_FILE", tmp_path / "panel.pkl")
    calls = []

    def fetch(day):
        calls.append(day)
        return [] if day.endswith("-07") else [{"T": "SPY", "c": 100.0 + len(calls), "v": 1}]
    from datetime import date
    p1 = em.load_panel(today=date(2026, 1, 12), years=0.03, fetch=fetch, log=lambda *_: None)
    n = len(calls)
    p2 = em.load_panel(today=date(2026, 1, 12), years=0.03, fetch=fetch, log=lambda *_: None)
    assert len(calls) == n and len(p2["closes"]) == len(p1["closes"]) > 0   # second load: no refetch, holidays remembered
