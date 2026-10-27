"""IV history backfill (synthetic Black–Scholes bars), the trainer's z-score signals, trade facts and alerts."""

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from worsey.options import builder, signals, vol_history
from worsey.options.pricer import bs_price
from worsey.options.view import View

TODAY = date(2026, 10, 16)
ROOT = Path(__file__).parents[1]


def smile(S, K, base=0.30, slope=0.4):
    """Put skew: vol rises as strike falls."""
    return base + slope * max(0.0, -math.log(K / S)) - 0.1 * max(0.0, math.log(K / S))


class FakeFetch:
    def __init__(self, spot: pd.Series, base: pd.Series):
        self.spot, self.base, self.calls = spot, base, 0

    def contracts(self, u, expiry, expired):
        return [{"ticker": f"O:{u}{expiry:%y%m%d}{k}{s:08d}", "strike": float(s), "kind": "call" if k == "C" else "put"}
                for s in range(50, 160, 5) for k in ("C", "P")]

    def closes(self, contract, start, end):
        self.calls += 1
        E = pd.Timestamp("20" + contract[-15:-9])
        kind = "call" if contract[-9] == "C" else "put"
        K = int(contract[-8:])
        out = {}
        for d, S in self.spot.items():
            if pd.Timestamp(start) <= d <= pd.Timestamp(end) and d < E:
                T = (E - d).days / 365
                out[d] = bs_price(S, K, T, 0.04, 0.0, smile(S, K, self.base[d]), kind)
        return pd.Series(out, dtype=float)


def _spot(days=400, seed=0):
    idx = pd.bdate_range(end=pd.Timestamp(TODAY), periods=days)
    r = np.random.default_rng(seed).normal(0, 0.012, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(r) - np.cumsum(r)[-1] * 0), index=idx)


def test_smile_points_recovers_skew():
    S, T = 100.0, 30 / 365
    q = pd.DataFrame([{"strike": k, "kind": "put" if k <= S else "call", "iv": smile(S, k)} for k in range(70, 131, 5)])
    p = vol_history.smile_points(S, T, q)
    assert abs(p["atm"] - 0.30) < 0.01
    assert p["put25"] > p["atm"] > p["call25"]


def test_constant_maturity_interpolates_total_variance():
    a, b = {"atm": 0.2, "put25": 0.25, "call25": 0.18}, {"atm": 0.3, "put25": 0.35, "call25": 0.28}
    cm = vol_history.constant_maturity([(60 / 365, b), (20 / 365, a)], 30)
    assert 0.2 < cm["atm"] < 0.3
    assert vol_history.constant_maturity([(20 / 365, a)], 30) == a      # flat within 15 days
    assert vol_history.constant_maturity([(60 / 365, a)], 30) is None   # never a long extrapolation


def test_backfill_from_synthetic_bars(tmp_path):
    spot = _spot()
    base = pd.Series(0.30, index=spot.index)
    f = FakeFetch(spot, base)
    h = vol_history.backfill("XYZ", spot, TODAY, f, days=200, strikes_per_kind=8, log=lambda *_: None)
    assert len(h) > 100 and (h.source == "backfill").all()
    assert abs(h.atm_front.median() - 0.30) < 0.02
    ok = h.dropna(subset=["put25_front", "call25_front"])
    assert len(ok) / len(h) > 0.95                                # most days get a 30d point
    assert (ok.put25_front > ok.call25_front).all()              # the put skew comes through
    vol_history.save("XYZ", h, tmp_path)
    assert len(vol_history.load("XYZ", tmp_path)) == len(h)


def test_signals_flag_cheap_iv_and_need_history():
    idx = pd.bdate_range(end=pd.Timestamp(TODAY), periods=260)
    rng = np.random.default_rng(1)
    h = pd.DataFrame({"spot": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 260))),
                      "atm_front": 0.30 + rng.normal(0, 0.02, 260), "atm_back": 0.29 + rng.normal(0, 0.01, 260),
                      "put25_front": 0.36 + rng.normal(0, 0.01, 260), "call25_front": 0.27 + rng.normal(0, 0.01, 260)},
                     index=idx)
    sig = signals.history_signals(h, {"spot": 100.0, "atm_front": 0.24, "atm_back": 0.29,
                                      "put25_front": 0.36, "call25_front": 0.27})
    assert sig["atm_iv_z"] < -2 and sig["history_days"] == 260
    assert sig["term_z"] < -1                                   # front well below back
    short = signals.history_signals(h.iloc[:30])
    assert short["atm_iv_z"] is None and "need" in short["note"]
    assert signals.history_signals(pd.DataFrame())["history_days"] == 0


def _quoter():
    from tests.test_options import quoter
    return quoter()


def test_alerts_and_facts_on_real_library():
    lib = builder.load_library(ROOT / "library" / "strategies")
    sig = {"atm_iv_z": -1.5, "call_skew_pct": 0.9, "call_skew": 0.04, "skew_rr_z": 1.3, "skew_rr": 0.08,
           "spot_vol_corr": -0.6, "term_z": 1.2, "term_spread": 0.03}
    up = View("NVDA", "up", TODAY + timedelta(days=21), target_move_pct=10)
    out = builder.propose(up, 100.0, TODAY, _quoter(), realized_vol=0.45, library=lib, signals=sig,
                          include=["short_risky"], implied_move=0.08)
    by = {c.id: c for c in out["candidates"]}
    cs = by["call_debit_spread"].metrics
    assert cs["facts"]["theta_per_day"] < 0 and cs["facts"]["max_payout_multiple"] > 1
    assert 0 < cs["facts"]["prob_past_short_strike"] < 1
    assert any(a["rule"] == "call_debit_spread#1" for a in cs["alerts"])        # upside skew rich
    sr = by["short_risky"].metrics
    assert len(sr["facts"]["stress"]) == 2 and sr["facts"]["stress"][1]["pnl"] < sr["facts"]["stress"][0]["pnl"] < 0
    assert {a["rule"] for a in sr["alerts"]} >= {"short_risky#3", "short_risky#4", "short_risky#6"}
    neutral = View("NVDA", "neutral", TODAY + timedelta(days=21))
    out = builder.propose(neutral, 100.0, TODAY, _quoter(), realized_vol=0.45, library=lib, signals=sig,
                          include=["call_calendar", "long_straddle"], implied_move=0.08)
    by = {c.id: c for c in out["candidates"]}
    assert any(a["rule"] == "long_straddle#3" for a in by["long_straddle"].metrics["alerts"])
    assert any(a["rule"] == "call_calendar#2" for a in by["call_calendar"].metrics["alerts"])


def test_live_library_fills_gaps_with_stand_ins():
    lib = builder.load_library_with_stand_ins()
    ids = {e["id"]: e for e in lib}
    assert "long_call" in ids and ids["long_call"].get("stand_in")             # The trainer hasn't written it
    assert not ids["call_debit_spread"].get("stand_in") and ids["call_debit_spread"]["_prose"]  # his file wins


def test_update_backfills_then_snapshots(tmp_path):
    spot = _spot()
    f = FakeFetch(spot, pd.Series(0.30, index=spot.index))

    class Data:
        def closes(self, tickers, days=750):
            return pd.DataFrame({tickers[0]: spot.iloc[-days:]})

        def chain(self, u, expiry_from=None, expiry_to=None, band=None):
            S, rows = float(spot.iloc[-1]), []
            for E in (pd.Timestamp(TODAY) + pd.Timedelta(days=d) for d in (21, 49, 112)):
                T = (E - pd.Timestamp(TODAY)).days / 365
                for k in range(60, 115, 5):
                    for kind in ("put", "call"):
                        px = bs_price(S, k, T, 0.04, 0.0, smile(S, k), kind)
                        rows.append({"expiry": E, "strike": float(k), "kind": kind, "mid": px, "bid": px, "ask": px, "iv": np.nan})
            return pd.DataFrame(rows)

    rep = vol_history.update(["XYZ"], Data(), TODAY, fetch=f, days=200, log=lambda *_: None, root=tmp_path)
    assert "error" not in rep["XYZ"], rep
    h = vol_history.load("XYZ", tmp_path)
    assert h.loc[pd.Timestamp(TODAY), "source"] == "snapshot" and (h.source == "backfill").any()
    n = f.calls
    vol_history.update(["XYZ"], Data(), TODAY, fetch=f, days=200, log=lambda *_: None, root=tmp_path)
    assert f.calls == n                                          # backfill runs once


def test_put_spread_protection_facts():
    idx = pd.bdate_range(end=pd.Timestamp(TODAY), periods=300)
    rng = np.random.default_rng(3)
    spy = rng.normal(0, 0.01, 300)
    closes = pd.DataFrame({"SPY": 100 * np.exp(np.cumsum(spy)),
                           "QQQ": 100 * np.exp(np.cumsum(1.3 * spy + rng.normal(0, 0.003, 300)))}, index=idx)
    book = signals.book_beta_notional({"SPY": 50_000.0, "QQQ": 50_000.0}, closes, "SPY")
    assert 1.2 < book["betas"]["QQQ"] < 1.4 and 110_000 < book["beta_notional"] < 120_000
    lib = builder.load_library(ROOT / "library" / "strategies")
    down = View("SPY", "down", TODAY + timedelta(days=30), target_move_pct=-8)
    out = builder.propose(down, 100.0, TODAY, _quoter(), realized_vol=0.2, library=lib, include=["put_debit_spread"])
    c = next(c for c in out["candidates"] if c.id == "put_debit_spread")
    assert any(a["rule"].startswith("put_debit_spread#2") for a in c.metrics["alerts"])
    hf = signals.hedge_facts(c.strategy, book)
    t20, t30 = hf["tail"]
    assert t20["book_hedged"] > t20["book_unhedged"]
    # below the short strike the spread stops paying: the extra 10% down is (almost) all unhedged
    assert abs((t30["book_hedged"] - t20["book_hedged"]) - (t30["book_unhedged"] - t20["book_unhedged"])) < 0.05 * abs(t30["book_unhedged"])
    assert hf["annualized_cost_pct_if_rolled"] > hf["premium_pct_of_book"] > 0
    al = signals.hedge_alerts(hf, 92.0, 97.0)
    assert {a["rule"] for a in al} == {"put_debit_spread#4", "put_debit_spread#6", "put_debit_spread#7", "put_debit_spread#8"}


def test_long_strangle_facts_and_position_alerts():
    lib = builder.load_library(ROOT / "library" / "strategies")
    v = View("NVDA", "neutral", TODAY + timedelta(days=21), vol_view="expand")
    out = builder.propose(v, 100.0, TODAY, _quoter(), realized_vol=0.45, library=lib, n=10, implied_move=0.08)
    by = {c.id: c for c in out["candidates"]}
    st, sd = by["long_strangle"].metrics, by["long_straddle"].metrics
    assert st["cost"] < sd["cost"]                                           # cheaper than the straddle
    assert 0.3 < st["facts"]["prob_both_otm"] < 0.8
    assert {a["rule"] for a in st["alerts"]} >= {"long_strangle#3", "long_strangle#4", "long_strangle#5"}
    pa = signals.strangle_position_alerts
    assert [a["step"] for a in pa(100, 123, 10, None, "kurtosis")] == [2]       # +23% → step 2
    assert pa(100, 123, 10, None, "kurtosis", last_alerted_step=2) == []        # once per step
    assert pa(100, 123, 10, None, "directional") == []                          # directional: no take-profit nudge
    assert pa(100, 105, 60, 1.7, None)[0]["rule"] == "long_strangle#7"          # longer-dated: vol signal
    assert pa(100, 105, 60, 1.2, None) == []
