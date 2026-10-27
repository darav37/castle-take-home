"""recent_markets_analyst's tape summary (DECISIONS #121): moves ranked against each asset's own volatility."""

from datetime import date

import numpy as np
import pandas as pd

from worsey import markets


class FakeData:
    def __init__(self, closes):
        self._c = closes

    def today(self):
        return date(2026, 10, 16)

    def closes(self, tickers, days=90):
        return self._c[[t for t in tickers if t in self._c.columns]].tail(days)

    def event_markets(self):
        return [{"id": "M1", "title": "Fed cut", "bid": 0.60, "ask": 0.62}]

    def market_history(self, mid):
        return pd.Series([0.40] * 10)


def test_a_quiet_asset_moving_2_percent_outranks_a_wild_one_moving_3():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2026-06-01", periods=90)
    quiet = 100 * np.cumprod(1 + rng.normal(0, 0.002, 90))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, 90))
    quiet[-1] = quiet[-6] * 1.02
    wild[-1] = wild[-6] * 1.03
    t = markets.tape_summary(FakeData(pd.DataFrame({"TLT": quiet, "SMH": wild}, index=idx)))
    assert t["biggest_moves"][0]["ticker"] == "TLT" and abs(t["biggest_moves"][0]["z_5d"]) > abs(t["biggest_moves"][1]["z_5d"])
    assert t["odds_shifts"][0]["shift"] == np.float64(0.61 - 0.40).round(10) or abs(t["odds_shifts"][0]["shift"] - 0.21) < 1e-9



def test_reference_search_finds_the_right_article_with_its_license():
    from worsey import knowledge
    if not knowledge.CORPORA["options"][0].exists():
        import pytest
        pytest.skip("corpus not built")
    hits = knowledge.search("put call parity relationship")
    assert hits and "parity" in hits[0]["title"].lower() and hits[0]["license"].startswith("CC BY-SA")
    gl = knowledge.search("contango futures price above spot")
    assert gl and any(h["source"] in ("CFTC", "Wikipedia") for h in gl)
