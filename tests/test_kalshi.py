"""Kalshi client parsing on real response shapes (tests/fixtures/live) + 16:00 ET sampling."""

import json
from pathlib import Path

import pandas as pd
import pytest

from worsey.data import kalshi

LIVE = Path(__file__).parent / "fixtures" / "live"


def test_parse_live_candles_dollar_strings():
    raw = json.loads((LIVE / "kalshi_candles.json").read_text(encoding="utf-8"))
    df = kalshi.parse_candles([c for c in raw["candlesticks"] if isinstance(c, dict)])
    assert df.iloc[0]["trade"] == pytest.approx(0.72) and df.iloc[0]["bid"] == pytest.approx(0.68)
    assert pd.isna(df.iloc[1]["trade"]) and df.iloc[1]["volume"] == 0  # no trade that period


def test_parse_historical_candle_keys():
    c = {"end_period_ts": 1781841600, "price": {"close": "0.0100"}, "yes_bid": {"close": "0.0000"},
         "yes_ask": {"close": "0.0200"}, "volume": "3"}
    df = kalshi.parse_candles([c])
    assert df.iloc[0]["trade"] == pytest.approx(0.01) and df.iloc[0]["ask"] == pytest.approx(0.02)


def test_normalize_market_from_live_fixture():
    raw = json.loads((LIVE / "kalshi_markets_open.json").read_text(encoding="utf-8"))
    m = kalshi.normalize_market(raw["markets"][0], {"category": "Sports", "series_ticker": "X", "mutually_exclusive": True})
    assert m["category"] == "Sports" and m["venue"] == "kalshi"
    for k in ("bid", "ask", "last"):
        assert m[k] is None or 0 <= m[k] <= 1


def _hour(ts_et: str, trade=None, bid=0.40, ask=0.44, vol=0.0):
    t = pd.Timestamp(ts_et, tz=kalshi.ET).tz_convert("UTC")
    return {"ts": t, "trade": trade, "bid": bid, "ask": ask, "volume": vol}


def test_sample_at_close_uses_last_print_before_4pm_and_flags_stale():
    rows = [
        _hour("2026-10-12 15:00", trade=0.50, vol=10),
        _hour("2026-10-12 16:00", trade=0.55, bid=0.53, ask=0.57, vol=5),  # ends at 16:00 → counts for the 12th
        _hour("2026-10-12 18:00", trade=0.90, vol=50),     # after the close → belongs to nobody's 4pm
        _hour("2026-10-13 10:00", bid=0.40, ask=0.44),     # quotes only, no trade
        _hour("2026-10-14 11:00", bid=0.40, ask=0.44),     # still no trade, same mid → stale
        _hour("2026-10-15 12:00", bid=0.50, ask=0.54),     # quote moved → not stale
    ]
    h = pd.DataFrame(rows).set_index("ts")
    d = kalshi.sample_at_close(h)
    assert d.loc["2026-10-12", "p"] == pytest.approx(0.55) and d.loc["2026-10-12", "source"] == "trade"
    assert d.loc["2026-10-13", "p"] == pytest.approx(0.42) and not d.loc["2026-10-13", "stale"]
    assert bool(d.loc["2026-10-14", "stale"]) and not bool(d.loc["2026-10-15", "stale"])


def test_sample_at_close_with_only_after_close_candles_is_empty_not_an_error():
    h = pd.DataFrame([_hour("2026-10-12 18:00", trade=0.90, vol=50)]).set_index("ts")
    d = kalshi.sample_at_close(h)
    assert d.empty and list(d.columns) == ["p", "stale", "source"]
