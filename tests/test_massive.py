"""Massive client logic without network: listing-date trim and next_url pagination."""

import pandas as pd

from worsey.data import massive


def test_daily_bars_trimmed_to_listing_date(monkeypatch):
    bars = {"results": [{"t": pd.Timestamp(d).value // 10**6, "c": c} for d, c in
                        [("2022-09-08", 24.1), ("2023-06-01", 25.0), ("2024-01-11", 26.0), ("2024-01-12", 27.0)]]}

    def fake_get(path, params=None):
        if path.startswith("/v3/reference/tickers/"):
            return {"results": {"list_date": "2024-01-05"}}
        return bars
    monkeypatch.setattr(massive, "_get", fake_get)
    df = massive.daily_bars("IBIT", "2021-01-01", "2026-01-01")
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-11", "2024-01-12"]  # pre-launch junk dropped
    assert len(massive.daily_bars("IBIT", "2021-01-01", "2026-01-01", trim_to_listing=False)) == 4


def test_paged_follows_next_url_without_api_key(monkeypatch):
    seen = []

    def fake_get(path, params=None):
        seen.append((path, dict(params or {})))
        if "cursor" not in (params or {}):
            return {"results": [1, 2], "next_url": "https://api.massive.com/v3/snapshot/options/NVDA?cursor=abc&apiKey=SHOULDNOTSTICK"}
        return {"results": [3]}
    monkeypatch.setattr(massive, "_get", fake_get)
    assert massive._paged("/v3/snapshot/options/NVDA", {"limit": 250}) == [1, 2, 3]
    assert seen[1] == ("/v3/snapshot/options/NVDA", {"cursor": "abc"})
