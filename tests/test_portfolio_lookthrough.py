import pandas as pd
import pytest

from worsey.lookthrough import carriers, concentrations, effective_exposure, sector_exposure
from worsey.portfolio import Portfolio, by_asset_class


def _pf():
    return Portfolio.from_dict({"name": "t", "positions": [
        {"ticker": "QQQ", "value_usd": 50_000},
        {"ticker": "NVDA", "value_usd": 30_000},
        {"ticker": "TLT", "shares": 200},
    ]})


def test_pricing_and_asset_classes():
    pr = _pf().priced({"QQQ": 500.0, "NVDA": 150.0, "TLT": 100.0})
    assert pr.at["QQQ", "shares"] == pytest.approx(100)
    assert pr.at["TLT", "value"] == pytest.approx(20_000)
    assert pr["weight"].sum() == pytest.approx(1.0)
    ac = by_asset_class(pr)
    assert ac.at["equity", "value"] == pytest.approx(80_000) and ac.at["rates", "weight"] == pytest.approx(0.2)


def test_position_needs_exactly_one_size():
    with pytest.raises(ValueError):
        Portfolio.from_dict({"positions": [{"ticker": "SPY"}]})


def test_lookthrough_finds_hidden_nvda_without_renormalizing():
    values = pd.Series({"QQQ": 50_000.0, "NVDA": 30_000.0, "TLT": 20_000.0})
    qqq = pd.DataFrame({"ticker": ["NVDA", "AAPL", "MSFT", "NQZ6"], "weight": [0.08, 0.08, 0.04, 0.01],
                        "name": ["NVIDIA", "APPLE", "MICROSOFT", "NASDAQ 100 E-MINI Dec26"],
                        "sector": ["Tech", "Tech", "Tech", None]})
    eff = effective_exposure(values, {"QQQ": qqq})
    assert eff.at["NVDA", "value"] == pytest.approx(30_000 + 0.08 * 50_000)   # published weight, not inflated
    assert eff.at["cash (via ETFs)", "value"] == pytest.approx(0.01 * 50_000)  # futures sleeve = cash
    assert eff.at["other holdings (QQQ)", "value"] == pytest.approx(0.79 * 50_000)
    assert eff["value"].sum() == pytest.approx(values.sum())
    c = concentrations(eff)
    assert "other holdings (QQQ)" not in c.index
    assert bool(c.at["NVDA", "hidden"]) and not bool(c.at["TLT", "hidden"])
    assert sector_exposure(eff)["Tech"] == pytest.approx((34_000 + 4_000 + 2_000) / 100_000)
    car = carriers(eff, ["NVDA"])
    assert set(car["position"]) == {"QQQ", "NVDA"}


def test_truncation_stops_at_coverage():
    from worsey.lookthrough import truncate
    h = pd.DataFrame({"ticker": list("ABCDE"), "weight": [0.5, 0.3, 0.15, 0.04, 0.01]})
    parts = {p[0]: p[1] for p in truncate("X", h, coverage=0.90)}
    assert set(parts) == {"A", "B", "C", "other holdings (X)"}  # 0.5+0.3 < 0.9, so C is kept, then stop
    assert parts["other holdings (X)"] == pytest.approx(0.05)


def test_live_qqq_fixture_shape():
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).parent / "fixtures" / "live" / "massive_etf_holdings_qqq.json").read_text(encoding="utf-8"))
    r = [x for x in raw["results"] if isinstance(x, dict)][0]
    assert {"constituent_ticker", "weight", "effective_date", "composite_ticker"} <= set(r)


def test_sic_to_sector_map():
    from worsey.sectors import sic_to_sector
    assert sic_to_sector(3674) == ("Information Technology", "Semiconductors")
    assert sic_to_sector("2834")[0] == "Health Care"          # pharma inside chemicals (283x)
    assert sic_to_sector(2810)[0] == "Materials"
    assert sic_to_sector(6798)[0] == "Real Estate"
    assert sic_to_sector(6021)[0] == "Financials"
    assert sic_to_sector(5961)[0] == "Consumer Discretionary"
    assert sic_to_sector(None) == ("unknown", None)


def test_direct_names_get_sectors_and_industries():
    from worsey.lookthrough import industry_exposure
    values = pd.Series({"QQQ": 50_000.0, "NVDA": 30_000.0})
    qqq = pd.DataFrame({"ticker": ["NVDA", "AAPL"], "weight": [0.5, 0.5], "sector": ["Information Technology"] * 2,
                        "industry": ["Semiconductors", "Computer Hardware"]})
    eff = effective_exposure(values, {"QQQ": qqq}, sector_of=lambda n: ("Information Technology", "Semiconductors"))
    assert industry_exposure(eff)["Semiconductors"] == pytest.approx(55_000 / 80_000)  # NVDA direct 30k + 25k via QQQ
