"""Portfolios: load from YAML or chat-stated holdings, price, break down by asset class (SPEC §2).

A position is given either as shares or as a target dollar value; value-based
positions are converted to shares at load-time prices, so the demo portfolios stay
round numbers regardless of the day they are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

PORTFOLIO_DIR = Path(__file__).resolve().parents[2] / "portfolios"

# Asset classes go through ETFs (SPEC §2). Anything not listed is a single-name equity.
ASSET_CLASS = {
    "TLT": "rates", "IEF": "rates", "SHY": "rates",
    "HYG": "credit", "LQD": "credit",
    "GLD": "commodities", "USO": "commodities", "SLV": "commodities", "CPER": "commodities",
    "IBIT": "crypto",
    "FXI": "international", "EWZ": "international", "EEM": "international", "ECH": "international",
    "KWEB": "international",
}


def asset_class(ticker: str) -> str:
    return ASSET_CLASS.get(ticker.upper(), "equity")


@dataclass
class Position:
    ticker: str
    shares: float | None = None
    value_usd: float | None = None

    def __post_init__(self):
        self.ticker = self.ticker.upper()
        if (self.shares is None) == (self.value_usd is None):
            raise ValueError(f"{self.ticker}: give exactly one of shares or value_usd")


@dataclass
class Portfolio:
    name: str
    positions: list[Position]
    cash_usd: float = 0.0
    meta: dict = field(default_factory=dict)
    options: list[dict] = field(default_factory=list)   # {"ticker": underlying, "option": {kind, strike, expiry, contracts}} (#209)

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        pos = [Position(p["ticker"], p.get("shares"), p.get("value_usd")) for p in d["positions"] if not p.get("option")]
        opts = [p for p in d["positions"] if p.get("option")]
        meta = {k: v for k, v in d.items() if k not in ("name", "positions", "cash_usd")}
        return cls(d.get("name", "custom"), pos, float(d.get("cash_usd", 0.0)), meta, opts)

    @classmethod
    def load(cls, name: str, directory: Path = PORTFOLIO_DIR) -> "Portfolio":
        return cls.from_dict(yaml.safe_load((directory / f"{name}.yaml").read_text(encoding="utf-8")))

    def priced(self, prices: dict[str, float]) -> pd.DataFrame:
        """One row per position: ticker, asset_class, shares, price, value, weight."""
        rows = []
        for p in self.positions:
            px = prices.get(p.ticker)
            if px is None or px <= 0:
                raise KeyError(f"no price for {p.ticker}")
            shares = p.shares if p.shares is not None else p.value_usd / px
            rows.append({"ticker": p.ticker, "asset_class": asset_class(p.ticker),
                         "shares": shares, "price": px, "value": shares * px})
        df = pd.DataFrame(rows).set_index("ticker")
        df["weight"] = df["value"] / (df["value"].sum() + self.cash_usd)
        return df

    def values(self, prices: dict[str, float]) -> pd.Series:
        return self.priced(prices)["value"]


def by_asset_class(priced: pd.DataFrame) -> pd.DataFrame:
    g = priced.groupby("asset_class")[["value", "weight"]].sum()
    return g.sort_values("value", ascending=False)
