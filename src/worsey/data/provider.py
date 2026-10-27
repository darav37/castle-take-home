"""One interface between the engines/tools and the data sources.

LiveMarketData wires the (unverified) Massive and Kalshi clients through the disk cache.
SyntheticMarketData generates labeled fake data for tests and offline plumbing runs; its
numbers must never appear in a demo or transcript (every payload carries synthetic=True).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from ..options.pricer import bs_price


class MarketData(Protocol):
    synthetic: bool
    def today(self) -> date: ...
    def closes(self, tickers: list[str], days: int = 750) -> pd.DataFrame: ...
    def prices(self, tickers: list[str]) -> dict[str, float]: ...
    def chain(self, underlying: str) -> pd.DataFrame: ...
    def holdings(self, etf: str) -> pd.DataFrame | None: ...
    def event_markets(self) -> list[dict]: ...
    def market_history(self, market_id: str) -> pd.Series: ...


class LiveMarketData:
    """Massive + Kalshi via the disk cache. Endpoints UNVERIFIED until checkpoint 1."""
    synthetic = False

    def __init__(self, as_of: date | None = None):
        self._today = as_of or date.today()
        self._cache: dict = {}

    def today(self):
        return self._today

    def closes(self, tickers, days=750):
        from . import massive
        start = (self._today - timedelta(days=int(days * 1.5))).isoformat()
        return massive.closes(list(dict.fromkeys(tickers)), start, self._today.isoformat()).tail(days)

    def prices(self, tickers):
        c = self.closes(tickers, days=5)
        return {t: float(c[t].dropna().iloc[-1]) for t in tickers}

    def opening_gaps(self, tickers) -> dict[str, float]:
        """Today's opening gap per ticker (today's open / the last close before today - 1); empty before the open or on a
        day without a session (DECISIONS #200)."""
        from . import massive
        c = self.closes(tickers, days=5)
        c = c[c.index < pd.Timestamp(self._today)]
        out = {}
        for t in tickers:
            o = massive.opening_price_today(t)
            prev = c[t].dropna() if t in c else pd.Series(dtype=float)
            if o is not None and len(prev):
                out[t] = o / float(prev.iloc[-1]) - 1
        return out

    def chain(self, underlying, expiry_from: date | None = None, expiry_to: date | None = None, band: float | None = None):
        """All pages over an expiry window, strikes within ±band of spot (CP-L F6; latency fix Oct 16). Memoized."""
        from .. import config
        from . import massive
        f = expiry_from or self._today
        t = expiry_to or (self._today + timedelta(days=config.CHAIN_MAX_DAYS))
        band = config.CHAIN_STRIKE_BAND if band is None else band
        key = ("chain", underlying, f, t, band)
        if key not in self._cache:
            S = self.prices([underlying])[underlying]
            self._cache[key] = massive.option_chain_snapshot(underlying, f.isoformat(), t.isoformat(), S * (1 - band), S * (1 + band))
        return self._cache[key]

    def earnings(self, ticker):
        """Next earnings date (Benzinga) and past realized earnings-day moves (from bars)."""
        from . import massive
        return massive.earnings(ticker, self._today, self.closes([ticker], days=1500)[ticker])

    def holdings(self, etf, sectors: bool = True):
        """Latest snapshot, with sector/industry for the names look-through keeps (top ≥ 90% + margin)."""
        from .. import config
        from . import massive
        h = massive.etf_holdings(etf)
        if h.empty or not sectors:
            return h
        h = h.sort_values("weight", ascending=False).reset_index(drop=True)
        keep = h["weight"].cumsum().shift(fill_value=0) < config.LOOKTHROUGH_COVERAGE + 0.02
        h["sector"], h["industry"] = None, None
        for i in h.index[keep]:
            h.loc[i, ["sector", "industry"]] = list(massive.sector(h.at[i, "ticker"]))
        return h

    def sector(self, ticker):
        from . import massive
        return massive.sector(ticker)

    def event_markets(self):
        from ..scenario.discovery import IN_SCOPE_CATEGORIES
        from . import kalshi
        if "markets" not in self._cache:
            self._cache["markets"] = {m["id"]: m for m in kalshi.open_markets(IN_SCOPE_CATEGORIES)}
        return list(self._cache["markets"].values())

    def market_history(self, market_id, days: int = 730):
        """Daily p at 16:00 ET, stale days NaN (CP-L F2)."""
        from datetime import datetime, timezone
        from . import kalshi
        m = self._cache.get("markets", {}).get(market_id)
        if m is None:
            raise KeyError(f"{market_id} not in the scanned universe")
        start = datetime.combine(self._today - timedelta(days=days), datetime.min.time(), tzinfo=timezone.utc)
        if m.get("open_date"):
            start = max(start, datetime.fromisoformat(m["open_date"]).replace(tzinfo=timezone.utc))
        return kalshi.p_series(m, start)


class BuiltMarketData(LiveMarketData):
    """Live prices/chains/holdings, but the event universe, p history and β come from `worsey build`;
    only market quotes refresh at query time (once per hour, cached)."""

    def __init__(self, build: dict | None = None, as_of: date | None = None, quote_tag: str | None = None, live: bool = False):
        """live=True (live chat on live data): today is the real date, Kalshi quotes refresh hourly
        (quote_tag None), and closes after the build date are appended from Massive. Replays pin as_of and quote_tag."""
        from ..build import load
        self.build = build or load()
        self.live = live
        super().__init__(as_of or (date.today() if live else date.fromisoformat(self.build["manifest"]["as_of"])))
        self.quote_tag = quote_tag

    def event_markets(self):
        from . import kalshi
        if "markets" not in self._cache:
            ms = [dict(m) for m in self.build["markets"]]
            q = kalshi.quotes([m["id"] for m in ms], tag=self.quote_tag)
            for m in ms:
                m.update({k: v for k, v in q.get(m["id"], {}).items() if v is not None})
            self._cache["markets"] = {m["id"]: m for m in ms if m.get("status", "active") in ("active", "open", None)}
        return list(self._cache["markets"].values())

    def market_history(self, market_id, days: int = 730):
        return self.build["p_history"][market_id].dropna(how="all")

    def betas(self) -> dict:
        return self.build["betas"]

    def structure(self) -> dict | None:
        return self.build.get("structure")

    def closes(self, tickers, days=750):
        if days > len(self.build["closes"]) + 5:  # longer than the build keeps (e.g. 2018 analogs): full cached history
            return super().closes(tickers, days)
        have = [t for t in dict.fromkeys(tickers) if t in self.build["closes"].columns]
        missing = [t for t in dict.fromkeys(tickers) if t not in have]
        out = self.build["closes"][have]
        if self.live and have:
            out = self._append_since_build(out)
        if missing:
            out = out.join(super().closes(missing, days), how="left")
        return out.tail(days)

    def _append_since_build(self, out: pd.DataFrame) -> pd.DataFrame:
        """Daily closes after the build date, from Massive (cached per day); the build's history is kept as is."""
        from . import massive
        last = out.index.max().date()
        if self._today <= last:
            return out
        key = ("since_build", tuple(out.columns), self._today)
        if key not in self._cache:
            try:
                self._cache[key] = massive.closes(list(out.columns), (last + timedelta(days=1)).isoformat(), self._today.isoformat())
            except Exception:  # noqa: BLE001 — stale beats broken: keep the build's closes
                self._cache[key] = pd.DataFrame()
        new = self._cache[key]
        return out if new is None or new.empty else pd.concat([out, new[new.index > out.index.max()]]).sort_index()


@dataclass
class SyntheticMarketData:
    """Deterministic fake world: a few event markets that genuinely move a few assets."""
    seed: int = 7
    as_of: date = date(2026, 10, 14)
    n_days: int = 500
    synthetic: bool = True
    _cache: dict = field(default_factory=dict)

    # event -> {asset: true β per +1.00 Δp}
    TRUTH = {
        "TARIFF-25-MAR": {"SPY": -0.20, "QQQ": -0.25, "NVDA": -0.40, "XLE": -0.10, "TLT": 0.04, "GLD": 0.06, "HYG": -0.05},
        "FED-CUT-DEC": {"SPY": 0.08, "QQQ": 0.10, "TLT": 0.06, "GLD": 0.03, "HYG": 0.03},
        "OIL-100-Q1": {"XLE": 0.25, "USO": 0.40, "SPY": -0.05, "TLT": -0.03},
    }
    VOL = {"SPY": 0.16, "QQQ": 0.21, "NVDA": 0.50, "TLT": 0.14, "GLD": 0.15, "XLE": 0.25, "HYG": 0.07,
           "USO": 0.35, "UUP": 0.07, "IEF": 0.07, "VIXY": 0.70, "AAPL": 0.28, "TSLA": 0.60, "XLK": 0.24}
    PX = {"SPY": 600, "QQQ": 520, "NVDA": 180, "TLT": 90, "GLD": 300, "XLE": 90, "HYG": 80, "USO": 75,
          "UUP": 28, "IEF": 95, "VIXY": 40, "AAPL": 230, "TSLA": 250, "XLK": 250}

    def today(self):
        return self.as_of

    def _world(self):
        if "w" in self._cache:
            return self._cache["w"]
        rng = np.random.default_rng(self.seed)
        idx = pd.bdate_range(end=pd.Timestamp(self.as_of), periods=self.n_days)
        ps, dps = {}, {}
        for i, m in enumerate(self.TRUTH):
            # random walk in log-odds, so p stays interior and jumps survive (like real markets)
            steps = rng.normal(0, 0.03, len(idx))
            steps[rng.choice(len(idx), 10, replace=False)] += rng.choice([-0.6, 0.6], 10)
            x = np.log((0.35 + 0.1 * i) / (0.65 - 0.1 * i)) + np.cumsum(steps - steps.mean())
            p = 1 / (1 + np.exp(-x))
            ps[m] = pd.Series(p, index=idx)
            dps[m] = ps[m].diff().fillna(0)
        mkt = rng.normal(0, 0.008, len(idx))
        rets = {}
        for t, v in self.VOL.items():
            r = rng.normal(0, v / np.sqrt(252), len(idx)) + (0.6 * mkt if t not in ("TLT", "GLD", "IEF", "UUP") else 0)
            for m, b in self.TRUTH.items():
                r = r + b.get(t, 0.0) * dps[m].to_numpy()
            rets[t] = r
        closes = pd.DataFrame({t: self.PX[t] * np.exp(np.cumsum(r) - np.cumsum(r)[-1]) for t, r in rets.items()}, index=idx)
        self._cache["w"] = (closes, ps)
        return self._cache["w"]

    def closes(self, tickers, days=750):
        c, _ = self._world()
        return c[[t for t in dict.fromkeys(tickers) if t in c.columns]].tail(days)

    def prices(self, tickers):
        c, _ = self._world()
        return {t: float(c[t].iloc[-1]) for t in tickers if t in c.columns}

    def opening_gaps(self, tickers) -> dict[str, float]:
        """Deterministic small gaps (tests of the after-the-open path)."""
        rng = np.random.default_rng(7)
        return {t: float(rng.normal(0, 0.006)) for t in tickers}

    def chain(self, underlying, expiry_from=None, expiry_to=None, band=None):
        S = self.prices([underlying])[underlying]
        iv0 = self.VOL.get(underlying, 0.3) * 1.1
        rows = []
        for d in (7, 14, 30, 45, 60, 91, 182):
            e = pd.Timestamp(self.as_of + timedelta(days=d))
            T = d / 365
            step = 1 if S < 100 else 5
            for K in np.arange(round(S * 0.6 / step) * step, S * 1.4, step):
                for kind in ("call", "put"):
                    iv = iv0 * (1 + 0.8 * max(0, 1 - K / S))  # put skew
                    m = bs_price(S, K, T, 0.04, 0, iv, kind)
                    rows.append({"expiry": e, "strike": float(K), "kind": kind, "bid": m * 0.98, "ask": m * 1.02,
                                 "mid": m, "iv": iv, "oi": 1000, "volume": 100})
        return pd.DataFrame(rows)

    def sector(self, ticker):
        return {"NVDA": ("Information Technology", "Semiconductors"), "AAPL": ("Information Technology", "Computer Hardware"),
                "MSFT": ("Information Technology", "Software"), "AMZN": ("Consumer Discretionary", "Internet Retail"),
                "XLE": ("Energy", None), "TSLA": ("Consumer Discretionary", "Automobiles")}.get(ticker, ("unknown", None))

    def holdings(self, etf, sectors: bool = True):
        if etf == "QQQ":
            return pd.DataFrame({"ticker": ["NVDA", "AAPL", "MSFT", "AMZN", "OTHER_QQQ"], "weight": [0.09, 0.08, 0.08, 0.05, 0.70],
                                 "sector": ["Information Technology"] * 3 + ["Consumer Discretionary", "unknown"],
                                 "industry": ["Semiconductors", "Computer Hardware", "Software", "Internet Retail", None]})
        if etf == "SPY":
            return pd.DataFrame({"ticker": ["NVDA", "AAPL", "MSFT", "AMZN", "OTHER_SPY"], "weight": [0.07, 0.06, 0.06, 0.04, 0.77],
                                 "sector": ["Information Technology"] * 3 + ["Consumer Discretionary", "unknown"],
                                 "industry": ["Semiconductors", "Computer Hardware", "Software", "Internet Retail", None]})
        return None

    def event_markets(self):
        _, ps = self._world()
        titles = {"TARIFF-25-MAR": "US average tariff rate 25% or more by March 31",
                  "FED-CUT-DEC": "Fed cuts rates at the December meeting",
                  "OIL-100-Q1": "WTI above $100 before April"}
        closes = {"TARIFF-25-MAR": "2027-03-31", "FED-CUT-DEC": "2026-12-10", "OIL-100-Q1": "2027-03-31"}
        return [{"id": m, "title": titles[m], "p": float(ps[m].iloc[-1]), "bid": float(ps[m].iloc[-1]) - 0.01,
                 "ask": float(ps[m].iloc[-1]) + 0.01, "last": float(ps[m].iloc[-1]), "volume": 50_000,
                 "close_date": closes[m], "open_date": str(ps[m].index[0].date()), "category": "Economics", "venue": "kalshi"}
                for m in self.TRUTH]

    def market_history(self, market_id):
        _, ps = self._world()
        return ps[market_id]
