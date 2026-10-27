"""ETF look-through (METHODOLOGY §3; ruling on C11).

w_name = w_direct + Σ_etf w_etf · h_name,etf

- Holdings h are the ETF's latest published snapshot, NOT renormalized.
- Each ETF is truncated to its top holdings covering ≥ 90% of weight; the residual goes to an
  "other holdings (ETF)" bucket, so a name is never overstated.
- Cash and futures sleeves (e.g. QQQ's Nasdaq E-mini, CP-L) are treated as cash.

Used for concentration, mapping name/sector events, and explanation. Never for macro β.
Holdings come in as {etf: DataFrame[ticker, weight, (name, asset_class, security_type, sector)]}.
A position with no holdings entry (a single stock, or TLT) looks through to itself.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config

CASH_LIKE = re.compile(r"\b(cash|dollar|money market|e-?mini|futures?|fut|t-?bill|treasury bill|repo)\b", re.I)


def _cash_like(row) -> bool:
    if str(row.get("asset_class") or "").lower() in ("cash", "futures", "future"):
        return True
    if str(row.get("security_type") or "").lower() in ("cash", "future", "futures"):
        return True
    return bool(CASH_LIKE.search(str(row.get("name") or "")))


def truncate(etf: str, h: pd.DataFrame, coverage: float | None = None) -> list[tuple]:
    """(name, weight, sector, industry) parts for one ETF: top names to `coverage`, cash, and the residual bucket."""
    coverage = config.LOOKTHROUGH_COVERAGE if coverage is None else coverage
    h = h.sort_values("weight", ascending=False).copy()
    h["cash"] = [_cash_like(r) for _, r in h.iterrows()]
    parts, kept = [], 0.0
    cash_w = float(h.loc[h["cash"], "weight"].sum())
    if cash_w:
        parts.append(("cash (via ETFs)", cash_w, "cash", None))
        kept += cash_w
    for _, r in h[~h["cash"]].iterrows():
        if kept >= coverage:
            break
        parts.append((r["ticker"], float(r["weight"]), r["sector"] if "sector" in h else None,
                      r["industry"] if "industry" in h else None))
        kept += float(r["weight"])
    residual = 1.0 - kept  # never renormalize: what we didn't keep is "other", not more NVDA
    if residual > 1e-9:
        parts.append((f"other holdings ({etf})", residual, "other", None))
    return parts


def effective_exposure(values: pd.Series, holdings: dict[str, pd.DataFrame], coverage: float | None = None,
                       sector_of=None) -> pd.DataFrame:
    """Dollar exposure per underlying name, with where it comes from.

    Rows indexed by name: value, weight (of the book), direct (held outright), via
    (dict position → dollars), sector.
    """
    total = values.sum()
    rows: dict[str, dict] = {}
    for pos, v in values.items():
        h = holdings.get(pos)
        parts = [(pos, 1.0, None, None)] if h is None or h.empty else truncate(pos, h, coverage)
        for name, w, sector, industry in parts:
            r = rows.setdefault(name, {"value": 0.0, "direct": 0.0, "via": {}, "sector": sector, "industry": industry})
            r["value"] += v * w
            r["via"][pos] = r["via"].get(pos, 0.0) + v * w
            if name == pos:
                r["direct"] += v * w
            if r["sector"] is None and sector is not None:
                r["sector"], r["industry"] = sector, industry
    if sector_of is not None:  # names held directly (NVDA, or TLT itself) have no holdings row to carry a sector
        for name, r in rows.items():
            if r["sector"] is None and not is_bucket(name):
                r["sector"], r["industry"] = sector_of(name)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["weight"] = df["value"] / total
    df["direct_weight"] = df["direct"] / total
    return df.sort_values("value", ascending=False)


def is_bucket(name: str) -> bool:
    return name.startswith("other holdings (") or name == "cash (via ETFs)"


def concentrations(eff: pd.DataFrame, top: int = 10, min_gap: float = 0.02) -> pd.DataFrame:
    """Top real names (buckets excluded), flagging those where look-through adds materially."""
    real = eff[[not is_bucket(n) for n in eff.index]]
    out = real.head(top)[["value", "weight", "direct_weight", "sector"]].copy()
    out["hidden"] = (out["weight"] - out["direct_weight"]) >= min_gap
    return out


def sector_exposure(eff: pd.DataFrame) -> pd.Series:
    return eff.groupby(eff["sector"].fillna("unknown"))["weight"].sum().sort_values(ascending=False)


def industry_exposure(eff: pd.DataFrame) -> pd.Series:
    """e.g. Semiconductors: the demo's concentration moment."""
    return eff.dropna(subset=["industry"]).groupby("industry")["weight"].sum().sort_values(ascending=False)


def carriers(eff: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Which positions carry exposure to the given names (e.g. chip export controls → NVDA, AMD, AVGO)."""
    rows = []
    for n in names:
        if n in eff.index:
            for pos, d in eff.at[n, "via"].items():
                rows.append({"name": n, "position": pos, "value": d})
    df = pd.DataFrame(rows, columns=["name", "position", "value"])
    return df.sort_values("value", ascending=False)
