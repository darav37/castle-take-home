"""Plan B event library v0 (PLAN_B §5, BUILD_PLAN B2).

Calendar first (code, no LLM): every scheduled release since 2000 with its surprise measures.
- Release dates: ALFRED vintage dates (a vintage is a publication), FOMC from federalreserve.gov.
- Market-based surprises (primary, PLAN_B §2, DECISIONS #155): release-day changes in the 2-year and 10-year yields
  and the 5-year breakeven (FRED, end of day).
- Data surprises (secondary, naive): first release minus the trailing-12 average of first releases (CPI MoM, payroll
  change, PCE MoM); GDP advance minus the last GDPNow vintage before it (2016+).
Unscheduled events (LLM under the SPEC §6.1 caps) and realized responses live in scripts/planb_b2.py.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

from ..data.cache import cached_call

FRED_API = "https://api.stlouisfed.org/fred"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
RELEASE_SERIES = {"cpi": "CPIAUCSL", "jobs": "PAYEMS", "gdp": "GDPC1", "pce": "PCEPI"}


def fred_api(path: str, **params) -> dict:
    """FRED API through the disk cache. The key is read from the environment and is never part of the cache key."""
    def fetch():
        r = requests.get(FRED_API + path, params=dict(params, api_key=os.environ["FRED_API_KEY"], file_type="json"),
                         timeout=60)
        r.raise_for_status()
        return r.json()
    return cached_call("fred", path, params, fetch)


def fred_daily(sid: str) -> pd.Series:
    obs = fred_api("/series/observations", series_id=sid)["observations"]
    return pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")})


def vintage_dates(sid: str) -> list[date]:
    return [date.fromisoformat(d) for d in fred_api("/series/vintagedates", series_id=sid, limit=10000)["vintage_dates"]]


def first_releases(sid: str) -> pd.DataFrame:
    """Initial-release values (FRED output_type=4): observation date, value, and the date it was published."""
    obs = fred_api("/series/observations", series_id=sid, output_type=4, realtime_start="1776-07-04",
                   realtime_end="9999-12-31")["observations"]
    rows = [{"obs": pd.Timestamp(o["date"]), "value": float(o[next(k for k in o if k not in ("date", "realtime_start", "realtime_end"))]),
             "published": pd.Timestamp(o["realtime_start"])} for o in obs
            if o.get(next((k for k in o if k not in ("date", "realtime_start", "realtime_end")), "x")) not in (".", "", None)]
    return pd.DataFrame(rows).sort_values("obs")


def web(url: str) -> str:
    def fetch():
        r = requests.get(url, headers=UA, timeout=45)
        r.raise_for_status()
        return {"text": r.text}
    return cached_call("web", url, None, fetch)["text"]


def web_fresh(url: str, tag: str) -> str:
    """As web(), but cached per `tag` (e.g. today's date): for pages that keep changing, like the current month's news."""
    def fetch():
        r = requests.get(url, headers=UA, timeout=45)
        r.raise_for_status()
        return {"text": r.text}
    return cached_call("web", url, {"fresh": tag}, fetch)["text"]


def fomc_dates(start_year: int, recent: list[dict]) -> list[dict]:
    """Scheduled FOMC decisions: federalreserve.gov historical pages (to 2020) plus the current calendar page
    (`recent`, from pmrm_p0b.fomc()). Unscheduled meetings and conference calls are kept, flagged."""
    out = []
    for y in range(start_year, 2021):
        t = re.sub(r"<[^>]+>", "|", web(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"))
        t = re.sub(r"\s+", " ", html.unescape(t))
        for m in re.finditer(rf"\|((?:{MONTHS})(?:/(?:{MONTHS}))?) (\d{{1,2}})(?:-(\d{{1,2}}))? (\(unscheduled\) )?(Meeting|Conference Call)[^|]* - {y}", t):
            mon = m.group(1).split("/")[-1]
            day = int(m.group(3) or m.group(2))
            try:
                d = datetime.strptime(f"{mon} {day} {y}", "%B %d %Y").date()
            except ValueError:
                continue
            out.append({"date": d, "release": "fomc", "scheduled": not m.group(4) and m.group(5) == "Meeting",
                        "source": "federalreserve.gov fomchistorical"})
    out += [{"date": r["date"], "release": "fomc", "scheduled": True, "source": r["source"]} for r in recent if r["release"] == "fomc"]
    return sorted({(r["date"], r["release"]): r for r in out}.values(), key=lambda r: r["date"])


def release_calendar(start: date, recent_fomc: list[dict]) -> pd.DataFrame:
    rows = fomc_dates(start.year, recent_fomc)
    for rel, sid in RELEASE_SERIES.items():
        if rel == "gdp":     # every GDP vintage is a release (advance, second, third estimate; annual updates)
            ds = vintage_dates(sid)
            src = f"ALFRED vintage dates ({sid})"
        else:                # CPI/jobs/PCE vintages include seasonal-factor revisions: keep first-publication days
            ds = sorted({p.date() for p in first_releases(sid)["published"]})
            src = f"ALFRED first-release dates ({sid})"
        rows += [{"date": d, "release": rel, "scheduled": True, "source": src} for d in ds if d >= start]
    cal = pd.DataFrame(rows)
    return cal[cal["date"] >= start].drop_duplicates(["date", "release"]).sort_values("date").reset_index(drop=True)


def data_surprises(start: date) -> pd.DataFrame:
    """Naive data surprises keyed by publication date (secondary to the market-based measures)."""
    out = []
    for rel, sid, kind in (("cpi", "CPIAUCSL", "mom"), ("pce", "PCEPI", "mom"), ("jobs", "PAYEMS", "diff")):
        fr = first_releases(sid).set_index("obs")
        lvl = fr["value"]
        chg = (lvl.pct_change() * 100) if kind == "mom" else lvl.diff()
        exp = chg.shift(1).rolling(12).mean()
        s = pd.DataFrame({"published": fr["published"], "actual": chg, "naive_expect": exp}).dropna()
        s["data_surprise"] = s["actual"] - s["naive_expect"]
        s["z"] = s["data_surprise"] / s["data_surprise"].rolling(60, min_periods=24).std()
        for _, r in s.iterrows():
            if r["published"].date() >= start:
                out.append({"date": r["published"].date(), "release": rel, "actual": r["actual"],
                            "expect": r["naive_expect"], "data_surprise": r["data_surprise"], "data_surprise_z": r["z"],
                            "expect_source": "trailing 12 first releases (naive)"})
    # GDP: advance estimate (first release) minus the last GDPNow vintage published before it
    g = first_releases("GDPC1").set_index("obs")
    gq = (g["value"].pct_change().add(1) ** 4 - 1) * 100
    try:
        gn = fred_api("/series/observations", series_id="GDPNOW", realtime_start="2011-01-01",
                      realtime_end="9999-12-31")["observations"]
        now = pd.DataFrame([{"obs": pd.Timestamp(o["date"]), "rt": pd.Timestamp(o["realtime_start"]), "v": float(o["value"])}
                            for o in gn if o["value"] not in (".", "")])
    except Exception:  # noqa: BLE001
        now = pd.DataFrame(columns=["obs", "rt", "v"])
    for q, pub in g["published"].items():
        if pub.date() < start or q not in gq.index or pd.isna(gq[q]):
            continue
        prior = now[(now["obs"] == q) & (now["rt"] < pub)]
        exp = float(prior.sort_values("rt")["v"].iloc[-1]) if len(prior) else np.nan
        out.append({"date": pub.date(), "release": "gdp", "actual": float(gq[q]), "expect": exp,
                    "data_surprise": float(gq[q]) - exp if not np.isnan(exp) else np.nan, "data_surprise_z": np.nan,
                    "expect_source": "GDPNow last vintage before release" if not np.isnan(exp) else "none"})
    return pd.DataFrame(out)


def market_surprises(cal: pd.DataFrame) -> pd.DataFrame:
    """Release-day changes (close before -> close of the release day) in DGS2, DGS10, T5YIE, in percentage points.
    FRED's H.15 yields are end-of-day; a release on a non-business day maps to the next business day."""
    s2, s10, be = fred_daily("DGS2"), fred_daily("DGS10"), fred_daily("T5YIE")
    days = s2.index

    def chg(s: pd.Series, d: date):
        i = s.index.searchsorted(pd.Timestamp(d))
        if i <= 0 or i >= len(s):
            return np.nan
        return float(s.iloc[i] - s.iloc[i - 1])

    out = cal.copy()
    out["d2y"] = [chg(s2, d) for d in cal["date"]]
    out["d10y"] = [chg(s10, d) for d in cal["date"]]
    out["d5y_breakeven"] = [chg(be, d) for d in cal["date"]]
    out["market_day"] = [days[min(days.searchsorted(pd.Timestamp(d)), len(days) - 1)].date() for d in cal["date"]]
    return out


# ---------------------------------------------------------------- unscheduled events: fixed schema (PLAN_B §5.1)

CATEGORIES = {
    "trade_tariffs": "tariff announcements, trade-war escalations or truces, export controls, trade deals",
    "sanctions": "new sanctions or sanctions relief on countries, companies or sectors (OFAC, EU, UK)",
    "conflict_war": "wars, military escalations, attacks, ceasefires, coups",
    "elections_politics": "national election results, surprise political outcomes, government shutdowns, debt-ceiling events",
    "regulatory_court": "major regulatory actions and court rulings affecting an industry or the economy",
    "disaster": "natural disasters: hurricanes, earthquakes, floods, wildfires with economic impact",
    "supply_shock": "supply disruptions: OPEC+ decisions, pipeline or refinery outages, shipping chokepoints, strikes, cyberattacks",
    "central_bank_unscheduled": "unscheduled central-bank actions or surprises outside regular meetings (any major central bank)",
    "health_pandemic": "pandemics, outbreaks and major public-health decisions",
    "financial_stress": "bank failures, sovereign-debt crises, market-structure shocks, major defaults",
}
SECTORS = ["energy", "materials", "industrials", "consumer_discretionary", "consumer_staples", "health_care", "financials",
           "banks", "insurance", "tech", "semiconductors", "software", "communications", "utilities", "real_estate",
           "homebuilders", "airlines", "transport_shipping", "autos", "defense", "agriculture", "metals_mining",
           "retail", "media", "biotech", "broad_market"]
SCHEMA = {
    "title": "short name", "category": "one of the category keys", "date_first_report": "YYYY-MM-DD (first credible public report, US Eastern)",
    "time_first_report_et": "HH:MM if known, else null", "actors": ["countries, organisations, companies"], "region": "country or region",
    "targeted_sectors": [f"from: {', '.join(SECTORS)}"], "targeted_tickers": ["US-listed tickers directly targeted, if any"],
    "commodities": ["e.g. crude oil, natural gas, wheat, copper"], "severity": "one of: minor, moderate, major, extreme",
    "anticipation": "one of: surprise, partly_expected, widely_expected", "prediction_market": "venue and question if one existed at the time, else null",
    "description": "one or two sentences, facts only, no market outcome", "sources": [{"url": "...", "publisher": "...", "date": "YYYY-MM-DD"}],
}
