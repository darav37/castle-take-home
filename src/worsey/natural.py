"""natural_world_analyst's code-side data (SPEC §1b; DECISIONS #116): commodities with weather inside.

Every number here is computed in code; the expert words it.
- Futures curves (Massive futures, on our plan): the next contracts' settlements per product, with carry: front-2nd
  and front-12th spreads, annualized roll yield, contango/backwardation, 5-day changes, the 3-2-1 crack.
- Weather (Open-Meteo, free): three models side by side, NOAA GFS, ECMWF IFS and ECMWF's AI model AIFS, as
  population-weighted heating/cooling degree days for gas demand and Corn Belt rain/heat for crops, vs a 10-year normal.
  Model disagreement is reported as the uncertainty.
- Spot (FRED, free): Henry Hub, WTI, Brent.
- Release calendar (code): EIA storage/petroleum, USDA Crop Progress and WASDE, CFTC positioning.
- Hooks that need the trainer's free keys: EIA (inventories, grid), USDA NASS (crop condition), ISO power data (PJM, ERCOT).
"""

from __future__ import annotations

import io
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

MONTHS = "FGHJKMNQUVXZ"
PRODUCTS = {   # Massive/CME product codes; months = which contract months are listed
    "CL": {"name": "WTI crude oil", "unit": "$/bbl", "months": MONTHS},
    "BZ": {"name": "Brent crude oil", "unit": "$/bbl", "months": MONTHS},
    "NG": {"name": "Henry Hub natural gas", "unit": "$/MMBtu", "months": MONTHS},
    "HO": {"name": "NY Harbor ULSD (heating oil)", "unit": "$/gal", "months": MONTHS},
    "RB": {"name": "RBOB gasoline", "unit": "$/gal", "months": MONTHS},
    "ZC": {"name": "Corn", "unit": "cents/bu", "months": "HKNUZ"},
    "ZS": {"name": "Soybeans", "unit": "cents/bu", "months": "FHKNQUX"},
    "ZW": {"name": "Chicago wheat", "unit": "cents/bu", "months": "HKNUZ"},
}
# population-weighted gas-demand basket (millions, metro areas, rough) and the Corn Belt
GAS_CITIES = {"New York": (40.71, -74.01, 19.5), "Chicago": (41.88, -87.63, 9.4), "Philadelphia": (39.95, -75.17, 6.2),
              "Boston": (42.36, -71.06, 4.9), "Washington": (38.90, -77.04, 6.3), "Detroit": (42.33, -83.05, 4.3),
              "Atlanta": (33.75, -84.39, 6.2), "Dallas": (32.78, -96.80, 7.9), "Houston": (29.76, -95.37, 7.3),
              "Minneapolis": (44.98, -93.27, 3.7)}
CORN_BELT = {"Des Moines IA": (41.59, -93.62), "Champaign IL": (40.12, -88.24), "Indianapolis IN": (39.77, -86.16),
             "Lincoln NE": (40.81, -96.70), "Mankato MN": (44.16, -94.00)}
MODELS = {"GFS (NOAA)": "gfs_seamless", "ECMWF IFS": "ecmwf_ifs025", "ECMWF AIFS (AI)": "ecmwf_aifs025_single"}


# ---------------------------------------------------------------- futures curves and carry

def _contract_month(product: str, ticker: str, today: date) -> date | None:
    """CLX6 / NGX26 → the contract month (1-digit years are read in today's decade, rolling forward)."""
    code, yy = ticker[len(product)], ticker[len(product) + 1:]
    if code not in MONTHS or not yy.isdigit():
        return None
    year = 2000 + int(yy) if len(yy) == 2 else (today.year // 10) * 10 + int(yy)
    if len(yy) == 1 and year < today.year - 1:
        year += 10
    return date(year, MONTHS.index(code) + 1, 1)


def outrights(product: str, today: date, rows: list[dict]) -> list[dict]:
    """Plain contracts only (not spreads like CLX6-CLF7 or minis), not yet expired, in expiry order."""
    import re
    rx = re.compile(rf"^{product}[FGHJKMNQUVXZ]\d{{1,2}}$")
    seen, out = set(), []
    for r in rows:
        d = r.get("details") or {}
        t, sd = d.get("ticker", ""), d.get("settlement_date")
        if not rx.match(t) or not sd or sd < today.isoformat() or t in seen:
            continue
        ses = r.get("session") or {}
        px = ses.get("settlement_price") or ses.get("previous_settlement") or (r.get("last_trade") or {}).get("price")
        month = _contract_month(product, t, today)
        if px is None or month is None:
            continue
        seen.add(t)
        out.append({"ticker": t, "month": month.strftime("%b %Y"), "expiry": sd, "settle": float(px),
                    "change_1d": ses.get("change")})
    return sorted(out, key=lambda x: x["expiry"])


def futures_curve(product: str, today: date | None = None, n: int = 12, snapshot=None, history=None) -> dict:
    """The live curve (Massive futures snapshot), plus carry computed in code."""
    from .data import massive
    today = today or date.today()
    rows = (snapshot or massive.futures_snapshot)(product)
    pts = outrights(product, today, rows)[:n]
    if len(pts) < 2:
        return {"product": product, "name": PRODUCTS[product]["name"], "curve": pts, "note": "not enough live contracts"}
    f1, f2, fl = pts[0]["settle"], pts[1]["settle"], pts[-1]["settle"]
    out = {"product": product, "name": PRODUCTS[product]["name"], "unit": PRODUCTS[product]["unit"], "curve": pts,
           "front": f1, "front_month": pts[0]["month"], "front_minus_second": f1 - f2, "front_minus_last": f1 - fl,
           "last_month": pts[-1]["month"], "shape": "backwardation" if f1 > f2 else "contango" if f1 < f2 else "flat",
           # roll yield of holding the front and rolling into the second, annualized by the months between them
           "roll_yield_annualized": (f1 / f2 - 1) * (12 / _months_between(pts[0]["month"], pts[1]["month"])) if f2 else None}
    hist = history or (lambda t: massive._get(f"/futures/v1/aggs/{t}", {"limit": 10}).get("results") or [])
    try:
        h1, h2 = (_settles(pts[i]["ticker"], hist) for i in (0, 1))
        if len(h1) >= 6 and len(h2) >= 6:
            out["front_change_5d"] = f1 - float(h1.iloc[-6])
            out["front_second_spread_change_5d"] = (f1 - f2) - (float(h1.iloc[-6]) - float(h2.iloc[-6]))
    except Exception:  # noqa: BLE001
        pass
    return out


def _settles(ticker: str, fetch) -> pd.Series:
    rows = fetch(ticker) or []
    s = {r["session_end_date"]: (r.get("settlement_price") or r.get("close")) for r in rows if r.get("session_end_date")}
    return pd.Series(s, dtype=float).sort_index()


def _months_between(a: str, b: str) -> int:
    da, db = datetime.strptime(a, "%b %Y"), datetime.strptime(b, "%b %Y")
    return max((db.year - da.year) * 12 + db.month - da.month, 1)


def crack_321(today: date | None = None) -> dict:
    """3-2-1 crack on the front contracts: (2·RBOB + 1·ULSD) in $/bbl minus 3·WTI, per barrel of crude."""
    cl, rb, ho = (futures_curve(p, today, 2) for p in ("CL", "RB", "HO"))
    if not all(x.get("front") for x in (cl, rb, ho)):
        return {"note": "a front contract is missing"}
    v = (2 * rb["front"] * 42 + ho["front"] * 42 - 3 * cl["front"]) / 3
    return {"crack_321_per_bbl": v, "wti": cl["front"], "rbob_per_gal": rb["front"], "ulsd_per_gal": ho["front"]}


# ---------------------------------------------------------------- weather: three models vs normal

def _get(url, params, daily: bool = False):
    from .data.http import get_json
    fmt = "%Y%m%d" if daily else "%Y%m%dT%H"   # forecasts refresh hourly; climatology daily
    return get_json("openmeteo", url, "", params, cache_tag=datetime.now(timezone.utc).strftime(fmt))


def _daily_mean_temp(lat, lon, model, days) -> pd.Series:
    if model.endswith("_single"):   # AIFS is served hourly only: daily mean from hourly
        d = _get("https://api.open-meteo.com/v1/forecast", {"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
                                                             "temperature_unit": "fahrenheit", "forecast_days": days, "models": model})
        h = pd.Series(d.get("hourly", {}).get("temperature_2m", []), index=pd.to_datetime(d.get("hourly", {}).get("time", [])), dtype=float)
        return h.resample("D").mean().dropna()
    d = _get("https://api.open-meteo.com/v1/forecast", {"latitude": lat, "longitude": lon, "daily": "temperature_2m_mean",
                                                         "temperature_unit": "fahrenheit", "forecast_days": days, "models": model})
    return pd.Series(d.get("daily", {}).get("temperature_2m_mean", []), index=pd.to_datetime(d.get("daily", {}).get("time", [])),
                     dtype=float).dropna()


def _normal_temp(lat, lon, days_ahead: list[date], years: int = 10) -> pd.Series:
    """10-year climatology for the same calendar days (archive), one call per location."""
    end = date.today() - timedelta(days=6)
    d = _get("https://archive-api.open-meteo.com/v1/archive", {"latitude": lat, "longitude": lon, "daily": "temperature_2m_mean",
                                                                "temperature_unit": "fahrenheit",
                                                                "start_date": (end - timedelta(days=365 * years)).isoformat(),
                                                                "end_date": end.isoformat()}, daily=True)
    s = pd.Series(d.get("daily", {}).get("temperature_2m_mean", []), index=pd.to_datetime(d.get("daily", {}).get("time", [])), dtype=float)
    doy = s.groupby(s.index.dayofyear).mean()
    return pd.Series({dd: float(doy.get(pd.Timestamp(dd).dayofyear, np.nan)) for dd in days_ahead})


def gas_weather(days: int = 14) -> dict:
    """Population-weighted HDD/CDD (base 65°F) for the next `days`, per model, vs the 10-year normal. The ~40 web calls
    run in parallel (one per city × model, plus normals), so the tool fits inside an expert's time budget."""
    from concurrent.futures import ThreadPoolExecutor
    tot = sum(w for *_, w in GAS_CITIES.values())
    horizon = [date.today() + timedelta(days=i) for i in range(days)]
    jobs = {(label, city): (lat, lon, model) for label, model in MODELS.items() for city, (lat, lon, _) in GAS_CITIES.items()}

    def fc(args):
        lat, lon, model = args
        try:
            return _daily_mean_temp(lat, lon, model, days).iloc[:days]
        except Exception:  # noqa: BLE001
            return None

    def nm(args):
        lat, lon = args
        try:
            return _normal_temp(lat, lon, horizon)
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(max_workers=12) as ex:
        fcs = dict(zip(jobs, ex.map(fc, jobs.values())))
        norms = dict(zip(GAS_CITIES, ex.map(nm, [(lat, lon) for lat, lon, _ in GAS_CITIES.values()])))
    per_model = {}
    for label in MODELS:
        series = {c: fcs[(label, c)] for c in GAS_CITIES}
        if any(v is None or v.empty for v in series.values()):
            continue
        per_model[label] = {"hdd": sum(float(np.clip(65 - series[c], 0, None).sum()) * GAS_CITIES[c][2] / tot for c in GAS_CITIES),
                            "cdd": sum(float(np.clip(series[c] - 65, 0, None).sum()) * GAS_CITIES[c][2] / tot for c in GAS_CITIES)}
    nh = sum(float(np.clip(65 - norms[c], 0, None).sum()) * GAS_CITIES[c][2] / tot for c in GAS_CITIES if norms[c] is not None)
    nc = sum(float(np.clip(norms[c] - 65, 0, None).sum()) * GAS_CITIES[c][2] / tot for c in GAS_CITIES if norms[c] is not None)
    hdds = [m["hdd"] for m in per_model.values()]
    return {"days": days, "by_model": per_model, "normal": {"hdd": nh, "cdd": nc},
            "hdd_vs_normal_by_model": {k: v["hdd"] - nh for k, v in per_model.items()},
            "model_spread_hdd": (max(hdds) - min(hdds)) if len(hdds) > 1 else None,
            "note": "population-weighted HDD/CDD, base 65°F; more HDD than normal = more heating demand for gas"}


def crop_weather(days: int = 14) -> dict:
    """Corn Belt rain and heat for the next `days` (GFS and ECMWF), with the crop-calendar context."""
    rows = {}
    for label, model in (("GFS (NOAA)", "gfs_seamless"), ("ECMWF IFS", "ecmwf_ifs025")):
        rain, tmax, n = [], [], 0
        for lat, lon in CORN_BELT.values():
            try:
                d = _get("https://api.open-meteo.com/v1/forecast", {"latitude": lat, "longitude": lon,
                                                                     "daily": "precipitation_sum,temperature_2m_max",
                                                                     "precipitation_unit": "inch", "temperature_unit": "fahrenheit",
                                                                     "forecast_days": days, "models": model})
            except Exception:  # noqa: BLE001
                continue
            dd = d.get("daily", {})
            rain.append(sum(x for x in (dd.get("precipitation_sum") or []) if x is not None))
            tmax += [x for x in (dd.get("temperature_2m_max") or []) if x is not None]
            n += 1
        if n:
            rows[label] = {"avg_rain_inches": float(np.mean(rain)), "avg_max_temp_f": float(np.mean(tmax)),
                           "hot_days_over_90f_share": float(np.mean([t > 90 for t in tmax])) if tmax else None}
    m = date.today().month
    stage = {(4, 5): "planting", (6,): "early growth", (7,): "corn pollination (most weather-sensitive)",
             (8,): "soybean pod fill (most weather-sensitive for beans)", (9, 10, 11): "harvest (US weather matters less; "
             "South American planting starts)", (12, 1, 2, 3): "US off-season; Brazil/Argentina crop weather drives"}
    season = next(v for k, v in stage.items() if m in k)
    return {"days": days, "by_model": rows, "crop_stage": season}


# ---------------------------------------------------------------- spot, calendar, keyed hooks

FRED = {"henry_hub": "DHHNGSP", "wti": "DCOILWTICO", "brent": "DCOILBRENTEU"}


def spot_prices() -> dict:
    """Daily spot from FRED's free CSV endpoint (no key)."""
    import requests
    out = {}
    for k, sid in FRED.items():
        try:
            txt = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=20).text
            df = pd.read_csv(io.StringIO(txt))
            df = df[pd.to_numeric(df.iloc[:, 1], errors="coerce").notna()]
            out[k] = {"price": float(df.iloc[-1, 1]), "date": str(df.iloc[-1, 0]), "series": sid}
        except Exception:  # noqa: BLE001
            out[k] = {"unavailable": True}
    return out


def release_calendar(today: date | None = None) -> list[dict]:
    """The next scheduled releases that move natural-world markets (US Eastern times; holidays can shift them)."""
    today = today or date.today()

    def next_wd(wd):   # Monday=0
        return today + timedelta(days=(wd - today.weekday()) % 7 or 7)
    rows = [{"release": "EIA Weekly Natural Gas Storage Report", "when": f"{next_wd(3)} 10:30 ET", "moves": "NG"},
            {"release": "EIA Weekly Petroleum Status Report", "when": f"{next_wd(2)} 10:30 ET", "moves": "CL, RB, HO, crack"},
            {"release": "CFTC Commitments of Traders", "when": f"{next_wd(4)} 15:30 ET", "moves": "positioning, all"}]
    if 4 <= today.month <= 11:
        rows.append({"release": "USDA Crop Progress", "when": f"{next_wd(0)} 16:00 ET", "moves": "ZC, ZS, ZW"})
    first = date(today.year + (today.month == 12), today.month % 12 + 1, 1) if today.day > 12 else today.replace(day=1)
    rows.append({"release": "USDA WASDE", "when": f"around {first.replace(day=10)}-{first.replace(day=12)} 12:00 ET (confirm on "
                 "USDA's schedule)", "moves": "ZC, ZS, ZW"})
    # how big these release hours usually are vs a normal hour (scripts/commod_releases.py, DECISIONS #212)
    try:
        import json
        from pathlib import Path
        R = json.loads((Path(__file__).resolve().parents[2] / "docs/pmrm/results/commod_releases.json").read_text(encoding="utf-8"))
        size = {"EIA Weekly Petroleum Status Report": ["EIA crude stocks | USO"], "EIA Weekly Natural Gas Storage Report": ["EIA gas storage | UNG"],
                "USDA WASDE": ["USDA WASDE | CORN", "USDA WASDE | SOYB", "USDA WASDE | WEAT"]}
        for r in rows:
            got = {k.split(" | ")[1]: round(R[k]["hour"]["size_vs_control_rms"], 1) for k in size.get(r["release"], []) if k in R}
            if got:
                r["first_hour_vs_a_normal_hour"] = got
    except Exception:  # noqa: BLE001
        pass
    return rows


def keyed_hooks() -> dict:
    """Sources that switch on when the trainer adds a free key to .env (never printed)."""
    have = lambda k: bool(os.environ.get(k))
    return {"EIA (inventories, grid load)": "on" if have("EIA_API_KEY") else "off: add EIA_API_KEY (free, eia.gov/opendata)",
            "USDA NASS (crop condition)": "on" if have("NASS_API_KEY") else "off: add NASS_API_KEY (free, quickstats.nass.usda.gov/api)",
            "PJM Data Miner (power prices, congestion)": "on" if have("PJM_API_KEY") else "off: add PJM_API_KEY (free, dataminer2.pjm.com)",
            "ERCOT public API (power prices)": "on" if have("ERCOT_API_KEY") else "off: add ERCOT_API_KEY (free, apiexplorer.ercot.com)"}


def eia_series(series_id: str) -> dict:
    """EIA API v2 by series id (e.g. NG.NW2_EPG0_SWO_R48_BCF.W for Lower-48 storage). Needs EIA_API_KEY."""
    key = os.environ.get("EIA_API_KEY")
    if not key:
        return {"available": False, "how": keyed_hooks()["EIA (inventories, grid load)"]}
    import requests
    d = requests.get(f"https://api.eia.gov/v2/seriesid/{series_id}", params={"api_key": key}, timeout=20).json()
    rows = (d.get("response") or {}).get("data") or []
    return {"available": True, "series": series_id, "latest": rows[:6]}
