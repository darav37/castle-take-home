"""Plan B checkpoint B0: data audit (docs/pmrm/BUILD_PLAN.md B0, PLAN_B §8). No LLM calls.

  python scripts/pmrm_b0_audit.py

Probes every data source Plan B needs, records whether it is reachable from this machine without new keys, how far
back it goes, and what it would take to use it. Writes docs/pmrm/results/b0_audit.json (the .md is written from it).
Every fetch goes through the disk cache (source "web" for public pages, the existing clients otherwise).
"""

from __future__ import annotations

import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pmrm_audit as A  # noqa: E402
from worsey.data import massive  # noqa: E402
from worsey.data.cache import cached_call  # noqa: E402

OUT = ROOT / "docs" / "pmrm" / "results"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

# Broad factor universe (PLAN_B §5.2): candidates, audited for history; the final list is frozen at B1.
UNIVERSE = {
    "us_sectors": ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
    "us_industries": ["KBE", "KRE", "KIE", "IAI", "XHB", "ITB", "XRT", "XME", "XOP", "OIH", "XES", "XBI", "IBB", "XPH",
                      "IHI", "XHS", "SOXX", "SMH", "IGV", "XSW", "FDN", "SKYY", "HACK", "IYZ", "XTL", "ITA", "XAR",
                      "IYT", "JETS", "XTN", "PAVE", "TAN", "ICLN", "URA", "LIT", "COPX", "GDX", "SIL", "SLX", "WOOD",
                      "MOO", "PBJ", "PEJ", "REM", "VNQ", "IYR", "MLPX", "AMLP", "PHO", "IPAY", "CARZ", "BOTZ"],
    "commodities": ["USO", "BNO", "UNG", "UGA", "DBA", "CORN", "WEAT", "SOYB", "GLD", "SLV", "CPER", "PPLT", "PALL", "DBB", "DBC"],
    "currencies": ["UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "FXF", "CYB", "CEW"],
    "country_equity": ["EWJ", "EWG", "EWU", "EWQ", "EWI", "EWP", "EWL", "EWN", "FXI", "MCHI", "KWEB", "EWH", "EWT",
                       "EWY", "INDA", "EWS", "EWA", "EWC", "EWZ", "EWW", "EZA", "TUR", "EPOL", "KSA", "ISRA", "EIS"],
    "rates_credit": ["SHY", "IEI", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "MBB", "BND"],
    "broad_equity": ["SPY", "QQQ", "IWM", "DIA", "VTV", "VUG", "MTUM", "USMV", "QUAL"],
}


def web(url: str, params: dict | None = None, text: bool = True, browser: bool = True) -> dict:
    """Cached GET: {status, text|len}. Transport errors are returned, never raised, and never cached.
    FRED drops connections from a browser user-agent, so its probes use the default one."""
    def fetch():
        r = requests.get(url, params=params, headers=UA if browser else None, timeout=45)
        return {"status": r.status_code, "text": r.text[:400000] if text else "", "len": len(r.content),
                "ctype": r.headers.get("content-type", "")}
    try:
        return cached_call("web", url + ("" if browser else "#plain"), params, fetch)
    except Exception as e:  # noqa: BLE001
        return {"status": None, "error": f"{type(e).__name__}: {e}"[:200], "text": "", "len": 0}


# ---------------------------------------------------------------- 1. FRED / ALFRED

def fred() -> dict:
    out = {}
    for sid in ("DGS2", "DGS10", "CPIAUCSL", "PAYEMS", "PCEPI", "GDPC1", "SOFR", "DFF"):
        r = web("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid}, browser=False)
        rows = [l for l in r["text"].splitlines()[1:] if l and not l.endswith(",.")]
        out[sid] = {"status": r["status"], "rows": len(rows), "first": rows[0].split(",")[0] if rows else None,
                    "last": rows[-1].split(",")[0] if rows else None}
    # first releases need ALFRED vintages: keyless CSV gives only the latest vintage
    api = web("https://api.stlouisfed.org/fred/series/vintagedates", {"series_id": "PAYEMS", "file_type": "json"}, browser=False)
    alf = web("https://alfred.stlouisfed.org/series", {"seid": "PAYEMS"}, browser=False)
    out["_alfred"] = {"api_without_key_status": api["status"], "api_message": api["text"][:160],
                      "alfred_page_status": alf["status"],
                      "note": "first releases (vintages) need the FRED API (free key: FRED_API_KEY) or ALFRED downloads"}
    return out


# ---------------------------------------------------------------- 2. nowcasts

def nowcasts() -> dict:
    g = web("https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx", text=False)
    g2 = web("https://www.atlantafed.org/cqer/research/gdpnow")
    c = web("https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting")
    out = {"gdpnow_xlsx": {"status": g["status"], "bytes": g["len"], "ctype": g.get("ctype")},
           "gdpnow_page": {"status": g2["status"]},
           "cleveland_nowcast_page": {"status": c["status"]}}
    # look for a downloadable archive link on the pages
    for k, r in (("gdpnow_links", g2), ("cleveland_links", c)):
        out[k] = sorted(set(re.findall(r'href="([^"]+\.(?:xlsx|csv|xls|json))"', r["text"])))[:10]
    return out


# ---------------------------------------------------------------- 3. futures history (Massive)

def futures() -> dict:
    out = {}
    for prod in ("ZQ", "SR3", "ZN", "ZT", "CL", "NG", "GC", "ES"):
        try:
            snap = massive.futures_snapshot(prod, max_pages=2)
        except Exception as e:  # noqa: BLE001
            out[prod] = {"error": f"{type(e).__name__}"[:80]}
            continue
        tick = [r.get("ticker") or (r.get("details") or {}).get("ticker") for r in snap]
        tick = [t for t in tick if t]
        res = {"live_contracts": len(tick), "sample": tick[:3]}
        if tick:
            t = sorted(tick)[0]
            try:
                d = massive._get(f"/futures/v1/aggs/{t}", {"limit": 50000, "sort": "asc"})
                rows = d.get("results") or []
                res["front_aggs"] = {"ticker": t, "rows": len(rows),
                                     "first": str(rows[0].get("window_start") or rows[0].get("t") or rows[0].get("session_end_date")) if rows else None}
            except Exception as e:  # noqa: BLE001
                res["front_aggs"] = {"error": f"{type(e).__name__}: {e}"[:120]}
        try:
            c = massive._get("/futures/v1/contracts", {"product_code": prod, "active": "false", "limit": 1000})
            rows = c.get("results") or []
            res["expired_contracts_listed"] = len(rows)
            res["oldest_expired"] = min((r.get("last_trade_date") or r.get("settlement_date") or "9999") for r in rows) if rows else None
        except Exception as e:  # noqa: BLE001
            res["expired_contracts"] = f"{type(e).__name__}: {e}"[:120]
        out[prod] = res
    return out


# ---------------------------------------------------------------- 4. Kalshi strike ladders per release

def ladders() -> dict:
    out = {}
    for series in ("KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXGDP", "KXFEDDECISION", "KXPCECORE", "KXU3"):
        ev = {}
        for m in A.all_markets(series):
            ev.setdefault(m["event"], []).append(m)
        sizes = [len(v) for v in ev.values()]
        vol = [sum(x["volume"] for x in v) for v in ev.values()]
        firsts = sorted(min(x["close"] for x in v) for v in ev.values() if any(x["close"] for x in v))
        out[series] = {"events": len(ev), "first_release": str(firsts[0].date()) if firsts else None,
                       "strikes_median": float(pd.Series(sizes).median()) if sizes else 0,
                       "events_ge_4_strikes": int(sum(s >= 4 for s in sizes)),
                       "events_ge_6_strikes": int(sum(s >= 6 for s in sizes)),
                       "median_event_volume": float(pd.Series(vol).median()) if vol else 0}
    return out


# ---------------------------------------------------------------- 5-6. consensus and event-library sources

def sources() -> dict:
    probes = {
        "gdelt_doc_api": ("https://api.gdeltproject.org/api/v2/doc/doc", {"query": "tariff", "mode": "timelinevolraw", "format": "json", "timespan": "1y"}),
        "gdelt_events_index": ("http://data.gdeltproject.org/events/index.html", None),
        "gdelt_v2_masterlist": ("http://data.gdeltproject.org/gdeltv2/masterfilelist.txt", None),
        "federal_register_api": ("https://www.federalregister.gov/api/v1/documents.json", {"per_page": 1, "conditions[term]": "tariff", "order": "oldest"}),
        "ustr_press": ("https://ustr.gov/about-us/policy-offices/press-office/press-releases", None),
        "ofac_recent_actions": ("https://ofac.treasury.gov/recent-actions", None),
        "ofac_sdn_csv": ("https://www.treasury.gov/ofac/downloads/sdn.csv", None),
        "emdat": ("https://public.emdat.be/", None),
        "acled_api": ("https://api.acleddata.com/acled/read", {"limit": 1}),
        "wikipedia_current_events": ("https://en.wikipedia.org/wiki/Portal:Current_events/October_2008", None),
        "philly_fed_spf": ("https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters", None),
        "iem_markets": ("https://iemweb.biz.uiowa.edu/pricehistory/", None),
        "predictit_api": ("https://www.predictit.org/api/marketdata/all/", None),
        "betfair_historic": ("https://historicdata.betfair.com/", None),
    }
    out = {}
    for k, (url, params) in probes.items():
        r = web(url, params)
        info = {"url": url, "status": r["status"], "bytes": r["len"]}
        if r.get("error"):
            info["error"] = r["error"]
        t = r["text"]
        if k == "federal_register_api" and r["status"] == 200:
            try:
                j = json.loads(t)
                info["count_tariff_docs"] = j.get("count")
                info["oldest"] = (j.get("results") or [{}])[0].get("publication_date")
            except ValueError:
                pass
        if k == "gdelt_v2_masterlist" and r["status"] == 200:
            lines = t.splitlines()
            info["first_file"] = lines[0].split()[-1][-40:] if lines else None
        if k == "gdelt_doc_api" and r["status"] == 200:
            info["json"] = t.strip().startswith("{")
        if k == "acled_api":
            info["note"] = t[:160]
        out[k] = info
    return out


# ---------------------------------------------------------------- 7. broad-universe price history

def universe() -> dict:
    out = {}
    for group, ts in UNIVERSE.items():
        for t in ts:
            try:
                b = massive.daily_bars(t, "1990-01-01", A.AUDIT_END.isoformat(), trim_to_listing=False)
                out[t] = {"group": group, "first": str(b.index.min().date()) if len(b) else None, "rows": int(len(b))}
            except Exception as e:  # noqa: BLE001
                out[t] = {"group": group, "error": f"{type(e).__name__}"[:60]}
    fx = {}
    for pair in ("C:EURUSD", "C:USDJPY", "C:GBPUSD", "C:USDCNH", "C:USDMXN"):
        try:
            d = massive._get(f"/v2/aggs/ticker/{pair}/range/1/day/1990-01-01/{A.AUDIT_END.isoformat()}",
                             {"adjusted": "true", "sort": "asc", "limit": 50000})
            rows = d.get("results") or []
            fx[pair] = {"rows": len(rows), "first": str(pd.to_datetime(rows[0]["t"], unit="ms").date()) if rows else None}
        except Exception as e:  # noqa: BLE001
            fx[pair] = {"error": f"{type(e).__name__}"[:60]}
    return {"etfs": out, "fx_spot": fx}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"audit_end": A.AUDIT_END.isoformat()}
    for name, fn in (("fred", fred), ("nowcasts", nowcasts), ("futures", futures), ("kalshi_ladders", ladders),
                     ("sources", sources), ("universe", universe)):
        print(name, "...", flush=True)
        try:
            rep[name] = fn()
        except Exception as e:  # noqa: BLE001
            rep[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(rep[name], default=str)[:600], flush=True)
    (OUT / "b0_audit.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT / "b0_audit.json")


if __name__ == "__main__":
    main()
