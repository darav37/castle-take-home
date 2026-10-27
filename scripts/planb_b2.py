"""Plan B checkpoint B2: event library v0 (BUILD_PLAN B2, PLAN_B §5).

  python scripts/planb_b2.py calendar              # scheduled releases since 2000 + surprise measures (no LLM)
  python scripts/planb_b2.py events                # DRY RUN: prints calls and estimated cost, calls nothing
  python scripts/planb_b2.py events --confirm      # LLM finds and structures unscheduled events (SPEC §6.1 caps)
  python scripts/planb_b2.py responses             # realized responses in the B1 factor space (no LLM)

The LLM only finds, dates and structures events into the fixed schema with cited sources. It never sees or produces
market outcomes; code checks the sources, assigns windows from the first credible report, and computes every number.
Event selection is by news prominence and category coverage, never by market reaction (PLAN_B §5.4).
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey import config  # noqa: E402
from worsey.data.cache import cached_call  # noqa: E402
from worsey.planb import library as LIB  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

OUT = ROOT / "docs" / "pmrm" / "results"
STORE = ROOT / "cache" / "pmrm" / "b2"
MODEL = config.EXPERT_MODEL
PER_CATEGORY = 20
YEARS = (2004, 2026)
PANEL = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "UUP", "USO", "UNG", "GLD", "DBA", "XLE", "XLF", "XLK", "XLI",
         "XLV", "XLP", "XLU", "XLY", "XLB", "KRE", "ITB", "JETS", "SMH", "ITA", "EEM", "FXI", "EWJ", "EWG", "EWW"]
# region text -> country equity ETF(s)
COUNTRY_ETF = {"brazil": "EWZ", "mexico": "EWW", "china": "FXI", "hong kong": "EWH", "japan": "EWJ", "germany": "EWG",
               "united kingdom": "EWU", "britain": "EWU", "uk": "EWU", "france": "EWQ", "italy": "EWI", "spain": "EWP",
               "switzerland": "EWL", "netherlands": "EWN", "taiwan": "EWT", "south korea": "EWY", "korea": "EWY",
               "india": "INDA", "singapore": "EWS", "australia": "EWA", "canada": "EWC", "south africa": "EZA",
               "turkey": "TUR", "poland": "EPOL", "saudi": "KSA", "israel": "EIS", "greece": "GREK", "russia": "ERUS",
               "eurozone": "EZU", "europe": "EZU", "argentina": "ARGT", "chile": "ECH", "indonesia": "EIDO"}


def country_etfs(region: str | None) -> list[str]:
    r = (region or "").lower()
    return sorted({etf for name, etf in COUNTRY_ETF.items() if re.search(r"\b" + re.escape(name) + r"\b", r)})   # whole words: "uk" != "ukraine"


SECTOR_ETF = {"energy": "XLE", "materials": "XLB", "industrials": "XLI", "consumer_discretionary": "XLY",
              "consumer_staples": "XLP", "health_care": "XLV", "financials": "XLF", "banks": "KRE", "insurance": "KIE",
              "tech": "XLK", "semiconductors": "SMH", "software": "IGV", "communications": "IYZ", "utilities": "XLU",
              "real_estate": "IYR", "homebuilders": "ITB", "airlines": "JETS", "transport_shipping": "IYT",
              "autos": "CARZ", "defense": "ITA", "agriculture": "MOO", "metals_mining": "XME", "retail": "XRT",
              "media": "IYZ", "biotech": "XBI", "broad_market": "SPY"}

PROMPT = """You are building a research library of dated real-world events for an event-impact study.

Category: {cat} ({desc}).

List {n} distinct events in this category between {y0} and {y1}, spread across the years (no more than 3 from any
single year). Choose events by how prominent they were in the news and by covering the range of the category, NOT by
how much financial markets moved: include several that had little or no market impact. Prefer events with a clear
first public report date. Use web search to confirm each date and to cite sources.

For each event, the date that matters is the FIRST CREDIBLE PUBLIC REPORT (a leak or first wire story counts), not
when the underlying thing happened or was later confirmed. Give its US Eastern time if a source states it.

Never mention market or price reactions anywhere. Return ONLY a JSON array, no prose, each item with exactly these
fields:
{schema}
"category" must be "{cat}". Use null for unknown fields; never guess a ticker or a time."""


# ---------------------------------------------------------------- calendar

def step_calendar():
    import pmrm_p0b as P
    start = date(2000, 1, 1)
    cal = LIB.release_calendar(start, P.fomc())
    ms = LIB.market_surprises(cal)
    ds = LIB.data_surprises(start)
    lib = ms.merge(ds, on=["date", "release"], how="left")
    STORE.mkdir(parents=True, exist_ok=True)
    lib.to_csv(STORE / "scheduled.csv", index=False)
    print(lib.groupby("release").agg(n=("date", "size"), first=("date", "min"), d2y_sd=("d2y", "std"),
                                     data_surprise_n=("data_surprise", "count")).to_string())


# ---------------------------------------------------------------- unscheduled events (LLM, capped)

def _text(resp: dict) -> str:
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


def _parse(txt: str) -> list[dict]:
    m = re.search(r"\[\s*\{.*\}\s*\]", txt, re.S)   # an array of objects (prose may contain "[]")
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except ValueError:
        return []


def _url_ok(u: str) -> bool:
    def fetch():
        try:
            r = requests.get(u, headers=LIB.UA, timeout=15, allow_redirects=True, stream=True)
            ok = r.status_code < 400
            r.close()
            return {"ok": ok}
        except Exception:  # noqa: BLE001
            return {"ok": False}
    return cached_call("urlcheck", u, None, fetch)["ok"]


def validate(ev: dict, cat: str) -> dict | None:
    try:
        d = date.fromisoformat(str(ev.get("date_first_report"))[:10])
    except ValueError:
        return None
    if not (date(YEARS[0], 1, 1) <= d <= date(YEARS[1], 10, 16)):
        return None
    out = {k: ev.get(k) for k in LIB.SCHEMA}
    out["category"] = cat
    out["date_first_report"] = d.isoformat()
    out["targeted_sectors"] = [s for s in (ev.get("targeted_sectors") or []) if s in LIB.SECTORS]
    out["targeted_tickers"] = [t for t in (ev.get("targeted_tickers") or []) if isinstance(t, str) and re.fullmatch(r"[A-Z.]{1,6}", t)]
    for k, allowed in (("severity", {"minor", "moderate", "major", "extreme"}),
                       ("anticipation", {"surprise", "partly_expected", "widely_expected"})):
        if out.get(k) not in allowed:
            out[k] = None
    out["sources"] = [s for s in (ev.get("sources") or []) if isinstance(s, dict) and str(s.get("url", "")).startswith("http")]
    out["sourcing"] = "cited" if out["sources"] else "llm_knowledge_only"   # the model ran out of searches; kept, flagged
    return out


def wiki_date_check(ev: dict) -> dict:
    """Free code check of the date: does Wikipedia's current-events portal list something matching this event on the
    stated day or the next two? Match = an actor, region or title word (>= 5 letters) in that day's section."""
    d = date.fromisoformat(ev["date_first_report"])
    words = {w.lower() for w in re.findall(r"[A-Za-z]{5,}", " ".join([ev.get("title") or ""] + list(ev.get("actors") or [])
                                                                      + [ev.get("region") or ""]))}
    words -= {"united", "states", "government", "announces", "announced", "against", "between", "national", "president"}
    hits = []
    for k in range(3):
        dd = d + timedelta(days=k)
        try:
            t = LIB.web(f"https://en.wikipedia.org/wiki/Portal:Current_events/{dd:%B}_{dd.year}")
        except Exception:  # noqa: BLE001
            continue
        anchor = f'id="{dd.year}_{dd:%B}_{dd.day}"'
        i = t.find(anchor)
        if i < 0:
            continue
        j = t.find('class="current-events-main', i + 200)          # the next day's section starts here
        sec = re.sub(r"<[^>]+>", " ", t[i:j if j > 0 else i + 60000]).lower()
        found = sorted(w for w in words if w in sec)
        if len(found) >= 2 or (found and len(words) <= 2):
            hits.append({"day": dd.isoformat(), "matched": found[:6]})
            break
    return {"wiki_checked": True, "wiki_match": hits[0] if hits else None}


def step_events(confirm: bool):
    from worsey.agent.llm import AnthropicLLM
    llm = AnthropicLLM() if confirm else None
    run = ICB.Run("planb_b2_events", MODEL, llm, confirmed=confirm)
    plan = run.plan(len(LIB.CATEGORIES), in_tok=60000, out_tok=10000, searches_per_call=6)
    if not confirm:
        return
    if not plan["within_caps"]:
        raise SystemExit("estimate exceeds the per-run caps; refusing (SPEC §6.1)")
    schema = json.dumps(LIB.SCHEMA, indent=1)
    events, raw_n = [], 0
    for cat, desc in LIB.CATEGORIES.items():
        prompt = PROMPT.format(cat=cat, desc=desc, n=PER_CATEGORY, y0=YEARS[0], y1=YEARS[1], schema=schema)
        k = ICB.key("planb_b2_events", {"category": cat, "n": PER_CATEGORY, "years": YEARS}, prompt, MODEL)
        resp = run.call(k, model=MODEL, max_tokens=16000,
                        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
                        messages=[{"role": "user", "content": prompt}])
        items = _parse(_text(resp))
        raw_n += len(items)
        good = [v for v in (validate(e, cat) for e in items) if v]
        print(f"  {cat:26} {len(items)} returned, {len(good)} valid", flush=True)
        events += good
    # dedupe (same date and similar title)
    seen, uniq = set(), []
    for e in sorted(events, key=lambda e: e["date_first_report"]):
        sig = (e["date_first_report"], re.sub(r"[^a-z]", "", (e.get("title") or "").lower())[:20])
        if sig not in seen:
            seen.add(sig)
            uniq.append(e)
    with ThreadPoolExecutor(8) as ex:
        oks = list(ex.map(lambda e: [_url_ok(s["url"]) for s in e["sources"]], uniq))
        wk = list(ex.map(wiki_date_check, uniq))
    for n, (e, o, w) in enumerate(zip(uniq, oks, wk)):
        for s, ok in zip(e["sources"], o):
            s["url_ok"] = ok
        e["sources_ok"] = sum(o)
        e.update(w)
        e["id"] = f"ev{n:04d}"
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "unscheduled.json").write_text(json.dumps(uniq, indent=1), encoding="utf-8")
    summ = run.summary() | {"returned": raw_n, "valid": len(events), "unique": len(uniq),
                            "cited": sum(e["sourcing"] == "cited" for e in uniq),
                            "cited_with_working_link": sum(e["sources_ok"] > 0 for e in uniq),
                            "wikipedia_date_match": sum(bool(e["wiki_match"]) for e in uniq)}
    (STORE / "events_run.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


# ---------------------------------------------------------------- realized responses (B1 factor space)

def step_responses():
    import pmrm_audit as A
    from planb_b1 import total_returns
    with open(ROOT / "cache" / "pmrm" / "b1" / "model.pkl", "rb") as f:
        m = pickle.load(f)
    sessions = m["factors_latent"].index
    F = m["factors_latent"]
    X = (m["anchors"] - m["anchor_mean"]) / m["anchor_sd"]
    sched = pd.read_csv(STORE / "scheduled.csv", parse_dates=["date"])
    uns = json.loads((STORE / "unscheduled.json").read_text(encoding="utf-8")) if (STORE / "unscheduled.json").exists() else []
    for e in uns:
        e.setdefault("origin", "curated_v0")
    wf = STORE / "wiki_events.json"
    if wf.exists():   # library growth (DECISIONS #161): drop near-duplicates of curated events (same category, ±2 days)
        cur = [(date.fromisoformat(e["date_first_report"]), e["category"]) for e in uns]
        added = dropped = 0
        for month in sorted(json.loads(wf.read_text(encoding="utf-8")).items()):
            for e in month[1]:
                d = date.fromisoformat(e["date_first_report"])
                if any(c == e["category"] and abs((d - dd).days) <= 2 for dd, c in cur):
                    dropped += 1
                    continue
                uns.append(e)
                added += 1
        print(f"wikipedia events: {added} added, {dropped} dropped as near-duplicates of curated events", flush=True)
    tick =set(PANEL) | set(SECTOR_ETF.values()) | {t for e in uns for t in e.get("targeted_tickers") or []} | set(COUNTRY_ETF.values())
    R = total_returns(sorted(tick), sessions, "1990-01-01")
    layer = m["stock_layer"]
    macro_days = set(pd.to_datetime(sched["date"]).dt.normalize())

    def first_session(d: pd.Timestamp, after_close: bool) -> int | None:
        i = sessions.searchsorted(d + (pd.Timedelta(days=1) if after_close else pd.Timedelta(0)))
        return int(i) if i < len(sessions) else None

    def window(i: int, n: int) -> dict:
        sl = slice(i, i + n)
        ws = sessions[sl]
        cum = lambda s: float(np.prod(1 + s.iloc[sl].fillna(0)) - 1)
        out = {"start": str(sessions[i - 1].date()), "end": str(ws[-1].date()), "sessions": len(ws),
               "factors_latent": {c: float(F[c].iloc[sl].sum()) for c in F.columns},
               "factors_anchored": {c: float(X[c].iloc[sl].sum()) for c in X.columns},
               "panel": {t: cum(R[t]) for t in PANEL if t in R}, "macro_release_in_window": bool(set(ws) & macro_days)}
        return out

    fitm = (sessions >= "2011-01-03") & (sessions <= pd.Timestamp(m["periods"]["val_end"]))
    extra = {}

    def loadings(t: str) -> dict | None:
        """B1 stock-layer loadings, or for panel ETFs (sectors, countries) the same regression fitted here on 2011-2020."""
        if t in layer and "anchored_loadings" in layer[t]:
            return layer[t]["anchored_loadings"]
        if t not in extra:
            s = R[t]
            d = pd.concat([(s - s[fitm].mean()) / s[fitm].std(), X], axis=1)[fitm].dropna()
            if len(d) < 250:
                extra[t] = None
            else:
                A1 = np.column_stack([np.ones(len(d)), d.iloc[:, 1:].to_numpy()])
                extra[t] = dict(zip(X.columns, np.linalg.lstsq(A1, d.iloc[:, 0].to_numpy(), rcond=None)[0][1:]))
        return extra[t]

    def abnormal(t: str, i: int, n: int) -> float | None:
        """Beyond-factor return of a targeted ticker: realized minus its anchored-factor prediction (B1 loadings)."""
        if t not in R:
            return None
        b = loadings(t)
        if b is None:
            return None
        s = R[t]
        sd = s.loc[: m["periods"]["val_end"]].std()
        pred_z = sum(X[c].iloc[i:i + n].fillna(0) * b[c] for c in X.columns)
        return float(s.iloc[i:i + n].sum() - (pred_z * sd).sum())

    recs = []
    for _, r in sched.iterrows():
        i = first_session(r["date"].normalize(), after_close=False)
        if i is None or i < 1:
            continue
        recs.append({"id": f"{r['release']}-{r['date'].date()}", "kind": "scheduled", "category": "macro_release",
                     "release": r["release"], "date": str(r["date"].date()),
                     "surprise": {k: (None if pd.isna(r.get(k)) else float(r[k])) for k in ("d2y", "d10y", "d5y_breakeven", "data_surprise", "data_surprise_z")},
                     "w1": window(i, 1), "w5": window(i, 5)})
    for e in uns:
        d = pd.Timestamp(e["date_first_report"])
        known = isinstance(e.get("time_first_report_et"), str) and bool(re.fullmatch(r"\d{1,2}:\d{2}", e["time_first_report_et"]))
        after = known and e["time_first_report_et"].zfill(5) >= "16:00"
        i = first_session(d, after)
        if i is None or i < 1:
            continue
        n1 = 1 if known else 2      # unknown time: the report day's session and the next, so after-close news is inside
        sect = {s: SECTOR_ETF[s] for s in e.get("targeted_sectors") or [] if s in SECTOR_ETF}
        rec = {**{k: e.get(k) for k in ("id", "title", "category", "date_first_report", "severity", "anticipation",
                                        "targeted_sectors", "targeted_tickers", "region", "prediction_market", "direction", "origin")},
               "kind": "unscheduled", "date": str(sessions[i].date()), "w1": window(i, n1), "w5": window(i, 5),
               "time_known": known,
               "targeted": {t: {"abnormal_w1": abnormal(t, i, n1), "abnormal_w5": abnormal(t, i, 5)}
                            for t in set(e.get("targeted_tickers") or []) | set(sect.values()) | set(country_etfs(e.get("region")))},
               "country_etfs": country_etfs(e.get("region")),
               "anticipation_measure": "lower bound (no prediction market price recorded)"}
        recs.append(rec)
    (STORE / "library.json").write_text(json.dumps(recs, default=float), encoding="utf-8")
    print(f"{len(recs)} records ({sum(r['kind'] == 'scheduled' for r in recs)} scheduled)")


# ---------------------------------------------------------------- direction labels (LLM, capped;)

DIRECTION_FIELDS = {
    "escalation": ["escalation", "de-escalation", "not_applicable"],
    "growth_shock": ["negative", "positive", "none"],
    "inflation_shock": ["up", "down", "none"],
    "oil_supply": ["disruption", "increase", "none"],
    "policy_stance": ["tighter", "looser", "none"],
    "risk_sentiment": ["risk_off", "risk_on", "neutral"],
}
SIGN_FACTORS = ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit"]
DIRECTION_PROMPT = """For each event below, label (A) the DIRECTION OF THE EVENT ITSELF along fixed economic channels,
judged from what the event is (its first report), and (B) separately, your expectation of the sign of each market
factor's move in the first sessions after it.

(A) fields and allowed values:
{fields}
(B) for each of {factors}: "+", "-" or "0" ("credit" = high-yield bonds minus Treasuries: "+" means credit spreads
tighten; "rates" = Treasury yields).

Answer from the description as of the first report. Return ONLY a JSON array with one object per event:
{{"id": ..., {afields}, "factor_signs": {{"market": "+|-|0", ...}}}}

Events:
{events}"""


def step_direction(confirm: bool):
    from worsey.agent.llm import AnthropicLLM
    uns = json.loads((STORE / "unscheduled.json").read_text(encoding="utf-8"))
    batches = [uns[i:i + 20] for i in range(0, len(uns), 20)]
    run = ICB.Run("planb_b2_direction", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    plan = run.plan(len(batches), in_tok=6000, out_tok=6000)
    if not confirm:
        return
    if not plan["within_caps"]:
        raise SystemExit("estimate exceeds the per-run caps; refusing (SPEC §6.1)")
    fields = "\n".join(f"- {k}: {' | '.join(v)}" for k, v in DIRECTION_FIELDS.items())
    afields = ", ".join(f'"{k}": ...' for k in DIRECTION_FIELDS)
    labels = {}
    for b in batches:
        evs = json.dumps([{"id": e["id"], "title": e["title"], "category": e["category"], "region": e.get("region"),
                           "description": e.get("description")} for e in b], indent=0)
        prompt = DIRECTION_PROMPT.format(fields=fields, factors=", ".join(SIGN_FACTORS), afields=afields, events=evs)
        k = ICB.key("planb_b2_direction", {"ids": [e["id"] for e in b]}, prompt, MODEL)
        resp = run.call(k, model=MODEL, max_tokens=12000, messages=[{"role": "user", "content": prompt}])
        for it in _parse(_text(resp)):
            if not isinstance(it, dict) or it.get("id") not in {e["id"] for e in b}:
                continue
            rec = {f: (it.get(f) if it.get(f) in vals else None) for f, vals in DIRECTION_FIELDS.items()}
            fs = it.get("factor_signs") or {}
            rec["factor_signs"] = {f: {"+": 1, "-": -1, "0": 0}.get(str(fs.get(f)), None) for f in SIGN_FACTORS}
            labels[it["id"]] = rec
    for e in uns:
        e["direction"] = labels.get(e["id"])
    (STORE / "unscheduled.json").write_text(json.dumps(uns, indent=1), encoding="utf-8")
    summ = run.summary() | {"labelled": sum(1 for e in uns if e.get("direction"))}
    (STORE / "direction_run.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calendar", "events", "responses", "direction"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    {"calendar": step_calendar, "responses": step_responses,
     "direction": lambda: step_direction(a.confirm)}.get(a.step, lambda: step_events(a.confirm))()


if __name__ == "__main__":
    main()
