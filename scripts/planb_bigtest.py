"""Big events: magnitude labels, first-report times, intraday windows and a blinded test (DECISIONS #176). Every LLM call is capped (SPEC §6.1 + $70 project cap) with a dry run first.

  python scripts/planb_bigtest.py label [--confirm]     # LLM: channel, magnitude, exposure, report time, anonymized text
  python scripts/planb_bigtest.py intraday              # code: minute-bar windows around in-hours first reports
  python scripts/planb_bigtest.py blind [--confirm]     # LLM: raw and assisted predictions from anonymized text only
  python scripts/planb_bigtest.py score                 # code: model / raw / assisted / magnitude-aware, daily and intraday

Events: every major or extreme unscheduled event in the library (2004-2026).
Hindsight control: the blinded predictions see only an anonymized description (no names, places, dates) and, for the
assisted LLM, anonymized analogs (category, severity, channel, magnitude and realized buckets; no titles or dates).
Writes cache/pmrm/bigtest/*.json and docs/pmrm/results/bigtest.json.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import planb_b4_threeway as B4  # noqa: E402
from planb_b2 import MODEL, _parse, _text  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

DIR = ROOT / "cache/pmrm/bigtest"
CHANNELS = ["oil_supply", "rates_fed", "credit_banking", "trade_flows", "risk_sentiment", "sector_specific", "fx_capital_flows", "none_clear"]
MAG = ["small", "medium", "large", "very_large"]
LABEL_PROMPT = """For each news event below, record facts known AT ITS FIRST PUBLIC REPORT (never any later market reaction or
later developments):
- "channel": the main way it could reach financial markets, one of: {channels}
- "magnitude": the size of the shock itself (not of any market move), one of: {mags}
- "quantity": the key number if the text or common knowledge of the event gives one at first report, as
  {{"value": number, "unit": "..."}} (e.g. barrels per day offline, tariff rate and goods value, bank assets, deaths,
  rate change in bp); null if none
- "exposed": up to 4 sectors or countries most directly exposed
- "time_et": the first report's US Eastern time as "HH:MM" if you know it with reasonable confidence, else null
- "time_confidence": high | medium | low | null
- "anon": a one- or two-sentence description with NO names of people, organizations, places or dates (e.g. "a large
  regional bank in an advanced economy fails after a deposit run"), keeping the type, scale and mechanism
Return ONLY a JSON array: [{{"id": ..., "channel": ..., "magnitude": ..., "quantity": ..., "exposed": [...],
"time_et": ..., "time_confidence": ..., "anon": "..."}}]

Events:
{events}"""

RAW_BLIND = """You are estimating how markets react to news events. Each event is described generically (no names, places or
dates). For each, as of its first public report, predict the move over the next one to two trading sessions for:
{factors}
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up ("flat" = within normal daily noise).
Return ONLY a JSON array: [{{"id": ..., "pred": {{"market": "...", ...}}}}]

Events:
{events}"""

ASSIST_BLIND = """You are estimating how markets react to news events. Each event is described generically (no names, places or
dates). For each, as of its first public report, predict the move over the next one to two trading sessions for:
{factors}
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up.

Each event also comes with evidence from a quantitative model built on thousands of past events: its severity, channel
and magnitude; the historical share of each bucket for events of the same severity; the model's estimate per factor
(scaled to the event's severity) and how much similar past events disagreed (band); the market regime at the time;
and five similar EARLIER events (described only by type, channel, magnitude) with how each factor actually moved.
Use the evidence and your judgment. Return ONLY a JSON array: [{{"id": ..., "pred": {{"market": "...", ...}}}}]

Events:
{events}"""


def load():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    X = ((ext["anchors"] - ext["anchor_mean"]) / ext["anchor_sd"]).fillna(0)
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    uns = sorted([r for r in lib if r["kind"] == "unscheduled"], key=lambda r: r["date"])
    src = {}
    for e in json.loads((ROOT / "cache/pmrm/b2/unscheduled.json").read_text(encoding="utf-8")):
        src[e["id"]] = e
    wf = ROOT / "cache/pmrm/b2/wiki_events.json"
    if wf.exists():
        for evs in json.loads(wf.read_text(encoding="utf-8")).values():
            for e in evs:
                src[e["id"]] = e
    return X, uns, src


def big_ids(uns):
    return [r["id"] for r in uns if r.get("severity") in ("major", "extreme")]


def step_label(confirm):
    from worsey.agent.llm import AnthropicLLM
    X, uns, src = load()
    ids = big_ids(uns)
    batches = [ids[i:i + 20] for i in range(0, len(ids), 20)]
    run = ICB.Run("planb_bigtest_label", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    run.plan(len(batches), in_tok=4000, out_tok=5000)
    print(f"{len(ids)} major/extreme events in {len(batches)} batches")
    if not confirm:
        return
    DIR.mkdir(parents=True, exist_ok=True)
    out = json.loads((DIR / "labels.json").read_text(encoding="utf-8")) if (DIR / "labels.json").exists() else {}
    for b in batches:
        evs = [{"id": i, "date": src.get(i, {}).get("date_first_report"), "category": src.get(i, {}).get("category"),
                "region": src.get(i, {}).get("region"), "description": (src.get(i, {}).get("description") or src.get(i, {}).get("title") or "")[:500]}
               for i in b]
        prompt = LABEL_PROMPT.format(channels=" | ".join(CHANNELS), mags=" | ".join(MAG), events=json.dumps(evs, indent=0, default=str))
        k = ICB.key("planb_bigtest_label", {"ids": b}, prompt, MODEL)
        try:
            resp = run.call(k, model=MODEL, max_tokens=8000, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as e:
            print("stopped:", e)
            break
        for it in _parse(_text(resp)):
            if isinstance(it, dict) and it.get("id") in b:
                it["channel"] = it.get("channel") if it.get("channel") in CHANNELS else "none_clear"
                it["magnitude"] = it.get("magnitude") if it.get("magnitude") in MAG else None
                out[it["id"]] = it
    (DIR / "labels.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(run.summary() | {"labelled": len(out),
                                       "with_time": sum(1 for v in out.values() if v.get("time_et")),
                                       "with_quantity": sum(1 for v in out.values() if v.get("quantity"))}))


INTRADAY = {"market": [("SPY", 1)], "rates_10y": [("IEF", -1)], "dollar": [("UUP", 1)], "oil": [("USO", 1)],
            "gold": [("GLD", 1)], "credit": [("HYG", 1), ("IEF", -1)], "defensive_vs_producers": [("XLU", 1), ("XLE", -1)],
            "asia_em": [("EEM", 1), ("SPY", -1)]}


def step_intraday():
    from worsey.data import massive
    import planb_b1 as B1
    X, uns, src = load()
    labels = json.loads((DIR / "labels.json").read_text(encoding="utf-8"))
    tick = sorted({t for v in INTRADAY.values() for t, _ in v})
    sessions = X.index
    R = B1.total_returns(tick, sessions, "1990-01-01")
    dsd = R.loc["2011":"2020"].std()
    out = {}
    for e in uns:
        lab = labels.get(e["id"])
        if not lab or not lab.get("time_et") or lab.get("time_confidence") == "low":
            continue
        try:
            hh, mm = map(int, str(lab["time_et"]).split(":")[:2])
        except ValueError:
            continue
        day = pd.Timestamp(src.get(e["id"], {}).get("date_first_report") or e["date"])
        t0 = pd.Timestamp(day.year, day.month, day.day, hh, mm).tz_localize("America/New_York")
        open_, close = t0.normalize() + pd.Timedelta(hours=9, minutes=30), t0.normalize() + pd.Timedelta(hours=16)
        if not (open_ <= t0 <= close - pd.Timedelta(minutes=30)) or t0.normalize().tz_localize(None) not in sessions:
            continue
        t_end = min(t0 + pd.Timedelta(minutes=120), close)
        minutes = (t_end - t0).total_seconds() / 60
        moves = {}
        for t in tick:
            try:
                rows = massive._get(f"/v2/aggs/ticker/{t}/range/1/minute/{day.date()}/{day.date()}",
                                    {"adjusted": "true", "sort": "asc", "limit": 50000}).get("results") or []
            except Exception:  # noqa: BLE001
                rows = []
            if not rows:
                continue
            ts = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert("America/New_York")
            c = pd.Series([r["c"] for r in rows], index=ts)
            a, b = c[c.index <= t0 - pd.Timedelta(minutes=5)], c[c.index <= t_end]
            if len(a) and len(b):
                moves[t] = b.iloc[-1] / a.iloc[-1] - 1
        z = {}
        for f, legs in INTRADAY.items():
            if all(t in moves for t, _ in legs):
                r = sum(s * moves[t] for t, s in legs)
                scale = np.sqrt(sum(dsd[t] ** 2 for t, _ in legs)) * np.sqrt(minutes / 390)
                z[f] = float(r / scale)
        if len(z) == len(INTRADAY):
            out[e["id"]] = {"t0": str(t0), "minutes": minutes, "z": z}
    (DIR / "intraday.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"intraday windows for {len(out)} events")


def evidence():
    """Daily responses, buckets, model and analogs as in B4 (time-ordered, earlier events only)."""
    X, uns, src = load()
    labels = json.loads((DIR / "labels.json").read_text(encoding="utf-8"))
    ss = X.index
    fi = [list(X.columns).index(f) for f in B4.FACTORS]

    def resp(r):
        a = ss.searchsorted(pd.Timestamp(r["w1"]["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(r["w1"]["end"])) + 1
        return X.iloc[a:b].to_numpy().sum(0)[fi]

    U = np.array([resp(r) for r in uns])
    dates = pd.to_datetime(pd.Series([r["date"] for r in uns]))
    pre = (dates < "2016-01-01").to_numpy()
    sd = U[pre].std(0)
    zU = U / sd
    sev = np.array([r.get("severity") or "moderate" for r in uns])
    cats = sorted({r["category"] for r in uns})
    Fd = np.column_stack([np.ones(len(uns)), np.array([IM.direction_features(e) for e in uns]),
                          np.array([[r["category"] == c for c in cats] for r in uns], float)])
    rates = {}
    for sv in ("major", "extreme"):
        mk = pre & (sev == sv)
        rates[sv] = {f: {B4.BUCKETS[b]: round(float(np.mean(np.searchsorted(B4.CUTS, zU[mk, j]) == b)), 2) for b in range(5)}
                     for j, f in enumerate(B4.FACTORS)}
    bval = np.zeros((5, len(B4.FACTORS)))
    for j in range(len(B4.FACTORS)):
        bins = np.searchsorted(B4.CUTS, zU[pre, j])
        for b in range(5):
            v = U[pre, j][bins == b]
            bval[b, j] = v.mean() if len(v) else 0.0
    return X, uns, src, labels, U, zU, sd, dates, sev, Fd, rates, bval


def step_blind(confirm):
    from worsey.agent.llm import AnthropicLLM
    X, uns, src, labels, U, zU, sd, dates, sev, Fd, rates, bval = evidence()
    regs = json.loads((ROOT / "cache/pmrm/b2/regimes.json").read_text(encoding="utf-8"))
    idx = [i for i, r in enumerate(uns) if r["id"] in labels and labels[r["id"]].get("anon") and dates[i] >= pd.Timestamp("2008-01-01")]
    bucket = lambda z: B4.BUCKETS[int(np.searchsorted(B4.CUTS, z))]
    model, scaled, analog_txt, band = {}, {}, {}, {}
    cache = {}
    for i in idx:
        key = str(dates[i])[:7]
        if key not in cache:
            tr = (dates < pd.Timestamp(key + "-01")).to_numpy()
            B = IM.ridge(Fd[tr], U[tr], 2.0)
            Ptr = Fd[tr] @ B
            scl = {}
            for sv in ("major", "extreme"):
                m = sev[tr] == sv
                x, y = Ptr[m].ravel(), U[tr][m].ravel()
                scl[sv] = float(np.clip((x @ y) / max(x @ x, 1e-9), 0, 6)) if m.sum() >= 20 else 1.0
            cache[key] = (B, tr, scl)
        B, tr, scl = cache[key]
        model[i] = Fd[i] @ B
        scaled[i] = model[i] * scl.get(sev[i], 1.0)
        earlier = np.where(tr)[0]
        sims = np.array([IM.direction_similarity(uns[i], uns[j]) for j in earlier])
        top = earlier[np.argsort(-sims)[:5]]
        band[i] = "wide" if np.mean(np.abs(zU[top] - zU[top].mean(0))) > 0.8 else "narrow"
        analog_txt[i] = [{"type": uns[k]["category"], "severity": uns[k].get("severity"),
                          "channel": labels.get(uns[k]["id"], {}).get("channel"), "magnitude": labels.get(uns[k]["id"], {}).get("magnitude"),
                          "moved": {f: bucket(zU[k, j]) for j, f in enumerate(B4.FACTORS)}} for k in top]
    ftxt = "\n".join(f"- {f}: {B4.FACTOR_TEXT[f]}" for f in B4.FACTORS)
    batches = [idx[a:a + 20] for a in range(0, len(idx), 20)]
    run = ICB.Run("planb_bigtest_blind", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    run.plan(len(batches), in_tok=2500, out_tok=2500)
    run.plan(len(batches), in_tok=9000, out_tok=2500)
    print(f"{len(idx)} events (2008+, labelled) in {len(batches)} batches per mode")
    if not confirm:
        return
    preds = json.loads((DIR / "blind_preds.json").read_text(encoding="utf-8")) if (DIR / "blind_preds.json").exists() else {}
    for b in batches:
        for mode, tmpl in (("raw", RAW_BLIND), ("assisted", ASSIST_BLIND)):
            evs = []
            for i in b:
                lab = labels[uns[i]["id"]]
                d = {"id": uns[i]["id"], "description": lab["anon"]}
                if mode == "assisted":
                    d.update({"severity": sev[i], "channel": lab.get("channel"), "magnitude": lab.get("magnitude"),
                              "typical_moves_for_this_severity": rates.get(sev[i]),
                              "model_estimate": {f: bucket(scaled[i][j] / sd[j]) for j, f in enumerate(B4.FACTORS)}, "band": band[i],
                              "regime": regs.get(uns[i]["id"], {}).get("labels"), "similar_earlier_events": analog_txt[i]})
                evs.append(d)
            prompt = tmpl.format(factors=ftxt, events=json.dumps(evs, indent=0, default=str))
            k = ICB.key("planb_bigtest_blind", {"mode": mode, "ids": [uns[i]["id"] for i in b]}, prompt, MODEL)
            try:
                resp = run.call(k, model=MODEL, max_tokens=5000, messages=[{"role": "user", "content": prompt}])
            except ICB.RunCapExceeded as e:
                print("stopped:", e)
                (DIR / "blind_preds.json").write_text(json.dumps(preds), encoding="utf-8")
                return
            for it in _parse(_text(resp)):
                if isinstance(it, dict) and it.get("pred"):
                    preds.setdefault(it["id"], {})[mode] = it["pred"]
    for i in idx:
        p = preds.setdefault(uns[i]["id"], {})
        p["model"], p["model_scaled"] = model[i].tolist(), scaled[i].tolist()
    (DIR / "blind_preds.json").write_text(json.dumps(preds), encoding="utf-8")
    print(json.dumps(run.summary() | {"events_with_preds": sum(1 for v in preds.values() if "raw" in v and "assisted" in v)}))


def step_score():
    X, uns, src, labels, U, zU, sd, dates, sev, Fd, rates, bval = evidence()
    preds = json.loads((DIR / "blind_preds.json").read_text(encoding="utf-8"))
    intr = json.loads((DIR / "intraday.json").read_text(encoding="utf-8")) if (DIR / "intraday.json").exists() else {}
    ids = {r["id"]: i for i, r in enumerate(uns)}
    have = [k for k, v in preds.items() if "raw" in v and "assisted" in v and k in ids]
    toval = lambda p: [bval[B4.BUCKETS.index(p.get(f)), j] if p.get(f) in B4.BUCKETS else 0.0 for j, f in enumerate(B4.FACTORS)]
    rng = np.random.default_rng(1729)
    out = {"events": len(have)}

    def table(rows, Yget, name):
        Y = np.array([Yget(k) for k in rows])
        P = {"model": np.array([preds[k]["model"] for k in rows]), "model (severity-scaled)": np.array([preds[k]["model_scaled"] for k in rows]),
             "raw LLM (blinded)": np.array([toval(preds[k]["raw"]) for k in rows]),
             "assisted LLM (blinded)": np.array([toval(preds[k]["assisted"]) for k in rows])}
        res = {"n": len(rows)}
        for m, Pm in P.items():
            res[m] = float(1 - ((Y - Pm) ** 2).sum() / (Y ** 2).sum())
            nz = Pm[:, 0] != 0
            res[m + " | market direction"] = float(np.mean(np.sign(Pm[nz, 0]) == np.sign(Y[nz, 0]))) if nz.any() else None
        diffs = []
        for _ in range(2000):
            ix = rng.integers(0, len(rows), len(rows))
            diffs.append((((Y[ix] - P["raw LLM (blinded)"][ix]) ** 2).sum() - ((Y[ix] - P["assisted LLM (blinded)"][ix]) ** 2).sum()) / (Y[ix] ** 2).sum())
        res["assisted - raw (90% range)"] = [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))]
        out[name] = res
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)

    table(have, lambda k: U[ids[k]], "daily window, all big events")
    for sv in ("extreme", "major"):
        table([k for k in have if sev[ids[k]] == sv], lambda k: U[ids[k]], f"daily window, {sv}")
    table([k for k in have if dates[ids[k]] >= pd.Timestamp("2026-01-01")], lambda k: U[ids[k]], "daily window, 2026 only")
    hi = [k for k in have if k in intr]
    if hi:
        table(hi, lambda k: np.array([intr[k]["z"][f] for f in B4.FACTORS]) * sd, "intraday window (2h after first report)")
        # how much of an intraday move is the event? variance vs normal 2-hour windows (z already in normal-window sd units)
        Z = np.array([[intr[k]["z"][f] for f in B4.FACTORS] for k in hi])
        out["intraday_variance_vs_normal"] = {f: float((Z[:, j] ** 2).mean()) for j, f in enumerate(B4.FACTORS)}
        print("intraday variance vs normal 2h windows:", {f: round(v, 2) for f, v in out["intraday_variance_vs_normal"].items()})
    W = json.loads((DIR / "windows.json").read_text(encoding="utf-8")) if (DIR / "windows.json").exists() else {}
    tw = [k for k in have if k in W]
    if tw:   # timing at scale (#177): first reaction (2h in hours, or prev close -> open + 60 min) and the move to the close
        table(tw, lambda k: np.array([W[k]["z_fast"][f] for f in B4.FACTORS]) * sd, "timed: first reaction")
        table(tw, lambda k: np.array([W[k]["z_to_close"][f] for f in B4.FACTORS]) * sd, "timed: report to the close")
        late = [k for k in tw if dates[ids[k]] >= pd.Timestamp("2025-06-01")]
        if len(late) >= 10:
            table(late, lambda k: np.array([W[k]["z_fast"][f] for f in B4.FACTORS]) * sd, "timed: first reaction, Jun 2025+")
    (ROOT / "docs/pmrm/results/bigtest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- timing at scale (#177)

STOP = set("the a an of and or to in on for with by from at as is are was were be after over into its their his her this that "
           "said says say new first two three four five more than about against amid".split())


def _tokens(txt: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", (txt or "").lower()) if w not in STOP}


def _news_day(day: pd.Timestamp) -> list[dict]:
    """Massive news from 18:00 ET the evening before through the end of the day (cached)."""
    from worsey.data import massive
    a = (day - pd.Timedelta(hours=6)).tz_localize("America/New_York").tz_convert("UTC")
    b = (day + pd.Timedelta(hours=23, minutes=59)).tz_localize("America/New_York").tz_convert("UTC")
    try:
        return massive._get("/v2/reference/news", {"published_utc.gte": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                   "published_utc.lte": b.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                   "limit": 1000, "order": "asc", "sort": "published_utc"}).get("results") or []
    except Exception:  # noqa: BLE001
        return []


def step_times():
    """News-based first-report times for 2022+ big events. Plain keyword overlap matched junk (18% agreement with the
    LLM's times: "united states", "russia" match every day), so each word is weighted by its rarity in headlines
    (IDF over all fetched days), only headlines are matched, and an article needs >= 2 matched words, one of them
    rare (in < 0.3% of headlines), and a total weight above a threshold. The threshold is reported against the LLM's
    high/medium-confidence times rather than tuned on them. Code only."""
    X, uns, src = load()
    labels = json.loads((DIR / "labels.json").read_text(encoding="utf-8"))
    cand = []
    for e in uns:
        if e["id"] not in labels or e["date"] < "2022-01-01":
            continue
        s0 = src.get(e["id"], e)
        kw = _tokens(" ".join([s0.get("title") or "", s0.get("description") or "", s0.get("region") or ""]))
        day = pd.Timestamp(s0.get("date_first_report") or e["date"])
        cand.append((e["id"], kw, _news_day(day)))
    df, n = {}, 0
    seen = set()
    for _, _, rows in cand:
        for r in rows:
            if r.get("id") in seen:
                continue
            seen.add(r.get("id"))
            n += 1
            for w in _tokens(r.get("title", "")):
                df[w] = df.get(w, 0) + 1
    idf = {w: np.log(n / (1 + c)) for w, c in df.items()}
    rare = np.log(n / (1 + 0.003 * n))
    out = {}
    for k, kw, rows in cand:
        for r in rows:
            hit = kw & _tokens(r.get("title", ""))
            ws = sorted((idf.get(w, np.log(n)) for w in hit), reverse=True)
            if len(ws) >= 2 and ws[0] >= rare and sum(ws[:3]) >= 13:
                t = pd.Timestamp(r["published_utc"]).tz_convert("America/New_York")
                out[k] = {"time_et": t.strftime("%Y-%m-%d %H:%M"), "matched": sorted(hit)[:6], "title": r.get("title", "")[:120],
                          "weight": round(float(sum(ws[:3])), 1)}
                break
    (DIR / "news_times.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    agree = []
    for k, v in out.items():
        lab = labels.get(k, {})
        lt = lab.get("time_et")
        if lt and lab.get("time_confidence") in ("high", "medium"):
            try:
                h1 = int(str(lt).split(":")[0]) * 60 + int(str(lt).split(":")[1])
                h2 = int(v["time_et"][11:13]) * 60 + int(v["time_et"][14:16])
                agree.append(abs(h1 - h2) <= 90)
            except ValueError:
                pass
    print(f"{n} distinct headlines; news times for {len(out)} of {len(cand)} events; agree with the LLM's medium/high time "
          f"within 90 min: {np.mean(agree) if agree else float('nan'):.0%} of {len(agree)}")


PICK_PROMPT = """For each market event below you get a list of timestamped financial-news headlines (US Eastern time) from the
event's day. Pick the EARLIEST headline that reports the event itself as having happened (or as breaking news of it).
Not a preview ("expected to", "ahead of"), not commentary on an older event, not an unrelated story that shares words.
If none qualifies, use null.

Return only a JSON array: [{{"id": "...", "pick": <headline number or null>}}, ...]

Events:
{events}"""


def step_newspick(confirm):
    """LLM check of the news matches (keyword/IDF matches alone gave 25% agreement): for each 2022+ big event, the
    12 highest-weight headlines that share a rare word with it, in time order; the LLM picks the earliest that
    reports the event. The pick's timestamp is the first-report time. Capped run, dry run first."""
    from worsey.agent.llm import AnthropicLLM
    X, uns, src = load()
    labels = json.loads((DIR / "labels.json").read_text(encoding="utf-8"))
    cand, df, seen, n = [], {}, set(), 0
    for e in uns:
        if e["id"] not in labels or e["date"] < "2022-01-01":
            continue
        s0 = src.get(e["id"], e)
        kw = _tokens(" ".join([s0.get("title") or "", s0.get("description") or "", s0.get("region") or ""]))
        rows = _news_day(pd.Timestamp(s0.get("date_first_report") or e["date"]))
        cand.append((e["id"], s0, kw, rows))
        for r in rows:
            if r.get("id") not in seen:
                seen.add(r.get("id"))
                n += 1
                for w in _tokens(r.get("title", "")):
                    df[w] = df.get(w, 0) + 1
    idf = {w: np.log(n / (1 + c)) for w, c in df.items()}
    rare = np.log(n / (1 + 0.003 * n))
    items = []
    for k, s0, kw, rows in cand:
        scored, titles = [], set()
        for r in rows:
            ti = r.get("title", "")
            hit = kw & _tokens(ti)
            ws = sorted((idf.get(w, np.log(n)) for w in hit), reverse=True)
            if ws and ws[0] >= rare and ti not in titles:
                titles.add(ti)
                scored.append((sum(ws[:3]), r["published_utc"], ti))
        top = sorted(sorted(scored, reverse=True)[:12], key=lambda x: x[1])
        if top:
            heads = [{"n": i + 1, "time_et": pd.Timestamp(t).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"), "headline": ti[:160]}
                     for i, (_, t, ti) in enumerate(top)]
            items.append({"id": k, "event": (s0.get("description") or s0.get("title") or "")[:400], "headlines": heads})
    batches = [items[i:i + 8] for i in range(0, len(items), 8)]
    run = ICB.Run("planb_bigtest_newspick", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    run.plan(len(batches), in_tok=4500, out_tok=400)
    print(f"{len(items)} events with candidate headlines in {len(batches)} batches")
    if not confirm:
        return
    out = {}
    for b in batches:
        prompt = PICK_PROMPT.format(events=json.dumps(b, indent=0, ensure_ascii=False))
        key = ICB.key("planb_bigtest_newspick", {"ids": [x["id"] for x in b]}, prompt, MODEL)
        try:
            resp = run.call(key, model=MODEL, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as e:
            print("stopped:", e)
            break
        byid = {x["id"]: x for x in b}
        for it in _parse(_text(resp)):
            if isinstance(it, dict) and it.get("id") in byid and isinstance(it.get("pick"), int):
                hs = byid[it["id"]]["headlines"]
                if 1 <= it["pick"] <= len(hs):
                    h = hs[it["pick"] - 1]
                    out[it["id"]] = {"time_et": h["time_et"], "title": h["headline"][:120], "source": "news+llm pick"}
    (DIR / "news_times.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    agree = []
    for k, v in out.items():
        lab = labels.get(k, {})
        if lab.get("time_et") and lab.get("time_confidence") in ("high", "medium"):
            try:
                h1 = int(str(lab["time_et"]).split(":")[0]) * 60 + int(str(lab["time_et"]).split(":")[1])
                h2 = int(v["time_et"][11:13]) * 60 + int(v["time_et"][14:16])
                agree.append(abs(h1 - h2) <= 90)
            except ValueError:
                pass
    print(json.dumps(run.summary() | {"picked": len(out), "agree_llm_time_90min": float(np.mean(agree)) if agree else None,
                                       "n_compared": len(agree)}))


def _minute_closes(t: str, day: pd.Timestamp):
    from worsey.data import massive
    try:
        rows = massive._get(f"/v2/aggs/ticker/{t}/range/1/minute/{day.date()}/{day.date()}",
                            {"adjusted": "true", "sort": "asc", "limit": 50000}).get("results") or []
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    ts = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert("America/New_York")
    return pd.Series([r["c"] for r in rows], index=ts)


def step_windows():
    """Reaction windows for every timed big event (news time first, else the LLM's medium/high-confidence time):
       in hours:    report - 5 min  -> report + 2 h (capped at the close)
       off hours:   last close before the report -> first next-session open + 60 min (the gap reaction)
    Plus, for reversion: the same session's close and the close 5 sessions later. Each move is standardized by the
    typical size of that kind of window, measured on 60 random sessions per ticker (same clock windows)."""
    import planb_b1 as B1
    X, uns, src = load()
    labels = json.loads((DIR / "labels.json").read_text(encoding="utf-8"))
    news = json.loads((DIR / "news_times.json").read_text(encoding="utf-8")) if (DIR / "news_times.json").exists() else {}
    tick = sorted({t for v in INTRADAY.values() for t, _ in v})
    sessions = X.index
    R = B1.total_returns(tick, sessions, "1990-01-01")
    dsd = R.loc["2011":"2020"].std()
    rng = np.random.default_rng(1729)
    # typical size of an off-hours window (prev close -> next open + 60 min) relative to a daily move, per ticker
    gap_ratio = {}
    sample_days = rng.choice(sessions[(sessions >= "2012-01-01") & (sessions <= "2024-12-31")], 60, replace=False)
    for t in tick:
        g = []
        for d in sample_days:
            d = pd.Timestamp(d)
            i = sessions.get_loc(d)
            prev = sessions[i - 1]
            cp = _minute_closes(t, prev)
            cd = _minute_closes(t, d)
            if cp is None or cd is None:
                continue
            a = cp[cp.index <= cp.index[0].normalize() + pd.Timedelta(hours=16)]
            b = cd[cd.index <= cd.index[0].normalize() + pd.Timedelta(hours=10, minutes=30)]
            if len(a) and len(b):
                g.append(b.iloc[-1] / a.iloc[-1] - 1)
        gap_ratio[t] = float(np.std(g) / dsd[t]) if len(g) > 20 else 0.6
    out = {}
    for e in uns:
        s0 = src.get(e["id"], e)
        lab = labels.get(e["id"], {})
        t_llm = None
        if lab.get("time_et") and lab.get("time_confidence") in ("high", "medium"):
            try:
                hh, mm = map(int, str(lab["time_et"]).split(":")[:2])
                day = pd.Timestamp(s0.get("date_first_report") or e["date"])
                t_llm = pd.Timestamp(day.year, day.month, day.day, hh, mm).tz_localize("America/New_York")
            except ValueError:
                pass
        t_news = pd.Timestamp(news[e["id"]]["time_et"]).tz_localize("America/New_York") if e["id"] in news else None
        # Massive's feed often lags the first report by hours, so a news time is an upper bound: prefer the LLM's
        # confident time, drop it when a news report came out > 90 min before it, use news only when there is no LLM time
        if t_llm is not None and t_news is not None and t_news < t_llm - pd.Timedelta(minutes=90):
            continue
        if t_llm is not None:
            t0, how = t_llm, ("llm+news" if t_news is not None and abs(t_news - t_llm) <= pd.Timedelta(minutes=90) else "llm")
        elif t_news is not None:
            t0, how = t_news, "news"
        else:
            continue
        d0 = t0.normalize().tz_localize(None)
        open_, close = t0.normalize() + pd.Timedelta(hours=9, minutes=30), t0.normalize() + pd.Timedelta(hours=16)
        in_hours = d0 in sessions and open_ <= t0 <= close - pd.Timedelta(minutes=30)
        if in_hours:
            sess = d0
        else:     # next session's open reaction
            k = sessions.searchsorted(d0 + (pd.Timedelta(days=1) if t0 >= close else pd.Timedelta(0)))
            if k >= len(sessions):
                continue
            sess = sessions[k]
        i_s = sessions.get_loc(sess)
        if i_s < 1 or i_s + 5 >= len(sessions):
            continue
        z_fast, z_close, z_after, raw = {}, {}, {}, {}
        moves = {}
        for t in tick:
            cs = _minute_closes(t, sess)
            if cs is None:
                continue
            if in_hours:
                a = cs[cs.index <= t0 - pd.Timedelta(minutes=5)]
                b = cs[cs.index <= min(t0 + pd.Timedelta(minutes=120), close)]
                if not (len(a) and len(b)):
                    continue
                start = a.iloc[-1]
                fast = b.iloc[-1] / start - 1
                scale_fast = np.sqrt(min(120, (close - t0).total_seconds() / 60) / 390)
            else:
                prev = _minute_closes(t, sessions[i_s - 1])
                if prev is None:
                    continue
                a = prev[prev.index <= prev.index[0].normalize() + pd.Timedelta(hours=16)]
                b = cs[cs.index <= cs.index[0].normalize() + pd.Timedelta(hours=10, minutes=30)]
                if not (len(a) and len(b)):
                    continue
                start = a.iloc[-1]
                fast = b.iloc[-1] / start - 1
                scale_fast = gap_ratio.get(t, 0.6)
            eod = cs[cs.index <= cs.index[0].normalize() + pd.Timedelta(hours=16)]
            to_close = eod.iloc[-1] / start - 1
            px = R[t]
            after = float(np.prod(1 + px.iloc[i_s + 1:i_s + 6].fillna(0)) - 1)
            moves[t] = (fast, scale_fast, to_close, after)
        for f, legs in INTRADAY.items():
            if all(t in moves for t, _ in legs):
                sdd = np.sqrt(sum(dsd[t] ** 2 for t, _ in legs))
                z_fast[f] = float(sum(s * moves[t][0] for t, s in legs) / (sdd * np.mean([moves[t][1] for t, _ in legs])))
                z_close[f] = float(sum(s * moves[t][2] for t, s in legs) / sdd)
                z_after[f] = float(sum(s * moves[t][3] for t, s in legs) / (sdd * np.sqrt(5)))
                raw[f] = [float(sum(s * moves[t][j] for t, s in legs)) for j in (0, 2, 3)]   # fast, to close, next 5 sessions
        if len(z_fast) == len(INTRADAY):
            out[e["id"]] = {"t0": str(t0), "source": how, "in_hours": bool(in_hours), "z_fast": z_fast, "z_to_close": z_close, "z_next5": z_after,
                             "raw": raw}
    (DIR / "windows.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"reaction windows for {len(out)} events ({sum(v['in_hours'] for v in out.values())} in hours, "
          f"{sum(v['source'] == 'news' for v in out.values())} news-timed); gap-window size vs a day: "
          + ", ".join(f"{t} {v:.2f}" for t, v in gap_ratio.items()))


def step_reversion():
    """Does the first reaction stick? For each factor: variance vs normal (fast window, to the close), and how much of
    the fast move survives to the close and over the next 5 sessions (regression slope of later on fast)."""
    W = json.loads((DIR / "windows.json").read_text(encoding="utf-8"))
    out = {"events": len(W)}
    for grp, keep in (("all", lambda v: True), ("in_hours", lambda v: v["in_hours"]), ("off_hours", lambda v: not v["in_hours"])):
        V = [v for v in W.values() if keep(v)]
        out[grp] = {"events": len(V)}
        for f in INTRADAY:
            zf = np.array([v["z_fast"][f] for v in V])
            zc = np.array([v["z_to_close"][f] for v in V])
            rf, rc, ra = (np.array([v["raw"][f][j] for v in V]) for j in range(3))
            big = np.abs(zf) >= np.quantile(np.abs(zf), 0.75)       # the quarter of events with the largest first reaction
            out[grp][f] = {"fast_variance_vs_normal": float((zf ** 2).mean()), "close_variance_vs_normal": float((zc ** 2).mean()),
                           "kept_at_close": float((rf @ rc) / (rf @ rf)),        # slope of the move-to-close on the first move
                           "kept_after_5_more_sessions": float((rf @ (rc + ra)) / (rf @ rf)),
                           "kept_at_close_biggest_quarter": float((rf[big] @ rc[big]) / (rf[big] @ rf[big])),
                           "same_sign_at_close": float(np.mean(np.sign(rf) == np.sign(rc)))}
    print(json.dumps(out, indent=1, default=lambda x: round(x, 2)))
    r = json.loads((ROOT / "docs/pmrm/results/bigtest.json").read_text(encoding="utf-8")) if (ROOT / "docs/pmrm/results/bigtest.json").exists() else {}
    r["reversion"] = out
    (ROOT / "docs/pmrm/results/bigtest.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["label", "intraday", "blind", "score", "times", "newspick", "windows", "reversion"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    {"label": lambda: step_label(a.confirm), "intraday": step_intraday, "blind": lambda: step_blind(a.confirm), "score": step_score,
     "times": step_times, "newspick": lambda: step_newspick(a.confirm), "windows": step_windows, "reversion": step_reversion}[a.step]()
