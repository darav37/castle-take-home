"""B4: model alone vs raw LLM vs LLM assisted by the model (DECISIONS #167).

  python scripts/planb_b4_threeway.py                 # DRY RUN: test sets, batches, estimated cost; no calls
  python scripts/planb_b4_threeway.py --confirm       # one capped run (SPEC §6.1); rerun to continue (cached)
  python scripts/planb_b4_threeway.py --score         # score whatever is cached (no calls)

Events: the unscheduled library. Responses: short-window moves of 8 named factors (B1 13-factor space).
Test sets:
  post-cutoff   events on or after --cutoff (default 2026-01-01): the LLM cannot remember their market reaction
  historical    a fixed random sample of 2016-2023 events (seed 1729): reported, marked as possibly flattered
Predictors (same events, same factors):
  model         ridge on direction + category, trained only on events before each test event (refit monthly)
  raw LLM       the event description only -> a bucket per factor
  assisted LLM  description + the model's estimate and band + the regime + the 5 most similar EARLIER events with
                their realized moves -> a bucket per factor
Buckets (strong down, down, flat, up, strong up) are cut at -1.5, -0.5, 0.5, 1.5 sd of event responses before 2016;
each bucket's number is the mean pre-2016 response in it, so LLM answers become numbers by code.
Score: R² vs zero over the 8 factors pooled, and direction hit rate on non-flat calls; chance by shuffling predictions
across events. The LLM never sees an outcome for the event it predicts; analogs shown are strictly earlier events.
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

from planb_b2 import MODEL, _parse, _text  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402
from worsey.planb import regime as RG  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

FACTORS = ["market", "rates_10y", "dollar", "oil", "gold", "credit", "defensive_vs_producers", "asia_em"]
FACTOR_TEXT = {"market": "US stock market (S&P 500)", "rates_10y": "10-year Treasury yield (up = yields rise)",
               "dollar": "US dollar", "oil": "crude oil", "gold": "gold", "credit": "high-yield credit vs Treasuries (up = credit outperforms)",
               "defensive_vs_producers": "defensive stocks (utilities) vs energy producers", "asia_em": "Asia/emerging-market stocks vs the S&P"}
BUCKETS = ["strong_down", "down", "flat", "up", "strong_up"]
CUTS = [-1.5, -0.5, 0.5, 1.5]
BATCH = 10
OUT = ROOT / "docs/pmrm/results/b4_threeway.json"

RAW = """You are estimating how markets react to news events. For each event below, as of its first public report, predict
the move over the next one to two trading sessions for each of these market factors:
{factors}
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up ("flat" = within normal daily noise; most
events move most factors only within noise). Return ONLY a JSON array: [{{"id": ..., "pred": {{"market": "...", ...}}}}]

Events:
{events}"""

RAWPLUS = """You are an experienced macro and cross-asset strategist estimating how markets react to news events. For each event
below, as of its first public report, predict the move over the next one to two trading sessions for each factor:
{factors}

Calibration (how event-window moves are distributed historically, per factor, across thousands of news events):
{base_rates}
Most news events, even prominent ones, move most factors only within normal daily noise; big moves are concentrated in
genuinely market-moving events. Each event also comes with the market regime at the time (stocks and bonds moving
together or apart, inflation level, Fed direction, volatility) and the market's recent context.

Think about each event first: how surprising was it, what channel would carry it to markets, how big relative to normal
news? Then answer. Return ONLY a JSON array: [{{"id": ..., "why": "one sentence", "pred": {{"market": "...", ...}}}}]
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up.

Events:
{events}"""


ASSISTEDPLUS = """You are an experienced macro and cross-asset strategist estimating how markets react to news events. For each event
below, as of its first public report, predict the move over the next one to two trading sessions for each factor:
{factors}

Calibration (how event-window moves are distributed historically, per factor, across thousands of news events):
{base_rates}
Most news events, even prominent ones, move most factors only within normal daily noise; big moves are concentrated in
genuinely market-moving events. Each event comes with the market regime at the time and the market's recent context.

Each event also comes with evidence from a quantitative model built on thousands of past events: the model's estimate
per factor (a bucket), how much past similar events disagreed (band: narrow | wide), and the five most similar EARLIER
events with how each factor actually moved after them. The model alone is weak on single events; the analogs are real
outcomes. Use them as evidence, and use your judgment where this event clearly differs from them.

Think about each event first: how surprising was it, what channel would carry it to markets, how big relative to its
analogs? Then answer. Return ONLY a JSON array: [{{"id": ..., "why": "one sentence", "pred": {{"market": "...", ...}}}}]
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up.

Events:
{events}"""


ASSISTEDSIZE = """You are estimating how markets react to news events. For each event below, as of its first public report, predict
the move over the next one to two trading sessions for each of these market factors:
{factors}
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up.

Each event comes with evidence from a quantitative model built on thousands of past events:
- its severity class and HOW BIG moves typically are for events of that class: the historical share of each bucket for
  events of the same severity (extreme events move markets far more than minor ones; use this to size your calls),
- the model's estimate per factor (a bucket, already scaled to the event's severity) and how much similar past events
  disagreed (band: narrow | wide),
- the market regime when the event happened,
- the five most similar EARLIER events and how each factor actually moved after them.
Use the evidence and your judgment; you may disagree with the model where the event clearly differs from its analogs.
Return ONLY a JSON array: [{{"id": ..., "pred": {{"market": "...", ...}}}}]

Events:
{events}"""


ASSISTED = """You are estimating how markets react to news events. For each event below, as of its first public report, predict
the move over the next one to two trading sessions for each of these market factors:
{factors}
Use exactly one bucket per factor: strong_down | down | flat | up | strong_up ("flat" = within normal daily noise).

For each event you also get evidence from a quantitative model built on thousands of past events:
- the model's estimate per factor (a bucket) and how much past similar events disagreed (band: narrow | wide),
- the market regime when the event happened,
- the five most similar EARLIER events and how each factor actually moved after them.
Use the evidence and your judgment; you may disagree with the model where the event clearly differs from its analogs.
Return ONLY a JSON array: [{{"id": ..., "pred": {{"market": "...", ...}}}}]

Events:
{events}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cutoff", default="2026-01-01")
    ap.add_argument("--historical", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(1729)
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    X = ((ext["anchors"] - ext["anchor_mean"]) / ext["anchor_sd"]).fillna(0)
    sessions = X.index
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    regs = json.loads((ROOT / "cache/pmrm/b2/regimes.json").read_text(encoding="utf-8"))
    uns = sorted([r for r in lib if r["kind"] == "unscheduled"], key=lambda r: r["date"])
    n = len(uns)
    fi = [list(X.columns).index(f) for f in FACTORS]

    def resp(r):
        s0 = sessions.searchsorted(pd.Timestamp(r["w1"]["start"])) + 1
        s1 = sessions.searchsorted(pd.Timestamp(r["w1"]["end"])) + 1
        return X.iloc[s0:s1].to_numpy().sum(0)[fi]

    U = np.array([resp(r) for r in uns])
    dates = pd.to_datetime(pd.Series([r["date"] for r in uns]))
    pre = (dates < "2016-01-01").to_numpy()
    sd = U[pre].std(0)
    zU = U / sd

    def bucket_of(z):
        return BUCKETS[int(np.searchsorted(CUTS, z))]

    bucket_val = np.zeros((len(BUCKETS), len(FACTORS)))
    for j in range(len(FACTORS)):
        bins = np.searchsorted(CUTS, zU[pre, j])
        for b in range(len(BUCKETS)):
            v = U[pre, j][bins == b]
            bucket_val[b, j] = v.mean() if len(v) else 0.0
    sev_all = np.array([r.get("severity") or "moderate" for r in uns])
    sev_rates, sev_scale = {}, {}
    for sv in ("minor", "moderate", "major", "extreme"):
        mk = pre & (sev_all == sv)
        if mk.sum() < 10:
            continue
        sev_rates[sv] = {f: {BUCKETS[b]: round(float(np.mean(np.searchsorted(CUTS, zU[mk, j]) == b)), 2) for b in range(len(BUCKETS))}
                         for j, f in enumerate(FACTORS)}
    # test sets
    post = np.where((dates >= a.cutoff).to_numpy())[0]
    hist_pool = np.where(((dates >= "2016-01-01") & (dates < "2024-01-01")).to_numpy())[0]
    hist = np.sort(rng.choice(hist_pool, min(a.historical, len(hist_pool)), replace=False))
    tests = {"post-cutoff": post, "historical (hindsight risk)": hist}
    # model alone: time-ordered ridge on direction + category, refit per month; analogs from earlier events only
    cats = sorted({r["category"] for r in uns})
    Fd = np.column_stack([np.ones(n), np.array([IM.direction_features(e) for e in uns]),
                          np.array([[r["category"] == c for c in cats] for r in uns], float)])
    model_pred, band, analogs = {}, {}, {}
    need = sorted(set(post) | set(hist))
    cache_B = {}
    for i in need:
        key = str(dates[i])[:7]
        if key not in cache_B:
            tr = (dates < pd.Timestamp(key + "-01")).to_numpy()
            cache_B[key] = (IM.ridge(Fd[tr], U[tr], 2.0), tr)
        B, tr = cache_B[key]
        model_pred[i] = Fd[i] @ B
        earlier = np.where(tr)[0]
        sims = np.array([IM.direction_similarity(uns[i], uns[j]) for j in earlier])
        top = earlier[np.argsort(-sims)[:5]]
        analogs[i] = top
        band[i] = "wide" if np.mean(np.abs(zU[top] - zU[top].mean(0))) > 0.8 else "narrow"
    print(f"events {n}; post-cutoff (>= {a.cutoff}): {len(post)}; historical sample: {len(hist)}", flush=True)

    scale_cache = {}

    def sev_scale_for(i, sv):
        key = (str(dates[i])[:7], sv)
        if key not in scale_cache:
            tr = (dates < pd.Timestamp(key[0] + "-01")).to_numpy() & (sev_all == sv)
            if tr.sum() < 20:
                scale_cache[key] = 1.0
            else:
                B = cache_B[key[0]][0]
                x = (Fd[tr] @ B).ravel()
                y = U[tr].ravel()
                scale_cache[key] = float(np.clip((x @ y) / max(x @ x, 1e-9), 0, 6))
        return scale_cache[key]

    def ev_text(i, assisted, context=False) -> dict:
        r = uns[i]
        d = {"id": r["id"], "date": r["date_first_report"] if r.get("date_first_report") else r["date"], "category": r["category"],
             "region": r.get("region"), "description": (r.get("description") or r.get("title") or "")[:500]}
        if context:
            d["regime"] = regs.get(r["id"], {}).get("labels")
            s0 = sessions.searchsorted(pd.Timestamp(r["w1"]["start"]))
            last5 = X.iloc[max(s0 - 4, 0):s0 + 1].to_numpy().sum(0)[fi]
            d["previous_week"] = {f: bucket_of(last5[j] / (sd[j] * np.sqrt(5) / 1.2)) for j, f in enumerate(FACTORS)}
        if assisted == "size":
            sv = r.get("severity") or "moderate"
            d["severity"] = sv
            d["typical_moves_for_this_severity"] = sev_rates.get(sv)
            scale = sev_scale_for(i, sv)
            d["model_estimate"] = {f: bucket_of(scale * model_pred[i][j] / sd[j]) for j, f in enumerate(FACTORS)}
            d["band"] = band[i]
            d["regime"] = regs.get(r["id"], {}).get("labels")
            d["similar_earlier_events"] = [{"date": uns[k]["date"], "title": (uns[k].get("title") or "")[:100], "severity": uns[k].get("severity"),
                                            "moved": {f: bucket_of(zU[k, j]) for j, f in enumerate(FACTORS)}} for k in analogs[i]]
        if assisted in (True, "both"):
            d["model_estimate"] = {f: bucket_of(model_pred[i][j] / sd[j]) for j, f in enumerate(FACTORS)}
            d["band"] = band[i]
            d["regime"] = regs.get(r["id"], {}).get("labels")
            d["similar_earlier_events"] = [{"date": uns[k]["date"], "title": (uns[k].get("title") or "")[:100],
                                            "moved": {f: bucket_of(zU[k, j]) for j, f in enumerate(FACTORS)}} for k in analogs[i]]
        return d

    from worsey.agent.llm import AnthropicLLM
    run = ICB.Run("planb_b4_threeway", MODEL, AnthropicLLM() if a.confirm else None, confirmed=a.confirm)
    nb = sum(int(np.ceil(len(v) / BATCH)) for v in tests.values())
    run.plan(nb, in_tok=3000, out_tok=1500)
    run.plan(nb, in_tok=11000, out_tok=1500)
    ftxt = "\n".join(f"- {f}: {FACTOR_TEXT[f]}" for f in FACTORS)
    rates = []
    for j, f in enumerate(FACTORS):
        bins = np.searchsorted(CUTS, zU[pre, j])
        rates.append(f"- {f}: " + ", ".join(f"{BUCKETS[b]} {np.mean(bins == b):.0%}" for b in range(len(BUCKETS))))
    base_rates = "\n".join(rates)
    preds = {"raw LLM": {}, "assisted LLM": {}, "raw LLM (well-prompted)": {}, "assisted LLM (well-prompted)": {}, "assisted LLM (size-aware)": {}}
    for tname, ids in tests.items():
        for b0 in range(0, len(ids), BATCH):
            chunk = ids[b0:b0 + BATCH]
            for mode, tmpl, assisted in (("raw LLM", RAW, False), ("assisted LLM", ASSISTED, True),
                                         ("raw LLM (well-prompted)", RAWPLUS, "context"),
                                         ("assisted LLM (well-prompted)", ASSISTEDPLUS, "both"),
                                         ("assisted LLM (size-aware)", ASSISTEDSIZE, "size")):
                if mode == "assisted LLM (size-aware)" and tname != "post-cutoff":
                    continue
                evs = [ev_text(i, assisted, context=(assisted in ("context", "both"))) for i in chunk]
                prompt = tmpl.format(factors=ftxt, events=json.dumps(evs, indent=0, default=str), base_rates=base_rates)
                k = ICB.key("planb_b4_threeway", {"mode": mode, "ids": [uns[i]["id"] for i in chunk]}, prompt, MODEL)
                if not a.confirm and run.cached(k) is None:
                    continue
                try:
                    resp_ = run.call(k, model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
                except ICB.RunCapExceeded as e:
                    print("stopped:", e, flush=True)
                    break
                byid = {it.get("id"): it.get("pred") or {} for it in _parse(_text(resp_)) if isinstance(it, dict)}
                for i in chunk:
                    p = byid.get(uns[i]["id"])
                    if p:
                        preds[mode][int(i)] = [bucket_val[BUCKETS.index(p.get(f)), j] if p.get(f) in BUCKETS else 0.0
                                               for j, f in enumerate(FACTORS)]
    print(json.dumps(run.summary()), flush=True)
    if not (a.confirm or a.score):
        return
    res = {"cutoff": a.cutoff, "factors": FACTORS, "run": run.summary(), "sets": {}}
    for tname, ids in tests.items():
        modes_here = [m for m in preds if any(int(i) in preds[m] for i in ids)]
        have = [i for i in ids if all(int(i) in preds[m] for m in modes_here)]
        if not have:
            continue
        Y = U[have]
        P = {"model": np.array([model_pred[i] for i in have]),
             "raw LLM": np.array([preds["raw LLM"][int(i)] for i in have]),
             "assisted LLM": np.array([preds["assisted LLM"][int(i)] for i in have]),
             "raw LLM (well-prompted)": np.array([preds["raw LLM (well-prompted)"][int(i)] for i in have]),
             "assisted LLM (well-prompted)": np.array([preds["assisted LLM (well-prompted)"][int(i)] for i in have])}
        if "assisted LLM (size-aware)" in modes_here:
            P["assisted LLM (size-aware)"] = np.array([preds["assisted LLM (size-aware)"][int(i)] for i in have])
        out = {"events": len(have)}
        for name, Pm in P.items():
            r2 = IM.r2_vs(Y, Pm, np.zeros_like(Y))
            nz = Pm != 0
            hit = float(np.mean(np.sign(Pm[nz]) == np.sign(Y[nz]))) if nz.any() else None
            nl = [IM.r2_vs(Y, Pm[rng.permutation(len(have))], np.zeros_like(Y)) for _ in range(200)]
            out[name] = {"r2_vs_zero": r2, "chance95": float(np.quantile(nl, 0.95)), "direction_hit": hit,
                         "share_non_flat": float(nz.mean()),
                         "r2_by_factor": {f: IM.r2_vs(Y[:, j], Pm[:, j], np.zeros(len(have))) for j, f in enumerate(FACTORS)}}
        # paired bootstrap over events: is assisted better than raw / than the model on the SAME events?
        def sse(Pm, ix):
            return ((Y[ix] - Pm[ix]) ** 2).sum()
        ss0 = lambda ix: (Y[ix] ** 2).sum()
        pairs = {"assisted - raw": ("assisted LLM", "raw LLM"), "assisted - model": ("assisted LLM", "model"), "raw - model": ("raw LLM", "model"),
                 "assisted - well-prompted raw": ("assisted LLM", "raw LLM (well-prompted)"),
                 "well-prompted raw - raw": ("raw LLM (well-prompted)", "raw LLM"), "well-prompted raw - model": ("raw LLM (well-prompted)", "model"),
                 "well-prompted assisted - well-prompted raw": ("assisted LLM (well-prompted)", "raw LLM (well-prompted)"),
                 "well-prompted assisted - model": ("assisted LLM (well-prompted)", "model"),
                 "well-prompted assisted - assisted": ("assisted LLM (well-prompted)", "assisted LLM"),
                 "size-aware assisted - assisted": ("assisted LLM (size-aware)", "assisted LLM"),
                 "size-aware assisted - well-prompted raw": ("assisted LLM (size-aware)", "raw LLM (well-prompted)")}
        pairs = {k: v for k, v in pairs.items() if v[0] in P and v[1] in P}
        out["paired_r2_difference"] = {}
        for nm, (a1, b1) in pairs.items():
            diffs = []
            for _ in range(2000):
                ix = rng.integers(0, len(have), len(have))
                diffs.append((sse(P[b1], ix) - sse(P[a1], ix)) / ss0(ix))
            obs = (sse(P[b1], np.arange(len(have))) - sse(P[a1], np.arange(len(have)))) / ss0(np.arange(len(have)))
            out["paired_r2_difference"][nm] = {"diff": float(obs), "ci90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))],
                                               "share_boot_above_0": float(np.mean(np.array(diffs) > 0))}
        out["per_event"] = {uns[i]["id"]: {"y": Y[k].tolist(), **{nm: P[nm][k].tolist() for nm in P}} for k, i in enumerate(have)}
        res["sets"][tname] = out
        print(tname, {k: (round(v["r2_vs_zero"], 4), round(v["chance95"], 4), v["direction_hit"] and round(v["direction_hit"], 2), round(v["share_non_flat"], 2))
                      for k, v in out.items() if k in P}, flush=True)
        print("   paired:", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci90"]], round(v["share_boot_above_0"], 2)) for k, v in out["paired_r2_difference"].items()}, flush=True)
        print("   by factor (assisted / raw / model):", {f: (round(out["assisted LLM"]["r2_by_factor"][f], 3), round(out["raw LLM"]["r2_by_factor"][f], 3), round(out["model"]["r2_by_factor"][f], 3)) for f in FACTORS}, flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
