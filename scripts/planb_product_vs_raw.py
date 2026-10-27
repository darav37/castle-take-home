"""The shipped chat vs an untrained LLM, on events after the LLM's training cutoff (DECISIONS #180).

  python scripts/planb_product_vs_raw.py run [--confirm]    # LLM: the product's event_analyst loop + an untrained LLM
  python scripts/planb_product_vs_raw.py score              # code: factor and book scores, paired bootstrap

Events: every major/extreme one-off event in 2026 (31). For each:
  product    the exact chat path: loop._analyze_event_planb (Plan B prompt, event_evidence, price_scenario) on the
             balanced book, with the engine in cutoff mode (model refitted on pre-2026 events; evidence only from earlier
             events), so nothing it sees knows the outcome
  untrained  the same model (Sonnet), asked directly: the event, the 8 factors and what the buckets mean, reason, answer
  model      the engine alone on the product's own event structure (no LLM judgment)
Scored: R² vs zero on the 8 factors (event window, as in #170/#176), and on 6 books' real returns over the window.
Every call is capped (SPEC §6.1 per-run caps + the $70 project cap), dry run first, cached under cache/pmrm/llm/.
"""

import argparse
import json
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
from worsey import config  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

DIR = ROOT / "cache/pmrm/product_vs_raw"
CUTOFF = "2026-01-01"
BOOKS = {"Balanced (demo book)": None, "Energy producers": {"XLE": 0.5, "XOP": 0.5}, "Gold miners": {"GDX": 1.0},
         "Latin America": {"EWZ": 0.5, "EWW": 0.25, "ECH": 0.25}, "Long Treasuries": {"TLT": 1.0},
         "Agriculture / weather": {"DBA": 0.5, "MOO": 0.5}}
RAW_PROMPT = """You are a markets analyst. Today is {date}. This event has just been reported:

"{title}"

How will it move each of these over the next one to two trading sessions?
- market: US stocks (S&P 500)
- rates_10y: the 10-year Treasury yield
- dollar: the US dollar
- oil: crude oil
- gold: gold
- credit: high-yield bonds vs Treasuries
- defensive_vs_producers: utilities vs energy producers
- asia_em: emerging-market stocks vs US stocks

For each, pick one bucket relative to that asset's typical move when news like this hits:
strong_down (below -1.5 sd), down (-1.5 to -0.5 sd), flat (-0.5 to +0.5 sd), up (+0.5 to +1.5 sd), strong_up (above +1.5 sd).
Most news moves most assets very little, so flat is often right. Think it through briefly, then end with only a JSON
object: {{"market": "...", "rates_10y": "...", "dollar": "...", "oil": "...", "gold": "...", "credit": "...",
"defensive_vs_producers": "...", "asia_em": "..."}}"""


class CappedLLM:
    """The chat's LLM interface, routed through a capped, cached experiment run (SPEC §6.1)."""

    def __init__(self, run):
        self.run = run

    def start_turn(self):
        pass

    def create(self, **kw):
        kw.pop("_source", None)
        k = ICB.key(self.run.job, {}, json.dumps(kw, sort_keys=True, default=str), kw.get("model", ""))
        return self.run.call(k, **kw)


def events():
    from worsey.planb import live
    live.set_cutoff(None)
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8")) if r["kind"] == "unscheduled"}
    return [dict(e, w1=lib[e["id"]]["w1"]) for e in live.pack()["events"]
            if e["date"] >= CUTOFF and e["severity"] in ("major", "extreme") and e["id"] in lib]


def step_run(confirm):
    from worsey.agent import tools
    from worsey.agent.llm import AnthropicLLM
    from worsey.agent.loop import _analyze_event_planb
    from worsey.agent.session import Session
    from worsey.data.provider import BuiltMarketData
    from worsey.planb import live
    evs = events()
    config.SCENARIO_ENGINE = "planb"
    prod = ICB.Run("planb_product_vs_raw_product", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    raw = ICB.Run("planb_product_vs_raw_untrained", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    prod.plan(len(evs) * 4, in_tok=9000, out_tok=900)
    raw.plan(len(evs), in_tok=500, out_tok=600)
    print(f"{len(evs)} events (2026, major/extreme)")
    if not confirm:
        return
    live.set_cutoff(CUTOFF)
    DIR.mkdir(parents=True, exist_ok=True)
    out = json.loads((DIR / "runs.json").read_text(encoding="utf-8")) if (DIR / "runs.json").exists() else {}
    data = BuiltMarketData()
    for e in evs:
        if e["id"] in out and out[e["id"]].get("product") and out[e["id"]].get("untrained"):
            continue
        rec = out.get(e["id"], {"date": e["date"], "title": e["title"]})
        try:
            if not rec.get("product"):
                s = Session(data)
                tools.load_portfolio(s, name="balanced")
                r = _analyze_event_planb(CappedLLM(prod), s, None, e["title"], f"What happens to my book if this happens: {e['title']}")
                priced = next((x["price_scenario"] for x in s.turn_results if "price_scenario" in x), None)
                rec["product"] = {"spec": s.__dict__.get("planb_spec"), "judged": (priced or {}).get("your_judgment"),
                                  "priced": priced is not None, "reply": r.get("expert_reply")}
            if not rec.get("untrained"):
                prompt = RAW_PROMPT.format(date=e["date"], title=e["title"])
                k = ICB.key(raw.job, {"id": e["id"]}, prompt, MODEL)
                resp = raw.call(k, model=MODEL, max_tokens=1200, messages=[{"role": "user", "content": prompt}])
                txt = _text(resp)
                js = txt[txt.rfind("{"):txt.rfind("}") + 1]
                rec["untrained"] = json.loads(js) if js else {}
        except ICB.RunCapExceeded as err:
            print("stopped:", err)
            out[e["id"]] = rec
            break
        out[e["id"]] = rec
        (DIR / "runs.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (DIR / "runs.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"product": prod.summary(), "untrained": raw.summary(), "events_done": len(out),
                      "priced": sum(1 for v in out.values() if (v.get("product") or {}).get("priced"))}))


def step_score():
    from worsey.data.provider import BuiltMarketData
    from worsey.planb import live
    runs = json.loads((DIR / "runs.json").read_text(encoding="utf-8"))
    evs = {e["id"]: e for e in events()}
    full_U = {e["id"]: np.array(e["U"]) for e in live.pack()["events"]}
    live.set_cutoff(CUTOFF)
    p = live.pack()
    F, LF = live.FACTORS, p["llm_factors"]
    jl = [F.index(f) for f in LF]
    bval = lambda f, b: p["bucket_value"][f].get(b, 0.0) if b in p["buckets"] else 0.0
    rows = []
    for k, v in runs.items():
        if k not in evs or not (v.get("product") or {}).get("spec") or not v.get("untrained"):
            continue
        spec = live._spec_ok(v["product"]["spec"])
        est = live.estimate(spec)
        prod = est.copy()          # the #180 product: the expert's buckets override the engine
        for f, b in (v["product"].get("judged") or {}).items():
            prod[F.index(f)] = bval(f, b)
        # the shipped product since #182: the same LLM-filled schema, numbers from the engine via price()'s own function
        shipped = live.scenario_draws(spec, np.zeros(len(F)), 0.0)["f"]
        unt = np.zeros(len(F))
        for f in LF:
            unt[F.index(f)] = bval(f, v["untrained"].get(f))
        rows.append({"id": k, "date": v["date"], "real": full_U[k], "product (#180, LLM buckets)": prod,
                     "product (shipped, #182)": shipped, "model": est, "untrained": unt,
                     "priced": v["product"]["priced"], "w1": evs[k]["w1"]})
    print(f"{len(rows)} events scored ({sum(r['priced'] for r in rows)} priced by the product loop)")
    meths = ["product (shipped, #182)", "product (#180, LLM buckets)", "untrained", "model"]
    rng = np.random.default_rng(1729)
    out = {"events": len(rows)}

    def r2(Y, P):
        return float(1 - ((Y - P) ** 2).sum() / (Y ** 2).sum())

    Y = np.array([r["real"][jl] for r in rows])
    P = {m: np.array([r[m][jl] for r in rows]) for m in meths}
    res = {m: r2(Y, P[m]) for m in meths}
    res["market direction"] = {m: float(np.mean(np.sign(P[m][P[m][:, 0] != 0, 0]) == np.sign(Y[P[m][:, 0] != 0, 0])))
                               if (P[m][:, 0] != 0).any() else None for m in meths}
    d = []
    for _ in range(4000):
        ix = rng.integers(0, len(rows), len(rows))
        d.append((((Y[ix] - P["untrained"][ix]) ** 2).sum() - ((Y[ix] - P["product (shipped, #182)"][ix]) ** 2).sum()) / (Y[ix] ** 2).sum())
    res["shipped - untrained (90% range)"] = [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))]
    res["shipped beats untrained (bootstrap share)"] = float(np.mean(np.array(d) > 0))
    res["shipped == engine alone"] = bool(np.allclose(P["product (shipped, #182)"], P["model"]))
    out["factors"] = res
    print("8 factors:", json.dumps(res, default=lambda x: round(x, 3)))

    # books: real returns over each event's window vs loadings x predicted factor moves
    data = BuiltMarketData()
    import yaml
    bal = yaml.safe_load((ROOT / "portfolios/balanced.yaml").read_text(encoding="utf-8"))
    bal_w = {}
    for ps in bal.get("positions", []):
        bal_w[ps["ticker"]] = float(ps.get("value_usd") or ps.get("value") or 0)
    tot = sum(bal_w.values()) or 1
    books = {b: (w if w is not None else {t: x / tot for t, x in bal_w.items()}) for b, w in BOOKS.items()}
    tick = sorted({t for w in books.values() for t in w})
    L = live.loadings(data, tick)
    closes = data.closes(tick, days=320)
    out["books"] = {}
    for b, w in books.items():
        held = [t for t in w if t in L["betas"].index and t in closes.columns]
        beta = sum(w[t] * L["betas"].loc[t].to_numpy() for t in held)
        y, preds = [], {m: [] for m in meths}
        for r in rows:
            a, z = pd.Timestamp(r["w1"]["start"]), pd.Timestamp(r["w1"]["end"])
            c = closes.loc[(closes.index >= a) & (closes.index <= z), held]
            if len(c) < 2:
                continue
            y.append(float(sum(w[t] * (c[t].iloc[-1] / c[t].iloc[0] - 1) for t in held)))
            for m in meths:
                preds[m].append(float(beta @ r[m]))
        y = np.array(y)
        rb = {"n": len(y)} | {m: r2(y, np.array(preds[m])) for m in meths}
        out["books"][b] = rb
        print(f"  {b:24s} n={len(y)} " + " | ".join(f"{m} {rb[m]:+.1%}" for m in meths))
    (ROOT / "docs/pmrm/results/product_vs_raw.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    step_run(a.confirm) if a.step == "run" else step_score()
