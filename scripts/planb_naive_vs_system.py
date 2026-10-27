"""The core test: does the system beat just asking the LLM? (DECISIONS #185)

  python scripts/planb_naive_vs_system.py run [--confirm]   # LLM: the naive arm (one call per event, all books)
  python scripts/planb_naive_vs_system.py score             # code: the system arm, the real outcomes, the scores

PREDECLARED (written before any result):
  Events     all 92 one-off events of 2026 (after the LLM's training cutoff); the 31 major/extreme reported separately.
  Books      12 portfolios of $100k: the 11 books of #178 (SPY, 60/40, tech, banks, energy, agriculture, gold miners,
             Latin America, utilities + REITs, long Treasuries, commodities) and the demo book (portfolios/balanced.yaml).
  Naive arm  the chat's expert model (Sonnet), no tools, no Worsey data. Given the date, the event and each book's
             holdings in dollars, it answers the user's question: the likely dollar impact over the next one to two
             trading sessions, a range it is 90% sure of, and a hedge (as a delta-one position: ticker, long/short,
             dollar notional). Fixed JSON format so code can read the numbers.
  System arm the shipped engine (live.scenario_draws, cutoff mode: refitted on pre-2026 events), the event's stored
             schema labels, each book's loadings from the 504 sessions before the event; its middle and 5%/95%
             quantiles in dollars. Hedge: the delta-one position in one liquid ETF that minimizes the book's variance
             under the event distribution (min-variance hedge; candidates SPY QQQ IWM TLT IEF GLD USO UUP HYG XLE EEM).
  Reality    each book's compounded total return over the event window (the library's short window) x $100k.
  Scores     weighted interval score on the 5/50/95% quantiles (WIS; lower is better; both arms reduced to the same
             three quantiles), 90%-range coverage, median absolute error of the middle, R2 of the middle vs zero;
             a predict-zero + normal-noise baseline for context. Hedges: share of each book's realized variance removed,
             and the average of its 10 worst event outcomes, hedged vs unhedged.
  Pass bar   the system beats the naive arm on WIS with >= 90% paired bootstrap support (over events), on all books
             together; reported also for market-like, unusual and the demo book, and for big events.
Every LLM call is capped (SPEC §6.1 + the $70 project cap), dry run first, cached under cache/pmrm/llm/.
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

import planb_b1 as B1  # noqa: E402
from planb_b2 import MODEL, _text  # noqa: E402
from planb_naive_books import BOOKS  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

DIR = ROOT / "cache/pmrm/naive_vs_system"
CUTOFF = "2026-01-01"
BOOK_USD = 100_000
LOOKBACK = 504
HEDGE_CANDIDATES = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "USO", "UUP", "HYG", "XLE", "EEM"]
PROMPT = """You are a markets analyst helping retail investors. Today is {date}. This news has just been reported:

"{title}"

For each portfolio below, answer the investor's question: "what could this do to my portfolio, and how could I hedge it?"
Give, for the next one to two trading sessions:
- expected_usd: your best estimate of the portfolio's dollar gain (+) or loss (-)
- low_usd, high_usd: a range you are 90% confident contains the actual dollar change
- hedge: one hedge as a delta-one position (if you'd use options, give the equivalent delta exposure): the ticker,
  "long" or "short", and the dollar notional. Use {{"ticker": null}} if you would not hedge.
- why: one short sentence

Portfolios:
{books}

Answer with only a JSON array, one object per portfolio, in order:
[{{"book": "...", "expected_usd": 0, "low_usd": 0, "high_usd": 0, "hedge": {{"ticker": "...", "side": "short", "notional_usd": 0}}, "why": "..."}}]"""


def books() -> dict:
    import yaml
    bal = yaml.safe_load((ROOT / "portfolios/balanced.yaml").read_text(encoding="utf-8"))
    pos = {p["ticker"]: float(p["value_usd"]) for p in bal["positions"]}
    tot = sum(pos.values())
    out = {"Demo book (balanced)": {t: v / tot for t, v in pos.items()}}
    out.update({b: w for b, (w, _) in BOOKS.items()})
    return out


def groups() -> dict:
    g = {b: grp for b, (_, grp) in BOOKS.items()}
    g["Demo book (balanced)"] = "demo"
    return g


def events():
    from worsey.planb import live
    live.set_cutoff(None)
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8")) if r["kind"] == "unscheduled"}
    return [dict(e, w1=lib[e["id"]]["w1"]) for e in live.pack()["events"] if e["date"] >= CUTOFF and e["id"] in lib]


def book_text(B: dict) -> str:
    lines = []
    for name, w in B.items():
        hold = ", ".join(f"{t} ${w[t] * BOOK_USD:,.0f}" for t in w)
        lines.append(f"- {name} (${BOOK_USD:,}): {hold}")
    return "\n".join(lines)


def step_run(confirm):
    from worsey.agent.llm import AnthropicLLM
    evs, B = events(), books()
    run = ICB.Run("planb_naive_vs_system", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    run.plan(len(evs), in_tok=1400, out_tok=1300)
    print(f"{len(evs)} events x {len(B)} books")
    if not confirm:
        return
    DIR.mkdir(parents=True, exist_ok=True)
    out = json.loads((DIR / "naive.json").read_text(encoding="utf-8")) if (DIR / "naive.json").exists() else {}
    for e in evs:
        if e["id"] in out:
            continue
        prompt = PROMPT.format(date=e["date"], title=e["title"], books=book_text(B))
        k = ICB.key(run.job, {"id": e["id"]}, prompt, MODEL)
        try:
            resp = run.call(k, model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as err:
            print("stopped:", err)
            break
        txt = _text(resp)
        try:
            arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
        except (ValueError, json.JSONDecodeError):
            arr = []
        out[e["id"]] = {"date": e["date"], "title": e["title"], "answers": arr}
        (DIR / "naive.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(run.summary() | {"events": len(out), "parsed": sum(1 for v in out.values() if len(v["answers"]) == len(B))}))


def wis(lo, mid, hi, y, alpha=0.10):
    """Weighted interval score for one central interval + the median (Bracher et al. 2021); lower is better."""
    interval = (hi - lo) + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)
    return float((0.5 * abs(y - mid) + (alpha / 2) * interval) / 1.5)


def step_score():
    from worsey.planb import live
    naive = json.loads((DIR / "naive.json").read_text(encoding="utf-8"))
    evs = {e["id"]: e for e in events()}
    B, G = books(), groups()
    names = list(B)
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    tick = sorted(set(proxy_t) | {t for w in B.values() for t in w} | set(HEDGE_CANDIDATES))
    R = B1.total_returns(tick, ss, "2003-01-01")
    Fz = live.factor_returns(R)
    F = live.FACTORS
    instr = {f"book:{b}": w for b, w in B.items()} | {f"inst:{t}": {t: 1.0} for t in HEDGE_CANDIDATES}
    T = pd.DataFrame({k: sum(wt * R[t] for t, wt in w.items()) for k, w in instr.items()})
    live.set_cutoff(CUTOFF)
    rows = []
    for k, v in naive.items():
        e = evs.get(k)
        if e is None or len(v["answers"]) != len(B):
            continue
        a = ss.searchsorted(pd.Timestamp(e["w1"]["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(e["w1"]["end"])) + 1
        if b <= a or b > len(ss):
            continue
        spec = live._spec_ok(e)
        base = live.scenario_draws(spec, np.zeros(len(F)), 0.0)
        f, dc = base["f"], base["draw"] - np.median(base["draw"], axis=0)
        rng = np.random.default_rng(1729)
        Fl = Fz.iloc[a - 1 - LOOKBACK:a - 1]
        fit = {}
        for col in T.columns:
            dd = pd.concat([T[col].iloc[a - 1 - LOOKBACK:a - 1].rename("y"), Fl], axis=1).dropna()
            used = [x for x in F if x in dd.columns]
            A = np.column_stack([np.ones(len(dd)), dd[used].to_numpy()])
            c = np.linalg.lstsq(A, dd["y"].to_numpy(), rcond=None)[0]
            fit[col] = (pd.Series(c[1:], index=used).reindex(F).fillna(0.0).to_numpy(), float((dd["y"].to_numpy() - A @ c).std()),
                        float(dd["y"].std()))
        real = {col: float(np.prod(1 + T[col].iloc[a:b]) - 1) for col in T.columns}
        z = rng.normal(0, 1, (live.N_DRAWS, len(T.columns))) * np.sqrt(1.5)
        cols = list(T.columns)
        # the event distribution for every book and hedge candidate (shared factor draws, independent own noise)
        sims = {col: fit[col][0] @ f + dc @ fit[col][0] + z[:, i] * fit[col][1] for i, col in enumerate(cols)}
        for j, name in enumerate(names):
            ans = v["answers"][j]
            col = f"book:{name}"
            x = sims[col] * BOOK_USD
            x = x - np.median(x) + fit[col][0] @ f * BOOK_USD
            lo, mid, hi = (float(q) for q in np.quantile(x, [0.05, 0.5, 0.95]))
            y = real[col] * BOOK_USD
            nsd = fit[col][2] * np.sqrt(b - a) * BOOK_USD
            try:
                n_mid, n_lo, n_hi = float(ans["expected_usd"]), float(ans["low_usd"]), float(ans["high_usd"])
            except (TypeError, ValueError, KeyError):
                continue
            n_lo, n_hi = min(n_lo, n_hi), max(n_lo, n_hi)
            # system hedge: min-variance delta-one in one candidate under the event distribution
            best = None
            for h in HEDGE_CANDIDATES:
                hs = sims[f"inst:{h}"]
                cov = np.cov(sims[col], hs)
                ratio = -cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
                left = cov[0, 0] + 2 * ratio * cov[0, 1] + ratio ** 2 * cov[1, 1]
                if best is None or left < best[2]:
                    best = (h, ratio, left)
            sys_hedge = best[1] * BOOK_USD * real[f"inst:{best[0]}"]
            hg = ans.get("hedge") or {}
            n_hedge = 0.0
            if hg.get("ticker") and f"inst:{hg['ticker']}" in real:
                sign = -1.0 if str(hg.get("side", "")).lower().startswith("s") else 1.0
                n_hedge = sign * float(hg.get("notional_usd") or 0) * real[f"inst:{hg['ticker']}"]
            elif hg.get("ticker"):
                n_hedge = np.nan     # a ticker we don't have prices for in this window
            rows.append({"event": k, "date": e["date"], "sev": spec["severity"], "book": name, "group": G[name], "y": y,
                         "sys_mid": mid, "sys_lo": lo, "sys_hi": hi, "nai_mid": n_mid, "nai_lo": n_lo, "nai_hi": n_hi,
                         "wis_sys": wis(lo, mid, hi, y), "wis_nai": wis(n_lo, n_mid, n_hi, y),
                         "wis_zero": wis(-1.645 * nsd, 0.0, 1.645 * nsd, y),
                         "sys_hedge_pnl": sys_hedge, "sys_hedge": best[0], "nai_hedge_pnl": n_hedge, "nai_hedge": hg.get("ticker")})
    live.set_cutoff(None)
    D = pd.DataFrame(rows)
    D.to_pickle(DIR / "rows.pkl")
    rng = np.random.default_rng(1729)

    def summ(d):
        ev = d["event"].unique()
        g = {k: x for k, x in d.groupby("event")}
        diffs = []
        for _ in range(2000):
            bs = pd.concat([g[k] for k in rng.choice(ev, len(ev))])
            diffs.append(bs["wis_nai"].mean() - bs["wis_sys"].mean())
        r2 = lambda m: float(1 - ((d.y - d[m]) ** 2).sum() / (d.y ** 2).sum())
        return {"events": int(len(ev)), "rows": int(len(d)),
                "wis": {"system": float(d.wis_sys.mean()), "naive_llm": float(d.wis_nai.mean()), "zero_plus_noise": float(d.wis_zero.mean())},
                "system_better_by_usd": float(d.wis_nai.mean() - d.wis_sys.mean()),
                "system_better_90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))],
                "system_better_support": float(np.mean(np.array(diffs) > 0)),
                "coverage_90": {"system": float(((d.y >= d.sys_lo) & (d.y <= d.sys_hi)).mean()),
                                "naive_llm": float(((d.y >= d.nai_lo) & (d.y <= d.nai_hi)).mean())},
                "range_width_usd_median": {"system": float((d.sys_hi - d.sys_lo).median()), "naive_llm": float((d.nai_hi - d.nai_lo).median())},
                "median_abs_error_usd": {"system": float((d.y - d.sys_mid).abs().median()), "naive_llm": float((d.y - d.nai_mid).abs().median())},
                "r2_middle": {"system": r2("sys_mid"), "naive_llm": r2("nai_mid")},
                "direction_hit": {"system": float(np.mean(np.sign(d.sys_mid) == np.sign(d.y))),
                                  "naive_llm": float(np.mean(np.sign(d.nai_mid[d.nai_mid != 0]) == np.sign(d.y[d.nai_mid != 0])))}}

    def hedges(d):
        out = {}
        for name, x in d.groupby("book"):
            y = x.y.to_numpy()
            r = {}
            for arm, col in (("system", "sys_hedge_pnl"), ("naive_llm", "nai_hedge_pnl")):
                h = np.nan_to_num(x[col].to_numpy())
                worst = np.argsort(y)[:10]
                r[arm] = {"variance_removed": float(1 - np.var(y + h) / np.var(y)),
                          "worst10_unhedged": float(y[worst].mean()), "worst10_hedged": float((y + h)[worst].mean())}
            r["naive_most_common_hedge"] = x.nai_hedge.value_counts().head(3).to_dict()
            r["system_most_common_hedge"] = x.sys_hedge.value_counts().head(3).to_dict()
            out[name] = r
        return out

    res = {"all books": summ(D), "market-like": summ(D[D.group == "market-like"]), "unusual": summ(D[D.group == "unusual"]),
           "demo book": summ(D[D.group == "demo"]), "big events (all books)": summ(D[D.sev.isin(["major", "extreme"])]),
           "by_book": {b: summ(D[D.book == b]) for b in names}, "hedges": hedges(D)}
    (ROOT / "docs/pmrm/results/naive_vs_system.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for k in ("all books", "market-like", "unusual", "demo book", "big events (all books)"):
        x = res[k]
        print(f"{k:24s} ev={x['events']} WIS sys ${x['wis']['system']:,.0f} naive ${x['wis']['naive_llm']:,.0f} zero ${x['wis']['zero_plus_noise']:,.0f} | "
              f"sys better by ${x['system_better_by_usd']:,.0f} (90% {x['system_better_90'][0]:,.0f} to {x['system_better_90'][1]:,.0f}, "
              f"support {x['system_better_support']:.0%}) | cov90 sys {x['coverage_90']['system']:.0%} naive {x['coverage_90']['naive_llm']:.0%} | "
              f"width sys ${x['range_width_usd_median']['system']:,.0f} naive ${x['range_width_usd_median']['naive_llm']:,.0f} | "
              f"MAE sys ${x['median_abs_error_usd']['system']:,.0f} naive ${x['median_abs_error_usd']['naive_llm']:,.0f} | "
              f"R2 sys {x['r2_middle']['system']:+.1%} naive {x['r2_middle']['naive_llm']:+.1%} | dir sys {x['direction_hit']['system']:.0%} naive {x['direction_hit']['naive_llm']:.0%}")
    for b, x in res["by_book"].items():
        print(f"   {b:26s} WIS sys ${x['wis']['system']:,.0f} naive ${x['wis']['naive_llm']:,.0f} support {x['system_better_support']:.0%} "
              f"cov sys {x['coverage_90']['system']:.0%} naive {x['coverage_90']['naive_llm']:.0%}")
    for b, x in res["hedges"].items():
        print(f"   hedge {b:26s} var removed sys {x['system']['variance_removed']:+.0%} naive {x['naive_llm']['variance_removed']:+.0%} | "
              f"worst10 unhedged ${x['system']['worst10_unhedged']:,.0f} sys ${x['system']['worst10_hedged']:,.0f} naive ${x['naive_llm']['worst10_hedged']:,.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    step_run(a.confirm) if a.step == "run" else step_score()
