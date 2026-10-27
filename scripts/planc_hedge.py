"""The fair hedge test: when is it actually worth paying premium to hedge an event? (DECISIONS #195)

  python scripts/planc_hedge.py build              # real option prices for every event x book, 2023-2026 (Massive, cached)
  python scripts/planc_hedge.py naive [--confirm]   # the naive LLM's hedge decisions, 2026 (capped; dry run without --confirm)
  python scripts/planc_hedge.py score

Design, fixed before scoring (the #185 hedge half was invalid; this fixes each flaw):
  Same instrument for every arm. For each event and book: the liquid ETF in MENU most correlated with the book over the
  252 sessions before, a put about 3% out of the money on the first monthly expiry at least 14 days after the window,
  sized to the book's beta to that ETF. Every arm decides only WHETHER to hedge, so the test scores the decision.
  Real prices. Bought at the close before the event window, sold at the window's last close, at the option's daily close
  +/- 3% of premium and 1 cent (a spread proxy). The timing is generous to hedging (the position is on before the news);
  if hedging doesn't pay even so, it doesn't pay.
  Arms:
    never          never hedge
    always         hedge every event (its average net P&L is the price of protection)
    cheap options  hedge when the option's implied vol is low vs the ETF's realized vol (rho0 = realized / implied > tau);
                   no event information, so it isolates "options are cheap"
    system         hedge when our event forecast is wider than what the option prices:
                   rho = R * realized / implied > tau, where R = the book's forecast sd / its normal sd, from the C1c gate's
                   factor quantiles (the best C1c rung) weighted by each factor's share of the book's normal variance
    naive LLM      the chat's Claude model, no tools, told the event, the book and the exact put with its premium (2026 only)
  tau for "cheap options" and "system" is chosen on 2023-25 and scored on 2026.
  Scores per book-event, in dollars on a $100k book:
    net hedge P&L (what the hedge returned minus what it cost)
    shortfall = loss beyond the book's normal bad case (1.645 x its normal sd over the window), after the hedge
    utility = P&L - lambda x shortfall, lambda = 1 (also reported for 0 and 3); paired bootstrap over events vs "never"
  and decision quality: of the book-events where the hedge paid for itself, how many each arm hedged, and vice versa.
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
from planb_naive_vs_system import books  # noqa: E402
from worsey.data import massive  # noqa: E402
from worsey.options.pricer import implied_vol  # noqa: E402
from worsey.planb import live  # noqa: E402

DIR = ROOT / "cache/pmrm/planc"
OUT = ROOT / "docs/pmrm/results"
MENU = ["SPY", "QQQ", "IWM", "XLF", "XLE", "GLD", "GDX", "EWZ", "XLU", "TLT", "EEM", "USO", "HYG"]
BOOK_USD = 100_000
OTM = 0.97
COST_PCT, COST_ABS = 0.03, 0.01
SPLITS = ("validation", "test")


def third_friday(y, m):
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def pick_put(und, S, after: pd.Timestamp):
    """First monthly expiry >= `after` with listed contracts; the put nearest OTM x S."""
    y, m = after.year, after.month
    for _ in range(4):
        e = third_friday(y, m)
        if e >= after:
            for exp in (e, e - pd.Timedelta(days=1)):        # holiday expiries move to Thursday
                c = massive.option_contracts(und, str(exp.date()), exp < pd.Timestamp("2026-10-20"))
                puts = [x for x in c if x["kind"] == "put"]
                if puts:
                    return exp, min(puts, key=lambda x: abs(x["strike"] - OTM * S))
        m = 1 if m == 12 else m + 1
        y = y + 1 if m == 1 else y
    return None, None


def step_build():
    D = pickle.load(open(DIR / "c0_events.pkl", "rb"))
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))}
    B = books()
    R = B1.total_returns(sorted(set(MENU) | {t for w in B.values() for t in w}), ss, "2003-01-01")
    px = massive.closes(MENU, "2014-01-01", "2026-10-20").reindex(ss)
    Bk = pd.DataFrame({b: sum(w * R[t] for t, w in wt.items()) for b, wt in B.items()})
    rows = []
    for r in [x for x in D["events"] if x["split"] in SPLITS]:
        w1 = lib[r["id"]]["w1"]
        a = ss.searchsorted(pd.Timestamp(w1["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(w1["end"])) + 1
        t_buy, t_sell = ss[a - 1], ss[b - 1]
        for name in B:
            if name not in r["book"]:
                continue
            look = slice(a - 253, a - 1)
            y = Bk[name].iloc[look]
            best = None
            for h in MENU:
                x = R[h].iloc[look]
                ok = y.notna() & x.notna()
                if ok.sum() < 200:
                    continue
                c = float(np.corrcoef(y[ok], x[ok])[0, 1])
                if best is None or c > best[1]:
                    best = (h, c, float(np.cov(y[ok], x[ok])[0, 1] / x[ok].var()), float(x[ok].std()))
            if best is None or best[1] <= 0.2:
                continue
            h, corr, beta, rv_d = best
            S = float(px[h].iloc[a - 1])
            exp, put = pick_put(h, S, t_sell + pd.Timedelta(days=14))
            if put is None:
                continue
            try:
                cl = massive.option_closes(put["ticker"], str((t_buy - pd.Timedelta(days=7)).date()), str(t_sell.date()))
            except Exception:  # noqa: BLE001
                continue
            if t_buy not in cl.index or t_sell not in cl.index:
                continue
            p0, p1 = float(cl[t_buy]), float(cl[t_sell])
            T = (exp - t_buy).days / 365
            iv = implied_vol(p0, S, put["strike"], T, 0.04, 0.0, "put")
            n_contracts = beta * BOOK_USD / (100 * S)
            buy, sell = p0 * (1 + COST_PCT) + COST_ABS, max(p1 * (1 - COST_PCT) - COST_ABS, 0.0)
            rows.append({"event": r["id"], "split": r["split"], "date": r["date"], "severity": r["severity"], "title": r["spec"]["title"],
                         "book": name, "hedge_etf": h, "corr": corr, "beta": beta, "S": S, "strike": put["strike"], "expiry": str(exp.date()),
                         "contract": put["ticker"], "premium_usd": n_contracts * 100 * buy, "hedge_pnl": n_contracts * 100 * (sell - buy),
                         "iv": iv, "rv_annual": rv_d * np.sqrt(252), "book_pnl": r["book"][name]["y"] * BOOK_USD,
                         "book_noise_usd": r["book"][name]["noise"] * BOOK_USD, "n": r["n"]})
        print(f"  {r['date']} {len(rows)} rows", flush=True) if len(rows) % 500 < 12 else None
    H = pd.DataFrame(rows)
    H.to_pickle(DIR / "hedge_rows.pkl")
    print(f"hedge rows: {len(H)} book-events, {H.event.nunique()} events; by split {H.groupby('split').event.nunique().to_dict()}; "
          f"median premium ${H.premium_usd.median():,.0f}; hedge ETFs {H.hedge_etf.value_counts().to_dict()}")


NAIVE_PROMPT = """You manage risk for retail investors. A news event has just been reported (date: {date}):
"{title}"

For each portfolio below ($100,000 each) you may buy one protective put, already chosen, held for the next one to two
trading sessions and then sold. The premium shown is what it costs today. Decide for each portfolio whether buying it is
worth the premium, given this news. Most of the time protection costs more than it saves; buy only if you expect this
event to cause an unusually large loss for that portfolio.

{lines}

Return ONLY a JSON array: [{{"book": "<name>", "hedge": true|false}}, ...] in the same order."""


def step_naive(confirm):
    from planb_b2 import MODEL, _text
    from worsey.pmrm.ic import budget as ICB
    H = pd.read_pickle(DIR / "hedge_rows.pkl")
    H = H[H.split == "test"]
    evs = list(H.groupby("event"))
    run = ICB.Run("planc_hedge_naive", MODEL, None, confirmed=confirm)
    if confirm:
        from worsey.agent.llm import AnthropicLLM
        run = ICB.Run("planc_hedge_naive", MODEL, AnthropicLLM(), confirmed=True)
    run.plan(len(evs), in_tok=900, out_tok=400)
    print(f"{len(evs)} events (2026)")
    if not confirm:
        return
    f = DIR / "hedge_naive.json"
    out = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    for eid, g in evs:
        if eid in out:
            continue
        lines = "\n".join(f"- {r.book}: put on {r.hedge_etf} (about {100 * (1 - r.strike / r.S):.0f}% below today's price, expiring "
                          f"{r.expiry}), premium ${r.premium_usd:,.0f}" for r in g.itertuples())
        prompt = NAIVE_PROMPT.format(date=g.date.iloc[0], title=g.title.iloc[0], lines=lines)
        k = ICB.key(run.job, {"id": eid}, prompt, MODEL)
        try:
            resp = run.call(k, model=MODEL, max_tokens=1200, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as err:
            print("stopped:", err)
            break
        txt = _text(resp)
        try:
            arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
        except (ValueError, json.JSONDecodeError):
            arr = []
        out[eid] = {str(x.get("book")): bool(x.get("hedge")) for x in arr if isinstance(x, dict)}
        f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(run.summary() | {"events": len(out)}))


def forecast_ratio(H):
    """R = the book's forecast sd / its normal sd over the window, from the best C1c rung's factor quantiles: each
    factor's forecast sd (noise units) weighted by its share of the book's normal variance; the rest at 1."""
    S = pd.read_pickle(DIR / "c1c_rungs.pkl")
    res = json.loads((OUT / "planc_c1c.json").read_text(encoding="utf-8"))
    best = min(("A gate", "B +C1c", "C +sign"), key=lambda k: res["crps"]["validation | all factors"]["rungs"][k]["crps_std"])
    S = S[S.rung == best]
    sd = {(r.event, r.target): (r.q[-1] - r.q[0]) / 3.29 for r in S.itertuples()}
    D = pickle.load(open(DIR / "c0_events.pkl", "rb"))
    ev = {r["id"]: r for r in D["events"]}
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))}
    B = books()
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    R = B1.total_returns(sorted(set(proxy_t) | {t for w in B.values() for t in w}), ss, "2003-01-01")
    Fz = live.factor_returns(R).reindex(columns=live.FACTORS)
    out = []
    for r in H.itertuples():
        e = ev[r.event]
        a = ss.searchsorted(pd.Timestamp(lib[r.event]["w1"]["start"])) + 1
        cov = np.nan_to_num(Fz.iloc[a - 505:a - 1].cov().to_numpy())
        beta = e["book"][r.book]["beta"]
        tot = (e["book"][r.book]["noise"] / np.sqrt(e["n"])) ** 2
        part = beta ** 2 * np.diag(cov)
        fac = float(beta @ cov @ beta)
        share = part / part.sum() * min(fac / tot, 1.0) if part.sum() > 0 else np.zeros(len(part))
        s = np.array([sd.get((r.event, f), 1.0) for f in live.FACTORS])
        out.append(float(np.sqrt(share @ s ** 2 + max(1 - share.sum(), 0.0))))
    return np.array(out), best


def step_score():
    H = pd.read_pickle(DIR / "hedge_rows.pkl")
    H = H[np.isfinite(H.iv) & (H.iv > 0)].copy()
    H["R"], best_rung = forecast_ratio(H)
    H["rho0"] = H.rv_annual / H.iv
    H["rho"] = H.R * H.rho0
    H["bad_case"] = -1.645 * H.book_noise_usd
    nf = DIR / "hedge_naive.json"
    naive = json.loads(nf.read_text(encoding="utf-8")) if nf.exists() else {}
    H["naive"] = [naive.get(r.event, {}).get(r.book) if r.event in naive else None for r in H.itertuples()]

    def outcome(d, hedge):
        pnl = d.book_pnl + np.where(hedge, d.hedge_pnl, 0.0)
        short = np.maximum(d.bad_case - pnl, 0.0)
        return pnl, short

    def utility(d, hedge, lam):
        pnl, short = outcome(d, hedge)
        return pnl - lam * short

    def boot(d, hedge, lam, n=2000, seed=1729):
        gain = pd.Series(utility(d, hedge, lam) - utility(d, np.zeros(len(d), bool), lam), index=d.index).groupby(d.event).agg(["sum", "count"])
        s, c = gain["sum"].to_numpy(), gain["count"].to_numpy()
        idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
        bs = s[idx].sum(1) / c[idx].sum(1)
        return float(s.sum() / c.sum()), float(np.mean(bs > 0))

    res = {"best_c1c_rung": best_rung, "rows": int(len(H)), "events": H.groupby("split").event.nunique().to_dict()}
    val, test = H[H.split == "validation"], H[H.split == "test"]
    taus = np.round(np.arange(0.5, 3.01, 0.05), 2)
    chosen = {}
    for arm, col in (("cheap options", "rho0"), ("system", "rho")):
        u = [(t, float(utility(val, val[col] > t, 1.0).mean())) for t in taus]
        chosen[arm] = max(u, key=lambda x: x[1])[0]
    res["tau"] = chosen
    paid = lambda d: d.hedge_pnl > 0                        # the hedge returned more than it cost
    for sp, d in (("validation", val), ("test", test)):
        arms = {"never": np.zeros(len(d), bool), "always": np.ones(len(d), bool),
                "cheap options": (d.rho0 > chosen["cheap options"]).to_numpy(), "system": (d.rho > chosen["system"]).to_numpy()}
        if sp == "test" and d.naive.notna().any():
            arms["naive LLM"] = d.naive.fillna(False).astype(bool).to_numpy()
        res[sp] = {}
        for arm, hedge in arms.items():
            pnl, short = outcome(d, hedge)
            row = {"share_hedged": float(hedge.mean()), "premium_per_book_event": float((d.premium_usd * hedge).mean()),
                   "net_hedge_pnl_per_book_event": float((d.hedge_pnl * hedge).mean()),
                   "mean_pnl": float(pnl.mean()), "mean_shortfall": float(short.mean()),
                   "worst_5pct_mean": float(np.sort(pnl)[:max(1, len(pnl) // 20)].mean()),
                   "hedged_when_it_paid": float(hedge[paid(d).to_numpy()].mean()) if paid(d).any() else None,
                   "paid_when_hedged": float(paid(d).to_numpy()[hedge].mean()) if hedge.any() else None}
            for lam in (0.0, 1.0, 3.0):
                g, sup = boot(d, hedge, lam)
                row[f"utility_gain_vs_never_lam{lam:g}"] = g
                row[f"support_lam{lam:g}"] = sup
            res[sp][arm] = row
            print(f"{sp:10s} {arm:13s} hedged {row['share_hedged']:5.1%} | premium ${row['premium_per_book_event']:6.0f} | net hedge P&L "
                  f"${row['net_hedge_pnl_per_book_event']:+7.1f} | shortfall ${row['mean_shortfall']:6.1f} | worst 5% ${row['worst_5pct_mean']:8.0f} | "
                  f"U(1) vs never ${row['utility_gain_vs_never_lam1']:+6.1f} ({row['support_lam1']:.0%}) | U(3) ${row['utility_gain_vs_never_lam3']:+6.1f} "
                  f"({row['support_lam3']:.0%}) | paid when hedged {row['paid_when_hedged'] if row['paid_when_hedged'] is None else round(row['paid_when_hedged'], 2)}",
                  flush=True)
        res[sp]["share_of_hedges_that_paid_if_always"] = float(paid(d).mean())
        big = d.severity.isin(["major", "extreme"])
        res[sp]["big_events_always_net_hedge_pnl"] = float(d.hedge_pnl[big].mean()) if big.any() else None
    # the evidence the product cites (live.hedge_verdict reads this block): 2023-26 pooled, always hedging
    allr = pd.concat([val, test])
    beyond = allr.book_pnl < allr.bad_case
    ev = {"book_events": int(len(allr)), "events": int(allr.event.nunique()), "years": "2023-2026",
          "hold": "1-2 sessions after the news", "net_per_100k": float(allr.hedge_pnl.mean()),
          "premium_per_100k": float(allr.premium_usd.mean()), "share_paid": float((allr.hedge_pnl > 0).mean()),
          "beyond_bad_case_share": float(beyond.mean()), "hedge_return_when_beyond": float(allr.hedge_pnl[beyond].mean()),
          "loss_when_beyond": float(allr.book_pnl[beyond].mean()), "by_severity": {}}
    for sv, d in allr.groupby("severity"):
        ev["by_severity"][sv] = {"book_events": int(len(d)), "events": int(d.event.nunique()), "net_per_100k": float(d.hedge_pnl.mean()),
                                 "share_paid": float((d.hedge_pnl > 0).mean()), "beyond_bad_case_share": float((d.book_pnl < d.bad_case).mean())}
    lam_be = []
    for d in (val, test):
        s0, s1 = np.maximum(d.bad_case - d.book_pnl, 0).mean(), np.maximum(d.bad_case - d.book_pnl - d.hedge_pnl, 0).mean()
        lam_be.append(float(-d.hedge_pnl.mean() / (s0 - s1)))
    ev["break_even_weight"] = lam_be
    by_etf = allr.groupby("hedge_etf").agg(net=("hedge_pnl", "mean"), premium=("premium_usd", "mean"), n=("hedge_pnl", "size"))
    ev["by_instrument"] = {k: {"net_per_100k": float(v.net), "premium_per_100k": float(v.premium), "book_events": int(v.n)} for k, v in by_etf.iterrows()}
    res["product_evidence"] = ev
    by = test.groupby("hedge_etf").agg(rows=("hedge_pnl", "size"), net=("hedge_pnl", "mean"), premium=("premium_usd", "mean"))
    res["test_by_hedge_etf_if_always"] = by.round(1).to_dict(orient="index")
    (OUT / "planc_hedge.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"tau: {chosen}; C1c rung used: {best_rung}; hedges that paid if always (2026): {res['test']['share_of_hedges_that_paid_if_always']:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "naive", "score"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    {"build": step_build, "naive": lambda: step_naive(a.confirm), "score": step_score}[a.step]()
