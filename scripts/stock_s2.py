"""Single-stock track, S2: do options price earnings moves fairly, and does a stock's own history say when a straddle is
cheap or dear? (docs/pmrm/SINGLE_STOCK_TRACK.md; DECISIONS #203). No LLM calls.

  python scripts/stock_s2.py build      # straddle prices around a sample of 2015-2026 earnings reports (Massive, cached)
  python scripts/stock_s2.py score

Sample: every 2026 report, 1,500 from 2023-25 and 1,000 from 2015-22 (seeded), from cache/pmrm/stock/earnings.pkl.
Straddle: the call and put nearest the money on the first listed expiry on or after the reaction day, bought at the close
before the reaction day and sold at its close, each leg's daily close +/- 3% of premium and 1 cent.
  implied move  = straddle price / spot (what options charge for the move)
  realized move = |the stock's total move on the reaction day|
  history       = the median |reaction-day move| of this stock's previous 8 reports (known before the report)
Rules, thresholds chosen on 2015-22 and scored on 2023-25 and 2026: buy the straddle when history > tau x implied, sell it
when history < tau' x implied; P&L per $ of premium, net of costs; paired bootstrap over reports vs doing nothing.
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

from worsey.data import massive  # noqa: E402

DIR = ROOT / "cache/pmrm/stock"
OUT = ROOT / "docs/pmrm/results"
COST_PCT, COST_ABS = 0.03, 0.01


def step_build():
    E = pd.read_pickle(DIR / "earnings.pkl").sort_values("date").reset_index(drop=True)
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    # this stock's previous 8 reaction-day moves (total), known before each report
    E["hist_med_abs"] = E.groupby("ticker").day1_total.transform(lambda s: s.abs().shift(1).rolling(8, min_periods=4).median())
    rng = np.random.default_rng(1729)
    pick = []
    for sp, n in (("test", None), ("validation", 1500), ("train", 1000)):
        idx = E.index[(E.split == sp) & (E.date >= "2015-01-01") & E.hist_med_abs.notna()]
        pick += list(idx) if n is None or len(idx) <= n else list(rng.choice(idx, n, replace=False))
    rows = []
    for k, i in enumerate(sorted(pick)):
        e = E.loc[i]
        d = pd.Timestamp(e.date)
        j = ss.searchsorted(d)
        a = j + 1 if (e.timing == "after close" and ss[j] == d) else j
        if a >= len(ss):
            continue
        t_buy, t_sell = ss[a - 1], ss[a]
        try:
            px = massive.daily_bars(e.ticker, str((t_buy - pd.Timedelta(days=10)).date()), str(t_sell.date()), trim_to_listing=False)["close"]
            if t_buy not in px.index or t_sell not in px.index:
                continue
            S0, S1 = float(px[t_buy]), float(px[t_sell])
            exp, legs = None, None
            for dd in range(0, 21):
                cand = t_sell + pd.Timedelta(days=dd)
                if cand.weekday() != 4 and dd < 20:
                    continue
                c = massive.option_contracts(e.ticker, str(cand.date()), cand < pd.Timestamp("2026-10-20"))
                if c:
                    exp = cand
                    calls = [x for x in c if x["kind"] == "call"]
                    puts = [x for x in c if x["kind"] == "put"]
                    if calls and puts:
                        k_ = min({x["strike"] for x in calls} & {x["strike"] for x in puts} or {0}, key=lambda s: abs(s - S0))
                        legs = (next(x for x in calls if x["strike"] == k_), next(x for x in puts if x["strike"] == k_))
                    break
            if legs is None:
                continue
            pr = []
            for leg in legs:
                cl = massive.option_closes(leg["ticker"], str((t_buy - pd.Timedelta(days=7)).date()), str(t_sell.date()))
                if t_buy not in cl.index or t_sell not in cl.index:
                    break
                pr.append((float(cl[t_buy]), float(cl[t_sell])))
            if len(pr) != 2:
                continue
        except Exception:  # noqa: BLE001
            continue
        buy = sum(p0 * (1 + COST_PCT) + COST_ABS for p0, _ in pr)
        sell = sum(max(p1 * (1 - COST_PCT) - COST_ABS, 0.0) for _, p1 in pr)
        mid0 = sum(p0 for p0, _ in pr)
        rows.append({"ticker": e.ticker, "date": e.date, "split": e.split, "S0": S0, "strike": legs[0]["strike"], "expiry": str(exp.date()),
                     "implied_move": mid0 / S0, "realized_move": abs(S1 / S0 - 1), "hist_med_abs": float(e.hist_med_abs),
                     "long_pnl_per_premium": (sell - buy) / buy, "short_pnl_per_premium": (mid0 * (1 - COST_PCT) - sum(p1 * (1 + COST_PCT) for _, p1 in pr)) / mid0,
                     "eps_surprise_pct": e.eps_surprise_pct, "day1_z": e.day1_z})
        if k % 250 == 0:
            print(f"  {k}/{len(pick)} reports, {len(rows)} priced", flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle(DIR / "straddles.pkl")
    print(f"straddles priced: {len(R)} of {len(pick)} sampled; by split {R.split.value_counts().to_dict()}")


def boot(x, n=2000, seed=1729):
    x = np.asarray(x)
    if len(x) == 0:
        return None, None
    bs = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.mean(bs > 0))


def step_score():
    R = pd.read_pickle(DIR / "straddles.pkl")
    R = R[(R.implied_move > 0) & (R.implied_move < 0.5)].copy()
    R["ratio_real_impl"] = R.realized_move / R.implied_move
    R["hist_vs_impl"] = R.hist_med_abs / R.implied_move
    res = {"priced": int(len(R)), "by_split": R.split.value_counts().to_dict()}
    for sp in ("train", "validation", "test"):
        d = R[R.split == sp]
        res[sp] = {"n": int(len(d)), "median_implied_move": float(d.implied_move.median()), "median_realized_move": float(d.realized_move.median()),
                   "median_realized_over_implied": float(d.ratio_real_impl.median()), "share_realized_bigger": float((d.ratio_real_impl > 1).mean()),
                   "buy_every_straddle_pnl_per_premium": boot(d.long_pnl_per_premium)[0],
                   "sell_every_straddle_pnl_per_premium": boot(d.short_pnl_per_premium)[0]}
    tr = R[R.split == "train"]
    taus = np.round(np.arange(0.6, 2.01, 0.05), 2)
    tau_buy = max(taus, key=lambda t: tr.long_pnl_per_premium[tr.hist_vs_impl > t].mean() if (tr.hist_vs_impl > t).sum() >= 50 else -9)
    tau_sell = max(taus, key=lambda t: tr.short_pnl_per_premium[tr.hist_vs_impl < t].mean() if (tr.hist_vs_impl < t).sum() >= 50 else -9)
    res["tau"] = {"buy_when_history_over_implied_above": float(tau_buy), "sell_when_below": float(tau_sell)}
    for sp in ("validation", "test"):
        d = R[R.split == sp]
        b, s_ = d[d.hist_vs_impl > tau_buy], d[d.hist_vs_impl < tau_sell]
        res[sp]["rule_buy"] = {"n": int(len(b)), "pnl_per_premium": boot(b.long_pnl_per_premium)[0], "support": boot(b.long_pnl_per_premium)[1]}
        res[sp]["rule_sell"] = {"n": int(len(s_)), "pnl_per_premium": boot(s_.short_pnl_per_premium)[0], "support": boot(s_.short_pnl_per_premium)[1]}
        q = pd.qcut(d.hist_vs_impl, 4, labels=False, duplicates="drop")
        res[sp]["by_history_vs_implied_quartile"] = d.groupby(q).agg(hist_vs_impl=("hist_vs_impl", "mean"), realized_over_implied=("ratio_real_impl", "median"),
                                                                     long_pnl=("long_pnl_per_premium", "mean")).round(3).to_dict(orient="index")
    (OUT / "stock_s2.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:3500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "score"])
    a = ap.parse_args()
    step_build() if a.step == "build" else step_score()
