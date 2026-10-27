"""The naive baseline vs the factor model, scored on real books including unusual ones (DECISIONS #178). No LLM calls: reuses the blinded big-event predictions from planb_bigtest.py.

  python scripts/planb_naive_books.py

Events: the blinded major/extreme events (2008-2026) with raw and assisted predictions; same daily window as #176.
Books: ETF baskets, split BEFORE scoring into market-like and unusual.
Predictors of each book's move over the event window (all time-safe: loadings from the 504 sessions before the event,
naive tables from earlier events only):
  zero               no move
  naive              book's market beta x average market move of earlier events with the same severity and the same
                     risk-sentiment label (risk off / on / neutral); the "stress test" baseline
  model              book's loadings on 8 factors x the factor model's prediction
  model (scaled)     same with the severity-scaled model
  raw / assisted     book's loadings x the blinded LLMs' factor buckets (numbers by code)
  oracle market      book's market beta x the REAL market move (how far a market-only view can go)
  oracle factors     book's 8-factor loadings x the REAL factor moves (how far the factor layer can go)
Writes docs/pmrm/results/naive_books.json.
"""

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

import planb_b1 as B1  # noqa: E402
import planb_b4_threeway as B4  # noqa: E402
import planb_bigtest as BT  # noqa: E402

BOOKS = {   # name: ({ticker: weight}, group)
    "Broad market (SPY)": ({"SPY": 1.0}, "market-like"),
    "Balanced 60/40": ({"SPY": 0.6, "IEF": 0.4}, "market-like"),
    "Tech growth (QQQ)": ({"QQQ": 1.0}, "market-like"),
    "Banks (KBE)": ({"KBE": 1.0}, "market-like"),
    "Energy producers": ({"XLE": 0.5, "XOP": 0.5}, "unusual"),
    "Agriculture / weather": ({"DBA": 0.5, "MOO": 0.5}, "unusual"),
    "Gold miners (GDX)": ({"GDX": 1.0}, "unusual"),
    "Latin America": ({"EWZ": 0.5, "EWW": 0.25, "ECH": 0.25}, "unusual"),
    "Utilities + REITs": ({"XLU": 0.5, "IYR": 0.5}, "unusual"),
    "Long Treasuries (TLT)": ({"TLT": 1.0}, "unusual"),
    "Commodities (DBC)": ({"DBC": 1.0}, "unusual"),
}
LOOKBACK = 504


def main():
    X, uns, src, labels, U, zU, sd, dates, sev, Fd, rates, bval = BT.evidence()
    preds = json.loads((BT.DIR / "blind_preds.json").read_text(encoding="utf-8"))
    ids = {r["id"]: i for i, r in enumerate(uns)}
    have = sorted([k for k, v in preds.items() if "raw" in v and "assisted" in v and k in ids], key=lambda k: ids[k])
    toval = lambda p: np.array([bval[B4.BUCKETS.index(p.get(f)), j] if p.get(f) in B4.BUCKETS else 0.0 for j, f in enumerate(B4.FACTORS)])
    ss = X.index
    F8 = X[B4.FACTORS]
    tick = sorted({t for w, _ in BOOKS.values() for t in w})
    R = B1.total_returns(tick, ss, "2003-01-01")
    book_ret = pd.DataFrame({b: sum(wt * R[t] for t, wt in w.items()) for b, (w, _) in BOOKS.items()})

    def window(i):
        a = ss.searchsorted(pd.Timestamp(uns[i]["w1"]["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(uns[i]["w1"]["end"])) + 1
        return a, b

    risk = np.array([(r.get("direction") or {}).get("risk_sentiment") or "neutral" for r in uns])
    risk = np.where(np.isin(risk, ["risk_off", "risk_on"]), risk, "neutral")
    mkt = U[:, B4.FACTORS.index("market")]

    rows = []
    for k in have:
        i = ids[k]
        a, b = window(i)
        if a < LOOKBACK + 1 or b <= a:
            continue
        earlier = (dates < dates[i]).to_numpy() & np.isin(sev, ["major", "extreme"])
        same = earlier & (sev == sev[i]) & (risk == risk[i])
        naive_m = mkt[same].mean() if same.sum() >= 10 else (mkt[earlier & (sev == sev[i])].mean() if (earlier & (sev == sev[i])).sum() else 0.0)
        Ftr = F8.iloc[a - 1 - LOOKBACK:a - 1].to_numpy()
        p = preds[k]
        P = {"model": np.array(p["model"]), "model (scaled)": np.array(p["model_scaled"]),
             "raw LLM": toval(p["raw"]), "assisted LLM": toval(p["assisted"])}
        for bk, (_, grp) in BOOKS.items():
            ytr = book_ret[bk].iloc[a - 1 - LOOKBACK:a - 1].to_numpy()
            ok = np.isfinite(ytr) & np.isfinite(Ftr).all(1)
            y = book_ret[bk].iloc[a:b].to_numpy()
            if ok.sum() < 250 or not np.isfinite(y).all():
                continue
            A = np.column_stack([np.ones(ok.sum()), Ftr[ok]])
            beta = np.linalg.lstsq(A, ytr[ok], rcond=None)[0][1:]
            am = np.column_stack([np.ones(ok.sum()), Ftr[ok, 0]])
            beta_m = np.linalg.lstsq(am, ytr[ok], rcond=None)[0][1]
            fit = A @ np.linalg.lstsq(A, ytr[ok], rcond=None)[0]
            cover = 1 - ((ytr[ok] - fit) ** 2).sum() / ((ytr[ok] - ytr[ok].mean()) ** 2).sum()
            row = {"event": k, "book": bk, "group": grp, "date": str(dates[i].date()), "sev": sev[i], "y": float(y.sum()),
                   "coverage": float(cover), "zero": 0.0, "naive": float(beta_m * naive_m),
                   "oracle market": float(beta_m * mkt[i]), "oracle factors": float(beta @ U[i])}
            for m, v in P.items():
                row[m] = float(beta @ v)
            rows.append(row)
    D = pd.DataFrame(rows)
    methods = ["naive", "model", "model (scaled)", "raw LLM", "assisted LLM", "oracle market", "oracle factors"]
    r2 = lambda d, m: float(1 - ((d.y - d[m]) ** 2).sum() / (d.y ** 2).sum())
    out = {"events": int(D.event.nunique()), "books": {}}
    print(f"{D.event.nunique()} events x {D.book.nunique()} books = {len(D)} rows\n")
    hdr = f"{'book':24s} {'cover':>6s} " + " ".join(f"{m[:13]:>13s}" for m in methods)
    print(hdr)
    for bk in BOOKS:
        d = D[D.book == bk]
        res = {"group": BOOKS[bk][1], "n": len(d), "daily_factor_coverage": float(d.coverage.mean())} | {m: r2(d, m) for m in methods}
        out["books"][bk] = res
        print(f"{bk:24s} {res['daily_factor_coverage']:6.0%} " + " ".join(f"{res[m]:+13.1%}" for m in methods))

    # pooled by group, each book standardized by its own event-move spread; paired bootstrap over events
    rng = np.random.default_rng(1729)
    D["s"] = D.groupby("book").y.transform(lambda v: np.sqrt((v ** 2).mean()))
    for c in ["y"] + methods:
        D[c + "_z"] = D[c] / D.s
    sets = {"all 2008-2026": D, "2026 only (post-cutoff)": D[D.date >= "2026-01-01"], "extreme": D[D.sev == "extreme"]}
    out["pooled"] = {}
    for sname, S in sets.items():
        for grp in ("market-like", "unusual"):
            d = S[S.group == grp]
            if d.event.nunique() < 10:
                continue
            res = {"events": int(d.event.nunique())} | {m: float(1 - ((d.y_z - d[m + "_z"]) ** 2).sum() / (d.y_z ** 2).sum()) for m in methods}
            ev = d.event.unique()
            g = {e: d[d.event == e] for e in ev}
            diffs = {"model - naive": [], "assisted - naive": [], "oracle factors - oracle market": []}
            for _ in range(1000):
                bs = pd.concat([g[e] for e in rng.choice(ev, len(ev))])
                den = (bs.y_z ** 2).sum()
                e2 = lambda m: ((bs.y_z - bs[m + "_z"]) ** 2).sum()
                diffs["model - naive"].append((e2("naive") - e2("model")) / den)
                diffs["assisted - naive"].append((e2("naive") - e2("assisted LLM")) / den)
                diffs["oracle factors - oracle market"].append((e2("oracle market") - e2("oracle factors")) / den)
            res["90% ranges"] = {k: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))] for k, v in diffs.items()}
            out["pooled"][f"{sname} | {grp}"] = res
            print(f"\n{sname} | {grp} ({res['events']} events): " + ", ".join(f"{m} {res[m]:+.1%}" for m in methods))
            print("   " + "; ".join(f"{k} [{a:+.1%}, {b:+.1%}]" for k, (a, b) in res["90% ranges"].items()))
    (ROOT / "docs/pmrm/results/naive_books.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
