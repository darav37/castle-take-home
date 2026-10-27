"""Event effects per factor and on targeted sectors/countries (Oct 21; the trainer: "try a bunch of different things"). No LLM.

Uses the 13 named factors (planb_b1_extended.py) and the grown library.
(1) Per factor, per category: LOO R² (vs zero) of a ridge model on direction labels + category, against chance
    (labels shuffled across events, 100 times).
(2) Targeted effects: every sector and country ETF's beyond-factor ("abnormal") move in each event window.
    (a) Size: is a targeted ETF's |abnormal move| bigger than the other sector/country ETFs' in the same window?
        (paired by event; chance by shuffling which ETFs count as targeted)
    (b) Direction: predict a targeted ETF's abnormal move from other events with the same category, risk direction and
        target (leave one out), vs zero; chance by shuffling the moves across targets.
Writes docs/pmrm/results/b3_targets.json.
"""

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
from planb_b2 import COUNTRY_ETF, SECTOR_ETF, country_etfs  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402


def main(perms: int = 100):
    rng = np.random.default_rng(1729)
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    X = (ext["anchors"] - ext["anchor_mean"]) / ext["anchor_sd"]
    sessions = X.index
    fit = (sessions >= "2011-01-03") & (sessions <= "2020-12-31")
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    uns = [r for r in lib if r["kind"] == "unscheduled"]
    n = len(uns)
    pos = {d: i for i, d in enumerate(sessions)}

    def span(w):
        a = sessions.searchsorted(pd.Timestamp(w["start"])) + 1
        b = sessions.searchsorted(pd.Timestamp(w["end"])) + 1
        return a, b

    spans = [span(r["w1"]) for r in uns]
    # (1) per-factor responses on the 13 named factors
    Xv = X.fillna(0).to_numpy()
    Fx = np.array([Xv[a:b].sum(0) for a, b in spans])
    cats = sorted({r["category"] for r in uns})
    feat = np.column_stack([np.ones(n), np.array([IM.direction_features(e) for e in uns]),
                            np.array([[r["category"] == c for c in cats] for r in uns], float)])
    P = IM.ridge_loo_predictions(feat, Fx, 2.0)
    nulls = [IM.ridge_loo_predictions(feat[rng.permutation(n)], Fx, 2.0) for _ in range(perms)]
    sev = np.array([r.get("severity") for r in uns])
    subsets = {"all": np.ones(n, bool), "extreme or major": np.isin(sev, ["extreme", "major"])}
    for c in cats:
        subsets[c] = np.array([r["category"] == c for r in uns])
    per = {}
    for sname, mk in subsets.items():
        per[sname] = {}
        for j, f in enumerate(X.columns):
            r2 = IM.r2_vs(Fx[mk, j], P[mk, j], np.zeros(mk.sum()))
            nl = np.array([IM.r2_vs(Fx[mk, j], q[mk, j], np.zeros(mk.sum())) for q in nulls])
            per[sname][f] = {"r2": r2, "null95": float(np.quantile(nl, 0.95)), "p": float((1 + (nl >= r2).sum()) / (1 + perms))}
        hits = [f"{f} {v['r2']:+.3f}" for f, v in per[sname].items() if v["p"] <= 0.05 and v["r2"] > 0]
        print(f"(1) {sname:26} n={mk.sum():4d}  factors with real skill: {', '.join(hits) or '-'}", flush=True)

    # (2) targeted sector/country ETFs: beyond-factor moves
    etfs = sorted(set(SECTOR_ETF.values()) | set(COUNTRY_ETF.values()))
    R = B1.total_returns(etfs, sessions, "1990-01-01")
    AB = {}
    for t in etfs:
        s = R[t]
        d = pd.concat([(s - s[fit].mean()) / s[fit].std(), X], axis=1)[fit].dropna()
        if len(d) < 250:
            continue
        b = np.linalg.lstsq(np.column_stack([np.ones(len(d)), d.iloc[:, 1:].to_numpy()]), d.iloc[:, 0].to_numpy(), rcond=None)[0][1:]
        AB[t] = (s - s[fit].std() * (Xv @ b)).to_numpy()      # daily beyond-factor return
    tick = sorted(AB)
    A = np.array([[np.nansum(AB[t][a:b]) if np.isfinite(AB[t][a:b]).all() and b > a else np.nan for t in tick] for a, b in spans])
    tgt = np.zeros_like(A, bool)
    for i, e in enumerate(uns):
        names = {SECTOR_ETF[s] for s in (e.get("targeted_sectors") or []) if s in SECTOR_ETF and s != "broad_market"} | set(country_etfs(e.get("region")))
        for t in names:
            if t in tick:
                tgt[i, tick.index(t)] = True
    # (a) size
    has = tgt.any(1) & (np.isfinite(A).sum(1) > 5)

    def size_ratio(T, mk):
        r = []
        for i in np.where(mk)[0]:
            a_, t_ = np.abs(A[i]), T[i]
            ok = np.isfinite(a_)
            if (t_ & ok).any() and (~t_ & ok).any():
                r.append(np.mean(a_[t_ & ok]) / np.mean(a_[~t_ & ok]))
        return (float(np.mean(r)) if r else np.nan), len(r)

    def size_test(mk):
        o, k = size_ratio(tgt, mk)
        nn = [size_ratio(np.array([rng.permutation(row) for row in tgt]), mk)[0] for _ in range(perms)]
        return {"events": k, "ratio": o, "null95": float(np.nanquantile(nn, 0.95)), "p": float((1 + (np.array(nn) >= o).sum()) / (1 + perms))}

    size = size_test(has)
    by_cat = {}
    for c in cats:
        mk = has & np.array([r["category"] == c for r in uns])
        if mk.sum() >= 15:
            by_cat[c] = size_test(mk)
    for sv in ("extreme", "major"):
        by_cat[f"severity: {sv}"] = size_test(has & (sev == sv))
    print(f"(2a) targeted |move| / other ETFs' |move|: {size['ratio']:.2f} (null95 {size['null95']:.2f}, p {size['p']:.3f}, {size['events']} events)", flush=True)
    for c, v in by_cat.items():
        print(f"      {c:26} {v['ratio']:.2f} (null95 {v['null95']:.2f}, p {v['p']:.3f}, n {v['events']})", flush=True)
    # (b) direction: LOO mean of the same (category, risk direction, target)
    rows = [(i, j) for i in range(n) for j in range(len(tick)) if tgt[i, j] and np.isfinite(A[i, j])]
    key = [(uns[i]["category"], (uns[i].get("direction") or {}).get("risk_sentiment"), tick[j]) for i, j in rows]
    y = np.array([A[i, j] for i, j in rows])

    def loo_pred(yv, shrink=0.0):
        """LOO group mean; shrink > 0 pulls small groups toward zero: sum / (count - 1 + shrink) (fixed before scoring)."""
        df = pd.DataFrame({"k": key, "y": yv})
        g = df.groupby("k")["y"]
        s_, c_ = g.transform("sum"), g.transform("count")
        return np.where(c_ > 1, (s_ - df["y"]) / ((c_ - 1) + shrink), 0.0)

    pr = loo_pred(y)
    r2 = float(1 - ((y - pr) ** 2).sum() / (y ** 2).sum())
    nl = [float(1 - ((yp - loo_pred(yp)) ** 2).sum() / (yp ** 2).sum()) for yp in (rng.permutation(y) for _ in range(perms))]
    hit = float(np.mean(np.sign(pr[pr != 0]) == np.sign(y[pr != 0])))
    dirn = {"pairs": len(y), "r2_vs_zero": r2, "null95": float(np.quantile(nl, 0.95)),
            "p": float((1 + (np.array(nl) >= r2).sum()) / (1 + perms)), "sign_hit": hit}
    print(f"(2b) targeted direction: R2 {r2:+.3f} (null95 {dirn['null95']:+.3f}, p {dirn['p']:.3f}), sign {hit:.2f}, {len(y)} event-target pairs", flush=True)
    for shrink in (5.0, 20.0):
        ps = loo_pred(y, shrink)
        r2s = float(1 - ((y - ps) ** 2).sum() / (y ** 2).sum())
        nls = [float(1 - ((yp - loo_pred(yp, shrink)) ** 2).sum() / (yp ** 2).sum()) for yp in (rng.permutation(y) for _ in range(perms))]
        dirn[f"shrunk_{int(shrink)}"] = {"r2_vs_zero": r2s, "null95": float(np.quantile(nls, 0.95)), "p": float((1 + (np.array(nls) >= r2s).sum()) / (1 + perms))}
        print(f"     shrunk ({shrink:.0f}): R2 {r2s:+.4f} (null95 {dirn[f'shrunk_{int(shrink)}']['null95']:+.4f}, p {dirn[f'shrunk_{int(shrink)}']['p']:.3f})", flush=True)
    (ROOT / "docs/pmrm/results/b3_targets.json").write_text(json.dumps(
        {"per_factor": per, "targeted_size": size, "targeted_size_by_category": by_cat, "targeted_direction": dirn,
         "etfs": tick}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
