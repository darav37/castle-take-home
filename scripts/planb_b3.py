"""Plan B checkpoint B3: event -> factor impact model, held-out tests (BUILD_PLAN B3, PLAN_B §2, §4, §5.3).
No new LLM calls (uses the B2 library and the B1 model).

  python scripts/planb_b3.py [--perms 200]

(a) Scheduled releases, held out by time: train on releases to 2020, test on 2021-2026 (the Kalshi era).
    Factor responses on the release day regressed on type-specific surprises with a shared low-rank structure.
    Two surprise measures: market-based (release-day changes in the 2y and 10y yields and the 5y breakeven) and data
    surprises (first release minus a naive expectation; GDP advance minus GDPNow). The market-based one is partly
    mechanical for rate-sensitive assets (the surprise IS a rates move), so rates factors are reported apart and the
    equity portfolios are the headline.
(b) Unscheduled events, leave one event out: analog prediction from the most similar other events (code-only
    retrieval, §5.3), against no information (zero), the category mean, and random analogs (the chance rate).
Scored in factor space and for portfolios (demo books, Study A, the Study B buckets) through each portfolio's factor
loadings fitted on 2011-2020 daily returns, so any book is handled the same way.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pmrm_audit as A  # noqa: E402
from planb_b1 import total_returns  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402

OUT = ROOT / "docs" / "pmrm" / "results"
B2 = ROOT / "cache" / "pmrm" / "b2"
TEST_FROM = "2021-01-01"
NONRATES = ["market", "dollar", "oil", "gold", "credit", "size", "value"]
RATES = ["rates_2y", "rates_10y"]


def portfolios() -> dict[str, dict[str, float]]:
    P = {}
    for f in sorted((ROOT / "portfolios").glob("*.yaml")):
        pos = yaml.safe_load(f.read_text()).get("positions", [])
        tot = sum(p["value_usd"] for p in pos)
        P[f"book: {f.stem}"] = {p["ticker"]: p["value_usd"] / tot for p in pos}
    P["study_a (XOM,DAL,MSFT)"] = {t: 1 / 3 for t in A.SEED_STOCKS}
    for b, ts in A.STOCK_POOL.items():
        P[f"bucket: {b}"] = {t: 1 / len(ts) for t in ts}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(1729)
    with open(ROOT / "cache" / "pmrm" / "b1" / "model.pkl", "rb") as f:
        m = pickle.load(f)
    F = m["factors_latent"]
    X = (m["anchors"] - m["anchor_mean"]) / m["anchor_sd"]
    sessions = F.index
    lib = json.loads((B2 / "library.json").read_text(encoding="utf-8"))
    P = portfolios()
    tick = sorted({t for w in P.values() for t in w})
    R = total_returns(tick, sessions, "1990-01-01")
    PR = pd.DataFrame({p: sum(R[t].fillna(0) * w for t, w in ws.items()) for p, ws in P.items()})
    for p, ws in P.items():   # a day counts only if every holding has a return
        PR.loc[R[list(ws)].isna().any(axis=1), p] = np.nan
    # portfolio loadings on the factor series, 2011-2020 daily (never touches the test period)
    fitm = (sessions >= "2011-01-03") & (sessions <= "2020-12-31")
    load = {}
    for name, Fac in (("anchored", X), ("latent", F)):
        load[name] = {}
        for p in P:
            d = pd.concat([PR[p], Fac], axis=1)[fitm].dropna()
            A1 = np.column_stack([np.ones(len(d)), d.iloc[:, 1:].to_numpy()])
            load[name][p] = np.linalg.lstsq(A1, d.iloc[:, 0].to_numpy(), rcond=None)[0][1:]
    acols, lcols = list(X.columns), list(F.columns)

    def window_ret(rec_w: dict, p: str) -> float:
        s = PR[p].loc[pd.Timestamp(rec_w["start"]) + pd.Timedelta(days=1): pd.Timestamp(rec_w["end"])]
        return float(np.prod(1 + s) - 1) if s.notna().all() and len(s) else np.nan

    def frame(recs, w="w1"):
        Ya = np.array([[r[w]["factors_anchored"][c] for c in acols] for r in recs])
        Yl = np.array([[r[w]["factors_latent"][c] for c in lcols] for r in recs])
        Yp = np.array([[window_ret(r[w], p) for p in P] for r in recs])
        return Ya, Yl, Yp

    def port_from_factors(Pa: np.ndarray, Pl: np.ndarray, version: str) -> np.ndarray:
        L = np.array([load[version][p] for p in P])          # portfolios x factors
        return (Pa if version == "anchored" else Pl) @ L.T

    rep = {"portfolios": list(P), "test_from": TEST_FROM}

    # ------------------------------------------------------------ (a) scheduled releases, by time
    sch = [r for r in lib if r["kind"] == "scheduled"]
    types = [r["release"] for r in sch]
    levels = sorted(set(types))
    dates = np.array([r["date"] for r in sch])
    tr, te = dates < TEST_FROM, dates >= TEST_FROM
    Ya, Yl, Yp = frame(sch)
    Y = np.hstack([Ya, Yl])
    out_a = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "by_type_test": {lv: int(sum(t == lv for t, x in zip(types, te) if x)) for lv in levels}}
    measures = {
        "market (2y, 10y, breakeven)": np.array([[r["surprise"][k] if r["surprise"][k] is not None else np.nan for k in ("d2y", "d10y", "d5y_breakeven")] for r in sch], float),
        "data surprise (actual - expected)": np.array([[r["surprise"]["data_surprise_z"] if r["surprise"]["data_surprise_z"] is not None else
                                                         (r["surprise"]["data_surprise"] if r["release"] == "gdp" and r["surprise"]["data_surprise"] is not None else np.nan)] for r in sch], float)}
    for mname, S in measures.items():
        mu, sd = np.nanmean(S[tr], 0), np.nanstd(S[tr], 0)
        Sz = (S - mu) / np.where(sd > 0, sd, 1)
        Xd = IM.design(types, Sz, levels)
        Xbase = IM.design(types, np.zeros_like(Sz), levels)       # intercepts only = the type mean
        # rank by blocked time-CV inside the training set
        idx = np.where(tr)[0]
        folds = np.array_split(idx, 5)
        cv = {}
        for rank in (1, 2, 3, 4, 6, Y.shape[1]):
            err = 0.0
            for f_ in folds:
                trn = np.setdiff1d(idx, f_)
                Bc = IM.reduced_rank(Xd[trn], np.nan_to_num(Y[trn]), rank)
                err += ((np.nan_to_num(Y[f_]) - Xd[f_] @ Bc) ** 2).sum()
            cv[rank] = err
        rank = min(cv, key=cv.get)
        B = IM.reduced_rank(Xd[tr], np.nan_to_num(Y[tr]), rank)
        Bb = IM.ridge(Xbase[tr], np.nan_to_num(Y[tr]), 1.0)
        Ph, Pb = Xd[te] @ B, Xbase[te] @ Bb
        Pa_h, Pl_h = Ph[:, :len(acols)], Ph[:, len(acols):]
        Pa_b, Pl_b = Pb[:, :len(acols)], Pb[:, len(acols):]
        ia = [acols.index(c) for c in NONRATES]
        ir = [acols.index(c) for c in RATES]
        res = {"rank": rank if rank < Y.shape[1] else "full",
               "factor_r2_nonrates": IM.r2_vs(Ya[te][:, ia], Pa_h[:, ia], Pa_b[:, ia]),
               "factor_r2_rates (mechanical)": IM.r2_vs(Ya[te][:, ir], Pa_h[:, ir], Pa_b[:, ir]),
               "factor_r2_latent": IM.r2_vs(Yl[te], Pl_h, Pl_b), "portfolios": {}}
        # portfolios, via factors (anchored and latent) and direct
        base_p = np.array([np.nanmean(Yp[tr][np.array(types)[tr] == t], 0) for t in np.array(types)[te]])
        for version in ("anchored", "latent"):
            pp = port_from_factors(Pa_h, Pl_h, version)
            ppb = port_from_factors(Pa_b, Pl_b, version)
            for j, p in enumerate(P):
                res["portfolios"].setdefault(p, {})[f"r2_via_{version}"] = IM.r2_vs(Yp[te][:, j], pp[:, j], ppb[:, j])
        Bd = IM.ridge(Xd[tr], np.nan_to_num(Yp[tr]), 1.0)
        pd_ = Xd[te] @ Bd
        for j, p in enumerate(P):
            res["portfolios"][p]["r2_direct"] = IM.r2_vs(Yp[te][:, j], pd_[:, j], base_p[:, j])
        # chance: shuffle surprises within type across test releases (model fixed)
        nulls = []
        te_idx = np.where(te)[0]
        for _ in range(a.perms):
            Sp = Sz.copy()
            for lv in levels:
                ii = te_idx[np.array(types)[te_idx] == lv]
                Sp[ii] = Sz[rng.permutation(ii)]
            Php = IM.design(types, Sp, levels)[te] @ B
            ppn = port_from_factors(Php[:, :len(acols)], Php[:, len(acols):], "anchored")
            nulls.append({"nonrates": IM.r2_vs(Ya[te][:, ia], Php[:, :len(acols)][:, ia], Pa_b[:, ia]),
                          "ports": [IM.r2_vs(Yp[te][:, j], ppn[:, j], port_from_factors(Pa_b, Pl_b, "anchored")[:, j]) for j in range(len(P))]})
        nn = np.array([n["nonrates"] for n in nulls])
        res["factor_r2_nonrates_null95"] = float(np.quantile(nn, 0.95))
        res["factor_r2_nonrates_p"] = float((1 + (nn >= res["factor_r2_nonrates"]).sum()) / (1 + len(nn)))
        npm = np.array([n["ports"] for n in nulls])
        for j, p in enumerate(P):
            res["portfolios"][p]["null95_via_anchored"] = float(np.quantile(npm[:, j], 0.95))
            res["portfolios"][p]["p_via_anchored"] = float((1 + (npm[:, j] >= res["portfolios"][p]["r2_via_anchored"]).sum()) / (1 + len(npm)))
        # by release type (market factor and the balanced book, via anchored)
        bt = {}
        ttypes = np.array(types)[te]
        jm = acols.index("market")
        jb = list(P).index("book: balanced")
        ppa = port_from_factors(Pa_h, Pl_h, "anchored")
        ppab = port_from_factors(Pa_b, Pl_b, "anchored")
        for lv in levels:
            mk = ttypes == lv
            if mk.sum() >= 10:
                bt[lv] = {"n": int(mk.sum()), "market_factor_r2": IM.r2_vs(Ya[te][mk, jm], Pa_h[mk, jm], Pa_b[mk, jm]),
                          "balanced_book_r2": IM.r2_vs(Yp[te][mk, jb], ppa[mk, jb], ppab[mk, jb])}
        res["by_type"] = bt
        # PLAN_B §4 rolling variant: each test year refitted on the previous 8 years only (same rank, no new tuning)
        yrs = np.array([int(d[:4]) for d in dates])
        Ph_r = np.full((te.sum(), Y.shape[1]), np.nan)
        Pb_r = np.full_like(Ph_r, np.nan)
        te_pos = {g: k for k, g in enumerate(np.where(te)[0])}
        for yv in sorted(set(yrs[te])):
            trw = (yrs >= yv - 8) & (yrs <= yv - 1)
            tey = te & (yrs == yv)
            Bw = IM.reduced_rank(Xd[trw], np.nan_to_num(Y[trw]), rank)
            Bbw = IM.ridge(Xbase[trw], np.nan_to_num(Y[trw]), 1.0)
            for g in np.where(tey)[0]:
                Ph_r[te_pos[g]] = Xd[g] @ Bw
                Pb_r[te_pos[g]] = Xbase[g] @ Bbw
        res["rolling_8y"] = {"factor_r2_nonrates": IM.r2_vs(Ya[te][:, ia], Ph_r[:, :len(acols)][:, ia], Pb_r[:, :len(acols)][:, ia]),
                             "portfolios_via_anchored": {}}
        ppr = port_from_factors(Ph_r[:, :len(acols)], Ph_r[:, len(acols):], "anchored")
        pprb = port_from_factors(Pb_r[:, :len(acols)], Pb_r[:, len(acols):], "anchored")
        for j, p in enumerate(P):
            res["rolling_8y"]["portfolios_via_anchored"][p] = IM.r2_vs(Yp[te][:, j], ppr[:, j], pprb[:, j])
        out_a[mname] = res
        print(f"(a) {mname}: rank {res['rank']}, nonrates R2 {res['factor_r2_nonrates']:.3f} (null95 {res['factor_r2_nonrates_null95']:.3f}), "
              f"rates {res['factor_r2_rates (mechanical)']:.3f}, latent {res['factor_r2_latent']:.3f}", flush=True)
    # stage 1 of the bridge: does the data surprise predict the rates reaction?
    br = {}
    for lv in levels:
        ii = [i for i, t in enumerate(types) if t == lv]
        ds = measures["data surprise (actual - expected)"][ii, 0]
        d2 = measures["market (2y, 10y, breakeven)"][ii, 0]
        ok = ~np.isnan(ds) & ~np.isnan(d2)
        if ok.sum() >= 20:
            br[lv] = {"n": int(ok.sum()), "corr_data_surprise_vs_d2y": float(np.corrcoef(ds[ok], d2[ok])[0, 1])}
    out_a["bridge_data_surprise_to_rates"] = br
    rep["scheduled"] = out_a

    # ------------------------------------------------------------ (b) unscheduled events, leave one out
    uns = [r for r in lib if r["kind"] == "unscheduled"]
    for w in ("w1", "w5"):
        Ua, Ul, Up = frame(uns, w)
        U = np.hstack([Ua, Ul])
        n = len(uns)
        dts = np.array([r["date"] for r in uns])
        # scale (Oct 20, library growth): similarity matrices once, closed-form leave-one-out for the ridge model
        Sb = np.array([[IM.similarity(x, y) for y in uns] for x in uns])
        Sd = np.array([[IM.direction_similarity(x, y) for y in uns] for x in uns])
        np.fill_diagonal(Sb, -np.inf)
        np.fill_diagonal(Sd, -np.inf)

        def from_matrix(S, mask=None, k=5):
            Pm, sp = np.full_like(U, np.nan), np.zeros(n)
            for i in range(n):
                row = S[i] if mask is None else np.where(mask[i], S[i], -np.inf)
                if np.isfinite(row).sum() < (k if mask is None else 10):
                    continue
                top = np.argpartition(-row, k)[:k]
                w_ = np.maximum(row[top], 1e-6)
                w_ = w_ / w_.sum()
                Pm[i] = w_ @ U[top]
                sp[i] = float(np.sqrt((w_[:, None] * (U[top] - Pm[i]) ** 2).sum(0).mean()))
            return Pm, sp

        preds = {}
        preds["analog (all others)"], spread = from_matrix(Sb)
        preds["analog (earlier events only)"], _ = from_matrix(Sb, mask=dts[None, :] < dts[:, None])
        cats = np.array([r["category"] for r in uns])
        cm = np.zeros_like(U)
        for c in set(cats):
            mk = cats == c
            tot, cnt = U[mk].sum(0), mk.sum()
            cm[mk] = (tot - U[mk]) / max(cnt - 1, 1)
        preds["category mean"] = cm
        preds["analog within direction"], _ = from_matrix(Sd)
        Fd = np.column_stack([np.ones(n), np.array([IM.direction_features(e) for e in uns])])
        preds["direction model (code)"] = IM.ridge_loo_predictions(Fd, U, 2.0)
        Sg = np.zeros_like(U)
        for i, e in enumerate(uns):
            for c, sgn in (((e.get("direction") or {}).get("factor_signs")) or {}).items():
                if c in acols and sgn is not None:
                    Sg[i, acols.index(c)] = sgn
        mean_abs_loo = (np.abs(U).sum(0)[None, :] - np.abs(U)) / (n - 1)
        preds["LLM factor signs (hindsight risk)"] = Sg * mean_abs_loo
        zero = np.zeros_like(U)
        ia = [acols.index(c) for c in NONRATES]
        res = {}
        for name, Pm in preds.items():
            ok = ~np.isnan(Pm).any(1)
            pa, pl = Pm[:, :len(acols)], Pm[:, len(acols):]
            pp = port_from_factors(pa, pl, "anchored")
            res[name] = {"events": int(ok.sum()),
                         "factor_r2_vs_zero_nonrates": IM.r2_vs(Ua[ok][:, ia], pa[ok][:, ia], zero[ok][:, :len(acols)][:, ia]),
                         "factor_r2_vs_zero_latent": IM.r2_vs(Ul[ok], pl[ok], zero[ok][:, len(acols):]),
                         "portfolio_r2_vs_zero": {p: IM.r2_vs(Up[ok][:, j], pp[ok][:, j], np.zeros(ok.sum())) for j, p in enumerate(P)},
                         "market_sign_hit": float(np.mean(np.sign(pa[ok][:, acols.index("market")]) == np.sign(Ua[ok][:, acols.index("market")]))),
                         "market_sign_hit_when_called": float(np.mean((np.sign(pa[ok][:, acols.index("market")]) == np.sign(Ua[ok][:, acols.index("market")]))[pa[ok][:, acols.index("market")] != 0])) if (pa[ok][:, acols.index("market")] != 0).any() else None,
                         "sign_hit_by_factor": {c: float(np.mean((np.sign(pa[ok][:, acols.index(c)]) == np.sign(Ua[ok][:, acols.index(c)]))[pa[ok][:, acols.index(c)] != 0])) if (pa[ok][:, acols.index(c)] != 0).any() else None for c in NONRATES + RATES}}
        # chance: random analogs (same k, same weighting) -> null distribution of the analog R²
        nulls, pnull = [], []
        for _ in range(a.perms):
            J = rng.integers(0, n - 1, (n, 5))
            J = J + (J >= np.arange(n)[:, None])          # never the event itself
            Pn = U[J].mean(1)
            nulls.append(IM.r2_vs(Ua[:, ia], Pn[:, :len(acols)][:, ia], zero[:, :len(acols)][:, ia]))
            ppn = port_from_factors(Pn[:, :len(acols)], Pn[:, len(acols):], "anchored")
            pnull.append([IM.r2_vs(Up[:, j], ppn[:, j], np.zeros(n)) for j in range(len(P))])
        nulls, pnull = np.array(nulls), np.array(pnull)
        an = res["analog (all others)"]
        an["random_analogs_null95_nonrates"] = float(np.quantile(nulls, 0.95))
        an["p_vs_random_analogs"] = float((1 + (nulls >= an["factor_r2_vs_zero_nonrates"]).sum()) / (1 + len(nulls)))
        an["portfolio_null95"] = {p: float(np.quantile(pnull[:, j], 0.95)) for j, p in enumerate(P)}
        # by severity, and the analog spread as an uncertainty band (does a wider band go with bigger errors?)
        sev = np.array([r.get("severity") or "unset" for r in uns])
        pa = preds["analog (all others)"][:, :len(acols)]
        bys = {}
        for s in ("extreme", "major", "moderate"):
            mk = sev == s
            if mk.sum() >= 5:
                bys[s] = {"n": int(mk.sum()), "nonrates_r2_vs_zero": IM.r2_vs(Ua[mk][:, ia], pa[mk][:, ia], np.zeros((mk.sum(), len(ia)))),
                          "realized_abs_market_mean": float(np.abs(Ua[mk][:, acols.index("market")]).mean())}
        an["by_severity"] = bys
        err = np.sqrt(((U - preds["analog (all others)"]) ** 2).mean(1))
        okb = ~np.isnan(err)
        an["corr_band_vs_error"] = float(np.corrcoef(spread[okb], err[okb])[0, 1])
        # exploratory (added after seeing direction fail; labelled as such): do analogs predict the SIZE of the move?
        # predicted |response| = similarity-weighted mean of the analogs' |responses|; baseline = mean |response| of all
        # other events; scored on |realized| (non-rates anchored factors and portfolio |moves|).
        absU, absP = np.abs(U), np.abs(Up)
        mag, magp, basem, basep = np.zeros_like(U), np.zeros_like(Up), np.zeros_like(U), np.zeros_like(Up)
        for i in range(n):
            top = np.argpartition(-Sb[i], 5)[:5]
            wts = np.maximum(Sb[i, top], 1e-6) / np.maximum(Sb[i, top], 1e-6).sum()
            mag[i], magp[i] = wts @ absU[top], wts @ np.nan_to_num(absP[top])
            o = np.arange(n) != i
            basem[i], basep[i] = absU[o].mean(0), np.nanmean(absP[o], 0)
        an["exploratory_magnitude"] = {
            "nonrates_r2_vs_mean_abs": IM.r2_vs(absU[:, ia], mag[:, ia], basem[:, ia]),
            "portfolios_r2_vs_mean_abs": {p: IM.r2_vs(absP[:, j], magp[:, j], basep[:, j]) for j, p in enumerate(P)},
            "corr_pred_vs_realized_abs_market": float(np.corrcoef(mag[:, acols.index("market")], absU[:, acols.index("market")])[0, 1])}
        rep[f"unscheduled_{w}"] = res
        print(f"(b) {w}: analog nonrates R2 vs zero {an['factor_r2_vs_zero_nonrates']:.3f} (random-analog null95 "
              f"{an['random_analogs_null95_nonrates']:.3f}, p {an['p_vs_random_analogs']:.3f}); category mean "
              f"{res['category mean']['factor_r2_vs_zero_nonrates']:.3f}; past-only {res['analog (earlier events only)']['factor_r2_vs_zero_nonrates']:.3f}", flush=True)
        for nm in ("analog within direction", "direction model (code)", "LLM factor signs (hindsight risk)"):
            print(f"    {nm}: nonrates R2 {res[nm]['factor_r2_vs_zero_nonrates']:.3f}, market sign {res[nm]['market_sign_hit_when_called']}", flush=True)
    (OUT / "b3.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT / "b3.json")


if __name__ == "__main__":
    main()
