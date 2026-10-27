"""PMRM checkpoint P0c: the basic question (BUILD_PLAN P0c, DECISIONS #151). No LLM calls.

Does conditioning on prediction-market odds moves predict portfolio P&L better than ignoring them, on held-out days?

  python scripts/pmrm_p0c.py [--shifts 50] [--strict]

Events: the stitched front-contract series of the "in" Study B families, plus liquid single markets from the current
build with >= 200 clean rows. Portfolios: Study A (XOM, DAL, MSFT), the demo `balanced` book, one equal-dollar portfolio
per Study B stock bucket. Daily 16:00 ET intervals from 2022, carried quotes (strict with --strict).
Rolling 120-row training window, refit every 20 rows, predictions frozen before each scored row, raw scoring.

Models, each against M0 (unconditional Gaussian from the training window):
  M1 single: portfolio = mean + slope x event move, variance = residual variance (one event at a time).
  M1 joint: Gaussian conditioning on every observed event move at once, shrunk covariance (Ledoit-Wolf-style
            intensity toward the diagonal, pairwise moments; a missing move is missing, never zero).
  Ridge joint (check): ridge on the events, penalty by blocked CV inside the training window only.
"Beyond the market": the same on portfolio return minus beta x SPY (beta from the training window).
An event move enters a prediction only if the event is valid and interior at the start of the day, |dp| <= 0.05 and
|z| <= 3 against its training sd (§6 support gate); otherwise that event is unobserved for the day.

Chance: every event series circularly shifted by its own random 20+ day offset against the portfolios, 50 times;
p = (1 + #null >= observed) / (1 + shifts). The count of "significant" pairs is compared with its own null
distribution (each shift scored against the others), so multiplicity is handled by measurement too.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pmrm_audit as A  # noqa: E402

OUT = ROOT / "docs" / "pmrm" / "results"
TRAIN, REFIT, MIN_TRAIN_X = 120, 20, 40
FAMILIES = {"fed": "KXFEDDECISION", "cpi": "KXCPI", "payrolls": "KXPAYROLLS", "recession": "KXRECSSNBER",
            "gdp": "KXGDP", "us_iran": "KXUSAIRANAGREEMENT", "house": "CONTROLH", "senate": "CONTROLS"}
BALANCED = {"SPY": 25000, "QQQ": 15000, "NVDA": 10000, "TLT": 15000, "GLD": 10000, "XLE": 10000, "HYG": 15000}


# ---------------------------------------------------------------- panel

def event_panel(calb, ivb, rd: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """dp per interval (NaN unless valid and interior at the start) and the start p, per event."""
    dps, meta = {}, {}
    for name, series in FAMILIES.items():
        rows, info = A.rolling_variable(series, None, calb, ivb)
        ok = rows[f"valid_{rd}"] & rows["interior"]
        dps[name] = rows["dp"].where(ok)
        meta[name] = {"kind": "family", "series": series, "clean": int(ok.sum()), **info}
        print(f"  {name:10} {int(ok.sum())} clean", flush=True)
    b = ROOT / "cache" / "build" / (ROOT / "cache" / "build" / "LATEST").read_text().strip()
    built = json.loads((b / "markets.json").read_text(encoding="utf-8"))
    by_series = {}
    for m in built:
        by_series.setdefault(m.get("series_ticker"), []).append(m["id"])
    for series, ids in by_series.items():
        if not series:
            continue
        mk = {m["ticker"]: m for m in A.all_markets(series)}
        for mid in ids:
            m = mk.get(mid)
            if m is None or m["open"] is None:
                continue
            rows, _ = A.fixed_variable(m, calb, ivb)
            ok = rows[f"valid_{rd}"] & rows["interior"]
            if ok.sum() >= 200:
                dps[mid] = rows["dp"].where(ok)
                meta[mid] = {"kind": "single", "series": series, "clean": int(ok.sum())}
                print(f"  single {mid} {int(ok.sum())} clean", flush=True)
    return pd.DataFrame(dps), meta


def portfolios(calb, ivb) -> pd.DataFrame:
    rets = {}
    tick = set(A.SEED_STOCKS) | set(BALANCED) | {t for ts in A.STOCK_POOL.values() for t in ts}
    for t in sorted(tick):
        rets[t], _ = A.stock_returns(t, calb, ivb)
    R = pd.DataFrame(rets)
    P = {"study_a (XOM,DAL,MSFT)": R[A.SEED_STOCKS].mean(axis=1, skipna=False)}
    w = pd.Series(BALANCED) / sum(BALANCED.values())
    P["balanced book"] = (R[list(BALANCED)] * w).sum(axis=1, min_count=len(w))
    for bucket, ts in A.STOCK_POOL.items():
        P[f"bucket: {bucket}"] = R[ts].mean(axis=1, skipna=False)
    return pd.DataFrame(P), R["SPY"]


# ---------------------------------------------------------------- models

def nls(y, m, v):
    return 0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)


def gate(x_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """Support gate for the scored day's event move: |dp| <= 0.05 and |z| <= 3 vs the training sd."""
    v = x_tr[~np.isnan(x_tr)]
    if len(v) < MIN_TRAIN_X:
        return np.full_like(x_te, np.nan)
    sd = v.std() or 1e-9
    ok = (np.abs(x_te) <= 0.05) & (np.abs(x_te - v.mean()) <= 3 * sd)
    return np.where(ok, x_te, np.nan)


def run_single(y: np.ndarray, x: np.ndarray, spy: np.ndarray | None) -> dict | None:
    """M1 single vs M0 on the rows where y is observed. Rows with no usable x are scored by both models as M0
    (identical), so they add zero to the paired gain; the summary also reports the conditioned-row count."""
    keep = ~np.isnan(y) & (spy is None or True)
    if spy is not None:
        keep &= ~np.isnan(spy)
    y, x = y[keep], x[keep]
    s = spy[keep] if spy is not None else None
    n = len(y)
    g, se0, se1, used = [], 0.0, 0.0, 0
    for s0 in range(TRAIN, n, REFIT):
        tr, te = slice(s0 - TRAIN, s0), slice(s0, min(s0 + REFIT, n))
        ytr, yte = y[tr].copy(), y[te].copy()
        if s is not None:      # beyond the market: residual on SPY with the training beta
            b = np.cov(ytr, s[tr])[0, 1] / max(s[tr].var(ddof=1), 1e-12)
            ytr, yte = ytr - b * s[tr], yte - b * s[te]
        m0, v0 = ytr.mean(), ytr.var(ddof=1)
        xtr = x[tr]
        ok = ~np.isnan(xtr)
        xte = gate(xtr, x[te])
        if ok.sum() >= MIN_TRAIN_X:
            # condition M0's own Gaussian (same mean and variance base) through the training correlation only, so a
            # gain can't come from the quoted days being calmer than the rest (Oct 20 fix)
            xm = xtr[ok].mean()
            rho = np.corrcoef(xtr[ok], ytr[ok])[0, 1] if xtr[ok].std() > 0 else 0.0
            slope = rho * math.sqrt(v0) / xtr[ok].std()
            m1 = np.where(np.isnan(xte), m0, m0 + slope * (np.nan_to_num(xte) - xm))
            vv = np.where(np.isnan(xte), v0, v0 * (1 - rho ** 2))
        else:
            m1, vv = np.full(len(yte), m0), np.full(len(yte), v0)
        used += int((~np.isnan(xte)).sum()) if ok.sum() >= MIN_TRAIN_X else 0
        g.append(nls(yte, m0, v0) - nls(yte, m1, vv))
        se0 += ((yte - m0) ** 2).sum()
        se1 += ((yte - m1) ** 2).sum()
    if not g:
        return None
    g = np.concatenate(g)
    return {"rows": int(len(g)), "conditioned_rows": used, "r2": 1 - se1 / se0 if se0 > 0 else 0.0,
            "gain": float(g.mean()), "_g": g}


def shrunk_cov(Z: np.ndarray) -> np.ndarray:
    """Pairwise covariance of standardized columns with missing values, shrunk toward the identity with a
    Ledoit-Wolf-style intensity (estimated on zero-filled standardized data), then made PSD."""
    d = Z.shape[1]
    C = np.eye(d)
    for a in range(d):
        for b in range(a + 1, d):
            ok = ~np.isnan(Z[:, a]) & ~np.isnan(Z[:, b])
            C[a, b] = C[b, a] = (Z[ok, a] * Z[ok, b]).mean() if ok.sum() >= 10 else 0.0
    X = np.nan_to_num(Z)
    n = len(X)
    S = X.T @ X / n
    pi = sum(((np.outer(r, r) - S) ** 2).sum() for r in X) / n
    gam = ((S - np.eye(d)) ** 2).sum()
    delta = float(np.clip(pi / n / gam if gam > 0 else 1.0, 0, 1))
    C = (1 - delta) * C + delta * np.eye(d)
    w, V = np.linalg.eigh(C)
    return (V * np.maximum(w, 1e-4)) @ V.T


def run_joint(y: np.ndarray, X: np.ndarray, spy: np.ndarray | None, ridge: bool = False) -> dict | None:
    keep = ~np.isnan(y)
    if spy is not None:
        keep &= ~np.isnan(spy)
    y, X = y[keep], X[keep]
    s = spy[keep] if spy is not None else None
    n = len(y)
    g, se0, se1, used = [], 0.0, 0.0, 0
    for s0 in range(TRAIN, n, REFIT):
        tr, te = slice(s0 - TRAIN, s0), slice(s0, min(s0 + REFIT, n))
        ytr, yte = y[tr].copy(), y[te].copy()
        if s is not None:
            b = np.cov(ytr, s[tr])[0, 1] / max(s[tr].var(ddof=1), 1e-12)
            ytr, yte = ytr - b * s[tr], yte - b * s[te]
        m0, v0 = ytr.mean(), ytr.var(ddof=1)
        cols = [j for j in range(X.shape[1]) if (~np.isnan(X[tr, j])).sum() >= MIN_TRAIN_X]
        m1, vv = np.full(len(yte), m0), np.full(len(yte), v0)
        if cols:
            Xtr = X[tr][:, cols]
            Xte = np.column_stack([gate(Xtr[:, k], X[te][:, cols[k]]) for k in range(len(cols))])
            mu = np.nanmean(Xtr, 0)
            sd = np.nanstd(Xtr, 0)
            sd[sd == 0] = 1e-9
            if not ridge:
                Z = np.column_stack([(ytr - m0) / math.sqrt(v0), (Xtr - mu) / sd])
                C = shrunk_cov(Z)
                for i in range(len(yte)):
                    o = ~np.isnan(Xte[i])
                    if not o.any():
                        continue
                    idx = 1 + np.where(o)[0]
                    Coo = C[np.ix_(idx, idx)]
                    cyo = C[0, idx]
                    zx = (Xte[i, o] - mu[o]) / sd[o]
                    w = np.linalg.solve(Coo, cyo)
                    m1[i] = m0 + math.sqrt(v0) * (w @ zx)
                    vv[i] = v0 * max(1 - cyo @ w, 1e-3)
                    used += 1
            else:
                Ztr = np.nan_to_num((Xtr - mu) / sd)       # ridge only: missing = training mean (documented)
                Zte = np.nan_to_num((Xte - mu) / sd)
                yc = ytr - m0
                best, lam_best = None, 1.0
                folds = np.array_split(np.arange(len(yc)), 5)
                for lam in (0.1, 1, 10, 100, 1000):
                    err = 0.0
                    for f in folds:
                        trn = np.setdiff1d(np.arange(len(yc)), f)
                        A_ = Ztr[trn]
                        beta = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ yc[trn])
                        err += ((yc[f] - Ztr[f] @ beta) ** 2).sum()
                    if best is None or err < best:
                        best, lam_best = err, lam
                beta = np.linalg.solve(Ztr.T @ Ztr + lam_best * np.eye(Ztr.shape[1]), Ztr.T @ yc)
                m1 = m0 + Zte @ beta
                vv = np.full(len(yte), max(best / len(yc), 1e-12))
                used += int((np.abs(Zte).sum(1) > 0).sum())
        g.append(nls(yte, m0, v0) - nls(yte, m1, vv))
        se0 += ((yte - m0) ** 2).sum()
        se1 += ((yte - m1) ** 2).sum()
    if not g:
        return None
    g = np.concatenate(g)
    return {"rows": int(len(g)), "conditioned_rows": used, "r2": 1 - se1 / se0 if se0 > 0 else 0.0,
            "gain": float(g.mean()), "_g": g}


def block_t(g: np.ndarray, block: int = 5) -> float:
    nb = len(g) // block
    if nb < 4:
        return float("nan")
    b = g[: nb * block].reshape(nb, block).mean(1)
    sd = b.std(ddof=1)
    return float(b.mean() / (sd / math.sqrt(nb))) if sd > 0 else 0.0


# ---------------------------------------------------------------- main

def shifted(E: pd.DataFrame, rng) -> pd.DataFrame:
    out = {}
    n = len(E)
    for c in E.columns:
        k = int(rng.integers(20, n - 20))
        out[c] = np.roll(E[c].to_numpy(), k)
    return pd.DataFrame(out, index=E.index)


def evaluate(E: pd.DataFrame, P: pd.DataFrame, spy: pd.Series) -> dict:
    res = {}
    S = spy.to_numpy()
    for beyond in (False, True):
        tag = "beyond_market" if beyond else "raw"
        for pname in P.columns:
            y = P[pname].to_numpy()
            if beyond and pname.startswith("bucket: placebo") is False and False:
                pass
            for ev in E.columns:
                r = run_single(y, E[ev].to_numpy(), S if beyond else None)
                if r:
                    res[(tag, pname, ev)] = r
            for kind, rid in (("ALL JOINTLY (shrunk cov)", False), ("ALL JOINTLY (ridge)", True)):
                r = run_joint(y, E.to_numpy(), S if beyond else None, ridge=rid)
                if r:
                    res[(tag, pname, kind)] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=int, default=50)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    rd = "strict" if args.strict else "carried"
    rng = np.random.default_rng(A.SPEC["seed"])
    calb = A.sessions(A.POOL_START, A.AUDIT_END)
    ivb = A.intervals(calb)
    elig = ivb["eligible"].to_numpy()
    E, meta = event_panel(calb, ivb, rd)
    P, spy = portfolios(calb, ivb)
    E, P, spy = E[elig].reset_index(drop=True), P[elig].reset_index(drop=True), spy[elig].reset_index(drop=True)
    print(f"panel: {len(E)} intervals, {E.shape[1]} events, {P.shape[1]} portfolios", flush=True)
    obs = evaluate(E, P, spy)
    nulls = []
    for i in range(args.shifts):
        nulls.append(evaluate(shifted(E, rng), P, spy))
        if (i + 1) % 10 == 0:
            print(f"  shifts {i + 1}/{args.shifts}", flush=True)
    rows = []
    for key, r in obs.items():
        nr2 = np.array([nd[key]["r2"] for nd in nulls if key in nd])
        ng = np.array([nd[key]["gain"] for nd in nulls if key in nd])
        rows.append({"scope": key[0], "portfolio": key[1], "event": key[2], "rows": r["rows"],
                     "conditioned_rows": r["conditioned_rows"], "r2": r["r2"], "r2_null95": float(np.quantile(nr2, 0.95)),
                     "gain": r["gain"], "gain_null95": float(np.quantile(ng, 0.95)),
                     "p_r2": (1 + (nr2 >= r["r2"]).sum()) / (1 + len(nr2)),
                     "p_gain": (1 + (ng >= r["gain"]).sum()) / (1 + len(ng)), "block_t": block_t(r["_g"])})
    df = pd.DataFrame(rows)
    # multiplicity, measured: how many pairs reach p_gain <= 0.05 in each null replicate (scored against the others)
    keys = list(obs)
    G = np.array([[nd[k]["gain"] if k in nd else np.nan for k in keys] for nd in nulls])   # shifts x pairs
    null_counts = []
    for i in range(len(nulls)):
        others = np.delete(G, i, axis=0)
        p = (1 + (others >= G[i]).sum(0)) / (1 + len(others))
        null_counts.append(int((p <= 0.05).sum()))
    summary = {}
    for scope in ("raw", "beyond_market"):
        sub = df[df["scope"] == scope]
        single = sub[~sub["event"].str.startswith("ALL JOINTLY")]
        jt = sub[sub["event"].str.startswith("ALL JOINTLY")]
        summary[scope] = {"pairs": int(len(single)), "pairs_p_gain_le_05": int((single["p_gain"] <= 0.05).sum()),
                          "pairs_r2_above_null95": int((single["r2"] > single["r2_null95"]).sum()),
                          "joint_cells": int(len(jt)), "joint_p_gain_le_05": int((jt["p_gain"] <= 0.05).sum()),
                          "joint_r2_above_null95": int((jt["r2"] > jt["r2_null95"]).sum())}
    allsig = int((df["p_gain"] <= 0.05).sum())
    out = {"reading": rd, "shifts": args.shifts, "panel": {"intervals": int(len(E)), "events": meta,
                                                           "portfolios": list(P.columns)},
           "summary": summary, "all_cells_p_le_05": allsig,
           "null_count_p_le_05": {"mean": float(np.mean(null_counts)), "p95": float(np.quantile(null_counts, 0.95)),
                                  "max": int(max(null_counts))},
           "cells": df.to_dict("records")}
    name = "p0c" + ("_strict" if args.strict else "") + ".json"
    (OUT / name).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary), allsig, out["null_count_p_le_05"])
    print("wrote", OUT / name)


if __name__ == "__main__":
    main()
