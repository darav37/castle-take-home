"""Plan C, C1c: surprise, positioning and the LLM's directional read, inside the real-event gate (PLAN_C §2a groups 1,
3, 4b; DECISIONS #194). No LLM calls.

  python scripts/planc_c1c.py fetch      # CFTC Commitments of Traders (legacy, futures only), 1986-2026, free
  python scripts/planc_c1c.py score

New features, all as of the close before the event window:
  positioning (CFTC)   12 futures markets (WTI, gold, 2y and 10y notes, dollar index, yen, euro, E-mini S&P, copper,
                       corn, soybeans, wheat): speculators' net position as a share of open interest, its z-score vs the
                       previous 3 years, and its 4-week change. A report (Tuesday positions) counts only from its Friday
                       release, so it's known at least 3 days later
  anticipation premium implied vol vs realized: VIX / SPY's 20-day realized vol, oil VIX / USO's, gold VIX / GLD's (log)
  LLM directional read the B2 schema's per-factor sign (market, 2y, 10y, dollar, oil, gold, credit), written from the
                       event text without outcomes (#160). It already calls real moves like #189's naive LLM (51% on
                       moves < 0.5x noise, 58% at 1-2x, 60% at 2-3x, 70% above 3x on 2004-2026), and today's engine
                       averages it away because most events are noise
Not built (and why): prediction-market odds (7 of 3,876 library events have a market); CBOE put/call (the free history
stops in Oct 2019, so nothing to score on); news-volume build-up (GDELT needs per-event search terms; left for C2,
where the text pipeline can supply them); options-implied moves beyond the CBOE vol indices (VIX, OVX, GVZ are in C1a).

The gate model (C1b, #192) is extended so the LLM's sign moves only the real-event part:
    z ~ (1 - g) * Normal(e * sign * s0, s0^2) + g * Normal(d * sign * s0, (k * s0)^2)
d is the real-event push in the LLM's direction, e a small push on every event (expected ~0). Rungs on the frozen test:
  0           zero + normal noise (the bar)
  A gate      C1b's gate on C1a features (refitted with the #193 controls)
  B +C1c      A plus positioning and premium in the gate and noise
  C +sign     B plus the LLM sign in the gate's real part (d, e) and |sign| as a gate feature
  D gate+sign A plus the LLM sign (added after the first run showed B overfits)
"""

import argparse
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import planb_b1 as B1  # noqa: E402
from planc_c1b_gate import STATE_PREFIX, auc, controls  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.planc import harness as H  # noqa: E402

DIR = ROOT / "cache/pmrm/planc"
RAW = DIR / "raw"
CFTC = {"067651": "wti", "088691": "gold", "043602": "t10y", "042601": "t2y", "098662": "usd", "097741": "yen", "099741": "euro",
        "13874A": "es", "085692": "copper", "002602": "corn", "005602": "soy", "001602": "wheat"}
SIGN_FACTORS = ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit"]


def step_fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    names = ["1986_2016"] + [str(y) for y in range(2017, 2027)]
    for n in names:
        f = RAW / f"deacot{n}.zip"
        if f.exists() and n != "2026":
            continue
        r = requests.get(f"https://www.cftc.gov/files/dea/history/deacot{n}.zip", headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
        print(f"  {f.name}: {len(r.content) / 1e6:.1f} MB", flush=True)
    parts = []
    for f in sorted(RAW.glob("deacot*.zip")):
        with zipfile.ZipFile(f) as z:
            d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False, dtype={"CFTC Contract Market Code": str},
                            usecols=["CFTC Contract Market Code", "As of Date in Form YYYY-MM-DD", "Open Interest (All)",
                                     "Noncommercial Positions-Long (All)", "Noncommercial Positions-Short (All)"])
        d = d.rename(columns={"CFTC Contract Market Code": "code", "As of Date in Form YYYY-MM-DD": "date", "Open Interest (All)": "oi",
                              "Noncommercial Positions-Long (All)": "nc_long", "Noncommercial Positions-Short (All)": "nc_short"})
        d["code"] = d.code.astype(str).str.strip()
        parts.append(d[d.code.isin(CFTC)])
    P = pd.concat(parts)
    P["date"] = pd.to_datetime(P.date)
    P["market"] = P.code.map(CFTC)
    P = P.drop_duplicates(["market", "date"]).sort_values(["market", "date"])
    P["net"] = (P.nc_long - P.nc_short) / P.oi
    P.to_pickle(DIR / "c1c_cftc.pkl")
    print(P.groupby("market").date.agg(["min", "max", "count"]))


def positioning(P: pd.DataFrame, day: pd.Timestamp) -> dict:
    """As of the close of `day`: only reports released (as-of date + 3 days) on or before it."""
    out = {}
    for m, g in P.groupby("market"):
        g = g[g.date + pd.Timedelta(days=3) <= day]
        if len(g) < 60:
            continue
        net = g.net.to_numpy()
        hist = net[-157:-1]
        out[f"pos_{m}_net"] = float(net[-1])
        out[f"pos_{m}_z"] = float((net[-1] - hist.mean()) / hist.std()) if hist.std() > 0 else 0.0
        out[f"pos_{m}_d4"] = float(net[-1] - net[-5])
    return out


def premium(R, FG, a, ss):
    out = {}
    for name, t, col in (("vix", "SPY", "fg_vix"), ("ovx", "USO", "fg_ovx"), ("gvz", "GLD", "fg_gvz")):
        rv = float(np.nanstd(R[t].iloc[a - 20:a]) * np.sqrt(252) * 100)
        iv = FG.get(col, np.nan)
        out[f"pr_{name}_iv_rv"] = float(np.log(iv / rv)) if rv > 0 and iv is not None and np.isfinite(iv) else np.nan
    return out


def fit_dir(Xs, Xa, z, sgn, lam, use_sign):
    """C1b's two-part model, with the LLM's sign shifting the real part (d) and, slightly, every event (e).
    Maximum likelihood with an exact gradient (log domain), L2 on the state and gate weights."""
    ps, pa = Xs.shape[1], Xa.shape[1]
    sg = np.asarray(sgn, float)

    def unpack(th):
        return th[:ps], th[ps:ps + pa], th[ps + pa], (th[ps + pa + 1], th[ps + pa + 2]) if use_sign else (0.0, 0.0)

    def f(th):
        v, w, c, (d, e) = unpack(th)
        u_raw = Xs @ v
        u = np.clip(u_raw, -3, 3)
        s0 = np.exp(u)
        a_raw = Xa @ w
        a = np.clip(a_raw, -20, 20)
        g = expit(a)
        cc = np.clip(c, -5, 5)
        k = 1 + np.exp(cc)
        t0 = z / s0 - e * sg
        t1 = z / (k * s0) - d * sg / k
        l0 = np.log1p(-g + 1e-15) - u - 0.5 * t0 ** 2
        l1 = np.log(g + 1e-300) - u - np.log(k) - 0.5 * t1 ** 2
        m = np.maximum(l0, l1)
        L = m + np.log(np.exp(l0 - m) + np.exp(l1 - m))
        r1 = np.exp(l1 - L)
        r0 = 1 - r1
        pen = lam * (np.sum(v[1:] ** 2) + np.sum(w[1:] ** 2)) + (lam * 0.01 * (d ** 2 + 10 * e ** 2) if use_sign else 0.0)
        nll = -L.sum() + 0.5 * len(z) * np.log(2 * np.pi) + pen
        du = r0 * (-1 + t0 * z / s0) + r1 * (-1 + t1 * z / (k * s0))
        du = du * (np.abs(u_raw) < 3)
        da = (r1 * (1 - g) - r0 * g) * (np.abs(a_raw) < 20)
        dk = r1 * (-1 / k + t1 ** 2 / k)
        gv = -(Xs.T @ du)
        gv[1:] += 2 * lam * v[1:]
        gw = -(Xa.T @ da)
        gw[1:] += 2 * lam * w[1:]
        gc = -dk.sum() * (k - 1) * (abs(c) < 5)
        grad = [gv, gw, [gc]]
        if use_sign:
            gd = -(r1 * t1 * sg / k).sum() + 2 * lam * 0.01 * d
            ge = -(r0 * t0 * sg).sum() + 2 * lam * 0.1 * e
            grad.append([gd, ge])
        return nll, np.concatenate([np.ravel(x) for x in grad])

    th0 = np.zeros(ps + pa + 1 + (2 if use_sign else 0))
    th0[ps] = -2.0
    th0[ps + pa] = 0.5
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return unpack(r.x)


def mix_q(s0, g, k, d, e, sgn, seed, n=4000):
    rng = np.random.default_rng(seed)
    real = rng.random(n) < g
    x = np.where(real, d * sgn * s0 + rng.normal(0, 1, n) * k * s0, e * sgn * s0 + rng.normal(0, 1, n) * s0)
    return H.quantiles_from_samples(x)


def design():
    """Events + ordinary control days with C1a, C1c and LLM-sign features (shared with C2)."""
    D = pickle.load(open(DIR / "c0_events.pkl", "rb"))
    FE = pickle.load(open(DIR / "c1a_features.pkl", "rb"))["X"]
    P = pd.read_pickle(DIR / "c1c_cftc.pkl")
    ev = {r["id"]: r for r in D["events"]}
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))}
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    R = B1.total_returns(["SPY", "USO", "GLD"], ss, "2003-01-01")
    C = controls(list(FE.columns))
    # C1c features for events and controls, as of the close before the window
    new = {}
    for i in FE.index:
        a = ss.searchsorted(pd.Timestamp(lib[i]["w1"]["start"])) + 1
        new[i] = positioning(P, ss[a - 1]) | premium(R, FE.loc[i].to_dict(), a, ss)
    for i in C.index:
        a = ss.get_loc(pd.Timestamp(i.replace("control-", "")))
        new[i] = positioning(P, ss[a - 1]) | premium(R, C.loc[i].to_dict(), a, ss)
    N = pd.DataFrame.from_dict(new, orient="index")
    c1c_cols = list(N.columns)
    print(f"C1c features: {len(c1c_cols)} (positioning {sum(c.startswith('pos_') for c in c1c_cols)}, premium "
          f"{sum(c.startswith('pr_') for c in c1c_cols)}); missing share on events {float(N.reindex(FE.index).isna().mean().mean()):.1%}", flush=True)
    E = FE.join(N)
    for j, fac in enumerate(live.FACTORS):
        E[f"z_{fac}"] = [ev[i]["factor_y"][j] / ev[i]["factor_noise"][j] if ev[i]["factor_noise"][j] > 0 else np.nan for i in E.index]
    sg = {f: [((lib[i].get("direction") or {}).get("factor_signs") or {}).get(f) or 0 for i in E.index] for f in SIGN_FACTORS}
    for f in SIGN_FACTORS:
        E[f"llm_sign_{f}"] = sg[f]
    E["split"] = [ev[i]["split"] for i in E.index]
    E["sev"] = [ev[i]["severity"] for i in E.index]
    E["date"] = [ev[i]["date"] for i in E.index]
    E["is_event"] = 1.0
    Cc = C.join(N).assign(split="train", sev="none", is_event=0.0, date=[i.replace("control-", "") for i in C.index])
    A = pd.concat([E, Cc], axis=0, sort=False)
    base_cols = list(FE.columns)
    for c in base_cols + c1c_cols + [f"llm_sign_{f}" for f in SIGN_FACTORS]:
        if c not in A:
            A[c] = 0.0
    fill0 = [c for c in base_cols if c.startswith(("sc_", "q_", "st_", "rg_"))] + [f"llm_sign_{f}" for f in SIGN_FACTORS]
    A[fill0] = A[fill0].fillna(0.0)
    tr = (A.split == "train").to_numpy()
    allc = base_cols + c1c_cols + ["is_event"] + [f"llm_abs_{f}" for f in SIGN_FACTORS]
    for f in SIGN_FACTORS:
        A[f"llm_abs_{f}"] = A[f"llm_sign_{f}"].abs()
    mu, sd = A.loc[tr, allc].mean(), A.loc[tr, allc].std().replace(0, 1)
    Zall = ((A[allc] - mu) / sd).fillna(0)
    sev_rank = A.sev.map({"minor": 0, "moderate": 1, "major": 2, "extreme": 3, "none": -1}).to_numpy()
    order = np.argsort(A.date.to_numpy()[tr], kind="stable")
    tri_all = np.where(tr)[0][order]
    return dict(D=D, FE=FE, ev=ev, lib=lib, ss=ss, C=C, N=N, c1c_cols=c1c_cols, E=E, A=A, base_cols=base_cols, tr=tr,
                allc=allc, Zall=Zall, sev_rank=sev_rank, tri_all=tri_all)


def step_score():
    g = design()
    D, FE, ev, lib, ss, C, N, c1c_cols, E, A, base_cols, tr, allc, Zall, sev_rank, tri_all = (g[k] for k in "D, FE, ev, lib, ss, C, N, c1c_cols, E, A, base_cols, tr, allc, Zall, sev_rank, tri_all".split(", "))
    res, rows = {}, []
    for fac in live.FACTORS:
        z = A[f"z_{fac}"].to_numpy()
        ok = np.isfinite(z)
        tri = tri_all[ok[tri_all]]
        fit_i, hold_i = tri[:int(0.8 * len(tri))], tri[int(0.8 * len(tri)):]
        has_sign = fac in SIGN_FACTORS
        sgn = A[f"llm_sign_{fac}"].to_numpy().astype(float) if has_sign else np.zeros(len(A))
        state_a = [c for c in base_cols if c.startswith(STATE_PREFIX)]
        variants = {
            "A gate": (state_a, base_cols + ["is_event"], False),
            "B +C1c": (state_a + c1c_cols, base_cols + c1c_cols + ["is_event"], False),
            "C +sign": (state_a + c1c_cols, base_cols + c1c_cols + ["is_event"] + ([f"llm_abs_{fac}"] if has_sign else []), has_sign),
            "D gate+sign": (state_a, base_cols + ["is_event"] + ([f"llm_abs_{fac}"] if has_sign else []), has_sign),
        }
        out = {}
        for vname, (scols, acols, use_sign) in variants.items():
            Xs = np.column_stack([np.ones(len(A)), Zall[scols].to_numpy()])
            small = ["is_event"] + [c for c in acols if c.startswith(("sc_sev_", "ms_", "fg_", "pos_", "pr_", "llm_abs_"))]
            best = None
            for gname, cols in (("severity+state", small), ("all features", acols)):
                Xa = np.column_stack([np.ones(len(A)), Zall[cols].to_numpy()])
                for lam in (3.0, 30.0, 300.0):
                    v, w, c, (d, e) = fit_dir(Xs[fit_i], Xa[fit_i], z[fit_i], sgn[fit_i], lam, use_sign)
                    s0 = np.exp(np.clip(Xs[hold_i] @ v, -3, 3))
                    g = expit(np.clip(Xa[hold_i] @ w, -20, 20))
                    k = 1 + np.exp(c)
                    ll = np.log((1 - g) * norm.pdf(z[hold_i], e * sgn[hold_i] * s0, s0) + g * norm.pdf(z[hold_i], d * sgn[hold_i] * s0, k * s0) + 1e-300).mean()
                    if best is None or ll > best[0]:
                        best = (ll, gname, cols, lam)
            _, gname, cols, lam = best
            Xa = np.column_stack([np.ones(len(A)), Zall[cols].to_numpy()])
            m = tr & ok
            v, w, c, (d, e) = fit_dir(Xs[m], Xa[m], z[m], sgn[m], lam, use_sign)
            k = 1 + np.exp(c)
            out[vname] = {"gate_features": gname, "lambda": lam, "k": float(k), "d_real_push": float(d), "e_every_event_push": float(e)}
            for sp in ("validation", "test"):
                mm = (A.split == sp).to_numpy() & ok
                s0 = np.exp(np.clip(Xs[mm] @ v, -3, 3))
                g = expit(np.clip(Xa[mm] @ w, -20, 20))
                big = np.abs(z[mm]) > 2
                for i, (zz, s, gg, sgi) in enumerate(zip(z[mm], s0, g, sgn[mm])):
                    q = mix_q(s, gg, k, d, e, sgi, seed=i)
                    rows.append({"split": sp, "event": A.index[mm][i], "target": fac, "rung": vname, "y": zz, "noise_sd": 1.0, "q": q,
                                 "severity": A.sev.iloc[np.where(mm)[0][i]], "sign": sgi})
                    if vname == "A gate":
                        rows.append({"split": sp, "event": A.index[mm][i], "target": fac, "rung": "0 noise", "y": zz, "noise_sd": 1.0,
                                     "q": H.quantiles_normal(0, 1), "severity": A.sev.iloc[np.where(mm)[0][i]], "sign": sgi})
                out[vname][sp] = {"auc_gate_for_big_move": auc(g, big), "auc_severity": auc(sev_rank[mm] + 1e-6 * np.arange(mm.sum()), big)}
        res[fac] = out
        print(f"{fac:24s} " + " | ".join(f"{k}: val AUC {v['validation']['auc_gate_for_big_move']:.2f} test {v['test']['auc_gate_for_big_move'] or float('nan'):.2f}"
                                         + (f" d={v['d_real_push']:+.2f} e={v['e_every_event_push']:+.2f}" if k == "C +sign" and fac in SIGN_FACTORS else "")
                                         for k, v in out.items()), flush=True)
    S = H.score_rows(pd.DataFrame(rows))
    S.to_pickle(DIR / "c1c_rungs.pkl")                  # with the 19 quantiles (the hedge test reads them)
    res["crps"] = {}
    for sp in ("validation", "test"):
        for sub, mask in (("all factors", lambda x: x), ("big events", lambda x: x[x.severity.isin(["major", "extreme"])]),
                          ("the 7 factors with an LLM sign", lambda x: x[x.target.isin(SIGN_FACTORS)])):
            d = mask(S[S.split == sp])
            by = {rg: H.summary(d[d.rung == rg]) for rg in sorted(d.rung.unique())}
            cmp_ = {"A vs 0": H.paired_better(d[d.rung == "A gate"], d[d.rung == "0 noise"]),
                    "B vs 0": H.paired_better(d[d.rung == "B +C1c"], d[d.rung == "0 noise"]),
                    "C vs 0": H.paired_better(d[d.rung == "C +sign"], d[d.rung == "0 noise"]),
                    "B vs A": H.paired_better(d[d.rung == "B +C1c"], d[d.rung == "A gate"]),
                    "C vs B": H.paired_better(d[d.rung == "C +sign"], d[d.rung == "B +C1c"]),
                    "D vs 0": H.paired_better(d[d.rung == "D gate+sign"], d[d.rung == "0 noise"]),
                    "D vs A": H.paired_better(d[d.rung == "D gate+sign"], d[d.rung == "A gate"])}
            # direction: on moves > 2x noise where the LLM gave a sign, does C's median point the right way?
            c = d[(d.rung == "C +sign") & (d.sign != 0) & (d.y.abs() > 2)]
            dd_ = d[(d.rung == "D gate+sign") & (d.sign != 0) & (d.y.abs() > 2)]
            raw = d[(d.rung == "0 noise") & (d.sign != 0) & (d.y.abs() > 2)]
            res["crps"][f"{sp} | {sub}"] = {"rungs": by, "comparisons": cmp_,
                                            "direction_hit_big_moves_C": float(np.mean(np.sign(c["median"]) == np.sign(c.y))) if len(c) else None,
                                            "direction_hit_big_moves_D": float(np.mean(np.sign(dd_["median"]) == np.sign(dd_.y))) if len(dd_) else None,
                                            "direction_hit_big_moves_llm_sign": float(np.mean(raw.sign == np.sign(raw.y))) if len(raw) else None,
                                            "n_big_signed": int(len(c))}
            print(f"{sp:10s} {sub:32s} " + " | ".join(f"{rg}: {x['crps_std']:.4f} ({x['coverage_90']:.0%})" for rg, x in by.items())
                  + " || " + "; ".join(f"{k} {v['b_minus_a']:+.4f} ({v['a_better_support']:.0%})" for k, v in cmp_.items())
                  + (f" || direction on big signed moves: C {res['crps'][f'{sp} | {sub}']['direction_hit_big_moves_C']:.0%}, D {res['crps'][f'{sp} | {sub}']['direction_hit_big_moves_D']:.0%}, LLM sign itself {res['crps'][f'{sp} | {sub}']['direction_hit_big_moves_llm_sign']:.0%} (n={len(c)})" if len(c) else ""), flush=True)
    (ROOT / "docs/pmrm/results/planc_c1c.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "score"])
    a = ap.parse_args()
    step_fetch() if a.step == "fetch" else step_score()
