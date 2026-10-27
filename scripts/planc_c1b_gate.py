"""Plan C, C1b part 2: the real-event gate (PLAN_C §2b;; DECISIONS #192). No LLM calls.

  python scripts/planc_c1b_gate.py

Per factor, the event-window move z (in units of its normal noise) is modelled as a two-part mixture:
    z ~ (1 - g) * Normal(0, s0^2) + g * Normal(0, (k * s0)^2)
    s0 = exp(v . x_state)        normal noise, which can depend on market state (vol, fear gauges, calendar)
    g  = sigmoid(w . x_all)      the probability this is a real event for this factor (event schema + state + size + story)
    k  = 1 + exp(c)              how much bigger a real event's move is
Controls: ordinary sessions before 2023 (worsey.planc.controls: ~770 since #193; the first run had 38), with the same state
features and no event features, so "bigger than an ordinary day" is learned, not assumed.
Fitted on train (< 2023) by maximum likelihood with L2; scored on validation (2023-25) and test (2026) with the frozen
harness (standardized CRPS from 19 quantiles) against rung 0 (Normal(0, 1)), and gate quality: does g rank the events
that really moved (|z| > 2) above those that didn't (AUC), vs severity alone.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey.planb import library as LB  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.planc import harness as H  # noqa: E402
from worsey.planc.controls import control_sessions  # noqa: E402

DIR = ROOT / "cache/pmrm/planc"
STATE_PREFIX = ("ms_", "fg_", "cal_weekday", "cal_month_end", "cal_opex_week", "cal_holiday_adjacent", "cal_days_to_fomc")


def auc(score, label):
    from scipy.stats import rankdata
    r = rankdata(score)
    label = np.asarray(label, bool)
    n1 = int(label.sum())
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def controls(F_cols: list[str], n_target: int = 3000) -> pd.DataFrame:
    """Random ordinary sessions before 2023 with the same state features (as planc_c1a computes them) and their z."""
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    X = ((ext["anchors"] - ext["anchor_mean"]) / ext["anchor_sd"])[live.FACTORS]
    ss = X.index
    Xa = X.to_numpy()
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    fred = {k: LB.fred_daily(s).sort_index() for k, s in
            {"vix": "VIXCLS", "vix3m": "VXVCLS", "ovx": "OVXCLS", "gvz": "GVZCLS", "real10": "DFII10", "be10": "T10YIE",
             "curve": "T10Y2Y", "baa": "BAA10Y"}.items()}
    fomc = sorted(pd.Timestamp(r["date"]) for r in lib if r.get("kind") == "scheduled" and r.get("release") == "fomc")
    # the first run (#192) excluded 3 sessions around every library event, which left 38 days; corrected in #193
    pick = control_sessions(ss, lib, n_target)
    rows = []
    for a in pick:
        n = 2
        f = {"id": f"control-{ss[a].date()}"}
        sd2y = np.nanstd(Xa[a - 505:a - 1], axis=0)
        for j, fac in enumerate(live.FACTORS):
            for k in (1, 5, 20):
                f[f"ms_{fac}_r{k}"] = float(np.nansum(Xa[a - k:a, j]) / (sd2y[j] * np.sqrt(k))) if sd2y[j] > 0 else 0.0
            f[f"ms_{fac}_vol20"] = float(np.nanstd(Xa[a - 20:a, j]) / sd2y[j]) if sd2y[j] > 0 else 1.0
            f[f"z_{fac}"] = float(np.nansum(Xa[a:a + n, j]) / (sd2y[j] * np.sqrt(n))) if sd2y[j] > 0 else np.nan
        for fac in ("market", "oil", "gold"):
            j = live.FACTORS.index(fac)
            path = np.nancumsum(Xa[a - 60:a, j])
            f[f"ms_{fac}_dd60"] = float((path[-1] - path.max()) / (sd2y[j] * np.sqrt(60))) if sd2y[j] > 0 else 0.0
        day = ss[a - 1]
        for k, s in fred.items():
            v = s[s.index <= day]
            v5 = s[s.index <= day - pd.Timedelta(days=7)]
            f[f"fg_{k}"] = float(v.iloc[-1]) if len(v) else np.nan
            f[f"fg_{k}_d5"] = float(v.iloc[-1] - v5.iloc[-1]) if len(v) and len(v5) else 0.0
        f["fg_vix_term"] = f["fg_vix3m"] / f["fg_vix"] if f.get("fg_vix") and np.isfinite(f["fg_vix"]) and np.isfinite(f["fg_vix3m"]) else np.nan
        d0 = ss[a]
        f["cal_weekday"] = float(d0.weekday())
        nxt = list(ss[a - 1:a + 6])
        f["cal_month_end"] = float(nxt[0].month != nxt[3].month)
        third_fri = pd.Timestamp(d0.year, d0.month, 15) + pd.offsets.Week(weekday=4)
        f["cal_opex_week"] = float(abs((d0 - third_fri).days) <= 4)
        f["cal_holiday_adjacent"] = float((ss[a] - ss[a - 1]).days > 3)
        nf = [x for x in fomc if x >= d0]
        f["cal_days_to_fomc"] = float(min((nf[0] - d0).days, 60)) if nf else 60.0
        rows.append(f)
    C = pd.DataFrame(rows).set_index("id")
    return C


def fit_factor(Xs, Xa, z, lam=3.0):
    """Maximum likelihood for the two-part mixture; Xs = state design (with intercept), Xa = all features (with intercept)."""
    ps, pa = Xs.shape[1], Xa.shape[1]

    def unpack(th):
        return th[:ps], th[ps:ps + pa], th[-1]

    def nll(th):
        v, w, c = unpack(th)
        s0 = np.exp(np.clip(Xs @ v, -3, 3))
        g = expit(np.clip(Xa @ w, -20, 20))
        k = 1 + np.exp(np.clip(c, -5, 5))
        l = (1 - g) * norm.pdf(z, 0, s0) + g * norm.pdf(z, 0, k * s0)
        return -np.log(l + 1e-300).sum() + lam * (np.sum(v[1:] ** 2) + np.sum(w[1:] ** 2))

    th0 = np.zeros(ps + pa + 1)
    th0[ps] = -2.0                     # start with few real events
    th0[-1] = 0.5
    r = minimize(nll, th0, method="L-BFGS-B", options={"maxiter": 400})
    return unpack(r.x)


def mixture_quantiles(s0, g, k, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.random(n) < g
    x = rng.normal(0, 1, n) * s0 * np.where(real, k, 1.0)
    return H.quantiles_from_samples(x)


def main():
    D = pickle.load(open(DIR / "c0_events.pkl", "rb"))
    FE = pickle.load(open(DIR / "c1a_features.pkl", "rb"))["X"]
    ev = {r["id"]: r for r in D["events"]}
    C = controls(list(FE.columns))
    state_cols = [c for c in FE.columns if c.startswith(STATE_PREFIX)]
    all_cols = list(FE.columns)
    E = FE.copy()
    for j, fac in enumerate(live.FACTORS):
        E[f"z_{fac}"] = [ev[i]["factor_y"][j] / ev[i]["factor_noise"][j] if ev[i]["factor_noise"][j] > 0 else np.nan for i in E.index]
    E["split"] = [ev[i]["split"] for i in E.index]
    E["sev"] = [ev[i]["severity"] for i in E.index]
    E["is_event"] = 1.0
    C = C.reindex(columns=[c for c in C.columns]).assign(split="train", sev="none", is_event=0.0)
    A = pd.concat([E, C], axis=0, sort=False)
    for c in all_cols:
        if c not in A:
            A[c] = 0.0
    A[[c for c in all_cols if c.startswith(("sc_", "q_", "st_", "rg_"))]] = A[[c for c in all_cols if c.startswith(("sc_", "q_", "st_", "rg_"))]].fillna(0.0)
    tr = (A.split == "train").to_numpy()
    cols_a = all_cols + ["is_event"]
    mu, sd = A.loc[tr, cols_a].mean(), A.loc[tr, cols_a].std().replace(0, 1)
    Za = ((A[cols_a] - mu) / sd).fillna(0).to_numpy()
    Zs = Za[:, [cols_a.index(c) for c in state_cols]]
    Za1, Zs1 = np.column_stack([np.ones(len(A)), Za]), np.column_stack([np.ones(len(A)), Zs])
    res, rows = {}, []
    small = ["is_event"] + [c for c in all_cols if c.startswith(("sc_sev_", "ms_", "fg_"))]
    small_idx = [0] + [1 + cols_a.index(c) for c in small]
    sev_rank = A.sev.map({"minor": 0, "moderate": 1, "major": 2, "extreme": 3, "none": -1}).to_numpy()
    for fac in live.FACTORS:
        z = A[f"z_{fac}"].to_numpy()
        ok = np.isfinite(z)
        m = tr & ok
        # choose the gate's feature set and penalty on the last 20% of TRAIN only (never on validation or test)
        tri = np.where(m)[0]
        order = np.argsort([str(i).replace("control-", "") if str(i).startswith("control-") else ev[i]["date"] for i in A.index[tri]])
        fit_i, hold_i = tri[order[:int(0.8 * len(order))]], tri[order[int(0.8 * len(order)):]]
        best = None
        for name, cols_idx in (("severity+state", small_idx), ("all features", list(range(Za1.shape[1])))):
            for lam in (3.0, 30.0, 300.0):
                vv, ww, cc = fit_factor(Zs1[fit_i], Za1[fit_i][:, cols_idx], z[fit_i], lam)
                s0h = np.exp(np.clip(Zs1[hold_i] @ vv, -3, 3))
                gh = expit(np.clip(Za1[hold_i][:, cols_idx] @ ww, -20, 20))
                kh = 1 + np.exp(cc)
                ll = np.log((1 - gh) * norm.pdf(z[hold_i], 0, s0h) + gh * norm.pdf(z[hold_i], 0, kh * s0h) + 1e-300).mean()
                if best is None or ll > best[0]:
                    best = (ll, name, cols_idx, lam)
        _, gname, cols_idx, lam = best
        v, w, c = fit_factor(Zs1[m], Za1[m][:, cols_idx], z[m], lam)
        k = 1 + np.exp(c)
        out = {"k_real_vs_normal": float(k), "gate_features": gname, "lambda": lam}
        for sp in ("validation", "test"):
            mm = (A.split == sp).to_numpy() & ok
            s0 = np.exp(np.clip(Zs1[mm] @ v, -3, 3))
            g = expit(np.clip(Za1[mm][:, cols_idx] @ w, -20, 20))
            big = np.abs(z[mm]) > 2
            for i, (zz, s, gg) in enumerate(zip(z[mm], s0, g)):
                rows.append({"split": sp, "event": A.index[mm][i], "target": fac, "y": zz, "noise_sd": 1.0,
                             "q_gate": mixture_quantiles(s, gg, k, seed=i), "q_zero": H.quantiles_normal(0, 1)})
            out[sp] = {"auc_gate_for_big_move": auc(g, big), "auc_severity_for_big_move": auc(sev_rank[mm] + 1e-6 * np.arange(mm.sum()), big),
                       "mean_g": float(g.mean()), "share_big": float(big.mean()),
                       "precision_top_decile": float(big[g >= np.quantile(g, 0.9)].mean()) if mm.sum() >= 10 else None}
        res[fac] = out
        f2 = lambda x, pct=False: "n/a" if x is None or not np.isfinite(x) else (f"{x:.0%}" if pct else f"{x:.2f}")
        print(f"{fac:24s} [{gname}, lam {lam:g}] k={k:.2f} | val AUC gate {f2(out['validation']['auc_gate_for_big_move'])} vs severity "
              f"{f2(out['validation']['auc_severity_for_big_move'])}, top-decile precision {f2(out['validation']['precision_top_decile'], True)} "
              f"(base {f2(out['validation']['share_big'], True)}) | test AUC {f2(out['test']['auc_gate_for_big_move'])} vs sev "
              f"{f2(out['test']['auc_severity_for_big_move'])}", flush=True)
    R = pd.DataFrame(rows)
    for sp in ("validation", "test"):
        d = R[R.split == sp]
        gate = H.score_rows(d.assign(q=d.q_gate).assign(rung="gate"))
        zero = H.score_rows(d.assign(q=d.q_zero).assign(rung="zero"))
        cmp_ = H.paired_better(gate, zero)
        res[f"crps_{sp}"] = {"gate": H.summary(gate), "zero": H.summary(zero), "gate_vs_zero": cmp_}
        print(f"{sp}: CRPS gate {H.summary(gate)['crps_std']:.4f} (cov90 {H.summary(gate)['coverage_90']:.0%}) vs zero "
              f"{H.summary(zero)['crps_std']:.4f} (cov90 {H.summary(zero)['coverage_90']:.0%}) | gate better by {cmp_['b_minus_a']:+.4f} "
              f"(90% {cmp_['range_90'][0]:+.4f} to {cmp_['range_90'][1]:+.4f}, support {cmp_['a_better_support']:.0%})")
    (ROOT / "docs/pmrm/results/planc_c1b_gate.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
