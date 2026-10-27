"""Plan B checkpoint B1: the broad factor model (BUILD_PLAN B1, PLAN_B §2, §5.2; DECISIONS #155). No LLM calls.

  python scripts/planb_b1.py

Universe: the B0 ETF candidates. Those with history from 2011 form the factor panel; later listings and individual
stocks (current S&P 500 members via SPY's holdings, the Study B pools, the demo books) are expressed in the factor
space afterwards. Periods: train 2011-2018, validation 2019-2020 (choose k), test 2021 to the audit end.
Writes docs/pmrm/results/b1.json and cache/pmrm/b1/model.pkl (loadings, factor series, standardization) for B3.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pmrm_audit as A  # noqa: E402
from pmrm_b0_audit import UNIVERSE  # noqa: E402
from worsey.data import massive  # noqa: E402
from worsey.data.cache import cached_call  # noqa: E402
from worsey.planb import factors as FM  # noqa: E402

OUT = ROOT / "docs" / "pmrm" / "results"
MODEL = ROOT / "cache" / "pmrm" / "b1"
START, PANEL_FROM = date(2003, 9, 10), pd.Timestamp("2011-01-03")
TRAIN_END, VAL_END = pd.Timestamp("2018-12-31"), pd.Timestamp("2020-12-31")
ANCHOR_ETFS = {"SPY", "UUP", "USO", "GLD", "HYG", "IEF", "IWM", "VTV", "VUG"}


# ---------------------------------------------------------------- data

def total_returns(tickers, sessions: pd.DatetimeIndex, start: str) -> pd.DataFrame:
    """Close-to-close total returns on consecutive XNYS sessions (split-adjusted closes + cash dividends on ex-date)."""
    out = {}
    for t in tickers:
        try:
            px = massive.daily_bars(t, start, A.AUDIT_END.isoformat(), trim_to_listing=False)["close"]
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: {type(e).__name__}", flush=True)
            continue
        px = px[~px.index.duplicated()].reindex(sessions)
        div = pd.Series(0.0, index=sessions)
        try:
            for d in massive.dividends(t):
                ex = pd.Timestamp(d.get("ex_dividend_date") or "1900-01-01")
                if ex in div.index and d.get("dividend_type") in (None, "CD", "SC"):
                    div[ex] += float(d.get("cash_amount") or 0)
        except Exception:  # noqa: BLE001
            pass
        r = (px + div) / px.shift(1) - 1
        out[t] = r.where(px.notna() & px.shift(1).notna())
    return pd.DataFrame(out)


def fred_change(sid: str, sessions: pd.DatetimeIndex) -> pd.Series:
    def fetch():
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=45)
        r.raise_for_status()
        return {"text": r.text}
    txt = cached_call("web", f"fred:{sid}", None, fetch)["text"]
    rows = [l.split(",") for l in txt.splitlines()[1:]]
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in rows if v not in (".", "")})
    return s.reindex(sessions).ffill(limit=2).diff()


def anchors(R: pd.DataFrame, sessions) -> pd.DataFrame:
    X = pd.DataFrame(index=sessions)
    X["market"] = R["SPY"]
    X["rates_2y"] = fred_change("DGS2", sessions)
    X["rates_10y"] = fred_change("DGS10", sessions)
    X["dollar"] = R["UUP"]
    X["oil"] = R["USO"]
    X["gold"] = R["GLD"]
    X["credit"] = R["HYG"] - R["IEF"]
    X["size"] = R["IWM"] - R["SPY"]
    X["value"] = R["VTV"] - R["VUG"]
    return X


def stock_list() -> dict[str, str]:
    """ticker -> source group: current S&P 500 (survivorship caveat), Study B pools, demo books."""
    out = {}
    try:
        h = massive.etf_holdings("SPY")
        for t in h["ticker"]:
            if isinstance(t, str) and t.isalpha() and t.isupper() and len(t) <= 5:
                out[t] = "sp500_current"
    except Exception as e:  # noqa: BLE001
        print("  SPY holdings:", type(e).__name__, flush=True)
    for bucket, ts in A.STOCK_POOL.items():
        for t in ts:
            out[t] = f"pool:{bucket}"
    for f in (ROOT / "portfolios").glob("*.yaml"):
        for p in yaml.safe_load(f.read_text()).get("positions", []):
            out.setdefault(p["ticker"], f"book:{f.stem}")
    return out


# ---------------------------------------------------------------- main

def main():
    cal = A.sessions(START, A.AUDIT_END)
    sessions = pd.DatetimeIndex(pd.to_datetime(cal["day"]))
    etfs = sorted({t for ts in UNIVERSE.values() for t in ts})
    group = {t: g for g, ts in UNIVERSE.items() for t in ts}
    print(f"ETF returns for {len(etfs)} ...", flush=True)
    R = total_returns(etfs, sessions, "1990-01-01")       # same call as B0 (cached)
    X = anchors(R, sessions)
    first = R.apply(lambda s: s.first_valid_index())
    panel = [t for t in R.columns if first[t] is not None and first[t] <= PANEL_FROM and t not in ANCHOR_ETFS
             and R.loc[PANEL_FROM:, t].notna().mean() > 0.97]
    late = [t for t in R.columns if t not in panel and t not in ANCHOR_ETFS]
    tr = (R.index >= PANEL_FROM) & (R.index <= TRAIN_END)
    va = (R.index > TRAIN_END) & (R.index <= VAL_END)
    te = R.index > VAL_END
    fit = tr | va
    Rp = R[panel]
    Ztr, mu, sd = FM.standardize(Rp[tr])
    Zva, _, _ = FM.standardize(Rp[va], mu, sd)
    print(f"panel {len(panel)} ETFs, late {len(late)}; train {tr.sum()} val {va.sum()} test {te.sum()} days", flush=True)

    # k on validation (ties within 0.002 go to the smaller k)
    lat_val = {k: FM.latent_cv_r2(Ztr, Zva, k) for k in range(1, 16)}
    best = max(lat_val.values())
    k_lat = min(k for k, v in lat_val.items() if v >= best - 0.002)
    Xs_tr = (X[tr] - X[fit].mean()) / X[fit].std()
    Xs_va = (X[va] - X[fit].mean()) / X[fit].std()
    hyb_val = {k: FM.hybrid_cv_r2(Ztr, Zva, Xs_tr, Xs_va, k) for k in range(0, 9)}
    bh = max(hyb_val.values())
    k_hyb = min(k for k, v in hyb_val.items() if v >= bh - 0.002)
    print("latent val:", {k: round(v, 3) for k, v in lat_val.items()}, "-> k", k_lat, flush=True)
    print("hybrid val:", {k: round(v, 3) for k, v in hyb_val.items()}, "-> k2", k_hyb, flush=True)

    # refit on train+val, score on test
    Zfit, mu, sd = FM.standardize(Rp[fit])
    Zte, _, _ = FM.standardize(Rp[te], mu, sd)
    Xm, Xsd = X[fit].mean(), X[fit].std()
    Xf, Xt = (X[fit] - Xm) / Xsd, (X[te] - Xm) / Xsd
    B = FM.regress_loadings(Zfit, Xf)
    anch_te = FM.r2(Zte[B.index].to_numpy(), FM.anchored_predict(B, Xt.fillna(0), B.index))
    test = {"latent": FM.latent_cv_r2(Zfit, Zte, k_lat), "anchored": anch_te,
            "hybrid": FM.hybrid_cv_r2(Zfit, Zte, Xf, Xt.fillna(0), k_hyb)}
    test_curve = {k: FM.latent_cv_r2(Zfit, Zte, k) for k in (1, 2, 3, 5, 8, 10, 15)}
    print("test R2:", {k: round(v, 3) for k, v in test.items()}, flush=True)

    # by group (test, latent at k_lat and anchored)
    L, psi = FM.pca_loadings(Zfit, k_lat)
    by_group = {}
    cols = np.array(Zfit.columns)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cols))
    halves = [cols[perm[: len(cols) // 2]], cols[perm[len(cols) // 2:]]]
    idx = {c: i for i, c in enumerate(cols)}
    P = pd.DataFrame(np.nan, index=Zte.index, columns=cols)
    for a, b in ((0, 1), (1, 0)):
        F = FM.gls_factors(Zte[halves[a]].to_numpy(), L[[idx[c] for c in halves[a]]], psi[[idx[c] for c in halves[a]]])
        P[halves[b]] = F @ L[[idx[c] for c in halves[b]]].T
    PA = pd.DataFrame(FM.anchored_predict(B, Xt.fillna(0), B.index), index=Zte.index, columns=B.index)
    for g in UNIVERSE:
        cs = [c for c in cols if group[c] == g]
        if cs:
            by_group[g] = {"n": len(cs), "latent": FM.r2(Zte[cs].to_numpy(), P[cs].to_numpy()),
                           "anchored": FM.r2(Zte[cs].to_numpy(), PA[cs].to_numpy())}

    # stability: rolling 3-year windows vs the full fit (latent subspace; anchored loading correlation)
    wins = [("2011-2013", "2011-01-01", "2013-12-31"), ("2014-2016", "2014-01-01", "2016-12-31"),
            ("2017-2019", "2017-01-01", "2019-12-31"), ("2020-2022", "2020-01-01", "2022-12-31"),
            ("2023-2026", "2023-01-01", "2026-12-31")]
    stab = {}
    for name, a, b in wins:
        m = (R.index >= a) & (R.index <= b)
        Zw, _, _ = FM.standardize(Rp[m])
        Lw, _ = FM.pca_loadings(Zw, k_lat)
        Bw = FM.regress_loadings(Zw, (X[m] - Xm) / Xsd)
        common = B.index.intersection(Bw.index)
        stab[name] = {"latent_subspace_vs_full": FM.subspace_similarity(L, Lw),
                      "latent_top3_vs_full": FM.subspace_similarity(L[:, :3], Lw[:, :3]),
                      "anchored_loading_corr": {f: float(np.corrcoef(B.loc[common, f], Bw.loc[common, f])[0, 1]) for f in B.columns}}

    # 2003 subset: early fit vs 2011-2020 fit on the same assets; early loadings scored on the 2021+ test
    early = [t for t in panel if first[t] <= pd.Timestamp("2004-01-31") and R.loc["2003-10-01":"2010-12-31", t].notna().mean() > 0.97]
    me = (R.index >= "2003-10-01") & (R.index <= "2010-12-31")
    Ze, mue, sde = FM.standardize(R.loc[me, early])
    Zf2, mu2, sd2 = FM.standardize(R.loc[fit, early])
    Le, _ = FM.pca_loadings(Ze, k_lat)
    L2, _ = FM.pca_loadings(Zf2, k_lat)
    Zte2, _, _ = FM.standardize(R.loc[te, early], mue, sde)
    stab_2003 = {"assets": len(early), "subspace_2003_10_vs_2011_20": FM.subspace_similarity(Le, L2),
                 "test_r2_with_2003_10_loadings": FM.latent_cv_r2(Ze, Zte2, k_lat),
                 "test_r2_with_2011_20_loadings": FM.latent_cv_r2(Zf2, FM.standardize(R.loc[te, early], mu2, sd2)[0], k_lat)}

    # factor series (latent from the panel, by GLS each day) for the whole history, and the stock layer
    Zall, _, _ = FM.standardize(Rp, mu, sd)
    F = pd.DataFrame(FM.gls_factors(Zall.to_numpy(), L, psi), index=R.index, columns=[f"F{i + 1}" for i in range(k_lat)])
    stocks = stock_list()
    print(f"stock layer: {len(stocks)} names ...", flush=True)
    S = total_returns(sorted(set(stocks) | set(late)), sessions, "2010-12-01")
    layer = {}
    for t in S.columns:
        s = S[t]
        fm = fit & s.notna().to_numpy()
        tm = te & s.notna().to_numpy()
        if fm.sum() < 250 or tm.sum() < 120:
            continue
        zs = (s - s[fm].mean()) / s[fm].std()
        res = {"source": stocks.get(t, "late_etf:" + group.get(t, "")), "fit_days": int(fm.sum())}
        for name, Fac in (("latent", F), ("anchored", (X - Xm) / Xsd)):
            Af = Fac[fm].fillna(0).to_numpy()
            bta = np.linalg.lstsq(np.column_stack([np.ones(len(Af)), Af]), zs[fm].to_numpy(), rcond=None)[0]
            pred = np.column_stack([np.ones(tm.sum()), Fac[tm].fillna(0).to_numpy()]) @ bta
            y = zs[tm].to_numpy()
            res[f"{name}_test_r2"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            if name == "anchored":
                res["anchored_loadings"] = dict(zip(X.columns, map(float, bta[1:])))
        layer[t] = res
    lay = pd.DataFrame(layer).T
    for c in ("latent_test_r2", "anchored_test_r2"):
        lay[c] = lay[c].astype(float)
    lay["src"] = lay["source"].str.split(":").str[0]
    stock_summary = lay.groupby("source")[["latent_test_r2", "anchored_test_r2"]].median().round(3).to_dict("index")
    src_summary = lay.groupby("src")[["latent_test_r2", "anchored_test_r2"]].agg(["median", "count"]).round(3)
    src_summary.columns = [f"{a}_{b}" for a, b in src_summary.columns]

    MODEL.mkdir(parents=True, exist_ok=True)
    with open(MODEL / "model.pkl", "wb") as f:
        pickle.dump({"panel": panel, "mu": mu, "sd": sd, "k_latent": k_lat, "k_hybrid": k_hyb, "L": L, "psi": psi,
                     "factors_latent": F, "anchors": X, "anchor_mean": Xm, "anchor_sd": Xsd, "B_anchored": B,
                     "stock_layer": layer, "periods": {"train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date())}}, f)
    rep = {"panel_n": len(panel), "late_n": len(late), "panel": panel, "days": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "anchors": list(X.columns), "val_latent": lat_val, "val_hybrid": hyb_val, "k_latent": k_lat, "k_hybrid": k_hyb,
           "test_r2": test, "test_latent_curve": test_curve, "by_group_test": by_group, "stability": stab,
           "stability_2003": stab_2003, "stock_layer_by_source": stock_summary,
           "stock_layer_by_src": src_summary.to_dict("index"), "stock_layer_n": int(len(lay))}
    (OUT / "b1.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("k_latent", "k_hybrid", "test_r2", "stability_2003", "stock_layer_by_src")}, default=float, indent=0))
    print("wrote", OUT / "b1.json")


if __name__ == "__main__":
    main()
