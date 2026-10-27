"""B1 revision: extend the named factors with the four the latent model found and the
named set missed (Asia/EM, healthcare, agriculture, defensives vs producers) and compare every version on the SAME scored
ETFs (all anchor ETFs of either set excluded from scoring). Same periods and scoring as planb_b1.py. No LLM calls.

  python scripts/planb_b1_extended.py
Adds `extended` to docs/pmrm/results/b1.json and saves cache/pmrm/b1/model_ext.pkl (extended anchors) for B3.
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
from worsey.planb import factors as FM  # noqa: E402

EXTRA_ETFS = {"EEM", "XLV", "DBA", "XLU", "XLE"}


def main():
    m = pickle.load(open(ROOT / "cache/pmrm/b1/model.pkl", "rb"))
    sessions = m["factors_latent"].index
    X9 = m["anchors"]
    E = B1.total_returns(sorted(EXTRA_ETFS), sessions, "1990-01-01")
    X13 = X9.copy()
    X13["asia_em"] = E["EEM"] - B1.total_returns(["SPY"], sessions, "1990-01-01")["SPY"]
    X13["healthcare"] = E["XLV"]
    X13["agriculture"] = E["DBA"]
    X13["defensive_vs_producers"] = E["XLU"] - E["XLE"]
    R = B1.total_returns(m["panel"], sessions, "1990-01-01")
    panel = [t for t in m["panel"] if t not in EXTRA_ETFS]        # common scored set for every version
    R = R[panel]
    tr = (sessions >= B1.PANEL_FROM) & (sessions <= B1.TRAIN_END)
    va = (sessions > B1.TRAIN_END) & (sessions <= B1.VAL_END)
    te = sessions > B1.VAL_END
    fit = tr | va
    Ztr, mu, sd = FM.standardize(R[tr])
    Zva, _, _ = FM.standardize(R[va], mu, sd)
    Zfit, muf, sdf = FM.standardize(R[fit])
    Zte, _, _ = FM.standardize(R[te], muf, sdf)
    out = {"scored_etfs": len(panel), "versions": {}}
    for name, X in (("named 9", X9), ("named 13 (extended)", X13)):
        Xm, Xs = X[fit].mean(), X[fit].std()
        Xtr, Xva = (X[tr] - Xm) / Xs, (X[va] - Xm) / Xs
        Xf, Xt = (X[fit] - Xm) / Xs, ((X[te] - Xm) / Xs).fillna(0)
        B = FM.regress_loadings(Zfit, Xf)
        anch = FM.r2(Zte[B.index].to_numpy(), FM.anchored_predict(B, Xt, B.index))
        val = {k: FM.hybrid_cv_r2(Ztr, Zva, Xtr, Xva.fillna(0), k) for k in range(0, 9)}
        best = max(val.values())
        k2 = min(k for k, v in val.items() if v >= best - 0.002)
        hyb = FM.hybrid_cv_r2(Zfit, Zte, Xf, Xt, k2)
        # per-factor share: drop one named factor at a time (test R² lost)
        drop = {}
        for c in X.columns:
            Bd = FM.regress_loadings(Zfit, Xf.drop(columns=c))
            drop[c] = anch - FM.r2(Zte[Bd.index].to_numpy(), FM.anchored_predict(Bd, Xt.drop(columns=c), Bd.index))
        out["versions"][name] = {"anchored_test_r2": anch, "hybrid_k2": k2, "hybrid_test_r2": hyb,
                                 "hybrid_val": val, "test_r2_lost_without_each_factor": drop}
        print(f"{name}: anchored {anch:.3f}, hybrid (+{k2} latent) {hyb:.3f}; lost without: "
              + ", ".join(f"{c} {v:.3f}" for c, v in sorted(drop.items(), key=lambda x: -x[1])), flush=True)
    lat = FM.latent_cv_r2(Zfit, Zte, m["k_latent"])
    out["versions"]["latent (9)"] = {"test_r2": lat}
    print(f"latent 9: {lat:.3f}")
    with open(ROOT / "cache/pmrm/b1/model_ext.pkl", "wb") as f:
        pickle.dump({"anchors": X13, "anchor_mean": X13[fit].mean(), "anchor_sd": X13[fit].std()}, f)
    p = ROOT / "docs/pmrm/results/b1.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["extended"] = out
    p.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
