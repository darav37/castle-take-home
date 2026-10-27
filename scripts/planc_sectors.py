"""Sector layer, step 1: do sector factors on top of the 13 macro factors explain more of stocks' and ETFs' daily moves
on days the model never saw? (DECISIONS #215). No LLM calls.

  python scripts/planc_sectors.py

Sector factor = a sector ETF's daily return beyond the 13 macro factors (its loadings on them fitted on the fit window),
in units of its own residual sd. Held-out design (as B1): loadings of every name on [macro] and on [macro + sectors]
fitted on 2019-2020 (504 sessions), then frozen; R2 of each name's daily return on 2021-01 .. 2026-10. Names: the
current S&P 500 (cached stock layer) and the 96 research ETFs. A stock that dominates its sector ETF (NVDA in SMH)
inflates its own gain mechanically; the median over ~500 stocks is robust to a few of those.
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
from stock_s0 import universe  # noqa: E402
from worsey.data import massive  # noqa: E402
from worsey.planb import live  # noqa: E402

SECTORS = {"semis": "SMH", "software": "IGV", "internet": "FDN", "regional banks": "KRE", "insurers": "KIE", "biotech": "XBI",
           "medtech": "IHI", "energy stocks": "XLE", "industrials": "XLI", "defense": "ITA", "homebuilders": "ITB",
           "transports": "IYT", "consumer discretionary": "XLY", "retail": "XRT", "staples": "XLP", "utilities": "XLU",
           "real estate": "IYR", "materials": "XLB", "metals & mining": "XME"}
FIT = ("2019-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-10-16")


def ols(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() < 250:
        return None
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    return np.linalg.lstsq(A, y[ok], rcond=None)[0]


def r2(X, y, c):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() < 250:
        return np.nan
    e = y[ok] - np.column_stack([np.ones(ok.sum()), X[ok]]) @ c
    return float(1 - (e ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


def sector_factors(R: pd.DataFrame, Fz: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each sector ETF's return beyond the macro factors (loadings from the fit window), / its residual sd there."""
    fit = (R.index >= FIT[0]) & (R.index <= FIT[1])
    out, meta = {}, {}
    for name, t in SECTORS.items():
        c = ols(Fz[fit].to_numpy(), R.loc[fit, t].to_numpy())
        res = R[t].to_numpy() - np.column_stack([np.ones(len(R)), Fz.to_numpy()]) @ c
        sd = float(np.nanstd(res[fit]))
        out[name] = res / sd
        meta[name] = {"etf": t, "macro_r2_fit": r2(Fz[fit].to_numpy(), R.loc[fit, t].to_numpy(), c), "residual_daily_sd": sd}
    return pd.DataFrame(out, index=R.index), meta


def main():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    etfs = list(pickle.load(open(ROOT / "cache/pmrm/b1/model.pkl", "rb"))["mu"].index)
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    RE = B1.total_returns(sorted(set(etfs) | set(proxy_t) | set(SECTORS.values())), ss, "1990-01-01")
    Fz = live.factor_returns(RE).reindex(columns=live.FACTORS)
    S, meta = sector_factors(RE, Fz)
    stocks = universe()
    RS = B1.total_returns(stocks, ss, "2010-12-01")
    fit = (ss >= FIT[0]) & (ss <= FIT[1])
    test = (ss >= TEST[0]) & (ss <= TEST[1])
    M, MS = Fz.to_numpy(), np.column_stack([Fz.to_numpy(), S.to_numpy()])
    rows = []
    sector_of = {}
    try:
        for t in stocks:
            sector_of[t] = massive.sector(t)[0]
    except Exception:  # noqa: BLE001
        pass
    for kind, R_ in (("stock", RS), ("etf", RE[[t for t in etfs if t in RE]])):
        for t in R_.columns:
            y = pd.to_numeric(R_[t], errors="coerce").to_numpy(float)
            c1, c2 = ols(M[fit], y[fit]), ols(MS[fit], y[fit])
            if c1 is None or c2 is None:
                continue
            a, b = r2(M[test], y[test], c1), r2(MS[test], y[test], c2)
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            # parsimonious version: macro + the ONE sector factor that helps most on the fit window (if it adds > 2 points)
            Sx = S.to_numpy()
            best, best_r2 = None, r2(M[fit], y[fit], c1)
            for k in range(Sx.shape[1]):
                Xk = np.column_stack([M, Sx[:, k]])
                ck = ols(Xk[fit], y[fit])
                if ck is not None and (rk := r2(Xk[fit], y[fit], ck)) > best_r2 + 0.02:
                    best, best_r2 = k, rk
            if best is None:
                c_ = c1
                b1 = a
            else:
                X1 = np.column_stack([M, Sx[:, best]])
                c_ = ols(X1[fit], y[fit])
                b1 = r2(X1[test], y[test], c_)
            own = [n for n, e in SECTORS.items() if e == t]
            top = int(np.argmax(np.abs(c2[1 + len(live.FACTORS):]))) if len(c2) > 1 + len(live.FACTORS) else None
            rows.append({"kind": kind, "name": t, "sector": sector_of.get(t) if kind == "stock" else None, "is_sector_etf": bool(own),
                         "r2_macro": a, "r2_macro_sectors": b, "gain": b - a, "r2_macro_best_sector": b1, "gain_best": b1 - a,
                         "best_sector": list(SECTORS)[best] if best is not None else None,
                         "top_sector": list(SECTORS)[top] if top is not None else None})
    D = pd.DataFrame(rows)
    res = {"fit": FIT, "test": TEST, "sector_factors": meta}
    for kind in ("stock", "etf"):
        d = D[(D.kind == kind) & (~D.is_sector_etf)]
        res[kind] = {"names": int(len(d)), "median_r2_macro": float(d.r2_macro.median()), "median_r2_macro_sectors": float(d.r2_macro_sectors.median()),
                     "median_gain": float(d.gain.median()), "share_improved": float((d.gain > 0).mean()),
                     "p25_gain": float(d.gain.quantile(0.25)), "p75_gain": float(d.gain.quantile(0.75)),
                     "best_one_sector": {"median_r2": float(d.r2_macro_best_sector.median()), "median_gain": float(d.gain_best.median()),
                                         "share_improved": float((d.gain_best > 0.001).mean()), "share_worse": float((d.gain_best < -0.001).mean()),
                                         "share_with_a_sector": float(d.best_sector.notna().mean())}}
        print(f"{kind:5s} ({len(d)} names, held out 2021-2026): median R2 macro {res[kind]['median_r2_macro']:.1%} -> macro+sectors "
              f"{res[kind]['median_r2_macro_sectors']:.1%}; median gain {res[kind]['median_gain']:+.1%}, improved {res[kind]['share_improved']:.0%} | best ONE sector: median R2 {res[kind]['best_one_sector']['median_r2']:.1%} "
              f"(gain {res[kind]['best_one_sector']['median_gain']:+.1%}; better {res[kind]['best_one_sector']['share_improved']:.0%}, worse {res[kind]['best_one_sector']['share_worse']:.0%}, "
              f"{res[kind]['best_one_sector']['share_with_a_sector']:.0%} get a sector)", flush=True)
    st = D[(D.kind == "stock") & D.sector.notna()]
    res["stock_by_sector"] = st.groupby("sector").agg(n=("gain", "size"), r2_macro=("r2_macro", "median"),
                                                       r2_macro_sectors=("r2_macro_sectors", "median"), gain=("gain", "median"),
                                                       r2_best_one=("r2_macro_best_sector", "median"), gain_best_one=("gain_best", "median")).round(3).to_dict(orient="index")
    print(pd.DataFrame(res["stock_by_sector"]).T.sort_values("gain", ascending=False).to_string())
    ex = D[D.name.isin(["NVDA", "AMD", "JPM", "KEY", "PFE", "MRNA", "XOM", "HD", "BA", "LMT", "SMH", "KBE", "XBI", "GDX"])]
    print(ex[["name", "kind", "r2_macro", "r2_macro_sectors", "r2_macro_best_sector", "best_sector"]].round(3).to_string(index=False))
    D.to_pickle(ROOT / "cache/pmrm/planc/sector_r2_rows.pkl")
    pickle.dump({"S": S, "meta": meta}, open(ROOT / "cache/pmrm/planc/sector_factors.pkl", "wb"))
    (ROOT / "docs/pmrm/results/planc_sectors.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
