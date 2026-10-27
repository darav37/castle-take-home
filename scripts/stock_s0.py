"""Single-stock track, S0: the corporate event set and its audit, earnings first (docs/pmrm/SINGLE_STOCK_TRACK.md;
; DECISIONS #202). No LLM calls.

  python scripts/stock_s0.py build     # Benzinga earnings for the S&P 500 + demo names, with each stock's reaction
  python scripts/stock_s0.py audit     # coverage, size of the reaction vs normal noise, does the surprise call direction,
                                       # does the move drift on or give back; headline volume per ticker

Reaction (code; the C0 residual definition): the stock's move beyond the 13 factors (betas on the live proxies over the
252 sessions before), over
  day 1     the first session that can react: after-close reports -> the next session; before-open and in-hours or
            unknown-time reports -> the report day (close before -> close)
  next 5    the 5 sessions after day 1 (drift or give-back)
each in units of the stock's own normal noise (residual sd x sqrt(sessions)). Splits as everywhere: train < 2023,
validation 2023-25, test 2026. Writes cache/pmrm/stock/earnings.pkl and docs/pmrm/results/stock_s0.json.
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

import planb_b1 as B1  # noqa: E402
from worsey.data import massive  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.planc import harness as H  # noqa: E402

DIR = ROOT / "cache/pmrm/stock"
OUT = ROOT / "docs/pmrm/results"
LOOK = 252


def universe() -> list[str]:
    names = set()
    h = massive.etf_holdings("SPY")
    for t in h["ticker"]:
        if isinstance(t, str) and t.isalpha() and t.isupper() and len(t) <= 5:
            names.add(t)
    names |= {"NVDA", "MRNA", "AAPL", "MSFT", "TSLA", "AMD", "META"}
    return sorted(names)


def step_build():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    names = universe()
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    Rp = B1.total_returns(proxy_t, ss, "1990-01-01")
    Fz = live.factor_returns(Rp).reindex(columns=live.FACTORS).to_numpy()
    S = B1.total_returns(names, ss, "2010-12-01")
    rows = []
    for k, t in enumerate(names):
        if t not in S:
            continue
        try:
            er = massive._get("/benzinga/v1/earnings", {"ticker": t, "limit": 100, "sort": "date.desc"}).get("results") or []
        except Exception:  # noqa: BLE001
            continue
        r = S[t].to_numpy()
        for e in er:
            if e.get("actual_eps") is None or not e.get("date"):
                continue
            d = pd.Timestamp(e["date"])
            hhmm = str(e.get("time") or "")
            i = ss.searchsorted(d)
            if i >= len(ss) - 6 or i < LOOK + 2:
                continue
            after = hhmm >= "16:00"
            a = i + 1 if (after and ss[i] == d) else i          # day 1 = the first session that can react
            if a + 6 > len(ss):
                continue
            lb = slice(a - 1 - LOOK, a - 1)
            y, Fl = r[lb], Fz[lb]
            ok = np.isfinite(y) & np.isfinite(Fl).all(1)
            if ok.sum() < 150 or not np.isfinite(r[a:a + 6]).all():
                continue
            A = np.column_stack([np.ones(ok.sum()), Fl[ok]])
            c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
            sd = float((y[ok] - A @ c).std())
            beta = c[1:]
            res = r[a:a + 6] - np.nan_to_num(Fz[a:a + 6]) @ beta
            est, act = e.get("estimated_eps"), e.get("actual_eps")
            rows.append({"ticker": t, "date": str(d.date()), "time": hhmm or None, "timing": "after close" if after else
                         ("before open" if hhmm and hhmm < "09:30" else "in hours or unknown"),
                         "split": H.split_of(str(d.date())), "eps_est": est, "eps_act": act,
                         "eps_surprise_pct": e.get("eps_surprise_percent"), "rev_surprise_pct": e.get("revenue_surprise_percent"),
                         "importance": e.get("importance"), "day1_z": float(res[0] / sd), "next5_z": float(res[1:6].sum() / (sd * np.sqrt(5))),
                         "day1_total": float(r[a]), "noise_day": sd})
        if k % 100 == 0:
            print(f"  {k}/{len(names)} tickers, {len(rows)} reports", flush=True)
    E = pd.DataFrame(rows)
    DIR.mkdir(parents=True, exist_ok=True)
    E.to_pickle(DIR / "earnings.pkl")
    print(f"earnings reports: {len(E)} across {E.ticker.nunique()} tickers, {E.date.min()} to {E.date.max()}")


def step_audit():
    E = pd.read_pickle(DIR / "earnings.pkl")
    E["year"] = E.date.str[:4]
    res = {"reports": int(len(E)), "tickers": int(E.ticker.nunique()), "first": E.date.min(), "last": E.date.max(),
           "by_year": E.groupby("year").size().to_dict(), "timing": E.timing.value_counts().to_dict(),
           "share_with_eps_estimate": float(E.eps_est.notna().mean()), "share_with_revenue_surprise": float(E.rev_surprise_pct.notna().mean()),
           "by_split": E.split.value_counts().to_dict()}
    z = E.day1_z
    res["size"] = {"rms_day1_vs_normal_day": float(np.sqrt((z ** 2).mean())), "median_abs_day1": float(z.abs().median()),
                   "share_over_2x": float((z.abs() > 2).mean()), "share_over_3x": float((z.abs() > 3).mean()),
                   "share_over_5x": float((z.abs() > 5).mean()), "normal_day_share_over_2x": 0.046,
                   "rms_next5": float(np.sqrt((E.next5_z ** 2).mean()))}
    # does the surprise call direction?
    out = {}
    for col in ("eps_surprise_pct", "rev_surprise_pct"):
        d = E[E[col].notna() & (E[col] != 0)]
        out[col] = {}
        for sp in ("train", "validation", "test"):
            x = d[d.split == sp]
            big = x[x.day1_z.abs() > 2]
            out[col][sp] = {"n": int(len(x)), "direction_hit_all": float((np.sign(x[col]) == np.sign(x.day1_z)).mean()) if len(x) else None,
                            "direction_hit_moves_over_2x": float((np.sign(big[col]) == np.sign(big.day1_z)).mean()) if len(big) else None,
                            "rank_corr": float(x[[col, "day1_z"]].corr(method="spearman").iloc[0, 1]) if len(x) > 10 else None}
    both = E[E.eps_surprise_pct.notna() & E.rev_surprise_pct.notna()]
    agree = both[np.sign(both.eps_surprise_pct) == np.sign(both.rev_surprise_pct)]
    out["eps_and_revenue_agree"] = {sp: {"n": int((agree.split == sp).sum()),
                                         "direction_hit": float((np.sign(agree[agree.split == sp].eps_surprise_pct) == np.sign(agree[agree.split == sp].day1_z)).mean())}
                                    for sp in ("train", "validation", "test")}
    res["surprise_direction"] = out
    # size by surprise decile (|eps surprise|)
    d = E[E.eps_surprise_pct.notna()].copy()
    d["dec"] = pd.qcut(d.eps_surprise_pct.abs().rank(method="first"), 5, labels=False)
    res["abs_move_by_abs_surprise_quintile"] = d.groupby("dec").day1_z.apply(lambda s: float(s.abs().mean())).to_dict()
    # drift or give-back after day 1
    d = E.copy()
    sg = np.sign(d.day1_z)
    res["after_day1"] = {"next5_mean_in_day1_direction": float((sg * d.next5_z).mean()),
                         "big_moves_next5_mean_in_direction": float((sg * d.next5_z)[d.day1_z.abs() > 3].mean()),
                         "share_big_moves_continuing": float(((sg * d.next5_z) > 0)[d.day1_z.abs() > 3].mean()),
                         "by_split_big": {sp: float((sg * d.next5_z)[(d.day1_z.abs() > 3) & (d.split == sp)].mean()) for sp in ("train", "validation", "test")}}
    # headline volume (Benzinga news, one page per ticker, a sample)
    sample = ["NVDA", "MRNA", "AAPL", "JPM", "XOM", "PFE", "TSLA", "WMT", "BA", "LLY"]
    vol = {}
    for t in sample:
        try:
            g = massive._get("/benzinga/v2/news", {"tickers": t, "limit": 1000, "sort": "published.desc"}).get("results") or []
        except Exception as ex:  # noqa: BLE001
            vol[t] = str(ex)[:60]
            continue
        yrs = pd.Series([x["published"][:4] for x in g if x.get("published")]).value_counts().sort_index()
        vol[t] = {"page_of_1000_spans": f"{g[-1]['published'][:10]} to {g[0]['published'][:10]}" if g else None,
                  "per_year_recent": yrs.tail(3).to_dict()}
    res["headline_volume_sample"] = vol
    (OUT / "stock_s0.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_year", "headline_volume_sample")}, indent=1, default=str)[:4000])
    print(json.dumps(vol, indent=0)[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "audit"])
    a = ap.parse_args()
    step_build() if a.step == "build" else step_audit()
