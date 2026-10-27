"""Sector layer, step 2a: when a company reports earnings, how do its sector peers move? (DECISIONS #216)
No LLM calls.

  python scripts/planc_spillover.py

Peers: the other S&P 500 stocks whose best sector (step 1, #215) is the same as the reporter's, excluding the reporter
(the sector ETF itself would be contaminated: NVDA is about a fifth of SMH). Each peer's move beyond the 13 macro
factors on the reporter's reaction day (loadings over the 252 sessions before), in units of its own residual noise; the
peer move = the average over peers, rescaled to the noise of such an average on ordinary days (so 1 = a normal day for
that peer group). Controls: the same peer groups on random ordinary sessions.
Scored: size vs ordinary days; direction from (a) the reporter's EPS surprise and (b) the reporter's own move; for all
reporters and for leaders (the 3 largest names in each sector by S&P 500 weight); by period.
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
from worsey.data import massive  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.planc import harness as H  # noqa: E402

LOOK = 252


def main():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    rows = pd.read_pickle(ROOT / "cache/pmrm/planc/sector_r2_rows.pkl")
    grp = rows[(rows.kind == "stock") & rows.best_sector.notna()].set_index("name").best_sector
    names = list(grp.index)
    E = pd.read_pickle(ROOT / "cache/pmrm/stock/earnings.pkl")
    E = E[E.ticker.isin(names)].copy()
    w = massive.etf_holdings("SPY")
    wcol = next((c for c in ("weight", "pct", "percent", "weight_pct") if c in w.columns), None)
    wt = dict(zip(w["ticker"], pd.to_numeric(w[wcol], errors="coerce"))) if wcol else {}
    leaders = set()
    for g, members in grp.groupby(grp).groups.items():
        top = sorted(members, key=lambda t: -(wt.get(t) or 0))[:3]
        leaders |= set(top)
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    Fz = live.factor_returns(B1.total_returns(proxy_t, ss, "1990-01-01")).reindex(columns=live.FACTORS).to_numpy()
    RS = B1.total_returns(names, ss, "2010-12-01").apply(pd.to_numeric, errors="coerce")
    R = RS.to_numpy()
    col = {t: i for i, t in enumerate(RS.columns)}

    cache = {}

    def resid_z(t, a):
        """Stock t's move beyond the macro factors on session a, in its residual sd (loadings over the 252 before)."""
        k = (t, a // 21)                                  # refit loadings monthly
        if k not in cache:
            y, X = R[a - 1 - LOOK:a - 1, col[t]], Fz[a - 1 - LOOK:a - 1]
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            if ok.sum() < 150:
                cache[k] = None
            else:
                A = np.column_stack([np.ones(ok.sum()), X[ok]])
                c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
                cache[k] = (c, float((y[ok] - A @ c).std()))
        if cache[k] is None or not np.isfinite(R[a, col[t]]) or not np.isfinite(Fz[a]).all():
            return np.nan
        c, sd = cache[k]
        return (R[a, col[t]] - c[0] - Fz[a] @ c[1:]) / sd

    def peer_move(g, a, exclude):
        zs = [resid_z(p, a) for p in grp.index[grp == g] if p != exclude]
        zs = [z for z in zs if np.isfinite(z)]
        return (float(np.mean(zs)), len(zs)) if len(zs) >= 4 else (np.nan, 0)

    out = []
    for r in E.itertuples():
        d = pd.Timestamp(r.date)
        i = ss.searchsorted(d)
        a = i + 1 if (r.timing == "after close" and i < len(ss) and ss[i] == d) else i
        if a < LOOK + 2 or a >= len(ss) - 1:
            continue
        g = grp[r.ticker]
        pm, n = peer_move(g, a, r.ticker)
        own = resid_z(r.ticker, a)
        if np.isfinite(pm) and np.isfinite(own):
            out.append({"ticker": r.ticker, "sector": g, "date": r.date, "split": H.split_of(r.date), "leader": r.ticker in leaders,
                        "surprise": r.eps_surprise_pct, "own_z": own, "peer_avg": pm, "n_peers": n})
    O = pd.DataFrame(out)
    # ordinary-day scale for each peer group's average (random sessions, not near any earnings of the group)
    rng = np.random.default_rng(1729)
    busy = {}
    for r in O.itertuples():
        busy.setdefault(r.sector, set()).add(r.date)
    scale = {}
    for g in grp.unique():
        vals = []
        for a in rng.choice(np.arange(LOOK + 2, len(ss) - 1), 250, replace=False):
            if str(ss[a].date()) in busy.get(g, set()):
                continue
            pm, n = peer_move(g, a, None)
            if np.isfinite(pm):
                vals.append(pm)
        scale[g] = float(np.std(vals)) if len(vals) > 50 else np.nan
    O["peer_z"] = O.peer_avg / O.sector.map(scale)
    O = O[np.isfinite(O.peer_z)]
    O.to_pickle(ROOT / "cache/pmrm/planc/spillover_rows.pkl")
    res = {"reports": int(len(O)), "peer_groups": {g: int((grp == g).sum()) for g in grp.unique()}, "leaders": sorted(leaders)}
    for sub, d in (("all reporters", O), ("leaders", O[O.leader])):
        big = d[d.peer_z.abs() > 2]
        r_ = {"reports": int(len(d)), "peer_rms": float(np.sqrt((d.peer_z ** 2).mean())), "peer_share_over_2x": float((d.peer_z.abs() > 2).mean()),
              "normal_day_share_over_2x": 0.046}
        x = d[d.surprise.notna() & (d.surprise != 0)]
        r_["direction_from_surprise"] = float((np.sign(x.surprise) == np.sign(x.peer_z)).mean())
        r_["direction_from_surprise_big_peer_moves"] = float((np.sign(x.surprise) == np.sign(x.peer_z))[x.peer_z.abs() > 2].mean()) if (x.peer_z.abs() > 2).any() else None
        r_["direction_from_reporter_move"] = float((np.sign(d.own_z) == np.sign(d.peer_z)).mean())
        bigown = d[d.own_z.abs() > 3]
        r_["direction_from_big_reporter_moves"] = float((np.sign(bigown.own_z) == np.sign(bigown.peer_z)).mean()) if len(bigown) else None
        r_["peer_slope_on_reporter_move"] = float((d.own_z @ d.peer_z) / (d.own_z @ d.own_z))
        r_["by_split"] = {sp: {"n": int(len(g_)), "rms": float(np.sqrt((g_.peer_z ** 2).mean())),
                               "dir_surprise": float((np.sign(g_.surprise) == np.sign(g_.peer_z))[g_.surprise.notna() & (g_.surprise != 0)].mean()),
                               "dir_reporter": float((np.sign(g_.own_z) == np.sign(g_.peer_z)).mean()),
                               "slope": float((g_.own_z @ g_.peer_z) / (g_.own_z @ g_.own_z))} for sp, g_ in d.groupby("split")}
        res[sub] = r_
        print(f"{sub}: {r_['reports']} reports | peer move RMS {r_['peer_rms']:.2f}x a normal day, >2x {r_['peer_share_over_2x']:.0%} (normal 4.6%) | "
              f"direction from surprise {r_['direction_from_surprise']:.0%} (big peer moves {r_['direction_from_surprise_big_peer_moves'] or float('nan'):.0%}) | "
              f"from the reporter's move {r_['direction_from_reporter_move']:.0%} (big reporter moves {r_['direction_from_big_reporter_moves'] or float('nan'):.0%}), slope {r_['peer_slope_on_reporter_move']:+.3f}", flush=True)
        print("   by split " + json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in r_["by_split"].items()}), flush=True)
    by = O[O.leader].groupby("sector").agg(n=("peer_z", "size"), rms=("peer_z", lambda z: float(np.sqrt((z ** 2).mean()))),
                                            dir_reporter=("own_z", lambda z: 0.0), slope=("own_z", lambda z: 0.0))
    for g in by.index:
        d = O[O.leader & (O.sector == g)]
        by.loc[g, "dir_reporter"] = float((np.sign(d.own_z) == np.sign(d.peer_z)).mean())
        by.loc[g, "slope"] = float((d.own_z @ d.peer_z) / (d.own_z @ d.own_z))
    res["leaders_by_sector"] = by.round(3).to_dict(orient="index")
    print(by.round(2).sort_values("slope", ascending=False).to_string())
    (ROOT / "docs/pmrm/results/planc_spillover.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
