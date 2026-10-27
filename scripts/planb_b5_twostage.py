"""B5 part 2: the forward-looking chain for releases (DECISIONS #172). No LLM calls.

Product question: "what if CPI prints 0.4% when Kalshi expects 0.25%?" Only the surprise is known, not the market's
reaction. Two stages:
  stage 1  Kalshi surprise (actual - implied mean, in implied sd) -> release-day 2y and 10y yield changes
           (per release type, leave one release out among the 2021-26 Kalshi releases)
  stage 2  predicted yield changes -> 49-industry and factor responses, through the long-history regime kernel
           (trained 1976-2020 on realized yield changes, settings from planb_longrel.py), using each release's regime
Compared on the same releases with: the type mean (baseline), stage 2 fed the ACTUAL yield changes (the upper bound for
this chain), and Kalshi surprise -> industries directly. Adds `twostage` to docs/pmrm/results/b5_ladders.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey.planb import impact as IM  # noqa: E402
from worsey.planb import longhist as LH  # noqa: E402
from worsey.planb import regime as RG  # noqa: E402


def main(perms: int = 100):
    rng = np.random.default_rng(1729)
    I = LH.industries()
    sessions = I.index[I.index >= "1976-01-01"]
    I = I.loc[sessions]
    X = LH.long_factors(sessions)
    state = LH.long_state(sessions, X)
    nonrates = ["market", "size", "value", "profitability", "investment", "momentum", "dollar", "oil"]
    # long-history training releases (1976-2020)
    rd = LH.release_dates()
    rd = rd[(rd["date"] <= "2020-12-31")]
    rd["i"] = sessions.searchsorted(rd["date"])
    rd = rd.drop_duplicates(["i", "release"]).reset_index(drop=True)
    d2, d10 = X["rates_2y"].to_numpy(), X["rates_10y"].to_numpy()
    S_tr = np.column_stack([np.where(np.isnan(d2[rd["i"]]), d10[rd["i"]], d2[rd["i"]]), d10[rd["i"]]])
    ok = ~np.isnan(S_tr).any(1)
    rd, S_tr = rd[ok].reset_index(drop=True), S_tr[ok]
    i_tr = rd["i"].to_numpy()
    t_tr = rd["release"].to_numpy()
    Y_tr = np.column_stack([X[nonrates].to_numpy()[i_tr], I.to_numpy()[i_tr]])
    mu, sd = np.nanmean(Y_tr, 0), np.nanstd(Y_tr, 0)
    Ytr = np.nan_to_num((Y_tr - mu) / sd)
    smu, ssd = S_tr.mean(0), S_tr.std(0)
    Z_tr = state[["sb_corr60", "infl_yoy", "ff_change_6m"]].to_numpy()[i_tr]
    zmu, zsd = np.nanmean(Z_tr, 0), np.nanstd(Z_tr, 0)
    levels = sorted(set(t_tr))

    def design(types, Sz, apart):
        return np.column_stack([IM.design(list(types), Sz, levels), IM.design(list(types), Sz * apart[:, None], levels)])

    D_tr = design(t_tr, (S_tr - smu) / ssd, (np.nan_to_num(Z_tr[:, 0]) <= 0).astype(float))
    ztr = np.nan_to_num((Z_tr - zmu) / zsd)

    def stage2(types, S, Z):
        """Kernel model (bandwidth 1, ridge 10, as chosen by leave-one-era-out) for each row."""
        Sz = (S - smu) / ssd
        D = design(types, Sz, (np.nan_to_num(Z[:, 0]) <= 0).astype(float))
        zz = np.nan_to_num((Z - zmu) / zsd)
        P = np.zeros((len(types), Ytr.shape[1]))
        for k in range(len(types)):
            w = np.exp(-((ztr - zz[k]) ** 2).sum(1) / 2.0) + 1e-6
            P[k] = D[k] @ RG.weighted_ridge(D_tr, Ytr, w, 10.0)
        return P

    # Kalshi releases (from planb_b5_ladders.py)
    L = pd.read_csv(ROOT / "cache/pmrm/b5/ladders.csv", parse_dates=["date"])
    L = L[L["actual"].notna() & L["surprise_z"].notna()].copy()
    L["i"] = sessions.searchsorted(L["date"])
    L = L[L["i"] < len(sessions)].reset_index(drop=True)
    idx = L["i"].to_numpy()
    types = L["release"].to_numpy()
    Y = np.nan_to_num((np.column_stack([X[nonrates].to_numpy()[idx], I.to_numpy()[idx]]) - mu) / sd)
    Sact = np.column_stack([d2[idx], d10[idx]])
    Z = state[["sb_corr60", "infl_yoy", "ff_change_6m"]].to_numpy()[idx]
    sz = L["surprise_z"].to_numpy()
    # stage 1: surprise -> (d2y, d10y), leave one release out within type
    S1 = np.zeros_like(Sact)
    for k in range(len(L)):
        tr = (np.arange(len(L)) != k) & (types == types[k])
        A1 = np.column_stack([np.ones(tr.sum()), sz[tr]])
        b = np.linalg.lstsq(A1, Sact[tr], rcond=None)[0]
        S1[k] = np.array([1.0, sz[k]]) @ b
    r2_stage1 = {c: float(1 - ((Sact[:, j] - S1[:, j]) ** 2).sum() / ((Sact[:, j] - Sact[:, j].mean()) ** 2).sum()) for j, c in enumerate(["2y", "10y"])}
    # baseline: type mean of the long-history training responses
    Xb = IM.design(list(t_tr), np.zeros((len(t_tr), 1)), levels)
    Bb = RG.weighted_ridge(Xb, Ytr, np.ones(len(t_tr)), 1.0)
    base = IM.design(list(types), np.zeros((len(types), 1)), levels) @ Bb
    cf, ci = list(range(8)), list(range(8, Y.shape[1]))

    def sc(P, cols):
        return float(1 - ((Y[:, cols] - P[:, cols]) ** 2).sum() / ((Y[:, cols] - base[:, cols]) ** 2).sum())

    P_actual = stage2(types, Sact, Z)          # upper bound: the real rates reaction known
    P_chain = stage2(types, S1, Z)             # forward-looking: only the Kalshi surprise known
    out = {"stage1_r2_kalshi_to_rates": r2_stage1, "releases": int(len(L)),
           "industries": {"chain (Kalshi surprise -> rates -> industries)": sc(P_chain, ci),
                          "stage 2 with actual rates (upper bound)": sc(P_actual, ci)},
           "factors": {"chain": sc(P_chain, cf), "stage 2 with actual rates": sc(P_actual, cf)}}
    nl = []
    for _ in range(perms):
        perm = rng.permutation(len(L))
        nl.append(sc(stage2(types, S1[perm], Z), ci))
    out["industries_chain_chance95"] = float(np.quantile(nl, 0.95))
    print(json.dumps(out, indent=1, default=lambda x: round(x, 4)), flush=True)
    p = ROOT / "docs/pmrm/results/b5_ladders.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["twostage"] = out
    p.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
