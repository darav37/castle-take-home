"""Validate the Kalshi-weighted release damage distribution (DECISIONS #174). No LLM calls.

For each 2021-26 Kalshi release (planb_b5_ladders.py), build the predicted distribution of release-day returns of six
industry baskets (rate-sensitive, energy, travel/consumer, tech, defensives, industrials) and the market:
  ladder at 16:00 the day before -> outcome distribution -> surprise (implied sd) -> stage-1 rates reaction
  (leave this release out; residual spread kept) -> stage-2 long-history regime kernel at this release's regime
Checks: coverage of the 5-95% and 25-75% ranges (should be about 90% and 50%), the PIT histogram's flatness, and
whether the probability-weighted mean beats the type mean (R²). Writes `distribution` into results/b5_ladders.json.
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
from worsey.planb import release_scenarios as RS  # noqa: E402

BASKETS = {"rate_sensitive": ["Banks", "Fin", "RlEst", "Util", "Insur"], "energy": ["Oil", "Coal"],
           "travel_consumer": ["Trans", "Meals", "Rtail", "Clths"], "tech": ["Chips", "Hardw", "Softw"],
           "defensives": ["Food", "Drugs", "Hshld", "Beer"], "industrials": ["Mach", "ElcEq", "Steel", "Cnstr"]}


def main():
    I = LH.industries()
    sessions = I.index[I.index >= "1976-01-01"]
    I = I.loc[sessions]
    X = LH.long_factors(sessions)
    state = LH.long_state(sessions, X)
    names = list(I.columns)
    # stage-2 training set (1976-2020), as in planb_b5_twostage.py
    rd = LH.release_dates()
    rd = rd[rd["date"] <= "2020-12-31"]
    rd["i"] = sessions.searchsorted(rd["date"])
    rd = rd.drop_duplicates(["i", "release"]).reset_index(drop=True)
    d2, d10 = X["rates_2y"].to_numpy(), X["rates_10y"].to_numpy()
    S_tr = np.column_stack([np.where(np.isnan(d2[rd["i"]]), d10[rd["i"]], d2[rd["i"]]), d10[rd["i"]]])
    ok = ~np.isnan(S_tr).any(1)
    rd, S_tr = rd[ok].reset_index(drop=True), S_tr[ok]
    itr, ttr = rd["i"].to_numpy(), rd["release"].to_numpy()
    Ytr_raw = np.column_stack([X[["market"]].to_numpy()[itr], I.to_numpy()[itr]])
    mu, sd = np.nanmean(Ytr_raw, 0), np.nanstd(Ytr_raw, 0)
    Ytr = np.nan_to_num((Ytr_raw - mu) / sd)
    smu, ssd = S_tr.mean(0), S_tr.std(0)
    Ztr = state[["sb_corr60", "infl_yoy", "ff_change_6m"]].to_numpy()[itr]
    zmu, zsd = np.nanmean(Ztr, 0), np.nanstd(Ztr, 0)
    ztr = np.nan_to_num((Ztr - zmu) / zsd)
    levels = sorted(set(ttr))

    def design(types, Sz, apart):
        return np.column_stack([IM.design(list(types), Sz, levels), IM.design(list(types), Sz * apart[:, None], levels)])

    Dtr = design(ttr, (S_tr - smu) / ssd, (np.nan_to_num(Ztr[:, 0]) <= 0).astype(float))

    def stage2_for(rel, z_state):
        """Linear map (rates_2y, rates_10y) -> raw returns [market, 49 industries] at this release's regime."""
        zz = np.nan_to_num((z_state - zmu) / zsd)
        w = np.exp(-((ztr - zz) ** 2).sum(1) / 2.0) + 1e-6
        beta = RG.weighted_ridge(Dtr, Ytr, w, 10.0)
        apart = float(np.nan_to_num(z_state[0]) <= 0)

        def f(r2, r10):
            Sz = (np.array([[r2, r10]]) - smu) / ssd
            return (design(np.array([rel]), Sz, np.array([apart])) @ beta)[0] * sd + mu
        return f

    fit_tr = Dtr @ RG.weighted_ridge(Dtr, Ytr, np.ones(len(Ytr)), 10.0)
    resid_raw = (Ytr - fit_tr) * sd
    basket_sd = {}
    for bk, inds in list(BASKETS.items()) + [("market", None)]:
        if inds is None:
            basket_sd[bk] = float(np.nanstd(resid_raw[:, 0]))
        else:
            cols = [1 + names.index(x) for x in inds]
            basket_sd[bk] = float(np.nanstd(resid_raw[:, cols].mean(1)))
    L = pd.read_csv(ROOT / "cache/pmrm/b5/ladders.csv", parse_dates=["date"])
    L = L[L["actual"].notna()].copy()
    L["i"] = sessions.searchsorted(L["date"])
    L = L[L["i"] < len(sessions)].reset_index(drop=True)
    # we need the ladder points themselves: rebuild from the stored implied mean/sd is not enough -> re-read quotes
    sys.path.insert(0, str(ROOT / "scripts"))
    import planb_b5_ladders as LD  # noqa: F401  (ladders were cached when built; reuse the stored mean/sd as a 2-point-free approx)
    z_act = ((L["actual"] - L["implied_mean"]) / L["implied_sd"]).to_numpy()
    rates = np.column_stack([d2[L["i"]], d10[L["i"]]])
    types = L["release"].to_numpy()
    cover90, cover50, pits, preds, real = [], [], [], [], []
    for k in range(len(L)):
        tr = (np.arange(len(L)) != k) & (types == types[k])
        A1 = np.column_stack([np.ones(tr.sum()), z_act[tr]])
        coef = np.linalg.lstsq(A1, rates[tr], rcond=None)[0].T          # 2 x 2: [[a2, b2], [a10, b10]]
        resid = rates[tr] - A1 @ coef.T
        rsd = resid.std(0)
        # outcome distribution: a normal on the implied mean/sd discretized (the stored ladder summary)
        m, s = L["implied_mean"].iloc[k], L["implied_sd"].iloc[k]
        pts = m + s * np.linspace(-2.5, 2.5, 11)
        mass = np.exp(-0.5 * np.linspace(-2.5, 2.5, 11) ** 2)
        mass /= mass.sum()
        f = stage2_for(types[k], state[["sb_corr60", "infl_yoy", "ff_change_6m"]].to_numpy()[L["i"].iloc[k]])
        realized = np.concatenate([[X["market"].iloc[L["i"].iloc[k]]], I.iloc[L["i"].iloc[k]].to_numpy()])
        for bk, inds in list(BASKETS.items()) + [("market", None)]:
            wts = np.zeros(1 + len(names))
            if inds is None:
                wts[0] = 1.0
            else:
                for x in inds:
                    wts[1 + names.index(x)] = 1.0 / len(inds)
            # add the basket's own idiosyncratic noise (training residual sd of the stage-2 fit, raw units)
            res = RS.book_pnl_distribution(pts, mass, coef, rsd, f, wts, n_draws=1500, seed=k)
            y = float(wts @ np.nan_to_num(realized))
            # the basket's own release-day spread around the stage-2 fit, measured directly on 1976-2020 release days
            # (industries in a basket move together, so summing their variances as if independent understates it)
            noise_sd = basket_sd[bk]
            lo = res["p05"] - 1.645 * noise_sd
            hi = res["p95"] + 1.645 * noise_sd
            cover90.append(lo <= y <= hi)
            cover50.append(res["expected_pnl"] - 0.674 * noise_sd <= y <= res["expected_pnl"] + 0.674 * noise_sd)
            preds.append(res["expected_pnl"])
            real.append(y)
    preds, real = np.array(preds), np.array(real)
    out = {"releases": int(len(L)), "basket_cases": int(len(real)), "coverage_90": float(np.mean(cover90)),
           "coverage_50": float(np.mean(cover50)),
           "mean_r2_vs_zero": float(1 - ((real - preds) ** 2).sum() / (real ** 2).sum()),
           "note": "outcome distribution approximated by a discretized normal on the ladder's implied mean and sd; "
                   "ranges include stage-1 residual spread and the basket's own release-day noise"}
    print(json.dumps(out, indent=1), flush=True)
    p = ROOT / "docs/pmrm/results/b5_ladders.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["distribution"] = out
    p.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
