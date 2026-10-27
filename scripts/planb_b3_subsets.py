"""B3 on the grown library, broken down by origin, severity and category (Oct 21). No LLM calls.

Leave-one-out predictions of the code-learned direction model (exact LOO ridge, fitted on ALL events) and the LLM's
factor signs, scored on subsets. Chance per subset: the same scoring with direction labels (or sign vectors) shuffled
across all events, 200 times. Adds `subsets` to docs/pmrm/results/b3.json.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worsey.planb import impact as IM  # noqa: E402

NON = ["market", "dollar", "oil", "gold", "credit", "size", "value"]


def main(perms: int = 200):
    rng = np.random.default_rng(1729)
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    m = pickle.load(open(ROOT / "cache/pmrm/b1/model.pkl", "rb"))
    acols = list(m["anchors"].columns)
    ia = [acols.index(c) for c in NON]
    jm = acols.index("market")
    uns = [r for r in lib if r["kind"] == "unscheduled"]
    n = len(uns)
    subsets = {"all": np.ones(n, bool),
               "origin: curated (194 hand-picked)": np.array([r.get("origin", "curated_v0") == "curated_v0" for r in uns]),
               "origin: wikipedia (systematic)": np.array([r.get("origin") == "wikipedia_portal_monthly" for r in uns])}
    for s in ("extreme", "major", "moderate", "minor"):
        subsets[f"severity: {s}"] = np.array([r.get("severity") == s for r in uns])
    subsets["severity: extreme or major"] = subsets["severity: extreme"] | subsets["severity: major"]
    for c in sorted({r["category"] for r in uns}):
        subsets[f"category: {c}"] = np.array([r["category"] == c for r in uns])
    out = {}
    for w in ("w1", "w5"):
        U = np.array([[r[w]["factors_anchored"][c] for c in acols] for r in uns])
        Fd = np.column_stack([np.ones(n), np.array([IM.direction_features(e) for e in uns])])
        S = np.zeros((n, len(acols)))
        for i, e in enumerate(uns):
            for c, sg in (((e.get("direction") or {}).get("factor_signs")) or {}).items():
                if c in acols and sg is not None:
                    S[i, acols.index(c)] = sg
        mean_abs = (np.abs(U).sum(0)[None, :] - np.abs(U)) / (n - 1)

        def preds(F, Sg):
            return IM.ridge_loo_predictions(F, U, 2.0), Sg * mean_abs

        def score(P, mk):
            r2 = IM.r2_vs(U[mk][:, ia], P[mk][:, ia], np.zeros((mk.sum(), len(ia))))
            called = mk & (P[:, jm] != 0)
            hit = float(np.mean(np.sign(P[called, jm]) == np.sign(U[called, jm]))) if called.any() else np.nan
            return r2, hit

        Pd, Ps = preds(Fd, S)
        nulls = [preds(Fd[np.r_[0:n][rng.permutation(n)]], S[rng.permutation(n)]) for _ in range(perms)]
        res = {}
        for name, mk in subsets.items():
            if mk.sum() < 20:
                continue
            od, os_ = score(Pd, mk), score(Ps, mk)
            nd = np.array([score(a, mk) for a, _ in nulls])
            ns = np.array([score(b, mk) for _, b in nulls])
            res[name] = {"n": int(mk.sum()),
                         "mean_abs_market_move": float(np.abs(U[mk, jm]).mean()),
                         "direction_model": {"r2": od[0], "r2_null95": float(np.nanquantile(nd[:, 0], 0.95)),
                                             "p": float((1 + (nd[:, 0] >= od[0]).sum()) / (1 + perms)),
                                             "market_hit": od[1], "hit_null95": float(np.nanquantile(nd[:, 1], 0.95))},
                         "llm_signs": {"r2": os_[0], "r2_null95": float(np.nanquantile(ns[:, 0], 0.95)),
                                       "p": float((1 + (ns[:, 0] >= os_[0]).sum()) / (1 + perms)),
                                       "market_hit": os_[1], "hit_null95": float(np.nanquantile(ns[:, 1], 0.95))}}
            d, l = res[name]["direction_model"], res[name]["llm_signs"]
            print(f"{w} {name:36} n={mk.sum():5d} |mkt|={res[name]['mean_abs_market_move']:.2f}  dir R2 {d['r2']:+.3f} (null95 {d['r2_null95']:+.3f}, p {d['p']:.3f}) hit {d['market_hit']:.2f}"
                  f"  | LLM R2 {l['r2']:+.3f} (p {l['p']:.3f}) hit {l['market_hit']:.2f} (null95 {l['hit_null95']:.2f})", flush=True)
        out[w] = res
    p = ROOT / "docs/pmrm/results/b3.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["subsets"] = out
    p.write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
