"""Plan C, C0: the frozen test's dataset and the baseline rungs (docs/research/plan_c.md §7a; DECISIONS #187). No LLM calls.

  python scripts/planc_c0.py build      # dataset: every one-off event x 13 factors, 96 ETF residuals, 12 books
  python scripts/planc_c0.py score      # rungs 0, 0b and 1 on validation (2023-25) and test (2026)

Targets per event (second-session close = the library's short window; the 2-hour horizon comes in C1):
  factor:<name>   the 13 factors' move U (research anchors, the units the engine works in)
  etf:<ticker>    the 96 ETFs' move beyond the 13 factors (window return - pre-event loadings x the factors' window move)
  book:<name>     the 12 portfolios of #185 (total return)
Noise per target: its daily sd (residual sd for ETF residuals) over the 504 sessions before the event x sqrt(window).
Rungs:
  0   predict zero, normal noise
  0b  predict zero, normal noise x the severity's event premium (estimated on earlier splits only); won #185
  1   today's engine (live.scenario_draws), refitted on events before the split (validation: < 2023; test: < 2026);
      it has no ETF-residual estimate, so its ETF residuals are zero with the ETF's own noise x sqrt(1.5) (as shipped)
Saved: cache/pmrm/planc/c0_events.pkl (dataset), cache/pmrm/planc/c0_rungs.pkl (scored rows), docs/pmrm/results/planc_c0.json.
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
from planb_naive_vs_system import books  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.planc import harness as H  # noqa: E402

DIR = ROOT / "cache/pmrm/planc"
LOOKBACK = 504
F = live.FACTORS


def step_build():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    X = ((ext["anchors"] - ext["anchor_mean"]) / ext["anchor_sd"])[F]
    ss = X.index
    etfs = list(pickle.load(open(ROOT / "cache/pmrm/b1/model.pkl", "rb"))["mu"].index)
    B = books()
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    R = B1.total_returns(sorted(set(proxy_t) | set(etfs) | {t for w in B.values() for t in w}), ss, "2003-01-01")
    Fz = live.factor_returns(R).reindex(columns=F)
    Bk = pd.DataFrame({b: sum(wt * R[t] for t, wt in w.items()) for b, w in B.items()})
    Xa, Fa, Ra, Ba = X.to_numpy(), Fz.to_numpy(), R[etfs].to_numpy(), Bk.to_numpy()
    live.set_cutoff(None)
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8")) if r["kind"] == "unscheduled"}
    out = []
    for e in live.pack()["events"]:
        if e["id"] not in lib:
            continue
        w1 = lib[e["id"]]["w1"]
        a = ss.searchsorted(pd.Timestamp(w1["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(w1["end"])) + 1
        if a < LOOKBACK + 2 or b <= a or b > len(ss):
            continue
        n = b - a
        lb = slice(a - 1 - LOOKBACK, a - 1)
        Fl, Fw = Fa[lb], np.nansum(Fa[a:b], axis=0)
        okF = np.isfinite(Fl).all(1)
        rec = {"id": e["id"], "date": e["date"], "split": H.split_of(e["date"]), "severity": e.get("severity") or "moderate",
               "category": e["category"], "n": n, "spec": {k: e.get(k) for k in ("category", "severity", "region", "direction",
                                                                                  "targeted_sectors", "anticipation", "title")},
               "factor_y": np.array(e["U"]), "factor_noise": np.nanstd(Xa[lb], axis=0) * np.sqrt(n), "etf": {}, "book": {}}
        for cols, arr, kind in ((etfs, Ra, "etf"), (list(B), Ba, "book")):
            for j, t in enumerate(cols):
                y = arr[lb, j]
                ok = okF & np.isfinite(y)
                yw = arr[a:b, j]
                if ok.sum() < 250 or not np.isfinite(yw).all():
                    continue
                A = np.column_stack([np.ones(ok.sum()), Fl[ok]])
                c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
                res = y[ok] - A @ c
                beta = c[1:]
                if kind == "etf":
                    rec["etf"][t] = {"y": float(yw.sum() - beta @ Fw), "noise": float(res.std() * np.sqrt(n)),
                                     "own_sd": float(res.std()), "beta": beta}
                else:
                    rec["book"][t] = {"y": float(np.prod(1 + yw) - 1), "noise": float(y[ok].std() * np.sqrt(n)),
                                      "own_sd": float(res.std()), "beta": beta}
        out.append(rec)
    DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump({"events": out, "etfs": etfs, "books": list(B), "factors": F}, open(DIR / "c0_events.pkl", "wb"))
    sp = pd.Series([r["split"] for r in out]).value_counts().to_dict()
    print(f"dataset: {len(out)} events {sp}; ETF residual targets per event (median) "
          f"{int(np.median([len(r['etf']) for r in out]))}; books {int(np.median([len(r['book']) for r in out]))}")


def premium(events: list[dict]) -> dict:
    """RMS of real factor moves in normal-noise units, by severity, over the given (earlier) events (#184)."""
    out = {}
    for sv in live.SEVERITIES:
        z = np.concatenate([r["factor_y"] / r["factor_noise"] for r in events if r["severity"] == sv] or [np.array([])])
        z = z[np.isfinite(z)]
        out[sv] = float(np.sqrt(np.mean(z ** 2))) if len(z) else 1.0
    return out


def step_score():
    D = pickle.load(open(DIR / "c0_events.pkl", "rb"))
    ev = D["events"]
    rows = []
    for split, cutoff, earlier in (("validation", "2023-01-01", ("train",)), ("test", "2026-01-01", ("train", "validation"))):
        prem = premium([r for r in ev if r["split"] in earlier])
        live.set_cutoff(cutoff)
        for r in [r for r in ev if r["split"] == split]:
            k = prem[r["severity"]]
            spec = live._spec_ok(r["spec"])
            base = live.scenario_draws(spec, np.zeros(len(F)), 0.0)
            f, dc = base["f"], base["draw"] - np.median(base["draw"], axis=0)
            rng = np.random.default_rng(1729)
            z = rng.normal(0, 1, live.N_DRAWS) * np.sqrt(1.5)
            targets = [(f"factor:{fac}", "factor", float(r["factor_y"][j]), float(r["factor_noise"][j]),
                        H.quantiles_from_samples(f[j] + dc[:, j])) for j, fac in enumerate(F)]
            for t, v in r["etf"].items():
                targets.append((f"etf:{t}", "etf residual", v["y"], v["noise"], H.quantiles_normal(0.0, v["own_sd"] * np.sqrt(1.5))))
            for t, v in r["book"].items():
                x = v["beta"] @ f + dc @ v["beta"] + z * v["own_sd"]
                x = x - np.median(x) + v["beta"] @ f
                targets.append((f"book:{t}", "book", v["y"], v["noise"], H.quantiles_from_samples(x)))
            for name, kind, y, sd, q1 in targets:
                if not (np.isfinite(y) and np.isfinite(sd) and sd > 0):
                    continue
                for rung, q in (("0 noise", H.quantiles_normal(0, sd)), ("0b noise+premium", H.quantiles_normal(0, sd * k)), ("1 engine", q1)):
                    rows.append({"split": split, "event": r["id"], "severity": r["severity"], "category": r["category"],
                                 "target": name, "kind": kind, "rung": rung, "y": y, "noise_sd": sd, "q": q})
        print(f"{split}: scored", flush=True)
    live.set_cutoff(None)
    S = H.score_rows(pd.DataFrame(rows))
    S.drop(columns=["q"]).to_pickle(DIR / "c0_rungs.pkl")
    res = {}
    for split in ("validation", "test"):
        res[split] = {}
        for kind in ("factor", "etf residual", "book"):
            for sub, mask in (("all", lambda d: d), ("big events", lambda d: d[d.severity.isin(["major", "extreme"])])):
                d = mask(S[(S.split == split) & (S.kind == kind)])
                if d.empty:
                    continue
                by = {rg: H.summary(d[d.rung == rg]) for rg in sorted(d.rung.unique())}
                cmp_ = {"1 vs 0": H.paired_better(d[d.rung == "1 engine"], d[d.rung == "0 noise"]),
                        "1 vs 0b": H.paired_better(d[d.rung == "1 engine"], d[d.rung == "0b noise+premium"]),
                        "0b vs 0": H.paired_better(d[d.rung == "0b noise+premium"], d[d.rung == "0 noise"])}
                res[split][f"{kind} | {sub}"] = {"rungs": by, "comparisons": cmp_}
                print(f"{split:10s} {kind:13s} {sub:10s} " + " | ".join(
                    f"{rg}: crps {x['crps_std']:.3f} cov90 {x['coverage_90']:.0%}" for rg, x in by.items())
                      + " || " + "; ".join(f"{k} {v['b_minus_a']:+.4f} ({v['a_better_support']:.0%})" for k, v in cmp_.items()))
    (ROOT / "docs/pmrm/results/planc_c0.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "score"])
    a = ap.parse_args()
    step_build() if a.step == "build" else step_score()
