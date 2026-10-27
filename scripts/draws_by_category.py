"""Does drawing a scenario's spread from past events of the same category (not just the same severity) help?
(the design brief; DECISIONS #239). No LLM calls.

  python scripts/draws_by_category.py history   # the range tests (#183, #185, #186) under both pools -> json
  python scripts/draws_by_category.py whatif    # the live what-if path under both pools, held out by year -> json
  python scripts/draws_by_category.py report    # -> docs/pmrm/results/draws_by_category.md

PREDECLARED ship rule: ship the category pool only if it is no worse on interval score overall at 90% bootstrap support
(the paired difference new - current has its 90% interval's upper end <= 0, or its mean <= 0 with support >= 90%) and
the 9-in-10 coverage stays within +-3 points of 90%; otherwise keep the severity pool.

history  The history-average path (live.scenario_draws), as the shipped range tests score it: planb_range_calibration
         (#183: 9-in-10 coverage, 1-in-20 exceedance), planb_naive_vs_system score (#185: 12 books x 92 events of 2026,
         interval score) and planb_naive_vs_system_positions score (#186: summed over positions). Each is run twice,
         with WORSEY_DRAW_POOL=severity and =category; their committed results files are restored afterwards.
whatif   The live what-if path (live.real_event_draws, used by "what if X" and the watchlist), which the range tests
         don't cover: the major and extreme C0 events of 2023-2026 that the live user-event labeller labelled (#236:
         its directions and sizes, unchanged), each on the 12 C0 books (each book's loadings, normal noise and own noise
         fitted before the event; its realized return over the event window). Held out by year: the library is cut off
         at Jan 1 of the event's year (live.set_cutoff), so the pools contain only earlier events. Scores per
         book-event: 9-in-10 coverage, 1-in-20 exceedance, interval score (5/50/95); paired difference with a bootstrap
         clustered by event; by category.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "cache/pmrm/draws_by_category"
RES = ROOT / "docs/pmrm/results"
HIST = [("planb_range_calibration.py", [], "range_calibration.json"),
        ("planb_naive_vs_system.py", ["score"], "naive_vs_system.json"),
        ("planb_naive_vs_system_positions.py", ["score"], "naive_vs_system_positions.json")]


def history():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for pool in ("severity", "category"):
        env = dict(os.environ, WORSEY_DRAW_POOL=pool, PYTHONIOENCODING="utf-8")
        for script, args, outj in HIST:
            print(f"{pool}: {script}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)] + args, env=env, cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
            res[f"{pool}|{outj}"] = json.loads((RES / outj).read_text(encoding="utf-8"))
    subprocess.run(["git", "checkout", "--"] + [f"docs/pmrm/results/{o}" for _, _, o in HIST] +
                   [f"docs/pmrm/results/{o.replace('.json', '.md')}" for _, _, o in HIST], cwd=ROOT, check=True)
    (OUT / "history.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("history done; committed results restored")


def interval_score(lo, hi, y, alpha=0.1):
    return (hi - lo) + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)


def whatif():
    from worsey.planb import live
    OUT.mkdir(parents=True, exist_ok=True)
    D = pickle.load(open(ROOT / "cache/pmrm/planc/c0_events.pkl", "rb"))
    L2 = json.loads((ROOT / "cache/pmrm/live_labeller/L2.json").read_text(encoding="utf-8"))
    j7 = {f: D["factors"].index(f) for f in live.FACTORS}
    evs = [e for e in D["events"] if e["severity"] in ("major", "extreme") and "2023" <= e["date"][:4] <= "2026" and e["id"] in L2 and e["book"]]
    rows = []
    for year in sorted({e["date"][:4] for e in evs}):
        live.set_cutoff(f"{year}-01-01")
        for e in [x for x in evs if x["date"][:4] == year]:
            lab = L2[e["id"]]
            spec = {"severity": e["severity"], "category": e["category"],
                    "factor_signs": {f: v[0] for f, v in lab.items() if v[0]}, "factor_sizes": {f: v[1] for f, v in lab.items() if v[0]}}
            fn = np.array([e["factor_noise"][j7[f]] for f in live.FACTORS])
            fd = np.where(fn > 0, fn, 1.0) / np.sqrt(2)
            for book, b in e["book"].items():
                if not np.isfinite(b.get("y", np.nan)):
                    continue
                db = np.asarray(b["beta"], float)
                out = {}
                for pool in ("severity", "category"):
                    live.DRAW_POOL = pool
                    d = live.real_event_draws(spec, db, float(b.get("own_sd") or 0.0), fd)
                    q05, q50, q95 = np.quantile(d["end"], [0.05, 0.5, 0.95])
                    out[pool] = {"is": interval_score(q05, q95, b["y"]), "in90": int(q05 <= b["y"] <= q95), "below5": int(b["y"] < q05),
                                 "pool": d["draw_pool"]}
                rows.append({"event": e["id"], "year": year, "category": e["category"], "severity": e["severity"], "book": book,
                             "noise": b.get("noise"), **{f"{p}_{k}": v for p in out for k, v in out[p].items()}})
        live.set_cutoff(None)
    live.DRAW_POOL = "severity"
    (OUT / "whatif_rows.json").write_text(json.dumps(rows, default=float), encoding="utf-8")
    print(f"{len(rows)} book-events from {len({r['event'] for r in rows})} events")


def paired(rows, a="category_is", b="severity_is", B=2000, seed=0):
    ev = sorted({r["event"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault(r["event"], []).append(r[a] / (r["noise"] or 1) - r[b] / (r["noise"] or 1))
    rng = np.random.default_rng(seed)
    means = [np.mean([x for k in rng.choice(ev, len(ev)) for x in by[k]]) for _ in range(B)]
    m = float(np.mean([x for v in by.values() for x in v]))
    return {"mean": m, "ci90": [float(np.percentile(means, 5)), float(np.percentile(means, 95))], "support_new_better": float(np.mean(np.array(means) < 0))}


def summary(rows):
    def s(sub):
        return {"book_events": len(sub), "events": len({r["event"] for r in sub}),
                **{p: {"coverage_90": float(np.mean([r[f"{p}_in90"] for r in sub])), "below_5": float(np.mean([r[f"{p}_below5"] for r in sub])),
                       "interval_score_in_noise_units": float(np.mean([r[f"{p}_is"] / (r["noise"] or 1) for r in sub]))} for p in ("severity", "category")},
                "new_minus_current_interval_score": paired(sub) if sub else None}
    out = {"all": s(rows)}
    for c in sorted({r["category"] for r in rows}):
        out[c] = s([r for r in rows if r["category"] == c])
    pools = {}
    for r in rows:
        pools[r["category_pool"]] = pools.get(r["category_pool"], 0) + 1
    out["pools_used_by_the_category_rule"] = pools
    return out


def report():
    H = json.loads((OUT / "history.json").read_text(encoding="utf-8")) if (OUT / "history.json").exists() else {}
    W = summary(json.loads((OUT / "whatif_rows.json").read_text(encoding="utf-8")))
    (RES / "draws_by_category.json").write_text(json.dumps({"history": H, "whatif": W}, indent=1, default=float), encoding="utf-8")
    print(json.dumps(W["all"], indent=1, default=float))
    for c, v in W.items():
        if c not in ("all", "pools_used_by_the_category_rule"):
            print(f"  {c:26} n={v['book_events']:4} cov {v['severity']['coverage_90']:.2f}->{v['category']['coverage_90']:.2f} "
                  f"IS {v['severity']['interval_score_in_noise_units']:.3f}->{v['category']['interval_score_in_noise_units']:.3f} "
                  f"diff {v['new_minus_current_interval_score']['mean']:+.3f} {v['new_minus_current_interval_score']['ci90']}")
    print(W["pools_used_by_the_category_rule"])


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("step", choices=["history", "whatif", "report"])
    {"history": history, "whatif": whatif, "report": report}[a.parse_args().step]()
