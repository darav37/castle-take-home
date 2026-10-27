"""Is direction a coin flip everywhere, or only on small moves? (DECISIONS #188, #189) No LLM calls.

  python scripts/planc_direction_by_size.py

Direction hit rate (sign of the forecast's middle vs the sign of the real move) by the real move's size in units of the
target's normal noise over the window (|z| bins). If the explanation in #188 is right, small moves are a coin flip and
big moves are called right more often. Sources (all existing runs, nothing refitted):
  engine     today's engine (rung 1 of C0) on validation 2023-25 and test 2026: 13 factors, 96 ETF residuals, 12 books
  naive LLM  #186's position-level answers (19 holdings, 92 events of 2026) and #185's portfolio answers (12 books)
Also: the theoretical best hit rate for an effect of size d (in noise units) known perfectly, Phi(d).
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

BINS = [0, 0.5, 1, 2, 3, np.inf]
LABELS = ["<0.5", "0.5-1", "1-2", "2-3", ">3"]


def table(z: np.ndarray, hit: np.ndarray) -> dict:
    out = {}
    b = pd.cut(np.abs(z), BINS, labels=LABELS, right=False)
    for lab in LABELS:
        m = np.asarray(b == lab)
        n = int(m.sum())
        if n:
            p = float(hit[m].mean())
            out[lab] = {"n": n, "hit": round(p, 3), "pm_90": round(1.645 * np.sqrt(p * (1 - p) / n), 3)}
    return out


def main():
    res = {}
    S = pd.read_pickle(ROOT / "cache/pmrm/planc/c0_rungs.pkl")
    E = S[(S.rung == "1 engine") & (S["median"].abs() > 1e-12)]
    for kind in ("factor", "etf residual", "book"):
        for split in ("validation", "test"):
            d = E[(E.kind == kind) & (E.split == split)]
            res[f"engine | {kind} | {split}"] = table((d.y / d.noise_sd).to_numpy(), (np.sign(d["median"]) == np.sign(d.y)).to_numpy())
    # naive LLM, positions (#186)
    import planb_b1 as B1
    from planb_naive_vs_system import events
    naive = json.loads((ROOT / "cache/pmrm/naive_vs_system/naive_positions.json").read_text(encoding="utf-8"))
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    H = sorted({t for v in naive.values() for t in v["answers"]})
    R = B1.total_returns(H, ss, "2003-01-01")
    evs = {e["id"]: e for e in events()}
    z, hit = [], []
    for k, v in naive.items():
        e = evs.get(k)
        if not e:
            continue
        a = ss.searchsorted(pd.Timestamp(e["w1"]["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(e["w1"]["end"])) + 1
        for t, ans in v["answers"].items():
            if t not in R or ans.get("expected_pct") in (None, 0):
                continue
            y = float(np.prod(1 + R[t].iloc[a:b]) - 1)
            sd = float(R[t].iloc[a - 505:a - 1].std() * np.sqrt(b - a))
            if np.isfinite(y) and sd > 0:
                z.append(y / sd)
                hit.append(np.sign(float(ans["expected_pct"])) == np.sign(y))
    res["naive LLM | positions | 2026"] = table(np.array(z), np.array(hit))
    # naive LLM and engine, books (#185)
    Bk = pd.read_pickle(ROOT / "cache/pmrm/naive_vs_system/rows.pkl")
    D = pickle.load(open(ROOT / "cache/pmrm/planc/c0_events.pkl", "rb"))
    noise = {(r["id"], b): v["noise"] * 100_000 for r in D["events"] for b, v in r["book"].items()}
    Bk = Bk.assign(sd=[noise.get((e, b), np.nan) for e, b in zip(Bk.event, Bk.book)]).dropna(subset=["sd"])
    nz = Bk[Bk.nai_mid != 0]
    res["naive LLM | books | 2026"] = table((nz.y / nz.sd).to_numpy(), (np.sign(nz.nai_mid) == np.sign(nz.y)).to_numpy())
    from scipy.stats import norm
    res["best possible with a perfectly known effect of size d (noise units)"] = {f"d={d}": round(float(norm.cdf(d)), 3)
                                                                                   for d in (0.1, 0.3, 0.5, 1, 2)}
    (ROOT / "docs/pmrm/results/direction_by_size.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k)
        print("   " + " | ".join(f"{b}: {x['hit']:.0%} (n={x['n']}, ±{x['pm_90']:.0%})" if isinstance(x, dict) else f"{b}: {x:.0%}"
                             for b, x in v.items()))


if __name__ == "__main__":
    main()
