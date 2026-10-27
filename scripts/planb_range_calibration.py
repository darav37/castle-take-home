"""Are the shipped ranges calibrated? (task 2026-10-22 §3; DECISIONS #182). No LLM calls.

  python scripts/planb_range_calibration.py [--width-scale K]

For every one-off event, scored out of sample:
  2026        cutoff 2026-01-01 (all 92 events of 2026)
  2016-2025   rolling yearly refits: each year's events scored with the model refitted on events before that year
Each event uses its stored schema labels (category, severity, direction, region, targets) and the shipped distribution
(live.scenario_draws, the function price() uses), on each of the 11 books of #178 with loadings from the 504 sessions
before the event, and on each factor. End-of-day ranges on all events; first-hours ranges on the timed events (#177),
with the first-reaction scaling estimated only from earlier timed events.

Scores: share inside the middle half (target 50%), inside the 9-in-10 range (90%), below the 1-in-20 bad case (5%) and
above the top 5% (5%); the mean signed error of the middle in units of the range's spread (bias); 90% error bars from a
bootstrap over events (books on the same event are not independent); by severity; CRPS against a predict-zero baseline
with the book's normal noise. --width-scale tests a proposed width fix (never tuned on the period it is scored on).
"""

import argparse
import json
import pickle
import sys
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import planb_b1 as B1  # noqa: E402
from planb_naive_books import BOOKS  # noqa: E402
from worsey.planb import live  # noqa: E402

LOOKBACK = 504
OUT = ROOT / "docs/pmrm/results/range_calibration.json"


def crps_sample(x: np.ndarray, y: float) -> float:
    x = np.sort(x)
    n = len(x)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2)


def crps_normal(sd: float, y: float) -> float:
    z = y / sd
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    phi = exp(-z * z / 2) / sqrt(2 * pi)
    return float(sd * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi)))


def reversion_before(W: dict, cutoff: str) -> dict | None:
    """#177's first-reaction stats from timed events before the cutoff only (as scripts/planb_pack.py computes them)."""
    V = [v for v in W.values() if v["t0"][:10] < cutoff]
    if len(V) < 30:
        return None
    out = {}
    for f in V[0]["raw"]:
        rf, rc = (np.array([v["raw"][f][j] for v in V]) for j in (0, 1))
        out[f] = {"kept_at_close": float((rf @ rc) / (rf @ rf)), "fast_to_close_sd": float(rf.std() / rc.std()),
                  "gave_back_half": 0.0, "flipped_by_close": 0.0}
    return out


def main(width_scale: float):
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    live.set_cutoff(None)
    full = [dict(e) for e in live.pack()["events"]]
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8")) if r["kind"] == "unscheduled"}
    W = json.loads((ROOT / "cache/pmrm/bigtest/windows.json").read_text(encoding="utf-8"))
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    tick = sorted(set(proxy_t) | {t for w, _ in BOOKS.values() for t in w})
    R = B1.total_returns(tick, ss, "2003-01-01")
    Fz = live.factor_returns(R)
    F = live.FACTORS
    book_ret = pd.DataFrame({b: sum(wt * R[t] for t, wt in w.items()) for b, (w, _) in BOOKS.items()})
    psd = live.pack()["proxy_sd"]
    rows, frows, hrows = [], [], []
    periods = [(f"{y}-01-01", f"{y + 1}-01-01", "2026" if y == 2026 else "2016-2025") for y in range(2016, 2027)]
    for c0, c1, label in periods:
        live.set_cutoff(c0)
        p = live.pack()
        rev = reversion_before(W, c0)
        if rev:
            p["reversion"] = rev
        for e in full:
            if not (c0 <= e["date"] < c1) or e["id"] not in lib:
                continue
            w1 = lib[e["id"]]["w1"]
            a = ss.searchsorted(pd.Timestamp(w1["start"])) + 1
            b = ss.searchsorted(pd.Timestamp(w1["end"])) + 1
            if a < LOOKBACK + 2 or b <= a or b > len(ss):
                continue
            spec = live._spec_ok(e)
            base = live.scenario_draws(spec, np.zeros(len(F)), 0.0)
            draw = base["draw"]
            med = np.median(draw, axis=0)
            # factors (end of day)
            for j, f in enumerate(F):
                dist = base["f"][j] + width_scale * (draw[:, j] - med[j])
                y = e["U"][j]
                frows.append({"event": e["id"], "period": label, "sev": spec["severity"], "factor": f, "pit": float(np.mean(dist < y)),
                              "z_err": float((y - base["f"][j]) / (dist.std() or 1))})
            # first hours, timed events (2023+ cutoffs have earlier timed events to estimate the scaling)
            if rev and e["id"] in W:
                for f, raw in W[e["id"]]["raw"].items():
                    j = F.index(f)
                    dist = base["fast_c"][j] + width_scale * base["fast_sc"][j] * (draw[:, j] - med[j])
                    y = raw[0] / psd[f]
                    hrows.append({"event": e["id"], "period": label, "sev": spec["severity"], "factor": f, "pit": float(np.mean(dist < y))})
            # books
            Fl = Fz.iloc[a - 1 - LOOKBACK:a - 1]
            for bk in BOOKS:
                yl = book_ret[bk].iloc[a - 1 - LOOKBACK:a - 1]
                dd = pd.concat([yl.rename("y"), Fl], axis=1).dropna()
                yw = book_ret[bk].iloc[a:b]
                if len(dd) < 250 or not np.isfinite(yw).all():
                    continue
                used = [f for f in F if f in dd.columns]
                A = np.column_stack([np.ones(len(dd)), dd[used].to_numpy()])
                coef = np.linalg.lstsq(A, dd["y"].to_numpy(), rcond=None)[0]
                resid = dd["y"].to_numpy() - A @ coef
                beta = pd.Series(coef[1:], index=used).reindex(F).fillna(0.0).to_numpy()
                d = live.scenario_draws(spec, beta, float(resid.std()))
                mid = float(beta @ d["f"])
                end = mid + width_scale * (d["end"] - mid)
                y = float(np.prod(1 + yw) - 1)
                sd_norm = float(dd["y"].std() * np.sqrt(b - a))
                rows.append({"event": e["id"], "period": label, "sev": spec["severity"], "book": bk, "pit": float(np.mean(end < y)),
                             "z_err": float((y - mid) / (end.std() or 1)), "crps": crps_sample(end[::4], y),
                             "crps_zero": crps_normal(sd_norm, y)})
        print(f"{c0[:4]}: {sum(r['period'] == label for r in rows)} book rows so far ({label})", flush=True)
    live.set_cutoff(None)
    D, Fd, H = pd.DataFrame(rows), pd.DataFrame(frows), pd.DataFrame(hrows)
    rng = np.random.default_rng(1729)

    def cover(d):
        pit = d["pit"].to_numpy()
        return {"middle_half": float(np.mean((pit >= 0.25) & (pit <= 0.75))), "nine_in_ten": float(np.mean((pit >= 0.05) & (pit <= 0.95))),
                "below_bad_case": float(np.mean(pit < 0.05)), "above_top": float(np.mean(pit > 0.95))}

    def summary(d, with_crps=False):
        ev = d["event"].unique()
        g = {k: v for k, v in d.groupby("event")}
        point = cover(d)
        bs = [cover(pd.concat([g[k] for k in rng.choice(ev, len(ev))])) for _ in range(300)]
        out = {"events": int(len(ev)), "rows": int(len(d))}
        for k, v in point.items():
            out[k] = [round(v, 3), round(float(np.quantile([b[k] for b in bs], 0.05)), 3), round(float(np.quantile([b[k] for b in bs], 0.95)), 3)]
        if "z_err" in d:
            out["mean_signed_error_sd"] = round(float(d["z_err"].mean()), 3)
        if with_crps:
            out["crps_skill_vs_zero"] = round(float(1 - d["crps"].sum() / d["crps_zero"].sum()), 3)
        lo, hi = point["below_bad_case"], point["above_top"]
        out["diagnosis"] = ("too narrow (misses on both sides)" if lo > 0.08 and hi > 0.08 else
                            "shifted down (middle too high)" if lo > 0.08 and lo > 2 * hi else
                            "shifted up (middle too low)" if hi > 0.08 and hi > 2 * lo else
                            "too wide" if point["nine_in_ten"] > 0.95 and point["middle_half"] > 0.6 else "calibrated")
        return out

    res = {"width_scale": width_scale}
    for per in ("2026", "2016-2025"):
        r = {"books_end_of_day": summary(D[D.period == per], True),
             "books_by_severity": {sv: summary(D[(D.period == per) & (D.sev == sv)], True) for sv in live.SEVERITIES
                                   if ((D.period == per) & (D.sev == sv)).any()},
             "books_by_book": {bk: summary(D[(D.period == per) & (D.book == bk)], True) for bk in BOOKS},
             "factors_end_of_day": {f: summary(Fd[(Fd.period == per) & (Fd.factor == f)]) for f in F},
             "first_hours_factors": {f: summary(H[(H.period == per) & (H.factor == f)]) for f in sorted(H.factor.unique())
                                     if ((H.period == per) & (H.factor == f)).sum() >= 10} if len(H) else {}}
        res[per] = r
        b = r["books_end_of_day"]
        print(f"\n{per}: books end of day ({b['events']} events, {b['rows']} rows): middle half {b['middle_half']}, 9-in-10 "
              f"{b['nine_in_ten']}, below bad case {b['below_bad_case']}, above top {b['above_top']}, bias {b['mean_signed_error_sd']} sd, "
              f"CRPS skill vs zero {b['crps_skill_vs_zero']} -> {b['diagnosis']}")
        for sv, x in r["books_by_severity"].items():
            print(f"   {sv:9s} {x['events']:4d} ev: half {x['middle_half'][0]:.2f} 9/10 {x['nine_in_ten'][0]:.2f} below {x['below_bad_case'][0]:.3f} "
                  f"above {x['above_top'][0]:.3f} bias {x['mean_signed_error_sd']:+.2f} crps {x['crps_skill_vs_zero']:+.3f} {x['diagnosis']}")
        for f, x in r["factors_end_of_day"].items():
            print(f"   factor {f:24s} half {x['middle_half'][0]:.2f} 9/10 {x['nine_in_ten'][0]:.2f} below {x['below_bad_case'][0]:.3f} "
                  f"above {x['above_top'][0]:.3f} bias {x['mean_signed_error_sd']:+.2f} {x['diagnosis']}")
        for f, x in r["first_hours_factors"].items():
            print(f"   first hours {f:20s} n={x['events']} half {x['middle_half'][0]:.2f} 9/10 {x['nine_in_ten'][0]:.2f} "
                  f"below {x['below_bad_case'][0]:.3f} above {x['above_top'][0]:.3f} {x['diagnosis']}")
    key = "base" if width_scale == 1.0 else f"width_{width_scale}"
    allres = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    allres[key] = res
    OUT.write_text(json.dumps(allres, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-scale", type=float, default=1.0)
    main(ap.parse_args().width_scale)
