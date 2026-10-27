"""Build the Plan B runtime pack that Worsey's chat uses (DECISIONS #179): everything the live engine needs, computed
here from the research caches, so the app never reads cache/pmrm/. No LLM calls.

  python scripts/planb_pack.py            # writes src/worsey/planb/pack.json

Contents:
  factors      13 named factors, their live ETF proxies and each proxy's daily sd over 2011-2020 (rates come from
               Treasury ETFs live, FRED yields in research; the proxy-vs-anchor correlation is recorded)
  model        ridge on direction + category -> 13 factor moves over the event window (all 3,876 one-off events),
               plus a least-squares size scale per severity (#175)
  buckets      per factor, the average move of pre-2016 events in each bucket (how code turns an LLM bucket into a
               number, #170 / #176) and how often each bucket happened, by severity
  events       every one-off event: date, title, category, severity, region, targets, direction, 13 factor moves
  releases     CPI / FOMC / jobs / GDP / PCE release days 2021-2026 (today's stock-bond regime, #165) with factor moves
  reversion    per factor, how much of the first reaction is kept at the close and how much wider the close is (#177)
  regime       the latest regime state and labels
"""

import json
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
import planb_b4_threeway as B4  # noqa: E402
import planb_bigtest as BT  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402
from worsey.planb.live import FACTORS, PROXIES  # noqa: E402

OUT = ROOT / "src/worsey/planb/pack.json"
r3 = lambda v: [round(float(x), 3) for x in v]


def main():
    X, uns, src, labels, U8, zU, sd8, dates, sev, Fd, rates, bval8 = BT.evidence()
    X = X[FACTORS]
    ss = X.index

    def resp(r):
        a = ss.searchsorted(pd.Timestamp(r["w1"]["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(r["w1"]["end"])) + 1
        return X.iloc[a:b].to_numpy().sum(0)

    U = np.array([resp(r) for r in uns])
    cats = sorted({r["category"] for r in uns})
    F = np.column_stack([np.ones(len(uns)), np.array([IM.direction_features(e) for e in uns]),
                         np.array([[r["category"] == c for c in cats] for r in uns], float)])
    B = IM.ridge(F, U, 2.0)
    P = F @ B
    scale = {}
    for sv in ("minor", "moderate", "major", "extreme"):
        m = sev == sv
        x, y = P[m].ravel(), U[m].ravel()
        scale[sv] = float(np.clip((x @ y) / max(x @ x, 1e-9), 0, 6)) if m.sum() >= 20 else 1.0

    pre = (dates < "2016-01-01").to_numpy()
    bucket_value, bucket_rates = {}, {}
    for j, f in enumerate(FACTORS):
        z = U[:, j] / U[pre, j].std()
        bins = np.searchsorted(B4.CUTS, z)
        bucket_value[f] = {B4.BUCKETS[b]: (float(U[pre & (bins == b), j].mean()) if (pre & (bins == b)).any() else 0.0) for b in range(5)}
        bucket_rates[f] = {sv: {B4.BUCKETS[b]: round(float(np.mean(bins[sev == sv] == b)), 3) for b in range(5)}
                           for sv in ("minor", "moderate", "major", "extreme") if (sev == sv).sum()}

    # live proxies vs research anchors (same factor, different instruments for rates)
    tick = sorted({t for legs in PROXIES.values() for t, _ in legs})
    R = B1.total_returns(tick, ss, "2003-01-01")
    prox = pd.DataFrame({f: sum(s * R[t] for t, s in legs) for f, legs in PROXIES.items()})
    fit = (ss >= "2011-01-01") & (ss <= "2020-12-31")
    proxy_sd = {f: float(prox.loc[fit, f].std()) for f in FACTORS}
    proxy_corr = {f: round(float(prox.loc[fit, f].corr(X.loc[fit, f])), 3) for f in FACTORS}

    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    rel = []
    for r in lib:
        if r["kind"] == "scheduled" and r["date"] >= "2021-01-01" and r.get("w1"):
            a = ss.searchsorted(pd.Timestamp(r["w1"]["start"])) + 1
            b = ss.searchsorted(pd.Timestamp(r["w1"]["end"])) + 1
            if b > a and b <= len(ss):
                rel.append({"type": r["release"], "date": r["date"], "U": r3(X.iloc[a:b].to_numpy().sum(0))})

    W = json.loads((BT.DIR / "windows.json").read_text(encoding="utf-8"))
    rev = {}
    for f in B4.FACTORS:
        rf, rc = (np.array([v["raw"][f][j] for v in W.values()]) for j in (0, 1))
        k = float((rf @ rc) / (rf @ rf))
        rev[f] = {"kept_at_close": round(k, 3), "fast_to_close_sd": round(float(rf.std() / rc.std()), 3),
                  "gave_back_half": round(float(np.mean((rc - rf) * np.sign(rf) < -0.5 * np.abs(rf))), 3),
                  "flipped_by_close": round(float(np.mean(np.sign(rf) != np.sign(rc))), 3)}

    regs = json.loads((ROOT / "cache/pmrm/b2/regimes.json").read_text(encoding="utf-8"))
    import re
    dated = [k for k in regs if regs[k].get("labels") and re.search(r"-\d{4}-\d{2}-\d{2}$", k)]
    last = max(dated, key=lambda k: (k[-10:], k))
    events = []
    for i, r in enumerate(uns):
        d = r.get("direction") or {}
        events.append({"id": r["id"], "date": r["date"], "title": (r.get("title") or "")[:140], "category": r["category"],
                       "severity": r.get("severity"), "region": r.get("region"), "targeted_sectors": r.get("targeted_sectors") or [],
                       "anticipation": r.get("anticipation"), "direction": {f: d.get(f) for f in IM.DIRECTION_FIELDS},
                       "U": r3(U[i])})
    pack = {"built": str(pd.Timestamp.now().date()), "data_through": str(ss[-1].date()), "factors": FACTORS,
            "proxy_sd": proxy_sd, "proxy_corr_with_research_anchor": proxy_corr,
            "model": {"categories": cats, "B": [r3(row) for row in B], "severity_scale": scale},
            "bucket_cuts_sd": B4.CUTS, "buckets": B4.BUCKETS, "bucket_value": bucket_value, "bucket_rates": bucket_rates,
            "llm_factors": B4.FACTORS, "events": events, "releases": rel, "reversion": rev,
            "regime": {"as_of": last[-10:], **regs[last]}}
    OUT.write_text(json.dumps(pack, separators=(",", ":")), encoding="utf-8")
    print(f"pack: {len(events)} events, {len(rel)} releases, {OUT.stat().st_size / 1e6:.2f} MB; proxy corr {proxy_corr}; "
          f"severity scale {scale}")


if __name__ == "__main__":
    main()
