"""Single-stock track, S3: after the open, does a stock's opening gap on news stick or fade by the close?
(docs/pmrm/SINGLE_STOCK_TRACK.md; DECISIONS #206). No LLM calls.

  python scripts/stock_s3.py

For news that came out before the open or after the previous close (earnings, S0; company news, S0b), the reacting
session splits into the gap (previous close -> open) and the rest of the day (open -> close). Both in units of the stock's
normal daily noise (total returns' sd over the 252 sessions before). Per news type: how much of the day's move the gap
already holds, the slope of the rest of the day on the gap (> 0 the move keeps going after the open, < 0 it fades), and
how often the rest of the day reverses at least half of a big gap. The baseline is ordinary days of the same stocks.
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

import pmrm_audit as A  # noqa: E402
from worsey.data import massive  # noqa: E402

DIR = ROOT / "cache/pmrm/stock"
OUT = ROOT / "docs/pmrm/results"


def ohlc(t, ss):
    rows = massive._get(f"/v2/aggs/ticker/{t}/range/1/day/2010-12-01/{A.AUDIT_END.isoformat()}",
                        {"adjusted": "true", "sort": "asc", "limit": 50000}).get("results") or []
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["t"], unit="ms").dt.normalize()
    return d[~d.index.duplicated()].reindex(ss)[["o", "c"]]


def main():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    earn = pd.read_pickle(DIR / "earnings.pkl")
    news = pd.read_pickle(DIR / "news_events.pkl")
    hh = news.published.str[11:16]
    news = news[(~news.near_earnings) & ((hh < "09:30") | (hh >= "16:00"))]
    ev = pd.concat([earn.assign(kind="Earnings", session=[None] * len(earn))[["ticker", "date", "timing", "kind", "split"]],
                    news.rename(columns={"channel": "kind"})[["ticker", "session", "kind", "split"]]], ignore_index=True)
    # earnings: the reacting session; news: already stored
    sess = []
    for r in ev.itertuples():
        if r.kind == "Earnings":
            d = pd.Timestamp(r.date)
            i = ss.searchsorted(d)
            a = i + 1 if (r.timing == "after close" and i < len(ss) and ss[i] == d) else i
            sess.append(ss[a] if a < len(ss) else pd.NaT)
        else:
            sess.append(pd.Timestamp(r.session))
    ev["sess"] = sess
    ev = ev.dropna(subset=["sess"])
    rows, base = [], []
    rng = np.random.default_rng(1729)
    for t, g in ev.groupby("ticker"):
        p = ohlc(t, ss)
        if p is None:
            continue
        c, o = p["c"].to_numpy(), p["o"].to_numpy()
        ret = c[1:] / c[:-1] - 1
        for r in g.itertuples():
            a = ss.get_loc(r.sess)
            if a < 260 or not (np.isfinite(c[a - 1]) and np.isfinite(o[a]) and np.isfinite(c[a])):
                continue
            sd = np.nanstd(ret[a - 253:a - 1])
            if not sd > 0:
                continue
            rows.append({"kind": r.kind, "split": r.split, "gap": (o[a] / c[a - 1] - 1) / sd, "rest": (c[a] / o[a] - 1) / sd})
        # ordinary days of the same stock (not an event session), a few per stock
        evs = {ss.get_loc(x) for x in g.sess}
        for a in rng.choice(np.arange(300, len(ss) - 1), 12, replace=False):
            if a in evs or not (np.isfinite(c[a - 1]) and np.isfinite(o[a]) and np.isfinite(c[a])):
                continue
            sd = np.nanstd(ret[a - 253:a - 1])
            if sd > 0:
                base.append({"kind": "(ordinary day)", "split": "all", "gap": (o[a] / c[a - 1] - 1) / sd, "rest": (c[a] / o[a] - 1) / sd})
    D = pd.concat([pd.DataFrame(rows), pd.DataFrame(base)], ignore_index=True)
    res = {}
    for k, d in D.groupby("kind"):
        g, rest = d.gap.to_numpy(), d.rest.to_numpy()
        big = np.abs(g) > 2
        day = g + rest
        res[k] = {"events": int(len(d)), "rms_gap": float(np.sqrt((g ** 2).mean())), "rms_rest": float(np.sqrt((rest ** 2).mean())),
                  "share_of_day_variance_in_gap": float((g ** 2).sum() / ((g ** 2).sum() + (rest ** 2).sum())),
                  "slope_rest_on_gap": float((g @ rest) / (g @ g)) if (g @ g) > 0 else None,
                  "big_gaps": int(big.sum()),
                  "big_gap_slope": float((g[big] @ rest[big]) / (g[big] @ g[big])) if big.sum() >= 20 else None,
                  "big_gaps_giving_back_half": float((np.sign(g[big]) * rest[big] < -0.5 * np.abs(g[big])).mean()) if big.sum() >= 20 else None,
                  "big_gaps_extending": float((np.sign(g[big]) * rest[big] > 0).mean()) if big.sum() >= 20 else None,
                  "day_close_same_sign_as_gap_big": float((np.sign(day[big]) == np.sign(g[big])).mean()) if big.sum() >= 20 else None}
    (OUT / "stock_s3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    T = pd.DataFrame(res).T[["events", "rms_gap", "rms_rest", "share_of_day_variance_in_gap", "slope_rest_on_gap", "big_gaps",
                             "big_gap_slope", "big_gaps_giving_back_half", "big_gaps_extending", "day_close_same_sign_as_gap_big"]]
    print(T.sort_values("rms_gap", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
