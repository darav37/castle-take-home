"""Single-stock track, S0b: non-earnings company news, selected by the news itself (Benzinga channels), not by big moves
(docs/pmrm/SINGLE_STOCK_TRACK.md; DECISIONS #205). No LLM calls.

  python scripts/stock_s0b.py fetch    # Benzinga news by channel, 2019-01 to 2026-10, in 10-day windows (cached)
  python scripts/stock_s0b.py build    # events on S&P 500 names (not within a session of an earnings report) + reactions
  python scripts/stock_s0b.py audit    # per news type: how big the move is, does the headline call direction, drift

An event = the first item of a channel naming one or two S&P 500 companies, per company and reacting session. The
reacting session: published before 16:00 ET -> that session (an in-hours item's session also holds the time before it),
after 16:00 -> the next. Reaction = the stock's move beyond the 13 factors (S0's definition), in units of its normal noise.
Direction from the headline where the type carries one (channel or title keywords, fixed before scoring):
  Upgrades +, Downgrades -; FDA approval + / rejection, complete response letter, delay -; guidance raised + / cut -;
  dividend raised + / cut or suspended -; buyback +; equity offering -; initiation at buy + / at sell -.
"""

import argparse
import html
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import planb_b1 as B1  # noqa: E402
from stock_s0 import universe  # noqa: E402
from worsey.data import massive  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.planc import harness as H  # noqa: E402

DIR = ROOT / "cache/pmrm/stock"
OUT = ROOT / "docs/pmrm/results"
CHANNELS = ["FDA", "M&A", "Guidance", "Upgrades", "Downgrades", "Legal", "Offerings", "Buybacks", "Management", "Rumors",
            "Contracts", "Dividends", "Initiation"]
LOOK = 252


def windows():
    d, out = pd.Timestamp("2019-01-01"), []
    while d < pd.Timestamp("2026-10-16"):
        e = min(d + pd.Timedelta(days=10), pd.Timestamp("2026-10-16"))
        out.append((str(d.date()), str(e.date())))
        d = e
    return out


def fetch_one(ch, a, b):
    try:
        return massive._get("/benzinga/v2/news", {"channels": ch, "limit": 1000, "sort": "published.asc",
                                                  "published.gte": a, "published.lte": b}).get("results") or []
    except Exception:  # noqa: BLE001
        return []


def step_fetch():
    jobs = [(ch, a, b) for ch in CHANNELS for a, b in windows()]
    print(f"{len(jobs)} channel-windows", flush=True)
    full = 0
    with ThreadPoolExecutor(8) as ex:
        for k, res in enumerate(ex.map(lambda j: fetch_one(*j), jobs)):
            full += len(res) >= 1000
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    print(f"done; {full} windows hit the 1,000-item page limit (those channels are undercounted there)")


def direction(ch: str, title: str) -> int:
    t = title.lower()
    if ch == "Upgrades":
        return 1
    if ch == "Downgrades":
        return -1
    if ch == "FDA":
        if re.search(r"complete response|\bcrl\b|reject|refus|declin|delay|halt|warning letter", t):
            return -1
        if re.search(r"approv|clearance|cleared|grants? .*designation", t):
            return 1
    if ch == "Guidance":
        if re.search(r"\b(cuts?|lowers?|reduces?|withdraws?|below)\b", t):
            return -1
        if re.search(r"\b(raises?|boosts?|lifts?|increases?|above|reaffirms? .*raises)\b", t):
            return 1
    if ch == "Dividends":
        if re.search(r"\b(cuts?|suspends?|eliminat|reduces?)\b", t):
            return -1
        if re.search(r"\b(raises?|increases?|hikes?|boosts?)\b", t):
            return 1
    if ch == "Buybacks" and re.search(r"buyback|repurchase", t):
        return 1
    if ch == "Offerings" and re.search(r"common stock|shares|equity offering|secondary", t) and not re.search(r"notes|debt|bond", t):
        return -1
    if ch == "Initiation":
        if re.search(r"\b(buy|outperform|overweight)\b", t):
            return 1
        if re.search(r"\b(sell|underperform|underweight)\b", t):
            return -1
    return 0


def step_build():
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    names = set(universe())
    earn = pd.read_pickle(DIR / "earnings.pkl")
    busy = set()
    for r in earn.itertuples():
        i = ss.searchsorted(pd.Timestamp(r.date))
        for j in range(i - 1, i + 3):
            busy.add((r.ticker, j))
    items = []
    for ch in CHANNELS:
        for a, b in windows():
            for x in fetch_one(ch, a, b):
                tk = [t for t in (x.get("tickers") or []) if t in names]
                if not tk or len(x.get("tickers") or []) > 2 or not x.get("published"):
                    continue
                ts = pd.Timestamp(x["published"]).tz_convert("America/New_York")
                for t in tk:
                    items.append((ch, t, ts, html.unescape(x.get("title") or "")))
    I = pd.DataFrame(items, columns=["channel", "ticker", "ts", "title"]).sort_values("ts")
    day = I.ts.dt.tz_localize(None).dt.normalize()
    after = I.ts.dt.hour >= 16
    idx = ss.searchsorted(day)
    I["a"] = np.where(after.to_numpy() & (idx < len(ss)) & (ss[np.minimum(idx, len(ss) - 1)] == day.to_numpy()), idx + 1, idx)
    I = I[(I.a > LOOK + 2) & (I.a + 6 < len(ss))].drop_duplicates(["channel", "ticker", "a"])
    I["near_earnings"] = [(t, a) in busy for t, a in zip(I.ticker, I.a)]
    proxy_t = sorted({t for legs in live.PROXIES.values() for t, _ in legs})
    Fz = live.factor_returns(B1.total_returns(proxy_t, ss, "1990-01-01")).reindex(columns=live.FACTORS).to_numpy()
    S = B1.total_returns(sorted(set(I.ticker)), ss, "2010-12-01")
    rows = []
    for r in I.itertuples():
        if r.ticker not in S:
            continue
        x = S[r.ticker].to_numpy()
        lb = slice(r.a - 1 - LOOK, r.a - 1)
        ok = np.isfinite(x[lb]) & np.isfinite(Fz[lb]).all(1)
        if ok.sum() < 150 or not np.isfinite(x[r.a:r.a + 6]).all():
            continue
        A = np.column_stack([np.ones(ok.sum()), Fz[lb][ok]])
        c = np.linalg.lstsq(A, x[lb][ok], rcond=None)[0]
        sd = float((x[lb][ok] - A @ c).std())
        res = x[r.a:r.a + 6] - np.nan_to_num(Fz[r.a:r.a + 6]) @ c[1:]
        d = str(ss[r.a].date())
        rows.append({"channel": r.channel, "ticker": r.ticker, "published": str(r.ts), "session": d, "split": H.split_of(d),
                     "title": r.title[:200], "near_earnings": r.near_earnings, "sign": direction(r.channel, r.title),
                     "day1_z": float(res[0] / sd), "next5_z": float(res[1:6].sum() / (sd * np.sqrt(5))), "noise_day": sd})
    E = pd.DataFrame(rows)
    E.to_pickle(DIR / "news_events.pkl")
    print(f"news events: {len(E)} ({(~E.near_earnings).sum()} away from earnings); by channel "
          f"{E[~E.near_earnings].channel.value_counts().to_dict()}")


def step_audit():
    E = pd.read_pickle(DIR / "news_events.pkl")
    res = {"events": int(len(E)), "near_earnings_excluded": int(E.near_earnings.sum())}
    X = E[~E.near_earnings].copy()
    hh = X.published.str[11:16]
    # timing-clean: an item published before the open or after the close is news the WHOLE reacting session follows;
    # an in-hours item's session also holds the hours before it (and analysts often write after a move)
    X["clean_timing"] = (hh < "09:30") | (hh >= "16:00")
    per = {}
    for ch, d in X.groupby("channel"):
        z = d.day1_z
        signed = d[d.sign != 0]
        big = signed[signed.day1_z.abs() > 2]
        sg = np.sign(d.day1_z)
        row = {"events": int(len(d)), "rms_day1": float(np.sqrt((z ** 2).mean())), "median_abs_day1": float(z.abs().median()),
               "share_over_2x": float((z.abs() > 2).mean()), "share_over_3x": float((z.abs() > 3).mean()),
               "signed_events": int(len(signed)),
               "direction_hit_signed": float((np.sign(signed.day1_z) == signed.sign).mean()) if len(signed) else None,
               "direction_hit_signed_moves_over_2x": float((np.sign(big.day1_z) == big.sign).mean()) if len(big) else None,
               "mean_day1_in_headline_direction": float((signed.sign * signed.day1_z).mean()) if len(signed) else None,
               "drift_next5_in_day1_direction": float((sg * d.next5_z).mean()),
               "drift_next5_in_headline_direction": float((signed.sign * signed.next5_z).mean()) if len(signed) else None,
               "by_split_direction_hit": {sp: float((np.sign(g.day1_z) == g.sign).mean()) for sp, g in signed.groupby("split")} if len(signed) else {}}
        c = d[d.clean_timing]
        cs = c[c.sign != 0]
        cb = cs[cs.day1_z.abs() > 2]
        row["clean_timing"] = {"events": int(len(c)), "rms_day1": float(np.sqrt((c.day1_z ** 2).mean())) if len(c) else None,
                               "median_abs_day1": float(c.day1_z.abs().median()) if len(c) else None,
                               "p90_abs_day1": float(c.day1_z.abs().quantile(0.9)) if len(c) else None,
                               "share_over_2x": float((c.day1_z.abs() > 2).mean()) if len(c) else None,
                               "direction_hit_signed": float((np.sign(cs.day1_z) == cs.sign).mean()) if len(cs) else None,
                               "direction_hit_signed_moves_over_2x": float((np.sign(cb.day1_z) == cb.sign).mean()) if len(cb) else None,
                               "direction_hit_2026": float((np.sign(cs[cs.split == "test"].day1_z) == cs[cs.split == "test"].sign).mean())
                               if (cs.split == "test").sum() >= 20 else None,
                               "drift_next5_in_headline_direction": float((cs.sign * cs.next5_z).mean()) if len(cs) else None,
                               "drift_next5_in_day1_direction": float((np.sign(c.day1_z) * c.next5_z).mean()) if len(c) else None}
        per[ch] = row
    res["by_channel"] = per
    res["normal_day_reference"] = {"rms": 1.0, "share_over_2x": 0.046, "share_over_3x": 0.003}
    (OUT / "stock_s0b.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    t = pd.DataFrame(per).T[["events", "rms_day1", "share_over_2x", "signed_events", "direction_hit_signed",
                             "direction_hit_signed_moves_over_2x", "mean_day1_in_headline_direction", "drift_next5_in_headline_direction"]]
    print(t.sort_values("rms_day1", ascending=False).round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "build", "audit"])
    a = ap.parse_args()
    {"fetch": step_fetch, "build": step_build, "audit": step_audit}[a.step]()
