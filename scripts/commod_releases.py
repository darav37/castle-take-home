"""Commodity releases and weather: do WASDE, EIA inventory reports and Gulf hurricanes move their markets beyond normal
noise, and does the surprise call direction? (DECISIONS #212). No LLM calls.

  python scripts/commod_releases.py fetch    # WASDE dates (USDA ESMIS archive), EIA weekly history, NOAA HURDAT2; minute bars
  python scripts/commod_releases.py score

Releases (minute bars, US Eastern):
  EIA Weekly Petroleum Status Report   Wednesdays 10:30 (Thursday after a Monday holiday)    USO
  EIA Weekly Natural Gas Storage       Thursdays 10:30 (Friday after a Monday holiday)        UNG
  USDA WASDE                           the archive's release time (12:00)                     DBA, CORN, SOYB, WEAT
Window: 5 minutes before -> 60 minutes after, and -> the close. Control: the same clock window one session earlier (not a
release day), so "bigger than normal" is measured, not assumed.
Surprise (EIA only; analysts' consensus isn't free): the week's stock change minus the average change in the same week
of the 5 previous years. A bigger build than the seasonal norm should push the price down.
Hurricanes (daily): US Gulf landfalls (HURDAT2 landfall records in the Gulf coast box, >= 64 kt), 2007-2024; the
2-session move of UNG, USO, UGA, XLE, KIE (insurers), ITB (homebuilders) after landfall, in units of each one's normal
2-session noise.
"""

import argparse
import io
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import planb_b1 as B1  # noqa: E402
from planc_c1d_fetch import minute_rows  # noqa: E402
from worsey.planc import harness as H  # noqa: E402

DIR = ROOT / "cache/pmrm/commod"
OUT = ROOT / "docs/pmrm/results"
UA = {"User-Agent": "Mozilla/5.0"}
SRC = {"crude.xls": "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls", "gas.xls": "https://ir.eia.gov/ngs/ngshistory.xls",
       "hurdat2.txt": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"}


def sessions():
    return pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))["anchors"].index


def wasde_dates() -> list[pd.Timestamp]:
    f = DIR / "wasde_dates.json"
    if f.exists():
        return [pd.Timestamp(x) for x in json.loads(f.read_text())]
    out = set()
    for page in range(0, 60):            # USDA ESMIS (the Cornell address redirects there and drops the page number)
        r = requests.get("https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates",
                         params={"page": page}, headers=UA, timeout=60)
        found = re.findall(r'datetime="(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"', r.text)
        if not found:
            break
        out |= set(found)
        if min(found) < "2008-01-01":
            break
    # the archive stamps "12:00:00Z" but WASDE is released at 12:00 US Eastern: keep the date, set noon ET
    ts = sorted(pd.Timestamp(x[:10]) + pd.Timedelta(hours=12) for x in out)
    f.write_text(json.dumps([str(t) for t in ts]))
    return ts


def eia_weekly(kind: str) -> pd.Series:
    if kind == "crude":
        d = pd.read_excel(DIR / "crude.xls", sheet_name="Data 1", header=2)
        s = pd.Series(d.iloc[:, 1].to_numpy(float), index=pd.to_datetime(d.iloc[:, 0])).sort_index()
        return s.diff().dropna()                                      # weekly change, thousand barrels
    d = pd.read_excel(DIR / "gas.xls", sheet_name="weekly_net_changes", header=None)
    d = d[pd.to_datetime(d[0], errors="coerce").notna()]
    num = d.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    return pd.Series(num.sum(axis=1).to_numpy(float), index=pd.to_datetime(d[0])).sort_index()   # Bcf, lower 48


def surprise(chg: pd.Series) -> pd.Series:
    """Change minus the mean change in the same ISO week of the 5 previous years."""
    wk = chg.index.isocalendar().week.to_numpy()
    yr = chg.index.year.to_numpy()
    out = {}
    for d, w, y, v in zip(chg.index, wk, yr, chg.to_numpy()):
        prev = chg[(wk == w) & (yr < y) & (yr >= y - 5)]
        if len(prev) >= 3:
            out[d] = v - prev.mean()
    return pd.Series(out)


def release_events(ss) -> pd.DataFrame:
    rows = []
    for kind, lag, ticks in (("crude", 5, ["USO"]), ("gas", 6, ["UNG"])):
        sur = surprise(eia_weekly(kind))
        for wk_end, s_ in sur.items():
            d = wk_end + pd.Timedelta(days=lag)
            monday = wk_end + pd.Timedelta(days=3)
            if monday not in ss:                                       # Monday holiday: one day later
                d += pd.Timedelta(days=1)
            if d not in ss or d < pd.Timestamp("2008-06-01"):
                continue
            for t in ticks:
                rows.append({"release": "EIA crude stocks" if kind == "crude" else "EIA gas storage", "ticker": t,
                             "time": d + pd.Timedelta(hours=10, minutes=30), "surprise": float(s_)})
    for ts in wasde_dates():
        if ts.normalize() in ss and ts >= pd.Timestamp("2008-06-01"):
            for t in ("DBA", "CORN", "SOYB", "WEAT"):
                rows.append({"release": "USDA WASDE", "ticker": t, "time": ts, "surprise": np.nan})
    E = pd.DataFrame(rows)
    i = ss.searchsorted(E.time.dt.normalize())
    E["control_day"] = [ss[k - 1] for k in i]
    return E


def mframe(t, day):
    rows = minute_rows(t, str(pd.Timestamp(day).date()))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return d


def window_moves(t, day, hhmm: tuple[int, int]):
    m = mframe(t, day)
    if m is None:
        return None
    t0 = pd.Timestamp(day).normalize() + pd.Timedelta(hours=hhmm[0], minutes=hhmm[1])
    a = m[m.index <= t0 - pd.Timedelta(minutes=5)]
    b = m[m.index <= t0 + pd.Timedelta(minutes=60)]
    c = m[m.index < pd.Timestamp(day).normalize() + pd.Timedelta(hours=16)]
    open_ = pd.Timestamp(day).normalize() + pd.Timedelta(hours=9, minutes=30)
    # thinly traded ETFs (CORN, SOYB, WEAT): the last trade that morning before the release, and a trade after it
    if a.empty or b.empty or c.empty or a.index[-1] < open_ or b.index[-1] <= t0:
        return None
    p0 = a.c.iloc[-1]
    return b.c.iloc[-1] / p0 - 1, c.c.iloc[-1] / p0 - 1


def step_fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SRC.items():
        f = DIR / name
        if not f.exists():
            r = requests.get(url, headers=UA, timeout=120)
            r.raise_for_status()
            f.write_bytes(r.content)
    print(f"WASDE releases: {len(wasde_dates())}", flush=True)
    E = release_events(sessions())
    jobs = sorted({(t, str(d.date())) for t, d in zip(E.ticker, E.time.dt.normalize())} | {(t, str(d.date())) for t, d in zip(E.ticker, E.control_day)})
    print(f"{len(E)} release-ticker events; {len(jobs)} minute days", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for k, _ in enumerate(ex.map(lambda j: minute_rows(*j), jobs)):
            if k % 1000 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    print("done", flush=True)


def hurricanes() -> list[tuple[pd.Timestamp, str]]:
    out, name = [], None
    for line in (DIR / "hurdat2.txt").read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        if p[0].startswith("AL"):
            name = p[1]
            continue
        if len(p) < 8 or p[2] != "L":
            continue
        d = pd.Timestamp(p[0])
        lat = float(p[4][:-1]) * (1 if p[4].endswith("N") else -1)
        lon = float(p[5][:-1]) * (-1 if p[5].endswith("W") else 1)
        if 2007 <= d.year <= 2024 and 25 <= lat <= 31.5 and -98 <= lon <= -81.5 and int(p[6]) >= 64:
            out.append((d, name))
    seen, res = set(), []
    for d, n in out:                       # one event per storm (first Gulf landfall)
        if (n, d.year) not in seen:
            seen.add((n, d.year))
            res.append((d, n))
    return res


def step_score():
    ss = sessions()
    E = release_events(ss)
    rows = []
    for r in E.itertuples():
        hhmm = (r.time.hour, r.time.minute)
        ev = window_moves(r.ticker, r.time.normalize(), hhmm)
        ct = window_moves(r.ticker, r.control_day, hhmm)
        if ev is None or ct is None:
            continue
        rows.append({"release": r.release, "ticker": r.ticker, "date": str(r.time.date()), "split": H.split_of(str(r.time.date())),
                     "surprise": r.surprise, "hour": ev[0], "to_close": ev[1], "c_hour": ct[0], "c_to_close": ct[1]})
    R = pd.DataFrame(rows)
    R.to_pickle(DIR / "release_rows.pkl")
    res = {}
    for (rel, t), d in R.groupby(["release", "ticker"]):
        out = {"events": int(len(d)), "first": d.date.min()}
        for w, cw in (("hour", "c_hour"), ("to_close", "c_to_close")):
            sd_c = d[cw].std()
            z = d[w] / sd_c
            out[w] = {"size_vs_control_rms": float(np.sqrt((d[w] ** 2).mean()) / np.sqrt((d[cw] ** 2).mean())),
                      "share_over_2x_control_sd": float((z.abs() > 2).mean()), "control_share_over_2x": float((d[cw].abs() / sd_c > 2).mean())}
            if d.surprise.notna().any():
                x = d[d.surprise.notna() & (d.surprise != 0)]
                big = x[(x[w].abs() / sd_c) > 2]
                out[w]["direction_hit_bearish_on_build"] = float((np.sign(x[w]) == -np.sign(x.surprise)).mean())
                out[w]["direction_hit_on_big_moves"] = float((np.sign(big[w]) == -np.sign(big.surprise)).mean()) if len(big) else None
                out[w]["rank_corr_surprise_move"] = float(x[["surprise", w]].corr(method="spearman").iloc[0, 1])
                out[w]["by_split_direction_hit"] = {sp: float((np.sign(g[w]) == -np.sign(g.surprise)).mean()) for sp, g in x.groupby("split")}
                out[w]["by_split_n"] = x.split.value_counts().to_dict()
        res[f"{rel} | {t}"] = out
        print(f"{rel:18s} {t:5s} n={len(d):4d} | first hour: size x{out['hour']['size_vs_control_rms']:.2f}, >2x {out['hour']['share_over_2x_control_sd']:.0%} "
              f"(control {out['hour']['control_share_over_2x']:.0%})"
              + (f", direction {out['hour']['direction_hit_bearish_on_build']:.0%} (big {out['hour']['direction_hit_on_big_moves'] or float('nan'):.0%}), "
                 f"rank corr {out['hour']['rank_corr_surprise_move']:+.2f}, by split {json.dumps({k: round(v, 2) for k, v in out['hour']['by_split_direction_hit'].items()})}"
                 if 'direction_hit_bearish_on_build' in out['hour'] else "")
              + f" | to close: size x{out['to_close']['size_vs_control_rms']:.2f}", flush=True)
    # hurricanes (daily)
    tick = ["UNG", "USO", "UGA", "XLE", "KIE", "ITB", "SPY"]
    Rd = B1.total_returns(tick, ss, "2003-01-01")
    hr = []
    for d, n in hurricanes():
        a = ss.searchsorted(d)
        if a < 260 or a + 2 > len(ss):
            continue
        row = {"date": str(d.date()), "storm": n}
        for t in tick[:-1]:
            x = Rd[t].to_numpy()
            sd = np.nanstd(x[a - 253:a - 1]) * np.sqrt(2)
            row[t] = float(np.nansum(x[a:a + 2]) / sd) if sd > 0 else np.nan
        hr.append(row)
    Hd = pd.DataFrame(hr)
    res["hurricanes"] = {"landfalls": int(len(Hd)), "storms": Hd.storm.tolist(),
                         "by_asset": {t: {"mean_z": float(Hd[t].mean()), "rms_z": float(np.sqrt((Hd[t] ** 2).mean())),
                                          "share_over_2x": float((Hd[t].abs() > 2).mean()), "share_up": float((Hd[t] > 0).mean())}
                                      for t in tick[:-1]}}
    print(f"hurricanes: {len(Hd)} Gulf landfalls 2007-2024 | " + " | ".join(f"{t} mean {v['mean_z']:+.2f} rms {v['rms_z']:.2f} up {v['share_up']:.0%}"
                                                                         for t, v in res["hurricanes"]["by_asset"].items()))
    (OUT / "commod_releases.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "score"])
    a = ap.parse_args()
    step_fetch() if a.step == "fetch" else step_score()
