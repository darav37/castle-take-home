"""Part 2, tier 1: is there a longshot bias in prediction-market odds, and how big? (docs/event_pipeline/probabilities.md
step 1; DECISIONS #227.) Code only, $0.

  python scripts/part2_longshot.py fetch   # resolved markets + their price on each snapshot date (cached; resumable)
  python scripts/part2_longshot.py fit     # calibration curve, bias test, correction -> results/part2_longshot.md

Sample (frozen rules):
  Kalshi      settled markets in the in-scope series (the build's categories; daily, weekly, hourly and 15-minute series
              dropped: those are mostly price-level ladders on an asset), closing 2023-01-01 .. 2026-06-30, volume >= 500
              contracts, at most 4 markets per event (ladders otherwise dominate), YES/NO result only
  Polymarket  closed markets of the top 2,000 events by volume under each topic tag the universe uses (#224), ending in the same window, volume
              >= $10k, at most 4 per event, resolved cleanly to 1/0
  Snapshots   calendar dates, the 1st and 15th of each month in the window: every market still open on date D gives one
              row (its price at D, its eventual outcome). "Still open at D" is known at D, so the sample does not depend
              on the outcome. (A first version priced markets "7 days before close"; close is when the market resolved,
              which comes early for "by date" markets that resolve YES, so the snapshot time depended on the outcome and
              biased the curve. Scheduled end dates don't fix it: Polymarket's endDate is often the whole event's last
              date, and Kalshi moves close_time when a market closes early.)
  Price       Kalshi: the daily candle ending at or before D (within 3 days): its last trade if it traded that day, else
              the bid/ask mid if the spread is <= 10 cents. Polymarket: the CLOB price-history point (12-hour resolution,
              the finest it serves for closed markets) at or before D, within 2 days. No usable price: no row.
Fit (fit):
  calibration by price bin (realized YES frequency vs mean price, Wilson intervals, and a bootstrap clustered by event);
  isotonic regression (pool-adjacent-violators) of outcome on price as the correction; the bias is "significant" in a
  bin when the clustered 95% interval for (realized - price) excludes 0. The correction is applied only in bins where it is.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worsey.data import kalshi as K  # noqa: E402
from worsey.data import polymarket as P  # noqa: E402
from worsey.data.http import get_json  # noqa: E402
from worsey.scenario import discovery as SD  # noqa: E402

DIR = ROOT / "cache/pmrm/part2"
OUT = ROOT / "docs/pmrm/results"
START, END = "2023-01-01", "2026-06-30"
DROP_FREQ = {"daily", "weekly", "hourly", "fifteen_min"}
PER_EVENT = 4
THREADS = 12
SNAPS = [f"{y}-{mo:02d}-{d:02d}" for y in range(2023, 2027) for mo in range(1, 13) for d in (1, 15) if START <= f"{y}-{mo:02d}-{d:02d}" <= END]


def _snap_ts(d: str) -> float:
    """A snapshot is taken at 16:00 New York time (20:00 UTC; the hour is immaterial at daily resolution)."""
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() + 20 * 3600


def _sample(rows: list[dict], key: str) -> list[dict]:
    """At most PER_EVENT markets per event, chosen by a stable hash of the ticker (not by price or outcome)."""
    by = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    out = []
    for ev, rs in by.items():
        rs.sort(key=lambda r: hashlib.md5(str(r["id"]).encode()).hexdigest())
        out += rs[:PER_EVENT]
    return out


# ---------------------------------------------------------------- Kalshi

def kalshi_markets() -> list[dict]:
    ser = [s for s in K.series() if s.get("category") in SD.IN_SCOPE_CATEGORIES and s.get("frequency") not in DROP_FREQ]

    def one(s):
        try:
            ms = K.settled_markets(s["ticker"])
        except Exception:  # noqa: BLE001
            return []
        out = []
        for m in ms:
            if m.get("result") not in ("yes", "no") or not (START <= (m.get("close_date") or "") <= END) or (m.get("volume") or 0) < 500:
                continue
            out.append({"venue": "kalshi", "id": m["id"], "event": m.get("event_ticker"), "series": s["ticker"], "category": s.get("category"),
                        "title": (m.get("title") or "")[:150], "sub": (m.get("subtitle") or "")[:60], "close": m.get("close_date"),
                        "settlement_ts": m.get("settlement_ts"), "open": m.get("open_date"), "y": 1 if m["result"] == "yes" else 0,
                        "volume": m.get("volume")})
        return out

    rows = []
    with ThreadPoolExecutor(THREADS) as ex:
        for k, r in enumerate(ex.map(one, ser)):
            rows += r
            if k % 500 == 0:
                print(f"  kalshi series {k}/{len(ser)}: {len(rows)} markets", flush=True)
    return _sample(rows, "event")


def _open_snaps(m: dict) -> list[str]:
    """Snapshot dates on which the market was open: after it opened, before the day it closed."""
    return [d for d in SNAPS if (m.get("open") or "0000") < d < (m.get("close") or "0000")]


def kalshi_prices(m: dict) -> dict:
    snaps = _open_snaps(m)
    if not snaps:
        return {}
    t0 = int(_snap_ts(snaps[0])) - 4 * 86400
    t1 = int(_snap_ts(snaps[-1])) + 86400
    frames = []
    for hist in (True, False):
        path = f"/historical/markets/{m['id']}/candlesticks" if hist else f"/series/{m['series']}/markets/{m['id']}/candlesticks"
        try:
            a = t0
            while a < t1:
                b = min(a + 900 * 86400, t1)
                frames.append(K.parse_candles(get_json("kalshi", K.BASE, path, {"start_ts": a, "end_ts": b, "period_interval": 1440}).get("candlesticks", [])))
                a = b
            break
        except Exception:  # noqa: BLE001
            frames = []
    if not frames:
        return {}
    c = pd.concat(frames)
    if c.empty:
        return {}
    c = c[~c.index.duplicated(keep="last")].sort_index()
    ts = np.array([t.timestamp() for t in c.index])
    out = {}
    for d in snaps:
        T = _snap_ts(d)
        i = np.searchsorted(ts, T, side="right") - 1
        if i < 0 or T - ts[i] > 3 * 86400:
            continue
        row = c.iloc[i]
        if pd.notna(row["trade"]) and row["volume"] > 0:
            out[d] = float(row["trade"])
        elif pd.notna(row["bid"]) and pd.notna(row["ask"]) and 0 <= row["ask"] - row["bid"] <= 0.10:
            out[d] = float((row["bid"] + row["ask"]) / 2)
    return out


# ---------------------------------------------------------------- Polymarket

def poly_markets() -> list[dict]:
    rows, seen = [], set()
    for tag in P.TAGS:
        for off in range(0, 2000, 100):     # the API refuses offsets past ~2,000: the top 2,000 events by volume per tag
            try:
                evs = get_json("polymarket", P.GAMMA, "/events", {"closed": "true", "tag_slug": tag, "end_date_min": START,
                                                                  "end_date_max": END + "T23:59:59Z", "limit": 100, "offset": off,
                                                                  "order": "volume", "ascending": "false"})
            except Exception:  # noqa: BLE001
                break
            if not evs:
                break
            for ev in evs:
                for m in ev.get("markets") or []:
                    if m["id"] in seen or (P._f(m.get("volumeNum")) or 0) < 10_000:
                        continue
                    try:
                        px = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                        tok = json.loads(m.get("clobTokenIds") or "[]")[0]
                    except (ValueError, IndexError, TypeError):
                        continue
                    if len(px) != 2 or sorted(px) != [0.0, 1.0] or not (START <= (m.get("endDate") or "")[:10] <= END):
                        continue
                    seen.add(m["id"])
                    rows.append({"venue": "polymarket", "id": m["id"], "event": f"E{ev.get('id')}", "series": tag, "category": tag,
                                 "title": (m.get("question") or "")[:150], "sub": (m.get("groupItemTitle") or "")[:60],
                                 "close": (m.get("closedTime") or m.get("endDate") or "")[:10], "token": tok, "y": int(px[0] == 1.0),
                                 "volume": P._f(m.get("volumeNum"))})
            if len(evs) < 100:
                break
        print(f"  polymarket {tag}: {len(rows)} markets", flush=True)
    return _sample(rows, "event")


def poly_prices(m: dict) -> dict:
    try:
        d = get_json("polymarket", P.CLOB, "/prices-history", {"market": m["token"], "interval": "max", "fidelity": 720})
    except Exception:  # noqa: BLE001
        return {}
    h = d.get("history") or []
    if not h:
        return {}
    ts = np.array([p["t"] for p in h])
    px = np.array([float(p["p"]) for p in h])
    out = {}
    first = datetime.fromtimestamp(ts[0], tz=timezone.utc).date().isoformat()
    for d in _open_snaps(dict(m, open=max(m.get("open") or "0000", first))):
        T = _snap_ts(d)
        i = np.searchsorted(ts, T, side="right") - 1
        if i >= 0 and T - ts[i] <= 2 * 86400:
            out[d] = float(px[i])
    return out


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    for name, lister, pricer in (("polymarket", poly_markets, poly_prices), ("kalshi", kalshi_markets, kalshi_prices)):
        if (DIR / f"{name}_priced.json").exists() and "--reprice" not in sys.argv:
            print(f"{name}: already priced (pass --reprice to redo)", flush=True)
            continue
        f = DIR / f"{name}_markets.json"
        ms = json.loads(f.read_text()) if f.exists() else lister()
        f.write_text(json.dumps(ms))
        print(f"{name}: {len(ms)} sampled markets; pricing...", flush=True)
        with ThreadPoolExecutor(THREADS) as ex:
            for k, (m, p) in enumerate(zip(ms, ex.map(pricer, ms))):
                m["p"] = p
                if k % 1000 == 0:
                    print(f"  {name} priced {k}/{len(ms)}", flush=True)
        (DIR / f"{name}_priced.json").write_text(json.dumps(ms))
        print(f"{name}: {sum(1 for m in ms if m['p'])} markets with a snapshot, {sum(len(m['p']) for m in ms)} snapshots", flush=True)


# ---------------------------------------------------------------- fit

BINS = [0, .02, .05, .10, .20, .35, .50, .65, .80, .90, .95, .98, 1.0001]


def pava(x, y):
    """Isotonic (non-decreasing) fit of y on x; returns the fitted values at sorted x and the sort order."""
    o = np.argsort(x, kind="mergesort")
    v, w, idx = list(y[o].astype(float)), [1.0] * len(o), [[i] for i in range(len(o))]
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            nv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            v[i:i + 2], w[i:i + 2], idx[i:i + 2] = [nv], [w[i] + w[i + 1]], [idx[i] + idx[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.empty(len(o))
    for val, ids in zip(v, idx):
        fit[ids] = val
    return x[o], fit


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def table(df: pd.DataFrame, B: int = 2000, seed: int = 0) -> list[dict]:
    """Per price bin: n, events, mean price, realized frequency, Wilson CI, clustered-bootstrap CI of (realized - price)."""
    rng = np.random.default_rng(seed)
    df = df.assign(bin=pd.cut(df["p"], BINS, right=False, labels=False))
    # per event x bin sums, then resample events with replacement as multinomial weights (the clustered bootstrap)
    ev_codes, ev_idx = np.unique(df["event"].to_numpy(), return_inverse=True)
    nb = len(BINS) - 1
    S = np.zeros((len(ev_codes), nb, 3))          # sum y, sum p, count
    np.add.at(S, (ev_idx, df["bin"].to_numpy().astype(int), 0), df["y"].to_numpy())
    np.add.at(S, (ev_idx, df["bin"].to_numpy().astype(int), 1), df["p"].to_numpy())
    np.add.at(S, (ev_idx, df["bin"].to_numpy().astype(int), 2), 1.0)
    W = rng.multinomial(len(ev_codes), np.full(len(ev_codes), 1 / len(ev_codes)), size=B).astype(float)   # B x events
    tot = np.einsum("be,ekc->bkc", W, S)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (tot[:, :, 0] - tot[:, :, 1]) / tot[:, :, 2]            # B x bins: realized - price
    rows = []
    for b, d in df.groupby("bin"):
        lo, hi = wilson(d["y"].sum(), len(d))
        bl, bh = np.nanpercentile(boots[:, int(b)], [2.5, 97.5])
        rows.append({"bin": f"{BINS[b]:.0%}–{min(BINS[b + 1], 1):.0%}", "lo": BINS[int(b)], "hi": min(BINS[int(b) + 1], 1.0), "n": len(d), "events": d["event"].nunique(),
                     "mean_price": d["p"].mean(), "realized": d["y"].mean(), "wilson_lo": lo, "wilson_hi": hi,
                     "gap": d["y"].mean() - d["p"].mean(), "gap_lo": bl, "gap_hi": bh, "significant": bool(bl > 0 or bh < 0)})
    return rows


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


EPS = 0.005


def _lg(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logit_fit(p, y, w=None):
    """Logistic recalibration: P(YES) = sigmoid(a + b * logit(price)), by Newton's method. b < 1 = prices too extreme
    (longshots overpriced AND favourites underpriced); a < 0 = YES overpriced across the board."""
    x, y = _lg(p), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    X = np.c_[np.ones_like(x), x]
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ beta))
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def logit_map(p, beta):
    return 1 / (1 + np.exp(-(beta[0] + beta[1] * _lg(p))))


def logit_ci(df, B=300, seed=1):
    """Event-clustered bootstrap of (a, b): resample events, refit."""
    rng = np.random.default_rng(seed)
    codes, idx = np.unique(df["event"].to_numpy(), return_inverse=True)
    p, y = df["p"].to_numpy(), df["y"].to_numpy()
    out = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes)).astype(float)
        out.append(logit_fit(p, y, cnt[idx]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def _frame(ms: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"venue": m["venue"], "event": m["venue"] + ":" + str(m["event"]), "market": m["id"], "category": m["category"],
                          "p": min(max(v, 0.0), 1.0), "y": m["y"], "date": d, "close": m["close"]}
                         for m in ms for d, v in (m.get("p") or {}).items()])


def _apply(p, xs, fs, bins_on):
    """The isotonic map inside the listed price bins; the raw price elsewhere."""
    p = np.asarray(p, float)
    b = np.clip(np.searchsorted(BINS, p, side="right") - 1, 0, len(BINS) - 2)
    return np.where(np.isin(b, bins_on), np.interp(p, xs, fs), p)


def fit():
    ms = []
    for name in ("kalshi", "polymarket"):
        f = DIR / f"{name}_priced.json"
        if f.exists():
            ms += json.loads(f.read_text())
    df = _frame(ms)
    res = {"snapshots": len(df), "markets": int(df["market"].nunique()), "events": int(df["event"].nunique()),
           "by_venue": {v: {"snapshots": int((df["venue"] == v).sum()), "markets": int(df.loc[df["venue"] == v, "market"].nunique()),
                            "events": int(df.loc[df["venue"] == v, "event"].nunique())} for v in df["venue"].unique()},
           "table": table(df)}
    for v in df["venue"].unique():
        res[f"table_{v}"] = table(df[df["venue"] == v])
    # out of sample: learn the map on markets that closed before 2025-07-01, score on markets that closed after
    tr, te = df[df["close"] < "2025-07-01"], df[df["close"] >= "2025-07-01"]
    xs, fs = pava(tr["p"].to_numpy(), tr["y"].to_numpy())
    sig_tr = [k for k, r in enumerate(table(tr)) if r["significant"]] if len(tr) else []
    bins_tr = [int(b) for b in sorted(pd.cut(tr["p"], BINS, right=False, labels=False).dropna().unique())]
    sig_bins = [bins_tr[k] for k in sig_tr]
    res["out_of_sample"] = {
        "train": "markets closed before 2025-07-01", "test": "markets closed 2025-07-01 .. 2026-06-30",
        "train_snapshots": len(tr), "test_snapshots": len(te), "significant_bins_in_train": [f"{BINS[b]:.0%}-{BINS[b + 1]:.0%}" for b in sig_bins],
        "brier_raw": brier(te["p"], te["y"]),
        "brier_isotonic_everywhere": brier(np.interp(te["p"], xs, fs), te["y"]),
        "brier_isotonic_significant_bins_only": brier(_apply(te["p"], xs, fs, sig_bins), te["y"]),
        "test_table": table(te)}
    xa, fa = pava(df["p"].to_numpy(), df["y"].to_numpy())
    grid = [.01, .02, .03, .05, .07, .10, .15, .20, .30, .40, .50, .60, .70, .80, .85, .90, .93, .95, .97, .98, .99]
    res["isotonic_map_all_data"] = {f"{g:.2f}": float(np.interp(g, xa, fa)) for g in grid}
    sig_all = [r["bin"] for r in res["table"] if r["significant"]]
    # the smooth correction: logistic recalibration, fit on train and scored on test, then refit on all data
    bt = logit_fit(tr["p"], tr["y"])
    oos = res["out_of_sample"]
    oos.update({"logit_train": {"a": float(bt[0]), "b": float(bt[1])},
                "brier_logit": brier(logit_map(te["p"], bt), te["y"]),
                "logloss_raw": logloss(te["p"], te["y"]), "logloss_isotonic_everywhere": logloss(np.interp(te["p"], xs, fs), te["y"]),
                "logloss_logit": logloss(logit_map(te["p"], bt), te["y"])})
    stable = [r["bin"] for r in oos["test_table"] if r["significant"] and r["bin"] in sig_all
              and np.sign(r["gap"]) == np.sign(next(x["gap"] for x in res["table"] if x["bin"] == r["bin"]))]
    ba = logit_fit(df["p"], df["y"])
    ci = logit_ci(df)
    res["logit_all_data"] = {"a": float(ba[0]), "b": float(ba[1]), "a_ci": [float(ci[0][0]), float(ci[1][0])],
                             "b_ci": [float(ci[0][1]), float(ci[1][1])],
                             "map": {f"{g:.2f}": float(logit_map(g, ba)) for g in grid}}
    for v in df["venue"].unique():
        bv = logit_fit(df.loc[df["venue"] == v, "p"], df.loc[df["venue"] == v, "y"])
        res[f"logit_{v}"] = {"a": float(bv[0]), "b": float(bv[1]), "map": {f"{g:.2f}": float(logit_map(g, bv)) for g in grid}}
    res["applied_correction"] = {"rule": "the logistic recalibration (all data) inside price bins where the bias is significant "
                                          "in the full sample AND in the out-of-sample test period with the same sign (event-"
                                          "clustered 95% intervals); the raw price elsewhere; both shown side by side",
                                 "bins": stable, "bins_significant_full_sample": sig_all}
    (OUT / "part2_longshot.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    # the committed parameters the chat's tier 1 reads (planb/probability.corrected)
    pack = {"source": "scripts/part2_longshot.py (DECISIONS #227)", "all": {k: la_ for k, la_ in res["logit_all_data"].items() if k in ("a", "b")},
            "by_venue": {v: {"a": res[f"logit_{v}"]["a"], "b": res[f"logit_{v}"]["b"]} for v in res["by_venue"]},
            "apply_in": [[r["lo"], r["hi"]] for r in res["table"] if r["bin"] in res["applied_correction"]["bins"]]}
    (ROOT / "src/worsey/planb/longshot.json").write_text(json.dumps(pack, indent=1), encoding="utf-8")
    report(res)
    print(json.dumps({k: v for k, v in res.items() if k in ("snapshots", "markets", "events", "by_venue")}, indent=1))
    print(json.dumps({k: v for k, v in res["out_of_sample"].items() if k != "test_table"}, indent=1, default=float))
    for name in ["table"] + [f"table_{v}" for v in df["venue"].unique()]:
        print("##", name)
        for r in res[name]:
            print(f"  {r['bin']:>9} n={r['n']:>6} ev={r['events']:>5} price {r['mean_price']:.3f} realized {r['realized']:.3f} "
                  f"gap {r['gap']:+.3f} [{r['gap_lo']:+.3f},{r['gap_hi']:+.3f}] {'*' if r['significant'] else ''}")


def _tab(rows) -> list[str]:
    out = ["| price bin | snapshots | events | mean price | realized | gap (realized - price), 95% CI clustered by event |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        star = " (significant)" if r["significant"] else ""
        out.append(f"| {r['bin']} | {r['n']:,} | {r['events']:,} | {r['mean_price']:.3f} | {r['realized']:.3f} | "
                   f"{r['gap']:+.3f} [{r['gap_lo']:+.3f}, {r['gap_hi']:+.3f}]{star} |")
    return out


def report(res: dict):
    o, la = res["out_of_sample"], res["logit_all_data"]
    L = ["# Part 2, tier 1: the longshot bias in prediction-market odds", "",
         "Generated by `scripts/part2_longshot.py` (code only, $0). Plan: `docs/event_pipeline/probabilities.md` step 1; DECISIONS #227.", "",
         "## Sample", "",
         f"- **{res['snapshots']:,} snapshots** of **{res['markets']:,} resolved markets** in **{res['events']:,} events**, closing 2023-01-01 to 2026-06-30 (all before Jul 1, 2026; later steps test on what came after).",
         *[f"- {v}: {d['snapshots']:,} snapshots, {d['markets']:,} markets, {d['events']:,} events" for v, d in res["by_venue"].items()],
         "- **Snapshots:** on the 1st and 15th of each month, every market still open contributes its price that day and its eventual outcome. Being open on a date is known on that date, so the sample doesn't depend on the outcome.",
         "- **First attempt, discarded:** prices 7 days before close. Close is when a market resolves, which is early for \"by date\" markets that resolve YES, so the snapshot time depended on the outcome; it showed the opposite bias (YES underpriced). Scheduled end dates don't fix it (Polymarket's end date is often the whole event's; Kalshi moves close_time on early closes).",
         "- **Filters:** Kalshi in-scope series, daily/weekly/intraday series dropped, volume >= 500 contracts; Polymarket topic tags (as in the universe), volume >= $10k; at most 4 markets per event (by a hash of the id); event-clustered intervals throughout.", "",
         "## Calibration: all markets", "", *_tab(res["table"]), ""]
    for v in res["by_venue"]:
        L += [f"## {v.capitalize()}", "", *_tab(res[f"table_{v}"]), "",
              f"Logistic recalibration for {v}: a = {res[f'logit_{v}']['a']:+.3f}, b = {res[f'logit_{v}']['b']:.3f}.", ""]
    L += ["## Out of sample (learn on markets closed before Jul 2025, score on markets closed Jul 2025 - Jun 2026)", "",
          f"| | Brier | log loss |", "|---|---|---|",
          f"| raw market price | {o['brier_raw']:.4f} | {o['logloss_raw']:.4f} |",
          f"| isotonic map | {o['brier_isotonic_everywhere']:.4f} | {o['logloss_isotonic_everywhere']:.4f} |",
          f"| logistic recalibration | {o['brier_logit']:.4f} | {o['logloss_logit']:.4f} |", "",
          "Test-period calibration (is the bias still there a year later?):", "", *_tab(o["test_table"]), "",
          "## The correction", "",
          f"- **Form:** P(YES) = sigmoid(a + b x logit(price)), all data: **a = {la['a']:+.3f}** [{la['a_ci'][0]:+.3f}, {la['a_ci'][1]:+.3f}], **b = {la['b']:.3f}** [{la['b_ci'][0]:.3f}, {la['b_ci'][1]:.3f}] (event-clustered bootstrap). a < 0: YES is a little overpriced across the board; b > 1: prices sit too close to 50%, so low prices are too high (longshots overpriced) and high prices too low (favourites underpriced).",
          f"- **Applied only** in price bins where the bias is significant in the full sample and again in the test year with the same sign: {', '.join(res['applied_correction']['bins']) or 'none'}. Elsewhere the raw price is used. Raw and corrected odds are always shown side by side.", "",
          "| market price | corrected |", "|---|---|", *[f"| {float(k):.0%} | {v:.1%} |" for k, v in la["map"].items()], ""]
    (OUT / "part2_longshot.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "fit"])
    ap.add_argument("--reprice", action="store_true")
    fetch() if ap.parse_args().step == "fetch" else fit()
