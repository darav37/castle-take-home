"""B5 part 1: prediction-market-implied surprises for scheduled releases (DECISIONS #172). No LLM calls.

  python scripts/planb_b5_ladders.py [--perms 200]

For every Kalshi CPI (MoM), jobs (payrolls change) and GDP (advance, annualized) event 2021-2026:
  implied     the ladder of "above X" markets at the last 16:00 ET before the release (carried book quotes, P0 rules),
              made monotonic (P(>k) non-increasing in k), turned into a distribution and its mean
  actual      the first-published number as of the release date (ALFRED vintage), rounded as published;
              cross-checked against the settled ladder (highest YES strike < actual <= lowest NO strike)
  surprise    actual - implied mean (also standardized by the implied sd)
Tests on these releases (leave-one-release-out within type, ridge, R² vs the type mean; chance from 200 shuffles):
  does the Kalshi surprise, vs the naive surprise and vs the release-day 2y yield change, predict
  (a) the 2y yield change itself (the "bridge"), (b) the release-day 49-industry cross-section and the non-rates factors?
Writes docs/pmrm/results/b5_ladders.json and cache/pmrm/b5/ladders.csv.
"""

import argparse
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

import pmrm_audit as A  # noqa: E402
from worsey.data import kalshi  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402
from worsey.planb import library as L  # noqa: E402
from worsey.planb import longhist as LH  # noqa: E402
from worsey.planb import regime as RG  # noqa: E402

STORE = ROOT / "cache/pmrm/b5"
SERIES = {"cpi": ("KXCPI", "CPIAUCSL"), "jobs": ("KXPAYROLLS", "PAYEMS"), "gdp": ("KXGDP", "GDPC1")}


def actual_value(rel: str, sid: str, release_day: pd.Timestamp) -> float | None:
    """The number as first published on release_day (ALFRED vintage on that day)."""
    d = release_day.date().isoformat()
    try:
        obs = L.fred_api("/series/observations", series_id=sid, realtime_start=d, realtime_end=d)["observations"]
    except Exception:  # noqa: BLE001
        return None
    v = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")}).sort_index()
    if len(v) < 2:
        return None
    if rel == "cpi":
        return round((v.iloc[-1] / v.iloc[-2] - 1) * 100, 1)
    if rel == "jobs":
        return (v.iloc[-1] - v.iloc[-2]) * 1000.0
    return round(((v.iloc[-1] / v.iloc[-2]) ** 4 - 1) * 100, 1)


def implied(strikes: np.ndarray, p_above: np.ndarray) -> tuple[float, float]:
    """Mean and sd of the distribution implied by P(X > k) on sorted strikes (monotone; tails at half a step out)."""
    p = np.minimum.accumulate(np.clip(p_above, 0, 1))          # non-increasing in k
    step = np.median(np.diff(strikes)) if len(strikes) > 1 else 1.0
    pts = np.concatenate([[strikes[0] - step / 2], (strikes[:-1] + strikes[1:]) / 2, [strikes[-1] + step / 2]])
    mass = np.concatenate([[1 - p[0]], p[:-1] - p[1:], [p[-1]]])
    mass = np.clip(mass, 0, None)
    mass = mass / mass.sum()
    m = float(mass @ pts)
    return m, float(np.sqrt(mass @ (pts - m) ** 2))


def build() -> pd.DataFrame:
    rows = []
    for rel, (series, sid) in SERIES.items():
        ms = kalshi._paged("/historical/markets", {"series_ticker": series, "limit": 1000}, "markets") + \
             kalshi._paged("/markets", {"series_ticker": series, "limit": 1000}, "markets")
        seen, ev = set(), {}
        for m in ms:
            if m["ticker"] in seen or m.get("strike_type") != "greater" or m.get("floor_strike") is None:
                continue
            seen.add(m["ticker"])
            ev.setdefault(m["event_ticker"], []).append(m)
        hist_set = {m["ticker"] for m in kalshi._paged("/historical/markets", {"series_ticker": series, "limit": 1000}, "markets")}
        for e, mk in ev.items():
            close = pd.Timestamp(max(m["close_time"] for m in mk)).tz_convert("America/New_York")
            release = (close + pd.Timedelta(days=1)).normalize() if close.hour >= 18 else close.normalize()
            if release.year < 2021 or release > pd.Timestamp("2026-10-16", tz="America/New_York"):
                continue
            # last 16:00 ET endpoint strictly before the release morning
            ep = (release - pd.Timedelta(days=1)).replace(hour=16)
            while ep.weekday() >= 5:
                ep -= pd.Timedelta(days=1)
            ep_utc = ep.tz_convert("UTC")
            ks, ps = [], []
            for m in sorted(mk, key=lambda m: float(m["floor_strike"])):
                mm = {"ticker": m["ticker"], "series": series, "historical": m["ticker"] in hist_set,
                      "open": pd.Timestamp(m["open_time"]).tz_convert("UTC") if m.get("open_time") else None,
                      "close": pd.Timestamp(m["close_time"]).tz_convert("UTC"), "settled": None}
                h = A.candles(mm, ep_utc - pd.Timedelta(days=4), ep_utc + pd.Timedelta(hours=1))
                q = A.endpoint_quotes(h, pd.Series([ep_utc], index=[ep_utc]))
                if not bool(q["quote_ok"].iloc[0]) and pd.isna(q["mid"].iloc[0]):
                    continue
                ks.append(float(m["floor_strike"]))
                ps.append(float(q["mid"].iloc[0]))
            if len(ks) < 3:
                continue
            ks, ps = np.array(ks), np.array(ps)
            mean, sd = implied(ks, ps)
            act = actual_value(rel, sid, release.tz_localize(None))
            yes = [float(m["floor_strike"]) for m in mk if m.get("result") == "yes"]
            no = [float(m["floor_strike"]) for m in mk if m.get("result") == "no"]
            lo, hi = (max(yes) if yes else -np.inf), (min(no) if no else np.inf)
            ok = act is not None and lo < act + 1e-9 and act <= hi + 1e-9
            rows.append({"release": rel, "event": e, "date": release.tz_localize(None).normalize(), "strikes": len(ks),
                         "implied_mean": mean, "implied_sd": sd, "actual": act, "settled_bracket": [lo, hi], "actual_in_bracket": ok})
        print(f"  {rel}: {sum(r['release'] == rel for r in rows)} events with a ladder", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(1729)
    STORE.mkdir(parents=True, exist_ok=True)
    df = build()
    df["surprise"] = df["actual"] - df["implied_mean"]
    df["surprise_z"] = df["surprise"] / df["implied_sd"].replace(0, np.nan)
    df.to_csv(STORE / "ladders.csv", index=False)
    print(df.groupby("release").agg(n=("event", "size"), in_bracket=("actual_in_bracket", "mean"),
                                    mean_abs_surprise=("surprise", lambda s: s.abs().mean())).to_string(), flush=True)
    # release-day responses and the competing surprise measures
    I = LH.industries()
    sessions = I.index[I.index >= "2020-01-01"]
    X = LH.long_factors(sessions)
    lib = pd.read_csv(ROOT / "cache/pmrm/b2/scheduled.csv", parse_dates=["date"])
    df = df[df["actual"].notna()].copy()
    df["i"] = sessions.searchsorted(df["date"])
    df = df[df["i"] < len(sessions)].reset_index(drop=True)
    idx = df["i"].to_numpy()
    nonrates = ["market", "size", "value", "profitability", "investment", "momentum", "dollar", "oil"]
    Fv, Iv = X[nonrates].to_numpy()[idx], I.loc[sessions].to_numpy()[idx]
    d2y = X["rates_2y"].to_numpy()[idx]
    naive = []
    for _, r in df.iterrows():
        m = lib[(lib["release"] == r["release"]) & (lib["date"].dt.normalize() == r["date"])]
        naive.append(float(m["data_surprise_z"].iloc[0]) if len(m) and pd.notna(m["data_surprise_z"].iloc[0]) else
                     (float(m["data_surprise"].iloc[0]) if len(m) and pd.notna(m["data_surprise"].iloc[0]) else np.nan))
    df["naive"] = naive
    types = df["release"].to_numpy()
    levels = sorted(set(types))
    out = {"events": df.groupby("release").size().to_dict(),
           "actual_in_settled_bracket": float(df["actual_in_bracket"].mean())}
    # (a) bridge: which surprise lines up with the market's rates reaction?
    out["bridge_corr_with_d2y"] = {lv: {"kalshi": float(np.corrcoef(df.loc[types == lv, "surprise_z"].fillna(0), d2y[types == lv])[0, 1]),
                                        "naive": float(pd.Series(df.loc[types == lv, "naive"].to_numpy()).corr(pd.Series(d2y[types == lv])))}
                                   for lv in levels}
    print("bridge corr with release-day 2y change:", json.dumps(out["bridge_corr_with_d2y"], default=lambda x: round(x, 2)), flush=True)

    # (b) predict release-day responses, leave one release out within type
    def loo_r2(Sv, Y):
        Y = (Y - np.nanmean(Y, 0)) / np.nanstd(Y, 0)
        M = ~np.isnan(Y)
        Y0 = np.nan_to_num(Y)
        P, B = np.zeros_like(Y0), np.zeros_like(Y0)
        for i in range(len(Y0)):
            tr = (np.arange(len(Y0)) != i) & (types == types[i])
            Xd = np.column_stack([np.ones(tr.sum()), np.nan_to_num(Sv[tr])])
            beta = RG.weighted_ridge(Xd, Y0[tr], np.ones(tr.sum()), 1.0)
            P[i] = np.concatenate([[1.0], np.nan_to_num(Sv[i])]) @ beta
            B[i] = Y0[tr].mean(0)
        return float(1 - (((Y0 - P) ** 2) * M).sum() / (((Y0 - B) ** 2) * M).sum())

    measures = {"kalshi surprise": df[["surprise_z"]].to_numpy(), "naive surprise": df[["naive"]].to_numpy(),
                "2y yield move (reaction)": d2y[:, None], "kalshi + 2y move": np.column_stack([df["surprise_z"], d2y])}
    res = {}
    for nm, Sv in measures.items():
        r_i, r_f = loo_r2(Sv, Iv), loo_r2(Sv, Fv)
        r_rates = loo_r2(Sv, d2y[:, None]) if nm in ("kalshi surprise", "naive surprise") else None
        nl = [loo_r2(Sv[rng.permutation(len(Sv))], Iv) for _ in range(a.perms if nm == "kalshi surprise" else 50)]
        res[nm] = {"industries_r2": r_i, "factors_r2": r_f, "predicts_2y_move_r2": r_rates, "industries_chance95": float(np.quantile(nl, 0.95))}
        print(f"{nm:26} industries {r_i:+.3f} (chance95 {res[nm]['industries_chance95']:+.3f})  factors {r_f:+.3f}"
              + (f"  predicts the 2y move {r_rates:+.3f}" if r_rates is not None else ""), flush=True)
    out["prediction"] = res
    (ROOT / "docs/pmrm/results/b5_ladders.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
