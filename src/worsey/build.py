"""`worsey build`: the offline Kalshi universe scan + cached β (SPEC §1a, §11b; METHODOLOGY §5.2).

Queries never bulk-scan: this runs once (nightly in production), and scan_risks reads its output,
refreshing only live quotes. Stages, each logged with counts:

  series category (Sports/Entertainment and other out-of-scope series excluded at SERIES level)
  → horizon (resolves within 12 months) → history (open ≥ 60 days) → liquidity (volume, two-sided quote)
  → reflexive price-level markets out (S&P/BTC/WTI levels are the asset itself, not events)
  → cap (top N by volume, round-robin across categories; standing-calendar series always kept) → hourly history fetched (p at 16:00 ET, stale masked)
  → tier-1(a) eligible (≥ 60 usable days, ≥ 5 jumps of 5pp) → β per asset in the build universe.

Output: cache/build/<UTC timestamp>/ {manifest.json, markets.json, betas.json, p_history.csv, closes.csv}
and cache/build/LATEST pointing at it.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import config
from .data import kalshi
from .data.cache import CACHE_ROOT
from .portfolio import PORTFOLIO_DIR
from .scenario import discovery, impact

BUILD_ROOT = CACHE_ROOT / "build"


def universe() -> list[str]:
    """Every ticker in a demo portfolio + reference/factor proxies (so any demo book has β everywhere)."""
    t = set(config.BUILD_EXTRA_TICKERS)
    for f in PORTFOLIO_DIR.glob("*.yaml"):
        for p in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("positions", []):
            if p.get("ticker"):
                t.add(p["ticker"].upper())
    return sorted(t)


def _series_categories() -> dict[str, str]:
    return {s["ticker"]: s.get("category") for s in kalshi.series(None)}


def _events_with_markets() -> list[dict]:
    return kalshi._paged("/events", {"status": "open", "limit": 200, "with_nested_markets": "true"}, "events", max_pages=400)


def run(data, today: date | None = None, max_markets: int | None = None, log=print, llm=None) -> Path:
    t_start = time.time()
    today = today or date.today()
    max_markets = config.BUILD_MAX_MARKETS if max_markets is None else max_markets
    counts: dict[str, int] = {}

    # 1. series-level category filter
    cats = _series_categories()
    counts["series_total"] = len(cats)
    counts["series_excluded_sports_entertainment"] = sum(1 for c in cats.values() if c in config.BUILD_EXCLUDED_CATEGORIES)
    in_scope = {s for s, c in cats.items() if c in discovery.IN_SCOPE_CATEGORIES}
    counts["series_in_scope"] = len(in_scope)

    events = _events_with_markets()
    counts["open_events"] = len(events)
    markets = []
    for ev in events:
        for m in ev.get("markets") or []:
            nm = kalshi.normalize_market(m, ev)
            nm["series_category"] = cats.get(ev.get("series_ticker"))
            markets.append(nm)
    counts["open_markets"] = len(markets)
    stage = [m for m in markets if m["series_ticker"] in in_scope]
    counts["after_series_category"] = len(stage)

    # 2. horizon, history, liquidity (discovery.bottom_up_filter uses the event/series category we attached)
    for m in stage:
        m["category"] = m["series_category"]
    h = [m for m in stage if m.get("close_date") and 0 < (date.fromisoformat(m["close_date"]) - today).days <= config.MAX_RESOLVE_DAYS]
    counts["after_horizon_12m"] = len(h)
    h = [m for m in h if m.get("open_date") and (today - date.fromisoformat(m["open_date"])).days >= config.MIN_HISTORY_DAYS]
    counts["after_open_60d"] = len(h)
    h = [m for m in h if (m.get("volume") or 0) >= config.BUILD_MIN_VOLUME
         and m.get("bid") is not None and m.get("ask") is not None and m["bid"] > 0 and m["ask"] < 1]
    counts["after_liquidity"] = len(h)
    h = [m for m in h if not discovery.is_reflexive(m)]
    counts["after_reflexive_price_filter"] = len(h)
    standing = [m for m in h if m.get("series_ticker") in set(config.BUILD_STANDING_SERIES)]
    rest = [m for m in h if m.get("series_ticker") not in set(config.BUILD_STANDING_SERIES)]
    capped = discovery.diversified_cap(rest + standing, max_markets)   # would it have made the cap on its own?
    capped_ids = {m["id"] for m in capped}
    for m in standing:
        m["via_standing_only"] = m["id"] not in capped_ids        # flagged on the row
    h = standing + discovery.diversified_cap(rest, max_markets)
    counts["standing_calendar_kept"] = len(standing)
    counts["after_cap_top_n_by_category_plus_standing"] = len(h)
    log(f"[build] markets: {counts}")

    # 3. equity side: closes for the build universe
    tick = universe()
    closes = data.closes(tick, days=config.BUILD_HISTORY_DAYS + 30)
    rets = closes.pct_change().dropna(how="all")
    counts["universe_tickers"] = len(tick)

    # 4. per market: hourly history → daily p at 16:00 ET → eligibility → β
    start = datetime.combine(today - timedelta(days=config.BUILD_HISTORY_DAYS), datetime.min.time(), tzinfo=timezone.utc)
    hist, betas, kept, skipped = {}, {}, [], []
    for i, m in enumerate(h):
        try:
            s0 = start
            if m.get("open_date"):
                s0 = max(start, datetime.fromisoformat(m["open_date"]).replace(tzinfo=timezone.utc))
            p = kalshi.p_series(m, s0)
        except Exception as e:  # noqa: BLE001 — one bad market never kills the build
            skipped.append({"id": m["id"], "reason": f"history fetch failed: {str(e)[:80]}"})
            continue
        dp = impact.align_dp(p, rets.index)
        el = discovery.tier1_eligible(dp)
        if not el["eligible"]:
            skipped.append({"id": m["id"], "reason": el["reason"]})
            continue
        b = impact.empirical_betas(rets, dp)
        betas[m["id"]] = {t: {"beta": x.beta, "se": x.se, "t": x.t, "n": x.n, "shrunk": x.shrunk} for t, x in b.items()}
        hist[m["id"]] = p
        kept.append(dict(m, jump_days=el["jump_days"], usable_days=el["days"]))
        if (i + 1) % 25 == 0:
            log(f"[build] {i + 1}/{len(h)} markets processed, {len(kept)} eligible")
    counts["history_fetched"] = len(h) - sum(1 for s in skipped if s["reason"].startswith("history fetch"))
    counts["tier1_eligible"] = len(kept)

    # 5. structure for the ranking: loadings, tier-3 shock maps, pooled tier-1(b) observations
    from .scenario import structural
    fac = impact.factor_returns(closes)
    loadings = impact.factor_loadings(rets, fac)
    counts["loadings_assets"] = len(loadings)
    priors = shock_maps(kept, llm, log)
    counts["shock_maps"] = sum(1 for k in kept if k["id"] in priors)
    notes = event_notes(kept, llm, log)
    counts["event_notes"] = len(notes)
    names = short_names(kept, llm, log)
    counts["short_names"] = len(names)
    pooled_obs = pooled_observations(data, tick, log)
    counts["pooled_obs"] = int(len(pooled_obs))
    counts["pooled_releases"] = {k: int(g["release"].nunique()) for k, g in pooled_obs.groupby("key")} if len(pooled_obs) else {}

    # 6. implied-vol history for the playbook signals (backfill once, then today's snapshot)
    from .options import vol_history
    vh = vol_history.update(config.VOL_HISTORY_TICKERS, data, today, days=config.VOL_HISTORY_DAYS, log=log)
    counts["vol_history"] = {u: r.get("rows", "error") for u, r in vh.items()}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = BUILD_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"built_at_utc": stamp, "as_of": today.isoformat(), "counts": counts, "seconds": round(time.time() - t_start, 1),
                "params": {"max_resolve_days": config.MAX_RESOLVE_DAYS, "min_open_days": config.MIN_HISTORY_DAYS,
                           "min_volume": config.BUILD_MIN_VOLUME, "max_markets": max_markets,
                           "history_days": config.BUILD_HISTORY_DAYS, "min_usable_days": config.MIN_HISTORY_DAYS,
                           "jump_pp": config.JUMP_PP, "min_jump_days": config.MIN_JUMP_DAYS,
                           "excluded_series_categories": sorted(config.BUILD_EXCLUDED_CATEGORIES),
                           "in_scope_categories": sorted(discovery.IN_SCOPE_CATEGORIES)},
                "universe": tick, "skipped": skipped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    (out / "markets.json").write_text(json.dumps(kept, indent=1, default=str), encoding="utf-8")
    (out / "betas.json").write_text(json.dumps(betas), encoding="utf-8")
    pd.DataFrame(hist).to_csv(out / "p_history.csv")
    closes.to_csv(out / "closes.csv")
    loadings.to_csv(out / "loadings.csv")
    (out / "priors.json").write_text(json.dumps(priors, indent=1), encoding="utf-8")
    (out / "notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")
    (out / "names.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    pooled_obs.to_csv(out / "pooled_obs.csv", index=False)
    (BUILD_ROOT / "LATEST").write_text(stamp, encoding="utf-8")
    log(f"[build] done in {manifest['seconds']}s → {out} ; counts {counts}")
    return out


PRIOR_CACHE = CACHE_ROOT / "priors"


def shock_maps(markets: list[dict], llm, log=print) -> dict:
    """Tier-3 shock maps (sign + size bucket per factor) from Claude, cached per market id so rebuilds don't re-ask."""
    from .scenario import structural
    PRIOR_CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for m in markets:
        f = PRIOR_CACHE / f"{m['id']}.json"
        if f.exists():
            out[m["id"]] = json.loads(f.read_text(encoding="utf-8"))
        else:
            todo.append(m)
    if todo and llm is None:
        log(f"[build] {len(todo)} markets have no shock map and no LLM was given: their prior is 0")
        return out
    for i in range(0, len(todo), config.SHOCK_MAP_BATCH):
        batch = todo[i:i + config.SHOCK_MAP_BATCH]
        got = structural.ask_shock_maps(llm, batch, config.EXPERT_MODEL)
        for mid, v in got.items():
            (PRIOR_CACHE / f"{mid}.json").write_text(json.dumps(v), encoding="utf-8")
            out[mid] = v
        log(f"[build] shock maps {min(i + config.SHOCK_MAP_BATCH, len(todo))}/{len(todo)}")
    return out


NOTES_CACHE = CACHE_ROOT / "notes"


def event_notes(markets: list[dict], llm, log=print) -> dict:
    """Number-free notes per event (what / channels / caveats), cached per market id: scan_risks quotes these
    instead of making a live expert call (latency, SPEC §11b)."""
    from .scenario import structural
    NOTES_CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for m in markets:
        f = NOTES_CACHE / f"{m['id']}.json"
        if f.exists():
            out[m["id"]] = json.loads(f.read_text(encoding="utf-8"))
        else:
            todo.append(m)
    if todo and llm is None:
        return out
    for i in range(0, len(todo), config.SHOCK_MAP_BATCH):
        batch = todo[i:i + config.SHOCK_MAP_BATCH]
        for mid, v in structural.ask_event_notes(llm, batch, config.EXPERT_MODEL).items():
            (NOTES_CACHE / f"{mid}.json").write_text(json.dumps(v), encoding="utf-8")
            out[mid] = v
        log(f"[build] event notes {min(i + config.SHOCK_MAP_BATCH, len(todo))}/{len(todo)}")
    return out


NAMES_CACHE = CACHE_ROOT / "names"


def short_names(markets: list[dict], llm, log=print) -> dict:
    """Short display names per market (cards never show truncated raw titles), cached per market id."""
    from .scenario import structural
    NAMES_CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for m in markets:
        f = NAMES_CACHE / f"{m['id']}.json"
        if f.exists():
            out[m["id"]] = json.loads(f.read_text(encoding="utf-8"))
        else:
            todo.append(m)
    if todo and llm is None:
        return out
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        for mid, v in structural.ask_short_names(llm, batch, config.EXPERT_MODEL).items():
            (NAMES_CACHE / f"{mid}.json").write_text(json.dumps(v), encoding="utf-8")
            out[mid] = v
        log(f"[build] short names {min(i + 40, len(todo))}/{len(todo)}")
    return out


def add_names(llm, path: Path | None = None, log=print) -> Path:
    """Add names.json to an existing build (no other step re-runs)."""
    path = path or latest()
    markets = json.loads((path / "markets.json").read_text(encoding="utf-8"))
    names = short_names(markets, llm, log)
    (path / "names.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    log(f"[build] {len(names)}/{len(markets)} short names → {path / 'names.json'}")
    return path


def pooled_observations(data, tickers: list[str], log=print) -> pd.DataFrame:
    """Tier 1(b) observations: one row per settled market in the pooled series (Fed, CPI, payrolls).
    surprise = outcome − p at the prior trading day's 16:00 ET; returns = prior close → release-day close."""
    from .scenario import structural
    closes = data.closes(tickers, days=1500)
    idx = closes.index
    rows = []
    for ser in config.POOLED_SERIES:
        ms = [m for m in kalshi.settled_markets(ser) if m.get("result") in ("yes", "no") and m.get("close_date")]
        for j, m in enumerate(ms):
            rel = pd.Timestamp(m["close_date"])
            if rel < pd.Timestamp(config.POOLED_START) or rel > idx[-1]:
                continue
            i1 = idx.searchsorted(rel)            # release day (or next session)
            if i1 >= len(idx) or i1 == 0:
                continue
            prev_day = idx[i1 - 1]
            try:
                d = kalshi.daily_history(m, (rel - pd.Timedelta(days=8)).to_pydatetime().replace(tzinfo=timezone.utc),
                                         (rel + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc))
            except Exception:  # noqa: BLE001 — one market never kills the build
                continue
            if d.empty or not isinstance(d.index, pd.DatetimeIndex):
                continue  # no candles in the window
            before = d[d.index <= prev_day]["p"].dropna()
            if before.empty:
                continue
            outcome = 1.0 if m["result"] == "yes" else 0.0
            r = closes.iloc[i1] / closes.iloc[i1 - 1] - 1
            rows.append({"key": structural.pooled_key(m), "market": m["id"], "release": str(idx[i1].date()),
                         "p_prev": float(before.iloc[-1]), "outcome": outcome, "surprise": outcome - float(before.iloc[-1]),
                         **{t: float(r[t]) for t in closes.columns if pd.notna(r[t])}})
            if (j + 1) % 100 == 0:
                log(f"[build] pooled {ser}: {j + 1}/{len(ms)}")
    return pd.DataFrame(rows)


def latest() -> Path | None:
    f = BUILD_ROOT / "LATEST"
    return BUILD_ROOT / f.read_text(encoding="utf-8").strip() if f.exists() else None


def load(path: Path | None = None) -> dict:
    """Read a build: manifest, markets (list), betas (market → asset → stats), p_history, closes."""
    path = path or latest()
    if path is None or not path.exists():
        raise FileNotFoundError("no build yet: run `worsey build`")
    ph = pd.read_csv(path / "p_history.csv", index_col=0, parse_dates=True)
    cl = pd.read_csv(path / "closes.csv", index_col=0, parse_dates=True)
    out = {"path": path, "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
           "markets": json.loads((path / "markets.json").read_text(encoding="utf-8")),
           "betas": json.loads((path / "betas.json").read_text(encoding="utf-8")), "p_history": ph, "closes": cl}
    if (path / "loadings.csv").exists():  # builds from Oct 16 on carry the ranking structure
        out["structure"] = {"loadings": pd.read_csv(path / "loadings.csv", index_col=0),
                            "priors": json.loads((path / "priors.json").read_text(encoding="utf-8")),
                            "pooled_obs": pd.read_csv(path / "pooled_obs.csv"),
                            "notes": json.loads((path / "notes.json").read_text(encoding="utf-8")) if (path / "notes.json").exists() else {},
                            "names": json.loads((path / "names.json").read_text(encoding="utf-8")) if (path / "names.json").exists() else {},
                            "standing": [m["id"] for m in out["markets"] if m.get("via_standing_only")]}
    return out
