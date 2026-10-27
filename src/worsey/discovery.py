"""The event universe for a book: what's coming, what would hurt it most, and the two matched (DECISIONS #218, #222).

Stream A, what's actually coming (topical):
  markets    prediction markets that close within the horizon and move markets (labels made once by the LLM,
             planb/market_labels.json): full impact on the book, odds, priced in            -> upcoming()
  calendar   releases (Fed, CPI, jobs, GDP: dates from the market universe; impact from replaying past releases on the
             book), held-name earnings (look-through exposure x the name's typical earnings move), EIA and WASDE (the
             book's exposure x how big those release days usually are, #212), the US election date  -> calendar()
  news       the last week's labelled stories from the daily feed (scripts/planb_prospective.py), as "the story develops
             further" candidates with a base rate from the library                          -> news_candidates()
Stream B, what would move THIS book most (impactful):
  proposals  the LLM, given the book's biggest exposures, proposes the specific events that would move it most, each in
             the same label format, linked to a market / calendar item / news story when one is the same thing
             (odds from the market; otherwise a base rate of how often big events of that kind happened, labelled)
Every candidate is priced by the same code (the "what if" engine: size calls, co-moves, sector legs, held-company legs,
options repriced). The match (watchlist()) groups them:
  headline       topical AND impactful (impact >= half the book's normal two-day swing), ranked by the risk left
  tail risks     impactful, but no live catalyst (proposals not linked to anything coming)
  noise for you  topical but too small for this book
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = Path(__file__).resolve().parent / "planb" / "market_labels.json"
FEED = ROOT / "cache/pmrm/prospective/frozen.jsonl"
GENERIC_COMPANY_NEWS = "Contracts"     # labels give a held company and a direction, not a news type: use a mid-sized type
SEV_SIZE = {"minor": "small", "moderate": "small", "major": "medium", "extreme": "large"}
RELEASE_SERIES = {"KXFEDDECISION": ("fomc", "Fed decision"), "KXFED": ("fomc", "Fed decision"), "KXCPI": ("cpi", "CPI report"),
                  "KXPAYROLLS": ("jobs", "jobs report"), "KXGDP": ("gdp", "GDP report")}


@lru_cache(maxsize=1)
def labels() -> dict:
    return json.loads(LABELS.read_text(encoding="utf-8"))["labels"] if LABELS.exists() else {}


# ---------------------------------------------------------------- shared pricing

def context(s) -> dict:
    """Everything the pricing needs about the book, computed once per request."""
    from . import stocks
    from .planb import live
    vals = s.priced["value"]
    tick = list(dict.fromkeys(list(vals.index) + live.REFS + [o.underlying for o in live._options(s)]))
    L = live.loadings(s.data, tick)
    held = [t for t in vals.index if t in L["betas"].index]
    Bh = L["betas"].loc[held]
    v = vals[held]
    cov = live.book_coverage(L, vals)
    return {"s": s, "vals": vals, "L": L, "held": held, "Bh": Bh, "v": v,
            "db": (Bh.T @ v).reindex(live.FACTORS).to_numpy(), "cov": cov,
            "fd": live.factor_returns(L["returns"]).reindex(columns=live.FACTORS).tail(live.LOOKBACK).std().fillna(1.0).to_numpy(),
            "SP": live.sector_pack()["sectors"], "SL": live.sector_loadings(s, L),
            "CP": live.country_pack()["countries"], "CL": live.country_loadings(s, L),
            "news": stocks._news_stats().get(GENERIC_COMPANY_NEWS, {}).get("clean_timing", {}),
            "opts": live.option_setup(s, L) if live._options(s) else [],
            "swing": float(cov["book_daily_sd_usd"]) * np.sqrt(2), "today": _today(s)}


def _today(s) -> pd.Timestamp:
    """What's 'coming' is judged from the wall clock when the data is live (prices can lag a few days); recorded and
    synthetic books keep their own date."""
    d, now = pd.Timestamp(s.data.today()), pd.Timestamp(date.today())
    return max(d, now) if abs((now - d).days) <= 30 else d


def combine_legs(sec_mv: dict, co_mv: dict, cty_mv: dict | None = None) -> tuple[dict, list, list]:
    """Per holding, its sector-leg, company-leg and country-leg moves from one event. With more than one, the largest,
    never the sum: a broad event's company leg is mostly the sector's move again, an event aimed at the company keeps its
    own (#232), and a country's own shock and a sector's are each the holding's move beyond the macro factors (#238).
    Returns (move per holding, holdings whose company leg was set aside, holdings whose company leg was the one kept
    over another leg)."""
    cty_mv = cty_mv or {}
    out, skipped, over = {}, [], []
    for t in set(sec_mv) | set(co_mv) | set(cty_mv):
        legs = {k: v[t] for k, v in (("sector", sec_mv), ("company", co_mv), ("country", cty_mv)) if t in v}
        k_best = max(legs, key=lambda k: abs(legs[k]))
        out[t] = legs[k_best]
        if "company" in legs and len(legs) > 1:
            (over if k_best == "company" else skipped).append(t)
    return out, sorted(skipped), sorted(over)


def impact(ctx: dict, lb: dict) -> dict:
    """The book's move if this outcome happened as a total surprise, from a label {severity, factor_signs ('+'/'-'),
    factor_sizes, sectors, companies}: the "what if" engine, per position, with options repriced."""
    from .planb import live
    held, Bh, v, L = ctx["held"], ctx["Bh"], ctx["v"], ctx["L"]
    spec = {"severity": lb.get("severity", "moderate"), "category": lb.get("category"),
            "factor_signs": {k: (1 if d == "+" else -1) for k, d in (lb.get("factor_signs") or {}).items() if d in ("+", "-")},
            "factor_sizes": lb.get("factor_sizes") or {}}
    d = live.real_event_draws(spec, ctx["db"], ctx["cov"]["daily_noise_usd"], ctx["fd"])
    mv = {t: float(Bh.loc[t].to_numpy() @ d["f"]) for t in held}
    sec_mv, co_mv = {}, {}
    names = set(held) | {o["pos"].underlying for o in ctx["opts"]}   # an option's underlier gets what the stock would (#235)
    for sec, dirn in lb.get("sectors") or []:
        z = ctx["SP"].get(sec, {}).get("two_day_move_median", 0.0) * np.sqrt(2) * (1 if dirn == "+" else -1)
        for t in names:
            if ctx["SL"].get(t, {}).get("sector") == sec:
                sec_mv[t] = sec_mv.get(t, 0.0) + ctx["SL"][t]["loading"] * z
    for tk, dirn in lb.get("companies") or []:
        if tk in names and ctx["news"].get("median_abs_day1"):
            co_mv[tk] = co_mv.get(tk, 0.0) + (1 if dirn == "+" else -1) * ctx["news"]["median_abs_day1"] * float(L["resid_sd"].get(tk, 0.0))
    cty_mv = {}
    for cty, dirn in lb.get("countries") or []:   # a country's own shock (#238): typical size, or the large end for a major event
        cp = (ctx.get("CP") or {}).get(cty, {})
        z = (cp.get("two_day_move_p90", 0.0) if lb.get("severity") in ("major", "extreme") else cp.get("two_day_move_median", 0.0))
        z *= np.sqrt(2) * (1 if dirn == "+" else -1)
        for t in names:
            if (ctx.get("CL") or {}).get(t, {}).get("country") == cty:
                cty_mv[t] = cty_mv.get(t, 0.0) + ctx["CL"][t]["loading"] * z
    legs, skipped, kept_over_sector = combine_legs(sec_mv, co_mv, cty_mv)
    for t, x in legs.items():
        if t in mv:
            mv[t] += x
    opt = {o["pos"].label: float(live.option_pnl(o, [float(o["beta"] @ d["f"]) + legs.get(o["pos"].underlying, 0.0)])[0])
           for o in ctx["opts"]}
    full = float(sum(float(v[t]) * mv[t] for t in held)) + sum(opt.values())
    # every holding's dollars in every draw: the draw's factor moves, the event's legs and the holding's own noise (shared
    # with its options, #235), options repriced in each draw; so each holding's average and 9-in-10 range are its own and
    # the holdings add up to the book draw by draw
    F = d["f"] + (d["draw"] - np.median(d["draw"], axis=0))
    if "_noise" not in ctx:
        ctx["_noise"] = (live.shared_noise(L, v, list(held) + [o["pos"].underlying for o in ctx["opts"]]) or {}).get("name", {})
    own = ctx["_noise"]
    pnl = {t: float(v[t]) * (F @ Bh.loc[t].to_numpy() + legs.get(t, 0.0) + own.get(t, 0.0)) for t in held}
    for o in ctx["opts"]:
        u = o["pos"].underlying
        pnl[o["pos"].label] = np.asarray(live.option_pnl(o, F @ np.asarray(o["beta"]) + legs.get(u, 0.0) + own.get(u, 0.0)), float)
    book = np.sum([np.broadcast_to(x, (len(F),)) for x in pnl.values()], axis=0)

    def spread(x) -> dict:
        x = np.broadcast_to(np.asarray(x, float), (len(F),))
        return {"mean_usd": float(x.mean()), "range_90_usd": [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))]}
    middle = {**{t: float(v[t]) * mv[t] for t in held}, **opt}
    pos = [{"ticker": k, "full_impact_usd": middle[k], **spread(x)} for k, x in pnl.items()]
    pos.sort(key=lambda r: -abs(r["mean_usd"]))
    return {"_lb": lb, "draw_pool": d.get("draw_pool"), "full_impact_usd": full, "bad_case_if_yes_1_in_20_usd": float(np.quantile(d["end"] - float(ctx["db"] @ d["f"]) + full, 0.05)),
            "book_if_yes": spread(book), "by_position": pos, "factor_moves": live.plain_moves(d["f"]),
            "company_legs_skipped": sorted(skipped), "company_legs_over_sector": sorted(kept_over_sector)}


def _with_odds(row: dict, p: float | None, source: str) -> dict:
    full = row["full_impact_usd"]
    row.update({"odds": p, "odds_source": source})
    if p is not None and source == "market":      # a market prices the outcome: split what's left
        # tier 1 (#233): the probability is the longshot-corrected price (#227); the priced-in split uses the raw price,
        # because that is what asset prices carry; the risk left uses the corrected probability
        from .planb import probability as PR
        c = PR.corrected(p, row.get("venue", "kalshi"))
        q = c["corrected"]
        row.update({"odds": q, "odds_raw": p, "tier": 1, "longshot_corrected": c["applied"],
                    "if_yes_usd": full * (1 - p), "if_no_usd": -full * p, "risk_left_usd": abs(full) * float(np.sqrt(q * (1 - q))),
                    "mostly_priced": "yes" if p >= 0.85 else ("no" if p <= 0.15 else None)})
    elif p is not None:                           # a base rate is about events of this KIND, not this exact one: context only
        row["odds_note"] = ("how often a big event of this kind happened within the horizon, 2016-2025; not the odds of "
                            "this exact event, and nothing is priced off it")
    return row


# ---------------------------------------------------------------- stream A: prediction markets

def market_candidates(ctx: dict, horizon_days: int = 91) -> list[dict]:
    from .planb import live
    s, lab, today = ctx["s"], labels(), ctx["today"]
    universe = _universe(s)
    rows = []
    for m in universe:
        lb = lab.get(m["id"])
        if not lb or not lb.get("moves_markets"):
            continue
        close = pd.Timestamp(m.get("close_date") or "2100-01-01")
        if not (today <= close <= today + pd.Timedelta(days=horizon_days)):
            continue
        p = live._p(m)
        if p is None or not (0.0 < p < 1.0):
            continue
        r = {"source": "market", "venue": m.get("venue", "kalshi"), "market_id": m["id"], "event": m.get("event_ticker"),
             "title": m.get("title") + (f" ({m['subtitle']})" if m.get("venue") == "polymarket" and m.get("subtitle") and m["subtitle"] not in (m.get("title") or "") else ""),
             "when": str(close.date()),
             "why_it_moves": lb.get("why"), **impact(ctx, lb)}
        rows.append(_with_odds(r, p, "market"))
    best = {}
    for r in rows:   # one market per event: the one with the most risk left
        if r["event"] not in best or r["risk_left_usd"] > best[r["event"]]["risk_left_usd"]:
            best[r["event"]] = r
    out = []
    for r in sorted(best.values(), key=lambda r: -r["risk_left_usd"]):   # the same question on both venues: keep one
        if not any(_same(r, o) for o in out):
            out.append(r)
    return out


def _universe(s) -> list[dict]:
    """Kalshi (the built universe, quotes refreshed hourly) plus Polymarket's topic-filtered feed when the data is live
    (replays pin a quote tag and stay Kalshi-only)."""
    try:
        ms = list(s.data.event_markets())
    except Exception:  # noqa: BLE001
        ms = []
    if not getattr(s.data, "synthetic", False) and getattr(s.data, "quote_tag", None) is None and hasattr(s.data, "build"):
        try:
            from .data import polymarket
            ms += polymarket.topical()
        except Exception:  # noqa: BLE001 — the second venue must never break discovery
            pass
    return ms


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 2 and w not in {"will", "the", "before", "2026", "2027", "by"}}


def _same(a: dict, b: dict) -> bool:
    if a.get("venue") == b.get("venue"):
        return False
    wa, wb = _words(a["title"]), _words(b["title"])
    close = abs((pd.Timestamp(a["when"]) - pd.Timestamp(b["when"])).days) <= 7
    return close and len(wa & wb) / max(1, len(wa | wb)) >= 0.5


def upcoming(s, horizon_days: int = 91, n: int = 8) -> dict:
    """Stream A's prediction markets alone (the upcoming_events tool, #218)."""
    s.require_portfolio()
    ctx = context(s)
    rows = market_candidates(ctx, horizon_days)
    for r in rows:
        r["odds_yes"], r["closes"] = r["odds"], r["when"]
    return _strip({"horizon_days": horizon_days, "book_value": float(ctx["vals"].sum()), "events_checked": len(rows), "rows": rows[:n],
            "how": "full impact = this book's move if the outcome were a total surprise (code's 'what if' engine); today's "
                   "odds already price part of it: if YES the book moves full x (1 - odds), if NO -full x odds; ranked by the "
                   "risk left to resolve, full x sqrt(odds x (1 - odds)); options repriced on their underlying's move (the "
                   "if-yes/if-no split is linear in the odds, a simplification for options)",
            "universe": "Kalshi markets in the built universe plus Polymarket's topic-filtered feed, closing within the horizon "
                        "and moving markets (labelled once by the LLM, judged on what a YES does; code makes every number)",
            "numbers": "every number is code's; quote them, never adjust them"})


# ---------------------------------------------------------------- stream A: the calendar

def _release_replay(ctx: dict, typ: str) -> dict | None:
    from .planb import live
    R = [r for r in live.pack()["releases"] if r["type"] == typ]
    if len(R) < 8:
        return None
    x = np.array([r["U"] for r in R]) @ ctx["db"]
    return {"typical_swing_usd": float(np.median(np.abs(x))), "bad_case_1_in_10_usd": float(np.quantile(x, 0.1)), "past_releases": len(R)}


def _us_election(today: date, horizon_days: int) -> date | None:
    for y in (today.year, today.year + 1):
        if y % 2:
            continue
        nov1 = date(y, 11, 1)
        d = nov1 + timedelta(days=(1 - nov1.weekday()) % 7)          # first Tuesday...
        if d.day == 1:
            d += timedelta(days=7)                                   # ...after the first Monday
        if today <= d <= today + timedelta(days=horizon_days):
            return d
    return None


def calendar(ctx: dict, horizon_days: int = 91) -> list[dict]:
    from . import natural, stocks
    from .planb import live
    s, today = ctx["s"], ctx["today"]
    out = []
    try:
        universe = s.data.event_markets()     # Kalshi lists each meeting and print: the release dates
    except Exception:  # noqa: BLE001
        universe = []
    nxt = {}
    for m in universe:                       # release dates from the market universe (Kalshi lists each meeting/print)
        key = RELEASE_SERIES.get(m.get("series_ticker"))
        close = pd.Timestamp(m.get("close_date") or "2100-01-01")
        if key and today <= close <= today + pd.Timedelta(days=horizon_days):
            if key[0] not in nxt or close < nxt[key[0]][0]:
                nxt[key[0]] = (close, key[1])
    for typ, (when, name) in nxt.items():
        rp = _release_replay(ctx, typ)
        if rp:
            out.append({"source": "calendar", "kind": "release", "title": f"Next {name}", "when": str(when.date()),
                        "swing_usd": rp["typical_swing_usd"], "bad_case_usd": rp["bad_case_1_in_10_usd"],
                        "how": f"this book replayed through {rp['past_releases']} past {name}s (today's stock-bond regime)"})
    try:                                     # held-name earnings (directly or through ETFs)
        eff = stocks._exposure(s)
        for t, r in eff.head(15).iterrows():
            if r["weight"] < 0.01 or not hasattr(s.data, "earnings"):
                continue
            e = s.data.earnings(t)
            nx = (e or {}).get("next")
            if not nx or not nx.get("date") or not (today <= pd.Timestamp(nx["date"]) <= today + pd.Timedelta(days=horizon_days)):
                continue
            past = [abs(m["move"]) for m in (e.get("past_moves") or [])][:8]
            if past:
                out.append({"source": "calendar", "kind": "earnings", "title": f"{t} earnings", "when": nx["date"],
                            "swing_usd": float(r["value"]) * float(np.median(past)),
                            "how": f"your ${float(r['value']):,.0f} in {t} (incl. through ETFs) x its median earnings-day move of the last {len(past)}"})
    except Exception:  # noqa: BLE001
        pass
    try:                                     # EIA and WASDE: the book's exposure x how big those days usually are (#212)
        R = json.loads((ROOT / "docs/pmrm/results/commod_releases.json").read_text(encoding="utf-8"))
        j_oil, j_ag = live.FACTORS.index("oil"), live.FACTORS.index("agriculture")
        oil_day = abs(ctx["db"][j_oil]) * ctx["fd"][j_oil]
        ag_day = abs(ctx["db"][j_ag]) * ctx["fd"][j_ag]
        for row in natural.release_calendar(today.date()):
            if "Petroleum" in row["release"] and oil_day > 0:
                out.append({"source": "calendar", "kind": "commodity release", "title": "EIA oil inventories", "when": row["when"],
                            "swing_usd": oil_day * R["EIA crude stocks | USO"]["to_close"]["size_vs_control_rms"],
                            "how": "your oil exposure x how big EIA days usually are vs a normal day"})
            if "WASDE" in row["release"] and ag_day > 0:
                out.append({"source": "calendar", "kind": "commodity release", "title": "USDA WASDE crop report", "when": row["when"],
                            "swing_usd": ag_day * R["USDA WASDE | DBA"]["to_close"]["size_vs_control_rms"],
                            "how": "your farm-commodity exposure x how big WASDE days usually are vs a normal day"})
    except Exception:  # noqa: BLE001
        pass
    d = _us_election(today.date(), horizon_days)
    if d:
        p = live.pack()
        U = np.array([e_U for e_U, e in zip(p["_U"], p["events"]) if e["category"] == "elections_politics" and e.get("severity") in ("major", "extreme")])
        if len(U):
            x = U @ ctx["db"]
            out.append({"source": "calendar", "kind": "election", "title": f"US elections ({d.isoformat()})", "when": d.isoformat(),
                        "swing_usd": float(np.median(np.abs(x))), "bad_case_usd": float(np.quantile(x, 0.1)),
                        "how": f"this book replayed through {len(U)} past big election and political events"})
    return sorted(out, key=lambda r: r["when"])


# ---------------------------------------------------------------- stream A: the news feed

@lru_cache(maxsize=1)
def _followup_rates() -> dict:
    """How often a big event of each category was followed by another big one of the same category within 30 days
    (library, 2016-2025): the base rate that a developing story produces a further market event."""
    from .planb import live
    ev = [e for e in live.pack()["events"] if e.get("severity") in ("major", "extreme") and "2016" <= e["date"][:4] <= "2025"]
    out = {}
    for c in live.CATEGORIES:
        dates = sorted(pd.Timestamp(e["date"]) for e in ev if e["category"] == c)
        if len(dates) >= 10:
            out[c] = float(np.mean([any(0 < (b - a).days <= 30 for b in dates[i + 1:i + 40]) for i, a in enumerate(dates)]))
    return out


def news_candidates(ctx: dict, days: int = 7) -> list[dict]:
    if not FEED.exists():
        return []
    since = (ctx["today"] - pd.Timedelta(days=days)).date()
    rates = _followup_rates()
    out, seen = [], set()
    for line in FEED.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["news_day"] < since.isoformat():
            continue
        e = r["event"]
        fs = ((e.get("direction") or {}).get("factor_signs")) or {}
        sev = e.get("severity") or "moderate"
        lb = {"severity": sev, "category": e.get("category"), "factor_signs": {k: ("+" if v > 0 else "-") for k, v in fs.items() if v},
              "factor_sizes": {k: SEV_SIZE.get(sev, "small") for k, v in fs.items() if v}}
        if not lb["factor_signs"] or (e.get("title") or "") in seen:
            continue
        seen.add(e.get("title") or "")
        row = {"source": "news", "title": f"Story develops further: {(e.get('title') or '')[:120]}", "when": f"reported {r['news_day']}",
               "category": e.get("category"), **impact(ctx, lb)}
        row = _with_odds(row, rates.get(e.get("category")), "base rate")
        if row.get("odds_note"):
            row["odds_note"] = "how often a big event of this kind was followed by another within 30 days, 2016-2025"
        out.append(row)
    return out


# ---------------------------------------------------------------- stream B: proposals

PROPOSE = """You are proposing the events that would move ONE portfolio the most over the next {days} days. Use the
book's biggest exposures below (from code: dollars per one-normal-day move of each factor, sector exposures, the largest
names including through ETFs, option positions). Propose {n} specific, plausible events (geopolitical, macro, policy,
sector or company) that would hit THESE exposures hardest, in either direction. Most should have a live catalyst; include
at least two tail risks: big for this book, plausible, but nothing listed below is their catalyst (links all null).

Today is {today}. For each, give what it would do to markets if it happened (directions and sizes only; code computes
every number). Link an upcoming item only when it is the catalyst: a market only if its YES is this exact event (its odds
are then used as this event's odds), a calendar item if the event happens at that release, a news story if the event is
that story developing further. Otherwise null: it's then a tail risk.

Return ONLY a JSON array:
[{{"title": "...", "category": "{cats}", "severity": "minor|moderate|major|extreme",
   "f": "<7 tokens for {factors}: sign (+, - or 0) then size (S, M or L), e.g. '-M +L 0S ...'>",
   "sectors": [["<sector>", "+|-"]], "companies": [["<TICKER>", "+|-"]], "countries": [["<country>", "+|-"]],
   "user": false, "why_this_book": "<at most 15 words>",{fh_field} "links": {{"market_id": "<id or null>", "calendar": <index or null>, "news": <index or null>}}}}]

Factors: market = US stocks; rates = Treasury yields ("+" = up); credit = high-yield minus Treasuries ("+" = spreads
tighten). Sectors from: {sectors}.
Countries: only those the event is about or directly hits (a war, a coup, sanctions, an election, a crisis in that
country), each with its own direction, from: {countries}. Never for global events. Usually empty.
Companies: only when the event is about that specific company (the company is the subject or is named as the direct
target: its drug, its earnings, its export licence, its CEO). Never for broad market, macro or sector events; their
effect on companies comes through the factors and sectors. Usually empty.

The book:
{book}

Upcoming items you may link (markets by id; calendar and news by index):
{items}{history_block}{user_block}"""

HISTORY_BLOCK = """

The book's history (code): the kinds of past events that hurt this book most (each past big event of that kind replayed
through today's exposures; bad case = the worst 1 in 10; the holdings that carry most of that bad case):
{rows}
Use it to choose what to propose: unless the user's events already cover them, at least 2 of your proposals must be
specific, current versions of the top categories (what would this kind of event look like now, and what's its catalyst?).
Set "from_history" to the category a proposal instantiates (one of the categories above), or null. Don't repeat these
numbers; code prices every proposal."""

USER_BLOCK = """

The user named these events themselves: {events}
FIRST return one object per user event, in the same format with "user": true, the title in the user's own words (made
specific only as far as they said), and links as above (a market only if its YES is exactly that event). THEN propose the
{n} other events with "user": false. They must be different from the user's events: not the same event, not a version of
it, and not another route to the same move (if the user named an oil spike, don't propose a Hormuz closure)."""


def exposures(ctx: dict) -> dict:
    from . import stocks
    from .planb import live
    db = ctx["db"]
    fac = sorted(((live.FACTOR_WORDS[f], float(db[j] * ctx["fd"][j])) for j, f in enumerate(live.FACTORS)), key=lambda x: -abs(x[1]))[:6]
    sec = {}
    for t, sl in ctx["SL"].items():
        if t in ctx["v"].index:
            sec[sl["sector"]] = sec.get(sl["sector"], 0.0) + float(ctx["v"][t]) * sl["loading"]
    try:
        names = [(t, float(r["value"])) for t, r in stocks._exposure(ctx["s"]).head(8).iterrows()]
    except Exception:  # noqa: BLE001
        names = [(t, float(ctx["v"][t])) for t in ctx["held"][:8]]
    return {"book_value": float(ctx["vals"].sum()), "factors_usd_per_normal_day": dict(fac),
            "sectors_usd_per_normal_sector_day": dict(sorted(sec.items(), key=lambda x: -abs(x[1]))[:5]),
            "largest_names_usd": dict(names), "options": [o["pos"].label for o in ctx["opts"]]}


def history(replay: list[dict] | None, k: int = 4) -> list[dict]:
    """The book's own history for the proposal call, from the risk card's replay (live.scan rows): the k one-off
    categories with the worst bad case, each with how often, what history says, the worst past example and the 2
    holdings that carry most of the bad case. Releases are left out: they're in the calendar."""
    rows = sorted([r for r in replay or [] if r.get("kind") == "one-off event"], key=lambda r: r["bad_case_1_in_10_usd"])[:k]
    out = []
    for r in rows:
        w = r.get("worst_past_example") or {}
        pos = sorted(((r.get("by_position") or {}).get("positions") or []), key=lambda p: p.get("in_the_books_bad_case_usd") or 0.0)
        out.append({"category": r["risk_id"].split(":", 1)[1], "kind": r.get("risk"),
                    "bad_case_1_in_10_usd": round(float(r["bad_case_1_in_10_usd"])), "how_often": r.get("how_often"),
                    "history_says": r.get("history_says"),
                    "worst_past_example": {"date": w.get("date"), "what": (w.get("what") or "")[:100],
                                           "this_book_usd": round(float(w["your_book_usd"])) if w.get("your_book_usd") is not None else None},
                    "holdings_carrying_the_bad_case": [p["ticker"] for p in pos[:2]]})
    return out


def propose(llm, ctx: dict, markets: list[dict], cal: list[dict], news: list[dict], days: int = 91, n: int = 6,
            user_events: list[str] | None = None, replay: list[dict] | None = None) -> list[dict]:
    """Stream B. With user_events, the same call first labels the user's own events (source "user") and then proposes
    n others that are not those events or other routes to the same move. With replay (the risk card's
    rows), the prompt carries the book's own history and each proposal says which category it instantiates
    (from_history, validated in code; its bad case is the replay's number, never the LLM's)."""
    from . import config
    from .planb import live
    from .agent.session import ToolError  # noqa: F401
    sectors = list(ctx["SP"])
    user_events = [str(u)[:200] for u in (user_events or []) if str(u).strip()][:4]
    shown = markets[:30]
    for u in user_events:     # the markets that might BE a user's event, wherever they rank
        wu = _words(u)
        shown += sorted([m for m in markets[30:] if m not in shown and wu & _words(m["title"])],
                        key=lambda m: -len(wu & _words(m["title"])))[:8]
    items = {"markets": [{"id": m["market_id"], "title": m["title"], "closes": m["when"]} for m in shown],
             "calendar": [{"index": i, "title": c["title"], "when": c["when"]} for i, c in enumerate(cal)],
             "news": [{"index": i, "title": x["title"][:110]} for i, x in enumerate(news)]}
    hist = history(replay)
    bad = {r["risk_id"].split(":", 1)[1]: float(r["bad_case_1_in_10_usd"]) for r in replay or [] if r.get("kind") == "one-off event"}
    prompt = PROPOSE.format(today=ctx["today"].date().isoformat(), days=days, n=n, cats="|".join(live.CATEGORIES), factors=", ".join(live.SIGNED), sectors=", ".join(sectors),
                            countries=", ".join(ctx.get("CP") or {}),
                            book=json.dumps(exposures(ctx), default=str), items=json.dumps(items, default=str, ensure_ascii=False),
                            fh_field=' "from_history": "<category or null>",' if hist else "",
                            history_block=HISTORY_BLOCK.format(rows="\n".join(json.dumps(h, ensure_ascii=False) for h in hist)) if hist else "",
                            user_block=USER_BLOCK.format(events=json.dumps(user_events, ensure_ascii=False), n=n) if user_events else "")
    resp = llm.create(model=config.EXPERT_MODEL, max_tokens=3500, messages=[{"role": "user", "content": prompt}], _source="discovery_propose")
    txt = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    try:
        arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    mk = {m["market_id"]: m for m in markets}
    rates = {c: v for c, v in _big_event_rates().items()}
    out = []
    for it in arr if isinstance(arr, list) else []:
        if not isinstance(it, dict):
            continue
        toks = str(it.get("f") or "").split()
        if len(toks) != len(live.SIGNED) or it.get("category") not in live.CATEGORIES:
            continue
        signs = {f: t[0] for f, t in zip(live.SIGNED, toks) if t[:1] in "+-" and len(t) > 1}
        sizes = {f: {"S": "small", "M": "medium", "L": "large"}.get(t[1:2].upper(), "small") for f, t in zip(live.SIGNED, toks) if f in signs}
        lb = {"severity": it.get("severity") if it.get("severity") in live.SEVERITIES else "moderate", "category": it.get("category"),
              "factor_signs": signs, "factor_sizes": sizes,
              "sectors": [list(x[:2]) for x in (it.get("sectors") or []) if isinstance(x, list) and len(x) >= 2 and x[0] in sectors and x[1] in ("+", "-")][:2],
              "companies": [[str(x[0]).upper(), x[1]] for x in (it.get("companies") or []) if isinstance(x, list) and len(x) >= 2 and x[1] in ("+", "-")][:2],
              "countries": [[x[0], x[1]] for x in (it.get("countries") or []) if isinstance(x, list) and len(x) >= 2 and x[0] in (ctx.get("CP") or {}) and x[1] in ("+", "-")][:3]}
        links = it.get("links") or {}
        row = {"source": "user" if (it.get("user") is True and user_events) else "proposal", "title": str(it.get("title"))[:140], "category": it["category"], "why_this_book": it.get("why_this_book"),
               "links": links, **impact(ctx, lb)}
        fh = it.get("from_history") if hist else None
        row["from_history"] = fh if fh in live.CATEGORIES else None
        if row["from_history"]:
            row["from_history_words"] = live.CATEGORY_WORDS[fh]
            row["from_history_bad_case_usd"] = bad.get(fh)
        m = mk.get(links.get("market_id")) if links.get("market_id") else None
        if m:   # the catalyst, and a hint for the relation check; the odds come from the probability step (#234)
            row["linked_to"] = f"market: {m['title']}"
            row["link_market_id"] = m["market_id"]
            row = _with_odds(row, rates.get(it["category"]), "base rate")
        else:
            row = _with_odds(row, rates.get(it["category"]), "base rate")
            ci, ni = links.get("calendar"), links.get("news")
            if isinstance(ci, int) and 0 <= ci < len(cal):
                row["linked_to"] = f"calendar: {cal[ci]['title']} ({cal[ci]['when']})"
            elif isinstance(ni, int) and 0 <= ni < len(news):
                row["linked_to"] = f"news: {news[ni]['title'][:90]}"
        out.append(row)
    return out


@lru_cache(maxsize=1)
def _big_event_rates(horizon_days: int = 91) -> dict:
    """P(at least one big event of each kind within the horizon), from how often they happened 2016-2025 (library)."""
    from .planb import live
    ev = [e for e in live.pack()["events"] if e.get("severity") in ("major", "extreme") and "2016" <= e["date"][:4] <= "2025"]
    return {c: float(1 - np.exp(-sum(1 for e in ev if e["category"] == c) / 10 * horizon_days / 365)) for c in live.CATEGORIES}


# ---------------------------------------------------------------- the match

SIZE_RANK = {"small": 1, "medium": 2, "large": 3}


def _main_moves(lb: dict | None) -> set[tuple[str, str]]:
    """An event's main moves: its largest-sized factors with their signs, e.g. {("oil", "+")}."""
    sg, sz = (lb or {}).get("factor_signs") or {}, (lb or {}).get("factor_sizes") or {}
    if not sg:
        return set()
    top = max(SIZE_RANK.get(sz.get(f), 1) for f in sg)
    return {(f, d) for f, d in sg.items() if SIZE_RANK.get(sz.get(f), 1) == top}


def same_move(user_lb: dict | None, cand_lb: dict | None) -> bool:
    """A candidate is another route to the user's event when their main moves share a factor and direction other than
    the broad market (most risk-off events share "stocks down"); if the user's event is only a stock-market move, the
    candidate must share it."""
    um, shared = _main_moves(user_lb), _main_moves(user_lb) & _main_moves(cand_lb)
    return any(f != "market" for f, _ in shared) or (bool(um) and um == shared)


def _verdict_words(full: float, swing: float) -> str:
    r = abs(full) / swing if swing else 0.0
    return "noise for your book" if r < 0.5 else ("small for your book" if r < 1 else "meaningful for your book")


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "_lb"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x

def _replay(s, horizon_days: int) -> list[dict]:
    """The risk card's replay (live.scan, code only), every category; the session's stored card scan is left as it was."""
    from .planb import live
    had, prev = "planb_scan" in s.__dict__, s.__dict__.get("planb_scan")
    try:
        return live.scan(s, horizon_days, n=len(live.CATEGORIES) + 5)["risks"]
    finally:
        if had:
            s.__dict__["planb_scan"] = prev
        else:
            s.__dict__.pop("planb_scan", None)


TIER_WORDS = {1: "market", "2a": "derived from markets", "2b": "bounded by markets", 3: "Worsey forecast"}


def _set_prob(r: dict, p: float, tier, label: str, **extra) -> None:
    full = r["full_impact_usd"]
    r.pop("odds_note", None)
    r.update({"odds": p, "odds_source": TIER_WORDS[tier], "tier": tier, "odds_label": label,
              "if_yes_usd": full * (1 - p), "if_no_usd": -full * p, "risk_left_usd": abs(full) * float(np.sqrt(p * (1 - p))), **extra})
    if tier == 3:
        r["priced_in_note"] = "no market prices this: the if-yes/if-no split assumes markets share this forecast"


def tier_probabilities(llm, ctx: dict, universe: list[dict], rows: list[dict], news: list[dict]) -> dict:
    """Tiers 1, 2a, 2b for every proposed or user event, in one batched relation call (#233, #234). Candidates per event:
    its top keyword-matched markets, the market the proposal call linked (a hint to check, never the answer), and the
    week's news stories (so a proposal that IS a listed story can be deduped). The only owner of these events' odds."""
    from .planb import probability as PR
    need = [r for r in rows if r.get("source") in ("proposal", "user")]
    stats = {"events": len(need), "tier1": 0, "tier2a": 0, "tier2b": 0, "tier3": 0, "base_rate_only": 0}
    if llm is None or not need:
        return stats
    today = ctx["today"].date()
    by_id = {m["id"]: m for m in universe}
    news_c = [{"id": f"news:{i}", "title": n["title"].replace("Story develops further: ", "")[:140], "close_date": None}
              for i, n in enumerate(news)]
    items, mkts = [], []
    for r in need:
        c = PR.candidates_for(r["title"], universe)
        h = by_id.get(r.get("link_market_id"))
        if h and h not in c and PR._p(h) is not None:
            c = [h] + c
        mkts.append(c)
        items.append((r["title"], c + news_c[:12]))
    try:
        rel = PR.classify_many(llm, items, today)
    except Exception:  # noqa: BLE001 — a failed call leaves the forecaster / base-rate path
        rel = {}
    for i, r in enumerate(need):
        rr = rel.get(i, [])
        r["same_news"] = [int(x["id"].split(":")[1]) for x in rr if x["id"].startswith("news:") and x["relation"] == "same"]
        tr = PR.tier_from([x for x in rr if not x["id"].startswith("news:")], mkts[i], universe, today)
        if tr["tier"] in (1, "2a", "2b"):
            _set_prob(r, tr["p"], tr["tier"], tr["label"], **({"odds_raw": tr["raw"]} if tr.get("raw") is not None else {}),
                      **({"bounds": tr["bounds"]} if tr.get("bounds") else {}))
            if tr["tier"] == 1:
                r["market_id"] = tr["market"]
            stats[f"tier{tr['tier']}"] += 1
        else:
            r["bounds"] = tr["bounds"]
            by = {m["id"]: m for m in mkts[i]}
            rels = [x for x in rr if x["relation"] in ("correlated", "E_implies_M", "M_implies_E", "exclusive") and x["id"] in by][:3]
            r["related_markets"] = [{"title": f"{by[x['id']].get('title')} {by[x['id']].get('subtitle') or ''}".strip()[:120],
                                     "p": PR.corrected(PR._p(by[x["id"]]), by[x["id"]].get("venue", "kalshi"))["corrected"],
                                     "relation": x["relation"]} for x in rels]
    return stats


def forecast_probabilities(llm, ctx: dict, yours: list[dict], castle: list[dict], thr: float, horizon_days: int = 91,
                           stats: dict | None = None) -> dict:
    """Tier 3: every user event still unpriced (always: the user asked), and the largest WORSEY_FORECAST_MAX_PER_SCAN (3)
    of Castle's unpriced proposals that pass the size screen. In parallel. Folded or deduped events are already gone."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from .planb import forecaster as FC
    stats = stats if stats is not None else {"tier3": 0}
    if llm is None:
        return stats
    n_castle = int(os.environ.get("WORSEY_FORECAST_MAX_PER_SCAN", 3))
    todo = [r for r in yours if r.get("tier") is None]
    todo += sorted([r for r in castle if r.get("source") == "proposal" and r.get("tier") is None and abs(r["full_impact_usd"]) >= thr],
                   key=lambda r: -abs(r["full_impact_usd"]))[:n_castle]
    today = ctx["today"].date().isoformat()

    def run(r):
        try:
            return FC.forecast(llm, r["title"], r.get("category"), horizon_days, today, r.get("bounds"), related=r.get("related_markets"))
        except Exception:  # noqa: BLE001
            return {"p": None}
    if todo:
        with ThreadPoolExecutor(len(todo)) as ex:
            for r, f in zip(todo, ex.map(run, todo)):
                if f.get("p") is not None:
                    _set_prob(r, f["p"], 3, "Worsey forecast: 3 agents (library base rate, similar past events, last 14 days' news); "
                                            "no market lists this", forecast={k: f[k] for k in ("agents", "spread", "clipped_to_bounds")})
                    stats["tier3"] = stats.get("tier3", 0) + 1
    if "events" in stats:
        stats["base_rate_only"] = stats["events"] - sum(stats.get(k, 0) for k in ("tier1", "tier2a", "tier2b", "tier3"))
    stats["forecast_runs"] = len(todo)
    return stats


def choose_three(pool: list[dict], n: int = 3) -> list[dict]:
    """Castle's picks: by risk left (|full move| x sqrt(p(1-p))), among events with a real probability (a
    market tier or a forecast; a base rate describes a kind of event, so it isn't eligible), skipping one that is the same
    move as a pick already made (so the three aren't three versions of "oil spikes")."""
    eligible = sorted([r for r in pool if r.get("tier") is not None and r.get("risk_left_usd") is not None],
                      key=lambda r: -r["risk_left_usd"])
    out = []
    for r in eligible:
        if any(r.get("_lb") and o.get("_lb") and same_move(o["_lb"], r["_lb"]) for o in out):
            continue
        out.append(r)
        if len(out) == n:
            break
    return out


def watchlist(s, llm=None, horizon_days: int = 91, user_events: list[str] | None = None, replay: list[dict] | None = None,
              progress=None) -> dict:
    """Both streams, priced by the same code, matched into: headline (topical and impactful), tail risks (impactful,
    no live catalyst), and noise for you (topical, too small for this book). The user's own events (user_events) come
    first and are always shown; anything that is the same event or another route to its main move is folded under it,
    so Castle's groups are everything OTHER than what the user asked about. replay = the risk card's
    rows (live.scan); the proposal call gets the book's own history from it, computed here when not given.
    progress(stage, data), if given, is called as stages finish ("listed", "proposed", "priced", "researching") so the
    chat can send messages as it goes (#234). Output: the user's events, Castle's three picks by risk left, and how many
    more were looked at."""
    s.require_portfolio()
    ctx = context(s)
    markets = market_candidates(ctx, horizon_days)
    cal = calendar(ctx, horizon_days)
    news = news_candidates(ctx)
    if progress:
        progress("listed", {"ctx": ctx, "markets": markets, "exposures": exposures(ctx)})
    props, err = [], None
    if llm is not None:
        try:
            if replay is None:
                replay = _replay(s, horizon_days)
            props = propose(llm, ctx, markets, cal, news, horizon_days, user_events=user_events, replay=replay)
        except Exception as e:  # noqa: BLE001 — proposals must never break the scan
            props, err = [], f"{type(e).__name__}: {e}"[:200]
    thr = 0.5 * ctx["swing"]
    yours = [p for p in props if p["source"] == "user"]
    props = [p for p in props if p["source"] != "user"]
    for u in yours:           # fold the same event, and other routes to its move, under the user's event
        u["verdict"] = _verdict_words(u["full_impact_usd"], ctx["swing"])
        rel = []

        def folded(r, u=u, rel=rel):
            hit = bool(r.get("market_id") and r.get("market_id") == u.get("market_id")) or same_move(u["_lb"], r.get("_lb"))
            if hit and r.get("source") != "proposal":
                rel.append({k: r.get(k) for k in ("source", "venue", "title", "when", "odds", "full_impact_usd")})
            return hit
        markets = [r for r in markets if not folded(r)]
        news = [r for r in news if not folded(r)]
        props = [r for r in props if not folded(r)]
        u["related"] = sorted(rel, key=lambda r: -abs(r.get("full_impact_usd") or 0))[:4]
    if progress:
        progress("proposed", {"yours": yours, "swing": ctx["swing"], "n_props": len(props)})

    def size(r):
        return abs(r.get("full_impact_usd", r.get("swing_usd", 0.0)))

    def rank(r):
        return r["risk_left_usd"] if r.get("risk_left_usd") is not None else size(r)

    universe = _universe(s) if (llm is not None and (yours or props)) else []
    prob_stats = tier_probabilities(llm, ctx, universe, yours + props, news)
    if progress:
        progress("priced", {"yours": yours, "swing": ctx["swing"]})
    # dedupe in code (#234): a proposal that IS a listed market (tier 1 on it) or a listed news story absorbs that item;
    # the proposal's framing stays, with the market's (corrected) odds
    drop_m, drop_n = set(), set()
    for p in props:
        hit = [m for m in markets if p.get("tier") == 1 and m.get("market_id") == p.get("market_id")]
        dn = [news[i] for i in p.get("same_news", []) if 0 <= i < len(news)]
        drop_m.update(id(m) for m in hit)
        drop_n.update(id(n_) for n_ in dn)
        if hit or dn:
            p["absorbs"] = [m["title"] for m in hit] + [n_["title"] for n_ in dn]
    markets = [m for m in markets if id(m) not in drop_m]
    news = [n_ for n_ in news if id(n_) not in drop_n]

    # topical vs tail in code (#234): topical = a market prices or derives it (tiers 1, 2a, 2b) or it names a calendar or
    # news catalyst; tail = everything else
    def is_topical(p):
        return p.get("tier") in (1, "2a", "2b") or str(p.get("linked_to") or "").startswith(("calendar:", "news:"))
    topical = markets + cal + news + [p for p in props if is_topical(p)]
    tails = [p for p in props if not is_topical(p) and size(p) >= thr]
    big_props = [p for p in props if size(p) >= thr]
    if progress:
        unpriced = sorted([p for p in big_props if p.get("tier") is None], key=lambda r: -size(r))[:3]
        progress("researching", {"forecasting": [r["title"] for r in yours if r.get("tier") is None] + [r["title"] for r in unpriced]})
    prob_stats = forecast_probabilities(llm, ctx, yours, big_props, thr, horizon_days, prob_stats)
    head = sorted([r for r in topical if size(r) >= thr], key=lambda r: -rank(r))
    seen, headline = set(), []
    for r in head:
        if r["title"] not in seen:
            seen.add(r["title"])
            headline.append(r)
    tails = sorted(tails, key=lambda r: -rank(r))
    for r in headline + tails:
        r["at_stake_usd"] = r.get("risk_left_usd") if r.get("risk_left_usd") is not None else size(r)
    noise = sorted([r for r in topical if size(r) < thr], key=lambda r: -(r.get("odds") or 1.0))
    picks = choose_three(headline + tails)
    more = [r for r in headline + tails if r not in picks]
    return _strip({"horizon_days": horizon_days, "book_value": float(ctx["vals"].sum()), "normal_2_day_swing_usd": ctx["swing"],
            "impact_threshold_usd": thr, "exposures": exposures(ctx),
            "yours": yours, "picks": picks, "more_count": len(more),
            "more": [{k: r.get(k) for k in ("title", "full_impact_usd", "odds", "tier")} for r in more[:10]],
            "headline": headline[:8], "tail_risks": tails[:5], "noise_for_you": noise[:6],
            "counts": {"markets": len(markets), "calendar": len(cal), "news": len(news), "proposals": len(props)},
            "proposals_error": err, "as_of": str(ctx["today"].date()), "probabilities": prob_stats, "user_events": user_events or [],
            "how": "stream A = what's coming (prediction markets, the calendar, the last week's news); stream B = events the "
                   "LLM proposed from this book's biggest exposures. Every candidate priced by code (the 'what if' engine). "
                   "Headline = topical and moves this book by at least half its normal two-day swing; tail risks = big for "
                   "this book but no live catalyst; noise for you = coming up but too small for this book. Odds carry a tier "
                   "(Part II): 1 = a market is the event, its price corrected for the measured longshot bias (odds_raw = the "
                   "market price; if yes: full x (1 - raw), if no: -full x raw); 2a = derived from a market's strike ladder "
                   "or deadline rungs; 2b = bounded by logically related markets; 3 = Worsey forecast (three agents reading the "
                   "library and the last 14 days' news; no market lists it; the priced-in split then assumes markets share "
                   "the forecast); otherwise a labelled base rate (how often big events of that kind happened, or how often "
                   "such stories developed further), context only. Say which tier each probability is. Yours = the "
                   "user's own events, priced the same way and always shown; 'related' under each = markets and stories that "
                   "are the same event or another route to its main move, folded there so the other groups are everything else. "
                   "Proposals are also chosen from the book's own history (the risk card's replay of past big events): "
                   "'from history' = the kind of past event a proposal is a current version of, with that kind's 1-in-10 bad "
                   "case for this book from the replay.",
            "numbers": "every number is code's; quote them, never adjust them"})
