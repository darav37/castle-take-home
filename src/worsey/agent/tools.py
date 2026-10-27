"""Agent tools (SPEC §7, §1b). Plain Python; every number the agent can say comes from here.

Each tool = strict JSON schema + handler(session, **input) → compact JSON. Tools are grouped
by who may call them: the router (front desk) or one of the experts.
"""

from __future__ import annotations

import json
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .. import charts, hedge, lookthrough
from ..options import book, builder, market_state
from ..options.view import View, ViewError
from ..portfolio import Portfolio, by_asset_class
from ..scenario import pipeline
from .session import Session, ToolError

# ---------------------------------------------------------------- compaction


def compact(x, sig: int = 4):
    """Round floats to `sig` significant figures, recursively; drop NaN/inf to None/'inf'."""
    if isinstance(x, bool) or x is None or isinstance(x, (str, int)):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "unbounded" if x > 0 else "-unbounded"
        return float(f"{x:.{sig}g}")
    if isinstance(x, dict):
        return {str(k): compact(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [compact(v, sig) for v in x]
    if hasattr(x, "item"):
        return compact(x.item(), sig)
    return str(x)


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


# ---------------------------------------------------------------- router tools


def filled_chain(s: Session, u: str, S: float, expiry_from=None, expiry_to=None):
    """Chain with IV backed out, memoized per session (one fill per window, not one per tool call)."""
    key = (u, expiry_from, expiry_to, s.data.today())
    memo = s.__dict__.setdefault("_chains", {})
    if key not in memo:
        memo[key] = market_state.fill_iv(s.data.chain(u, expiry_from, expiry_to), S, s.data.today())
    return memo[key]


GROUP_BOOK = 100_000.0


def apply_channel(s: Session) -> None:
    """Privacy by construction (DECISIONS #110): in a group channel the member's book is scaled to $100k BEFORE any tool,
    expert or reply sees it, so real dollar amounts can't reach a group reply. In a DM (or after "post it here") the
    real book is used. Book-dependent caches are dropped when the scale changes."""
    real = s.priced_real
    if real is None:
        return
    total = float(real["value"].sum())
    view = real
    if s.channel == "group" and total > 0 and abs(total - GROUP_BOOK) > 0.5:
        view = real.copy()
        view["value"] = real["value"] * (GROUP_BOOK / total)
    if s.priced is None or not view["value"].equals(s.priced["value"]):
        s.priced = view
        s.last_scan = s.active_event = s.last_hedges = None


def load_portfolio(s: Session, name: str | None = None, holdings: list | None = None) -> dict:
    if name:
        pf = Portfolio.load(name)
    elif holdings:
        pf = Portfolio.from_dict({"name": "chat", "positions": holdings})
    else:
        raise ToolError("give a portfolio name or holdings")
    prices = s.data.prices(pf.tickers)
    missing = [t for t in pf.tickers if t not in prices]
    if missing:
        raise ToolError(f"no prices for {missing}")
    s.portfolio, s.priced_real, s.priced = pf, pf.priced(prices), None
    s.last_scan = s.active_event = None
    apply_channel(s)
    from ..options import scenario as OS
    s.__dict__["options"] = [OS.OptionPosition.from_dict(o) for o in pf.options]   # repriced in scenarios (#209)
    if s.__dict__.get("remember"):   # live chat: each load is a dated book snapshot (DECISIONS #114)
        from . import memory
        memory.snapshot_book(s)
    ac = by_asset_class(s.priced)
    scaled = s.channel == "group" and abs(float(s.priced_real["value"].sum()) - float(s.priced["value"].sum())) > 0.5
    return {"portfolio": pf.name, "total_value": float(s.priced["value"].sum()),
            **({"amounts": "per $100k of the member's book (group chat: real dollars go to their DM)"} if scaled else {}),
            "positions": s.priced[["value", "weight", "asset_class"]].reset_index().to_dict("records"),
            "by_asset_class": ac.reset_index().to_dict("records"), **({"synthetic_data": True} if s.data.synthetic else {}),
            **({"options": [{"position": o.label, "value_usd": OS.value_now(s, o)} for o in s.__dict__["options"]]}
               if s.__dict__["options"] else {})}


def name_exposure(s: Session, names: list[str]) -> list[dict]:
    """How much of the book is in specific companies, through every fund held, stock AND debt. Searches each fund's FULL latest holdings (no
    top-N, no 'other holdings' bucket) by ticker or company name; bond lines ("CRWV 9.75 10/01/31") count as debt."""
    vals, book = s.priced["value"], float(s.priced["value"].sum())
    out = []
    for q in names:
        Q, ql = q.strip().upper(), q.strip().lower()
        parts, dates = [], set()
        if Q in vals.index:
            parts.append({"via": "held directly", "kind": "stock", "weight_in_fund": 1.0, "usd": float(vals[Q]), "lines": 1})
        for etf in vals.index:
            h = s.data.holdings(etf, sectors=False)
            if h is None or h.empty or etf == Q:
                continue
            tk = h["ticker"].astype(str).str.upper()
            nm = (h["name"] if "name" in h else h["ticker"]).astype(str).str.lower()
            hit = h[(tk == Q) | tk.str.startswith(Q + " ") | (nm.str.contains(ql, regex=False) if len(ql) >= 4 else False)]
            if hit.empty:
                continue
            dates |= set(hit.get("effective_date", pd.Series(dtype=str)).astype(str))
            for kind, g in hit.groupby(hit.get("asset_class", pd.Series("Equity", index=hit.index)).astype(str).str.lower()
                                       .map(lambda a: "debt" if "fixed" in a or "bond" in a else "stock")):
                w = float(g["weight"].sum())
                parts.append({"via": etf, "kind": kind, "weight_in_fund": w, "usd": float(vals[etf]) * w, "lines": int(len(g)),
                              "examples": (g["name"] if "name" in g else g["ticker"]).astype(str).head(3).tolist()})
        usd = sum(p["usd"] for p in parts)
        out.append({"name": q, "total_usd": usd, "share_of_book": usd / book if book else None,
                    "stock_usd": sum(p["usd"] for p in parts if p["kind"] == "stock"),
                    "debt_usd": sum(p["usd"] for p in parts if p["kind"] == "debt"), "through": parts,
                    "holdings_as_of": sorted(dates)[-1] if dates else None,
                    **({} if parts else {"note": f"not held directly and not in any held fund's latest holdings: {sorted(set(vals.index))}"})})
    return out


def exposures(s: Session, top: int = 8, names: list | None = None) -> dict:
    s.require_portfolio()
    if names:   # a specific company: full holdings, every fund, stock and debt
        return {"names": name_exposure(s, names)}
    vals = s.priced["value"]
    # equity ETFs only: bond/gold funds look through to bond lines or a trust, which says nothing about name concentration
    eq = s.priced.index[s.priced["asset_class"] == "equity"]
    hold = {t: h for t in eq if (h := s.data.holdings(t)) is not None and not h.empty}
    ac = s.priced["asset_class"]
    # non-equity positions (TLT, GLD, HYG…) are their own asset class, not "unknown" equities
    sector_of = lambda n: (f"({ac[n]})", None) if n in ac.index and ac[n] != "equity" else s.data.sector(n)
    eff = lookthrough.effective_exposure(vals, hold, sector_of=sector_of)
    conc = lookthrough.concentrations(eff, top=top)
    return {"top_names": conc.reset_index(names="name").to_dict("records"),
            "sectors": lookthrough.sector_exposure(eff).to_dict(),
            "industries": lookthrough.industry_exposure(eff).head(5).to_dict(),
            "etfs_looked_through": sorted(hold), "etfs_without_holdings": sorted(set(vals.index) - set(hold) - set(eff.index))}


def hedge_exposures(s: Session, pos_moves: dict, underlyings: list[str]) -> dict:
    """β-weighted dollar exposure of the scenario-affected book (positions that lose if YES) to each hedge underlying:
    Σ Vᵢ·βᵢ,und with βᵢ,und from the last year of daily returns."""
    from .. import config
    vals = s.priced["value"]
    affected = [t for t in vals.index if pos_moves.get(t, 0.0) < 0]
    und = [u for u in dict.fromkeys(underlyings) if u]
    if not affected or not und:
        return {}
    r = s.data.closes(list(dict.fromkeys(list(vals.index) + und)), days=config.HEDGE_BETA_DAYS + 5).pct_change().dropna(how="all")
    out = {}
    for u in und:
        if u not in r:
            continue
        var = float(r[u].var())
        betas = {t: (float(r[t].cov(r[u])) / var if var > 0 and t in r else 0.0) for t in affected}
        out[u] = abs(sum(float(vals[t]) * betas[t] for t in affected))
    return out


def compare_hedges_tool(s: Session, event_id: str | None = None, bad_case_weight: float = 1.0) -> dict:
    s.require_portfolio()
    today = s.data.today()
    tmpl = s.__dict__.get("active_template")
    weak = s.__dict__.get("template_weak_match")
    pb = s.__dict__.get("planb_scenario")
    if pb is not None and event_id in (None, pb["id"]):
        return _planb_hedges(s, pb, bad_case_weight)
    if tmpl is not None and s.active_event is None and event_id in (None, tmpl["type"], weak):
        # unpriced template event (no market): hedge the conditional loss over a 3-month horizon, no contract row
        from .. import config
        ref = s.data.closes(["SPY", "TLT", "GLD", "UUP", "USO"], days=2400)
        from ..scenario import impact
        for a in tmpl["analogs"]:  # reference assets' analog moves too, so index/cross legs can be sized
            mv = impact.window_moves(ref, pd.Timestamp(a["date"]), 1, a.get("after", 1))
            for t in ref.columns:
                a["moves"].setdefault(t, float(mv[t]))
        typ = [a for a in tmpl["analogs"] if a.get("typical")] or tmpl["analogs"]   # the median episode(s)
        mv_yes = {t: float(np.mean([a["moves"].get(t, 0.0) for a in typ]))
                  for t in set().union(*[a["moves"] for a in typ])}
        sc = hedge.Scenario(tmpl["type"], None, today + timedelta(days=config.HORIZON_DAYS),
                            tmpl["pnl_if_it_happens"], 0.0, mv_yes, {t: 0.0 for t in mv_yes})
        row = None
    else:
        # the event under discussion survives a portfolio reload ("about 60k" → hedge the same event at the new size)
        eid0 = event_id or s.active_event or s.__dict__.get("last_event_id")
        if not s.last_scan:
            if not eid0:
                raise ToolError("no event scan yet; ask event_analyst to scan first")
            scan_events(s)   # a named event after a reload: rebuild the scan for the new book, then price it below
        eid = eid0
        rows = {e["event_id"]: e for e in s.last_scan["events"] + s.last_scan.get("drift", [])}
        if eid not in rows:  # e.g. the portfolio was just reloaded at the real size: re-price that market for the new book
            m = next((m for m in s.data.event_markets() if m["id"] == eid), None)
            if m is not None:
                _price_into_scan(s, m, flag="repriced_for_hedge")
                rows = {e["event_id"]: e for e in s.last_scan["events"] + s.last_scan.get("drift", [])}
        if eid not in rows:
            raise ToolError(f"unknown event {eid}; known: {sorted(rows)}")
        s.active_event = eid
        row, betas = rows[eid], s.last_scan["betas"][eid]
        p = row["p"]
        mv_yes = {t: b["used"] * (1 - p) for t, b in betas.items()}
        mv_no = {t: -b["used"] * p for t, b in betas.items()}
        sc = hedge.Scenario(eid, p, pd.Timestamp(row["close_date"]).date(), row["pnl_yes"], row["pnl_no"], mv_yes, mv_no)

    def und(t):
        if t is None:
            return None
        S = s.data.prices([t])[t]
        from .. import config
        ch = filled_chain(s, t, S, sc.resolution, sc.resolution + pd.Timedelta(days=config.HEDGE_EXPIRY_PAD_DAYS).to_pytimedelta())
        return hedge.Underlying(t, S, builder.ChainQuoter(ch))

    # sector leg = the held equity ETF/name (other than SPY) that loses most if YES
    eq = s.priced[s.priced["asset_class"] == "equity"].index
    contrib = {t: s.priced.at[t, "value"] * mv_yes.get(t, 0.0) for t in eq if t != "SPY"}
    sector = min(contrib, key=contrib.get) if contrib and min(contrib.values()) < 0 else None
    crosses = hedge.pick_cross_asset(mv_yes)
    expo = hedge_exposures(s, {t: mv_yes.get(t, 0.0) for t in s.priced.index}, ["SPY", sector, *crosses])
    book = float(s.priced["value"].sum())
    out = hedge.compare_hedges(sc, today, index=und("SPY"), sector=und(sector), cross=[und(t) for t in crosses],
                               book_value=book, exposures=expo)
    if sc.p is None:
        # unpriced template: also price the SAME hedges against the worst analog
        worst = min(tmpl["analogs"], key=lambda a: a["book_pnl"])
        mv_w = {t: float(worst["moves"].get(t, 0.0)) for t in mv_yes}
        sc_w = hedge.Scenario(f"{tmpl['type']} (stress: {worst['date']})", None, sc.resolution, worst["book_pnl"], 0.0,
                              mv_w, {t: 0.0 for t in mv_w})
        stress = hedge.compare_hedges(sc_w, today, index=und("SPY"), sector=und(sector), cross=[und(t) for t in crosses],
                                      book_value=book, exposures=expo)
        out["case"] = "typical (median episode)"
        out["stress"] = {"case": f"worst analog: {worst['what']} ({worst['date']})", "loss_if_it_happens": worst["book_pnl"],
                         "rows": stress["rows"],
                         "note": "same hedges, same sizes, priced against the worst past episode instead of the typical (median) one"}
    # one convention: the agent sees your book after each hedge, never the hedge leg's own P&L
    for rows in (out.get("rows") or [], (out.get("stress") or {}).get("rows") or []):
        for r in rows:
            r.pop("pnl_if_yes", None)
            r.pop("pnl_if_no", None)
    s.last_hedges = out
    return out


def chart(s: Session, kind: str, ref: str | None = None) -> dict:
    if kind == "scan":
        if not s.last_scan:
            raise ToolError("no scan to chart")
        return {"path": charts.scenario_bars(s.last_scan["events"])}
    if not s.last_strategies:
        raise ToolError("no strategies to chart")
    cands = {c.id: c for c in s.last_strategies["candidates"]}
    c = cands.get(ref) or next(iter(cands.values()))
    return {"path": charts.payoff(c.strategy, s.last_strategies.get("_dist"), name=f"payoff_{c.id}")}


def _planb_hedges(s: Session, pb: dict, bad_case_weight: float = 1.0) -> dict:
    """Hedges for a Plan B scenario (DECISIONS #179): the same comparator, priced against the scenario's middle case
    (if it happens) and against its bad case (1 in 20), both from code. No prediction-contract row; odds, when the
    expert linked a Kalshi market, are reported beside the table."""
    from .. import config
    today = s.data.today()
    horizon = today + timedelta(days=config.HORIZON_DAYS)
    zero = {t: 0.0 for t in pb["moves_center"]}
    sc = hedge.Scenario(pb["id"], None, horizon, pb["pnl_center"], 0.0, pb["moves_center"], zero)
    sc_bad = hedge.Scenario(pb["id"] + " (bad case)", None, horizon, pb["pnl_bad"], 0.0, pb["moves_bad"], zero)

    def und(t):
        if t is None:
            return None
        S = s.data.prices([t])[t]
        ch = filled_chain(s, t, S, horizon, horizon + pd.Timedelta(days=config.HEDGE_EXPIRY_PAD_DAYS).to_pytimedelta())
        return hedge.Underlying(t, S, builder.ChainQuoter(ch))

    eq = s.priced[s.priced["asset_class"] == "equity"].index
    contrib = {t: s.priced.at[t, "value"] * pb["moves_center"].get(t, 0.0) for t in eq if t != "SPY"}
    sector = min(contrib, key=contrib.get) if contrib and min(contrib.values()) < 0 else None
    crosses = hedge.pick_cross_asset(pb["moves_center"])
    expo = hedge_exposures(s, {t: pb["moves_center"].get(t, 0.0) for t in s.priced.index}, ["SPY", sector, *crosses])
    book = float(s.priced["value"].sum())
    legs = dict(index=und("SPY"), sector=und(sector), cross=[und(t) for t in crosses], book_value=book, exposures=expo)
    out = hedge.compare_hedges(sc, today, **legs)
    bad = hedge.compare_hedges(sc_bad, today, **legs)
    out["case"] = "middle case (if it happens)"
    out["scenario"] = pb["title"]
    out["stress"] = {"case": "bad case (1 in 20, end of day)", "loss_if_it_happens": pb["pnl_bad"], "rows": bad["rows"],
                     "note": "same hedges, same sizes, priced against the scenario's bad case instead of its middle"}
    if pb.get("p") is not None:
        out["kalshi_odds"] = {"p": pb["p"], "closes": pb.get("closes")}
    for rows in (out.get("rows") or [], out["stress"]["rows"]):
        for r in rows:
            r.pop("pnl_if_yes", None)
            r.pop("pnl_if_no", None)
    from ..planb import live
    out["should_you_hedge"] = live.hedge_verdict(pb, out.get("rows") or [], bad_case_weight)   # code's verdict (#200)
    s.last_hedges = out
    return out


def event_evidence(s: Session, title: str, category: str, severity: str, direction: dict | None = None,
                   region: str | None = None, targeted_sectors: list | None = None, anticipation: str | None = None,
                   factor_signs: dict | None = None, factor_sizes: dict | None = None, market_moving: bool = True) -> dict:
    from ..planb import live
    spec = {"title": title, "category": category, "severity": severity, "direction": direction or {}, "region": region,
            "targeted_sectors": targeted_sectors or [], "anticipation": anticipation or "surprise", "factor_signs": factor_signs or {},
            "factor_sizes": factor_sizes or {}, "market_moving": market_moving}
    return live.evidence(s, spec)


def stress_scenario_tool(s: Session, title: str, shocks: dict, why: dict | None = None) -> dict:
    from ..planb import live
    return live.stress_scenario(s, title, shocks, why)


def news_base_rates_tool(s: Session, news_type: str, ticker: str | None = None) -> dict:
    from .. import stocks
    return stocks.news_base_rates(s, news_type, ticker)


def upcoming_events_tool(s: Session, horizon_days: int = 91, n: int = 8) -> dict:
    from .. import discovery
    return discovery.upcoming(s, horizon_days=horizon_days, n=n)


def earnings_calendar_tool(s: Session, days: int = 45) -> dict:
    from .. import stocks
    return stocks.earnings_calendar(s, days=days)


_COUNTRIES = list(json.loads((Path(__file__).resolve().parents[1] / "planb" / "countries.json").read_text(encoding="utf-8"))["countries"])


def price_scenario(s: Session, market_id: str | None = None, already_reported_before_open: bool = False,
                   company_news: list | None = None, sector_news: list | None = None, country_news: list | None = None) -> dict:
    from ..planb import live
    return live.price(s, market_id=market_id, already_reported_before_open=already_reported_before_open, company_news=company_news,
                      sector_news=sector_news, country_news=country_news)


def scan_risks_planb(s: Session, horizon_days: int = 91, n: int = 6) -> dict:
    from ..planb import live
    return live.scan(s, horizon_days=horizon_days, n=n)


# ---------------------------------------------------------------- event_analyst tools


def short_name(s: Session, e: dict) -> str:
    """Display name: the build's validated short name, else the cleaned full title (never truncated)."""
    from ..scenario import structural
    names = ((s.data.structure() or {}).get("names") or {}) if hasattr(s.data, "structure") else {}
    return names.get(e["event_id"]) or structural.clean_title(e.get("event") or e["event_id"])


def scan_events(s: Session, horizon_days: int = 90, n: int = 6) -> dict:
    s.require_portfolio()
    vals, ac = s.priced["value"], s.priced["asset_class"]
    closes = s.data.closes(list(vals.index) + pipeline.REFERENCE)
    mk = s.data.event_markets()
    hist = {m["id"]: s.data.market_history(m["id"]) for m in mk}
    cached = s.data.betas() if hasattr(s.data, "betas") else None
    struct = s.data.structure() if hasattr(s.data, "structure") else None
    out = pipeline.scan(vals, ac, closes, mk, hist, s.data.today(), horizon_days=horizon_days, n=n, cached_betas=cached,
                        structure=struct)
    s.last_scan = out
    keep = ("event_id", "event", "short", "market", "p", "p_confidence", "B", "beta_tier", "beta_se", "pnl_yes", "pnl_no", "sigma",
            "pnl_yes_by_class", "resolves_in_horizon", "drivers", "theme", "flags", "close_date",
            "above_divider", "divider_reason", "evidence", "B_hat", "B_prior", "prior_tier", "prior_why", "shrink_weight",
            "pnl_yes_pct_of_book", "sigma_pct_of_book", "sigma_vs_top_multiple")
    book = float(vals.sum())
    top = max((abs(e.get("sigma") or 0) for e in out["events"]), default=0) or None
    for e in out["events"] + out.get("drift", []):  # derived figures from code, so replies never compute them
        e["short"] = short_name(s, e)
        e["pnl_yes_pct_of_book"] = e["pnl_yes"] / book
        e["sigma_pct_of_book"] = (e.get("sigma") or e.get("sigma_drift") or 0) / book
        if top and e.get("sigma"):
            e["sigma_vs_top_multiple"] = top / e["sigma"]
    return {"events": [{k: e.get(k) for k in keep} for e in out["events"]],
            "drift": [{k: e.get(k) for k in keep + ("sigma_drift",)} for e in out.get("drift", [])],
            "skipped": out.get("skipped", [])[:10],
            "themes": out["themes"], "joint": out["joint"], "horizon_days": horizon_days,
            "ranking": out.get("ranking"), "ranking_note": "rows above the divider first (strong evidence), then 'weak evidence'; "
            "each group by σ of resolution P&L", **({"synthetic_data": True} if s.data.synthetic else {})}


def event_type(query: str) -> str | None:
    from .. import config
    q = query.lower()
    return next((t for t, kws in config.EVENT_TYPE_KEYWORDS.items() if any(k in q for k in kws)), None)


def event_template(s: Session, kind: str) -> dict:  # noqa: C901
    """Unpriced tier-2 template (METHODOLOGY §4.4–4.5): how THIS book's holdings moved on the type's analog dates.
    Code computes every move; no probability (no market), so it is conditional: "if it happens"."""
    from .. import config
    from ..scenario import impact
    s.require_portfolio()
    analogs = config.TEMPLATE_ANALOGS.get(kind)
    if not analogs:
        raise ToolError(f"no template for event type {kind!r}")
    vals = s.priced["value"]
    closes = s.data.closes(list(vals.index), days=2400)
    rows = []
    for a in analogs:
        d = pd.Timestamp(a["date"])
        if closes.empty or d < closes.index[0] or d > closes.index[-1]:
            continue
        mv = impact.window_moves(closes, d, before=1, after=a.get("after", 1))
        pnl = {t: float(vals[t] * mv[t]) for t in vals.index if t in mv and pd.notna(mv[t])}
        rows.append({"date": a["date"], "what": a["what"], "after": a.get("after", 1), "moves": {t: float(mv[t]) for t in pnl},
                     "book_pnl": sum(pnl.values()), "by_position": pnl})
    if not rows:
        raise ToolError("no price history for the analog dates")
    book = [r["book_pnl"] for r in rows]
    # typical = the median episode: the middle one, or the mean of the middle two, so the typical
    # by-position split and the hedge's typical moves come from the same episode(s) and sum to the median exactly
    order = sorted(range(len(rows)), key=lambda i: book[i])
    mid = order[(len(rows) - 1) // 2: len(rows) // 2 + 1]
    for i in mid:
        rows[i]["typical"] = True
    per_pos = {t: float(np.mean([rows[i]["by_position"].get(t, 0.0) for i in mid])) for t in vals.index}
    typical = float(np.median(book))
    out = {"type": kind, "status": "unpriced · template (tier 2, unreviewed: surprise scores pending the trainer's review)",
            "pnl_if_it_happens": typical, "typical_is": "median episode", "average": float(np.mean(book)),
            "range": [float(min(book)), float(max(book))],
            "pnl_if_it_happens_pct_of_book": typical / float(vals.sum()),
            "by_position": per_pos, "analogs": rows,
            "note": "conditional on the event happening; no market prices it, so there is no probability and no σ"}
    s.__dict__["active_template"] = out   # "how would I hedge that?" can follow
    return out


def _price_into_scan(s: Session, market: dict, flag: str) -> None:
    """Price one market for this book and add its row to the current scan (events or drift)."""
    vals, ac = s.priced["value"], s.priced["asset_class"]
    closes = s.data.closes(list(vals.index) + pipeline.REFERENCE)
    cached = s.data.betas() if hasattr(s.data, "betas") else None
    struct = s.data.structure() if hasattr(s.data, "structure") else None
    out = pipeline.scan(vals, ac, closes, [market], {market["id"]: s.data.market_history(market["id"])}, s.data.today(),
                        horizon_days=s.last_scan.get("horizon_days", 91), n=1, cached_betas=cached, structure=struct)
    for key in ("events", "drift"):
        for r in out.get(key, []):
            r["flags"] = list(r.get("flags", [])) + [flag]
            s.last_scan.setdefault(key, []).append(r)
    s.last_scan["betas"].update(out.get("betas", {}))


def find_event(s: Session, query: str, k: int = 3) -> dict:
    """Pointed question about an event that may not be in the top rows (SPEC §3.3 Case 2, METHODOLOGY §6 step 1):
    keyword-retrieve candidate markets from the whole universe, price each in code, add them to the scan, and
    return the candidates with match scores so the expert can judge the fit (or say nothing fits)."""
    from ..scenario import matching
    s.require_portfolio()
    if not s.last_scan:
        scan_events(s)
    known = {e["event_id"] for e in s.last_scan["events"] + s.last_scan.get("drift", [])}
    universe = s.data.event_markets()
    cands = matching.keyword_candidates(query, universe, n=k, text_keys=("title", "subtitle", "rules_primary"))
    if not cands:
        raise ToolError(f"no market in the universe matches {query!r}: treat it as unpriced")
    found = []
    for c in cands:
        if c["id"] not in known:
            _price_into_scan(s, c, flag="found_by_query")
        priced = any(e["event_id"] == c["id"] for e in s.last_scan["events"] + s.last_scan.get("drift", []))
        why = None
        if not priced:
            from ..scenario import discovery
            why = ("a price-level market on a tradable asset: that's the asset itself, not an event" if discovery.is_reflexive(c)
                   else "no usable history (needs ≥ 60 days and ≥ 5 jumps of 5pp) or outside 12 months")
        found.append({"event_id": c["id"], "title": c.get("title"), "keyword_score": c["_score"], "priced": priced, "why_not": why})
    return {"query": query, "candidates": found,
            "note": "keyword retrieval only: the expert must check variable, threshold, deadline and resolution source"}


def explain_event(s: Session, event_id: str) -> dict:
    if not s.last_scan:
        raise ToolError("no scan yet")
    rows = {e["event_id"]: e for e in s.last_scan["events"] + s.last_scan.get("drift", [])}
    if event_id not in rows:
        raise ToolError(f"unknown event {event_id}; known: {sorted(rows)}")
    s.active_event = event_id
    s.__dict__["last_event_id"] = event_id   # survives a reload (load_portfolio clears active_event)
    e, b = rows[event_id], s.last_scan["betas"][event_id]
    vals = s.priced["value"]
    pos = [{"ticker": t, "value": float(vals[t]), "beta": b[t]["beta"], "beta_used": b[t]["used"], "t_stat": b[t]["t"],
            "pnl_if_yes": float(vals[t]) * b[t]["used"] * (1 - e["p"])} for t in vals.index if t in b]
    pos.sort(key=lambda r: r["pnl_if_yes"])
    corr = {k: v for k, v in s.last_scan.get("corr", {}).get(event_id, {}).items() if k != event_id}
    # channel assets, whether held or not: the data check on a story like "Iran → oil → inflation → rates → small caps"
    cached = (s.data.betas() if hasattr(s.data, "betas") else {}).get(event_id) or s.last_scan["betas"].get(event_id, {})
    names = {"SPY": "US stocks", "IWM": "small caps", "USO": "oil", "TLT": "long bonds (up = yields down)", "GLD": "gold",
             "UUP": "US dollar", "HYG": "high-yield credit"}
    channels = [{"asset": a, "channel": names[a], "move_if_yes": cached[a].get("shrunk", 0.0) * (1 - e["p"]),
                 "move_if_no": -cached[a].get("shrunk", 0.0) * e["p"], "t_stat": cached[a].get("t")}
                for a in names if a in cached]
    e["short"] = short_name(s, e)
    return {"event": {k: e.get(k) for k in ("event_id", "event", "short", "market", "p", "p_source", "p_confidence", "beta_tier", "B",
                                            "beta_se", "pnl_yes", "pnl_no", "sigma", "close_date", "flags")},
            "positions": pos,
            "priced_in": {"p": e["p"], "move_if_yes_share": 1 - e["p"], "move_if_no_share": -e["p"],
                          "note": "only the unpriced part moves you: YES pays β(1−p), NO pays −βp"},
            "shrinkage": "β·t²/(t²+4)", "dp_correlation_with_others": corr,
            "channels": channels, "channels_note": "|t| < 2 means the data can't confirm that link"}


# ---------------------------------------------------------------- options_desk tools


def set_view(s: Session, **fields) -> dict:
    today = s.data.today()
    if s.view is None or (fields.get("underlying") and fields["underlying"].upper() != s.view.underlying):
        missing = [f for f, ok in (("direction", "direction" in fields), ("horizon", "horizon_days" in fields or "horizon" in fields))
                   if not ok]
        if missing:  # strict: an ambiguous new view gets one clarifying question, never a guess
            raise ToolError(f"ambiguous view: missing {' and '.join(missing)}. Ask the user ONE short question for it; "
                            "don't guess.")
    if "horizon_days" in fields:
        fields["horizon"] = today + timedelta(days=fields.pop("horizon_days"))
    assumed = None
    direction = fields.get("direction", getattr(s.view, "direction", None))
    has_target = "target_move_pct" in fields or (s.view is not None and s.view.target_move_pct is not None
                                                 and fields.get("direction", s.view.direction) == s.view.direction)
    horizon_known = fields.get("horizon") is not None or s.view is not None
    if direction in ("up", "down") and not has_target and horizon_known:
        # no size given: default to a 1σ move from 20-day realized vol over the horizon, and say so
        u = (fields.get("underlying") or s.view.underlying).upper()
        h = fields.get("horizon") or s.view.horizon
        rv = market_state.realized_vol(s.data.closes([u], days=120)[u], 20)
        mv = round(rv * math.sqrt(max((h - today).days, 1) / 365) * 100, 1)
        fields["target_move_pct"] = mv if direction == "up" else -mv
        assumed = {"target_move_pct": fields["target_move_pct"], "why": "no size given: a 1σ move from 20-day realized vol"}
    try:
        if s.view is None or fields.get("underlying", getattr(s.view, "underlying", None)) != s.view.underlying:
            v = View(**fields)
            v.validate(today)
            s.view, diff = v, None
        else:
            v, diff = s.view.update(**fields)
            v.validate(today)
            s.view = v
    except (ViewError, TypeError) as e:
        raise ToolError(str(e))
    return {"view": s.view.to_dict(), "diff": diff, **({"assumed": assumed} if assumed else {})}


def market_state_tool(s: Session, underlying: str) -> dict:
    u = underlying.upper()
    closes = s.data.closes([u], days=120)[u]
    earn = s.data.earnings(u) if hasattr(s.data, "earnings") else None
    exp_e, past = None, None
    S0 = float(closes.iloc[-1])
    if earn and earn.get("next"):
        ch0 = filled_chain(s, u, S0)
        ed = pd.Timestamp(earn["next"]["date"])
        after = sorted(e for e in pd.to_datetime(ch0["expiry"]).unique() if e > ed)
        exp_e = after[0] if after else None   # first expiry after the report holds the event
        past = [m["move"] for m in earn["past_moves"]]
    ms = market_state.summarize(filled_chain(s, u, S0), closes, s.data.today(), earnings_expiry=exp_e, past_earnings_moves=past)
    if earn and earn.get("next"):
        ms["earnings_next"] = earn["next"]
        ms["earnings_past_moves"] = earn["past_moves"][:8]
    if s.view is not None and s.view.underlying == u:
        ch = filled_chain(s, u, ms["spot"])
        exps = sorted(ch["expiry"].unique())
        e = next((x for x in exps if pd.Timestamp(x).date() >= s.view.horizon), exps[-1])
        ms["implied_move_to_horizon"] = market_state.implied_move(ch, ms["spot"], e)
        ms["horizon_expiry"] = str(pd.Timestamp(e).date())
    return ms


def _library():
    """WORSEY_LIBRARY_DIR pins one directory (replays/tests). Otherwise the trainer's library wins by id, and the
    front-matter-only fixtures stand in for strategies he hasn't written yet (marked `stand_in`)."""
    d = os.environ.get("WORSEY_LIBRARY_DIR")
    if d:
        return builder.load_library(Path(d))
    return builder.load_library_with_stand_ins()


def _vol_signals(u: str, chain, S: float) -> dict:
    """The trainer's z-score signals: today's chain scored against the cached IV history (options/vol_history.py)."""
    from ..options import signals, vol_history
    try:
        return signals.history_signals(vol_history.load(u), vol_history.chain_row(chain, S))
    except Exception as e:  # noqa: BLE001 — a signal failure never blocks pricing
        return {"history_days": 0, "note": f"vol signals unavailable: {str(e)[:80]}"}


def _protection(s: Session, u: str, candidates: list) -> None:
    """The trainer's put-spread rules 4/6/7/8: with a book loaded, size each put spread to the book's beta-adjusted
    notional to the underlying and show the uncovered tail and yearly cost (code numbers only)."""
    from ..options import signals
    cands = [c for c in candidates if c.id == "put_debit_spread"]
    if not cands or getattr(s, "priced", None) is None:
        return
    try:
        vals = s.priced["value"].astype(float).to_dict()
        closes = s.data.closes(sorted(set(vals) | {u}), days=300)
        book = signals.book_beta_notional(vals, closes, u)
    except Exception as e:  # noqa: BLE001 — protection facts never block pricing
        for c in cands:
            c.notes.append(f"no protection sizing: {str(e)[:80]}")
        return
    for c in cands:
        hf = signals.hedge_facts(c.strategy, book)
        ks = min((l.strike for l in c.strategy.legs if l.qty < 0), default=None)
        be = (c.strategy.breakevens() or [None])[0]
        al = signals.hedge_alerts(hf, ks, be)
        for a in al:
            a["darin_rule"] = c.entry.get("_rules", {}).get(int(a["rule"].split("#")[1]))
        c.metrics["protection"] = hf
        c.metrics.setdefault("alerts", []).extend(al)


def propose_strategies(s: Session, n: int = 3, include: list | None = None) -> dict:
    if s.view is None:
        raise ToolError("no view set; call set_view first")
    lib = _library()
    if not lib:
        raise ToolError("strategy library is empty (library/strategies/ is the trainer's to write)")
    v, today = s.view, s.data.today()
    ms = market_state_tool(s, v.underlying)
    ch = filled_chain(s, v.underlying, ms["spot"])
    sig = _vol_signals(v.underlying, ch, ms["spot"])
    term = ms.get("term") or []
    mctx = {"rv20": ms.get("rv20"), "iv_rv": ms.get("iv_rv"), "atm_iv_front": term[0]["atm_iv"] if term else None,
            "earnings_hist_abs_moves": ms.get("earnings_hist_abs_moves"), "implied_vs_hist": ms.get("implied_vs_hist"),
            "front_implied_move": ms.get("implied_move_to_horizon")}
    out = builder.propose(v, ms["spot"], today, builder.ChainQuoter(ch), ms["rv20"], lib, builder.load_playbook(),
                          implied_move=ms.get("implied_move_to_horizon"), iv_rv=ms.get("iv_rv"), n=n, include=include,
                          signals=sig, market_ctx=mctx, unlocked=set(s.__dict__.get("unlocked_gates") or ()))
    diff = builder.diff_proposals(s.last_strategies, out) if s.last_strategies else None
    out["_dist"] = v.distribution(today, ms["rv20"])
    out["_spot"] = ms["spot"]
    s.last_strategies = out
    ctx = out["context"]
    if out["candidates"]:  # derived comparisons, computed here (grounding)
        c0 = out["candidates"][0].metrics["cost"]
        for c in out["candidates"]:
            c.metrics["cost_vs_first"] = c.metrics["cost"] / c0 if c0 else None
            c.metrics["cost_minus_first"] = c.metrics["cost"] - c0
    budget = (v.constraints or {}).get("risk_budget_usd")
    for c in out["candidates"] if budget else []:   # sizing in code: how many of each the member's risk budget buys
        ml = c.metrics.get("max_loss")
        if isinstance(ml, (int, float)) and math.isfinite(ml) and ml < 0:
            k = int(budget // abs(ml))
            c.metrics["sized_to_budget"] = {"risk_budget": budget, "contracts": k, "total_cost": k * c.metrics["cost"],
                                            "total_max_loss": k * ml}
        else:
            c.metrics["sized_to_budget"] = {"risk_budget": budget, "contracts": None,
                                            "note": "undefined risk: size it by margin and the stress loss, not by max loss"}
    _protection(s, v.underlying, out["candidates"])
    md = out["market_distribution"]
    return {"view": out["view"], "your_distribution": out["distribution"], "market_distribution": md,
            "view_vs_market": {"view_move": ctx.get("view_move"), "implied_move_to_horizon": ctx.get("implied_move"),
                               "view_over_implied": ctx.get("view_over_implied"), "iv_rv": ctx.get("iv_rv"),
                               "your_sd_over_market_sd": out["distribution"]["sd"] / md["sd"] if md["sd"] else None},
            "market": {k: ms.get(k) for k in
            ("spot", "rv20", "iv_rv", "implied_move_to_horizon", "skew_25d_front")},
            "vol_signals": sig,
            "candidates": [{"id": c.id, "notes": c.notes, **c.metrics} for c in out["candidates"]],
            "dropped": out["dropped"], "teaching": out["teaching"], "changes_vs_last": diff}


def show_structure(s: Session, structure_id: str) -> dict:
    """The full card (payoff chart, view vs market, sizing) for ONE structure the member settled on, after they said yes
    to seeing it."""
    pool = {}
    if s.last_strategies:
        for c in s.last_strategies.get("candidates", []) + s.last_strategies.get("also_ran", []):
            pool[c.id] = {"id": c.id, "notes": c.notes, **c.metrics}
    for c in s.__dict__.get("custom_structures", []):
        pool[c["id"]] = c
    if structure_id not in pool:
        raise ToolError(f"no priced structure {structure_id!r}; priced: {sorted(pool)}")
    v = s.view.to_dict() if s.view is not None else {}
    last = s.last_strategies or {}
    return {"view": v, "your_distribution": last.get("distribution"), "market": {"spot": last.get("_spot")},
            "candidates": [pool[structure_id]], "full": True}


def price_structure(s: Session, legs: list, name: str = "custom") -> dict:
    """Price a structure the expert chooses (strikes or deltas, expiries by days) against the current view."""
    if s.view is None:
        raise ToolError("no view yet: set the view (view_and_strategies) first")
    v, today = s.view, s.data.today()
    ms = market_state_tool(s, v.underlying)
    ch = filled_chain(s, v.underlying, ms["spot"])
    for sp in legs:
        if sp.get("strike") is None and sp.get("delta") is None and sp.get("kind") != "stock":
            raise ToolError("each option leg needs a strike or a delta")
    out = builder.price_custom(v, ms["spot"], today, builder.ChainQuoter(ch), ms["rv20"], legs, name=name)
    budget = (v.constraints or {}).get("risk_budget_usd")
    ml = out.get("max_loss")
    if budget and isinstance(ml, (int, float)) and math.isfinite(ml) and ml < 0:
        k = int(budget // abs(ml))
        out["sized_to_budget"] = {"risk_budget": budget, "contracts": k, "total_cost": k * out["cost"], "total_max_loss": k * ml}
    s.__dict__.setdefault("custom_structures", []).append(out)
    return {"view": v.to_dict(), "spot": ms["spot"], "structure": out}


def view_and_strategies(s: Session, include: list | None = None, n: int = 3, confirmed: bool = False, **fields) -> dict:
    """Coarse options step (SPEC §11b rule 3): set/update the view, read the market, price candidates: one call.
    A NEW view is priced only after the intake:
    `confirmed` says the member answered the intake questions (exact view, sizing) or asked to skip them."""
    new_view = bool(fields.get("underlying")) and (s.view is None or str(fields["underlying"]).upper() != s.view.underlying)
    if new_view and not confirmed:
        raise ToolError("intake first: before pricing a new view, restate it in one line and ask the member 2-3 questions "
                        "(exact view: how far, by when, what drives it, how sure; sizing: how much they want to risk, defined "
                        "risk only?). You may name structure types in words, no prices. Call again with confirmed=true once "
                        "they've answered or said to just price it.")
    v = set_view(s, **fields) if fields else {"view": s.view.to_dict() if s.view is not None else None, "diff": None}
    out = propose_strategies(s, n=n, include=include)
    out["view_diff"] = v["diff"]
    return out


# ---------------------------------------------------------------- registry

def _as_of_et(iso: str | None) -> str | None:
    """'2026-10-17T18:31:00+00:00' → 'as of 2:31pm ET, Oct 17', so a reply can say when without converting anything."""
    if not iso:
        return None
    t = pd.Timestamp(iso).tz_convert("America/New_York")
    return f"as of {t.strftime('%I:%M%p').lstrip('0').lower()} ET, {t.strftime('%b')} {t.day}"


def live_quote(s: Session, ticker: str, expiry: str | None = None, strike: float | None = None,
               kind: str | None = None) -> dict:
    """The latest price, fetched live from Massive even when the rest of the chat runs on the cached snapshot
. A stock/ETF/crypto ticker alone → its latest minute bar. With expiry + strike + kind → that option
    contract's live bid/ask/mid, IV and greeks, plus the underlying's live price. Always states when it's as of."""
    from ..data import massive
    t = ticker.upper().strip()
    if expiry is None and strike is None and kind is None:
        q = massive.live_price(t)
        if not q:
            raise ToolError(f"no recent price for {t} from Massive")
        return {"ticker": t, **q, "as_of": _as_of_et(q.get("as_of_utc")), "source": "Massive, live (latest minute bar)",
                "note": "say when the price is from (`as_of`); the rest of this chat's analysis uses the cached snapshot"}
    if not (expiry and strike and kind in ("call", "put")):
        raise ToolError("an option quote needs expiry (YYYY-MM-DD), strike and kind (call or put)")

    def row(r):
        d, q, g, u = r.get("details") or {}, r.get("last_quote") or {}, r.get("greeks") or {}, r.get("underlying_asset") or {}
        ts = q.get("last_updated")
        return {"contract": d.get("ticker"), "expiry": d.get("expiration_date"), "strike": d.get("strike_price"),
                "kind": d.get("contract_type"), "bid": q.get("bid"), "ask": q.get("ask"), "mid": q.get("midpoint"),
                "last_trade": (r.get("last_trade") or {}).get("price"), "iv": r.get("implied_volatility"),
                "delta": g.get("delta"), "open_interest": r.get("open_interest"),
                "underlying_price": u.get("price"), "timeframe": q.get("timeframe"),
                "as_of": _as_of_et(pd.Timestamp(ts, unit="ns", tz="UTC").isoformat()) if ts else None}
    rows = massive.live_option(t, expiry, float(strike), kind)
    if rows:
        return {"ticker": t, "option": row(rows[0]), "source": "Massive, live option snapshot",
                "note": "say when the quote is from (`as_of`); the rest of this chat's analysis uses the cached snapshot"}
    near = massive.live_option(t, expiry, float(strike), kind, near=True)
    near = sorted(near, key=lambda r: (abs(pd.Timestamp((r.get("details") or {}).get("expiration_date")) - pd.Timestamp(expiry)),
                                       abs(((r.get("details") or {}).get("strike_price") or 0) - float(strike))))[:6]
    if not near:
        raise ToolError(f"no listed {t} {kind}s near {strike} around {expiry}")
    return {"ticker": t, "exact_contract_listed": False, "nearest_listed": [row(r) for r in near],
            "source": "Massive, live option snapshot"}


def company_facts(s: Session, ticker: str) -> dict:
    """What Massive has on a company: profile, the last 4 quarters (with margins computed here), next earnings and past
    earnings moves, recent news. Each part degrades on its own: `unavailable` lists what couldn't be fetched
    (e.g. not in the cached build when market data is offline). Public articles come from web search, not from here."""
    from ..data import massive
    t, out, missing = ticker.upper().strip(), {}, []
    out["ticker"] = t
    try:
        ov = massive.overview(t)
        if ov:
            out["profile"] = {k: ov.get(k) for k in ("name", "sic_description", "market_cap", "homepage_url", "total_employees")}
            out["profile"]["description"] = (ov.get("description") or "")[:700]
        else:
            missing.append("profile")
    except Exception:  # noqa: BLE001
        missing.append("profile")
    try:
        qs = []
        for r in massive.income_statements(t):
            rev = r.get("revenue") or 0.0
            qs.append({"period_end": r.get("period_end"), "fiscal_year": r.get("fiscal_year"), "fiscal_quarter": r.get("fiscal_quarter"),
                       "revenue": rev, "gross_profit": r.get("gross_profit"), "operating_income": r.get("operating_income"),
                       "net_income": r.get("consolidated_net_income_loss") or r.get("net_income_loss_attributable_common_shareholders"),
                       "gross_margin": (r.get("gross_profit") or 0.0) / rev if rev else None,
                       "operating_margin": (r.get("operating_income") or 0.0) / rev if rev else None})
        if qs:
            out["quarters"] = qs
            if len(qs) >= 2 and qs[1]["revenue"]:
                out["revenue_growth_qoq"] = qs[0]["revenue"] / qs[1]["revenue"] - 1   # derived here, so replies never compute it
        else:
            missing.append("financials")
    except Exception:  # noqa: BLE001
        missing.append("financials")
    try:
        closes = s.data.closes([t])
        earn = massive.earnings(t, s.data.today(), closes[t]) if t in closes else None
        if earn:
            out["earnings"] = {"next": earn.get("next"), "past_moves": (earn.get("past_moves") or [])[:8]}
        else:
            missing.append("earnings")
    except Exception:  # noqa: BLE001
        missing.append("earnings")
    try:
        items = massive.news(t)
        out["news"] = [{"title": n.get("title"), "publisher": (n.get("publisher") or {}).get("name"), "published": n.get("published_utc"),
                        "url": n.get("article_url"), "summary": (n.get("description") or "")[:300]} for n in items]
        if not items:
            missing.append("news")
    except Exception:  # noqa: BLE001
        missing.append("news")
    if s.priced is not None and t in s.priced.index:
        out["held_value"] = float(s.priced.at[t, "value"])
    out["unavailable"] = missing
    return out


def set_restrictions(s: Session, **fields) -> dict:
    """The member's trading restrictions for ETF matching (etf_desk intake), validated in code, kept per member."""
    from .. import etfmatch
    prof = s.__dict__.setdefault("restrictions", {})
    cur = dict(prof.get(s.member) or {})
    cur.update({k: v for k, v in fields.items() if v is not None})
    try:
        r = etfmatch.Restrictions(**cur)
    except (TypeError, ValueError) as e:
        raise ToolError(str(e)) from e
    from dataclasses import asdict
    prof[s.member] = asdict(r)
    return {"restrictions": prof[s.member], "never_allowed": "leveraged and inverse funds"}


def replicate_with_etfs(s: Session, target: str | None = None, basket: list | None = None, k: int = 3,
                        notional: float | None = None) -> dict:
    """Run the ETF matching engine (SPEC §5a) under the member's restrictions."""
    from .. import etfmatch
    prof = (s.__dict__.get("restrictions") or {}).get(s.member)
    if prof is None:
        raise ToolError("no restrictions yet: run the intake (set_restrictions) first")
    tgt = {b["ticker"].upper(): float(b["weight"]) for b in basket} if basket else (target or "").upper()
    if not tgt:
        raise ToolError("give target (a ticker) or basket")
    try:
        out = etfmatch.replicate(tgt, etfmatch.Restrictions(**prof), k=k, notional=notional, data=s.data, today=s.data.today())
    except ValueError as e:
        raise ToolError(str(e)) from e
    s.__dict__["last_etf_match"] = out
    return out


def my_profile(s: Session, action: str = "show", item_id: str | None = None) -> dict:
    """The member's own profile (SPEC §1e-1): "what do you know about me" / "forget <id>". In a group chat it goes to
    their DM, never into the group."""
    from . import profiles
    prof = profiles.load(s.member)
    if action == "forget":
        before = len(prof["lines"])
        prof["lines"] = [l for l in prof["lines"] if l["id"] != item_id]
        profiles.save(prof)
        done = len(prof["lines"]) < before
        msg = f"Removed line {item_id}." if done else f"No line {item_id}."
    else:
        msg = profiles.render(prof)
    if s.channel == "group":
        s.__dict__.setdefault("dm_out", []).append(msg)
        return {"sent_to_dm": True, "say": "tell them it's in their DM, nothing else about it in the group"}
    return {"profile": msg}


def recall(s: Session, query: str = "") -> dict:
    """Accumulated knowledge (DECISIONS #114): this member's relevant past recommendations with outcomes computed by code
    now, plus the active advisory lessons. Principles always win over lessons."""
    from . import memory
    rows = memory.recall(s.member, s.data, query)
    return {"episodes": rows, "lessons": [{"text": l["text"], "kind": l["kind"]} for l in memory.lessons()],
            "note": "outcomes computed from prices now; entries marked demo are seeded demo data; lessons are advisory"}


def book_history(s: Session) -> dict:
    """The member's book snapshots over time, value then vs now (code state, never memory)."""
    from . import memory
    return {"snapshots": memory.book_history(s.member, s.data)}


SECTOR_DIR = Path(__file__).resolve().parents[3] / "library" / "sectors"


def sector_pack(s: Session, sector: str) -> dict:
    """A sector knowledge pack for equity_analyst (DECISIONS #117): the trainer's prose (library/sectors/<id>.md, empty while a
    stub) plus code-computed facts on the sector ETF's top 10 holdings: 1y return, vol, beta to SPY, median earnings move."""
    import numpy as np
    import yaml
    from ..data import massive
    f = SECTOR_DIR / f"{sector}.md"
    if not f.exists():
        raise ToolError(f"no pack for {sector!r}; packs: {sorted(p.stem for p in SECTOR_DIR.glob('*.md'))}")
    text = f.read_text(encoding="utf-8")
    meta = yaml.safe_load(text.split("---")[1]) if text.startswith("---") else {}
    body = text.split("---", 2)[2].strip() if text.startswith("---") else text
    prose = body if any(line.strip() and not line.startswith("#") for line in body.splitlines()) else ""
    h = massive.etf_holdings(meta["etf"]).sort_values("weight", ascending=False)
    peers = list(dict.fromkeys(t for t in h["ticker"].astype(str).str.upper() if t.isalpha()))[:10]   # holdings can repeat tickers
    closes = s.data.closes(peers + ["SPY"], days=260)
    r = closes.pct_change().dropna(how="all")
    facts = []
    for t in peers:
        if t not in r.columns or r[t].dropna().empty:
            continue
        x = r[[t, "SPY"]].dropna()
        beta = float(np.cov(x[t], x["SPY"])[0, 1] / np.var(x["SPY"])) if len(x) > 20 else None
        try:
            past = massive.earnings(t, s.data.today(), closes[t]).get("past_moves") or []
        except Exception:  # noqa: BLE001
            past = []
        facts.append({"ticker": t, "return_1y": float(closes[t].dropna().iloc[-1] / closes[t].dropna().iloc[0] - 1),
                      "vol_annual": float(r[t].std() * np.sqrt(252)), "beta_spy": beta,
                      "median_abs_earnings_move": float(np.median([abs(m["move"]) for m in past])) if past else None})
    return {"sector": sector, "name": meta.get("name"), "etf": meta.get("etf"), "pack": prose or None,
            "pack_status": meta.get("status"), "peer_facts": facts,
            "note": "pack prose is the trainer's (empty while a stub); peer facts computed from prices and earnings history"}


# ---------------------------------------------------------------- natural_world_analyst tools (DECISIONS #116)

def futures_curve_tool(s: Session, product: str) -> dict:
    from .. import natural
    return natural.futures_curve(product.upper(), s.data.today())


def crack_spread(s: Session) -> dict:
    from .. import natural
    return natural.crack_321(s.data.today())


def weather_gas(s: Session, days: int = 14) -> dict:
    from .. import natural
    return natural.gas_weather(days)


def weather_crops(s: Session, days: int = 14) -> dict:
    from .. import natural
    return natural.crop_weather(days)


def commodity_spot(s: Session) -> dict:
    from .. import natural
    return natural.spot_prices()


def release_calendar_tool(s: Session) -> dict:
    from .. import natural
    return {"releases": natural.release_calendar(s.data.today())}


NATURAL_WORDS = r"\b(oil|crude|brent|wti|gas|gasoline|diesel|opec|hormuz|corn|soy|wheat|harvest|drought|hurricane|temperature|" \
                r"hottest|heat|snow|rain|el ni|la ni|enso|power|electric|grid|freeze|frost|wildfire)\b"


def natural_markets(s: Session) -> dict:
    """Prediction markets on weather and commodities in the current universe, with their odds."""
    import re
    names = ((s.data.structure() or {}).get("names") or {}) if hasattr(s.data, "structure") else {}
    out = []
    for m in s.data.event_markets():
        title = f"{m.get('title', '')} {m.get('subtitle', '')}"
        if not re.search(NATURAL_WORDS, title, re.I):
            continue
        b, a = m.get("bid"), m.get("ask")
        p = (b + a) / 2 if b is not None and a is not None else m.get("last")
        out.append({"market": m["id"], "name": names.get(m["id"]) or m.get("title"), "p": p, "close": m.get("close_time") or m.get("close_date")})
    return {"markets": out[:20]}


def fundamentals(s: Session, series: str | None = None) -> dict:
    """EIA / NASS / ISO hooks: which are on (keys present), and an EIA series if asked (e.g. NG.NW2_EPG0_SWO_R48_BCF.W)."""
    from .. import natural
    out = {"sources": natural.keyed_hooks()}
    if series:
        out["eia"] = natural.eia_series(series)
    return out


def reference(s: Session, query: str, k: int = 4) -> dict:
    """Search the options/derivatives reference corpus (Wikipedia CC BY-SA, SEC and CFTC public domain); cite what you use."""
    from .. import knowledge
    hits = knowledge.search(query, "options", k=k)
    return {"query": query, "results": hits,
            "note": "cite the source (title + url, and the author for Stack Exchange answers) for anything you use; Wikipedia and Stack Exchange text is CC BY-SA 4.0"}


def tape_summary_tool(s: Session) -> dict:
    """Code-computed tape (DECISIONS #121): biggest moves vs their own vol, IV and skew changes, odds shifts."""
    from .. import markets
    held = list(s.priced.index) if s.priced is not None else []
    return markets.tape_summary(s.data, held)


def market_digest(s: Session) -> dict:
    """This week's shared 'what the tape cares about' digest, written at build time (`worsey digest`)."""
    from .. import markets
    d = markets.digest()
    if d is None:
        raise ToolError("no digest yet (run `worsey digest`); use tape_summary instead")
    return {"text": d["text"], "as_of": d["as_of"], "age_days": d["age_days"], "sources": d["sources"][:8]}


def correct_previous(s: Session, original_excerpt: str, correction: str) -> dict:
    """Etiquette (DECISIONS #126): when an earlier reply of Worsey's turns out wrong (a bug, stale data, a number that
    didn't hold), post a short correction unprompted, quoting the original. The server posts it after this turn."""
    s.__dict__.setdefault("corrections", []).append({"quote": original_excerpt[:200], "text": correction[:400]})
    return {"queued": True, "note": "posted as its own message, quoting the original"}


VIEW_PROPS = {
    "underlying": {"type": "string", "minLength": 1},
    "direction": {"type": "string", "enum": ["up", "down", "neutral"]},
    "target_move_pct": {"type": "number", "minimum": -80, "maximum": 300},
    "horizon_days": {"type": "integer", "minimum": 1, "maximum": 730},
    "vol_view": {"type": "string", "enum": ["expand", "contract", "none"]},
    "conviction": {"type": "string", "enum": ["low", "med", "high"]},
    "event": {"type": "string"},
    "constraints": _obj({"defined_risk_only": {"type": "boolean"}, "max_premium_usd": {"type": "number", "minimum": 0},
                         "risk_budget_usd": {"type": "number", "minimum": 0}}, []),
}

# ---------------------------------------------------------------- paper option book (SPEC §4.3;)

def _book_ctx(s: Session, u: str) -> dict:
    """Spot, chain quoter and σ_day for one underlying, from the same data path as everything else."""
    u = u.upper()
    S = float(s.data.prices([u])[u])
    ch = filled_chain(s, u, S)
    closes = s.data.closes([u], days=60)[u]
    rv = market_state.realized_vol(closes, 20) if len(closes) > 21 else None
    return {"S": S, "chain": ch, "quote": builder.ChainQuoter(ch).quote, "sigma_day": book.sigma_day_from(rv), "rv20": rv}


def _private(s: Session) -> None:
    if s.channel == "group":
        raise ToolError("positions are private: ask the member to open their DM with Worsey (they can say 'post it here' "
                        "to share a summary)")


def _gap_event(s: Session, u: str, last_expiry: str) -> dict | None:
    try:
        e = s.data.earnings(u) if hasattr(s.data, "earnings") else None
    except Exception:  # noqa: BLE001  (earnings is context, never a blocker)
        return None
    nxt = (e or {}).get("next")
    if nxt and s.data.today().isoformat() <= str(nxt["date"]) <= last_expiry:
        return {"name": f"{u} earnings", "date": str(nxt["date"])}
    return None


def _front_skew(c: dict, legs: list[dict]) -> float | None:
    try:
        return market_state.skew_25d(c["chain"], c["S"], pd.Timestamp(min(lg["expiry"] for lg in legs)))
    except Exception:  # noqa: BLE001  (skew is context)
        return None


def positions_check(s: Session, member: str | None = None) -> dict:
    """Every open position: marks, greeks in trader units, P&L, and drift/staleness alerts against its intent."""
    _private(s)
    return book_status(s, member or s.member)


def book_status(s: Session, m: str) -> dict:
    """positions_check without the channel guard: the server's daily DM check calls this (its output goes to the DM only)."""
    out, today = [], s.data.today()
    for pos in book.open_positions(m):
        c = _book_ctx(s, pos["underlying"])
        g = book.position_greeks(pos, c["S"], today, c["quote"])
        live = [lg for lg in pos["legs"] if lg["expiry"] >= today.isoformat()]
        skew = _front_skew(c, live) if pos.get("intent") == "skew" and live else None
        gap = _gap_event(s, pos["underlying"], max(lg["expiry"] for lg in live)) if live else None
        out.append({"position": {k: pos.get(k) for k in ("id", "underlying", "intent", "opened", "targets", "note")},
                    "greeks": g, "alerts": book.check(pos, g, today, c["sigma_day"], skew, gap), "sigma_day": c["sigma_day"]})
    by_u = {}
    for r in out:
        n = by_u.setdefault(r["position"]["underlying"], {k: 0.0 for k in ("delta_sh", "dollar_delta", "vega_pt", "theta_day", "dgamma_1pct")})
        for k in n:
            n[k] += r["greeks"]["net"][k]
    return {"positions": out, "net_by_underlying": by_u,
            "n_alerts_high": sum(a["severity"] == "high" for r in out for a in r["alerts"]),
            "note": "net greeks are per underlying; never add deltas across underlyings" if len(by_u) > 1 else None}


def open_position(s: Session, intent: str, structure_id: str | None = None, underlying: str | None = None,
                  legs: list | None = None, confirmed: bool = False, note: str = "") -> dict:
    """Add a position to the member's paper book, from a priced structure or stated legs, with what it's FOR."""
    _private(s)
    today = s.data.today()
    if structure_id:
        pool = {}
        if s.last_strategies:
            for cnd in s.last_strategies.get("candidates", []) + s.last_strategies.get("also_ran", []):
                pool[cnd.id] = cnd.metrics
        for cst in s.__dict__.get("custom_structures", []):
            pool[cst["id"]] = cst
        if structure_id not in pool:
            raise ToolError(f"no priced structure {structure_id!r}; priced: {sorted(pool)}")
        underlying = underlying or (s.view.underlying if s.view is not None else None)
        dated = book.legs_from_structure(pool[structure_id], today)
    else:
        if not (underlying and legs):
            raise ToolError("give structure_id, or underlying + legs [{kind, strike, expiry YYYY-MM-DD, qty}]")
        dated = legs
    if not underlying:
        raise ToolError("which underlying?")
    if not confirmed:
        return {"confirm_first": True, "underlying": underlying.upper(), "intent": intent, "legs": dated,
                "note": "echo these legs and the intent back to the member in one line; call again with confirmed=true once they say yes"}
    c = _book_ctx(s, underlying)
    try:
        pos = book.new_position(s.member, underlying, dated, intent, c["S"], today, c["quote"],
                                skew25=_front_skew(c, dated), note=note)
    except KeyError as e:
        raise ToolError(f"not a listed contract: {e}") from None
    except ValueError as e:
        raise ToolError(str(e)) from None
    out = {"opened": pos["id"], "intent": intent, "entry_greeks": pos["entry_greeks"], "targets": pos["targets"],
           "legs": pos["legs"], "paper_only": True}
    if intent in book.VOL_INTENTS:   # a vol trade starts flat: offer the opening hedge now rather than flag it tomorrow
        t = book.hedge_ticket(pos, c["S"], today, c["quote"], c["sigma_day"], at_close=False)
        if not t["inside_band"]:
            out["opening_hedge"] = {k: t[k] for k in ("trade_shares", "trade_notional", "net_delta_sh", "band_sh")}
    return out


def _open_pos(s: Session, position_id: str) -> dict:
    pos = next((p for p in book.open_positions(s.member) if p["id"] == position_id), None)
    if pos is None:
        raise ToolError(f"no open position {position_id!r}; open: {[p['id'] for p in book.open_positions(s.member)]}")
    return pos


def hedge_ticket_tool(s: Session, position_id: str, at: str = "close", target_delta: float | None = None) -> dict:
    """Shares to trade to bring a position back to its delta target, with a spot ladder and the band."""
    _private(s)
    pos = _open_pos(s, position_id)
    c = _book_ctx(s, pos["underlying"])
    t = book.hedge_ticket(pos, c["S"], s.data.today(), c["quote"], c["sigma_day"], target_delta, at_close=(at == "close"))
    return {**t, "paper_only": True, "approve": "if the member says yes, call record_hedge with these shares"}


def record_hedge(s: Session, position_id: str, shares: float, confirmed: bool = False) -> dict:
    """Paper book only: record a hedge the member approved (never sent to a broker)."""
    _private(s)
    if not confirmed:
        raise ToolError("ask the member to approve the hedge first (paper book), then call with confirmed=true")
    pos = _open_pos(s, position_id)
    S = float(s.data.prices([pos["underlying"]])[pos["underlying"]])
    p = book.record_hedge(s.member, position_id, float(shares), S, s.data.today())
    return {"recorded": position_id, "shares": shares, "price": S, "hedge_shares_now": p["hedge_shares"], "paper_only": True}


def close_position_tool(s: Session, position_id: str, confirmed: bool = False) -> dict:
    _private(s)
    if not confirmed:
        raise ToolError("confirm with the member before closing (paper book), then call with confirmed=true")
    try:
        p = book.close_position(s.member, position_id, s.data.today())
    except KeyError:
        raise ToolError(f"no position {position_id!r}") from None
    return {"closed": p["id"], "paper_only": True}


PLANB_CATEGORIES = ["central_bank_unscheduled", "conflict_war", "disaster", "elections_politics", "financial_stress",
                    "health_pandemic", "regulatory_court", "sanctions", "supply_shock", "trade_tariffs"]
PLANB_DIRECTIONS = {"escalation": ["escalation", "de-escalation", "not_applicable"], "growth_shock": ["negative", "positive", "none"],
                    "inflation_shock": ["up", "down", "none"], "oil_supply": ["disruption", "increase", "none"],
                    "policy_stance": ["tighter", "looser", "none"], "risk_sentiment": ["risk_off", "risk_on", "neutral"]}

TOOLS = {
    "load_portfolio": (load_portfolio, "Load a demo portfolio by name, or holdings the user typed. Prices it and returns the asset-class breakdown.",
                       _obj({"name": {"type": "string", "enum": ["balanced", "tech_heavy", "policy_exposed"]},
                             "holdings": {"type": "array", "items": _obj({"ticker": {"type": "string"}, "shares": {"type": "number", "minimum": 0},
                                                                         "value_usd": {"type": "number", "minimum": 0}}, ["ticker"])}}, [])),
    "exposures": (exposures, "ETF look-through: real single-name and sector exposure, top concentrations. For a SPECIFIC company "
                  "('how much of my book is CoreWeave?', 'do I own any Tesla through my ETFs?') pass names=[ticker or company "
                  "name]: searches every held fund's full holdings, stock and bonds, with dollar amounts and the holdings date.",
                  _obj({"top": {"type": "integer", "minimum": 1, "maximum": 20},
                        "names": {"type": "array", "items": {"type": "string", "minLength": 2}, "maxItems": 5}}, [])),
    "compare_hedges": (compare_hedges_tool, "Hedge comparator for one scanned event (default: the active one): do nothing vs prediction contract vs index put spread vs puts on the holding hit hardest vs cross-asset legs (oil/bonds/gold/dollar, calls or puts by direction).",
                       _obj({"event_id": {"type": "string"}}, [])),
    "chart": (chart, "Chart the event scan (P&L if YES by asset class). NOT for options payoffs: those go to options_desk, "
                     "which knows which structure the member picked.",
              _obj({"kind": {"type": "string", "enum": ["scan"]}, "ref": {"type": "string"}}, ["kind"])),
    "scan_events": (scan_events, "Bottom-up scan of prediction markets: β per position, P&L if YES/NO, ranked by σ of resolution P&L, plus themes and the joint tail.",
                    _obj({"horizon_days": {"type": "integer", "minimum": 7, "maximum": 365}, "n": {"type": "integer", "minimum": 1, "maximum": 10}}, [])),
    "explain_event": (explain_event, "Drivers by position (β, t-stat, P&L if YES), the priced-in math, and correlation with other events.",
                      _obj({"event_id": {"type": "string"}}, ["event_id"])),
    "set_view": (set_view, "Create or update the options view. Pass only fields that change; returns the diff.",
                 _obj(VIEW_PROPS, [])),
    "market_state": (market_state_tool, "Vol picture for an underlying: spot, IV term structure, 25Δ skew, realized vol, IV/RV, implied move to the view horizon.",
                     _obj({"underlying": {"type": "string", "minLength": 1}}, ["underlying"])),
    "view_and_strategies": (view_and_strategies, "One call: create/update the view (pass only changed fields), read the vol "
                            "picture and price rule-built strategies, with E[P&L] under the user's view, view vs implied move, "
                            "and what changed since last time. `include` = structures the user named.", None),
    "live_quote": (live_quote, "The LATEST price, fetched live (the chat otherwise uses a cached snapshot). Use when the user asks "
                   "for the current/latest/live price of a stock, ETF, crypto (X:BTCUSD) or a specific option contract "
                   "(give expiry YYYY-MM-DD, strike, kind call/put; if that contract isn't listed it returns the nearest ones). "
                   "Say when the quote is as of.",
                   _obj({"ticker": {"type": "string", "minLength": 1}, "expiry": {"type": "string"},
                         "strike": {"type": "number", "minimum": 0.01}, "kind": {"type": "string", "enum": ["call", "put"]}},
                        ["ticker"])),
    "futures_curve": (futures_curve_tool, "Live futures curve for a commodity (CL WTI, BZ Brent, NG Henry Hub gas, HO heating oil, "
                      "RB gasoline, ZC corn, ZS soybeans, ZW wheat): settlements by month, contango/backwardation, front-2nd "
                      "and front-12th spreads, annualized roll yield, 5-day changes.",
                      _obj({"product": {"type": "string", "enum": ["CL", "BZ", "NG", "HO", "RB", "ZC", "ZS", "ZW"]}}, ["product"])),
    "crack_spread": (crack_spread, "The 3-2-1 crack on front futures (refining margin, $ per barrel of crude).", _obj({}, [])),
    "weather_gas": (weather_gas, "Population-weighted heating/cooling degree days for the next days from three models (NOAA GFS, "
                    "ECMWF IFS, ECMWF AIFS AI) vs the 10-year normal, and the model spread (forecast uncertainty).",
                    _obj({"days": {"type": "integer", "minimum": 3, "maximum": 16}}, [])),
    "weather_crops": (weather_crops, "Corn Belt rain and heat for the next days (GFS, ECMWF) with the current crop stage.",
                      _obj({"days": {"type": "integer", "minimum": 3, "maximum": 16}}, [])),
    "commodity_spot": (commodity_spot, "Daily spot: Henry Hub, WTI, Brent (FRED).", _obj({}, [])),
    "release_calendar": (release_calendar_tool, "Next market-moving releases: EIA storage and petroleum, USDA Crop Progress and "
                         "WASDE, CFTC positioning.", _obj({}, [])),
    "natural_markets": (natural_markets, "Prediction markets on weather and commodities (oil, gas, crops, hurricanes, temperature) "
                        "with their odds.", _obj({}, [])),
    "fundamentals": (fundamentals, "Which fundamentals sources are on (EIA inventories and grid, USDA NASS crop condition, PJM/ERCOT "
                     "power); with `series`, an EIA series.", _obj({"series": {"type": "string"}}, [])),
    "sector_pack": (sector_pack, "Sector knowledge pack: the trainer's notes on the sector's drivers, KPIs, event calendar and common "
                    "mistakes (when written), plus code-computed peer facts (1y return, vol, beta to SPY, median earnings move) "
                    "for the sector ETF's top holdings. Packs: semis, energy.",
                    _obj({"sector": {"type": "string", "enum": ["semis", "energy"]}}, ["sector"])),
    "reference": (reference, "Look up an options/derivatives term or concept (greeks, skew, put-call parity, pin risk, spreads, "
                  "futures terms...) in a cited reference corpus: Wikipedia, SEC Investor.gov, CFTC glossary.",
                  _obj({"query": {"type": "string", "minLength": 2}, "k": {"type": "integer", "minimum": 1, "maximum": 8}}, ["query"])),
    "tape_summary": (tape_summary_tool, "Fresh, code-computed tape: the week's biggest moves (z vs each asset's own vol), implied "
                     "vol and skew changes, and the biggest prediction-market odds shifts.", _obj({}, [])),
    "market_digest": (market_digest, "This week's shared digest of what the tape cares about (cited), written at build time.",
                      _obj({}, [])),
    "recall": (recall, "This member's relevant past recommendations from Worsey (strategies, hedges, events, ETF matches, what "
               "they said they did), each with its outcome computed from prices now, plus active advisory lessons. Use it to "
               "build on history ('last month's put spread expired worthless'); principles always win over lessons.",
               _obj({"query": {"type": "string"}}, [])),
    "book_history": (book_history, "The member's book snapshots over time, with value then vs now.", _obj({}, [])),
    "my_profile": (my_profile, "The member's profile of what they've told Worsey ('what do you know about me?'), or remove a "
                   "line ('forget <id>'). In the group it is sent to their DM.",
                   _obj({"action": {"type": "string", "enum": ["show", "forget"]}, "item_id": {"type": "string"}}, [])),
    "correct_previous": (correct_previous, "Post a short correction of one of YOUR earlier replies that turned out wrong "
                         "(bug, stale data, a figure that didn't hold), quoting the original. Only for real errors of yours.",
                         _obj({"original_excerpt": {"type": "string", "minLength": 3}, "correction": {"type": "string", "minLength": 3}},
                              ["original_excerpt", "correction"])),
    "set_restrictions": (set_restrictions, "Record the member's trading restrictions for ETF matching: allowed fund kinds "
                         "(broad, sector, thematic, country, bond, commodity, single_stock, other), shorting, gross cap, cash, "
                         "approved list, banned tickers, minimum holding days. Pass only what you learned; it merges.",
                         _obj({"allowed": {"type": "array", "items": {"type": "string", "enum": ["broad", "sector", "thematic", "country",
                                                                                                   "bond", "commodity", "single_stock", "other"]}},
                               "allow_short": {"type": "boolean"}, "gross_cap": {"type": "number", "minimum": 1, "maximum": 3},
                               "allow_cash": {"type": "boolean"},
                               "approved_list": {"type": "array", "items": {"type": "string"}},
                               "banned": {"type": "array", "items": {"type": "string"}},
                               "min_hold_days": {"type": "integer", "minimum": 0, "maximum": 365}}, [])),
    "replicate_with_etfs": (replicate_with_etfs, "Find the few permitted ETFs (+ cash) that track a stock or basket most closely "
                            "(min tracking error, total returns), tested on 6 unseen months vs SPY, the best single ETF and a "
                            "holdings-based mix, with what's lost. Uses the restrictions set earlier. k = number of ETFs (default 3).",
                            _obj({"target": {"type": "string"},
                                  "basket": {"type": "array", "items": _obj({"ticker": {"type": "string"}, "weight": {"type": "number", "minimum": 0}},
                                                                            ["ticker", "weight"])},
                                  "k": {"type": "integer", "minimum": 1, "maximum": 10},
                                  "notional": {"type": "number", "minimum": 0}}, [])),
    "company_facts": (company_facts, "Company data from Massive: profile, last 4 quarters (revenue, margins), next earnings and "
                      "past earnings moves, recent news headlines with links. `unavailable` lists parts that couldn't be fetched.",
                      _obj({"ticker": {"type": "string", "minLength": 1}}, ["ticker"])),
    "show_structure": (show_structure, "Send the FULL card (payoff chart, your view vs the market, sizing) for one structure the "
                       "member has settled on, by id. Only after they said yes to seeing it.",
                       _obj({"structure_id": {"type": "string", "minLength": 1}}, ["structure_id"])),
    "price_structure": (price_structure, "Price ANY structure you choose against the current view: legs as [{kind: call|put|stock, "
                        "qty: signed contracts (+ buy, - sell), strike OR delta (abs, e.g. 0.30), days (default: view horizon)}]. "
                        "Same metrics as the candidates (E[P&L] under the view and at market, max gain/loss, breakevens, greeks, "
                        "each leg's delta). Use it to compare other strikes or structures than the rule-built ones.",
                        _obj({"legs": {"type": "array", "minItems": 1, "maxItems": 4,
                                       "items": _obj({"kind": {"type": "string", "enum": ["call", "put", "stock"]},
                                                      "qty": {"type": "number"}, "strike": {"type": "number", "minimum": 0},
                                                      "delta": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                                                      "days": {"type": "integer", "minimum": 1, "maximum": 730}}, ["kind"])},
                              "name": {"type": "string"}}, ["legs"])),
    "positions": (positions_check, "The member's open paper positions: marks, greeks (delta shares and $, $gamma per 1%, vega $ "
                  "per vol pt, theta $/day, charm), P&L incl. hedges, and DRIFT alerts against what each was put on for "
                  "(delta outside the band, stale hedge, vega bled or flipped, now mostly directional, near expiry, gap "
                  "risk). DM only.", _obj({}, [])),
    "open_position": (open_position, "Add a position to the member's paper book, from a priced structure_id or stated legs, "
                      "with its intent (what it's FOR: delta_neutral_vol, long_vega, short_vega, skew, directional, hedge, "
                      "income). First call returns the parsed legs to echo back; call again with confirmed=true after they "
                      "say yes. Records entry greeks and targets so drift can be checked later.",
                      _obj({"intent": {"type": "string", "enum": list(book.INTENTS)}, "structure_id": {"type": "string"},
                            "underlying": {"type": "string"}, "confirmed": {"type": "boolean"}, "note": {"type": "string"},
                            "legs": {"type": "array", "maxItems": 4, "items": _obj({
                                "kind": {"type": "string", "enum": ["call", "put"]}, "strike": {"type": "number"},
                                "expiry": {"type": "string"}, "qty": {"type": "number"}, "entry_mid": {"type": "number"}},
                                ["kind", "strike", "expiry", "qty"])}}, ["intent"])),
    "hedge_ticket": (hedge_ticket_tool, "Shares to trade to bring one position back to its delta target, at the close (spot "
                     "unchanged: the difference is charm) or now, with a ladder at -2%..+2% and ±1σ, and whether it's inside "
                     "the no-trade band.", _obj({"position_id": {"type": "string"}, "at": {"type": "string", "enum": ["close", "now"]},
                                                 "target_delta": {"type": "number"}}, ["position_id"])),
    "record_hedge": (record_hedge, "Record a hedge the member approved, on the PAPER book only (never a broker).",
                     _obj({"position_id": {"type": "string"}, "shares": {"type": "number"}, "confirmed": {"type": "boolean"}},
                          ["position_id", "shares"])),
    "close_position": (close_position_tool, "Close a paper position after the member confirms.",
                       _obj({"position_id": {"type": "string"}, "confirmed": {"type": "boolean"}}, ["position_id"])),
    "event_evidence": (event_evidence, "Plan B evidence for ONE event you have structured: code's estimate per factor (in "
                       "words), the 5 closest past events and what each would do to THIS book today, the regime, how well the "
                       "factors cover the book, and Kalshi markets that might price it. Call it first, then price_scenario.",
                       _obj({"title": {"type": "string", "minLength": 3},
                             "category": {"type": "string", "enum": PLANB_CATEGORIES},
                             "severity": {"type": "string", "enum": ["minor", "moderate", "major", "extreme"]},
                             "direction": _obj({f: {"type": "string", "enum": v} for f, v in PLANB_DIRECTIONS.items()}, []),
                             "region": {"type": "string"},
                             "targeted_sectors": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                             "anticipation": {"type": "string", "enum": ["surprise", "partly_expected", "widely_expected"]},
                             "factor_signs": _obj({f: {"type": "string", "enum": ["+", "-", "0"]} for f in
                                                   ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit"]}, []),
                             "factor_sizes": _obj({f: {"type": "string", "enum": ["small", "medium", "large"]} for f in
                                                   ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit"]}, []),
                             "market_moving": {"type": "boolean", "description": "false when there's no reason to expect this "
                                               "to move markets (code then shows only the book's normal range)"}},
                            ["title", "category", "severity"])),
    "upcoming_events": (upcoming_events_tool, "Upcoming events that could move THIS book, with odds: prediction markets "
                        "closing within the horizon that move markets, each with its full impact on the book (if it were a "
                        "total surprise), what's already priced at today's odds (if YES: full x (1 - odds); if NO: -full x "
                        "odds), the risk left to resolve, and the positions it hits. Use for 'what could hit me', 'what's "
                        "coming up', or when odds matter.",
                        _obj({"horizon_days": {"type": "integer", "minimum": 7, "maximum": 365},
                              "n": {"type": "integer", "minimum": 1, "maximum": 12}}, [])),
    "earnings_calendar": (earnings_calendar_tool, "Single-name risk: the companies this book holds (directly or through its "
                          "ETFs) that report earnings in the next `days`, with the dollars riding on each, what options price "
                          "for the move, and the stock's past reaction moves, plus what the research says (options price "
                          "earnings about right; direction after the report is barely better than a coin flip). Use for "
                          "'what could hit me', single stocks, or 'when do my stocks report?'.",
                          _obj({"days": {"type": "integer", "minimum": 7, "maximum": 120}}, [])),
    "stress_scenario": (stress_scenario_tool, "Black-swan mode, for an event with no real precedent (the user calls it "
                        "unprecedented, or the closest past events are clearly much smaller): YOU write the factor shock in "
                        "plain units from your own understanding of the event (what it does to supply, growth, inflation, "
                        "policy, fear), with a short why per factor. Don't anchor on smaller past events: in the gut check "
                        "(#208) that made shocks worse. Code maps it through every holding (and fills unset factors with "
                        "their typical co-move). Units: market/dollar/oil/gold/healthcare/agriculture in % moves; rates_2y/rates_10y in "
                        "basis points of yield; credit, size, value, asia_em, defensive_vs_producers in % relative returns. "
                        "The shock is shown to the user as your assumption.",
                        _obj({"title": {"type": "string"},
                              "shocks": {"type": "object", "additionalProperties": {"type": "number"},
                                         "description": "factor name -> plain-unit move over the next 1-2 sessions"},
                              "why": {"type": "object", "additionalProperties": {"type": "string"}}}, ["title", "shocks"])),
    "news_base_rates": (news_base_rates_tool, "What company news of one kind usually does to a stock: typical size of the "
                        "move vs a normal day, how often the headline's direction held (incl. 2026), drift after, scaled to the "
                        "ticker and the user's exposure. You pick news_type; code gives every number.",
                        _obj({"news_type": {"type": "string", "enum": ["FDA", "M&A", "Guidance", "Upgrades", "Downgrades", "Legal",
                                                                       "Offerings", "Buybacks", "Management", "Rumors", "Contracts",
                                                                       "Dividends", "Initiation"]},
                              "ticker": {"type": "string"}}, ["news_type"])),
    "price_scenario": (price_scenario, "Code prices the event you structured (every number from the engine, DECISIONS "
                       "#182): first-hours and end-of-day ranges if it happens, the bad case (1 in 20), tail probabilities, "
                       "the nearest past events' worst and best on this book, by position and by factor. market_id only if a "
                       "Kalshi market from the evidence really is this event (its odds are then shown). "
                       "already_reported_before_open=true only when the event has REALLY happened and was first reported "
                       "before today's US open: code then adds the after-the-open range from today's opening prices.",
                       _obj({"market_id": {"type": "string"}, "already_reported_before_open": {"type": "boolean"},
                             "company_news": {"type": "array", "maxItems": 4, "items": _obj({
                                 "ticker": {"type": "string"},
                                 "news_type": {"type": "string", "enum": ["FDA", "M&A", "Guidance", "Upgrades", "Downgrades", "Legal",
                                                                          "Offerings", "Buybacks", "Management", "Rumors", "Contracts",
                                                                          "Dividends", "Initiation"]},
                                 "direction": {"type": "string", "enum": ["positive", "negative"]},
                                 "size": {"type": "string", "enum": ["typical", "big"]}}, ["ticker", "news_type", "direction"])},
                             "sector_news": {"type": "array", "maxItems": 3, "items": _obj({
                                 "sector": {"type": "string", "enum": ["semis", "software", "internet", "regional banks", "insurers", "biotech",
                                                                       "medtech", "energy stocks", "industrials", "defense", "homebuilders",
                                                                       "transports", "consumer discretionary", "retail", "staples", "utilities",
                                                                       "real estate", "materials", "metals & mining"]},
                                 "direction": {"type": "string", "enum": ["positive", "negative"]},
                                 "size": {"type": "string", "enum": ["typical", "big"]}}, ["sector", "direction"])},
                             "country_news": {"type": "array", "maxItems": 3, "items": _obj({
                                 "country": {"type": "string", "enum": _COUNTRIES},
                                 "direction": {"type": "string", "enum": ["positive", "negative"]},
                                 "size": {"type": "string", "enum": ["typical", "big"]}}, ["country", "direction"])}}, [])),
    "propose_strategies": (propose_strategies, "Rule-built strategies for the current view from the trainer's library, priced on the chain, with E[P&L] under the user's view and what changed since last time.",
                           _obj({"n": {"type": "integer", "minimum": 1, "maximum": 5},
                                 "include": {"type": "array", "items": {"type": "string"}}}, [])),
}

VIEW_AND_STRATEGIES = _obj({**VIEW_PROPS, "include": {"type": "array", "items": {"type": "string"}},
                            "confirmed": {"type": "boolean"},
                            "n": {"type": "integer", "minimum": 1, "maximum": 5}}, [])

TOOLS["view_and_strategies"] = (TOOLS["view_and_strategies"][0], TOOLS["view_and_strategies"][1], VIEW_AND_STRATEGIES)

GROUPS = {
    # entry tools (scan_risks, analyze_event, compare_hedges) and the experts are added in agent/loop.py.
    # Oct 17 (DECISIONS #100): the view goes back to options_desk; hedges are an entry tool that runs experts.
    "router": ["load_portfolio", "exposures", "chart", "correct_previous", "my_profile", "recall", "book_history"],
    "event_analyst": ["explain_event", "correct_previous", "recall"],  # the scan itself is code (scan_risks); the expert judges and explains it
    "options_desk": ["view_and_strategies", "price_structure", "show_structure", "market_state", "reference", "correct_previous", "recall",
                     "positions", "open_position", "hedge_ticket", "record_hedge", "close_position"],  # coarse: one call per step
    "equity_analyst": ["company_facts", "sector_pack", "correct_previous", "recall"],  # plus Anthropic web search (server tool), added in agent/loop.py
    "etf_desk": ["set_restrictions", "replicate_with_etfs", "correct_previous", "recall"],
    "natural_world_analyst": ["futures_curve", "crack_spread", "weather_gas", "weather_crops", "commodity_spot", "release_calendar",
                              "natural_markets", "fundamentals", "reference", "correct_previous", "recall"],  # + web search
    "recent_markets_analyst": ["tape_summary", "market_digest", "company_facts", "correct_previous", "recall"],  # + web search   # stateful by warm sessions (#108, #122)
}


PLANB_GROUPS = {"event_analyst": ["event_evidence", "price_scenario", "stress_scenario", "upcoming_events", "earnings_calendar", "news_base_rates", "correct_previous", "recall"],   # DECISIONS #179
                # single-stock risk (#204): Plan B only, so v1's tool lists (and the recorded demos) stay byte-identical
                "router": GROUPS["router"] + ["earnings_calendar", "upcoming_events"],
                "equity_analyst": GROUPS["equity_analyst"] + ["earnings_calendar", "news_base_rates"]}


def group_tools(group: str) -> list[str]:
    from .. import config
    return PLANB_GROUPS.get(group, GROUPS[group]) if config.SCENARIO_ENGINE == "planb" else GROUPS[group]


def schemas(group: str) -> list[dict]:
    """Tool definitions for the API, in a fixed order (stable prefix for prompt caching)."""
    return [{"name": n, "description": TOOLS[n][1], "input_schema": TOOLS[n][2]} for n in group_tools(group)]
