"""Discovery, bottom-up half (SPEC §3.2 step 2, §3.3 Case 1).

Bottom-up: filter the market universe to liquid markets inside the horizon with enough
history to estimate a β, then rank by σ (scenario/impact.rank_events).
Top-down proposals come from event_analyst (Claude) and are merged here; events with no
market are kept as 'unpriced' and listed separately, ranked by conditional loss.
"""

from __future__ import annotations

from datetime import date, datetime

# Kalshi event categories in scope (verified CP-L: category lives on the event/series, F5).
# Sports, Entertainment, Mentions, Social etc. are out. Weather is in scope for `natural_world_analyst` (was meteorologist, #116).
# Crypto is out (Oct 16): its markets are price levels/returns of crypto assets, i.e. reflexive, not events.
IN_SCOPE_CATEGORIES = {"Economics", "Financials", "Politics", "Elections", "World", "Companies", "Commodities",
                       "Climate and Weather", "Science and Technology", "AI"}


import re

# Price-level markets on tradable assets are not events: "S&P 500 above 6845 on Dec 31" is SPY itself, so its
# β to the book is mechanical and would top every ranking. Excluded before ranking (Oct 16, flagged for the trainer).
_ASSET = r"(s&p ?500|s&p|\bsp ?500\b|\bspx\b|nasdaq|dow jones|\bdow\b|russell|bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bxrp\b|dogecoin|" \
         r"\bwti\b|brent|crude oil price|gold price|\bgold\b|silver|treasury yield|10-year|10y|\busd/|eur/usd|vix)"
_LEVEL = r"(above|below|between|reach|hit|settle|close|high|low|max|min|price|\$\d)"


def is_reflexive(m: dict) -> bool:
    t = f"{m.get('title') or ''} {m.get('subtitle') or ''}".lower()
    return bool(re.search(_ASSET, t)) and bool(re.search(_LEVEL, t))


def diversified_cap(markets: list[dict], n: int, key: str = "category") -> list[dict]:
    """Top n by volume, round-robin across categories, so one hot category (crypto, elections) can't take
    every slot from thinner but relevant ones (tariffs, Fed)."""
    groups: dict[str, list[dict]] = {}
    for m in sorted(markets, key=lambda m: -(m.get("volume") or 0)):
        groups.setdefault(m.get(key) or "?", []).append(m)
    out, i = [], 0
    while len(out) < n and any(i < len(g) for g in groups.values()):
        for g in groups.values():
            if i < len(g) and len(out) < n:
                out.append(g[i])
        i += 1
    return out


def _to_date(x) -> date | None:
    if x is None:
        return None
    if isinstance(x, date):
        return x if not isinstance(x, datetime) else x.date()
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).date()


def bottom_up_filter(markets: list[dict], today: date, horizon_days: int | None = None, max_resolve_days: int | None = None,
                     min_history_days: int = 60, min_volume: float = 1000.0) -> list[dict]:
    """Candidate markets for the bottom-up scan (METHODOLOGY §5.2).

    Keeps markets resolving within 12 months (not just within the horizon: C5, post-horizon
    events become drift rows), open ≥ min_history_days, with some liquidity, and whose event
    category is in scope. Category lives on the event (CP-L F5): a missing category is excluded,
    never waved through. Each kept market gets `resolves_in_horizon`.
    """
    from .. import config
    horizon_days = config.HORIZON_DAYS if horizon_days is None else horizon_days
    max_resolve_days = config.MAX_RESOLVE_DAYS if max_resolve_days is None else max_resolve_days
    out = []
    for m in markets:
        close, opened = _to_date(m.get("close_date")), _to_date(m.get("open_date"))
        if close is None or opened is None:
            continue
        days = (close - today).days
        if not (0 < days <= max(max_resolve_days, horizon_days)):
            continue
        if (today - opened).days < min_history_days:
            continue
        if (m.get("volume") or 0) < min_volume:
            continue
        if m.get("category") not in IN_SCOPE_CATEGORIES:
            continue
        if is_reflexive(m):
            continue
        out.append(dict(m, resolves_in_horizon=days <= horizon_days))
    return out


def tier1_eligible(dp, min_days: int | None = None, jump: float | None = None, min_jumps: int | None = None) -> dict:
    """Tier 1(a) needs history AND jumps (METHODOLOGY §4.1): ≥ 60 non-stale Δp days and ≥ 5 days with |Δp| ≥ 5pp.

    Without jumps, β is just noise. Returns {eligible, days, jump_days, reason}.
    """
    from .. import config
    min_days = config.MIN_HISTORY_DAYS if min_days is None else min_days
    jump = config.JUMP_PP if jump is None else jump
    min_jumps = config.MIN_JUMP_DAYS if min_jumps is None else min_jumps
    good = dp.dropna()
    n, j = int(len(good)), int((good.abs() >= jump - 1e-12).sum())
    ok = n >= min_days and j >= min_jumps
    reason = None if ok else (f"only {n} usable days (< {min_days})" if n < min_days else f"only {j} jump days ≥ {jump:.0%} (< {min_jumps})")
    return {"eligible": ok, "days": n, "jump_days": j, "reason": reason}


def merge(bottom_up: list[dict], top_down: list[dict]) -> dict:
    """Merge the two candidate lists. Top-down events matched to a bottom-up market are deduped
    onto it; unmatched top-down events become 'unpriced'.

    top_down dicts carry 'matched_market_id' (set by matching) or None.
    """
    by_id = {m["id"]: dict(m, sources=["bottom_up"]) for m in bottom_up}
    unpriced = []
    for e in top_down:
        mid = e.get("matched_market_id")
        if mid and mid in by_id:
            by_id[mid]["sources"].append("top_down")
            by_id[mid].setdefault("top_down_events", []).append(e)
        elif mid:
            by_id[mid] = dict(e, id=mid, sources=["top_down"])
        else:
            unpriced.append(dict(e, status="unpriced"))
    return {"priced": list(by_id.values()), "unpriced": unpriced}


def rank_unpriced(events: list[dict]) -> list[dict]:
    """No p ⇒ no σ. Rank by conditional loss (P&L if it happens), worst first."""
    return sorted(events, key=lambda e: e.get("pnl_if_happens", 0.0))
