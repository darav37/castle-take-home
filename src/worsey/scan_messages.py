"""The scan's messages, sent as each stage finishes (DECISIONS #234). Written by code from the pipeline's
own numbers, so they arrive in seconds and can't drift; the analyst writes only the last message (the three events to
watch, with their probabilities).

  1. your book at a glance      after the listed candidates are priced (seconds)
  2. your events                after the proposal call has labelled and priced them
  3. researching                before the forecasts run (says which events and why it takes a moment)
"""

from __future__ import annotations


def _usd(x: float, sign: bool = True) -> str:
    s = f"${abs(x):,.0f}"
    return (("+" if x >= 0 else "−") + s) if sign else s


def _positions(r: dict, k: int = 3) -> str:
    pos = [p for p in (r.get("by_position") or []) if isinstance(p, dict) and abs(p.get("full_impact_usd", 0)) >= 1][:k]
    return ", ".join(f"{p['ticker']} {_usd(p['full_impact_usd'])}" for p in pos)


def book(table: dict, data: dict) -> str:
    """Message 1: the normal swing, the biggest exposures, and the kinds of past events that hurt this book most."""
    swing = data["ctx"]["swing"]
    ex = data.get("exposures") or {}
    fac = [(k, v) for k, v in (ex.get("factors_usd_per_normal_day") or {}).items()][:3]
    risks = sorted([r for r in (table or {}).get("risks", []) if r.get("kind") == "one-off event"],
                   key=lambda r: r.get("bad_case_1_in_10_usd", 0))[:3]
    lines = [f"**Your book at a glance.** A normal two-day swing is about ±{_usd(swing, False)}; anything smaller is noise for you."]
    if fac:
        lines.append("Biggest exposures (dollars per normal day of each): " + ", ".join(f"{k} {_usd(v)}" for k, v in fac) + ".")
    if risks:
        lines.append("The kinds of past events that hurt it most (1-in-10 bad case): "
                     + "; ".join(f"{r['risk']} {_usd(r['bad_case_1_in_10_usd'])}" for r in risks) + ".")
    lines.append("Now checking what's coming up and what would move it most…")
    return "\n\n".join(lines)


def yours(rows: list[dict], swing: float) -> str | None:
    """Message 2: the user's own events, each with its move, top positions and an honest verdict."""
    if not rows:
        return None
    out = ["**The events you asked about:**"]
    for r in rows:
        v = r.get("verdict") or ""
        honest = " (inside your normal swing: noise for your book)" if "noise" in v else ""
        pos = _positions(r)
        rel = r.get("related") or []
        folded = f" Markets on the same move: {', '.join(x['title'][:60] for x in rel[:2])}." if rel else ""
        out.append(f"- **{r['title']}:** about {_usd(r['full_impact_usd'])} if it happens{honest}"
                   + (f" ({pos})" if pos else "") + f".{folded}")
    out.append("Odds for these come next.")
    return "\n".join(out)


def _pct(p: float) -> str:
    return f"{p:.0%}" if p >= 0.1 else f"{p:.1%}"


def probability(r: dict) -> str:
    """The event's probability and where it comes from, in words."""
    p, tier = r.get("odds"), r.get("tier")
    if p is None:
        return "Probability: no market lists it and it wasn't forecast"
    if tier is None:   # a base rate describes a kind of event, not this one
        return (f"Probability: not estimated. For context, a big event of this kind happened in about {_pct(p)} of windows "
                f"this long (2016–2025)")
    src = {1: "market", "2a": "derived from related markets", "2b": "bounded by related markets", 3: "Worsey forecast; no market lists it"}[tier]
    if tier == 1 and r.get("odds_raw") is not None and r.get("longshot_corrected"):
        src = f"market, corrected for longshot bias from {_pct(r['odds_raw'])}"
    return f"Probability: {_pct(p)} ({src})"


def event(r: dict, swing: float) -> str:
    """One event: what it is, its probability, and if it happens, each holding's average dollar move with its 9-in-10
    range, then the book's. Every number comes from the pipeline; holdings add up to the book draw by draw."""
    book = r.get("book_if_yes") or {}
    lines = [f"**{r['title']}**"]
    why = r.get("why_this_book") or r.get("why")
    if why:
        lines.append(why)
    lines.append(probability(r))
    if book:
        lo, hi = book["range_90_usd"]
        ratio = abs(book["mean_usd"]) / swing if swing else 0.0
        verdict = "noise for your book" if ratio < 0.5 else ("small for your book" if ratio < 1 else "meaningful for your book")
        lines.append(f"If it happens, your book: {_usd(book['mean_usd'])} on average (9 in 10: {_usd(lo)} to {_usd(hi)}); {verdict}")
        for p in r.get("by_position") or []:
            if "mean_usd" in p:
                a, b = p["range_90_usd"]
                lines.append(f"- {p['ticker']}: {_usd(p['mean_usd'])} (9 in 10: {_usd(a)} to {_usd(b)})")
    return "\n".join(lines)


def researching(titles: list[str]) -> str | None:
    """Message 3: which events are being researched, and why it takes a moment."""
    if not titles:
        return None
    t = "; ".join(x if len(x) <= 80 else x[:80].rsplit(" ", 1)[0] + "…" for x in titles[:6])
    return (f"No market lists {'this' if len(titles) == 1 else 'these'} yet, so I'm researching the odds myself (past events "
            f"like {'it' if len(titles) == 1 else 'them'} and this fortnight's news): {t}. About a minute…")
