"""Paper option book: positions, live greeks, drift and staleness, hedge tickets (SPEC §4.3;).

Every position records what it was put on FOR (its intent) and the greeks it had then. The check compares today's greeks
with that intent, so "your straddle is now a +40-share delta bet" or "your vega target has bled 35%" is computed by code,
never by a model. Paper only: approved hedges update this book; nothing is ever sent to a broker.

Units (per position, after qty × multiplier): delta in shares, $ delta, gamma as shares of delta per 1% move,
$ gamma per 1% (½·Γ·S²·0.01²), vega in $ per vol point, theta in $ per day, charm in shares of delta per day.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

from ..data.cache import CACHE_ROOT
from .pricer import bs_greeks, bs_price

STORE = CACHE_ROOT / "positions"
MULT = 100
INTENTS = ("delta_neutral_vol", "long_vega", "short_vega", "skew", "directional", "hedge", "income")
VOL_INTENTS = ("delta_neutral_vol", "long_vega", "short_vega", "skew")

# defaults: band = delta change from a 0.5σ_day move (SPEC §4.3.3)
BAND_SIGMA = 0.5
VEGA_DRIFT = 0.25          # vega more than 25% away from what it was put on with
STALE_HEDGE_DAYS = 3       # outside the band and no hedge for this many days = stale
NEAR_EXPIRY_DAYS = 7


# ---------------------------------------------------------------- store (one JSON file per member)

def _path(member: str) -> Path:
    return STORE / f"{''.join(c for c in member if c.isalnum() or c in '-_') or 'anon'}.json"


def load(member: str) -> list[dict]:
    p = _path(member)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save(member: str, rows: list[dict]) -> None:
    p = _path(member)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


def open_positions(member: str, underlying: str | None = None) -> list[dict]:
    return [p for p in load(member) if p.get("status") == "open" and (underlying is None or p["underlying"] == underlying)]


# ---------------------------------------------------------------- greeks

def _T(expiry: str, today: date, extra_days: float = 0.0) -> float:
    return max((date.fromisoformat(expiry) - today).days - extra_days, 0.0) / 365


def leg_greeks(leg: dict, S: float, today: date, iv: float, r: float = 0.04, q: float = 0.0) -> dict:
    """One leg, in position units (qty × multiplier)."""
    n = leg["qty"] * MULT
    T = _T(leg["expiry"], today)
    g = bs_greeks(S, leg["strike"], T, r, q, max(iv, 1e-6), leg["kind"])
    g1 = bs_greeks(S, leg["strike"], _T(leg["expiry"], today, 1), r, q, max(iv, 1e-6), leg["kind"])
    return {"delta_sh": g.delta * n, "gamma_sh_1pct": g.gamma * S * 0.01 * n, "dgamma_1pct": 0.5 * g.gamma * S * S * 1e-4 * n,
            "vega_pt": g.vega / 100 * n, "theta_day": g.theta / 365 * n, "charm_sh_day": (g1.delta - g.delta) * n,
            "mark": bs_price(S, leg["strike"], T, r, q, max(iv, 1e-6), leg["kind"]), "iv": iv, "dte": (date.fromisoformat(leg["expiry"]) - today).days}


SUMS = ("delta_sh", "gamma_sh_1pct", "dgamma_1pct", "vega_pt", "theta_day", "charm_sh_day")


def position_greeks(pos: dict, S: float, today: date, quote, r: float = 0.04, q: float = 0.0) -> dict:
    """quote(kind, strike, expiry_date) -> (iv, mid). Marks at the chain mid; greeks at the mid-implied IV (one-IV rule)."""
    tot = {k: 0.0 for k in SUMS}
    legs, value, missing = [], 0.0, []
    for lg in pos["legs"]:
        exp = date.fromisoformat(lg["expiry"])
        if exp < today:
            legs.append({**lg, "expired": True})
            continue
        try:
            iv, mid = quote(lg["kind"], lg["strike"], exp)
        except KeyError:
            iv, mid = lg.get("entry_iv") or 0.3, None
            missing.append(f"{lg['kind']} {lg['strike']:g} {lg['expiry']}")
        g = leg_greeks(lg, S, today, iv, r, q)
        if mid is not None:
            g["mark"] = mid
        for k in SUMS:
            tot[k] += g[k]
        value += g["mark"] * lg["qty"] * MULT
        legs.append({**lg, **g})
    sh = pos.get("hedge_shares", 0.0)
    tot["delta_sh"] += sh
    tot["dollar_delta"] = tot["delta_sh"] * S
    entry_cost = sum((lg.get("entry_mid") or 0) * lg["qty"] * MULT for lg in pos["legs"])
    hedge_pnl = sum(h["shares"] * (S - h["price"]) for h in pos.get("hedges", []))
    return {"id": pos["id"], "underlying": pos["underlying"], "spot": S, "legs": legs, "hedge_shares": sh,
            "net": {k: round(v, 2) for k, v in tot.items()}, "option_value": round(value, 2),
            "pnl_options": round(value - entry_cost, 2), "pnl_hedges": round(hedge_pnl, 2),
            "pnl_total": round(value - entry_cost + hedge_pnl, 2), **({"unquoted_legs": missing} if missing else {})}


# ---------------------------------------------------------------- drift and staleness (the point of this module)

def band_shares(net: dict, S: float, sigma_day: float) -> float:
    """Delta change from a BAND_SIGMA·σ_day move: |gamma shares per 1%| × (0.5·σ_day in %). Floor of 1 share."""
    return max(abs(net["gamma_sh_1pct"]) * BAND_SIGMA * sigma_day * 100, 1.0)


def check(pos: dict, g: dict, today: date, sigma_day: float, skew_now: float | None = None,
          event_before_expiry: dict | None = None) -> list[dict]:
    """Alerts for one position: each {kind, severity (high|medium|info), text, numbers}. Text is code-written."""
    out = []
    net, S = g["net"], g["spot"]
    intent = pos.get("intent", "directional")
    tgt = pos.get("targets", {})
    band = band_shares(net, S, sigma_day)
    ed = pos.get("entry_greeks", {}).get("delta_sh", 0.0)
    tdelta = tgt.get("delta_sh", 0.0 if intent in VOL_INTENTS else ed)
    gap = net["delta_sh"] - tdelta
    last_hedge = max([h["date"] for h in pos.get("hedges", [])] + [pos["opened"]])
    days_since = (today - date.fromisoformat(last_hedge)).days

    if intent in VOL_INTENTS and abs(gap) > band:
        sev = "high" if abs(gap) > 3 * band else "medium"
        out.append({"kind": "delta_drift", "severity": sev,
                    "text": f"delta is {net['delta_sh']:+.0f} shares against a target of {tdelta:+.0f} (band ±{band:.0f}); "
                            f"trade {-gap:+.0f} shares to get back",
                    "numbers": {"delta_sh": net["delta_sh"], "target": tdelta, "band": band, "trade": -gap}})
        if days_since >= STALE_HEDGE_DAYS:
            out.append({"kind": "stale_hedge", "severity": "high",
                        "text": f"no re-hedge for {days_since} days while outside the band",
                        "numbers": {"days_since_hedge": days_since}})

    if intent in ("directional", "hedge", "income") and abs(ed) >= 1 and abs(net["delta_sh"] - ed) > max(0.5 * abs(ed), band):
        out.append({"kind": "delta_since_entry", "severity": "medium" if intent == "hedge" else "info",
                    "text": f"delta is {net['delta_sh']:+.0f} shares vs {ed:+.0f} at entry: the position's exposure has "
                            f"{'shrunk' if abs(net['delta_sh']) < abs(ed) else 'grown'} with the move and time",
                    "numbers": {"delta_sh": net["delta_sh"], "entry": ed}})

    ev = pos.get("entry_greeks", {}).get("vega_pt")
    if intent in ("long_vega", "short_vega", "delta_neutral_vol", "skew") and ev:
        tv = tgt.get("vega_pt", ev)
        if tv and abs(net["vega_pt"] / tv - 1) > VEGA_DRIFT:
            out.append({"kind": "vega_drift", "severity": "medium",
                        "text": f"vega is ${net['vega_pt']:,.0f} per vol point against ${tv:,.0f} at entry "
                                f"({net['vega_pt'] / tv - 1:+.0%}); the vol exposure isn't what it was put on for",
                        "numbers": {"vega_pt": net["vega_pt"], "target": tv}})
        if (ev > 0) != (net["vega_pt"] > 0) and abs(net["vega_pt"]) > 1:
            out.append({"kind": "vega_flip", "severity": "high",
                        "text": "vega has changed sign since entry: the position is now the opposite vol trade",
                        "numbers": {"vega_pt": net["vega_pt"], "entry": ev}})

    if intent in VOL_INTENTS and net["vega_pt"]:
        d_move = abs(net["dollar_delta"]) * sigma_day       # $ P&L of a 1σ day from delta
        v_move = abs(net["vega_pt"])                        # $ P&L of a 1 vol point move
        if d_move > 2 * v_move:
            out.append({"kind": "not_a_vol_trade", "severity": "medium",
                        "text": f"a 1σ day moves it ${d_move:,.0f} through delta vs ${v_move:,.0f} per vol point: "
                                "it's now mostly a directional position",
                        "numbers": {"delta_pnl_1sd": d_move, "vega_pnl_1pt": v_move}})

    if intent == "skew" and skew_now is not None and pos.get("entry_skew25") is not None:
        ch = skew_now - pos["entry_skew25"]
        out.append({"kind": "skew_since_entry", "severity": "info",
                    "text": f"25-delta skew is {skew_now:.1%} vs {pos['entry_skew25']:.1%} at entry ({ch * 100:+.1f} pts)",
                    "numbers": {"skew_now": skew_now, "skew_entry": pos["entry_skew25"]}})

    eg = pos.get("entry_greeks", {}).get("gamma_sh_1pct")
    if eg and (eg > 0) != (net["gamma_sh_1pct"] > 0) and abs(net["gamma_sh_1pct"]) > 0.5:
        out.append({"kind": "gamma_flip", "severity": "high",
                    "text": "gamma has changed sign since entry: big moves now hurt where they used to help (or the reverse)",
                    "numbers": {"gamma_sh_1pct": net["gamma_sh_1pct"], "entry": eg}})

    live = [lg for lg in g["legs"] if not lg.get("expired")]
    if any(lg.get("expired") for lg in g["legs"]):
        out.append({"kind": "expired_leg", "severity": "high", "text": "a leg has expired: settle it and re-check the hedge",
                    "numbers": {}})
    near = [lg for lg in live if lg["dte"] <= NEAR_EXPIRY_DAYS]
    if near:
        out.append({"kind": "near_expiry", "severity": "medium",
                    "text": f"{min(l['dte'] for l in near)} days to the front expiry: gamma and charm get large, delta "
                            f"moves about {net['charm_sh_day']:+.0f} shares a day from time alone",
                    "numbers": {"dte": min(l["dte"] for l in near), "charm_sh_day": net["charm_sh_day"]}})
    if event_before_expiry:
        out.append({"kind": "gap_risk", "severity": "medium",
                    "text": f"{event_before_expiry['name']} on {event_before_expiry['date']} is before expiry: a delta "
                            "hedge doesn't protect a jump",
                    "numbers": event_before_expiry})
    if g.get("unquoted_legs"):
        out.append({"kind": "stale_quote", "severity": "medium",
                    "text": "no live quote for " + ", ".join(g["unquoted_legs"]) + "; greeks use the entry IV",
                    "numbers": {}})
    return out


# ---------------------------------------------------------------- hedge ticket (SPEC §4.3.3)

def hedge_ticket(pos: dict, S: float, today: date, quote, sigma_day: float, target_delta: float | None = None,
                 at_close: bool = True, r: float = 0.04, q: float = 0.0) -> dict:
    """Shares to trade so net delta = target, at the close if spot is unchanged (time decay of delta = charm), plus a
    ladder at −2..+2% and ±1σ_day. IV held at today's mid-implied."""
    tgt = target_delta if target_delta is not None else pos.get("targets", {}).get("delta_sh", 0.0)
    frac = 0.25 if at_close else 0.0          # fraction of a day to the close; the weekend effect is shown via charm

    def delta_at(spot: float) -> float:
        d = pos.get("hedge_shares", 0.0)
        for lg in pos["legs"]:
            exp = date.fromisoformat(lg["expiry"])
            if exp < today:
                continue
            try:
                iv, _ = quote(lg["kind"], lg["strike"], exp)
            except KeyError:
                iv = lg.get("entry_iv") or 0.3
            d += bs_greeks(spot, lg["strike"], _T(lg["expiry"], today, frac), r, q, max(iv, 1e-6), lg["kind"]).delta * lg["qty"] * MULT
        return d

    d0 = delta_at(S)
    g = position_greeks(pos, S, today, quote, r, q)
    band = band_shares(g["net"], S, sigma_day)
    ladder = []
    moves = [("-2%", -0.02), ("-1σ", -sigma_day), ("-1%", -0.01), ("0", 0.0), ("+1%", 0.01), ("+1σ", sigma_day), ("+2%", 0.02)]
    for lbl, m in sorted(moves, key=lambda x: x[1]):
        d = delta_at(S * (1 + m))
        ladder.append({"move": lbl, "spot": round(S * (1 + m), 2), "net_delta_sh": round(d, 1), "trade_sh": round(tgt - d)})
    trade = round(tgt - d0)
    return {"id": pos["id"], "underlying": pos["underlying"], "spot": S, "at": "close" if at_close else "now",
            "net_delta_sh": round(d0, 1), "target_delta_sh": tgt, "trade_shares": trade,
            "trade_notional": round(abs(trade) * S, 2), "band_sh": round(band, 1), "inside_band": abs(tgt - d0) <= band,
            "charm_sh_day": g["net"]["charm_sh_day"], "ladder": ladder,
            "note": "inside the band: no trade needed" if abs(tgt - d0) <= band else None}


# ---------------------------------------------------------------- writes (paper book only)

def new_position(member: str, underlying: str, legs: list[dict], intent: str, S: float, today: date, quote,
                 targets: dict | None = None, skew25: float | None = None, note: str = "", r: float = 0.04, q: float = 0.0) -> dict:
    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {INTENTS}")
    clean = []
    for lg in legs:
        exp = date.fromisoformat(str(lg["expiry"]))
        iv, mid = quote(lg["kind"], float(lg["strike"]), exp)
        clean.append({"kind": lg["kind"], "strike": float(lg["strike"]), "expiry": exp.isoformat(), "qty": float(lg["qty"]),
                      "entry_mid": float(lg.get("entry_mid") or mid), "entry_iv": iv})
    pos = {"id": "pos-" + uuid.uuid4().hex[:6], "member": member, "underlying": underlying.upper(), "legs": clean,
           "intent": intent, "opened": today.isoformat(), "entry_spot": S, "hedge_shares": 0.0, "hedges": [],
           "status": "open", "note": note, "entry_skew25": skew25, "ts": time.time()}
    g = position_greeks(pos, S, today, quote, r, q)["net"]
    pos["entry_greeks"] = g
    t = dict(targets or {})
    if intent in VOL_INTENTS:
        t.setdefault("delta_sh", 0.0)
        t.setdefault("vega_pt", g["vega_pt"])
    else:
        t.setdefault("delta_sh", g["delta_sh"])
    pos["targets"] = t
    rows = load(member)
    rows.append(pos)
    save(member, rows)
    return pos


def record_hedge(member: str, pos_id: str, shares: float, price: float, today: date) -> dict:
    rows = load(member)
    for p in rows:
        if p["id"] == pos_id:
            p["hedge_shares"] = p.get("hedge_shares", 0.0) + shares
            p.setdefault("hedges", []).append({"date": today.isoformat(), "shares": shares, "price": price, "ts": time.time()})
            save(member, rows)
            return p
    raise KeyError(pos_id)


def close_position(member: str, pos_id: str, today: date) -> dict:
    rows = load(member)
    for p in rows:
        if p["id"] == pos_id:
            p["status"], p["closed"] = "closed", today.isoformat()
            save(member, rows)
            return p
    raise KeyError(pos_id)


def legs_from_structure(structure: dict, today: date) -> list[dict]:
    """A priced structure's legs (days to expiry) as dated legs. Stock legs become hedge shares, handled by the caller."""
    return [{"kind": l["kind"], "strike": l["strike"], "qty": l["qty"], "entry_mid": l.get("mid"),
             "expiry": (today + timedelta(days=int(l["days"]))).isoformat()} for l in structure.get("legs", []) if l["kind"] != "stock"]


def sigma_day_from(rv_annual: float | None, iv: float | None = None) -> float:
    v = rv_annual if rv_annual and math.isfinite(rv_annual) and rv_annual > 0 else (iv or 0.3)
    return v / math.sqrt(252)


def daily_note(status: dict) -> str | None:
    """The once-a-day DM (code-written, no model): only positions with high or medium alerts, most severe first."""
    rank = {"high": 0, "medium": 1}
    lines = []
    for r in status.get("positions", []):
        al = sorted([a for a in r["alerts"] if a["severity"] in rank], key=lambda a: rank[a["severity"]])
        if al:
            p = r["position"]
            lines.append(f"{p['underlying']} {p['intent'].replace('_', ' ')} ({p['id']}): " + "; ".join(a["text"] for a in al[:3]) + ".")
    if not lines:
        return None
    return ("**Position check.** " + " ".join(lines[:4]) +
            " Ask me for a hedge ticket on any of these (paper book; nothing is sent to a broker).")
