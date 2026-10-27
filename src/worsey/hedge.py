"""The bridge: hedge comparator (SPEC §5).

Given a scenario (an event with p and per-position dollar betas, or a user view of an
index move), build 3–5 rule-generated hedges across instrument types and compute, in
code, the same metrics for each so the basis-risk trade-off is visible in one table.

Sizing (stated assumption, an open question in SPEC §12): each hedge is sized linearly
so its P&L if YES offsets the portfolio's loss if YES. Option hedges are valued at the
event's resolution date by BSM repricing, not by delta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .options.builder import Quoter, pick_expiry, snap
from .options.pricer import MULT, Leg, Strategy
from . import config
from .options.view import View

BASIS_RISK = {
    "do_nothing": "None: you keep the full exposure. At the market's odds the expected resolution P&L is about zero; the risk is the swing.",
    "prediction_market": "None to the event itself; full to the size of your loss (β is an estimate).",
    "index_put_spread": "Pays on the market move whatever the cause; can miss if the hit is sector-specific. Sized to your β-weighted exposure, so it covers what the index move explains, not more.",
    "sector_puts": "Targets the holdings driving the loss; thinner liquidity and higher vol.",
    "cross_asset_call": "Pays through a cross-asset channel (e.g. flight to safety, oil); relies on a correlation that can break.",
    "cross_asset_put": "Pays if this asset falls when the event hits (e.g. bonds selling off on an inflation scare); relies on a correlation that can break.",
}


@dataclass
class Scenario:
    event_id: str
    p: float | None               # market probability now; None for an unpriced (template) event
    resolution: date              # when the event resolves (hedges are valued then)
    loss_if_yes: float            # portfolio P&L if YES (negative for a loss), = β$·(1−p)
    pnl_if_no: float              # = −β$·p
    asset_moves_yes: dict         # ticker -> return if YES (βᵢ·(1−p)), for the hedge underlyings
    asset_moves_no: dict          # ticker -> return if NO (−βᵢ·p)


@dataclass
class Underlying:
    ticker: str
    S: float
    quoter: Quoter


def _row(kind, name, cost, pnl_yes, pnl_no, sc: Scenario, detail: dict) -> dict:
    loss = -sc.loss_if_yes  # positive dollars lost if YES
    gross_yes = pnl_yes + cost  # what the hedge pays out if YES, before its cost
    return {
        "hedge": kind,
        "name": name,
        "cost": cost,
        "pnl_if_yes": pnl_yes,
        "pnl_if_no": pnl_no,
        "coverage": pnl_yes / loss if loss > 0 else None,
        "cost_per_dollar_covered": cost / gross_yes if gross_yes > 0 else None,
        "residual_if_yes": sc.loss_if_yes + pnl_yes,
        "book_if_no": sc.pnl_if_no + pnl_no,
        "cost_to_loss_multiple": cost / loss if loss > 0 else None,   # "costs 6x the loss it covers": from code
        "basis_risk": BASIS_RISK[kind],
        **detail,
    }


def do_nothing(sc: Scenario, book_value: float | None = None) -> dict:
    """Always a row (SPEC §1f-0 rule 6, METHODOLOGY §12): the unhedged book, with its own risk numbers."""
    priced = sc.p is not None
    B = sc.loss_if_yes / (1 - sc.p) if priced and sc.p < 1 else 0.0  # portfolio $ per +1.00 Δp
    row = {
        "hedge": "do_nothing", "name": "Do nothing", "cost": 0.0, "pnl_if_yes": 0.0, "pnl_if_no": 0.0,
        "coverage": 0.0, "cost_per_dollar_covered": None,
        "residual_if_yes": sc.loss_if_yes, "book_if_no": sc.pnl_if_no,
        # unpriced (template) events have no probability: these stay empty rather than invented
        "expected_resolution_pnl": sc.p * sc.loss_if_yes + (1 - sc.p) * sc.pnl_if_no if priced else None,
        "sigma_resolution": abs(B) * (sc.p * (1 - sc.p)) ** 0.5 if priced else None,
        "basis_risk": BASIS_RISK["do_nothing"] if priced else
        "None: you keep the full exposure. No market prices this event, so there's no probability to weigh it by.",
    }
    if book_value:
        row["loss_if_yes_pct_of_book"] = sc.loss_if_yes / book_value
    return row


def prediction_market_hedge(sc: Scenario, price: float | None = None, fee_per_contract: float = 0.0) -> dict:
    """Buy YES contracts ($1 payout) so the payout if YES offsets the loss if YES."""
    px = sc.p if price is None else price  # pay the ask in practice; mid by default
    loss = -sc.loss_if_yes
    n = loss / (1 - px) if px < 1 else 0.0
    cost = n * (px + fee_per_contract)
    return _row("prediction_market", f"Buy {sc.event_id} YES", cost, n * (1 - px) - n * fee_per_contract, -cost, sc,
                {"contracts": n, "price": px})


def option_hedge(kind: str, und: Underlying, opt: str, sc: Scenario, today: date, exposure_usd: float | None = None,
                 r: float = 0.04, q: float = 0.0) -> dict | None:
    """An option hedge sized and struck to the scenario, not stretched to "cover the full loss".

    - Notional: `exposure_usd`, the β-weighted dollar exposure of the scenario-affected book to this underlying
      (positions that lose if YES, Σ Vᵢ·βᵢ,und). Contracts = notional / (spot · 100). Never inflated.
    - Strikes from the horizon vol σ_h = ATM IV·√(time to resolution): long leg HEDGE_LONG_SIGMA·σ_h OTM (never ITM),
      short leg HEDGE_SHORT_SIGMA·σ_h OTM. Expiry: first listed on/after resolution.
    - Coverage is whatever that position pays if YES, as a share of the event loss; cost > loss is flagged.
    Returns None when the underlying doesn't move in the hedge's direction if YES.
    """
    mv_yes = sc.asset_moves_yes.get(und.ticker)
    mv_no = sc.asset_moves_no.get(und.ticker, 0.0)
    if mv_yes is None or (opt in ("put_spread", "put") and mv_yes >= 0) or (opt == "call" and mv_yes <= 0):
        return None
    view = View(und.ticker, "neutral", sc.resolution)  # only used for expiry selection
    exp = pick_expiry("horizon", view, und.quoter.expiries(), None)
    T = max((exp - today).days, 1) / 365
    t_res = max((sc.resolution - today).days, 0) / 365
    ks = sorted(und.quoter.strikes(exp))
    atm_iv, _ = und.quoter.quote("put" if opt != "call" else "call", snap(und.S, ks), exp)
    sig_h = atm_iv * math.sqrt(max(t_res, config.HEDGE_MIN_HORIZON_DAYS / 365))
    below = [k for k in ks if k <= und.S] or ks[:1]
    above = [k for k in ks if k >= und.S] or ks[-1:]
    if opt == "put_spread":
        t1 = und.S * math.exp(-config.HEDGE_LONG_SIGMA * sig_h)
        k1 = max([k for k in below if k <= t1] or below[:1])          # highest strike at/below the target: OTM
        t2 = und.S * math.exp(-config.HEDGE_SHORT_SIGMA * sig_h)
        lower = [k for k in ks if k < k1]
        k2 = max([k for k in lower if k <= t2] or lower[:1] or [k1])
        if k2 >= k1:
            return {"hedge": kind, "name": f"{und.ticker} put spread", "skipped": "no listed strike below the long leg"}
        specs = [("put", 1, k1), ("put", -1, k2)]
        label = f"{und.ticker} {k1:g}/{k2:g} put spread"
    elif opt == "put":
        t1 = und.S * math.exp(-config.HEDGE_LONG_SIGMA * sig_h)
        k1 = max([k for k in below if k <= t1] or below[:1])
        specs, label = [("put", 1, k1)], f"{und.ticker} {k1:g} put"
    else:
        t1 = und.S * math.exp(config.HEDGE_LONG_SIGMA * sig_h)
        k1 = min([k for k in above if k >= t1] or above[-1:])
        specs, label = [("call", 1, k1)], f"{und.ticker} {k1:g} call"
    legs = []
    for kd, qty, k in specs:
        iv, mid = und.quoter.quote(kd, k, exp)
        legs.append(Leg(kd, qty, k, T, iv, premium=mid))
    one = Strategy(label, legs, und.S, r, q)
    notional = abs(exposure_usd) if exposure_usd else -sc.loss_if_yes / max(abs(mv_yes), 1e-9)
    n = notional / (und.S * MULT)                     # contracts covering the β-weighted exposure, no more
    per_yes = one.pnl(und.S * (1 + mv_yes), t_res)
    per_no = one.pnl(und.S * (1 + mv_no), t_res)
    cost = n * one.cost()
    loss = -sc.loss_if_yes
    flags = []
    if per_no > 1e-6:
        flags.append("pays_if_no")                    # should not happen by construction (property-tested)
    if loss > 0 and cost > loss:
        flags.append("cost_exceeds_loss")             # sanity check: premium should be on the order of the loss
    if per_yes <= 0:
        flags.append("does_not_pay_if_yes")           # the scenario move is too small to beat the premium
    return _row(kind, label, cost, n * per_yes, n * per_no, sc,
                {"contracts": n, "notional_usd": notional, "underlying": und.ticker, "expiry": exp.isoformat(),
                 "flags": flags, "horizon_sigma": sig_h, "underlying_move_if_yes": mv_yes,
                 "legs": [{"kind": l.kind, "qty": l.qty, "strike": l.strike, "iv": l.iv} for l in legs]})


def _derived(out: dict, book_value: float | None) -> dict:
    """Every derived figure a reply might quote, computed here (grounding, Oct 16)."""
    rows = [r for r in out["rows"] if "cost" in r]
    cheapest = min((r["cost"] for r in rows if r["hedge"] != "do_nothing"), default=None)
    for r in rows:
        if book_value:
            r["cost_pct_of_book"] = r["cost"] / book_value
        if cheapest and r["hedge"] != "do_nothing":
            r["cost_vs_cheapest_multiple"] = r["cost"] / cheapest
    return out


def compare_hedges(sc: Scenario, today: date, index: Underlying | None = None, sector: Underlying | None = None,
                   cross: Underlying | list[Underlying] | None = None, include_market: bool = True, r: float = 0.04,
                   book_value: float | None = None, exposures: dict | None = None) -> dict:
    """exposures: underlying → β-weighted dollar exposure of the scenario-affected book (notional per hedge)."""
    """The §5 table: do nothing vs prediction contract vs index put spread vs sector puts vs cross-asset calls."""
    rows = [do_nothing(sc, book_value)]
    include_market = include_market and sc.p is not None  # no market, no contract to buy
    if sc.loss_if_yes >= 0:
        return {"event_id": sc.event_id, "p": sc.p, "unpriced": sc.p is None, "rows": rows, "note": "the book gains if YES: nothing to hedge"}
    if include_market:
        rows.append(prediction_market_hedge(sc))
    crosses = cross if isinstance(cross, list) else [cross]
    legs = [("index_put_spread", index, "put_spread"), ("sector_puts", sector, "put_spread")]
    for c in crosses:  # a cross-asset channel hedges with a call if the asset rises on YES, a put if it falls
        if c is not None:
            up = sc.asset_moves_yes.get(c.ticker, 0.0) > 0
            legs.append(("cross_asset_call" if up else "cross_asset_put", c, "call" if up else "put"))
    for kind, und, opt in legs:
        if und is None:
            continue
        row = option_hedge(kind, und, opt, sc, today, (exposures or {}).get(und.ticker), r)
        if row is None:  # wrong direction; the 'too small' case returns its own skipped row
            rows.append({"hedge": kind, "name": und.ticker, "skipped": f"{und.ticker} doesn't move the right way if YES"})
        else:
            rows.append(row)
    return _derived({"event_id": sc.event_id, "p": sc.p, "unpriced": sc.p is None, "loss_if_yes": sc.loss_if_yes, "pnl_if_no": sc.pnl_if_no, "rows": rows,
                     "sizing": "each hedge sized linearly to offset the loss if YES; options valued at resolution by BSM"}, book_value)


def pick_cross_asset(asset_moves_yes: dict, candidates=("TLT", "GLD", "UUP", "USO"), n: int = 2,
                     min_move: float = 0.005) -> list[str]:
    """Cross-asset channels: the candidates that move most (either way) if YES, biggest first.
    A rise → calls (flight to safety, oil spike); a fall → puts (e.g. bonds on an inflation scare).
    Channels that barely move (< 0.5% if YES) are dropped: no option on them can pay for itself."""
    mv = {t: m for t, m in asset_moves_yes.items() if t in candidates and abs(m or 0.0) >= min_move}
    return sorted(mv, key=lambda t: -abs(mv[t]))[:n]
