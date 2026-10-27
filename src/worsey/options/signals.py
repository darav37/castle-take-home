"""The trainer's playbook signals, computed (library/strategies/*.md, "Playbook rules (machine-usable)").

Two kinds:
- history signals: z-scores / percentiles of today's surface vs its own trailing ~1y (options/vol_history.py)
- trade facts: per-candidate numbers the rules ask Worsey to always show (theta/day, payout multiple,
  stress P&L, vol exposure, event coverage, moneyness drift).

Every one is a flag, not a trade signal. The LLM explains them
in the trainer's words from the strategy file; it never computes them.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from .pricer import Strategy

MIN_OBS = 60          # fewer rows than this → "not enough history yet", never a z-score
WINDOW = 252          # ~1y of trading days
CORR_WINDOW = 63      # ~3m for spot–vol correlation


# ---------------------------------------------------------------- history signals


def _z(series: pd.Series, value: float) -> tuple[float | None, int]:
    s = series.dropna().iloc[-WINDOW:]
    if len(s) < MIN_OBS or value is None or (isinstance(value, float) and math.isnan(value)) or s.std() == 0:
        return None, len(s)
    return float((value - s.mean()) / s.std()), len(s)


def _pct(series: pd.Series, value: float) -> float | None:
    s = series.dropna().iloc[-WINDOW:]
    if len(s) < MIN_OBS or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float((s < value).mean())


def history_signals(hist: pd.DataFrame, today_row: dict | None = None) -> dict:
    """Signals for today vs trailing history. today_row (vol_history.chain_row) overrides the last row
    when given, so a live chain is scored against history built before today."""
    h = hist.copy()
    if today_row:
        h = pd.concat([h, pd.DataFrame([today_row], index=[h.index.max() + pd.Timedelta(days=1) if len(h) else pd.Timestamp.today()])])
    if h.empty:
        return {"history_days": 0, "note": "no IV history for this underlying yet"}
    for c in ("atm_front", "put25_front", "call25_front", "atm_back"):
        if c not in h:
            h[c] = np.nan
    h = h.apply(pd.to_numeric, errors="coerce")
    put_call = h.put25_front - h.call25_front          # risk-reversal skew
    call_wing = h.call25_front - h.atm_front           # upside call skew
    term = h.atm_front - h.atm_back                    # front − back (> 0 = inverted)
    hist_part = h.iloc[:-1] if today_row else h
    last = h.iloc[-1]
    hp = lambda s: s.loc[hist_part.index]
    out = {"history_days": int(hist_part.atm_front.notna().sum())}
    out["atm_iv_z"], _ = _z(hp(h.atm_front), last.atm_front)
    out["skew_rr_z"], _ = _z(hp(put_call), put_call.iloc[-1])
    out["call_skew_pct"] = _pct(hp(call_wing), call_wing.iloc[-1])
    out["term_z"], _ = _z(hp(term), term.iloc[-1])
    out["skew_rr"], out["call_skew"], out["term_spread"] = (float(x) if pd.notna(x) else None
                                                          for x in (put_call.iloc[-1], call_wing.iloc[-1], term.iloc[-1]))
    # realized skew: corr(daily return, daily change in front ATM IV) over ~3m
    r = np.log(h.spot.astype(float)).diff()
    dv = h.atm_front.diff()
    both = pd.concat([r, dv], axis=1).dropna().iloc[-CORR_WINDOW:]
    out["spot_vol_corr"] = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) >= 20 else None
    if out["history_days"] < MIN_OBS:
        out["note"] = f"only {out['history_days']} days of IV history; z-scores need {MIN_OBS}"
    return out


# ---------------------------------------------------------------- trade facts


def stress(st: Strategy, moves=(-0.10, -0.20), vol_up=(0.10, 0.20), skew_steepen: float = 0.05) -> list[dict]:
    """P&L for a down move with ATM vol up and puts bid harder. Puts below spot get
    `skew_steepen` extra vol per 10% OTM on top of the parallel bump; spot moves, no time passes."""
    out = []
    for m, dv in zip(moves, vol_up):
        legs = []
        for l in st.legs:
            if l.kind == "stock":
                legs.append(l)
                continue
            extra = skew_steepen * max(0.0, math.log(st.S / l.strike)) / 0.10 if l.kind == "put" else 0.0
            legs.append(replace(l, iv=l.iv + dv + extra))
        bumped = Strategy(st.name, legs, st.S, st.r, st.q, dict(st.meta))
        out.append({"spot_move": m, "atm_vol_up": dv, "pnl": bumped.value(st.S * (1 + m)) - st.cost()})
    return out


def trade_facts(st: Strategy, metrics: dict, implied_move: float | None, entry: dict) -> dict:
    """Always-show numbers from the trainer's rules, for one priced candidate."""
    g = st.greeks()
    cost = metrics["cost"]
    f = {"theta_per_day": g["theta"],
         "theta_pct_of_premium": g["theta"] / abs(cost) if cost else None,
         "vega_per_vol_pt": g["vega"], "delta": g["delta"],
         "vol_exposure": "long vol (a vol drop hurts)" if g["vega"] > 0 else "short vol (a vol drop helps)",
         "dte_first_leg": round(st.first_expiry * 365)}
    mg = metrics.get("max_gain")
    if cost > 0 and mg is not None and math.isfinite(mg):
        f["max_payout_multiple"] = (mg + cost) / cost        # best-case value ÷ premium
    shorts = [l for l in st.legs if l.qty < 0 and l.kind in ("call", "put")]
    longs = [l for l in st.legs if l.qty > 0 and l.kind in ("call", "put")]
    if entry["id"] in ("call_debit_spread", "put_debit_spread") and shorts and longs:
        ks, kl = shorts[0], longs[0]
        from .pricer import bs_greeks
        p = abs(bs_greeks(st.S, ks.strike, ks.T, st.r, st.q, ks.iv, ks.kind).delta)
        f["prob_past_short_strike"] = p                       # |Δ| of the short strike as the proxy
        width = abs(ks.strike - kl.strike) * 100
        f["value_pct_of_max"] = (metrics["cost"] / width) if width else None
        f["long_minus_short_iv"] = kl.iv - ks.iv            # < 0 for a put spread: you sell the richer vol
        f["spot_vs_strikes"] = ("above short strike" if (st.S > ks.strike if ks.kind == "call" else st.S < ks.strike)
                                else "below long strike" if (st.S < kl.strike if kl.kind == "call" else st.S > kl.strike)
                                else "between strikes")
    if entry["id"] == "long_strangle":
        from .pricer import bs_greeks
        opts = [l for l in st.legs if l.kind in ("call", "put")]
        kp = [l for l in opts if l.kind == "put"]
        kc = [l for l in opts if l.kind == "call"]
        f["max_leg_delta"] = max(abs(bs_greeks(st.S, l.strike, l.T, st.r, st.q, l.iv, l.kind).delta) for l in opts)
        if kp and kc:   # risk-neutral P(K_put < S_T < K_call) = N(d2(K_put)) − N(d2(K_call)) at each leg's own IV
            def n_d2(l):
                d2 = (math.log(st.S / l.strike) + (st.r - st.q - l.iv ** 2 / 2) * l.T) / (l.iv * math.sqrt(l.T))
                return 0.5 * math.erfc(-d2 / math.sqrt(2))
            f["prob_both_otm"] = max(0.0, n_d2(kp[0]) - n_d2(kc[0]))
    if entry.get("risk") == "undefined" or entry["id"] in ("long_risky", "short_risky"):
        f["stress"] = stress(st)
    if implied_move:
        f["implied_move"] = implied_move
    f["_legs"] = st.legs          # for alerts only; stripped before the tool returns
    return f


def alerts(entry: dict, facts: dict, sig: dict, ctx: dict) -> list[dict]:
    """The trainer's machine rules → alerts. Each carries the rule's file and number so the explainer can use his
    wording from library/strategies/<file>.md; numbers come from here."""
    sid, out = entry["id"], []
    add = lambda rule, level, **nums: out.append({"rule": f"{sid}#{rule}", "level": level, **nums})
    dte = facts.get("dte_first_leg")
    has_event = ctx.get("has_event")
    z = lambda k: sig.get(k)

    if sid == "call_debit_spread":
        if z("call_skew_pct") is not None and z("call_skew_pct") >= 0.80:
            add(1, "info", call_skew=z("call_skew"), call_skew_percentile=z("call_skew_pct"))
        if dte is not None and dte < 5 and not has_event:
            add(4, "warn", dte=dte)
        add(6, "info", theta_per_day=facts["theta_per_day"])
        if facts.get("spot_vs_strikes") == "above short strike":
            add(7, "info", exposure="short vega now; most of the upside is spent")
        if (v := facts.get("value_pct_of_max")) is not None and 0.5 <= v and dte and dte > 5:
            add(8, "info", value_pct_of_max=v)

    if sid == "put_debit_spread":
        # rule 1: inherit the call-spread rules, mirrored
        if dte is not None and dte < 5 and not has_event:
            add(1, "warn", dte=dte, inherited="call_debit_spread#4")
        add(1, "info", theta_per_day=facts["theta_per_day"], inherited="call_debit_spread#6")
        if (v := facts.get("value_pct_of_max")) is not None and 0.5 <= v and dte and dte > 5:
            add(1, "info", value_pct_of_max=v, inherited="call_debit_spread#8")
        add(2, "info", skew_rr=z("skew_rr"), skew_rr_z=z("skew_rr_z"),
            long_minus_short_strike_iv=facts.get("long_minus_short_iv"))
        if facts.get("spot_vs_strikes") == "above short strike":     # put spread: spot below the short strike
            add(5, "warn", note="through the short strike; rolling down now likely costs more (vol spiking)")

    if sid == "long_strangle":
        add(1, "info", implied_move=facts.get("implied_move"), inherited="long_straddle#1")
        if has_event:
            add(1, "warn", implied_move=facts.get("implied_move"), inherited="long_straddle#10")
        if z("atm_iv_z") is not None and z("atm_iv_z") <= -1:
            add(1, "info", atm_iv_z=z("atm_iv_z"), inherited="long_straddle#3")
        add(1, "info", theta_per_day=facts["theta_per_day"], theta_pct_of_premium=facts.get("theta_pct_of_premium"),
            inherited="long_straddle#6")
        add(3, "warn", prob_expire_worthless=facts.get("prob_both_otm"))
        if facts.get("max_leg_delta") is not None and facts["max_leg_delta"] <= 0.30:
            add(4, "info", max_leg_delta=facts["max_leg_delta"])
        add(5, "info", ask="short-term kurtosis play, or a directional view with both legs out of the money?")

    if sid == "long_straddle":
        add(1, "info", implied_move=facts.get("implied_move"))
        if dte is not None and dte <= 1 or has_event:
            add(2, "info", earnings_hist_abs_moves=ctx.get("earnings_hist_abs_moves"),
                implied_vs_hist=ctx.get("implied_vs_hist"))
        if z("atm_iv_z") is not None and z("atm_iv_z") <= -1:
            add(3, "info", atm_iv_z=z("atm_iv_z"))
            rv, iv = ctx.get("rv20"), ctx.get("atm_iv_front")
            if (rv and iv and rv < iv) or not has_event:
                add(4, "warn", rv20=rv, atm_iv=iv, catalyst_in_window=bool(has_event))
        if dte is not None and dte > 10 and not has_event:
            add(5, "info", iv_rv=ctx.get("iv_rv"))
        add(6, "info", theta_per_day=facts["theta_per_day"], theta_pct_of_premium=facts.get("theta_pct_of_premium"))
        if has_event:
            add(10, "warn", implied_move=facts.get("implied_move"))

    if sid in ("call_calendar", "reverse_calendar"):
        if z("term_z") is not None:
            if (sid == "call_calendar" and z("term_z") >= 1) or (sid == "reverse_calendar" and z("term_z") <= -1):
                add(2, "info", term_spread=z("term_spread"), term_z=z("term_z"), event_in_front=bool(has_event))
        add(5, "warn", delta=facts.get("delta"))
        im, k = ctx.get("front_implied_move"), next((l.strike for l in facts.get("_legs", [])), None)
        if im and k and abs(ctx["spot"] - k) / ctx["spot"] > 0.5 * im:
            add(6, "warn", spot=ctx["spot"], strike=k, front_implied_move=im)
        if dte is not None and dte <= 5:
            add(7, "warn", dte=dte)

    if sid in ("long_risky", "short_risky"):
        add(3, "warn", delta=facts.get("delta"),
            trade="mostly a delta trade unless delta-hedged; hedged it is a skew trade")
        if z("skew_rr_z") is not None and abs(z("skew_rr_z")) >= 1:
            add(4, "info", skew_rr=z("skew_rr"), skew_rr_z=z("skew_rr_z"))
        if z("spot_vol_corr") is not None:
            add(5, "info", spot_vol_corr=z("spot_vol_corr"), implied_skew=z("skew_rr"))
        add(6, "warn" if sid == "short_risky" else "info", stress=facts.get("stress"))
    return out


# ---------------------------------------------------------------- protection (put spread as a hedge)


def book_beta_notional(values: dict[str, float], closes: pd.DataFrame, underlying: str, window: int = 252) -> dict:
    """Σ value_i · β_i to the hedge underlying, β from daily log returns over ~1y (OLS). Positions without
    enough history count at β = 1 and are listed."""
    r = np.log(closes.astype(float)).diff().iloc[-window:]
    if underlying not in r:
        raise KeyError(f"no closes for {underlying}")
    x = r[underlying]
    betas, assumed = {}, []
    for t, v in values.items():
        if t == underlying:
            betas[t] = 1.0
            continue
        if t not in r or r[t].notna().sum() < 60:
            betas[t] = 1.0
            assumed.append(t)
            continue
        both = pd.concat([r[t], x], axis=1).dropna()
        betas[t] = float(np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / both.iloc[:, 1].var())
    return {"beta_notional": float(sum(values[t] * betas[t] for t in values)), "betas": betas,
            "beta_assumed_1": assumed, "book_value": float(sum(values.values()))}


def hedge_facts(st: Strategy, book: dict, tail_moves=(-0.20, -0.30)) -> dict:
    """The trainer's put-spread rules 4, 6, 7: size to beta-adjusted notional, show the uncovered tail and the
    yearly cost of keeping it. Payoffs at expiry, spot-only (the book moves β × the index move)."""
    S, bn, bv = st.S, book["beta_notional"], book["book_value"]
    n = max(1, round(bn / (S * 100)))                       # contracts covering the beta-adjusted notional
    premium = st.cost() * n
    dte = max(1, round(st.first_expiry * 365))
    g = st.greeks()
    tail = []
    for m in tail_moves:
        book_pnl = bn * m
        spread = st.payoff_at(np.array([S * (1 + m)]), t_elapsed=st.first_expiry)[0] * n
        tail.append({"index_move": m, "book_unhedged": book_pnl, "book_hedged": book_pnl + spread,
                     "spread_pnl": spread})
    return {"contracts_for_beta_notional": n, "beta_notional": bn, "book_value": bv,
            "hedge_ratio_delta": abs(g["delta"] * S * n) / bn if bn else None,
            "premium_total": premium, "premium_pct_of_book": premium / bv if bv else None,
            "annualized_cost_pct_if_rolled": premium / bv * 365 / dte if bv else None,
            "dte": dte, "tail": tail, "beta_assumed_1": book.get("beta_assumed_1")}


def hedge_alerts(facts: dict, short_strike: float, breakeven: float | None) -> list[dict]:
    sid = "put_debit_spread"
    out = [{"rule": f"{sid}#4", "level": "warn", "tail": facts["tail"], "short_strike": short_strike},
           {"rule": f"{sid}#6", "level": "info", "contracts": facts["contracts_for_beta_notional"],
            "beta_notional": facts["beta_notional"], "hedge_ratio_delta": facts["hedge_ratio_delta"]},
           {"rule": f"{sid}#7", "level": "info", "annualized_cost_pct_if_rolled": facts["annualized_cost_pct_if_rolled"]}]
    if facts["dte"] <= 45:
        out.append({"rule": f"{sid}#8", "level": "warn", "premium_total": facts["premium_total"],
                    "loses_premium_unless_below": breakeven})
    return out


# ---------------------------------------------------------------- open-position alerts (strangle rules 6–8)


def strangle_position_alerts(entry_cost: float, value_now: float, dte: int, wing_iv_change_z: float | None,
                             purpose: str | None, last_alerted_step: int = 0, nearer_leg: str | None = None) -> list[dict]:
    """The trainer's strangle management rules for an OPEN position (call from position management, not proposals).
    purpose: 'kurtosis' | 'directional' | None. Profit alerts at each +10% of entry premium, once per step;
    for weeks-to-months tenors (> 21 DTE) a vol alert when the wing IV's daily change is ≥ 1.5σ."""
    sid, out = "long_strangle", []
    gain = (value_now - entry_cost) / entry_cost if entry_cost > 0 else 0.0
    step = int(math.floor(gain / 0.10 + 1e-9)) if gain > 0 else 0
    if step > last_alerted_step and (purpose != "directional" and dte <= 21):
        out.append({"rule": f"{sid}#6", "level": "info", "gain_pct": gain, "step": step})
    if dte > 21 and wing_iv_change_z is not None and wing_iv_change_z >= 1.5:
        out.append({"rule": f"{sid}#7", "level": "info", "wing_iv_change_z": wing_iv_change_z, "gain_pct": gain})
    if purpose == "directional" and nearer_leg:
        out.append({"rule": f"{sid}#8", "level": "info", "nearer_leg": nearer_leg, "gain_pct": gain})
    return out
