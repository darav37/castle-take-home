"""Option positions in event scenarios: full repricing with a measured spot-vol response (DECISIONS #209).

The event engine hands over a move distribution per underlying (the historical estimate's draws, a black-swan shock, or
past events replayed); this module turns it into option P&L:
  spot      S1 = S * (1 + move)
  vol       IV1 = IV + response(move) * move, where response is this underlying's measured change in at-the-money IV
            per unit spot return, separately for down and up moves (a year of cached IV history, cache/vol_history;
            else the CBOE vol index for it from FRED: VIX for SPY, Nasdaq VIX for QQQ, oil VIX for USO, gold VIX for
            GLD; else SPY's response scaled by the ratio of the two IVs)
  time      T1 = T - the event window (2 sessions)
  value     Black-Scholes at (S1, IV1, T1) minus today's value, x 100 x contracts
Full repricing captures delta, gamma, vega, vanna, volga and theta together; greek_split() reports each piece at one
scenario and the remainder vs the full repricing, so the P&L can be explained.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..planb import library as LB
from .pricer import bs_greeks, bs_price

R_RATE, WINDOW_DAYS = 0.04, 2
VOL_INDEX = {"SPY": "VIXCLS", "QQQ": "VXNCLS", "USO": "OVXCLS", "GLD": "GVZCLS"}


@dataclass
class OptionPosition:
    underlying: str
    kind: str            # "call" | "put"
    strike: float
    expiry: date
    contracts: float     # + long, - short

    @property
    def label(self) -> str:
        return f"{self.underlying} {self.strike:g} {self.kind} {self.expiry:%b %d %Y} x{self.contracts:g}"

    @classmethod
    def from_dict(cls, d: dict) -> "OptionPosition":
        o = d.get("option") or d
        return cls(str(d.get("ticker") or o["underlying"]).upper(), str(o["kind"]).lower(), float(o["strike"]),
                   pd.Timestamp(o["expiry"]).date(), float(o.get("contracts", 1)))


def bs_vec(S, K, T, sig, kind: str, r: float = R_RATE) -> np.ndarray:
    S, T, sig = np.asarray(S, float), np.maximum(np.asarray(T, float), 1e-6), np.maximum(np.asarray(sig, float), 1e-4)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def spot_vol_response(s, u: str) -> dict:
    """Change in at-the-money IV (decimal) per unit spot return, down moves and up moves separately (fits through 0)."""
    from . import vol_history
    def fit(dv, ret):
        ok = np.isfinite(dv) & np.isfinite(ret)
        dv, ret = dv[ok], ret[ok]
        out = {}
        for side, m in (("down", ret < 0), ("up", ret > 0)):
            x, y = ret[m], dv[m]
            out[side] = float(x @ y / (x @ x)) if len(x) >= 30 and (x @ x) > 0 else None
        return out, int(ok.sum())
    try:
        h = vol_history.load(u)
        if h is not None and len(h) >= 120:
            h = h.sort_values("date")
            b, n = fit(h["atm_front"].diff().to_numpy(), h["spot"].pct_change().to_numpy())
            if b["down"] is not None and b["up"] is not None:
                return {**b, "source": f"{n} days of {u}'s own at-the-money IV (cache/vol_history)"}
    except Exception:  # noqa: BLE001
        pass
    if u in VOL_INDEX:
        try:
            v = LB.fred_daily(VOL_INDEX[u]) / 100
            c = s.data.closes([u], days=750)[u]
            d = pd.concat([v.reindex(c.index).ffill().diff().rename("dv"), c.pct_change().rename("r")], axis=1).dropna().tail(750)
            b, n = fit(d.dv.to_numpy(), d.r.to_numpy())
            if b["down"] is not None and b["up"] is not None:
                return {**b, "source": f"{n} days of the CBOE vol index for {u} ({VOL_INDEX[u]})"}
        except Exception:  # noqa: BLE001
            pass
    base = spot_vol_response(s, "SPY") if u != "SPY" else {"down": -0.8, "up": -0.4, "source": "fallback"}
    return {"down": base["down"], "up": base["up"], "source": f"SPY's response ({base['source']}); scaled by IV in iv_now"}


def iv_now(s, pos: OptionPosition) -> tuple[float, float, float, str]:
    """(spot, IV, years to expiry, where the IV came from) for this contract today."""
    from ..agent.tools import filled_chain
    from . import market_state
    S = float(s.data.prices([pos.underlying])[pos.underlying])
    T = max((pd.Timestamp(pos.expiry) - pd.Timestamp(s.data.today())).days, 1) / 365
    try:
        ch = filled_chain(s, pos.underlying, S)
        ch = ch.assign(expiry=pd.to_datetime(ch["expiry"]))
        same = ch[(ch.kind == pos.kind) & (ch.expiry == pd.Timestamp(pos.expiry)) & np.isclose(ch.strike, pos.strike)]
        if len(same) and np.isfinite(same.iloc[0].get("iv", np.nan)):
            return S, float(same.iloc[0]["iv"]), T, "this contract's quote"
        term = market_state.atm_iv(ch, S)
        term["expiry"] = pd.to_datetime(term["expiry"])
        near = term.iloc[(term.expiry - pd.Timestamp(pos.expiry)).abs().argsort()[:1]]
        if len(near) and np.isfinite(near.iloc[0].iv):
            return S, float(near.iloc[0].iv), T, f"at-the-money IV of the nearest listed expiry ({near.iloc[0].expiry.date()})"
    except Exception:  # noqa: BLE001
        pass
    r = s.data.closes([pos.underlying], days=70)[pos.underlying].pct_change().dropna()
    return S, float(r.tail(60).std() * np.sqrt(252)), T, "60-day realized vol (no usable option quote)"


def value_now(s, pos: OptionPosition) -> float:
    S, iv, T, _ = iv_now(s, pos)
    return float(bs_vec(S, pos.strike, T, iv, pos.kind)) * 100 * pos.contracts


def scenario_pnl(pos: OptionPosition, S: float, iv: float, T: float, move: np.ndarray, resp: dict,
                 days: int = WINDOW_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """P&L in dollars per scenario move (full repricing) and the IV used in each."""
    move = np.asarray(move, float)
    b = np.where(move < 0, resp["down"], resp["up"])
    iv1 = np.maximum(iv + b * move, 0.03)
    T1 = max(T - days / 365, 1 / 365)
    v0 = float(bs_vec(S, pos.strike, T, iv, pos.kind))
    v1 = bs_vec(S * (1 + move), pos.strike, T1, iv1, pos.kind)
    return (v1 - v0) * 100 * pos.contracts, iv1


def greek_split(pos: OptionPosition, S: float, iv: float, T: float, move: float, resp: dict, days: int = WINDOW_DAYS) -> dict:
    """The P&L at one scenario, by Greek (today's Greeks; vanna and volga by finite differences), and the remainder vs
    full repricing (what the Taylor expansion misses on a big move)."""
    mult = 100 * pos.contracts
    dS = S * move
    dv = float(np.where(move < 0, resp["down"], resp["up"]) * move)
    dv = max(iv + dv, 0.03) - iv
    dt = min(days / 365, T - 1 / 365)
    g = bs_greeks(S, pos.strike, T, R_RATE, 0.0, iv, pos.kind)
    h, e = 0.01 * S, 0.01
    vega_f = lambda S_, s_: (bs_price(S_, pos.strike, T, R_RATE, 0.0, s_ + e, pos.kind) - bs_price(S_, pos.strike, T, R_RATE, 0.0, s_ - e, pos.kind)) / (2 * e)
    vanna = (vega_f(S + h, iv) - vega_f(S - h, iv)) / (2 * h)
    volga = (vega_f(S, iv + e) - vega_f(S, iv - e)) / (2 * e)
    vega = vega_f(S, iv)
    theta_year = (bs_price(S, pos.strike, T - 1e-4, R_RATE, 0.0, iv, pos.kind) - bs_price(S, pos.strike, T, R_RATE, 0.0, iv, pos.kind)) / 1e-4
    parts = {"delta": g.delta * dS, "gamma": 0.5 * g.gamma * dS ** 2, "vega": vega * dv, "vanna": vanna * dS * dv,
             "volga": 0.5 * volga * dv ** 2, "theta": theta_year * dt}
    parts = {k: float(v * mult) for k, v in parts.items()}
    full = float(scenario_pnl(pos, S, iv, T, np.array([move]), resp, days)[0][0])
    # exact split by full repricing, one input at a time (adds up to the full P&L on any size of move)
    S1, iv1, T1 = S * (1 + move), iv + dv, max(T - days / 365, 1 / 365)
    V = lambda S_, s_, T_: float(bs_vec(S_, pos.strike, T_, s_, pos.kind)) * mult
    effects = {"spot (delta, gamma and beyond)": V(S1, iv, T) - V(S, iv, T),
               "vol at the new spot (vega, vanna, volga and beyond)": V(S1, iv1, T) - V(S1, iv, T),
               "time (theta)": V(S1, iv1, T1) - V(S1, iv1, T)}
    return {"move": move, "iv_change": dv, "by_effect_usd": effects, "full_repricing_usd": full,
            "by_greek_usd": parts, "sum_of_greeks_usd": sum(parts.values()), "remainder_usd": full - sum(parts.values()),
            "greeks_note": "today's Greeks (a Taylor expansion) explain small moves; on big moves the remainder grows, so "
                           "by_effect (full repricing one input at a time) is the explanation to use"}
