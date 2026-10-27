"""Market state for an underlying (SPEC §4.1): spot, IV term structure, 25-delta skew,
realized vol, IV/RV, and the earnings implied move vs history.

Input is a normalized chain (see data/massive.option_chain_snapshot). If the chain has no
IV column (or NaNs), IV is backed out of mids here, so nothing downstream cares which.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from .pricer import bs_greeks, implied_vol, straddle_implied_move


def realized_vol(closes: pd.Series, window: int) -> float:
    r = np.log(closes).diff().dropna().tail(window)
    return float(r.std(ddof=1) * math.sqrt(252))


def fill_iv(chain: pd.DataFrame, S: float, today: date, r: float = 0.04, q: float = 0.0) -> pd.DataFrame:
    """One consistent IV per contract (CP-L F7, F16; the trainer's ruling Oct 16).

    - `iv_mid`: IV backed out of the quoted mid with BSM.
    - `iv`: the IV used for EVERYTHING: pricing, greeks, skew, term structure, scenario repricing.
      Mid-implied where a mid exists (config.PRICE_LEGS_WITH = "mid_iv"), so every number is consistent
      with what you'd actually pay; vendor IV only where no usable mid exists.
    - `iv_vendor`: Massive's IV, kept for display only. `iv_source`: 'vendor' if Massive supplied one,
      'computed' if not (flagged per contract).
    - Greeks are recomputed from `iv` for every contract (vendor greeks kept as `*_vendor`).
    """
    from .. import config
    c = chain.copy()
    c["T"] = (pd.to_datetime(c["expiry"]) - pd.Timestamp(today)).dt.days / 365
    c = c[c["T"] > 0].copy()
    if "iv" not in c:
        c["iv"] = np.nan
    has_mid = c["mid"].notna() & (c["mid"] > 0)
    c["iv_mid"] = np.nan
    c.loc[has_mid, "iv_mid"] = [implied_vol(m, S, k, t, r, q, kind)
                                for m, k, t, kind in c.loc[has_mid, ["mid", "strike", "T", "kind"]].itertuples(index=False)]
    vendor = c["iv"].notna() & (c["iv"] > 0)
    c["iv_vendor"] = c["iv"].where(vendor)
    c["iv_source"] = np.where(vendor, "vendor", np.where(c["iv_mid"].notna(), "computed", None))
    if config.PRICE_LEGS_WITH == "mid_iv":
        c["iv"] = c["iv_mid"].where(c["iv_mid"].notna(), c["iv_vendor"])
    else:
        c["iv"] = c["iv_vendor"].where(vendor, c["iv_mid"])
    c["iv_price"] = c["iv"]  # kept for callers that ask for the pricing IV explicitly
    for col in ("delta", "gamma", "theta", "vega"):
        c[f"{col}_vendor"] = c[col] if col in c else np.nan
    c = c.dropna(subset=["iv"]).copy()
    g = [bs_greeks(S, k, t, r, q, v, kind) for k, t, v, kind in c[["strike", "T", "iv", "kind"]].itertuples(index=False)]
    c["delta"] = [x.delta for x in g]
    c["gamma"] = [x.gamma for x in g]
    c["theta"] = [x.theta / 365 for x in g]   # per day
    c["vega"] = [x.vega / 100 for x in g]     # per vol point
    return c


def atm_iv(chain: pd.DataFrame, S: float) -> pd.DataFrame:
    """ATM IV per expiry: average of the call and put IVs at the strike nearest spot."""
    rows = []
    for exp, g in chain.groupby("expiry"):
        k = g.loc[(g["strike"] - S).abs().idxmin(), "strike"]
        at = g[g["strike"] == k]
        rows.append({"expiry": exp, "T": at["T"].iloc[0], "strike": k,
                     "iv": at["iv"].mean(), "iv_call": at.loc[at.kind == "call", "iv"].mean(),
                     "iv_put": at.loc[at.kind == "put", "iv"].mean()})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def skew_25d(chain: pd.DataFrame, S: float, expiry, r: float = 0.04, q: float = 0.0) -> float:
    """IV(25Δ put) − IV(25Δ call) for one expiry, by nearest-delta strikes. Positive = puts bid."""
    g = chain[chain["expiry"] == expiry].dropna(subset=["iv"])
    def nearest(kind, target):
        s = g[g["kind"] == kind]
        d = s.apply(lambda x: bs_greeks(S, x["strike"], x["T"], r, q, x["iv"], kind).delta, axis=1)
        return s.loc[(d - target).abs().idxmin(), "iv"] if len(s) else np.nan
    return float(nearest("put", -0.25) - nearest("call", 0.25))


def implied_move(chain: pd.DataFrame, S: float, expiry, r: float = 0.04, q: float = 0.0) -> float:
    """ATM straddle / spot for one expiry."""
    a = atm_iv(chain[chain["expiry"] == expiry], S).iloc[0]
    return straddle_implied_move(S, a.strike, a["T"], r, q, a.iv_call, a.iv_put)


def event_move_from_term(iv_pre: float, iv_post: float, T_post: float) -> float:
    """Event-day sd from an expiry before the event (iv_pre) and one after it (iv_post, T_post).

    Assume one diffusive vol σ_d for ordinary days, read off the pre-event expiry
    (σ_d ≈ iv_pre). The post-event expiry's total variance is σ_d²·T_post + event variance, so
    event_var = T_post·(iv_post² − iv_pre²). Expected absolute move ≈ 0.8 × this sd.
    """
    return math.sqrt(max(T_post * (iv_post ** 2 - iv_pre ** 2), 0.0))


def vol_after_event(iv_post: float, T_post: float, event_sd: float) -> float:
    """The IV an expiry should show once the event has passed (for the vol-crush scenario)."""
    return math.sqrt(max(iv_post ** 2 - event_sd ** 2 / T_post, 1e-6))


def summarize(chain: pd.DataFrame, closes: pd.Series, today: date, r: float = 0.04, q: float = 0.0,
              earnings_expiry=None, past_earnings_moves: list[float] | None = None) -> dict:
    """Everything market_state() returns to the agent, compact and rounded by the tool layer."""
    S = float(closes.iloc[-1])
    ch = chain if "iv_mid" in chain else fill_iv(chain, S, today, r, q)
    term = atm_iv(ch, S)
    rv20, rv60 = realized_vol(closes, 20), realized_vol(closes, 60)
    front = term.iloc[0]
    out = {
        "spot": S,
        "term": [{"expiry": str(pd.Timestamp(x.expiry).date()), "atm_iv": x.iv} for x in term.itertuples()],
        "rv20": rv20, "rv60": rv60,
        "iv_rv": front.iv / rv20 if rv20 > 0 else None,
        "skew_25d_front": skew_25d(ch, S, front.expiry, r, q),
        "term_slope": (term.iloc[-1].iv - front.iv) if len(term) > 1 else 0.0,
        "iv_source": ch["iv_source"].value_counts().to_dict(),
    }
    if earnings_expiry is not None:
        out["earnings_implied_move"] = implied_move(ch, S, earnings_expiry, r, q)
        if past_earnings_moves:
            h = np.abs(np.asarray(past_earnings_moves, float))
            out["earnings_hist_abs_moves"] = h.tolist()
            out["earnings_hist_mean_abs"] = float(h.mean())
            out["implied_vs_hist"] = out["earnings_implied_move"] / float(h.mean()) if h.mean() > 0 else None
    return out
