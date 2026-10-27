"""Black-Scholes-Merton pricer, greeks, implied vol, and multi-leg strategy metrics (SPEC §4.1).

Conventions:
- T is in years (calendar days / 365). r and q are continuous annual rates.
- sigma is annual vol as a decimal (0.45 = 45%).
- A leg's qty is signed in contracts: +1 long, -1 short. One contract = MULT shares.
- Every leg is priced off its own IV (no single-vol shortcut), so skew shows up in spreads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

SQRT2 = math.sqrt(2.0)
INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:  # scalar math.erfc is ~50× faster than scipy's norm.cdf
    return 0.5 * math.erfc(-x / SQRT2)


def _npdf(x: float) -> float:
    return INV_SQRT2PI * math.exp(-0.5 * x * x)

MULT = 100
Kind = Literal["call", "put", "stock"]


# ---------------------------------------------------------------- single option

def _d1d2(S, K, T, r, q, sigma):
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vt
    return d1, d1 - vt


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: Kind) -> float:
    if kind == "stock":
        return S
    if T <= 0 or sigma <= 0:
        # at/after expiry (or zero vol): discounted intrinsic on the forward
        F = S * math.exp((r - q) * max(T, 0))
        disc = math.exp(-r * max(T, 0))
        return disc * (max(F - K, 0.0) if kind == "call" else max(K - F, 0.0))
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float  # per $1 in spot
    gamma: float  # per $1 in spot, per $1
    vega: float   # per 1.00 vol (divide by 100 for per vol point)
    theta: float  # per year (divide by 365 for per day)
    rho: float    # per 1.00 rate


def bs_greeks(S, K, T, r, q, sigma, kind: Kind) -> Greeks:
    if kind == "stock":
        return Greeks(1.0, 0.0, 0.0, 0.0, 0.0)
    if T <= 0 or sigma <= 0:
        itm = (S > K) if kind == "call" else (S < K)
        d = (1.0 if kind == "call" else -1.0) if itm else 0.0
        return Greeks(d, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    eq, er, sq = math.exp(-q * T), math.exp(-r * T), math.sqrt(T)
    pdf = _npdf(d1)
    gamma = eq * pdf / (S * sigma * sq)
    vega = S * eq * pdf * sq
    if kind == "call":
        delta = eq * _ncdf(d1)
        theta = -S * eq * pdf * sigma / (2 * sq) - r * K * er * _ncdf(d2) + q * S * eq * _ncdf(d1)
        rho = K * T * er * _ncdf(d2)
    else:
        delta = -eq * _ncdf(-d1)
        theta = -S * eq * pdf * sigma / (2 * sq) + r * K * er * _ncdf(-d2) - q * S * eq * _ncdf(-d1)
        rho = -K * T * er * _ncdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


def implied_vol(price: float, S, K, T, r, q, kind: Kind, lo: float = 1e-4, hi: float = 5.0) -> float:
    """Back out IV from a price. Returns NaN if the price is outside no-arbitrage bounds."""
    if T <= 0 or kind == "stock":
        return float("nan")
    f = lambda s: bs_price(S, K, T, r, q, s, kind) - price
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-8)


def straddle_implied_move(S, K, T, r, q, sigma_call, sigma_put) -> float:
    """ATM straddle / spot: the market's priced move to expiry (≈ 0.8·σ·√T)."""
    return (bs_price(S, K, T, r, q, sigma_call, "call") + bs_price(S, K, T, r, q, sigma_put, "put")) / S


# ---------------------------------------------------------------- strategies

@dataclass(frozen=True)
class Leg:
    kind: Kind
    qty: float               # signed contracts (stock legs: signed shares / MULT)
    strike: float = 0.0
    T: float = 0.0           # years to expiry at entry
    iv: float = 0.0
    premium: float | None = None  # quoted mid per share; if None, use model price

    def price(self, S, r, q, t_elapsed=0.0, iv_shift=0.0) -> float:
        return bs_price(S, self.strike, max(self.T - t_elapsed, 0.0), r, q, max(self.iv + iv_shift, 1e-6), self.kind)


@dataclass
class Strategy:
    name: str
    legs: list[Leg]
    S: float
    r: float = 0.04
    q: float = 0.0
    meta: dict = field(default_factory=dict)

    # cost convention: positive = you pay (debit), negative = you receive (credit)
    def cost(self) -> float:
        tot = 0.0
        for l in self.legs:
            px = l.premium if l.premium is not None else l.price(self.S, self.r, self.q)
            tot += l.qty * px * MULT
        return tot

    def value(self, S, t_elapsed=0.0, iv_shift=0.0) -> float:
        return sum(l.qty * l.price(S, self.r, self.q, t_elapsed, iv_shift) * MULT for l in self.legs)

    def pnl(self, S, t_elapsed=0.0, iv_shift=0.0) -> float:
        return self.value(S, t_elapsed, iv_shift) - self.cost()

    def greeks(self) -> dict:
        g = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for l in self.legs:
            x = bs_greeks(self.S, l.strike, l.T, self.r, self.q, l.iv, l.kind)
            g["delta"] += l.qty * x.delta * MULT
            g["gamma"] += l.qty * x.gamma * MULT
            g["vega"] += l.qty * x.vega * MULT / 100   # $ per vol point
            g["theta"] += l.qty * x.theta * MULT / 365  # $ per day
        return g

    @property
    def first_expiry(self) -> float:
        ts = [l.T for l in self.legs if l.kind != "stock"]
        return min(ts) if ts else 0.0

    def payoff_at(self, S_grid: np.ndarray, t_elapsed: float | None = None, iv_shift: float = 0.0) -> np.ndarray:
        """P&L across spot at t_elapsed (default: first expiry; later legs keep time value)."""
        t = self.first_expiry if t_elapsed is None else t_elapsed
        return np.array([self.pnl(s, t, iv_shift) for s in S_grid])

    def grid(self, n: int = 2001, width: float = 4.0) -> np.ndarray:
        # from ~0 (spot can't go negative, so the downside is always bounded) to (1+width)·S
        return np.linspace(self.S * 1e-4, self.S * (1 + width), n)

    def max_gain_loss(self) -> tuple[float, float]:
        """(max gain, max loss) at first expiry. ±inf when the payoff is unbounded (only possible upside)."""
        g = self.grid()
        p = self.payoff_at(g)
        hi_slope = p[-1] - p[-2]  # still sloping at 5× spot ⇒ unbounded as S→∞
        mg = math.inf if hi_slope > 1e-6 else float(p.max())
        ml = -math.inf if hi_slope < -1e-6 else float(p.min())
        return mg, ml

    def breakevens(self) -> list[float]:
        g = self.grid(n=8001)
        p = self.payoff_at(g)
        out = []
        for i in range(len(g) - 1):
            if p[i] == 0:
                out.append(float(g[i]))
            elif p[i] * p[i + 1] < 0:
                out.append(float(brentq(lambda s: self.pnl(s, self.first_expiry), g[i], g[i + 1])))
        return out

    def expected_pnl(self, dist: "MoveDistribution", t_elapsed: float | None = None, iv_shift: float = 0.0) -> float:
        """E[P&L] under the user's distribution of spot at t_elapsed (default: the view horizon)."""
        t = dist.T if t_elapsed is None else t_elapsed
        S_pts, w = dist.quadrature(self.S)
        return float(sum(wi * self.pnl(si, t, iv_shift) for si, wi in zip(S_pts, w)))

    def prob_profit(self, dist: "MoveDistribution", t_elapsed: float | None = None) -> float:
        t = dist.T if t_elapsed is None else t_elapsed
        S_pts, w = dist.quadrature(self.S)
        return float(sum(wi for si, wi in zip(S_pts, w) if self.pnl(si, t) > 0))

    def with_iv_crush(self, iv_after: dict[float, float]) -> "Strategy":
        """Copy with each leg's IV replaced by iv_after[leg.T] (post-event vol by expiry)."""
        legs = [replace(l, iv=iv_after.get(l.T, l.iv)) for l in self.legs]
        return Strategy(self.name + " (post-crush)", legs, self.S, self.r, self.q, dict(self.meta))


@dataclass(frozen=True)
class MoveDistribution:
    """The user's view as a lognormal on spot at horizon T.

    mean_move is the expected % move (0.08 = +8%); sd is the log-return stdev over the horizon.
    See options/view.py for how a view's conviction sets sd.
    """
    mean_move: float
    sd: float
    T: float

    def quadrature(self, S0: float, n: int = 801):
        # dense grid on the log return (±8 sd). Gauss-Hermite was tried and is ~0.5% off on
        # kinked payoffs; a plain grid converges cleanly through the strikes.
        x = np.linspace(-8, 8, n)
        w = norm.pdf(x)
        w = w / w.sum()
        mu = math.log(1 + self.mean_move) - 0.5 * self.sd ** 2  # so E[S_T]/S0 = 1 + mean_move
        return S0 * np.exp(mu + self.sd * x), w
