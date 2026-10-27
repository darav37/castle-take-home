"""Strategy candidates from a view (SPEC §4.1, §6, §6a).

Pipeline: library front-matter → filter by (direction, vol view) → set strikes/expiries
by rules → price on the real chain → metrics → playbook rules promote/demote → rank.

No model picks a strike. The library says *which* structures fit; the strike
and expiry vocabulary below says *where* the legs go.

Strike rules (leg 'strike' in front-matter):
  atm | target | half_target | atm_or_slightly_otm | otm_pct:<x> | itm_pct:<x> | delta:<d>
Expiry rules (leg 'expiry', default 'horizon'):
  horizon (first listed expiry on/after the view horizon) | front (first expiry after the
  event, or the nearest one) | back (next expiry ≥ 21 days after 'horizon')
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd
import yaml

from . import signals as signals_mod

from .pricer import Leg, MoveDistribution, Strategy, bs_greeks, bs_price
from .view import View

LIB_DIR = Path(__file__).resolve().parents[3] / "library"


# ---------------------------------------------------------------- library + playbook

def load_library(directory: Path = LIB_DIR / "strategies") -> list[dict]:
    """Front-matter of each strategy file (prose stays in the file for the explainer)."""
    out = []
    for f in sorted(Path(directory).glob("*.md")):
        txt = f.read_text(encoding="utf-8")
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", txt, re.S)
        if not m:
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        fm["_path"] = str(f)
        fm["_prose"] = m.group(2).strip()
        fm["_rules"] = playbook_section(fm["_prose"])
        for k in ("direction", "vol_view"):
            if isinstance(fm.get(k), str):
                fm[k] = [fm[k]]
        out.append(fm)
    return out


def playbook_section(prose: str) -> dict[int, str]:
    """The trainer's numbered 'Playbook rules (machine-usable)' → {n: rule text}, so an alert can carry his words."""
    m = re.search(r"^## Playbook rules.*?$(.*?)(?=^## |\Z)", prose, re.S | re.M)
    out, cur = {}, None
    for line in (m.group(1) if m else "").splitlines():
        h = re.match(r"^(\d+)\.\s+(.*)", line)
        if h:
            cur = int(h.group(1))
            out[cur] = h.group(2).strip()
        elif cur is not None and line.strip():
            out[cur] += " " + line.strip()
    return out


STAND_IN_DIR = LIB_DIR.parent / "tests" / "fixtures" / "strategies"


def load_library_with_stand_ins(directory: Path = LIB_DIR / "strategies", stand_ins: Path = STAND_IN_DIR) -> list[dict]:
    """The trainer's files win by id; a fixture fills any id he hasn't written yet, flagged stand_in (no prose)."""
    lib = load_library(directory)
    have = {e["id"] for e in lib}
    for e in load_library(stand_ins) if Path(stand_ins).exists() else []:
        if e["id"] not in have:
            e["stand_in"] = True
            lib.append(e)
    return lib


def load_playbook(path: Path = LIB_DIR / "playbook" / "playbook_rules.yaml") -> list[dict]:
    if not Path(path).exists():
        return []
    return [r for r in (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []) if r.get("status", "approved") == "approved"]


_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
        "in": lambda a, b: a in b}


def rule_fires(rule: dict, ctx: dict) -> bool:
    """A rule's 'when' is a list of {metric, op, value}; all must hold. Missing metric ⇒ no fire."""
    for c in rule.get("when", []):
        v = ctx.get(c["metric"])
        if v is None or not _OPS[c["op"]](v, c["value"]):
            return False
    return True


# ---------------------------------------------------------------- quoting

class Quoter(Protocol):
    def expiries(self) -> list[date]: ...
    def strikes(self, expiry: date) -> list[float]: ...
    def quote(self, kind: str, strike: float, expiry: date) -> tuple[float, float]: ...  # (iv, mid)


@dataclass
class FlatQuoter:
    """A synthetic surface for tests/offline demos: listed expiries, a strike grid, IV(T, K)."""
    S: float
    today: date
    expiry_list: list[date]
    iv_fn: Callable[[float, float], float]  # (T, K) -> iv
    step: float = 1.0
    r: float = 0.04
    q: float = 0.0

    def expiries(self):
        return self.expiry_list

    def strikes(self, expiry):
        lo, hi = self.S * 0.5, self.S * 1.5
        k0 = math.floor(lo / self.step) * self.step
        return [round(k0 + i * self.step, 6) for i in range(int((hi - k0) / self.step) + 1)]

    def quote(self, kind, strike, expiry):
        T = (expiry - self.today).days / 365
        iv = self.iv_fn(T, strike)
        return iv, bs_price(self.S, strike, T, self.r, self.q, iv, kind)


@dataclass
class ChainQuoter:
    """Quotes off a real chain (with IV filled by market_state.fill_iv)."""
    chain: pd.DataFrame

    def expiries(self):
        return sorted({pd.Timestamp(e).date() for e in self.chain["expiry"]})

    def strikes(self, expiry):
        return sorted(self.chain.loc[self.chain["expiry"] == pd.Timestamp(expiry), "strike"].unique())

    def quote(self, kind, strike, expiry):
        row = self.chain[(self.chain["expiry"] == pd.Timestamp(expiry)) & (self.chain["strike"] == strike) & (self.chain["kind"] == kind)]
        if row.empty:
            raise KeyError((kind, strike, expiry))
        col = "iv_price" if "iv_price" in row else "iv"  # mid-implied by default (config.PRICE_LEGS_WITH)
        return float(row[col].iloc[0]), float(row["mid"].iloc[0])


# ---------------------------------------------------------------- rules → legs

def pick_expiry(rule: str, view: View, expiries: list[date], event_date: date | None) -> date:
    after = [e for e in expiries if e >= view.horizon]
    base = after[0] if after else expiries[-1]
    if rule == "horizon":
        return base
    if rule == "front":
        if event_date:
            post = [e for e in expiries if e > event_date]
            if post:
                return post[0]
        return expiries[0]
    if rule == "back":
        later = [e for e in expiries if (e - base).days >= 21]
        return later[0] if later else expiries[-1]
    raise ValueError(f"unknown expiry rule {rule}")


def target_strike(rule: str, kind: str, view: View, S: float, T: float, iv_guess: float, r: float, q: float) -> float:
    """The ideal strike before snapping to a listed one."""
    move = (view.target_move_pct or 0.0) / 100
    sgn = 1 if kind == "call" else -1
    if rule == "atm":
        return S
    if rule == "target":
        return S * (1 + move)
    if rule == "half_target":
        return S * (1 + move / 2)
    if rule == "atm_or_slightly_otm":
        return S * (1 + sgn * 0.02)
    if rule.startswith("otm_pct:"):
        return S * (1 + sgn * float(rule.split(":")[1]) / 100)
    if rule.startswith("itm_pct:"):
        return S * (1 - sgn * float(rule.split(":")[1]) / 100)
    if rule.startswith("delta:"):
        d = float(rule.split(":")[1])  # e.g. 0.25 (abs delta)
        lo, hi = S * 0.3, S * 3
        f = lambda k: abs(bs_greeks(S, k, T, r, q, iv_guess, kind).delta) - d
        from scipy.optimize import brentq
        return brentq(f, lo, hi)
    raise ValueError(f"unknown strike rule {rule}")


def snap(k: float, strikes: list[float]) -> float:
    return min(strikes, key=lambda s: abs(s - k))


@dataclass
class Candidate:
    id: str
    strategy: Strategy
    metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    score: float = 0.0
    entry: dict = field(default_factory=dict)


def build_legs(entry: dict, view: View, S: float, today: date, quoter: Quoter, r: float, q: float,
               event_date: date | None) -> list[Leg]:
    legs = []
    exps = quoter.expiries()
    for spec in entry["legs"]:
        kind = spec["type"]
        side = 1 if spec["side"] == "long" else -1
        qty = side * float(spec.get("ratio", 1))
        if kind == "stock":
            legs.append(Leg("stock", qty, 0.0, 0.0, 0.0, premium=S))
            continue
        exp = pick_expiry(spec.get("expiry", "horizon"), view, exps, event_date)
        T = max((exp - today).days, 1) / 365
        atm_iv, _ = quoter.quote(kind, snap(S, quoter.strikes(exp)), exp)
        k = snap(target_strike(spec["strike"], kind, view, S, T, atm_iv, r, q), quoter.strikes(exp))
        iv, mid = quoter.quote(kind, k, exp)
        legs.append(Leg(kind, qty, k, T, iv, premium=mid))
    # a spread whose two strikes snapped together is no spread: drop it
    return legs


def _degenerate(legs: list[Leg]) -> bool:
    opts = [(l.kind, l.strike, l.T) for l in legs if l.kind != "stock"]
    return len(opts) != len(set(opts)) and len(legs) > 1


# ---------------------------------------------------------------- shared metrics

def leg_rows(legs: list[Leg], S: float, r: float, q: float) -> list[dict]:
    """Legs as shown to the expert, each with its own delta (per share, signed for calls/puts, before qty)."""
    return [{"kind": l.kind, "qty": l.qty, "strike": l.strike, "days": round(l.T * 365), "iv": l.iv, "mid": l.premium,
             "delta": (bs_greeks(S, l.strike, l.T, r, q, max(l.iv, 1e-6), l.kind).delta if l.kind != "stock" else 1.0)}
            for l in legs]


def price_custom(view: View, S: float, today: date, quoter: Quoter, realized_vol: float, specs: list[dict],
                 r: float = 0.04, q: float = 0.0, name: str = "custom") -> dict:
    """Price ANY structure the expert asks for: each leg
    by strike or by |delta|, expiry by days (default: the view horizon). Same metrics as the rule-built candidates."""
    legs = []
    exps = quoter.expiries()
    for sp in specs:
        kind = sp["kind"]
        qty = float(sp.get("qty", 1))
        if kind == "stock":
            legs.append(Leg("stock", qty, 0.0, 0.0, 0.0, premium=S))
            continue
        target_day = today + __import__("datetime").timedelta(days=int(sp["days"])) if sp.get("days") else view.horizon
        exp = next((e for e in exps if e >= target_day), exps[-1])
        T = max((exp - today).days, 1) / 365
        strikes = quoter.strikes(exp)
        if sp.get("delta") is not None:
            atm_iv, _ = quoter.quote(kind, snap(S, strikes), exp)
            k = snap(target_strike(f"delta:{abs(float(sp['delta']))}", kind, view, S, T, atm_iv, r, q), strikes)
        else:
            k = snap(float(sp["strike"]), strikes)
        iv, mid = quoter.quote(kind, k, exp)
        legs.append(Leg(kind, qty, k, T, iv, premium=mid))
    st = Strategy(name, legs, S, r, q, meta={"custom": True})
    dist = view.distribution(today, realized_vol)
    mkt = market_distribution(view, S, today, quoter, r, q)
    dir_only = MoveDistribution(dist.mean_move, mkt.sd, dist.T)
    mg, ml = st.max_gain_loss()
    ev, evm, evd = st.expected_pnl(dist), st.expected_pnl(mkt), st.expected_pnl(dir_only)
    return {"id": name, "custom": True, "cost": st.cost(), "max_gain": mg, "max_loss": ml, "breakevens": st.breakevens(),
            "expected_pnl": ev, "prob_profit": st.prob_profit(dist), "expected_pnl_market": evm,
            "prob_profit_market": st.prob_profit(mkt), "edge_direction": evd - evm, "edge_vol": ev - evd,
            "vol_effect": "vol works for you" if ev - evd >= 0 else "you pay for vol", **st.greeks(),
            "legs": leg_rows(legs, S, r, q)}


# ---------------------------------------------------------------- main entry

def market_distribution(view: View, S: float, today: date, quoter: Quoter, r: float, q: float) -> MoveDistribution:
    """The market's distribution of spot at the view horizon: lognormal, centred on the forward, sd from the
    ATM implied vol of the first expiry on/after the horizon. Under it, a fairly priced option has E[P&L] ≈ 0."""
    T = view.T(today)
    exp = pick_expiry("horizon", view, quoter.expiries(), None)
    k = snap(S, quoter.strikes(exp))
    iv = 0.5 * (quoter.quote("call", k, exp)[0] + quoter.quote("put", k, exp)[0])
    return MoveDistribution(math.exp((r - q) * T) - 1, iv * math.sqrt(T), T)


def propose(view: View, S: float, today: date, quoter: Quoter, realized_vol: float, library: list[dict],
            playbook: list[dict] | None = None, r: float = 0.04, q: float = 0.0, event_date: date | None = None,
            iv_after_event: dict | None = None, implied_move: float | None = None, iv_rv: float | None = None,
            n: int = 3, include: list[str] | None = None, signals: dict | None = None,
            market_ctx: dict | None = None, unlocked: set | None = None) -> dict:
    """Candidates for a view, with metrics, playbook notes, and the dropped list with reasons.
    include: structures the user asked for by name; they bypass the vol-view filter (never the direction or
    risk constraints) and are always shown, with a note."""
    include = set(include or [])
    dist = view.distribution(today, realized_vol)
    mkt = market_distribution(view, S, today, quoter, r, q)
    dir_only = MoveDistribution(dist.mean_move, mkt.sd, dist.T)   # your direction, the market's vol
    kept, dropped = [], []
    for e in library:
        if view.direction not in e.get("direction", []):
            continue
        # gates unlock from the member's track record (profiles.sophistication, DECISIONS #120), replacing "asked by
        # name" (#104); a locked structure the member names is still shown, flagged as locked
        if e.get("gate") and e["gate"] not in (unlocked or set()) and e["id"] not in include:
            dropped.append({"id": e["id"], "reason": f"gated ({e['gate']}): unlocks from the member's track record"})
            continue
        if view.vol_view not in e.get("vol_view", ["none"]) and e["id"] not in include:
            dropped.append({"id": e["id"], "reason": f"vol view '{view.vol_view}' not in {e.get('vol_view')}"})
            continue
        if view.constraints.get("defined_risk_only") and e.get("risk") != "defined":
            dropped.append({"id": e["id"], "reason": "undefined risk, but defined_risk_only"})
            continue
        legs = build_legs(e, view, S, today, quoter, r, q, event_date)
        if _degenerate(legs):
            dropped.append({"id": e["id"], "reason": "strikes collapsed onto each other at this target"})
            continue
        st = Strategy(e.get("name", e["id"]), legs, S, r, q, meta={"library": e["id"]})
        cost = st.cost()
        cap = view.constraints.get("max_premium_usd")
        if cap is not None and cost > cap:
            dropped.append({"id": e["id"], "reason": f"costs {cost:.0f} > max premium {cap}"})
            continue
        mg, ml = st.max_gain_loss()
        ev = st.expected_pnl(dist)
        m = {
            "cost": cost, "max_gain": mg, "max_loss": ml, "breakevens": st.breakevens(),
            "expected_pnl": ev, "prob_profit": st.prob_profit(dist),
            # the same trade under the market's own (implied) distribution, and where your edge comes from
            #: edge_direction = your move at the market's vol; edge_vol = what your vol view adds/costs
            "expected_pnl_market": (evm := st.expected_pnl(mkt)), "prob_profit_market": st.prob_profit(mkt),
            "edge_direction": (evd := st.expected_pnl(dir_only)) - evm, "edge_vol": ev - evd,
            "vol_effect": "vol works for you" if ev - evd >= 0 else "you pay for vol",   # sign in words, from code
            **st.greeks(),
            "legs": leg_rows(legs, S, r, q),
        }
        if iv_after_event:
            # reprice the day after the event at the post-event vols, spot unchanged
            crushed = st.with_iv_crush(iv_after_event)
            m["pnl_vol_crush_flat_spot"] = crushed.pnl(S, t_elapsed=1 / 365)
        risk = abs(ml) if math.isfinite(ml) and ml != 0 else abs(cost) or 1.0
        c = Candidate(e["id"], st, m, score=ev / risk)
        c.entry = e
        kept.append(c)

    # playbook rules promote/demote with a stated reason
    view_move = abs(view.target_move_pct or 0) / 100
    ctx = {"view_move": view_move, "implied_move": implied_move, "iv_rv": iv_rv,
           "view_over_implied": view_move / implied_move if implied_move else None,
           "days": (view.horizon - today).days, "has_event": event_date is not None}
    sig = signals or {}
    # The trainer's history signals are playbook metrics too (z-scores; None when history is short ⇒ no fire)
    for k in ("atm_iv_z", "skew_rr_z", "term_z", "call_skew_pct", "spot_vol_corr"):
        ctx[k] = sig.get(k)
    actx = {**ctx, **(market_ctx or {}), "spot": S}
    for c in kept:
        f = signals_mod.trade_facts(c.strategy, c.metrics, implied_move, c.entry)
        al = signals_mod.alerts(c.entry, f, sig, actx)
        for a in al:   # The trainer's own rule text, from the strategy file (the explainer words it from this)
            a["darin_rule"] = c.entry.get("_rules", {}).get(int(a["rule"].split("#")[1]))
        c.metrics["alerts"] = al
        f.pop("_legs", None)
        c.metrics["facts"] = f
        if c.entry.get("stand_in"):
            c.notes.append("stand-in structure: the trainer hasn't written this strategy's file yet")
    for rule in playbook or []:
        if rule_fires(rule, ctx):
            for c in kept:
                if c.id in rule.get("demote", []):
                    c.score -= 1.0
                    c.notes.append(f"demoted by {rule['id']}: {rule.get('say', '')}")
                if c.id in rule.get("promote", []):
                    c.score += 1.0
                    c.notes.append(f"promoted by {rule['id']}: {rule.get('say', '')}")

    for c in kept:
        if c.metrics["expected_pnl"] < 0:
            c.notes.append("negative E[P&L] under your own view")
        if c.id in include:
            c.notes.append("asked for by name")
    kept.sort(key=lambda c: (c.id not in include, -c.score))  # named structures first
    teaching = None
    longs = [c for c in kept if len(c.strategy.legs) == 1 and c.strategy.legs[0].qty > 0]
    if longs and all(c.metrics["expected_pnl"] < 0 for c in longs):
        teaching = "long_premium_loses_under_view"
    return {"view": view.to_dict(), "distribution": {"mean_move": dist.mean_move, "sd": dist.sd, "T": dist.T},
            "market_distribution": {"mean_move": mkt.mean_move, "sd": mkt.sd, "T": mkt.T},
            "context": ctx, "candidates": kept[:n], "also_ran": kept[n:], "dropped": dropped, "teaching": teaching}


def diff_proposals(old: dict, new: dict) -> dict:
    """Which strategies dropped out / came in when the view changed (SPEC §4.1 update loop)."""
    a = [c.id for c in old["candidates"]]
    b = [c.id for c in new["candidates"]]
    reasons = {d["id"]: d["reason"] for d in new["dropped"]}
    return {"out": [{"id": i, "reason": reasons.get(i, "outranked")} for i in a if i not in b],
            "in": [i for i in b if i not in a], "kept": [i for i in b if i in a]}
