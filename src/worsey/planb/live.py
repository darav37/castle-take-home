"""Plan B in the chat (DECISIONS #179): the scenario engine behind scan_risks, analyze_event and compare_hedges.

What the research showed, and how it is used here:
- The map (#156, #163, #178): a book's event risk is its exposure to 13 named factors. A market-only view explains 37% of
  an unusual book's event moves; the factors explain 77%. Loadings are fitted on the last 504 sessions of each position.
- The forecaster is code (#180, #182): in the shipped chat the expert's own factor calls tied an untrained LLM, and the
  engine alone was best, so every number now comes from the engine. The LLM identifies the event, fills the schema,
  matches a Kalshi market and explains the result (the project notes rule 6).
- Ranges, not points (#175, #177): severity-scaled ranges from the real spread of past events of that severity
  (calibrated, #175), an end-of-day range and a tighter first-hours range (the first move is kept on average, but about
  1 in 4 big events gives back half of it by the close).
- Most headlines don't move books (#162, #171): the category's learnability is reported, so the expert can say so.

Everything comes from `pack.json` (built by scripts/planb_pack.py from the research caches) plus live closes.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import impact as IM

FACTORS = ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit", "size", "value", "asia_em", "healthcare",
           "agriculture", "defensive_vs_producers"]
# live proxies: the research used FRED yields for rates; live, Treasury ETFs stand in (sign flipped: price up = yield down)
PROXIES = {"market": [("SPY", 1)], "rates_2y": [("SHY", -1)], "rates_10y": [("IEF", -1)], "dollar": [("UUP", 1)],
           "oil": [("USO", 1)], "gold": [("GLD", 1)], "credit": [("HYG", 1), ("IEF", -1)], "size": [("IWM", 1), ("SPY", -1)],
           "value": [("VTV", 1), ("VUG", -1)], "asia_em": [("EEM", 1), ("SPY", -1)], "healthcare": [("XLV", 1)],
           "agriculture": [("DBA", 1)], "defensive_vs_producers": [("XLU", 1), ("XLE", -1)]}
FACTOR_WORDS = {"market": "US stocks", "rates_2y": "2-year yields", "rates_10y": "10-year yields", "dollar": "the dollar",
                "oil": "oil", "gold": "gold", "credit": "credit (high yield vs Treasuries)", "size": "small vs large caps",
                "value": "value vs growth", "asia_em": "emerging markets vs US", "healthcare": "healthcare",
                "agriculture": "farm commodities", "defensive_vs_producers": "utilities vs energy producers"}
CATEGORIES = ["central_bank_unscheduled", "conflict_war", "disaster", "elections_politics", "financial_stress",
              "health_pandemic", "regulatory_court", "sanctions", "supply_shock", "trade_tariffs"]
CATEGORY_WORDS = {"central_bank_unscheduled": "surprise central-bank move", "conflict_war": "war or military escalation",
                  "disaster": "natural disaster", "elections_politics": "election or political shock",
                  "financial_stress": "financial stress (bank failure, credit event)", "health_pandemic": "pandemic or health scare",
                  "regulatory_court": "regulation or court ruling", "sanctions": "sanctions", "supply_shock": "supply shock",
                  "trade_tariffs": "tariffs or trade war"}
# what history says about each category's broad-market effect within days (#162, #164); the expert must say it plainly
LEARNABLE = {"financial_stress": "learnable: past events moved broad markets consistently",
             "health_pandemic": "learnable: past events moved broad markets consistently",
             "central_bank_unscheduled": "learnable: past events moved rates and markets consistently",
             "trade_tariffs": "specific: moves the market, healthcare and credit a little; no consistent broad effect otherwise",
             "sanctions": "specific: moves utilities vs energy producers; little broad effect",
             "supply_shock": "specific: moves oil; little broad effect",
             "conflict_war": "weak: past wars and escalations had no consistent broad-market effect within days, except the biggest",
             "disaster": "weak: past disasters had no consistent broad-market effect within days",
             "elections_politics": "weak: mostly moves the country's own market",
             "regulatory_court": "weak: mostly moves the targeted sector"}
SEVERITIES = ["minor", "moderate", "major", "extreme"]
BUCKET_WORDS = {"strong_down": "a large fall", "down": "a moderate fall", "flat": "little change", "up": "a moderate rise",
                "strong_up": "a large rise"}
TAIL_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]   # loss thresholds as a share of the book, rounded in dollars
DIRECTION_VALUES = {"escalation": ["escalation", "de-escalation", "not_applicable"], "growth_shock": ["negative", "positive", "none"],
                    "inflation_shock": ["up", "down", "none"], "oil_supply": ["disruption", "increase", "none"],
                    "policy_stance": ["tighter", "looser", "none"], "risk_sentiment": ["risk_off", "risk_on", "neutral"]}
RELEASE_WORDS = {"cpi": "CPI (inflation) release", "fomc": "Fed decision (FOMC)", "jobs": "jobs report",
                 "gdp": "GDP release", "pce": "PCE inflation release"}
CATEGORY_KEYWORDS = {"central_bank_unscheduled": "Fed emergency cut", "conflict_war": "war invasion military strike ceasefire",
                     "disaster": "hurricane earthquake", "elections_politics": "election president",
                     "financial_stress": "bank default recession", "health_pandemic": "pandemic outbreak",
                     "regulatory_court": "Supreme Court ruling", "sanctions": "sanctions", "supply_shock": "oil OPEC production",
                     "trade_tariffs": "tariff trade"}
RELEASE_SERIES = {"fomc": "KXFEDDECISION", "cpi": "KXCPI", "gdp": "KXGDP", "jobs": "KXPAYROLLS"}
LOOKBACK = 504
REFS = ["SPY", "TLT", "GLD", "UUP", "USO", "HYG", "IEF"]   # hedge underlyings: their moves are needed for the hedge table
N_DRAWS = 4000


_CUTOFF: str | None = None   # evaluation only: pretend today is this date (no event on or after it is known)


def set_cutoff(date: str | None) -> None:
    """For out-of-sample tests of the shipped engine (scripts/planb_product_vs_raw.py): drop events on or after `date`
    and refit the event model and severity scales on what's left, so neither the model nor the evidence can see the
    events being scored."""
    global _CUTOFF
    _CUTOFF = date
    pack.cache_clear()


@lru_cache(maxsize=1)
def pack() -> dict:
    p = json.loads((Path(__file__).with_name("pack.json")).read_text(encoding="utf-8"))
    if _CUTOFF:
        p["events"] = [e for e in p["events"] if e["date"] < _CUTOFF]
        p["releases"] = [r for r in p["releases"] if r["date"] < _CUTOFF]
        U = np.array([e["U"] for e in p["events"]])
        F = np.array([features(e, p["model"]["categories"]) for e in p["events"]])
        B = IM.ridge(F, U, 2.0)
        P = F @ B
        sev0 = np.array([e.get("severity") or "moderate" for e in p["events"]])
        for sv in SEVERITIES:
            m = sev0 == sv
            x, y = P[m].ravel(), U[m].ravel()
            p["model"]["severity_scale"][sv] = float(np.clip((x @ y) / max(x @ x, 1e-9), 0, 6)) if m.sum() >= 20 else 1.0
        p["model"]["B"] = B.tolist()
    p["_U"] = np.array([e["U"] for e in p["events"]])
    p["_B"] = np.array(p["model"]["B"])
    sev = np.array([e.get("severity") or "moderate" for e in p["events"]])
    P = np.array([features(e, p["model"]["categories"]) for e in p["events"]]) @ p["_B"]
    P *= np.array([p["model"]["severity_scale"].get(s, 1.0) for s in sev])[:, None]
    p["_resid"] = {s: p["_U"][sev == s] - P[sev == s] for s in SEVERITIES}
    cat = np.array([e.get("category") for e in p["events"]])
    E = p["_U"] - P
    p["_resid_cat"] = {(c, s): E[(cat == c) & (sev == s)] for c in set(cat) for s in SEVERITIES}
    p["_resid_cat"].update({(c, "big"): E[(cat == c) & np.isin(sev, ["major", "extreme"])] for c in set(cat)})
    p["_sev"] = sev
    return p


def features(e: dict, cats: list[str]) -> np.ndarray:
    return np.concatenate([[1.0], IM.direction_features(e), [float(e.get("category") == c) for c in cats]])


def _bucket(z: float) -> str:
    p = pack()
    return p["buckets"][int(np.searchsorted(p["bucket_cuts_sd"], z))]


def _zsd(f: str) -> float:
    """Typical size of an event-window move for factor f (pre-2016 spread), so moves read as buckets."""
    p = pack()
    j = FACTORS.index(f)
    return float(np.std(p["_U"][:, j])) or 1.0


# ---------------------------------------------------------------- the map: factor loadings of the book


def factor_returns(r: pd.DataFrame) -> pd.DataFrame:
    """Daily factor proxies in units of their 2011-2020 daily sd (the units the event moves are in)."""
    p = pack()
    out = {}
    for f, legs in PROXIES.items():
        if all(t in r.columns for t, _ in legs):
            out[f] = sum(s * r[t] for t, s in legs) / p["proxy_sd"][f]
    return pd.DataFrame(out)


def loadings(data, tickers: list[str]) -> dict:
    """Each ticker's return per 1 sd of each factor (OLS on the last 504 sessions), its own daily noise, and how much
    of its daily moves the factors explain."""
    want = list(dict.fromkeys(list(tickers) + [t for legs in PROXIES.values() for t, _ in legs]))
    closes = data.closes(want, days=LOOKBACK + 5)
    r = closes.pct_change().iloc[1:]
    F = factor_returns(r)
    used = [f for f in FACTORS if f in F.columns]
    betas, resid_sd, r2 = {}, {}, {}
    for t in dict.fromkeys(tickers):
        if t not in r.columns:
            continue
        d = pd.concat([r[t], F[used]], axis=1).dropna()
        if len(d) < 120:
            continue
        A = np.column_stack([np.ones(len(d)), d[used].to_numpy()])
        y = d[t].to_numpy()
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        e = y - A @ c
        betas[t] = pd.Series(c[1:], index=used).reindex(FACTORS).fillna(0.0)
        resid_sd[t] = float(e.std())
        r2[t] = float(1 - e.var() / y.var()) if y.var() > 0 else 0.0
    return {"betas": pd.DataFrame(betas).T.reindex(columns=FACTORS).fillna(0.0) if betas else pd.DataFrame(columns=FACTORS),
            "resid_sd": pd.Series(resid_sd), "r2": pd.Series(r2), "factors_used": used, "returns": r}


def book_coverage(L: dict, vals: pd.Series) -> dict:
    """How much of the BOOK's daily moves the factors explain (the "is your book covered?" check, #178)."""
    r = L["returns"]
    held = [t for t in vals.index if t in r.columns and t in L["betas"].index]
    if not held:
        return {"share_explained": None, "daily_noise_usd": 0.0, "book_daily_sd_usd": 0.0}
    w = vals[held] / vals[held].sum()
    br = (r[held].fillna(0) * w).sum(axis=1)
    fit = L["betas"].loc[held].T @ w
    F = factor_returns(r)[[f for f in FACTORS if f in factor_returns(r).columns]]
    pred = (F * fit.reindex(F.columns).to_numpy()).sum(axis=1)
    e = br - pred
    e = e - e.mean()
    share = float(1 - e.var() / br.var()) if br.var() > 0 else None
    weak = [t for t in held if L["r2"].get(t, 1) < 0.35]
    return {"share_explained": share, "daily_noise_usd": float(e.std() * vals[held].sum()),
            "book_daily_sd_usd": float(br.std() * vals[held].sum()), "poorly_covered_positions": weak,
            "verdict": ("well covered" if share is not None and share >= 0.8 else
                        "partly covered: some of this book's risk lives outside the 13 factors" if share is not None and share >= 0.5 else
                        "poorly covered: most of this book's risk lives outside the 13 factors, so treat the numbers as partial")}


def _dollar_betas(L: dict, vals: pd.Series) -> np.ndarray:
    held = [t for t in vals.index if t in L["betas"].index]
    return (L["betas"].loc[held].T @ vals[held]).reindex(FACTORS).fillna(0.0).to_numpy()


# ---------------------------------------------------------------- evidence for the expert


SIGNED = ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit"]


@lru_cache(maxsize=1)
def real_event() -> dict:
    """The real-event part of Plan C's gate model, frozen for the chat (scripts/planc_real_event_pack.py, #213)."""
    return json.loads(Path(__file__).with_name("real_event.json").read_text(encoding="utf-8"))


def _spec_ok(spec: dict) -> dict:
    from ..agent.session import ToolError
    if spec.get("category") not in CATEGORIES:
        raise ToolError(f"category must be one of {CATEGORIES}")
    if spec.get("severity") not in SEVERITIES:
        raise ToolError(f"severity must be one of {SEVERITIES}")
    d = dict(spec.get("direction") or {})
    for f, vals in DIRECTION_VALUES.items():
        if d.get(f) not in vals:
            d[f] = vals[-1]
    out = dict(spec, direction=d)
    fs = spec.get("factor_signs") or {}
    bad = {k: v for k, v in fs.items() if k not in SIGNED or v not in ("+", "-", "0")}
    if bad:
        raise ToolError(f"factor_signs: keys from {SIGNED}, values '+', '-' or '0'; got {bad}")
    out["factor_signs"] = {k: {"+": 1, "-": -1, "0": 0}[v] for k, v in fs.items()}
    fz = spec.get("factor_sizes") or {}
    bad = {k: v for k, v in fz.items() if k not in SIGNED or v not in ("small", "medium", "large")}
    if bad:
        raise ToolError(f"factor_sizes: keys from {SIGNED}, values small, medium or large; got {bad}")
    out["factor_sizes"] = dict(fz)
    out["market_moving"] = spec.get("market_moving", True) is not False
    out.setdefault("anticipation", "surprise")
    out["targeted_sectors"] = list(spec.get("targeted_sectors") or [])
    return out


def estimate(spec: dict) -> np.ndarray:
    p = pack()
    return features(spec, p["model"]["categories"]) @ p["_B"] * p["model"]["severity_scale"].get(spec["severity"], 1.0)


DRAW_POOL = os.environ.get("WORSEY_DRAW_POOL", "severity")   # "severity" (shipped) or "category" (#239, on test)


def _residuals(sev: str) -> np.ndarray:
    p = pack()
    R = p["_resid"].get(sev)
    return R if R is not None and len(R) >= 20 else p["_resid"]["major"]


def resid_pool(spec: dict) -> tuple[np.ndarray, str]:
    """The past events whose leftover factor vectors give a scenario its shape and co-movement (#239). With the
    category pool: the event's own category and severity if there are at least 20, else its category's major + extreme
    events (for a major or extreme event), else the severity pool, else "major". Returns (vectors, pool name)."""
    p = pack()
    sev, cat = spec.get("severity") or "moderate", spec.get("category")
    if DRAW_POOL == "category" and cat:
        R = p["_resid_cat"].get((cat, sev))
        if R is not None and len(R) >= 20:
            return R, f"{cat}, {sev}"
        if sev in ("major", "extreme"):
            R = p["_resid_cat"].get((cat, "big"))
            if R is not None and len(R) >= 20:
                return R, f"{cat}, major or extreme"
    R = p["_resid"].get(sev)
    if R is not None and len(R) >= 20:
        return R, sev
    return p["_resid"]["major"], "major"


def _p(m: dict) -> float | None:
    from ..scenario.probability import market_prob
    try:
        return float(market_prob(m.get("bid"), m.get("ask"), m.get("last"))["p"])
    except (ValueError, TypeError):
        return None


def _markets(data, query: str, k: int = 3) -> list[dict]:
    """Kalshi markets that might price this event (keyword retrieval; the expert judges the fit)."""
    from ..scenario import matching
    try:
        universe = data.event_markets()
        cands = matching.keyword_candidates(query, universe, n=k, text_keys=("title", "subtitle", "rules_primary"))
    except Exception:  # noqa: BLE001 — markets are optional context; the scenario stands without them
        return []
    out = []
    for c in cands:
        p = _p(c)
        if p is not None:
            out.append({"market_id": c["id"], "title": c.get("title"), "p": p, "closes": c.get("close_date"),
                        "keyword_score": c.get("_score")})
    return out


def _next_release(universe: list[dict], typ: str, today: pd.Timestamp) -> dict | None:
    """The next Kalshi-listed release of this type: its date and, for the Fed, the most likely outcome."""
    ser = RELEASE_SERIES.get(typ)
    ms = [m for m in universe if m.get("series_ticker") == ser and m.get("close_date") and pd.Timestamp(m["close_date"]) >= today]
    if not ms:
        return None
    first = min(pd.Timestamp(m["close_date"]) for m in ms)
    nxt = [m for m in ms if pd.Timestamp(m["close_date"]) == first]
    priced = [(p, m) for m in nxt if (p := _p(m)) is not None]
    out = {"next_on_kalshi": str(first.date()), "outcomes_listed": len(nxt)}
    if typ == "fomc" and priced:
        p, m = max(priced, key=lambda x: x[0])
        out["most_likely"] = {"outcome": m.get("subtitle") or m.get("title"), "p": p, "market_id": m["id"]}
    return out


def evidence(s, spec: dict) -> dict:
    """Code's evidence for one event, for the expert to judge: the model's estimate per factor (as a bucket), the
    severity's base rates, the 5 closest past events and what each would do to THIS book today, the regime, how well the
    factors cover the book, and Kalshi markets that might price it."""
    s.require_portfolio()
    p = pack()
    spec = _spec_ok(spec)
    vals = s.priced["value"]
    L = loadings(s.data, list(vals.index))
    db = _dollar_betas(L, vals)
    est = estimate(spec)
    pool = p["events"]
    sims = np.array([IM.direction_similarity(spec, e) for e in pool])
    top = np.argsort(-sims)[:5]
    analogs = [{"date": pool[i]["date"], "what": pool[i]["title"], "category": pool[i]["category"], "severity": pool[i]["severity"],
                "similarity": float(sims[i]),
                "factor_moves": {f: _bucket(p["_U"][i, j] / _zsd(f)) for j, f in enumerate(FACTORS) if f in p["llm_factors"]},
                "factor_moves_plain": plain_moves(p["_U"][i]),
                "your_book_if_it_happened_today": float(db @ p["_U"][i])} for i in top]
    same = (p["_sev"] == spec["severity"]) & np.array([e["category"] == spec["category"] for e in pool])
    yrs = [int(e["date"][:4]) for e in pool]
    recent = np.array([2016 <= y <= 2025 for y in yrs])
    big = np.isin(p["_sev"], ["major", "extreme"]) & np.array([e["category"] == spec["category"] for e in pool])
    s.__dict__["planb_spec"] = spec
    s.__dict__["planb_analogs"] = analogs
    return {"event": {k: spec.get(k) for k in ("title", "category", "severity", "region", "direction", "targeted_sectors")},
            "history_says": LEARNABLE[spec["category"]],
            "code_estimate": {FACTOR_WORDS[f]: BUCKET_WORDS[_bucket(est[j] / _zsd(f))] for j, f in enumerate(FACTORS)
                              if f in p["llm_factors"]},
            "code_estimate_book_usd": float(db @ est),
            "closest_past_events": analogs,
            "past_events_like_this": {"same_category_and_severity": int(same.sum()),
                                      "big_ones_per_year_2016_2025": round(float((big & recent).sum()) / 10, 1)},
            "regime_now": {"as_of": p["regime"]["as_of"], **p["regime"]["labels"]},
            "coverage": book_coverage(L, vals),
            "markets": _markets(s.data, spec.get("title") or CATEGORY_KEYWORDS[spec["category"]]),
            "next": "check whether a market in `markets` really is this event (variable, threshold, deadline, resolution "
                    "source), then call price_scenario; code prices it, you never adjust the numbers"}


# ---------------------------------------------------------------- option positions (DECISIONS #209)


def _options(s) -> list:
    return s.__dict__.get("options") or []


def option_setup(s, L: dict) -> list[dict]:
    """Per option: today's spot, IV, time, value, its underlying's factor loadings and own noise, and its spot-vol
    response. L must include the underlyings."""
    from ..options import scenario as OS
    out = []
    for o in _options(s):
        S, iv, T, src = OS.iv_now(s, o)
        u = o.underlying
        out.append({"pos": o, "S": S, "iv": iv, "T": T, "iv_source": src, "resp": OS.spot_vol_response(s, u),
                    "beta": L["betas"].loc[u].to_numpy() if u in L["betas"].index else np.zeros(len(FACTORS)),
                    "own": float(L["resid_sd"].get(u, 0.0)), "value": float(OS.value_now(s, o))})
    return out


def option_pnl(o: dict, move) -> np.ndarray:
    from ..options import scenario as OS
    return OS.scenario_pnl(o["pos"], o["S"], o["iv"], o["T"], np.asarray(move, float), o["resp"])[0]


def option_row(o: dict, middle_move: float, draws_pnl: np.ndarray | None, bad_move: float | None) -> dict:
    from ..options import scenario as OS
    mid = float(option_pnl(o, [middle_move])[0])
    row = {"position": o["pos"].label, "kind": "option", "value_now_usd": o["value"], "underlying_move_middle": middle_move,
           "pnl_middle_usd": mid, "iv_now": o["iv"], "iv_source": o["iv_source"],
           "vol_response": {k: o["resp"][k] for k in ("down", "up", "source")},
           "max_loss_usd": -o["value"] if o["pos"].contracts > 0 else None}
    if draws_pnl is not None:
        row.update({"bad_case_1_in_20_usd": float(np.quantile(draws_pnl, 0.05)), "good_case_1_in_20_usd": float(np.quantile(draws_pnl, 0.95))})
    if bad_move is not None:
        row["greeks_at_the_bad_case"] = OS.greek_split(o["pos"], o["S"], o["iv"], o["T"], float(bad_move), o["resp"])
    return row


# ---------------------------------------------------------------- plain units and black-swan stress (DECISIONS #208)

# what one unit of each factor means to a person: % moves of the proxy (relative returns for the spreads), basis points
# for yields (the bond proxies' duration). The factor scale is the proxy's daily sd (pack proxy_sd), as the book mapping
# uses it, so a plain shock maps through each holding's loadings exactly.
FACTOR_UNITS = {"market": "% US stocks", "rates_2y": "bp 2-year yield", "rates_10y": "bp 10-year yield", "dollar": "% dollar",
                "oil": "% oil (USO)", "gold": "% gold", "credit": "% high-yield bonds minus Treasuries",
                "size": "% small caps minus large", "value": "% value minus growth", "asia_em": "% emerging markets minus US",
                "healthcare": "% healthcare stocks", "agriculture": "% farm commodities",
                "defensive_vs_producers": "% utilities minus energy producers"}
DURATION = {"rates_2y": 1.9, "rates_10y": 7.5}


def to_factor(f: str, x: float) -> float:
    sd = pack()["proxy_sd"][f]
    return (x / 1e4 * DURATION[f]) / sd if f in DURATION else (x / 100) / sd


def to_plain(f: str, u: float) -> float:
    sd = pack()["proxy_sd"][f]
    return u * sd / DURATION[f] * 1e4 if f in DURATION else u * sd * 100


def plain_moves(u: np.ndarray) -> dict:
    return {f: f"{to_plain(f, u[j]):+.1f}{'bp' if f in DURATION else '%'}" for j, f in enumerate(FACTORS)}


def stress_scenario(s, title: str, shocks: dict, why: dict | None = None) -> dict:
    """A black-swan scenario: the analyst writes an explicit factor shock in plain units (from its own understanding of
    the event; anchoring on smaller past events made shocks worse in the gut check, #208), and code maps it through every
    holding. The shock is an assumption shown to the user, not a forecast.
    Factors the analyst didn't set get their typical co-move given the ones set (the conditional mean over past big event
    windows), marked as filled in. Shocks beyond the biggest move in any past event window are flagged."""
    from ..agent.session import ToolError
    s.require_portfolio()
    p = pack()
    bad = [k for k in shocks if k not in FACTORS]
    if bad or not shocks:
        raise ToolError(f"shocks must use factor names {FACTORS}; got {bad or 'nothing'}")
    U = p["_U"][np.isin(p["_sev"], ["major", "extreme"])]
    C = np.cov(U.T)
    set_i = [FACTORS.index(k) for k in shocks]
    rest = [j for j in range(len(FACTORS)) if j not in set_i]
    x = np.array([to_factor(k, float(v)) for k, v in shocks.items()])
    f = np.zeros(len(FACTORS))
    f[set_i] = x
    f[rest] = C[np.ix_(rest, set_i)] @ np.linalg.solve(C[np.ix_(set_i, set_i)], x)
    lo, hi = p["_U"].min(0), p["_U"].max(0)
    unit = lambda j: "bp" if FACTORS[j] in DURATION else "%"
    flags = [f"{FACTOR_WORDS[FACTORS[j]]} {to_plain(FACTORS[j], f[j]):+.1f}{unit(j)} is beyond any move in 3,876 past event "
             f"windows ({to_plain(FACTORS[j], lo[j]):+.1f} to {to_plain(FACTORS[j], hi[j]):+.1f}{unit(j)})"
             for j in range(len(FACTORS)) if f[j] < lo[j] or f[j] > hi[j]]
    vals = s.priced["value"]
    tick = list(dict.fromkeys(list(vals.index) + REFS + [o.underlying for o in _options(s)]))
    L = loadings(s.data, tick)
    held = [t for t in vals.index if t in L["betas"].index]
    book = float(vals.sum())
    pos = []
    opt_rows = []
    for o in option_setup(s, L):   # options fully repriced at each scale of the shock (#209)
        mv = float(o["beta"] @ f)
        p05, p1, p15 = option_pnl(o, [0.5 * mv, mv, 1.5 * mv])
        band = 1.645 * o["own"] * np.sqrt(1.5)
        lo_, hi_ = option_pnl(o, [mv - band, mv + band])
        book += o["value"]
        pos.append({"ticker": o["pos"].label, "value": o["value"], "move": mv, "pnl_usd": float(p1),
                    "pnl_half_shock_usd": float(p05), "pnl_1_5x_shock_usd": float(p15),
                    "plus_or_minus_its_own_noise_usd": float(max(abs(lo_ - p1), abs(hi_ - p1)))})
        opt_rows.append(option_row(o, mv, None, mv))
    for t in held:
        mv = float(L["betas"].loc[t].to_numpy() @ f)
        band = 1.645 * float(L["resid_sd"].get(t, 0.0)) * np.sqrt(1.5)
        pos.append({"ticker": t, "value": float(vals[t]), "move": mv, "pnl_usd": float(vals[t]) * mv,
                    "pnl_half_shock_usd": float(vals[t]) * mv * 0.5, "pnl_1_5x_shock_usd": float(vals[t]) * mv * 1.5,
                    "plus_or_minus_its_own_noise_usd": abs(float(vals[t])) * band})
    pos.sort(key=lambda r: r["pnl_usd"])
    tot = sum(r["pnl_usd"] for r in pos)
    cov = book_coverage(L, vals)
    noise = 1.645 * cov["daily_noise_usd"] * np.sqrt(1.5)
    pm = plain_moves(f)
    shown = {}
    for j, f_ in enumerate(FACTORS):
        if f_ in shocks or abs(to_plain(f_, f[j])) >= (1.0 if f_ in DURATION else 0.1):
            shown[FACTOR_WORDS[f_]] = {"move": pm[f_], "set_by": "analyst" if f_ in shocks else "filled in (typical co-move)"}
            if (why or {}).get(f_):
                shown[FACTOR_WORDS[f_]]["why"] = why[f_]
    out = {"scenario": title, "kind": "stress (black swan)", "book_value": book, "shock": shown, "beyond_history": flags,
           "book_usd": {"if_the_shock_is_half": sum(r["pnl_half_shock_usd"] for r in pos), "as_written": tot,
                        "if_it_is_1_5x": sum(r["pnl_1_5x_shock_usd"] for r in pos),
                        "plus_or_minus_noise_beyond_the_factors": noise},
           **({"options": opt_rows} if opt_rows else {}),
           "by_position": pos, "coverage": cov,
           "what_this_is": "a scenario assumption the analyst wrote, mapped by code through each "
                           "holding's factor exposures; not a forecast, and not tested on held-out events the way the "
                           "historical estimate is",
           "numbers": "the shock is the analyst's (shown, editable); everything after it is code's"}
    moves = {t: float(L["betas"].loc[t].to_numpy() @ f) for t in L["betas"].index}
    s.__dict__["planb_scenario"] = {"id": _slug(title), "title": title, "p": None, "closes": None, "pnl_center": tot,
                                    "pnl_bad": min(sum(r["pnl_1_5x_shock_usd"] for r in pos), sum(r["pnl_half_shock_usd"] for r in pos)) - noise,
                                    "moves_center": moves, "moves_bad": {t: 1.5 * m for t, m in moves.items()},
                                    "severity": "extreme", "book_daily_sd_usd": cov["book_daily_sd_usd"], "book_value": book}
    return out


# ---------------------------------------------------------------- the numbers


def _first_hours(f_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-hours center and spread scale per factor, from #177 (first move kept at the close on average; the first
    move's spread relative to the close's)."""
    rev = pack()["reversion"]
    m = rev["market"]
    k = np.array([(rev.get(f) or m)["kept_at_close"] for f in FACTORS])
    sc = np.array([(rev.get(f) or m)["fast_to_close_sd"] for f in FACTORS])
    return f_end / np.clip(k, 0.5, 2.0), sc


def _q(x: np.ndarray) -> dict:
    a = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"likely_range": [float(a[1]), float(a[3])], "wide_range": [float(a[0]), float(a[4])], "middle": float(a[2])}


def _slug(t: str) -> str:
    return "scenario:" + (re.sub(r"[^a-z0-9]+", "-", (t or "event").lower()).strip("-")[:40] or "event")


def shared_noise(L: dict, vals: pd.Series, names: list[str], seed: int = 1734) -> dict | None:
    """The book's own noise beyond the factors, drawn jointly with the own noise of each held name that an option is
    written on (#235): one draw per name per scenario, shared by the stock and its options, so a protective put offsets
    the stock's own drops within each draw. Jointly normal with the measured covariance of the daily residuals (the book's
    and each name's), so the book's own noise keeps exactly its spread (daily_noise_usd) and each name its own (resid_sd).
    Returns {"idio": book dollars per draw, "name": {ticker: return per draw}}, both over ~1.5 sessions; None if no name."""
    r = L["returns"]
    held = [t for t in vals.index if t in r.columns and t in L["betas"].index]
    names = [u for u in dict.fromkeys(names) if u in held]
    if not names:
        return None
    F = factor_returns(r).reindex(columns=FACTORS)
    w = vals[held] / vals[held].sum()

    def resid(y, beta):
        e = y - (F.fillna(0) * beta).sum(axis=1)
        return e - e.mean()
    br = (r[held].fillna(0) * w).sum(axis=1)
    cols = [resid(br, (L["betas"].loc[held].T @ w).reindex(FACTORS).to_numpy()) * float(vals[held].sum())]
    cols += [resid(r[u].fillna(0), L["betas"].loc[u].reindex(FACTORS).to_numpy()) for u in names]
    E = pd.concat(cols, axis=1).dropna().tail(LOOKBACK).to_numpy()
    C = np.cov(E.T) + 1e-12 * np.eye(E.shape[1])
    Z = np.random.default_rng(seed).normal(0, 1, (N_DRAWS, E.shape[1])) @ np.linalg.cholesky(C).T * np.sqrt(1.5)
    return {"idio": Z[:, 0], "name": {u: Z[:, k + 1] for k, u in enumerate(names)}}


def scenario_draws(spec: dict, db: np.ndarray, noise_sd: float, seed: int = 1729, idio: np.ndarray | None = None) -> dict:
    """The shipped distribution for one event on one book (used by price() and by the calibration test, #182).
    db: the book's exposure per 1 sd of each factor (dollars, or returns for a $1 book); noise_sd: the book's own daily
    noise beyond the factors, same units. Middle = the engine's estimate through the exposures; spread = the real
    spread of past events of this severity (residual factor vectors) plus the book's own noise over ~1.5 sessions,
    recentered so the middle is the estimate. First hours: the same with #177's first-reaction scaling."""
    f = estimate(spec)
    rng = np.random.default_rng(seed)
    R, pool = resid_pool(spec)
    draw = R[rng.integers(0, len(R), N_DRAWS)]
    if idio is None:   # event windows span 1-2 sessions; a shared draw comes from shared_noise() (#235)
        idio = rng.normal(0, 1, N_DRAWS) * noise_sd * np.sqrt(1.5)
    noise = draw @ db + idio
    end = float(db @ f) + noise - np.median(noise)
    fast_c, fast_sc = _first_hours(f)
    noise_f = (draw * fast_sc) @ db + idio * 0.7
    fast = float(db @ fast_c) + noise_f - np.median(noise_f)
    return {"f": f, "draw": draw, "end": end, "fast": fast, "fast_c": fast_c, "fast_sc": fast_sc, "draw_pool": pool}


def _event_size(f: np.ndarray, factor_daily_sd: np.ndarray, sec_moves: dict | None = None) -> dict:
    """How big the event is in the markets it moves, whatever any one book holds (#220): the largest factor move (and
    sector move) in units of that market's normal two-session swing."""
    z = np.abs(f) / np.maximum(factor_daily_sd * np.sqrt(2), 1e-12)
    j = int(np.argmax(z)) if len(z) else 0
    top = {"market": FACTOR_WORDS[FACTORS[j]], "move": plain_moves(f)[FACTORS[j]], "vs_its_normal_2_day_swing": float(z[j])}
    for sec, mv in (sec_moves or {}).items():
        if abs(mv) / np.sqrt(2) > top["vs_its_normal_2_day_swing"]:
            top = {"market": f"{sec} stocks (beyond the market)", "move": f"{mv / np.sqrt(2):+.1f}x a normal 2-day swing",
                   "vs_its_normal_2_day_swing": float(abs(mv) / np.sqrt(2))}
    r = top["vs_its_normal_2_day_swing"]
    top["call"] = "big market event" if r >= 1.5 else ("market event" if r >= 0.75 else "barely a market event")
    return top


def _verdict(middle: float, book_daily_sd: float, spec: dict, event: dict | None = None) -> dict:
    """Is this worth calling an event for THIS book? (#219) The middle against the book's normal 2-session swing."""
    swing = float(book_daily_sd) * np.sqrt(2)
    ratio = abs(middle) / swing if swing > 0 else 0.0
    if not spec.get("market_moving", True):
        call, say = "noise", "no reason to expect this to move markets: your book's normal range applies"
    elif ratio < 0.5:
        call, say = "noise", "for your book this is within normal day-to-day noise"
    elif ratio < 1.0:
        call, say = "small", "a small move for your book, less than a normal two-day swing"
    else:
        call, say = "meaningful", "a move bigger than your book's normal two-day swing"
    out = {"call": call, "say": say, "middle_vs_normal_2_day_swing": ratio, "normal_2_day_swing_usd": swing}
    if event:
        out["the_event_itself"] = event
        if event["call"] != "barely a market event" and call == "noise":
            out["say"] = (f"a real {event['call'].replace('big ', 'big ')} ({event['market']} {event['move']}), but your holdings are "
                          "mostly out of its path: for your book it's within normal noise")
    return out


def real_event_draws(spec: dict, db: np.ndarray, noise_sd: float, factor_daily_sd: np.ndarray, seed: int = 1729,
                     idio: np.ndarray | None = None) -> dict:
    """The scenario if the event is a real market mover (a user's "what if" posits it; DECISIONS #213). Per factor the
    analyst signed: middle = push x sign x the factor's normal noise over the window, spread = real-event width x that
    noise (Plan C's gate model, real part). Factors signed 0, or never signed by the model: normal noise, middle 0. The
    co-movement across factors comes from past events of this severity (their residual vectors, rescaled to these
    widths). Same output shape as scenario_draws."""
    RE = real_event()
    signs, sizes = spec.get("factor_signs") or {}, spec.get("factor_sizes") or {}
    s_win = factor_daily_sd * np.sqrt(2)                          # event windows: the report day and the next session
    mid, width = np.zeros(len(FACTORS)), s_win.copy()
    for j, f in enumerate(FACTORS):
        sg = signs.get(f, 0)
        if sg:
            sm = RE["size_map"].get(sizes.get(f, "medium"))
            if sm:   # the measured move in the analyst's direction for this size call (#214)
                mid[j] = sm["move_in_llm_direction"] * sg * s_win[j]
                width[j] = max(sm["spread"], 1.0) * s_win[j]
            else:
                mid[j] = RE["factors"][f]["push_in_llm_direction"] * sg * s_win[j]
                width[j] = RE["factors"][f]["real_event_width"] * s_win[j]
    # factors the analyst didn't sign (and the six the model never signs): their typical co-move given the signed ones,
    # in units of each factor's own noise, from past big event windows (as stress_scenario fills them, #208)
    set_i = [j for j, f in enumerate(FACTORS) if f in signs]
    rest = [j for j in range(len(FACTORS)) if FACTORS[j] not in signs]
    if set_i and any(mid[set_i]):
        Ub = pack()["_U"][np.isin(pack()["_sev"], ["major", "extreme"])]
        Cz = np.corrcoef(Ub.T)
        x = mid[set_i] / s_win[set_i]
        mid[rest] = (Cz[np.ix_(rest, set_i)] @ np.linalg.solve(Cz[np.ix_(set_i, set_i)] + 1e-6 * np.eye(len(set_i)), x)) * s_win[rest]
    rng = np.random.default_rng(seed)
    Rr, pool = resid_pool(spec)
    draw = Rr[rng.integers(0, len(Rr), N_DRAWS)]
    sd = draw.std(0)
    draw = draw / np.where(sd > 0, sd, 1.0) * width
    if idio is None:   # a shared draw comes from shared_noise() (#235)
        idio = rng.normal(0, 1, N_DRAWS) * noise_sd * np.sqrt(1.5)
    dc = draw - np.median(draw, axis=0)
    noise = dc @ db + idio
    end = float(db @ mid) + noise - np.median(noise)
    fast_c, fast_sc = _first_hours(mid)
    noise_f = (dc * fast_sc) @ db + idio * 0.7
    fast = float(db @ fast_c) + noise_f - np.median(noise_f)
    return {"f": mid, "draw": draw, "end": end, "fast": fast, "fast_c": fast_c, "fast_sc": fast_sc, "width": width, "draw_pool": pool}


# ---------------------------------------------------------------- the sector layer (DECISIONS #215-217)


@lru_cache(maxsize=1)
def sector_pack() -> dict:
    return json.loads(Path(__file__).with_name("sectors.json").read_text(encoding="utf-8"))


def sector_loadings(s, L: dict) -> dict:
    """Second stage on top of the macro loadings (which stay as tested): each ticker's best sector factor (a sector ETF's
    move beyond the macro factors, in its own noise) and its loading on it, fitted on what the macro factors leave over
    the same 504 sessions; kept only if it explains > 2% more of the ticker's daily moves (as #215 tested)."""
    memo = s.__dict__.setdefault("_sector_loadings", {})
    tick = [t for t in L["betas"].index]
    key = (tuple(tick), str(s.data.today()))
    if key in memo:
        return memo[key]
    SP = sector_pack()["sectors"]
    etfs = {n: v["etf"] for n, v in SP.items()}
    r = L["returns"]
    F = factor_returns(r).reindex(columns=FACTORS)
    try:
        cs = s.data.closes(sorted(set(etfs.values())), days=LOOKBACK + 5).pct_change().iloc[1:].reindex(r.index)
    except Exception:  # noqa: BLE001
        memo[key] = {}
        return {}

    def resid(y):
        d = pd.concat([y.rename("y"), F], axis=1).dropna()
        if len(d) < 120:
            return None
        A = np.column_stack([np.ones(len(d)), d[FACTORS].fillna(0).to_numpy()])
        c = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)[0]
        return pd.Series(d["y"].to_numpy() - A @ c, index=d.index)

    sf = {}
    for n, e in etfs.items():
        if e in cs:
            res = resid(cs[e])
            if res is not None and res.std() > 0:
                sf[n] = res / res.std()
    out = {}
    for t in tick:
        if t not in r:
            continue
        y = r[t]
        e = resid(y)
        if e is None or y.var() <= 0:
            continue
        best = None
        for n, x in sf.items():
            d = pd.concat([e.rename("e"), x.rename("x")], axis=1).dropna()
            if len(d) < 120 or (d.x @ d.x) <= 0:
                continue
            b = float(d.e @ d.x / (d.x @ d.x))
            gain = float((1 - ((d.e - b * d.x) ** 2).sum() / (d.e ** 2).sum()) * d.e.var() / y.var())
            if gain > 0.02 and (best is None or gain > best[2]):
                best = (n, b, gain)
        if best:
            out[t] = {"sector": best[0], "loading": best[1], "extra_share_explained": best[2]}
    memo[key] = out
    return out


@lru_cache(maxsize=1)
def country_pack() -> dict:
    return json.loads(Path(__file__).with_name("countries.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def domicile() -> dict:
    d = json.loads(Path(__file__).with_name("domicile.json").read_text(encoding="utf-8"))["countries"]
    return {t: c for c, ts in d.items() for t in ts}


def country_loadings(s, L: dict) -> dict:
    """The country layer (#238), a second stage beside the sector layer: each holding's country and its loading on that
    country's factor (the country ETF's daily move beyond the 13 macro factors, in its own noise), over the same 504
    sessions. Which country:
      a country ETF          its own country (true by construction: its country factor is its own leftover move)
      a foreign stock / ADR  its home country (planb/domicile.json), kept if it adds > 2% of the stock's daily moves
      anything else          the best-fitting country, kept only if it adds > 2% held out (fit on the first 70% of the
                             window, scored on the rest) beyond the macro factors AND its sector, so a US energy stock
                             doesn't pick up "Canada" just because Canada is oil-heavy
: home country for foreign stocks (best fit picked UBS -> Germany, BHP -> Peru), best fit only for the rest."""
    memo = s.__dict__.setdefault("_country_loadings", {})
    tick = [t for t in L["betas"].index]
    key = (tuple(tick), str(s.data.today()))
    if key in memo:
        return memo[key]
    CP = country_pack()["countries"]
    etfs = {n: v["etf"] for n, v in CP.items()}
    own = {e: n for n, e in etfs.items()}
    dom = domicile()
    r = L["returns"]
    F = factor_returns(r).reindex(columns=FACTORS)
    try:
        cs = s.data.closes(sorted(set(etfs.values())), days=LOOKBACK + 5).pct_change().iloc[1:].reindex(r.index)
    except Exception:  # noqa: BLE001
        memo[key] = {}
        return {}

    def resid(y):
        d = pd.concat([y.rename("y"), F], axis=1).dropna()
        if len(d) < 120:
            return None
        A = np.column_stack([np.ones(len(d)), d[FACTORS].fillna(0).to_numpy()])
        c = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)[0]
        return pd.Series(d["y"].to_numpy() - A @ c, index=d.index)

    cf = {}
    for n, e in etfs.items():
        x = resid(cs[e]) if e in cs else None
        if x is not None and x.std() > 0:
            cf[n] = x / x.std()
    SL = sector_loadings(s, L)
    SF = {}
    out = {}
    for t in tick:
        if t not in r:
            continue
        y = r[t]
        e = resid(y)
        if e is None or y.var() <= 0:
            continue

        def fit(n, ee):
            d = pd.concat([ee.rename("e"), cf[n].rename("x")], axis=1).dropna()
            if len(d) < 120 or (d.x @ d.x) <= 0:
                return None
            b = float(d.e @ d.x / (d.x @ d.x))
            return b, float((1 - ((d.e - b * d.x) ** 2).sum() / (d.e ** 2).sum()) * d.e.var() / y.var()), d
        if t in own and own[t] in cf:
            f_ = fit(own[t], e)
            if f_:
                out[t] = {"country": own[t], "loading": f_[0], "extra_share_explained": f_[1], "how": "country ETF"}
            continue
        if t in dom:
            if dom[t] in cf:
                f_ = fit(dom[t], e)
                if f_ and f_[1] > 0.02:
                    out[t] = {"country": dom[t], "loading": f_[0], "extra_share_explained": f_[1], "how": "home country"}
            continue
        ee = e   # best fit: beyond the macro factors and the holding's sector
        sl = SL.get(t)
        if sl:
            if sl["sector"] not in SF:
                se = sector_pack()["sectors"][sl["sector"]]["etf"]
                try:
                    xs = resid(s.data.closes([se], days=LOOKBACK + 5)[se].pct_change().iloc[1:].reindex(r.index))
                    SF[sl["sector"]] = xs / xs.std() if xs is not None and xs.std() > 0 else None
                except Exception:  # noqa: BLE001
                    SF[sl["sector"]] = None
            if SF.get(sl["sector"]) is not None:
                ee = (e - sl["loading"] * SF[sl["sector"]].reindex(e.index).fillna(0))
        best = None
        for n in cf:
            d = pd.concat([ee.rename("e"), cf[n].rename("x")], axis=1).dropna()
            if len(d) < 200:
                continue
            cut = int(len(d) * 0.7)
            a, b_ = d.iloc[:cut], d.iloc[cut:]
            if (a.x @ a.x) <= 0:
                continue
            beta = float(a.e @ a.x / (a.x @ a.x))
            g_tr = 1 - ((a.e - beta * a.x) ** 2).sum() / (a.e ** 2).sum()
            g_te = (1 - ((b_.e - beta * b_.x) ** 2).sum() / (b_.e ** 2).sum()) * float(ee.var() / y.var())
            if best is None or g_tr > best[1]:
                best = (n, g_tr, g_te)
        if best and best[2] > 0.02:
            f_ = fit(best[0], ee)
            if f_:
                out[t] = {"country": best[0], "loading": f_[0], "extra_share_explained": float(best[2]), "how": "best fit (held out)"}
    memo[key] = out
    return out


def country_moves(news: list[dict] | None) -> dict:
    """country -> its factor's move in normal-noise units: the median 2-session move, or the 90th percentile when big."""
    CP = country_pack()["countries"]
    out = {}
    for leg in news or []:
        c, dirn, size = leg.get("country"), leg.get("direction"), leg.get("size", "typical")
        if c not in CP or dirn not in ("positive", "negative"):
            from ..agent.session import ToolError
            raise ToolError(f"country_news: country one of {sorted(CP)}, direction 'positive' or 'negative'")
        z = (CP[c]["two_day_move_p90"] if size == "big" else CP[c]["two_day_move_median"]) * np.sqrt(2) * (1 if dirn == "positive" else -1)
        out[c] = out.get(c, 0.0) + z
    return out


def sector_legs(s, legs: list[dict], L: dict, company: list[dict]) -> tuple[list[dict], dict]:
    """Sector headline legs (user's direction; size typical = the sector's median big 2-day move, big = its 90th
    percentile) and automatic peer spillover from company legs on sector names (step 2a's measured fraction). Returns the
    legs and, per sector, the factor move in daily noise units."""
    from ..agent.session import ToolError
    SP = sector_pack()["sectors"]
    SL = sector_loadings(s, L)
    moves, out = {}, []
    for leg in legs or []:
        sec, dirn, size = leg.get("sector"), leg.get("direction"), leg.get("size", "typical")
        if sec not in SP or dirn not in ("positive", "negative"):
            raise ToolError(f"sector_news: sector one of {sorted(SP)}, direction 'positive' or 'negative'")
        z = (SP[sec]["two_day_move_p90"] if size == "big" else SP[sec]["two_day_move_median"]) * np.sqrt(2) * (1 if dirn == "positive" else -1)
        moves[sec] = moves.get(sec, 0.0) + z
        out.append({"sector": sec, "direction": dirn, "size": size, "sector_move_vs_a_normal_day": z,
                    "how": f"{'the 90th percentile' if size == 'big' else 'the median'} 2-session move of the {sec} factor (beyond the "
                           f"market and macro factors, 2011-2026), in the direction you described"})
    for c in company or []:
        sl = SL.get(c["ticker"])
        if sl and SP.get(sl["sector"], {}).get("spillover_fraction", 0) > 0:
            rs = float(L["resid_sd"].get(c["ticker"], np.nan))
            if np.isfinite(rs) and rs > 0:
                z = SP[sl["sector"]]["spillover_fraction"] * c["move_middle"] / rs
                moves[sl["sector"]] = moves.get(sl["sector"], 0.0) + z
                out.append({"sector": sl["sector"], "from_company_news": c["ticker"], "sector_move_vs_a_normal_day": z,
                            "how": f"peers of a reporting {sl['sector']} name move about {SP[sl['sector']]['spillover_fraction']:.0%} of its move "
                                   f"(earnings spillover, #216), same direction {SP[sl['sector']]['spillover_same_direction']:.0%} of the time"})
    return out, moves


def company_legs(s, legs: list[dict], L: dict) -> list[dict]:
    """Single-name headline legs on top of the event (DECISIONS #213): for each {ticker, news_type, direction}, the
    typical move for that kind of company news (the S0b study, timing-clean, #205) in that stock's own noise beyond the
    factors, in the user's direction, applied to the book's exposure to the name (directly and through its ETFs)."""
    from .. import stocks
    from ..agent.session import ToolError
    stats = stocks._news_stats()
    try:
        eff = stocks._exposure(s)
    except Exception:  # noqa: BLE001
        eff = pd.DataFrame(columns=["value"])
    out = []
    for leg in legs or []:
        t, typ, dirn = str(leg.get("ticker", "")).upper(), leg.get("news_type"), leg.get("direction")
        size = leg.get("size", "typical")
        if typ not in stats or dirn not in ("positive", "negative"):
            raise ToolError(f"company_news: news_type one of {sorted(stats)}, direction 'positive' or 'negative'")
        c = stats[typ]["clean_timing"]
        Lt = L if t in L["resid_sd"].index else loadings(s.data, [t])
        rs = float(Lt["resid_sd"].get(t, np.nan))
        if not np.isfinite(rs):
            raise ToolError(f"no price history for {t}")
        sg = 1 if dirn == "positive" else -1
        expo = float(eff.at[t, "value"]) if t in eff.index else 0.0
        z_mid = c.get("p90_abs_day1") if size == "big" and c.get("p90_abs_day1") else c["median_abs_day1"]
        out.append({"ticker": t, "news_type": stocks.NEWS_TYPES.get(typ, typ), "direction": dirn, "size": size, "your_exposure_usd": expo,
                    "move_middle": sg * z_mid * rs, "move_sd": c["rms_day1"] * rs,
                    "headline_direction_held_historically": c["direction_hit_signed"],
                    "how": f"{'the large end (top 10%)' if size == 'big' else 'typical size'} of {stocks.NEWS_TYPES.get(typ, typ)} moves for S&P 500 stocks ({c['events']:,} events, "
                           f"2019-2026), in {t}'s own daily noise beyond the market; direction as you described it"})
    return out


def tail_probabilities(end: np.ndarray, book: float, k: int = 2) -> list[dict]:
    """P(loss > X) at round thresholds (a share of the book, rounded in dollars), from the same draws as the ranges.
    Keeps the first k thresholds whose probability is between 2% and 60%."""
    out = []
    for lv in TAIL_LEVELS:
        x = float(np.round(lv * book, -1 if lv * book < 1000 else -2))
        pr = float(np.mean(end < -x))
        if 0.02 <= pr <= 0.6:
            out.append({"loss_over_usd": x, "share_of_book": lv, "probability": pr})
    return out[:k]


# the ordinary-day rule at the open (C1d, DECISIONS #193): across the 12 test books an opening move sticks about one for
# one (median slope 1.05 in dollars) and 90% of a 2-session window's noise is still ahead
OPEN_SLOPE, OPEN_NOISE_LEFT = 1.05, 0.90


def after_open(s, v: pd.Series, book_daily_sd_usd: float) -> dict | None:
    """Once markets have opened on news that broke before the open: the book's opening move, and the range to tomorrow's
    close from the ordinary-day rule. None when today's open isn't known (before 09:30, no session, replays)."""
    gaps_fn = getattr(s.data, "opening_gaps", None)
    if gaps_fn is None:
        return None
    try:
        g = gaps_fn(list(v.index))
    except Exception:  # noqa: BLE001
        return None
    have = [t for t in v.index if t in g]
    if not have or v[have].sum() < 0.8 * v.sum():
        return None
    move = float(sum(v[t] * g[t] for t in have))
    mid = OPEN_SLOPE * move
    sd = OPEN_NOISE_LEFT * book_daily_sd_usd * np.sqrt(2)
    return {"book_opening_move_usd": move, "middle_to_tomorrows_close_usd": mid,
            "likely_range_usd": [mid - 0.674 * sd, mid + 0.674 * sd], "range_9_in_10_usd": [mid - 1.645 * sd, mid + 1.645 * sd],
            "how": "markets have already reacted: the opening move usually sticks (about one for one) and most of a normal "
                   "2-day range is still ahead; this beat the pre-news estimate by 11-26% on 2023-26 (DECISIONS #193)"}


def price(s, market_id: str | None = None, already_reported_before_open: bool = False, company_news: list | None = None,
          sector_news: list | None = None, country_news: list | None = None) -> dict:
    """Code prices the event the expert structured (#182: no LLM adjustment of any number). End of day: the engine's
    estimate through the book's loadings with the real spread of past events of this severity plus the book's own
    noise; first hours from #177; the nearest past events' best and worst on this book; tail probabilities."""
    from ..agent.session import ToolError
    s.require_portfolio()
    spec = s.__dict__.get("planb_spec")
    if spec is None:
        raise ToolError("call event_evidence first")
    p = pack()
    vals = s.priced["value"]
    tick = list(dict.fromkeys(list(vals.index) + REFS + [o.underlying for o in _options(s)]))
    L = loadings(s.data, tick)
    held = [t for t in vals.index if t in L["betas"].index]
    Bh = L["betas"].loc[held]
    v = vals[held]
    db = (Bh.T @ v).reindex(FACTORS).to_numpy()
    book = float(vals.sum())
    cov = book_coverage(L, vals)
    sn = shared_noise(L, vals, [o.underlying for o in _options(s)])   # options on held names share their noise (#235)
    idio = sn["idio"] if sn else None
    d = scenario_draws(spec, db, cov["daily_noise_usd"], idio=idio)
    hist = {"middle_usd": float(db @ d["f"]), "range_9_in_10_usd": [float(np.quantile(d["end"], 0.05)), float(np.quantile(d["end"], 0.95))]}
    mode = "history's average for events like this"
    fd = factor_returns(L["returns"]).reindex(columns=FACTORS).tail(LOOKBACK).std().fillna(1.0).to_numpy()
    if not spec.get("market_moving", True):   # the analyst sees no reason it moves markets: normal noise only (#219)
        d = real_event_draws(dict(spec, factor_signs={}), db, cov["daily_noise_usd"], fd, idio=idio)
        mode = "not a market event: your book's normal range, nothing added"
    elif spec.get("factor_signs"):   # a "what if" with the analyst's directions: the event as a real market mover (#213)
        d = real_event_draws(spec, db, cov["daily_noise_usd"], fd, idio=idio)
        mode = "if it's a real market mover (directions from the analyst, sizes from the real-event model)"
    f, end, fast, draw = d["f"], d["end"], d["fast"], d["draw"]
    legs = company_legs(s, company_news, L) if company_news else []
    if legs:   # each headline leg's draws, added to the book (independent of the event's factor draws)
        zl = np.random.default_rng(1732).normal(0, 1, (N_DRAWS, len(legs)))
        for k_, lg in enumerate(legs):
            pnl_leg = lg["your_exposure_usd"] * (lg["move_middle"] + zl[:, k_] * lg["move_sd"])
            end = end + pnl_leg
            fast = fast + pnl_leg * 0.8
            lg["pnl_middle_usd"] = lg["your_exposure_usd"] * lg["move_middle"]
            lg["pnl_range_9_in_10_usd"] = [float(np.quantile(pnl_leg, 0.05)), float(np.quantile(pnl_leg, 0.95))]
    sec_legs, sec_moves = sector_legs(s, sector_news, L, legs) if (sector_news or legs) else ([], {})
    SL = sector_loadings(s, L) if sec_moves else {}
    sec_mv = {t: sl["loading"] * sec_moves[sl["sector"]] for t, sl in SL.items() if sl["sector"] in sec_moves}
    for lg in legs:                     # a company's own headline already carries its own sector part
        if lg["ticker"] in sec_mv and any(x.get("from_company_news") == lg["ticker"] for x in sec_legs):
            sec_mv[lg["ticker"]] = sum(SL[lg["ticker"]]["loading"] * x["sector_move_vs_a_normal_day"] for x in sec_legs
                                       if x["sector"] == SL[lg["ticker"]]["sector"] and x.get("from_company_news") != lg["ticker"])
    cty_legs = []
    if country_news:                    # country legs (#238): each holding's country loading x the country's move
        cmv = country_moves(country_news)
        CL = country_loadings(s, L)
        for t, cl in CL.items():
            if cl["country"] in cmv:
                x = cl["loading"] * cmv[cl["country"]]
                if abs(x) > abs(sec_mv.get(t, 0.0)):   # with a sector move too: the larger of the two, never the sum
                    sec_mv[t] = x
        for c, z in cmv.items():
            reach = sorted(t for t in v.index if CL.get(t, {}).get("country") == c)
            cty_legs.append({"country": c, "country_move_vs_a_normal_day": z, "holdings_it_reaches": reach,
                             "your_book_usd": sum(float(v[t]) * CL[t]["loading"] * z for t in reach),
                             "how": f"the {'90th percentile' if any(l.get('size') == 'big' and l.get('country') == c for l in country_news) else 'median'} "
                                    f"2-session move of {c}'s factor (its ETF beyond the macro factors), in the direction you described"})
    if sec_mv:                          # sector legs: each holding's sector loading x the sector's move, with spread
        zs = np.random.default_rng(1733).normal(0, 0.5, N_DRAWS)
        sec_pnl = sum(float(v[t]) * mv for t, mv in sec_mv.items() if t in v.index)
        end = end + sec_pnl * (1 + zs)
        fast = fast + sec_pnl * 0.8 * (1 + zs)
        for x in sec_legs:
            x["your_book_usd"] = sum(float(v[t]) * SL[t]["loading"] * x["sector_move_vs_a_normal_day"] for t in v.index
                                     if t in SL and SL[t]["sector"] == x["sector"] and t != x.get("from_company_news"))
            x["holdings_it_reaches"] = sorted(t for t in v.index if t in SL and SL[t]["sector"] == x["sector"] and t != x.get("from_company_news"))
    opt_rows, opt_mid = [], 0.0
    if _options(s):   # each option fully repriced on the same factor draws as the book (#209)
        dc = draw - np.median(draw, axis=0)
        z = np.random.default_rng(1730).normal(0, 1, N_DRAWS)
        zl_ = np.random.default_rng(1732).normal(0, 1, (N_DRAWS, len(legs))) if legs else None
        zs_ = np.random.default_rng(1733).normal(0, 0.5, N_DRAWS)
        for o in option_setup(s, L):
            u = o["pos"].underlying
            # the underlier moves exactly as the stock would (#235): the same sector leg and company legs, on the same
            # spread draws; its own noise is the stock's own draw when the name is held, else independent
            leg_draw = sum((lg["move_middle"] + zl_[:, k_] * lg["move_sd"] for k_, lg in enumerate(legs) if lg["ticker"] == u),
                           np.zeros(N_DRAWS)) if legs else np.zeros(N_DRAWS)
            sec_draw = sec_mv.get(u, 0.0) * (1 + zs_)
            own = sn["name"][u] if sn and u in sn["name"] else z * o["own"] * np.sqrt(1.5)
            move = o["beta"] @ f + dc @ o["beta"] + sec_draw + leg_draw + own
            mid_move = float(o["beta"] @ f) + sec_mv.get(u, 0.0) + sum(lg["move_middle"] for lg in legs if lg["ticker"] == u)
            pnl = option_pnl(o, move)
            end = end + pnl
            book += o["value"]
            row = option_row(o, mid_move, pnl, float(move[np.argsort(pnl)[int(0.05 * N_DRAWS)]]))
            row["underlier_legs"] = {"sector": sec_mv.get(u, 0.0), "company": sum(lg["move_middle"] for lg in legs if lg["ticker"] == u),
                                     "noise_shared_with_the_stock": bool(sn and u in sn["name"])}
            opt_mid += row["pnl_middle_usd"]
            opt_rows.append(row)
    leg_mv = {}
    for lg in legs:
        leg_mv[lg["ticker"]] = leg_mv.get(lg["ticker"], 0.0) + lg["move_middle"]
    pos = [{"ticker": t, "value": float(v[t]),
            "move_if_it_happens": float(Bh.loc[t].to_numpy() @ f) + leg_mv.get(t, 0.0) + sec_mv.get(t, 0.0),
            "pnl_if_it_happens": float(v[t] * (Bh.loc[t].to_numpy() @ f + leg_mv.get(t, 0.0) + sec_mv.get(t, 0.0))),
            **({"of_which_from_the_event": float(v[t] * (Bh.loc[t].to_numpy() @ f))} if (t in leg_mv or t in sec_mv) else {}),
            **({"of_which_from_the_headline": float(v[t] * leg_mv[t])} if t in leg_mv else {}),
            **({"of_which_from_the_sector": float(v[t] * sec_mv[t])} if t in sec_mv else {})} for t in held]
    pos.sort(key=lambda r: r["pnl_if_it_happens"])
    contrib = {FACTOR_WORDS[fac]: float(db[j] * f[j]) for j, fac in enumerate(FACTORS) if abs(db[j] * f[j]) >= 0.0005 * book}
    bad_i = int(np.argsort(end)[int(0.05 * N_DRAWS)])          # the draw at the bad end (1 in 20)
    f_bad = f + draw[bad_i] - np.median(draw, axis=0)
    moves_center = {t: float(L["betas"].loc[t].to_numpy() @ f) for t in L["betas"].index}
    moves_bad = {t: float(L["betas"].loc[t].to_numpy() @ f_bad) for t in L["betas"].index}
    analogs = s.__dict__.get("planb_analogs") or []
    near = sorted(analogs, key=lambda a: a["your_book_if_it_happened_today"])
    nearest = ({"worst": {k: near[0][k] for k in ("date", "what")} | {"your_book_usd": near[0]["your_book_if_it_happened_today"]},
                "best": {k: near[-1][k] for k in ("date", "what")} | {"your_book_usd": near[-1]["your_book_if_it_happened_today"]}}
               if near else None)
    tails = tail_probabilities(end, book)
    mkt = None
    if market_id:
        m = next((m for m in s.data.event_markets() if m["id"] == market_id), None)
        if m is not None and _p(m) is not None:
            mkt = {"market_id": market_id, "title": m.get("title"), "p": _p(m), "closes": m.get("close_date"),
                   "note": "the expert judged this market to be the event; its odds are Kalshi's"}
    opened = after_open(s, v, cov["book_daily_sd_usd"]) if already_reported_before_open else None
    rev = p["reversion"]["market"]
    title = spec.get("title") or CATEGORY_WORDS[spec["category"]]
    q_end, q_fast = _q(end), _q(fast)
    out = {"scenario": title, "severity": spec["severity"], "book_value": book,
           "if_it_happens": {"first_hours": q_fast, "end_of_day": q_end},
           "bad_case_1_in_20_end_of_day": float(np.quantile(end, 0.05)),
           "nearest_past_events_on_this_book": nearest,
           "tail_probabilities_end_of_day": tails,
           "reply_facts": {"middle_usd": q_end["middle"], "range_9_in_10_usd": q_end["wide_range"],
                           "worst_nearest_past_event": (nearest or {}).get("worst"), "tail": tails[0] if tails else None},
           "after_the_open": opened or ("today's open isn't known yet (before 09:30 or no session): the estimate above "
                                         "stands; ask again after the open" if already_reported_before_open else None),
           "mode": mode,
           "verdict": _verdict(q_end["middle"], cov["book_daily_sd_usd"], spec,
                               _event_size(f, fd, sec_moves) if spec.get("market_moving", True) and spec.get("factor_signs") else None),
           "history_average_for_events_like_this": hist,
           **({"factor_middle_moves": {FACTOR_WORDS[f_]: plain_moves(f)[f_] for f_ in SIGNED},
               "real_event_model": real_event()["evidence"]} if spec.get("factor_signs") else {}),
           **({"company_news": legs} if legs else {}),
           **({"sector_news": sec_legs} if sec_legs else {}),
           **({"country_news": cty_legs} if cty_legs else {}),
           "by_position": pos, "by_factor_usd": contrib,
           **({"options": opt_rows, "options_how": "each option fully repriced (spot, its implied vol via the measured "
               "spot-vol response, 2 sessions of time) on the same draws as the book; the book's range includes them"}
              if opt_rows else {}),
           "code_estimate": {FACTOR_WORDS[fac]: BUCKET_WORDS[_bucket(f[j] / _zsd(fac))] for j, fac in enumerate(FACTORS)
                             if fac in p["llm_factors"]},
           "first_move_sticks": {"on_average": "yes: the close is about the first move or a bit more",
                                 "share_of_big_events_giving_back_half_by_close": rev["gave_back_half"],
                                 "share_flipping_sign_by_close": rev["flipped_by_close"]},
           "coverage": cov,
           "probability": mkt or "no market prices this event: say it's conditional ('if it happens'), never supply odds",
           **({"priced_in": {"market_odds": mkt["p"], "if_it_happens_usd": q_end["middle"] * (1 - mkt["p"]),
                             "if_it_doesnt_usd": -q_end["middle"] * mkt["p"],
                             "how": "today's odds already carry odds x the move: if it happens the book moves the middle x (1 - "
                                    "odds); if it doesn't, -middle x odds (#218)"}} if mkt else {}),
           "history_says": LEARNABLE[spec["category"]],
           "ranges_mean": "likely = middle half of outcomes, 9 in 10 = the wide range; from the real spread of past events "
                          "of this severity plus this book's own daily noise",
           "numbers": "every number here is code's; quote them, never adjust or add your own"}
    s.__dict__["planb_scenario"] = {"id": _slug(title), "title": title, "p": (mkt or {}).get("p"),
                                    "closes": (mkt or {}).get("closes"),
                                    "pnl_center": float(db @ f) + opt_mid + sum(lg.get("pnl_middle_usd", 0.0) for lg in legs)
                                                  + sum(float(v[t]) * mv for t, mv in sec_mv.items() if t in v.index),
                                    "pnl_bad": float(np.quantile(end, 0.05)), "moves_center": moves_center, "moves_bad": moves_bad,
                                    "severity": spec["severity"], "book_daily_sd_usd": cov["book_daily_sd_usd"], "book_value": book}
    s.active_event = None
    s.__dict__.pop("active_template", None)
    return out


# ---------------------------------------------------------------- should you hedge at all? (DECISIONS #199, #200)

HEDGE_EVIDENCE = Path(__file__).resolve().parents[3] / "docs/pmrm/results/planc_hedge.json"


@lru_cache(maxsize=1)
def hedge_evidence() -> dict:
    """The fair hedge test's pooled evidence (2023-26, real option prices; scripts/planc_hedge.py)."""
    return json.loads(HEDGE_EVIDENCE.read_text(encoding="utf-8"))["product_evidence"]


def hedge_verdict(sc: dict, rows: list[dict], bad_case_weight: float = 1.0) -> dict:
    """Code's answer to "is any hedge worth its premium?". The default is no: over 576 events of 2023-26, a protective put
    held through the news cost ~$135 per $100k net and paid for itself ~3 times in 10. Two exceptions: extreme events
    (about break-even, the bad case ~4x likelier) and a user who weighs a loss beyond their normal bad case several times
    more than everyday ups and downs (break-even weight 3-6x). Dollar figures are the test's, scaled to this book."""
    ev = hedge_evidence()
    sev = sc.get("severity") or "moderate"
    by = ev["by_severity"].get(sev, ev)
    scale = float(sc.get("book_value") or 100_000) / 100_000
    weight_needed = float(min(ev["break_even_weight"]))
    normal_bad_2d = 1.645 * float(sc.get("book_daily_sd_usd") or 0.0) * np.sqrt(2)     # the book's whole daily swing
    priced = [r for r in rows if r.get("hedge") not in (None, "do_nothing") and r.get("cost")]
    cheapest = min(priced, key=lambda r: r["cost"]) if priced else None
    if sev == "extreme":
        verdict, why = "worth considering", "extreme events are the one case where protection has come close to paying for itself"
    elif bad_case_weight >= weight_needed:
        verdict, why = "worth considering", "you said a loss beyond your normal bad case would hurt much more than everyday swings"
    else:
        verdict, why = "skip by default", "for news like this, protection has usually cost more than it saved"
    return {"verdict": verdict, "why": why,
            "your_normal_2_day_bad_case_usd": -normal_bad_2d,
            "cheapest_hedge": {"name": cheapest.get("name"), "cost_usd": cheapest["cost"]} if cheapest else None,
            "history": {"what": f"protective puts bought before {ev['events']} news events of {ev['years']} and sold "
                                f"{ev['hold']}, on books like yours",
                        "events_like_this": sev, "share_that_paid_for_themselves": by["share_paid"],
                        "net_cost_per_hedge_usd": -by["net_per_100k"] * scale,
                        "chance_of_a_loss_beyond_your_normal_bad_case": by["beyond_bad_case_share"],
                        "when_that_happened_the_hedge_covered": ev["hedge_return_when_beyond"] / -ev["loss_when_beyond"]},
            "worth_it_if": f"a loss beyond your normal bad case would hurt you about {weight_needed:.0f}x more than "
                           "everyday ups and downs (say so and Worsey re-checks)",
            "numbers": "code's, from the fair hedge test (DECISIONS #199); quote them, never adjust them"}


# ---------------------------------------------------------------- discovery: what could hit this book


def scan(s, horizon_days: int = 91, n: int = 6) -> dict:
    """Code-only ranking for THIS book: every kind of big event in the library replayed through today's loadings (how
    bad the bad case is, how often such events happen), plus scheduled releases in today's stock-bond regime, plus
    Kalshi markets that price something of that kind inside the horizon."""
    s.require_portfolio()
    p = pack()
    vals = s.priced["value"]
    L = loadings(s.data, list(dict.fromkeys(list(vals.index) + [o.underlying for o in _options(s)])))
    db = _dollar_betas(L, vals)
    book = float(vals.sum())
    pnl = p["_U"] @ db
    opts = option_setup(s, L) if _options(s) else []
    for o in opts:                     # each option repriced on every past event's move of its underlying (#209)
        pnl = pnl + option_pnl(o, p["_U"] @ o["beta"])
        book += o["value"]
    big = np.isin(p["_sev"], ["major", "extreme"])
    yrs = np.array([int(e["date"][:4]) for e in p["events"]])
    cats = np.array([e["category"] for e in p["events"]])
    today = pd.Timestamp(s.data.today())
    try:
        universe = s.data.event_markets()
    except Exception:  # noqa: BLE001
        universe = []
    # per position (DECISIONS #207): the same replay with
    # each holding's own exposures. They add up exactly to the book's factor P&L; each holding's own range adds its noise
    # beyond the factors (residual daily sd x sqrt(1.5), as the position-level test #186 scored)
    held = [t for t in vals.index if t in L["betas"].index]
    Bv = L["betas"].loc[held].to_numpy() * vals[held].to_numpy()[:, None]
    own = (L["resid_sd"].reindex(held).fillna(0.0) * vals[held].abs()).to_numpy() * np.sqrt(1.5)
    Zown = np.random.default_rng(1729).normal(0, 1, (20, len(held)))

    def by_position(Ux: np.ndarray) -> dict:
        P = Ux @ Bv.T                                         # past events x positions, dollars
        if opts:
            P = np.column_stack([P] + [option_pnl(o, Ux @ o["beta"]) for o in opts])
        tot = P.sum(1)
        worst = np.argsort(tot)[:max(1, int(round(0.1 * len(tot))))]
        bad_avg = float(tot[worst].mean())
        pos = []
        for k, o in enumerate(opts):
            j = len(held) + k
            mv = Ux @ o["beta"]
            zz = np.random.default_rng(1731 + k).normal(0, 1, (20, 1)) * o["own"] * np.sqrt(1.5)
            draws = option_pnl(o, (mv[None, :] + zz).ravel())
            pos.append({"ticker": o["pos"].label, "value": o["value"], "typical_usd": float(np.median(P[:, j])),
                        "own_bad_case_1_in_10_usd": float(np.quantile(draws, 0.1)),
                        "own_good_case_1_in_10_usd": float(np.quantile(draws, 0.9)),
                        "in_the_books_bad_case_usd": float(P[worst, j].mean()),
                        "share_of_books_bad_case": float(P[worst, j].mean() / bad_avg) if bad_avg < 0 else None})
        for j, t in enumerate(held):
            draws = (P[:, j][None, :] + Zown[:, j][:, None] * own[j]).ravel()
            pos.append({"ticker": t, "value": float(vals[t]), "typical_usd": float(np.median(P[:, j])),
                        "own_bad_case_1_in_10_usd": float(np.quantile(draws, 0.1)),
                        "own_good_case_1_in_10_usd": float(np.quantile(draws, 0.9)),
                        "in_the_books_bad_case_usd": float(P[worst, j].mean()),
                        "share_of_books_bad_case": float(P[worst, j].mean() / bad_avg) if bad_avg < 0 else None})
        pos.sort(key=lambda r: r["in_the_books_bad_case_usd"])
        return {"books_worst_1_in_10_average_usd": bad_avg, "positions": pos}

    rows = []
    for c in CATEGORIES:
        m = big & (cats == c)
        if m.sum() < 5:
            continue
        x = pnl[m]
        worst = int(np.argmin(np.where(m, pnl, np.inf)))
        wp = np.concatenate([p["_U"][worst] @ Bv.T, [float(option_pnl(o, [p["_U"][worst] @ o["beta"]])[0]) for o in opts]])
        rows.append({"risk_id": f"category:{c}", "kind": "one-off event", "risk": CATEGORY_WORDS[c],
                     "typical_usd": float(np.median(x)), "bad_case_1_in_10_usd": float(np.quantile(x, 0.1)),
                     "good_case_1_in_10_usd": float(np.quantile(x, 0.9)),
                     "how_often": f"{(m & (yrs >= 2016) & (yrs <= 2025)).sum() / 10:.1f} big ones a year (2016-2025)",
                     "worst_past_example": {"date": p["events"][worst]["date"], "what": p["events"][worst]["title"],
                                            "your_book_usd": float(pnl[worst]),
                                            "by_position_usd": {t: float(wp[j]) for j, t in enumerate(held + [o["pos"].label for o in opts])}},
                     "by_position": by_position(p["_U"][m]),
                     "history_says": LEARNABLE[c]})
    for typ in ("fomc", "cpi", "jobs", "gdp", "pce"):
        R = [r for r in p["releases"] if r["type"] == typ]
        if len(R) < 8:
            continue
        UR = np.array([r["U"] for r in R])
        x = UR @ db + sum((option_pnl(o, UR @ o["beta"]) for o in opts), np.zeros(len(R)))
        rows.append({"risk_id": f"release:{typ}", "kind": "scheduled release", "risk": RELEASE_WORDS[typ],
                     "typical_usd": float(np.median(x)), "bad_case_1_in_10_usd": float(np.quantile(x, 0.1)),
                     "good_case_1_in_10_usd": float(np.quantile(x, 0.9)),
                     "how_often": f"{len(R)} release days since 2021 (today's regime: stocks and bonds move apart)",
                     "history_says": "releases move rates first; how much reaches stocks depends on the regime",
                     "by_position": by_position(np.array([r["U"] for r in R])),
                     "kalshi": _next_release(universe, typ, today)})
    rows.sort(key=lambda r: r["bad_case_1_in_10_usd"])
    rows = rows[:n]
    sector_risks = []
    try:                                 # a bad day in each sector the book leans on (#217)
        SP, SL = sector_pack()["sectors"], sector_loadings(s, L)
        for sec, meta in SP.items():
            hold = [(t, float(vals[t]) * SL[t]["loading"]) for t in vals.index if t in SL and SL[t]["sector"] == sec]
            if hold:
                expo = sum(x for _, x in hold)
                move = meta["two_day_move_p95"] * np.sqrt(2)
                sector_risks.append({"sector": sec, "a_bad_2_day_sector_move_1_in_20_usd": -abs(expo) * move,
                                     "by_holding_usd": {t: -abs(x) * move * np.sign(x * expo) for t, x in hold}})
        sector_risks.sort(key=lambda r: r["a_bad_2_day_sector_move_1_in_20_usd"])
    except Exception:  # noqa: BLE001
        sector_risks = []
    cov = book_coverage(L, vals)
    out = {"book_value": book, "horizon_days": horizon_days, "risks": rows, "coverage": cov,
           **({"sector_risks": sector_risks[:4], "sector_risks_means": "a 1-in-20 two-session move of that sector beyond the "
               "market and macro factors, through each holding's sector loading (#215-217)"} if sector_risks else {}),
           "regime_now": {"as_of": p["regime"]["as_of"], **p["regime"]["labels"]},
           "exposures_per_1sd_usd": {FACTOR_WORDS[f]: float(db[j]) for j, f in enumerate(FACTORS) if abs(db[j]) >= 0.001 * book},
           "method": "each past big event replayed through your book's current factor exposures; bad case = the worst "
                     "1 in 10 of those. One-off events have no odds here (ask about one to look for a market); releases "
                     "show the next date Kalshi lists",
           "by_position_means": "per risk, each holding: its typical move, its own 1-in-10 bad and good case (factor moves plus "
                                "its own noise), and its part of the book's bad case (the average of the book's worst 1 in 10 "
                                "past events, split by holding; the parts add up to that average)"}
    s.__dict__["planb_scan"] = out
    return out
