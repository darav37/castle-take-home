"""ETF matching for restricted traders (Option 2; SPEC §5a; DECISIONS #122).

A user who can't trade a stock (bank compliance, say) gives the exposure they want and what they're allowed to trade;
this returns the small portfolio of permitted ETFs (plus cash) that tracks it most closely, tested out of sample, with
what exposure is lost. Pure code: every number here is computed, the etf_desk expert only words it.

  "closest" = lowest tracking error: std of the daily return gap (target − replica), annualized ×√252, on total returns.
  weights   = ETFs + a cash line summing to 1; long-only unless the user's restrictions allow shorts (gross cap).
  selection = shortlist by correlation → greedy forward selection with refits → swap pass → half-sample stability check.
  test      = train 2y, test the next 6m with weights held buy-and-hold (holding periods are common in restricted
              accounts); daily-rebalanced as a second line; baselines SPY, best single permitted ETF, holdings-based;
              rolling windows when history allows.
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data.cache import CACHE_ROOT

PANEL_FILE = CACHE_ROOT / "etf_panel" / "panel.pkl"
TRAIN_DAYS, TEST_DAYS, PANEL_YEARS = 504, 126, 3.5
ANN = math.sqrt(252)
CASH_PROXY = "BIL"          # 1–3 month T-bill ETF; falls back to a flat 4%/yr if absent
FLAT_RF = 0.04 / 252
KINDS = ("broad", "sector", "thematic", "country", "bond", "commodity", "single_stock", "other")


# ---------------------------------------------------------------- restrictions (from etf_desk's intake)

@dataclass
class Restrictions:
    """What this user may trade. Leveraged and inverse funds are never allowed (daily reset decay)."""
    allowed: list = field(default_factory=lambda: ["broad", "sector", "thematic", "country", "bond", "commodity", "other"])
    allow_short: bool = False
    gross_cap: float = 1.5          # Σ|w| when shorting is allowed
    allow_cash: bool = True
    approved_list: list | None = None   # if the firm has one: only these tickers
    banned: list = field(default_factory=list)
    min_hold_days: int | None = None

    def __post_init__(self):
        bad = set(self.allowed) - set(KINDS)
        if bad:
            raise ValueError(f"unknown fund kinds {sorted(bad)}; known: {list(KINDS)}")
        if not 1.0 <= self.gross_cap <= 3.0:
            raise ValueError("gross_cap must be between 1 and 3")


# ---------------------------------------------------------------- fund classification

# "UltraShort" / "UltraPro" are one word, so \bultra\b alone misses them (seen live Oct 17: SSG, REW got through)
_LEV = re.compile(r"(\b[1-4](\.\d+)?x\b|\bultra ?(pro|short)\b(?![- ](term|duration|bond|treasury|income))|"
                  r"\bultra\b(?![- ]short[- ](term|duration|bond|treasury|income))|\bleverag|\bdaily\b)", re.I)
_INV = re.compile(r"(\binverse\b|\bbear\b|-[1-4](\.\d+)?x|\bultra ?short\b(?![- ](term|duration|bond|treasury|income))|"
                  r"\bshort\b(?![- ]?(term|duration|maturity|-dated|dated|treasury|bond|t-bill|income)))", re.I)
_SINGLE = re.compile(r"(yieldmax|option income strategy|single[- ]stock|\btarget (12|25|50)\b)", re.I)
_BOND = re.compile(r"(bond|treasur|muni|municipal|fixed income|aggregate|\btips\b|corporate|floating rate|mortgage|\bclo\b|"
                   r"t-bill|ultra[- ]short|income fund|credit)", re.I)
_COMMOD = re.compile(r"(gold|silver|platinum|palladium|\boil\b|crude|natural gas|commodit|copper|agricultur|bitcoin|ether|crypto)", re.I)
_COUNTRY = re.compile(r"(international|emerging|developed|europe|eurozone|asia|pacific|japan|china|india|brazil|mexico|canada|"
                      r"korea|taiwan|germany|united kingdom|\buk\b|france|australia|latin america|frontier|ex-us|ex us|world|global|"
                      r"saudi|israel|vietnam|indonesia|africa|chile|italy|spain|switzerland|singapore|hong kong)", re.I)
_THEME = re.compile(r"(innovation|robot|artificial intelligence|\bai\b|cyber|cloud|clean|solar|wind|blockchain|genomic|space|"
                    r"electric vehicle|\bev\b|cannabis|esports|metaverse|internet|fintech|lithium|uranium|disruptive|future|"
                    r"autonomous|drone|quantum|hydrogen|nuclear|infrastructure)", re.I)
_SECTOR = re.compile(r"(select sector|financial|technology|tech\b|health|energy|utilit|industrial|materials|real estate|reit|"
                     r"consumer|communication|semiconductor|biotech|bank|insurance|homebuild|pharma|retail|aerospace|defense|"
                     r"transport|software|oil & gas|exploration|metals & mining|gold miners|regional)", re.I)
_BROAD = re.compile(r"(s&p 500|total (stock )?market|russell|nasdaq[- ]100|\bdow\b|mid[- ]?cap|small[- ]?cap|large[- ]?cap|"
                    r"\bvalue\b|\bgrowth\b|dividend|momentum|quality|low vol|minimum vol|equal weight|msci usa|\bcore\b|"
                    r"\b500\b|1000|2000|3000|broad market)", re.I)


def classify_name(name: str) -> dict:
    """First pass from the fund's name: kind plus leveraged / inverse / single-stock flags (heuristic, refined by holdings)."""
    n = name or ""
    flags = {"leveraged": bool(_LEV.search(n)) and not _BOND.search(n), "inverse": bool(_INV.search(n)),
             "single_stock": bool(_SINGLE.search(n))}
    if flags["single_stock"]:
        kind = "single_stock"
    elif _COMMOD.search(n) and not re.search(r"miners|producers|equity", n, re.I):
        kind = "commodity"
    elif _BOND.search(n) and not re.search(r"dividend|equity", n, re.I):
        kind = "bond"
    elif _COUNTRY.search(n):
        kind = "country"
    elif _THEME.search(n):
        kind = "thematic"
    elif _SECTOR.search(n):
        kind = "sector"
    elif _BROAD.search(n):
        kind = "broad"
    else:
        kind = "other"
    return {"kind": kind, **flags, "by": "name"}


def classify_holdings(base: dict, h: pd.DataFrame | None) -> dict:
    """Refine from holdings: mostly bonds → bond; one name ≥ 50% → single-stock; one sector ≥ 60% → sector."""
    if h is None or h.empty:
        return base
    out = dict(base, by="holdings", n_holdings=int(len(h)))
    ac = h.get("asset_class", pd.Series(dtype=str)).astype(str).str.lower()
    w = h["weight"].astype(float)
    names = (h["name"] if "name" in h else h["ticker"]).astype(str)
    # swap-based or geared funds: index swaps in the holdings, short lines, or gross exposure well above 100%
    if names.str.contains(r"\bswap\b", case=False, regex=True).any() or (w < -0.05).any() or w.abs().sum() > 1.3:
        out.update(leveraged=True, derivatives=True, inverse=bool(out.get("inverse") or (w < -0.05).any()))
        return out
    if w.sum() > 0 and w[ac.str.contains("fixed|bond", na=False)].sum() / w.sum() >= 0.6:
        out["kind"] = "bond"
        return out
    eq = w[ac.str.contains("equity", na=False)] if ac.str.contains("equity", na=False).any() else w
    if len(eq) and eq.max() / max(eq.sum(), 1e-9) >= 0.5 and base["kind"] not in ("commodity",):
        out["kind"], out["single_stock"] = "single_stock", True
        return out
    if "sector" in h and h["sector"].notna().any() and base["kind"] in ("broad", "other"):
        sec = h.dropna(subset=["sector"]).groupby("sector")["weight"].sum()
        top = float(sec.max() / max(sec.sum(), 1e-9)) if len(sec) else 0.0
        out["top_sector_share"] = round(top, 3)
        if top >= 0.6:
            out["kind"] = "sector"
        elif base["kind"] == "other" and len(h) >= 100 and top < 0.45:
            out["kind"] = "broad"
    return out


def permitted(info: dict, ticker: str, r: Restrictions) -> tuple[bool, str]:
    if info.get("leveraged") or info.get("inverse"):
        return False, "leveraged or inverse (never allowed)"
    if ticker in set(r.banned):
        return False, "banned by the user"
    if r.approved_list is not None and ticker not in set(r.approved_list):
        return False, "not on the approved list"
    if info["kind"] not in r.allowed:
        return False, f"{info['kind']} funds not allowed"
    return True, ""


# ---------------------------------------------------------------- data: one compact price panel for everything

def _weekdays(start: date, end: date) -> list[date]:
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def load_panel(today: date | None = None, years: float = PANEL_YEARS, fetch=None, log=print) -> dict:
    """Daily closes for every US ticker, from grouped daily bars (one call per day), stored as ONE compact float32
    table and extended incrementally. `fetch(day) -> rows` is injectable for tests; default is Massive."""
    from .data import massive
    fetch = fetch or massive.grouped_daily
    today = today or date.today()
    start = today - timedelta(days=int(365.25 * years))
    closes, fetched = pd.DataFrame(dtype="float32"), set()
    if PANEL_FILE.exists():
        stored = pickle.loads(PANEL_FILE.read_bytes())
        if "closes_df" in stored:
            closes, fetched = stored["closes_df"], set(stored["days_fetched"])
        else:   # first format (nested dicts, Oct 17): convert once
            closes = pd.DataFrame.from_dict(stored["closes"], orient="index").astype("float32")
            closes.index = pd.to_datetime(closes.index)
            fetched = set(stored["days_fetched"])
            _save(closes, fetched)
    todo = [d for d in _weekdays(start, today - timedelta(days=1)) if d.isoformat() not in fetched]
    new_rows = {}
    for i, d in enumerate(todo):
        rows = fetch(d.isoformat())
        if rows:   # holidays come back empty and are remembered as fetched
            new_rows[pd.Timestamp(d)] = {r["T"]: float(r["c"]) for r in rows if r.get("c")}
        fetched.add(d.isoformat())
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            if new_rows:
                closes = pd.concat([closes, pd.DataFrame.from_dict(new_rows, orient="index").astype("float32")]).sort_index()
                closes = closes[~closes.index.duplicated(keep="last")]
                new_rows = {}
            _save(closes, fetched)
            log(f"[etfmatch] price panel {i + 1}/{len(todo)} days")
    closes = closes.sort_index()
    return {"closes": closes[closes.index >= pd.Timestamp(start)].astype("float64")}


def _save(closes: pd.DataFrame, fetched: set) -> None:
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PANEL_FILE.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"closes_df": closes, "days_fetched": sorted(fetched)}))
    tmp.replace(PANEL_FILE)


def total_returns(closes: pd.Series, divs: list[dict] | None) -> pd.Series:
    """Daily total return: price change plus cash dividends on their ex-dates."""
    r = closes.pct_change()
    for dv in divs or []:
        ex, amt = dv.get("ex_dividend_date"), dv.get("cash_amount")
        if not ex or not amt:
            continue
        t = pd.Timestamp(ex)
        if t in closes.index:
            i = closes.index.get_loc(t)
            if i > 0 and closes.iloc[i - 1] > 0:
                r.iloc[i] += float(amt) / float(closes.iloc[i - 1])
    return r


# ---------------------------------------------------------------- the fit

def fit(y: np.ndarray, X: np.ndarray, cash_r: np.ndarray, r: Restrictions) -> tuple[np.ndarray, float]:
    """Weights w (per column of X) and cash c minimizing the mean squared daily gap y − Xw − c·cash, with
    Σw + c = 1, c ≥ 0 (if cash allowed, else c = 0), w ≥ 0 (long-only) or Σ|w| ≤ gross_cap (shorts allowed).
    Mean squared gap (not just its variance) so a level drift between target and replica is penalized too."""
    n = X.shape[1]
    if r.allow_short:   # w = p − q with p, q ≥ 0: keeps the gross constraint smooth
        def unpack(z):
            return z[:n] - z[n:2 * n], z[2 * n] if r.allow_cash else 0.0
        z0 = np.r_[np.full(n, 1.0 / n), np.zeros(n), [0.0] if r.allow_cash else []]
        bounds = [(0, r.gross_cap)] * (2 * n) + ([(0, 1)] if r.allow_cash else [])
        cons = [{"type": "eq", "fun": lambda z: unpack(z)[0].sum() + unpack(z)[1] - 1.0},
                {"type": "ineq", "fun": lambda z: r.gross_cap - z[:2 * n].sum()}]
    else:
        def unpack(z):
            return z[:n], z[n] if r.allow_cash else 0.0
        z0 = np.r_[np.full(n, 1.0 / n), [0.0] if r.allow_cash else []]
        bounds = [(0, 1)] * n + ([(0, 1)] if r.allow_cash else [])
        cons = [{"type": "eq", "fun": lambda z: unpack(z)[0].sum() + unpack(z)[1] - 1.0}]

    def obj(z):
        w, c = unpack(z)
        e = y - X @ w - c * cash_r
        return float(np.mean(e * e)) * 1e4

    res = minimize(obj, z0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    w, c = unpack(res.x)
    w = np.where(np.abs(w) < 1e-4, 0.0, w)
    return w, float(max(c, 0.0))


def tracking_error(y, X, w, c, cash_r) -> float:
    return float(np.std(y - X @ w - c * cash_r, ddof=1) * ANN)


def select(y: pd.Series, R: pd.DataFrame, cash_r: pd.Series, r: Restrictions, k: int = 3, shortlist: int = 40,
           min_gain: float = 0.05) -> dict:
    """Shortlist by correlation, greedy forward selection with refits, then a swap pass."""
    corr = R.corrwith(y)
    pool = list((corr.abs() if r.allow_short else corr).sort_values(ascending=False).index[:shortlist])
    yv, cv = y.values, cash_r.values

    def te_of(cols):
        X = R[cols].values
        w, c = fit(yv, X, cv, r)
        return tracking_error(yv, X, w, c, cv), w, c

    chosen, best = [], math.inf
    for _ in range(k):
        trial = [(te_of(chosen + [t])[0], t) for t in pool if t not in chosen]
        if not trial:
            break
        te, t = min(trial)
        if chosen and te > best * (1 - min_gain):
            break                    # the next ETF barely helps: keep the portfolio small (fewer trades)
        chosen, best = chosen + [t], te
    improved = True
    while improved:                  # swap pass: greedy can lock in an early pick
        improved = False
        for i in range(len(chosen)):
            for t in pool:
                if t in chosen:
                    continue
                cand = chosen[:i] + [t] + chosen[i + 1:]
                te = te_of(cand)[0]
                if te < best - 1e-9:
                    chosen, best, improved = cand, te, True
    te, w, c = te_of(chosen)
    return {"tickers": chosen, "weights": dict(zip(chosen, map(float, w))), "cash": c, "train_te": te,
            "shortlist": pool, "corr": {t: float(corr[t]) for t in pool}}


# ---------------------------------------------------------------- evaluation

def _path_returns(R: pd.DataFrame, weights: dict, cash: float, cash_r: pd.Series, buy_and_hold: bool) -> pd.Series:
    if not buy_and_hold:
        return sum(R[t] * w for t, w in weights.items()) + cash * cash_r
    growth = (1 + R[list(weights)]).cumprod()
    value = sum(growth[t] * w for t, w in weights.items()) + cash * (1 + cash_r).cumprod()
    return value.pct_change().fillna(value.iloc[0] - 1.0)


def metrics(y: pd.Series, p: pd.Series) -> dict:
    gap = y - p
    beta = float(np.cov(y, p, ddof=1)[0, 1] / np.var(p, ddof=1)) if np.var(p) > 0 else None
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else None
    tr_y, tr_p = float((1 + y).prod() - 1), float((1 + p).prod() - 1)
    return {"tracking_error": float(np.std(gap, ddof=1) * ANN), "correlation": corr, "r2": corr ** 2 if corr is not None else None,
            "beta": beta, "target_return": tr_y, "replica_return": tr_p, "return_gap": tr_p - tr_y}


def evaluate(target: pd.Series, R: pd.DataFrame, cash_r: pd.Series, r: Restrictions, k: int,
             holdings_weights: dict | None = None, end: int | None = None) -> dict:
    """One train/test split ending at row `end` (default: the last row)."""
    end = len(target) if end is None else end
    tr = slice(end - TEST_DAYS - TRAIN_DAYS, end - TEST_DAYS)
    te_ = slice(end - TEST_DAYS, end)
    y_tr, R_tr, c_tr = target.iloc[tr], R.iloc[tr], cash_r.iloc[tr]
    y_te, R_te, c_te = target.iloc[te_], R.iloc[te_], cash_r.iloc[te_]
    sel = select(y_tr, R_tr, c_tr, r, k=k)
    one = select(y_tr, R_tr, c_tr, r, k=1)
    out = {"train": [str(y_tr.index[0].date()), str(y_tr.index[-1].date())],
           "test": [str(y_te.index[0].date()), str(y_te.index[-1].date())], "selection": sel,
           "test_metrics": {
               "replica_buy_and_hold": metrics(y_te, _path_returns(R_te, sel["weights"], sel["cash"], c_te, True)),
               "replica_daily_rebalanced": metrics(y_te, _path_returns(R_te, sel["weights"], sel["cash"], c_te, False)),
               "best_single_etf": dict(metrics(y_te, _path_returns(R_te, one["weights"], one["cash"], c_te, True)),
                                       ticker=one["tickers"][0] if one["tickers"] else None)}}
    if "SPY" in R.columns:
        out["test_metrics"]["spy_alone"] = metrics(y_te, R_te["SPY"])
    if holdings_weights:
        hw = {t: v for t, v in holdings_weights.items() if t in R.columns}
        if hw:
            s = sum(hw.values())
            hw = {t: v / s for t, v in hw.items()}
            out["test_metrics"]["holdings_based"] = dict(metrics(y_te, _path_returns(R_te, hw, 0.0, c_te, True)),
                                                         weights=hw)
    return out


def stability(target: pd.Series, R: pd.DataFrame, cash_r: pd.Series, r: Restrictions, k: int, chosen: list) -> dict:
    """Refit on each half of the training window: if the chosen ETFs change a lot, the match is fragile."""
    end = len(target) - TEST_DAYS
    half = TRAIN_DAYS // 2
    sets = []
    for a, b in ((end - TRAIN_DAYS, end - half), (end - half, end)):
        sets.append(set(select(target.iloc[a:b], R.iloc[a:b], cash_r.iloc[a:b], r, k=k)["tickers"]))
    overlap = min(len(s & set(chosen)) for s in sets) / max(len(chosen), 1)
    return {"first_half": sorted(sets[0]), "second_half": sorted(sets[1]), "overlap_with_full": overlap,
            "stable": overlap >= (len(chosen) - 1) / max(len(chosen), 1)}


# ---------------------------------------------------------------- what exposure is lost

def lookthrough(weights: dict, target_names: dict, holdings: dict) -> dict:
    """The replica's direct weight in the target name(s), and the names that fill the rest."""
    agg: dict[str, float] = {}
    for etf, w in weights.items():
        h = holdings.get(etf)
        if h is None or h.empty:
            agg[f"{etf} (no holdings data)"] = agg.get(f"{etf} (no holdings data)", 0.0) + w
            continue
        for tk, hw in zip(h["ticker"].astype(str).str.upper(), h["weight"].astype(float)):
            agg[tk] = agg.get(tk, 0.0) + w * hw
    in_target = sum(agg.get(t, 0.0) * 1.0 for t in target_names)   # replica weight in the names you wanted
    wanted = sum(target_names.values())
    others = sorted(((t, v) for t, v in agg.items() if t not in target_names), key=lambda x: -abs(x[1]))[:6]
    return {"replica_weight_in_target": in_target, "target_weight": wanted,
            "share_of_target_held_directly": in_target / wanted if wanted else None,
            "top_other_names": [{"name": t, "weight": v} for t, v in others]}


def earnings_capture(target_r: pd.Series, replica_r: pd.Series, past: list[dict]) -> dict:
    """On the stock's past earnings sessions, how much of its move the replica captured."""
    rows = []
    for m in past or []:
        a, b = pd.Timestamp(m.get("from", "")), pd.Timestamp(m.get("to", ""))
        if pd.isna(a) or pd.isna(b) or b not in replica_r.index or a not in replica_r.index:
            continue
        rep = float((1 + replica_r.loc[(replica_r.index > a) & (replica_r.index <= b)]).prod() - 1)
        rows.append({"date": m["date"], "stock_move": m["move"], "replica_move": rep,
                     "captured": rep / m["move"] if abs(m["move"]) > 1e-4 else None})
    caps = [r["captured"] for r in rows if r["captured"] is not None]
    return {"reports": rows, "median_captured": float(np.median(caps)) if caps else None}


def factor_gaps(target_r: pd.Series, replica_r: pd.Series, closes: pd.DataFrame) -> dict:
    from .scenario import impact
    fac = impact.factor_returns(closes)
    L = impact.factor_loadings(pd.DataFrame({"target": target_r, "replica": replica_r}).dropna(), fac.dropna(how="all"))
    if L is None or L.empty or not {"target", "replica"} <= set(L.index):
        return {}
    return {f: {"target": float(L.at["target", f]), "replica": float(L.at["replica", f]),
                "gap": float(L.at["replica", f] - L.at["target", f])} for f in L.columns}


# ---------------------------------------------------------------- the whole pipeline

def _target_series(closes: pd.DataFrame, target: dict, divs: dict) -> pd.Series:
    w = pd.Series(target, dtype=float)
    w = w / w.sum()
    return sum(total_returns(closes[t], divs.get(t)) * float(wt) for t, wt in w.items())


def replicate(target, restrictions: Restrictions | None = None, k: int = 3, notional: float | None = None, data=None,
              today: date | None = None, shortlist: int = 40, log=print) -> dict:
    """target: a ticker ("NVDA") or a basket ({"NVDA": 0.5, "AMD": 0.5}). Returns the replica, its out-of-sample record
    against the baselines, rolling windows, stability, and what exposure is lost. Every number is computed here."""
    from .data import massive
    r = restrictions or Restrictions()
    tgt = {target.upper(): 1.0} if isinstance(target, str) else {t.upper(): float(v) for t, v in target.items()}
    today = today or date.today()
    closes = load_panel(today, log=log)["closes"]
    missing = [t for t in tgt if t not in closes.columns]
    if missing:
        raise ValueError(f"no price history for {missing}")
    need = TRAIN_DAYS + TEST_DAYS
    rolling_ends = [0, TEST_DAYS, 2 * TEST_DAYS]          # offsets from the end: main window + 2 earlier
    span = need + rolling_ends[-1]
    win = closes.iloc[-(span + 1):] if len(closes) > span else closes
    rolling_ends = [e for e in rolling_ends if len(win) - 1 - e >= need]
    cov = win.notna().mean()

    # universe: active ETFs with clean history, classified by name, filtered by the user's rules
    names = {e["ticker"]: e.get("name", "") for e in massive.etf_list()}
    info, excluded = {}, {}
    for t, nm in names.items():
        if t not in win.columns or cov.get(t, 0) < 0.95 or t in tgt:
            continue
        info[t] = classify_name(nm)
        ok, why = permitted(info[t], t, r)
        if not ok:
            excluded[why] = excluded.get(why, 0) + 1
    pool = [t for t in info if permitted(info[t], t, r)[0]]
    if not pool:
        raise ValueError("no permitted ETFs with enough history under these restrictions")

    # shortlist on price returns, then refine each shortlisted fund's kind from its holdings
    R_px = win[pool].pct_change().iloc[1:]
    y_px = sum(win[t].pct_change().iloc[1:] * w for t, w in tgt.items()) / sum(tgt.values())
    train_px = slice(len(y_px) - need, len(y_px) - TEST_DAYS)
    corr = R_px.iloc[train_px].corrwith(y_px.iloc[train_px])
    ranked = list((corr.abs() if r.allow_short else corr).dropna().sort_values(ascending=False).index)
    holdings, final = {}, []
    for t in ranked:
        if len(final) >= shortlist:
            break
        try:
            h = data.holdings(t, sectors=True) if data is not None else massive.etf_holdings(t)
        except Exception:  # noqa: BLE001 — no holdings: keep the name-based class
            h = None
        holdings[t] = h
        info[t] = classify_holdings(dict(info[t]), h) if h is not None and len(h) else info[t]
        ok, why = permitted(info[t], t, r)
        if ok:
            final.append(t)
        else:
            excluded[why + " (from holdings)"] = excluded.get(why + " (from holdings)", 0) + 1

    # total returns (dividends) for what's actually fitted and compared
    need_divs = set(final) | set(tgt) | {"SPY", CASH_PROXY}
    divs = {}
    for t in need_divs:
        try:
            divs[t] = massive.dividends(t)
        except Exception:  # noqa: BLE001
            divs[t] = []
    R = pd.DataFrame({t: total_returns(win[t], divs.get(t)) for t in final + (["SPY"] if "SPY" in win and "SPY" not in final else [])}).iloc[1:]
    y = _target_series(win, tgt, divs).iloc[1:]
    cash_r = (total_returns(win[CASH_PROXY], divs.get(CASH_PROXY)).iloc[1:] if CASH_PROXY in win.columns
              else pd.Series(FLAT_RF, index=y.index)).fillna(FLAT_RF)
    R, y = R.fillna(0.0), y.fillna(0.0)
    fit_cols = [c for c in R.columns if c in final]

    # holdings-based baseline: the permitted funds that own the most of the target
    hw = {}
    for t in final:
        h = holdings.get(t)
        if h is None or h.empty:
            continue
        tk = h["ticker"].astype(str).str.upper()
        hw[t] = float(sum(h.loc[tk == name, "weight"].astype(float).sum() * w for name, w in tgt.items()))
    hw = dict(sorted(((t, v) for t, v in hw.items() if v > 0), key=lambda x: -x[1])[:3])

    main = evaluate(y, R[fit_cols], cash_r, r, k, holdings_weights=hw)
    if "SPY" in R.columns and "spy_alone" not in main["test_metrics"]:
        te_ = slice(len(y) - TEST_DAYS, len(y))
        main["test_metrics"]["spy_alone"] = metrics(y.iloc[te_], R["SPY"].iloc[te_])
    rolling = []
    for off in rolling_ends[1:]:
        ev = evaluate(y, R[fit_cols], cash_r, r, k, end=len(y) - off)
        rolling.append({"test": ev["test"], "tickers": ev["selection"]["tickers"],
                        "replica_te": ev["test_metrics"]["replica_buy_and_hold"]["tracking_error"],
                        "best_single_te": ev["test_metrics"]["best_single_etf"]["tracking_error"]})
    sel = main["selection"]
    stab = stability(y, R[fit_cols], cash_r, r, k, sel["tickers"])

    # what exposure is lost
    weights = {t: w for t, w in sel["weights"].items() if abs(w) > 0}
    tr = slice(len(y) - need, len(y) - TEST_DAYS)
    replica_r = sum(R[t] * w for t, w in weights.items()) + sel["cash"] * cash_r
    full_h = {}
    for t in weights:
        try:
            full_h[t] = massive.etf_holdings(t)
        except Exception:  # noqa: BLE001
            full_h[t] = None
    lost = {"oos_r2": main["test_metrics"]["replica_buy_and_hold"]["r2"],
            "lookthrough": lookthrough(weights, tgt, full_h),
            "factor_gaps": factor_gaps(y.iloc[tr], replica_r.iloc[tr], win.iloc[1:].iloc[tr]),
            "expense_ratios": "not available on our data plan (ETF Global profiles return 403)",
            "positions": len(weights) + (1 if sel["cash"] > 1e-4 else 0)}
    if len(tgt) == 1:
        t0 = next(iter(tgt))
        try:
            past = massive.earnings(t0, today, win[t0]).get("past_moves") or []
        except Exception:  # noqa: BLE001
            past = []
        lost["earnings_capture"] = earnings_capture(y, replica_r, past)

    order = None
    if notional:
        last = win.iloc[-1]
        order = [{"ticker": t, "weight": w, "usd": notional * w,
                  "shares": round(notional * w / float(last[t]), 2) if last.get(t) else None, "price": float(last[t])}
                 for t, w in weights.items()] + ([{"ticker": "cash", "weight": sel["cash"], "usd": notional * sel["cash"]}]
                                                 if sel["cash"] > 1e-4 else [])
    return {"target": tgt, "restrictions": asdict(r), "k": k,
            "universe": {"etfs_with_history": len(info), "permitted": len(pool), "shortlist": len(final),
                         "excluded_by_reason": dict(sorted(excluded.items(), key=lambda x: -x[1]))},
            "replica": {"weights": weights, "cash": sel["cash"], "kinds": {t: info[t]["kind"] for t in weights},
                        "names": {t: names.get(t, "") for t in weights}},
            "order": order, "train": main["train"], "test": main["test"], "test_metrics": main["test_metrics"],
            "rolling": rolling, "stability": stab, "lost": lost,
            "method": "min tracking error of daily total returns; ETFs + cash; greedy + swap; OOS buy-and-hold"}
