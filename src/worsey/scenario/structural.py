"""Structure for the ranking: pooled tier 1(b), the structural prior, and empirical Bayes.

Why: on real data, tier 1(a) alone has almost no power at the book level (F22: 1% of rows with |t| ≥ 2,
cross-sectional τ² ≈ 0), so a σ ranking of raw betas is mostly winner's curse.

1. Tier 1(b), pooled recurring series (METHODOLOGY §4.1b): each settled Fed/CPI/payrolls market is one
   observation, surprise = outcome − p at the prior day's 16:00 ET, reaction = close-before → release-day close.
   Book-level B̂ by regression of the book's window return on surprise, errors clustered by release.
2. Structural prior (METHODOLOGY §4.2–4.3): B_prior = Σᵢ Vᵢ·(Lᵢ·s), where L are factor loadings and s is the
   event's shock map. s comes from an approved tier-2 template when one exists, else from Claude's tier-3
   sign + size bucket per factor (numbers from config). Prior sd = 100% of each bucket.
3. Empirical Bayes toward the prior: r = B̂ − B_prior; τ² = max(0, mean(r² − se² − s_p²)) (method of moments);
   with v = τ² + s_p²: B_post = B_prior + r·v/(v + se²), sd_post = √(v·se²/(v + se²)).
   the trainer's formula is the s_p = 0 case; s_p (the prior's own uncertainty) keeps the evidence test from
   becoming degenerate when τ² = 0 (flagged in DECISIONS).
4. Divider: a row ranks above it only with strong evidence (|B_post|/sd_post ≥ 2) or a tier-2 mechanism with
   material exposure; everything else is "weak evidence", below.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import config

FACTORS = ["SPY", "TLT", "UUP", "USO", "GLD", "CREDIT", "VIXY"]
FACTOR_LABELS = {"SPY": "US stocks", "TLT": "long Treasuries (up = yields down)", "UUP": "US dollar", "USO": "oil",
                 "GLD": "gold", "CREDIT": "high-yield vs Treasuries (up = spreads tighter)", "VIXY": "volatility"}


# ---------------------------------------------------------------- pooled tier 1(b)

def pooled_key(market: dict) -> str | None:
    """Which pooled series a market belongs to: Fed by outcome (H0 = hold, C25 = cut 25…), CPI/payrolls by series."""
    mid, ser = market.get("id", ""), market.get("series_ticker") or ""
    for s in config.POOLED_SERIES:
        if ser == s or mid.startswith(s + "-") or mid.startswith(s.replace("KX", "", 1) + "-"):
            if s == "KXFEDDECISION":
                return f"{s}:{mid.rsplit('-', 1)[-1]}"
            return f"{s}:above"
    return None


_POOL_MEMO: dict = {}


def pooled_book_beta(obs: pd.DataFrame, values: pd.Series, key: str) -> dict | None:
    """Memoized per (observations, key, book): every CPI threshold market shares one pooled regression."""
    memo_key = (id(obs), key, tuple(sorted(values.round(2).items())))
    if memo_key not in _POOL_MEMO:
        _POOL_MEMO[memo_key] = _pooled_book_beta(obs, values, key)
    return _POOL_MEMO[memo_key]


def _pooled_book_beta(obs: pd.DataFrame, values: pd.Series, key: str) -> dict | None:
    """Book-level tier-1(b) β: regress Σ Vᵢ·rᵢ over each release window on the surprise; cluster by release.
    obs columns: key, release, surprise, and one return column per asset."""
    d = obs[obs["key"] == key]
    cols = [t for t in values.index if t in d.columns]
    if len(d) < 5 or d["release"].nunique() < config.POOLED_MIN_RELEASES or len(cols) < len(values):
        return None
    y = (d[cols] * values[cols]).sum(axis=1)
    res = sm.OLS(y, d[["surprise"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["release"])[0]})
    per_asset = {}
    for t in cols:
        r = sm.OLS(d[t], d[["surprise"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["release"])[0]})
        per_asset[t] = {"beta": float(r.params["surprise"]), "se": float(r.bse["surprise"])}
    return {"B": float(res.params["surprise"]), "se": float(res.bse["surprise"]), "n": int(len(d)),
            "releases": int(d["release"].nunique()), "per_asset": per_asset}


# ---------------------------------------------------------------- structural prior

def shock_from_buckets(signs: dict) -> tuple[dict, dict]:
    """{factor: [sign, bucket]} → (shock per +1.00 Δp, sd) using the tier-3 bucket sizes; sd = 100% of bucket."""
    s, sd = {}, {}
    for f in FACTORS:
        sign, bucket = (signs.get(f) or [0, "none"])
        m = config.PRIOR_BUCKETS.get(bucket, 0.0)
        s[f], sd[f] = float(np.sign(sign)) * m, m * config.PRIOR_SD_FRACTION
    return s, sd


def asset_prior_betas(loadings: pd.DataFrame, shock: dict) -> dict:
    """βᵢ_prior = Lᵢ·s; a factor proxy held directly takes its own shock (SPY, TLT, …; HYG via CREDIT)."""
    out = {}
    for a in loadings.index:
        if a in shock:
            out[a] = shock[a]
        else:
            out[a] = float(sum(loadings.at[a, f] * shock.get(f, 0.0) for f in loadings.columns if f in shock))
    return out


def book_prior(values: pd.Series, loadings: pd.DataFrame, shock: dict, sd: dict) -> tuple[float, float, dict]:
    """(B_prior, s_prior, per-asset prior β). Assets outside the loadings get a zero prior."""
    beta = asset_prior_betas(loadings, shock)
    for t in values.index:  # a factor proxy held directly takes its own shock even if it isn't in the loadings table
        if t in FACTORS:
            beta[t] = shock.get(t, 0.0)
    B = float(sum(values[t] * beta.get(t, 0.0) for t in values.index))
    # book exposure to each factor (a held proxy counts 1 on its own factor), times that factor's prior sd
    expo = {f: 0.0 for f in FACTORS}
    for t, v in values.items():
        if t in FACTORS:
            expo[t] += v
        elif t in loadings.index:
            for f in loadings.columns:
                if f in expo:
                    expo[f] += v * float(loadings.at[t, f])
    s_p = math.sqrt(sum((expo[f] * sd.get(f, 0.0)) ** 2 for f in FACTORS))
    return B, s_p, beta


# ---------------------------------------------------------------- empirical Bayes + divider

def empirical_bayes(rows: list[dict]) -> dict:
    """In place: each row gets B_post, sd_post, weight, evidence; returns {tau2, n}.
    rows need B_hat, se, B_prior, s_prior."""
    use = [r for r in rows if r.get("se") and math.isfinite(r["se"])]
    if not use:
        return {"tau2": 0.0, "n": 0}
    m = np.mean([(r["B_hat"] - r["B_prior"]) ** 2 - r["se"] ** 2 - r["s_prior"] ** 2 for r in use])
    tau2 = float(max(0.0, m))
    for r in rows:
        se2 = r["se"] ** 2 if r.get("se") and math.isfinite(r["se"]) else math.inf
        v = tau2 + r["s_prior"] ** 2
        w = v / (v + se2) if math.isfinite(se2) and (v + se2) > 0 else 0.0
        r["weight"] = w
        r["B_post"] = r["B_prior"] + w * (r["B_hat"] - r["B_prior"])
        r["sd_post"] = math.sqrt(v * se2 / (v + se2)) if math.isfinite(se2) and (v + se2) > 0 else math.sqrt(v)
        r["evidence"] = abs(r["B_post"]) / r["sd_post"] if r["sd_post"] > 0 else 0.0
    return {"tau2": tau2, "n": len(use)}


def above_divider(row: dict, book_value: float) -> tuple[bool, str]:
    if row.get("evidence", 0.0) >= config.DIVIDER_EVIDENCE:
        return True, f"strong evidence (|B|/sd = {row['evidence']:.1f})"
    material = abs(row.get("B_prior", 0.0)) * math.sqrt(max(row.get("p", 0.5) * (1 - row.get("p", 0.5)), 0.0)) \
        >= config.DIVIDER_MATERIAL_PCT * book_value
    if row.get("prior_tier") == "2" and material:
        return True, "tier-2 mechanism with material exposure"
    return False, "weak evidence"


# ---------------------------------------------------------------- tier-3 shock maps from Claude (build time)

SHOCK_PROMPT = """You map prediction-market events to the direction of factor moves if the event resolves YES.
Factors: {factors}.
For each market, give each factor a sign (+1, -1, 0) and a size bucket (none, small, medium, large) for the move
if YES, versus NO, judged on economic mechanism only. Most events move most factors not at all: say "none" unless
there is a clear channel. Never give numbers. Reply with JSON only:
{{"MARKET_ID": {{"SPY": [sign, "bucket"], ...all factors..., "why": "one short line"}}, ...}}

Markets:
{markets}"""


def ask_shock_maps(llm, markets: list[dict], model: str) -> dict:
    """One call per batch; validated: unknown factors/buckets are dropped to (0, none)."""
    lines = "\n".join(f"- {m['id']}: {m.get('title')} ({m.get('subtitle') or ''})" for m in markets)
    text = SHOCK_PROMPT.format(factors=", ".join(f"{f} = {FACTOR_LABELS[f]}" for f in FACTORS), markets=lines)
    resp = llm.create(model=model, max_tokens=8000, messages=[{"role": "user", "content": text}],
                      output_config={"effort": "low"})
    raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    data = {}
    if m:
        try:
            data = __import__("json").loads(m.group(0))
        except ValueError:
            data = {}
    out = {}
    for mk in markets:
        d = data.get(mk["id"]) or {}
        clean = {}
        for f in FACTORS:
            v = d.get(f)
            ok = isinstance(v, list) and len(v) == 2 and v[0] in (-1, 0, 1) and v[1] in config.PRIOR_BUCKETS
            clean[f] = [int(v[0]), v[1]] if ok else [0, "none"]
        clean["why"] = str(d.get("why", ""))[:200]
        out[mk["id"]] = clean
    return out


# ---------------------------------------------------------------- precomputed event notes (latency, Oct 16)

NOTES_PROMPT = """For each prediction market below, write a short note a retail investor can read on a phone.
No numbers of any kind (no probabilities, prices, percentages, dates beyond the title's). Reply with JSON only:
{{"MARKET_ID": {{"what": "one plain line on what resolves it", "channels": "one line: which markets it would move and why",
"caveats": "one line: what the market does NOT capture, or resolution quirks"}}, ...}}

Markets:
{markets}"""


def ask_event_notes(llm, markets: list[dict], model: str) -> dict:
    """Portfolio-independent commentary per event, written once in `worsey build` (SPEC §11b rule 1)."""
    import json as _json
    lines = "\n".join(f"- {m['id']}: {m.get('title')} ({m.get('subtitle') or ''}). Rules: {(m.get('rules_primary') or '')[:300]}"
                      for m in markets)
    resp = llm.create(model=model, max_tokens=8000, messages=[{"role": "user", "content": NOTES_PROMPT.format(markets=lines)}],
                      output_config={"effort": "low"})
    raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    mm = re.search(r"\{.*\}", raw, re.S)
    try:
        data = _json.loads(mm.group(0)) if mm else {}
    except ValueError:
        data = {}
    out = {}
    for mk in markets:
        d = data.get(mk["id"]) or {}
        note = {k: str(d.get(k, ""))[:240] for k in ("what", "channels", "caveats")}
        if any(ch.isdigit() for v in note.values() for ch in v):  # notes must carry no numbers (grounding)
            note = {k: re.sub(r"\S*\d\S*", "", v).strip() for k, v in note.items()}
        out[mk["id"]] = note
    return out


NAMES_PROMPT = """For each prediction market below, write a short display name for a phone screen: at most 32 characters,
plain words, no question mark, no "Will". Keep the threshold and the month if the title has them, copied exactly
(e.g. "Gas above $4.60 by Dec 31", "Nov CPI above 0.4%", "Emergency Fed cut in 2026"). Use no number that is not in
the title. Reply with JSON only: {{"MARKET_ID": "short name", ...}}

Markets:
{markets}"""

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def clean_title(title: str) -> str:
    """Deterministic fallback name: the full title, de-markdowned, without "Will" and "?" (never truncated)."""
    t = re.sub(r"\s+", " ", str(title or "").replace("**", "")).strip()
    t = re.sub(r"^(will|is|are|does|do)\s+(the\s+)?", "", t, flags=re.I).rstrip("?").strip()
    return t[:1].upper() + t[1:]


def valid_short_name(name: str, title: str, subtitle: str = "") -> bool:
    """A short name is accepted only if it is short and every number in it appears in the market's own text."""
    if not name or len(name) > 40:
        return False
    src = {n.replace(",", "") for n in _NUM.findall(f"{title} {subtitle}")}
    return all(n.replace(",", "") in src for n in _NUM.findall(name))


def ask_short_names(llm, markets: list[dict], model: str) -> dict:
    """Short display names per market, written once in `worsey build`; validated in code."""
    import json as _json
    lines = "\n".join(f"- {m['id']}: {m.get('title')} ({m.get('subtitle') or ''})" for m in markets)
    resp = llm.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": NAMES_PROMPT.format(markets=lines)}],
                      output_config={"effort": "low"})
    raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    mm = re.search(r"\{.*\}", raw, re.S)
    try:
        data = _json.loads(mm.group(0)) if mm else {}
    except ValueError:
        data = {}
    out = {}
    for mk in markets:
        name = str(data.get(mk["id"]) or "").strip().rstrip("?")
        if valid_short_name(name, mk.get("title") or "", mk.get("subtitle") or ""):
            out[mk["id"]] = name
    return out
