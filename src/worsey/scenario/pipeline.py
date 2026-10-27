"""Bottom-up scan, composed (METHODOLOGY §2, §4.1, §5.2; SPEC §3.3 Case 1, type A).

markets → Δp (16:00 ET, stale masked) → tier-1(a) eligibility (history + jumps) → β per
position (HAC, shrunk) → priced-in P&L → rank:
- events resolving inside the horizon by σ = |B|·√(p(1−p));
- events resolving after it (≤ 12 months) as *drift rows*, by σ_drift = |B|·σ(Δp over H) (C5).
Then Δp correlation, themes and the joint scenario over the in-horizon events.
The agent's scan_risks entry tool calls this.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from . import discovery, impact, joint, probability

# β is estimated for the book AND these, so hedges/explanations have index/haven moves
REFERENCE = ["SPY", "TLT", "GLD", "UUP", "USO", "IWM", "HYG"]  # channel assets: hedges and causal-story checks need their β
TRADING_DAYS_PER_YEAR = 252


def drift_sigma(dp: pd.Series, horizon_days: int) -> float:
    """σ(Δp over H): daily Δp std scaled by √(trading days in H) (METHODOLOGY §2.3)."""
    good = dp.dropna()
    if len(good) < 2:
        return float("nan")
    h = max(horizon_days * TRADING_DAYS_PER_YEAR / 365, 1)
    return float(good.std(ddof=1) * math.sqrt(h))


def _threshold(m: dict) -> float | None:
    st = m.get("strike_type")
    v = m.get("floor_strike") if st in ("greater", "greater_or_equal") else m.get("cap_strike") if st in ("less", "less_or_equal") else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _mid(m: dict) -> float | None:
    try:
        return probability.market_prob(m.get("bid"), m.get("ask"), m.get("last"))["p"]
    except ValueError:
        return None


def is_ladder(ms: list[dict], tol: float | None = None) -> tuple[bool, str]:
    """A ladder only if the markets share ONE numeric variable with parsed, ordered thresholds and prices
    roughly monotone across them. "Which countries…" or "will A / will B" are not ladders."""
    from .. import config
    tol = config.LADDER_MONOTONE_TOL if tol is None else tol
    types = {m.get("strike_type") for m in ms}
    if len(types) != 1 or not types <= {"greater", "greater_or_equal", "less", "less_or_equal"}:
        return False, f"strike types {sorted(map(str, types))} are not one numeric comparator"
    th = [_threshold(m) for m in ms]
    if any(x is None for x in th) or len(set(th)) != len(th):
        return False, "thresholds missing or duplicated"
    ps = [_mid(m) for m in ms]
    if any(p is None for p in ps):
        return False, "missing prices"
    order = sorted(range(len(ms)), key=lambda i: th[i])
    seq = [ps[i] for i in order]
    falling = next(iter(types)) in ("greater", "greater_or_equal")  # P(X ≥ k) falls as k rises
    steps = [b - a for a, b in zip(seq, seq[1:])]
    bad = [s for s in steps if (s > tol if falling else s < -tol)]
    if bad:
        return False, f"prices not monotone across thresholds (worst break {max(abs(x) for x in bad):.2f})"
    return True, f"ladder on {len(ms)} ordered thresholds"


def structural_groups(markets: list[dict]) -> tuple[list[list[str]], list[list[str]], list[dict]]:
    """Logical constraints from the venue, not from data (METHODOLOGY §8.2). Within one event:
    mutually_exclusive → one categorical group; a strict ladder (is_ladder) → one shared latent;
    anything else → separate events, correlated normally. Returns (ladders, exclusive, decisions)."""
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        if m.get("event_ticker"):
            by_event.setdefault(m["event_ticker"], []).append(m)
    ladders, exclusive, log = [], [], []
    for ev, ms in by_event.items():
        if len(ms) < 2:
            continue
        ids = [m["id"] for m in ms]
        if ms[0].get("mutually_exclusive"):
            exclusive.append(ids)
            log.append({"event": ev, "as": "exclusive", "why": "venue flags outcomes mutually exclusive"})
            continue
        ok, why = is_ladder(ms)
        if ok:
            ladders.append([m["id"] for m in sorted(ms, key=_threshold)])
        log.append({"event": ev, "as": "ladder" if ok else "separate", "why": why})
    return ladders, exclusive, log


def scan(values: pd.Series, asset_classes: pd.Series, closes: pd.DataFrame, markets: list[dict], histories: dict[str, pd.Series],
         today: date, horizon_days: int = 91, n: int = 6, k_shrink: float | None = None,
         templates: dict | None = None, loadings: pd.DataFrame | None = None, cached_betas: dict | None = None,
         structure: dict | None = None) -> dict:
    """Score every candidate, then rank:
    1. B̂: pooled tier 1(b) where the market belongs to a recurring series with enough releases, else tier 1(a).
    2. Structural prior B_prior (shock map × loadings): tier 2 if an approved template exists, else tier 3.
    3. Empirical Bayes across ALL candidates, shrinking toward the prior (structural.empirical_bayes).
    4. Per-position β_post = β_prior + w·(β̂ − β_prior), so the P&L split adds up to B_post.
    5. Rank above-divider rows first (strong evidence / tier-2 mechanism), then "weak evidence" rows.
    cached_betas: market → asset → {beta, se, …} from `worsey build` (tier 1(a) skipped here).
    structure: {loadings, priors, pooled_obs, standing} from `worsey build`.
    """
    from .. import config
    from . import structural
    st = structure or {}
    L = st.get("loadings") if st.get("loadings") is not None else loadings
    rets = closes.pct_change().dropna(how="all")
    book = float(values.sum())
    cand = discovery.bottom_up_filter(markets, today, horizon_days=horizon_days)
    scored, skipped, ps = [], [], {}
    meta = {m["id"]: m for m in cand}
    for m in cand:
        dp = impact.align_dp(histories[m["id"]], rets.index)
        try:
            q = probability.market_prob(m.get("bid"), m.get("ask"), m.get("last"))
        except ValueError:
            skipped.append({"event_id": m["id"], "title": m.get("title"), "reason": "no live quote"})
            continue
        # --- 1. B̂ and its standard error
        key = structural.pooled_key(m)
        pooled = structural.pooled_book_beta(st["pooled_obs"], values, key) if key and st.get("pooled_obs") is not None else None
        if pooled:
            tier, B_hat, se = "1b", pooled["B"], pooled["se"]
            beta_hat = {t_: x["beta"] for t_, x in pooled["per_asset"].items()}
            evidence_n = {"releases": pooled["releases"], "obs": pooled["n"]}
            elig = {"jump_days": None}
            # reference assets (hedges, channels) still need β̂ for this event: use the market's own history
            ref_b = impact.empirical_betas(rets[[c for c in rets.columns if c not in beta_hat]], dp, k=k_shrink) \
                if (cached_betas or {}).get(m["id"]) is None else {}
            all_b = {**{t_: impact.Beta(x["beta"], x["se"], pooled["n"], 1) for t_, x in pooled["per_asset"].items()},
                     **ref_b, **{a_: impact.Beta(x["beta"], x["se"], x["n"], 1) for a_, x in ((cached_betas or {}).get(m["id"]) or {}).items()
                                 if a_ not in pooled["per_asset"]}}
        else:
            cb = (cached_betas or {}).get(m["id"])
            if cb is not None:
                elig = {"eligible": True, "jump_days": m.get("jump_days")}
                all_b = {a_: impact.Beta(x["beta"], x["se"], x["n"], tier=1, shrunk=x["shrunk"]) for a_, x in cb.items() if a_ in rets.columns}
                missing = [a_ for a_ in values.index if a_ not in all_b]
                if missing:  # a typed-in holding outside the build universe: estimate it now (cheap, local)
                    all_b.update(impact.empirical_betas(rets[missing], dp, k=k_shrink))
            else:
                elig = discovery.tier1_eligible(dp)
                if not elig["eligible"]:
                    skipped.append({"event_id": m["id"], "title": m.get("title"), "reason": elig["reason"]})
                    continue
                all_b = impact.empirical_betas(rets, dp, k=k_shrink)
            chk = impact.portfolio_regression_check(rets, values, dp)
            tier, B_hat, se = "1a", chk["B"], chk["se"]
            beta_hat = {t_: x.beta for t_, x in all_b.items()}
            evidence_n = {"days": chk["n"]}
        # --- 2. structural prior
        tmpl = (templates or {}).get(m["id"]) or (templates or {}).get(m.get("type") or "")
        prior_tier, shock, sd = None, {}, {}
        if tmpl is not None:
            shock, sd, prior_tier = tmpl["shock"], tmpl.get("shock_se", {}), "2"
        elif (st.get("priors") or {}).get(m["id"]):
            shock, sd = structural.shock_from_buckets(st["priors"][m["id"]])
            prior_tier = "3"
        if L is not None and shock:
            B_prior, s_prior, beta_prior = structural.book_prior(values, L, shock, sd)
            beta_prior = {**{a_: 0.0 for a_ in beta_hat}, **beta_prior}
        else:
            B_prior, s_prior, beta_prior = 0.0, 0.0, {a_: 0.0 for a_ in beta_hat}
        ps[m["id"]] = histories[m["id"]].reindex(rets.index)  # stale/missing stay NaN for Δlogit
        scored.append({"m": m, "q": q, "dp": dp, "tier": tier, "B_hat": B_hat, "se": se, "beta_hat": beta_hat,
                       "all_b": all_b, "B_prior": B_prior, "s_prior": s_prior, "beta_prior": beta_prior,
                       "prior_tier": prior_tier, "prior_why": (st.get("priors") or {}).get(m["id"], {}).get("why"),
                       "evidence_n": evidence_n, "jump_days": elig.get("jump_days"), "p": q["p"]})
    # --- 3. empirical Bayes toward the structural prior, across every candidate
    eb = structural.empirical_bayes(scored)
    rows, drift_rows, betas_all = [], [], {}
    standing = set(st.get("standing") or [])
    for r in scored:
        m, q, w = r["m"], r["q"], r["weight"]
        post = {a_: r["beta_prior"].get(a_, 0.0) + w * (b_ - r["beta_prior"].get(a_, 0.0)) for a_, b_ in r["beta_hat"].items()}
        betas_all[m["id"]] = {a_: {"beta": x.beta, "se": x.se, "t": x.t, "n": x.n, "shrunk": x.shrunk,
                                   "used": post.get(a_, r["beta_prior"].get(a_, 0.0) + w * (x.beta - r["beta_prior"].get(a_, 0.0)))}
                              for a_, x in r["all_b"].items()}
        above, why = structural.above_divider({**r, "p": q["p"]}, book)
        flags = []
        if probability.longshot_flag(q["p"]):
            flags.append("longshot")
        if not above:
            flags.append("weak_evidence")
        if not m["resolves_in_horizon"]:
            flags.append("drift_only")
        if m["id"] in standing:
            flags.append("standing_calendar_only")
        market = {"venue": m.get("venue", "kalshi"), "ticker": m["id"], "match": "exact", "reason": "bottom-up market scan"}
        held = {t_: post.get(t_, 0.0) for t_ in values.index}
        row = impact.event_row(m["id"], q["p"], values, held, asset_classes, beta_tier=r["tier"], market=market,
                               beta_se=r["sd_post"], p_confidence=q["confidence"], resolves_in_horizon=m["resolves_in_horizon"],
                               flags=flags, title=m.get("title"), p_source=q["source"], close_date=m.get("close_date"),
                               jump_days=r["jump_days"], category=m.get("category"),
                               B_hat=r["B_hat"], B_hat_se=r["se"], B_prior=r["B_prior"], prior_sd=r["s_prior"],
                               prior_tier=r["prior_tier"], prior_why=r["prior_why"], B_post=r["B_post"],
                               shrink_weight=w, evidence=r["evidence"], above_divider=above, divider_reason=why,
                               evidence_n=r["evidence_n"])
        if m["resolves_in_horizon"]:
            rows.append(row)
        else:
            row["sigma_drift"] = abs(row["B"]) * drift_sigma(r["dp"], horizon_days)
            drift_rows.append(row)

    def order(rs, key):
        return sorted(rs, key=lambda x: (not x["above_divider"], -(x[key] if x[key] == x[key] else -1)))

    drift = order(drift_rows, "sigma_drift")[:n]
    out = {"events": [], "drift": drift, "skipped": skipped, "horizon_days": horizon_days,
           "ranking": {"method": "empirical Bayes toward the structural prior", "tau2": eb["tau2"], "candidates": eb["n"],
                       "above_divider": sum(r["above_divider"] for r in rows + drift_rows),
                       "rule": f"above the divider: |B_post|/sd ≥ {config.DIVIDER_EVIDENCE:g}, or a tier-2 mechanism with material exposure"},
           "betas": {r["event_id"]: betas_all[r["event_id"]] for r in drift}}
    if not rows:
        out["note"] = "no in-horizon markets passed the filters"
        return out
    ranked = order(rows, "sigma")[:n]
    ids = [r["event_id"] for r in ranked]
    Bs = pd.Series({r["event_id"]: r["B"] for r in ranked})
    dep = joint.correlation(pd.DataFrame(ps)[ids])
    corr, fallback = joint.fill_missing(dep["R"], Bs)
    ladders, exclusive, structure_log = structural_groups([meta[i] for i in ids])
    sim = (joint.simulate(pd.Series({r["event_id"]: r["p"] for r in ranked}), corr, Bs, ladders=ladders, exclusive=exclusive)
           if len(ids) > 1 else None)
    th = joint.themes(corr)
    for e in ranked:  # theme + correlation flags (METHODOLOGY §13)
        grp = next(g for g in th if e["event_id"] in g)
        e["theme"] = "+".join(grp) if len(grp) > 1 else None
        e["flags"] += [f"correlated_with:{o}" for o in grp if o != e["event_id"]]
    out.update({"events": ranked, "corr": corr.round(2).to_dict(), "themes": th, "joint": sim,
                "dependence": {"method": dep["method"], "shrinkage": dep["delta"], "overlap_days": dep["n"],
                               "fallback_pairs": fallback, "ladders": ladders, "exclusive": exclusive,
                               "structure": structure_log}})
    out["betas"].update({i: betas_all[i] for i in ids})
    return out
