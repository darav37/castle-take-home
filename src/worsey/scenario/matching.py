"""Event → market matching (SPEC §3.2 step 3).

Split of labour: Claude parses the user's event and each candidate market into
EventSchema; this module retrieves candidates (keywords) and scores the mismatch with
fixed rules, so every match decision is reproducible and logged with reasons.

Outcomes: exact | derived_strike | derived_time | no_fit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

STOP = {"the", "a", "an", "of", "by", "in", "on", "to", "will", "be", "or", "and", "for", "at", "than", "more", "less", "before", "after", "is"}


@dataclass
class EventSchema:
    variable: str                 # canonical, e.g. "us_avg_tariff_rate", "fed_funds_upper", "us_cpi_yoy"
    comparator: str               # ">=", ">", "<=", "<", "=="
    threshold: float | None
    unit: str = ""
    deadline: date | None = None
    jurisdiction: str = ""
    resolution_source: str = ""
    why: str = ""
    ladder: list[dict] = field(default_factory=list)  # sibling strikes [{threshold, p}] for derived_strike
    timing: str = "hazard"        # "hazard" (can happen any day) | "release" (only on decision/release dates, e.g. FOMC)
    decision_dates: list[date] = field(default_factory=list)  # for timing="release": the scheduled dates

    def to_dict(self):
        d = asdict(self)
        d["deadline"] = self.deadline.isoformat() if self.deadline else None
        d["decision_dates"] = [x.isoformat() for x in self.decision_dates]
        return d


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1}


def keyword_candidates(query: str, markets: list[dict], n: int = 10, text_keys=("title", "subtitle", "rules_primary")) -> list[dict]:
    """Rank markets by token overlap (Jaccard) with the query. Embeddings are Tier C."""
    q = tokens(query)
    scored = []
    for m in markets:
        mt = tokens(" ".join(str(m.get(k, "")) for k in text_keys))
        if not mt:
            continue
        s = len(q & mt) / len(q | mt)
        if s > 0:
            scored.append((s, m))
    scored.sort(key=lambda x: -x[0])
    return [dict(m, _score=round(s, 3)) for s, m in scored[:n]]


_DIR = {">=": 1, ">": 1, "<=": -1, "<": -1, "==": 0}


def score_match(want: EventSchema, cand: EventSchema, deadline_tol_days: int = 3,
                max_time_ratio: float = 3.0, today: date | None = None) -> dict:
    """Decide exact / derived / no_fit, with reasons. Rules, in order:

    1. variable and jurisdiction must agree, or no fit.
    2. comparator direction must agree (a 'below' market can't price an 'above' event,
       except as 1 − p for the complement, which we treat as exact with a flag).
    3. threshold: equal → ok; different → derived_strike if the candidate carries a ladder
       spanning the wanted threshold, else no fit.
    4. deadline: within tolerance → ok; different → derived_time (constant hazard) if the
       ratio is ≤ max_time_ratio, else no fit — stretching a hazard 10× is a guess.
    5. resolution source differences are flagged, never silently ignored.
    """
    reasons, flags = [], []
    if want.variable != cand.variable:
        return {"decision": "no_fit", "reasons": [f"different variable: {cand.variable} vs {want.variable}"], "flags": []}
    if want.jurisdiction and cand.jurisdiction and want.jurisdiction != cand.jurisdiction:
        return {"decision": "no_fit", "reasons": [f"different jurisdiction: {cand.jurisdiction} vs {want.jurisdiction}"], "flags": []}

    complement = False
    dw, dc = _DIR.get(want.comparator), _DIR.get(cand.comparator)
    if dw != dc:
        if dw and dc and dw == -dc and want.threshold == cand.threshold:
            complement = True
            flags.append("complement: p = 1 − market p")
        else:
            return {"decision": "no_fit", "reasons": [f"comparator {cand.comparator} can't price {want.comparator}"], "flags": []}

    decision = "exact"
    if want.threshold != cand.threshold:
        ts = [r["threshold"] for r in cand.ladder]
        if ts and want.threshold is not None and min(ts) <= want.threshold <= max(ts):
            decision = "derived_strike"
            reasons.append(f"threshold {want.threshold}{want.unit} interpolated from ladder {sorted(ts)}")
        else:
            return {"decision": "no_fit", "reasons": [f"threshold {cand.threshold} ≠ {want.threshold} and no ladder spans it"], "flags": flags}

    if want.deadline and cand.deadline and abs((want.deadline - cand.deadline).days) > deadline_tol_days:
        today = today or date.today()
        if decision == "derived_strike":
            flags.append("derived on both strike and time")
        if want.timing == "release" or cand.timing == "release":
            # METHODOLOGY §6: release-driven events move only on decision dates, so count those, not days
            dates = sorted(set(want.decision_dates) | set(cand.decision_dates))
            n = sum(1 for x in dates if today < x <= cand.deadline)
            n_new = sum(1 for x in dates if today < x <= want.deadline)
            if not dates:
                return {"decision": "no_fit", "reasons": ["release-driven event but no decision dates given"], "flags": flags}
            if n == 0:
                return {"decision": "no_fit", "reasons": [f"no decision date before the market's deadline {cand.deadline}"], "flags": flags}
            decision = "derived_time" if decision == "exact" else decision
            reasons.append(f"deadline {cand.deadline} → {want.deadline} by meeting count ({n} → {n_new} decision dates)")
            return {"decision": decision, "reasons": reasons, "flags": flags, "complement": complement,
                    "time_rule": {"kind": "meetings", "n": n, "n_new": n_new}}
        T, Tn = (cand.deadline - today).days, (want.deadline - today).days
        if T <= 0 or Tn <= 0:
            return {"decision": "no_fit", "reasons": ["deadline already passed"], "flags": flags}
        ratio = max(T, Tn) / min(T, Tn)
        if ratio > max_time_ratio:
            return {"decision": "no_fit", "reasons": [f"deadline gap too large to rescale ({ratio:.1f}× > {max_time_ratio}×)"], "flags": flags}
        decision = "derived_time" if decision == "exact" else decision
        reasons.append(f"deadline {cand.deadline} → {want.deadline} via constant hazard (T′/T = {Tn}/{T} days)")
        return {"decision": decision, "reasons": reasons, "flags": flags, "complement": complement,
                "time_rule": {"kind": "hazard", "T": T, "T_new": Tn}}

    if want.resolution_source and cand.resolution_source and want.resolution_source != cand.resolution_source:
        flags.append(f"resolution source differs: {cand.resolution_source} vs {want.resolution_source}")
    if decision == "exact":
        reasons.append("same variable, threshold, deadline")
    return {"decision": decision, "reasons": reasons, "flags": flags, "complement": complement}
