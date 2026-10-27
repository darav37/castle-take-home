"""Grounding check (SPEC §1: the model never produces a number). Fails closed.

Every number in a reply must match a number in a tool result from this conversation (or the
user's own words), allowing only for display rounding: $4,100 / $4.1k / -12% / 0.12 / 12 bp.
Derived numbers count too: multiples ("9x", "9×", "9 times", "nine times", "twice", "double"),
ratios and percentages must appear in a tool output, computed in code; there is no fuzzy match.
Exempt: small integers (≤ 10) used as plain counts ("3 hedges", "2 weeks") and 4-digit years;
never multiples.

If a reply fails, the loop asks for one rewrite; if that still fails, `strip` removes every
sentence, bullet or table row that carries an unsupported number.
"""

from __future__ import annotations

import math
import re

NUM = re.compile(r"(?<![\w.])([-+−]?)\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kKmMbB](?![a-zA-Z])|%|bps?\b|x\b|×|times\b)?")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
SPELLED = re.compile(r"\b(" + "|".join(WORDS) + r")[\s-]+(?:times|fold)\b|\b(twice|double[ds]?|triple[ds]?|tripled|quadruple[ds]?|halves|half as)\b", re.I)
SPELLED_VAL = {"twice": 2, "double": 2, "doubles": 2, "doubled": 2, "triple": 3, "triples": 3, "tripled": 3,
               "quadruple": 4, "quadruples": 4, "quadrupled": 4, "halves": 0.5, "half as": 0.5}


def reply_numbers(text: str) -> list[dict]:
    out = []
    for m in NUM.finditer(text):
        sign, digits, suffix = m.group(1), m.group(2), (m.group(3) or "")
        v = float(digits.replace(",", ""))
        dec = len(digits.split(".")[1]) if "." in digits else 0
        s = suffix.lower()
        mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get(s[:1], 1.0) if s[:1] in ("k", "m", "b") and s not in ("bp", "bps") else 1.0
        if s in ("×", "times"):
            s = "x"
        out.append({"raw": m.group(0).strip(), "value": v * mult, "neg": sign in "-−" and sign != "", "suffix": s,
                    "decimals": dec, "mult": mult, "span": m.span()})
    for m in SPELLED.finditer(text):
        word = (m.group(1) or m.group(2)).lower()
        v = WORDS.get(word, SPELLED_VAL.get(word))
        if v is not None:
            out.append({"raw": m.group(0), "value": float(v), "neg": False, "suffix": "x", "decimals": 0, "mult": 1.0,
                        "span": m.span()})
    return out


def _leaves(obj, acc):
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _leaves(v, acc)
    elif isinstance(obj, str):
        for n in reply_numbers(obj):
            acc.append(n["value"])
    return acc


def _matches(n: dict, src: float) -> bool:
    cands = [abs(src)]
    if n["suffix"] == "%":
        cands.append(abs(src) * 100)  # fractions shown as percentages
    if n["suffix"].startswith("bp"):
        cands.append(abs(src) * 1e4)
    # only display rounding: half a unit in the last shown digit ($4.1k ↔ 4,132; 18% ↔ 0.1823)
    tol = 0.5 * 10 ** (-n["decimals"]) * n["mult"] + 1e-9
    return any(abs(c - n["value"]) <= tol for c in cands)


def check(reply: str, tool_results: list, user_text: str = "") -> dict:
    sources = _leaves(tool_results, [])
    user_vals = {n["value"] for n in reply_numbers(user_text)}
    flagged = []
    for n in reply_numbers(reply):
        v = n["value"]
        plain = n["suffix"] == ""
        if plain and v <= 10 and float(v).is_integer():
            continue  # counts in prose ("3 hedges"); never multiples
        if plain and 1900 <= v <= 2100 and float(v).is_integer():
            continue  # years
        if v in user_vals and n["suffix"] != "x":
            continue
        if not any(_matches(n, s) for s in sources):
            flagged.append(n["raw"])
    return {"ok": not flagged, "ungrounded": flagged}


def strip(reply: str, ungrounded: list[str]) -> str:
    """Fail closed: drop every table row, bullet or sentence that carries an unsupported number."""
    bad = [u for u in ungrounded if u]
    if not bad:
        return reply
    out = []
    for line in reply.splitlines():
        if not any(b in line for b in bad):
            out.append(line)
            continue
        if line.lstrip().startswith("|"):
            continue  # table row: drop it whole
        prefix = re.match(r"^\s*(?:[-*•]|\d+\.)\s+", line)
        body = line[prefix.end():] if prefix else line
        kept = [s for s in re.split(r"(?<=[.!?])\s+", body) if s and not any(b in s for b in bad)]
        if kept:
            out.append((prefix.group(0) if prefix else "") + " ".join(kept))
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- reply shape
# The card carries the numbers. A reply is a bold one-line headline + ≤ 3 short lines; no markdown tables.
# The trainer's rule: fail on a table or > 6 lines. "Short lines" is read as: no bullet lists and the whole reply
# ≤ MAX_CHARS (DECISIONS #97). A per-line cap was tried and dropped: it failed
# replies that were short overall and the enforced cut then removed the line that carried the point.
MAX_LINES, MAX_CHARS = 6, 600
TABLE = re.compile(r"^\s*\|.*\|")
BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def format_check(reply: str) -> dict:
    lines = [l for l in reply.splitlines() if l.strip()]
    problems = []
    if any(TABLE.match(l) for l in lines):
        problems.append("markdown table")
    if any(BULLET.match(l) for l in lines):
        problems.append("bullet list")
    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines (max {MAX_LINES})")
    if len(reply.strip()) > MAX_CHARS:
        problems.append(f"{len(reply.strip())} characters (max {MAX_CHARS})")
    return {"ok": not problems, "problems": problems}


def enforce_format(reply: str) -> str:
    """Fail closed on shape: drop table rows and bullet markers, then keep whole lines (headline first) within the
    limits."""
    lines = [BULLET.sub("", l).strip() for l in reply.splitlines() if l.strip() and not TABLE.match(l)]
    out, total = [], 0
    for l in lines[:MAX_LINES]:
        if total + len(l) + 1 > MAX_CHARS:
            continue
        out.append(l)
        total += len(l) + 1
    return "\n".join(out)


# ---------------------------------------------------------------- hedge numbers: one convention
CARD_FIELDS = ("cost", "residual_if_yes", "book_if_no")


def hedge_card_values(h: dict) -> list[float]:
    """The figures the hedge card shows: your book after each hedge, if YES / if NO, and each hedge's cost."""
    vals = [h.get("loss_if_yes"), h.get("pnl_if_no")]
    for rows in (h.get("rows") or [], (h.get("stress") or {}).get("rows") or []):
        vals += [r.get(k) for r in rows for k in CARD_FIELDS]
    if h.get("stress"):
        vals.append(h["stress"].get("loss_if_it_happens"))
    return [float(v) for v in vals if isinstance(v, (int, float))]


SIZING_KEYS = {"notional_usd", "contracts", "price", "strike", "legs", "underlying", "expiry", "horizon_sigma",
               "underlying_move_if_yes", "sizing", "exposure", "exposures", "p"}


def _outcomes_only(obj):
    """The hedge result without its sizing/context fields (position size, notional, strikes): those can legitimately
    appear in a reply ("per $100k of IWM") and are not hedge outcomes in another convention."""
    if isinstance(obj, dict):
        return {k: _outcomes_only(v) for k, v in obj.items() if k not in SIZING_KEYS}
    if isinstance(obj, list):
        return [_outcomes_only(v) for v in obj]
    return obj


def hedge_mismatches(reply: str, hedge: dict) -> list[str]:
    """Dollar figures in the reply that are hedge numbers in a different convention: they match some number in the
    hedge result but none the card shows."""
    card = hedge_card_values(hedge)
    every = _leaves(_outcomes_only(hedge), [])
    bad = []
    for n in reply_numbers(reply):
        if "$" not in n["raw"]:
            continue
        if not any(_matches(n, v) for v in card) and any(_matches(n, v) for v in every):
            bad.append(n["raw"])
    return bad
