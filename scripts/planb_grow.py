"""Grow the Plan B event library from Wikipedia's Current events portal (DECISIONS #161).

  python scripts/planb_grow.py --from 2004-01 --to 2026-10             # DRY RUN: months, calls, estimated cost
  python scripts/planb_grow.py --from 2004-01 --to 2026-10 --confirm   # one capped run (SPEC §6.1); rerun to continue

Per month, code collects the relevant news items (free, cached; src/worsey/planb/wikinews.py). One Claude call picks the
events that mattered that month BY NEWS PROMINENCE AND CATEGORY COVERAGE, never by market reaction, and returns only
compact labels (category, severity, anticipation, sectors, tickers, region, the six direction channels, factor signs).
Title, description, date and source links come from the Wikipedia item itself, so the model writes no facts. Every
answer is cached: a rerun or a stopped run continues where it left off at no cost. Each invocation is one run under
the per-run caps; the global daily cap (agent/budget.py) still applies.
Output: cache/pmrm/b2/wiki_events.json (merged into the library by `planb_b2.py responses`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from planb_b2 import DIRECTION_FIELDS, MODEL, SIGN_FACTORS, STORE, _parse, _text  # noqa: E402
from worsey.planb import library as LIB  # noqa: E402
from worsey.planb import wikinews as W  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

MAX_ITEMS, MAX_EVENTS = 300, 15
OUTFILE = STORE / "wiki_events.json"

PROMPT = """Below are numbered news items from one month ({month}), from Wikipedia's Current events portal.

Pick the events from this month that belong in a library of economically relevant events: up to {maxev}, fewer if the
month was quiet. Choose by how prominent the event was in the news and to cover the range of categories, NOT by how much
markets moved; include some that likely had little market effect. One pick per distinct event (use its first item).
Skip sport, crime of local interest only, and routine items.

For each pick return compact labels only (no prose):
- "i": the item number
- "cat": one of {cats}
- "sev": minor | moderate | major | extreme
- "ant": surprise | partly_expected | widely_expected
- "sec": list from: {sectors}
- "tic": US-listed tickers directly targeted ([] if none; never guess)
- "reg": country or region
- "dir": {{{dirfields}}}
- "sig": expected sign of the move right after, for {factors}, as a string like "-,+,0,..." in that order
  (market = US stocks; rates = Treasury yields, "+" = yields up; dollar = the US dollar; oil and gold = their prices;
  credit = high-yield bonds minus Treasuries, "+" = credit spreads tighten)

Return ONLY a JSON array.

Items:
{items}"""


def months(a: str, b: str) -> list[tuple[int, int]]:
    y, m = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    out = []
    while (y, m) <= (y2, m2):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def select_items(items: list[dict]) -> list[dict]:
    """Deterministic cap: business/economy items first, then the rest in date order."""
    biz = [x for x in items if "business" in (x.get("section") or "").lower() or "econom" in (x.get("section") or "").lower()]
    rest = [x for x in items if x not in biz]
    return (biz + rest)[:MAX_ITEMS]


def build_event(it: dict, lab: dict, month: str, i: int) -> dict | None:
    d = date.fromisoformat(it["day"])
    cat = lab.get("cat") if lab.get("cat") in LIB.CATEGORIES else None
    if cat is None:
        return None
    dirs = lab.get("dir") or {}
    signs = [s.strip() for s in str(lab.get("sig") or "").split(",")]
    portal = f"https://en.wikipedia.org/wiki/Portal:Current_events/{d:%B}_{d.year}#{d.year}_{d:%B}_{d.day}"
    text = it["text"]
    return {
        "id": f"w{month.replace('-', '')}{i:03d}", "origin": "wikipedia_portal_monthly",
        "title": text.split(": ", 1)[-1][:120], "category": cat, "date_first_report": d.isoformat(),
        "time_first_report_et": None, "actors": [], "region": lab.get("reg"),
        "targeted_sectors": [s for s in (lab.get("sec") or []) if s in LIB.SECTORS],
        "targeted_tickers": [t for t in (lab.get("tic") or []) if isinstance(t, str) and t.isupper() and 1 <= len(t) <= 6],
        "commodities": [], "severity": lab.get("sev") if lab.get("sev") in ("minor", "moderate", "major", "extreme") else None,
        "anticipation": lab.get("ant") if lab.get("ant") in ("surprise", "partly_expected", "widely_expected") else None,
        "prediction_market": None, "description": text,
        "sources": [{"url": u, "publisher": None, "date": d.isoformat()} for u in it["links"]] + [{"url": portal, "publisher": "Wikipedia", "date": d.isoformat()}],
        "sourcing": "wikipedia_item", "sources_ok": None, "wiki_match": {"day": d.isoformat(), "matched": ["source item"]},
        "direction": {**{f: (dirs.get(f) if dirs.get(f) in vals else None) for f, vals in DIRECTION_FIELDS.items()},
                      "factor_signs": {f: {"+": 1, "-": -1, "0": 0}.get(signs[k] if k < len(signs) else "", None)
                                       for k, f in enumerate(SIGN_FACTORS)}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="a", default="2004-01")
    ap.add_argument("--to", dest="b", default="2026-10")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    ms = months(args.a, args.b)
    from worsey.agent.llm import AnthropicLLM
    run = ICB.Run("planb_grow_wikipedia", MODEL, AnthropicLLM() if args.confirm else None, confirmed=args.confirm)
    run.plan(len(ms), in_tok=22000, out_tok=1800)
    cats = " | ".join(LIB.CATEGORIES)
    dirfields = ", ".join(f'"{k}": "{"|".join(v)}"' for k, v in DIRECTION_FIELDS.items())
    have = json.loads(OUTFILE.read_text(encoding="utf-8")) if OUTFILE.exists() else {}
    done_now = 0
    for (y, m) in ms:
        key_m = f"{y}-{m:02d}"
        items = select_items(W.month_items(y, m))
        if not items:
            continue
        listing = "\n".join(f"[{k}] {it['day']} | {it['text'][:240]}" for k, it in enumerate(items))
        prompt = PROMPT.format(month=key_m, maxev=MAX_EVENTS, cats=cats, sectors=", ".join(LIB.SECTORS),
                               dirfields=dirfields, factors=", ".join(SIGN_FACTORS), items=listing)
        k = ICB.key("planb_grow_wikipedia", {"month": key_m, "n_items": len(items)}, prompt, MODEL)
        if not args.confirm and run.cached(k) is None:
            continue
        try:
            resp = run.call(k, model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as e:
            print(f"stopped at {key_m}: {e}", flush=True)
            break
        evs = []
        for lab in _parse(_text(resp)):
            try:
                i = int(lab.get("i"))
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(items):
                ev = build_event(items[i], lab, key_m, i)
                if ev:
                    evs.append(ev)
        have[key_m] = evs
        done_now += 1
        if done_now % 10 == 0:
            OUTFILE.write_text(json.dumps(have, indent=0), encoding="utf-8")
            print(f"  {key_m}: {len(evs)} events; run so far {run.summary()['calls']} calls, ${run.summary()['est_usd']}", flush=True)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(have, indent=0), encoding="utf-8")
    tot = sum(len(v) for v in have.values())
    print(json.dumps(run.summary() | {"months_done": len(have), "events_total": tot}))


if __name__ == "__main__":
    main()
