"""Coverage test for the event universe (docs/event_pipeline/universe.md, DECISIONS #224): for each big 2026 event in the
library, would the universe have contained it BEFORE it happened?

  python scripts/universe_coverage.py            # DRY RUN: candidates found, LLM calls and cost; nothing is called
  python scripts/universe_coverage.py --confirm  # one capped judging call per batch of events

Per event (2026 unscheduled, severity major or extreme):
  market     Kalshi markets that were OPEN on the event day (opened before it, closed on or after it), retrieved by
             keyword from the in-scope series (the event title's names and places, plus the category's words); the LLM
             judges whether one was this event ("exact": its YES was this event or its immediate outcome) or the same
             developing story ("story": e.g. "US strikes Iran by June" before an Israeli strike on Iran), or neither
  news       a library event sharing a story name (a country, person, place or company; generic words dropped) in the
             14 days before: the story was already in the news feed
  book loss  the trainer's example book replayed through the event's measured factor moves (live.pack): the share of the
             2026 event-driven loss that was covered by a market or a developing story
Proposals (stream B) are not backtested: that would need the LLM to propose for past dates without hindsight.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey import config  # noqa: E402
from worsey.data import kalshi as K  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402
from worsey.scenario import discovery as SD  # noqa: E402

OUT = ROOT / "docs/pmrm/results"
CAT_WORDS = {"trade_tariffs": ["tariff", "trade"], "sanctions": ["sanction"], "conflict_war": ["war", "strike", "ceasefire", "military", "invade", "attack"],
             "elections_politics": ["election", "prime minister", "shutdown"], "regulatory_court": ["supreme court", "court", "ruling"],
             "disaster": ["hurricane", "earthquake"], "supply_shock": ["oil", "opec", "hormuz", "strait", "crude", "gas price"],
             "central_bank_unscheduled": ["rate", "fed ", "ecb", "central bank"], "health_pandemic": ["pandemic", "outbreak", "measles"],
             "financial_stress": ["default", "bank", "trillionaire", "bankrupt"]}
STOP = {"The", "United", "States", "U.S.", "US", "President", "Donald", "Trump", "Trump's", "A", "An", "In", "On", "After", "Prime", "Minister",
        "Secretary", "Executive", "Office", "Department", "Court", "Supreme", "House", "Oil", "Crude", "Brent", "West", "Texas", "Intermediate",
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "Navy", "Central", "Command"}
JUDGE = """For each event below, decide whether a prediction market that was OPEN on the event day (listed with it) was
already about it. Answer per event:
  "exact"  a market whose YES (or NO) resolved on this event or its immediate outcome, so its odds were a forecast of it
  "story"  no exact market, but one on the same developing story (same actors and issue), so the story was being priced
  "none"   neither
Return ONLY a JSON array: [{{"id": "...", "verdict": "exact|story|none", "market": "<ticker or null>", "why": "<10 words>"}}]

{events}"""


def events():
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    uns = [r for r in lib if r["kind"] == "unscheduled"]
    big = [r for r in uns if r["date"] >= "2026-01-01" and r.get("severity") in ("major", "extreme")]
    return big, uns


def names(r) -> set[str]:
    t = (r.get("title") or "") + " " + (r.get("description") or "") + " " + (r.get("region") or "")
    w = {x.strip(".,'\"()") for x in re.findall(r"\b[A-Z][a-zA-Z'.-]{2,}\b", t)}
    return {x for x in w if x not in STOP and len(x) > 2}


def candidates(r, series_list) -> list[dict]:
    ws = [w.lower() for w in names(r)]
    cw = CAT_WORDS.get(r["category"], [])
    scored = []
    for s in series_list:
        t = (s.get("title") or "").lower()
        hit = sum(2 for w in ws if re.search(rf"\b{re.escape(w)}\b", t)) + sum(1 for w in cw if w in t)
        if hit >= 1:
            scored.append((hit, s["ticker"]))
    scored.sort(reverse=True)
    day = r["date"]
    out = []
    for _, st in scored[:20]:
        try:
            ms = K.settled_markets(st) + [K.normalize_market(m, {"series_ticker": st}) for m in
                                          K._paged("/markets", {"series_ticker": st, "status": "open", "limit": 1000}, "markets", max_pages=5)]
        except Exception:  # noqa: BLE001
            continue
        for m in ms:
            if (m.get("open_date") or "9999") < day <= (m.get("close_date") or "0000"):
                out.append({"ticker": m["id"], "title": (m.get("title") or "")[:110], "sub": (m.get("subtitle") or "")[:40],
                            "open": m.get("open_date"), "close": m.get("close_date")})
    seen, uniq = set(), []
    for m in out:   # one row per event family keeps the list short: drop strike-ladder siblings with the same title
        k = m["title"]
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    return uniq[:15]


GENERIC = {"U.S", "BBC", "Reuters", "Bank", "Kingdom", "Union", "Global", "North", "South", "East", "West", "New", "News", "Middle",
           "American", "Federal", "Reserve", "International", "National", "Council", "Party", "York", "Group", "Times", "Street",
           "Journal", "Associated", "Press", "Bloomberg", "Government", "Ministry", "Minister", "Agency", "Commission", "Treasury",
           "Prices", "Price", "Market", "Markets", "Stocks", "Index", "Exchange", "Including", "Composite", "Dow", "Jones", "Nasdaq"}


def rare(ws: set[str], uns) -> set[str]:
    """Names that identify a story ('Iran', 'Hormuz', 'Starmer'), not generic title-case words ('U.S', 'Federal')."""
    return {w for w in ws if w not in GENERIC and not w.startswith("S&")}


def precursor(r, uns) -> str | None:
    d = datetime.fromisoformat(r["date"])
    ws = rare(names(r), uns)
    for u in uns:
        if u["id"] == r["id"]:
            continue
        du = datetime.fromisoformat(u["date"])
        if d - timedelta(days=14) <= du < d and ws & names(u):
            return f"{u['date']}: {(u.get('title') or '')[:80]}"
    return None


def book_moves(ids):
    """The trainer's example book through each event's measured factor moves (live.pack), in dollars."""
    import argparse as ap
    from worsey import discovery as D
    from worsey.agent import tools
    from worsey.agent.session import Session
    from worsey.cli import _data
    from worsey.planb import live
    s = Session(_data(ap.Namespace(synthetic=False, quote_tag=None)))
    tools.load_portfolio(s, holdings=[{"ticker": "PFE", "value_usd": 20000}, {"ticker": "WDC", "value_usd": 15000}, {"ticker": "TSLA", "value_usd": 20000},
                                      {"ticker": "URA", "value_usd": 15000}, {"ticker": "GLD", "value_usd": 20000}])
    ctx = D.context(s)
    p = live.pack()
    idx = {e["id"]: k for k, e in enumerate(p["events"])}
    return {i: float(np.asarray(p["_U"][idx[i]]) @ ctx["db"]) for i in ids if i in idx}, ctx["swing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    big, uns = events()
    ser = [s for s in K.series() if s.get("category") in SD.IN_SCOPE_CATEGORIES]
    rows = []
    for r in big:
        c = candidates(r, ser)
        rows.append({"id": r["id"], "date": r["date"], "category": r["category"], "severity": r["severity"],
                     "title": (r.get("title") or "")[:140], "candidates": c, "precursor": precursor(r, uns)})
        print(f"  {r['date']} {r['category']:<22} {len(c):>2} open markets | precursor: {bool(rows[-1]['precursor'])}", flush=True)
    from worsey.agent.llm import AnthropicLLM
    run = ICB.Run("universe_coverage", config.EXPERT_MODEL, AnthropicLLM() if a.confirm else None, confirmed=a.confirm)
    batches = [rows[i:i + 8] for i in range(0, len(rows), 8)]
    run.plan(len(batches), in_tok=7000, out_tok=900)
    if not a.confirm:
        return
    verdict = {}
    for b in batches:
        txt = json.dumps([{"id": r["id"], "date": r["date"], "event": r["title"],
                           "open_markets": [f"{m['ticker']} | {m['title']} {m['sub']} | {m['open']}..{m['close']}" for m in r["candidates"]]} for r in b],
                         ensure_ascii=False, indent=0)
        pr = JUDGE.format(events=txt)
        resp = run.call(ICB.key("universe_coverage", {"ids": [r["id"] for r in b]}, pr, config.EXPERT_MODEL),
                        model=config.EXPERT_MODEL, max_tokens=2000, messages=[{"role": "user", "content": pr}])
        t = "".join(x.get("text", "") for x in resp.get("content", []) if x.get("type") == "text")
        try:
            for it in json.loads(t[t.find("["):t.rfind("]") + 1]):
                verdict[it.get("id")] = it
        except (ValueError, json.JSONDecodeError):
            pass
    moves, swing = book_moves([r["id"] for r in rows])
    for r in rows:
        v = verdict.get(r["id"], {})
        r["market_verdict"] = v.get("verdict", "none") if r["candidates"] else "none"
        r["market"], r["why"] = v.get("market"), v.get("why")
        r["covered"] = r["market_verdict"] in ("exact", "story") or bool(r["precursor"])
        r["book_move_usd"] = moves.get(r["id"])
    n = len(rows)
    loss = [r for r in rows if (r["book_move_usd"] or 0) < 0]
    tot = sum(-r["book_move_usd"] for r in loss)
    summ = {"events": n,
            "exact_market": sum(r["market_verdict"] == "exact" for r in rows),
            "story_market": sum(r["market_verdict"] == "story" for r in rows),
            "news_precursor": sum(bool(r["precursor"]) for r in rows),
            "covered_any": sum(r["covered"] for r in rows),
            "covered_strict": sum(r["market_verdict"] == "exact" or bool(r["precursor"]) for r in rows),
            "book": "PFE 20k, WDC 15k, TSLA 20k, URA 15k, GLD 20k (options left out: no 2026 option marks)",
            "book_normal_2_day_swing_usd": swing,
            "book_loss_events": len(loss), "book_loss_total_usd": tot,
            "book_loss_share_covered": (sum(-r["book_move_usd"] for r in loss if r["covered"]) / tot) if tot else None,
            "book_loss_share_market": (sum(-r["book_move_usd"] for r in loss if r["market_verdict"] in ("exact", "story")) / tot) if tot else None,
            "book_loss_share_strict": (sum(-r["book_move_usd"] for r in loss if r["market_verdict"] == "exact" or r["precursor"]) / tot) if tot else None,
            "llm": run.summary()}
    (OUT / "universe_coverage.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=str), encoding="utf-8")
    lines = ["# Universe coverage: was each big 2026 event in the universe beforehand?", "",
             f"{n} big 2026 events (library, unscheduled, major or extreme). Generated by `scripts/universe_coverage.py`.", "",
             f"- an exact prediction market open on the day: **{summ['exact_market']}**",
             f"- a market on the same developing story: **{summ['story_market']}**",
             f"- the story already in the news feed in the 14 days before: **{summ['news_precursor']}**",
             f"- covered strictly (an exact market, or the story already in the news): **{summ['covered_strict']} of {n}**; "
             f"counting same-story markets too: {summ['covered_any']} of {n}",
             f"- The trainer's example book (stocks only): {len(loss)} of these events lost money, ${tot:,.0f} in total; share of that loss covered: "
             f"**{(summ['book_loss_share_strict'] or 0):.0%} strictly**, {(summ['book_loss_share_covered'] or 0):.0%} counting same-story markets", "",
             "Caveats: \"story\" verdicts are loose (some links are thin, e.g. travel or import-volume markets for a conflict); "
             "markets were searched across all of Kalshi's in-scope series, not only the build's liquid subset; the news check is a "
             "shared story name in the 14 days before, on library titles cut at ~100 characters, so it is a lower bound and says the story was running, not that its next step was forecast; "
             "stream B proposals are not backtested.", "",
             "| date | event | market | news before | book |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["date"]):
        mk = f"{r['market_verdict']}" + (f": {r['market']}" if r.get("market") and r["market_verdict"] != "none" else "")
        bm = "" if r["book_move_usd"] is None else f"{r['book_move_usd']:+,.0f}"
        lines.append(f"| {r['date']} | {r['title'][:80]} | {mk} | {'yes' if r['precursor'] else 'no'} | {bm} |")
    (OUT / "universe_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
