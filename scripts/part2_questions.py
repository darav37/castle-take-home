"""Part 2, step 4: the forecasting question set (docs/event_pipeline/probabilities.md; DECISIONS #229). Code only, $0.

  python scripts/part2_questions.py     -> cache/pmrm/part2/questions.json, docs/pmrm/results/part2_questions.md

Rules (frozen before any forecast is made):
  window     markets that resolved between 2026-07-01 and 2026-10-23 (after the model's knowledge, ~June 2026)
  forecast   date T, a calendar date: Jul 1, Jul 15, Aug 1, Aug 15 or Sep 1. For each market, among the dates when it was
  date       open and at least 14 days before it closed, one is picked by a hash of the market id. (Not "N days before
             close": close is when a market resolves, so that would make T depend on the outcome; see #227.)
  scope      Kalshi in-scope series (no daily/weekly/intraday), volume >= 500; Polymarket topic tags, volume >= $10k;
             price-level markets on an asset dropped (the build's reflexive filter); one question per event (hash)
  priced     the market's price at T between 2% and 98% (known at T; nearly-decided questions test nothing)
  benchmark  that price is kept hidden from every forecaster arm and used only as the benchmark
"""

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part2_longshot as L  # noqa: E402
from worsey.data import kalshi as K  # noqa: E402
from worsey.data import polymarket as P  # noqa: E402
from worsey.data.http import get_json  # noqa: E402
from worsey.scenario import discovery as SD  # noqa: E402

W0, W1 = "2026-07-01", "2026-10-23"
TS = ["2026-07-01", "2026-07-15", "2026-08-01", "2026-08-15", "2026-09-01"]
OUT = ROOT / "cache/pmrm/part2/questions.json"


def _h(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()


def kalshi_rows() -> list[dict]:
    ser = [s for s in K.series() if s.get("category") in SD.IN_SCOPE_CATEGORIES and s.get("frequency") not in L.DROP_FREQ]
    rows = []
    from concurrent.futures import ThreadPoolExecutor

    def one(s):
        try:
            ms = K.settled_markets(s["ticker"])
        except Exception:  # noqa: BLE001
            return []
        return [{"venue": "kalshi", "id": m["id"], "event": m.get("event_ticker"), "series": s["ticker"], "category": s.get("category"),
                 "title": m.get("title"), "sub": m.get("subtitle"), "rules": (m.get("rules_primary") or "")[:600],
                 "open": m.get("open_date"), "close": m.get("close_date"), "y": 1 if m["result"] == "yes" else 0, "volume": m.get("volume")}
                for m in ms if m.get("result") in ("yes", "no") and W0 <= (m.get("close_date") or "") <= W1 and (m.get("volume") or 0) >= 500
                and not SD.is_reflexive(m)]
    with ThreadPoolExecutor(L.THREADS) as ex:
        for r in ex.map(one, ser):
            rows += r
    return rows


def poly_rows() -> list[dict]:
    rows, seen = [], set()
    for tag in P.TAGS:
        for off in range(0, 2000, 100):
            try:
                evs = get_json("polymarket", P.GAMMA, "/events", {"closed": "true", "tag_slug": tag, "end_date_min": W0,
                                                                  "end_date_max": W1 + "T23:59:59Z", "limit": 100, "offset": off,
                                                                  "order": "volume", "ascending": "false"})
            except Exception:  # noqa: BLE001
                break
            for ev in evs or []:
                for m in ev.get("markets") or []:
                    if m["id"] in seen or (P._f(m.get("volumeNum")) or 0) < 10_000:
                        continue
                    try:
                        px = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                        tok = json.loads(m.get("clobTokenIds") or "[]")[0]
                    except (ValueError, IndexError, TypeError):
                        continue
                    close = (m.get("closedTime") or "")[:10]
                    if len(px) != 2 or sorted(px) != [0.0, 1.0] or not (W0 <= close <= W1):
                        continue
                    if SD.is_reflexive({"title": m.get("question"), "subtitle": m.get("groupItemTitle")}):
                        continue
                    seen.add(m["id"])
                    rows.append({"venue": "polymarket", "id": m["id"], "event": f"E{ev.get('id')}", "series": tag, "category": tag,
                                 "title": m.get("question"), "sub": m.get("groupItemTitle"), "rules": (m.get("description") or "")[:600],
                                 "open": (m.get("startDate") or "")[:10] or None, "close": close, "token": tok, "y": int(px[0] == 1.0),
                                 "volume": P._f(m.get("volumeNum"))})
            if not evs or len(evs) < 100:
                break
    return rows


def main():
    L.SNAPS = TS                         # the pricers read the snapshot calendar from here
    rows = poly_rows() + kalshi_rows()
    print(f"{len(rows)} resolved markets in the window", flush=True)
    for r in rows:                       # eligible forecast dates: open, and at least 14 days before close
        el = [t for t in TS if (r.get("open") or "0000") < t and (date.fromisoformat(r["close"]) - date.fromisoformat(t)).days >= 14]
        r["T"] = sorted(el, key=lambda t: _h(r["id"] + t))[0] if el else None
    rows = [r for r in rows if r["T"]]
    by = {}
    for r in rows:                       # one market per event (hash), before pricing, so the choice can't depend on price
        k = r["venue"] + ":" + str(r["event"])
        if k not in by or _h(r["id"]) < _h(by[k]["id"]):
            by[k] = r
    cand = list(by.values())
    print(f"{len(cand)} events with an eligible forecast date; pricing at T...", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def price(r):                        # L.SNAPS = all five dates (set above, never changed in threads); keep T's
        f = L.poly_prices if r["venue"] == "polymarket" else L.kalshi_prices
        return f(dict(r, open=r.get("open") or "0000")).get(r["T"])
    with ThreadPoolExecutor(L.THREADS) as ex:
        for r, p in zip(cand, ex.map(price, cand)):
            r["market_p_at_T"] = p
    qs = [r for r in cand if r["market_p_at_T"] is not None and 0.02 <= r["market_p_at_T"] <= 0.98]
    qs.sort(key=lambda r: _h(r["id"]))
    for k, q in enumerate(qs):
        q["qid"] = f"q{k:04d}"
        q["horizon_days"] = (date.fromisoformat(q["close"]) - date.fromisoformat(q["T"])).days
    pilot = [q["qid"] for q in qs if q["venue"] == "kalshi"][:15] + [q["qid"] for q in qs if q["venue"] == "polymarket"][:15]
    OUT.write_text(json.dumps({"rules": __doc__, "questions": qs, "pilot": pilot}, indent=1, default=str), encoding="utf-8")
    import collections
    c = collections.Counter((q["venue"], q["category"]) for q in qs)
    md = ["# Part 2: the forecasting question set", "", "Generated by `scripts/part2_questions.py` (code only). Rules in the script's docstring.", "",
          f"- **{len(qs)} questions** (one per event), resolved 2026-07-01 .. 2026-10-23; forecast dates {', '.join(TS)}",
          f"- YES rate {sum(q['y'] for q in qs) / max(1, len(qs)):.0%}; median horizon {sorted(q['horizon_days'] for q in qs)[len(qs) // 2] if qs else 0} days",
          f"- **Pilot: {len(pilot)} questions** (the first 15 per venue in a hash order)", "",
          "| venue | category | questions |", "|---|---|---|", *[f"| {v} | {cat} | {n} |" for (v, cat), n in c.most_common()], "",
          "## Pilot questions (market price at T is hidden from the forecasters)", "", "| qid | T | question | resolved | market at T |", "|---|---|---|---|---|"]
    for q in qs:
        if q["qid"] in pilot:
            md.append(f"| {q['qid']} | {q['T']} | {(q['title'] or '')[:90]}{(' — ' + q['sub']) if q.get('sub') and q['sub'] not in (q['title'] or '') else ''} | "
                      f"{q['close']} {'YES' if q['y'] else 'NO'} | {q['market_p_at_T']:.2f} |")
    (ROOT / "docs/pmrm/results/part2_questions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(qs)} questions; pilot {len(pilot)}")


if __name__ == "__main__":
    main()
