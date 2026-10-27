"""Discovery layer, step 1: label every prediction market once: what would YES do to markets? (DECISIONS #218)

  python scripts/discovery_label.py            # dry run: markets to label and the cost under the caps (SPEC section 6.1)
  python scripts/discovery_label.py --confirm  # label (cached; resumes) -> src/worsey/planb/market_labels.json
  python scripts/discovery_label.py --polymarket [--confirm]   # also the topic-filtered Polymarket feed (#224)
  python scripts/discovery_label.py --polymarket --only-companies [--confirm]   # relabel labels that name companies (#232)

For each market in the built universe, from its title and rules only, the LLM says whether a YES would move markets and,
if so, how: per factor a direction and a size (small / medium / large, the same calls the "what if" engine sizes from
measured outcomes, #214), a severity, and optional sector or company legs. These are labels; code makes every number
from them. Stored in a committed file so the live chat makes no LLM call for discovery.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from planb_b2 import MODEL, _parse, _text  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

OUT = ROOT / "src/worsey/planb/market_labels.json"
SECTORS = ["semis", "software", "internet", "regional banks", "insurers", "biotech", "medtech", "energy stocks", "industrials",
           "defense", "homebuilders", "transports", "consumer discretionary", "retail", "staples", "utilities", "real estate",
           "materials", "metals & mining"]
FACTORS = ["market", "rates_2y", "rates_10y", "dollar", "oil", "gold", "credit"]
COUNTRIES = list(json.loads((ROOT / "src/worsey/planb/countries.json").read_text(encoding="utf-8"))["countries"])
PROMPT = """For each prediction market below, judge what a YES outcome would do to US markets in the first one to two
sessions after it becomes known, relative to the market NOT having priced it at all (ignore today's odds).

Return ONLY a JSON array, one object per market:
[{{"id": ..., "moves_markets": true|false, "severity": "minor|moderate|major|extreme",
   "f": "<7 tokens, one per factor in this order: {factors}; each a sign (+, - or 0) then a size (S small, M medium,
         L large), e.g. '-M +L 0S ...'>",
   "sectors": [["<sector>", "+|-"]], "companies": [["<TICKER>", "+|-"]], "countries": [["<country>", "+|-"]],
   "why": "<at most 12 words>"}}]

Factors: market = US stocks; rates = Treasury yields ("+" = yields up); credit = high-yield bonds minus Treasuries ("+" =
spreads tighten). Sectors, only if YES hits one directly, from: {sectors}.
Companies: only when the event is about that specific company (the company is the subject or is named as the direct
target: its drug, its earnings, its export licence, its CEO). Never for broad market, macro or sector events; their
effect on companies comes through the factors and sectors. Usually empty.
Countries: only those a YES is about or directly hits (a war, a coup, sanctions, an election, a crisis in that country),
each with its own direction, from: {countries}. Never for global outcomes. Usually empty.
Use moves_markets false (and "0S" everywhere) for markets with no meaningful market impact (celebrity, sports, trivia,
most AI-model trivia). Be conservative: most outcomes are small for most factors.

Markets:
{markets}"""


def universe(polymarket: bool = False):
    import argparse as _a
    from datetime import date

    from worsey.cli import _data
    d = _data(_a.Namespace(synthetic=False, quote_tag=None))
    mk = d.event_markets()
    if polymarket:      # the second venue, topic-filtered (DECISIONS #224); labels are per market, so reruns label only new ones
        from worsey.data import polymarket as P
        today = date.today().isoformat()
        mk += [m for m in P.topical(tag=today) if (m.get("close_date") or "0000") >= today]
    return mk


def key(m):
    return hashlib.sha256((m["id"] + "|" + (m.get("title") or "") + "|" + (m.get("rules_primary") or "")[:300]).encode()).hexdigest()[:16]


DEMONYMS = {"Brazil": ["Brazil", "Brazilian"], "Mexico": ["Mexico", "Mexican"], "Argentina": ["Argentin"], "Chile": ["Chile"],
            "Peru": ["Peru"], "Canada": ["Canada", "Canadian"], "Japan": ["Japan"], "China": ["China", "Chinese", "Beijing", "Xi Jinping"],
            "Hong Kong": ["Hong Kong"], "Taiwan": ["Taiwan"], "South Korea": ["Korea"], "India": ["India"], "Australia": ["Australia"],
            "Singapore": ["Singapore"], "Indonesia": ["Indonesia"], "Malaysia": ["Malaysia"], "Thailand": ["Thai"], "Germany": ["German"],
            "United Kingdom": ["UK ", "U.K.", "Britain", "British", "England"], "France": ["France", "French", "Macron"], "Italy": ["Ital"],
            "Spain": ["Spain", "Spanish"], "Switzerland": ["Swiss", "Switzerland"], "Netherlands": ["Netherlands", "Dutch"],
            "Sweden": ["Swed"], "Poland": ["Poland", "Polish"], "Greece": ["Greece", "Greek"], "Turkey": ["Turkey", "Turkish", "Erdogan"],
            "Israel": ["Israel", "Netanyahu", "Gaza"], "Saudi Arabia": ["Saudi"], "South Africa": ["South Africa"]}


def mentions_country(m: dict) -> bool:
    text = f"{m.get('title') or ''} {m.get('subtitle') or ''} {(m.get('rules_primary') or '')[:300]}"
    return any(w in text for ws in DEMONYMS.values() for w in ws)


def main(confirm: bool, polymarket: bool = False, only_companies: bool = False, only_countries: bool = False):
    mk = universe(polymarket)
    have = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"labels": {}}
    before = json.loads(json.dumps(have))
    if only_countries:   # #238: relabel only market-moving labels whose market names a country; keep every other field
        todo = [m for m in mk if have["labels"].get(m["id"], {}).get("moves_markets") and mentions_country(m)]
        print(f"{len(todo)} market-moving labels in today's universe name a country and get relabelled (countries field only)")
    elif only_companies:   # #232: relabel only labels that name companies, under the new company rule
        todo = [m for m in mk if have["labels"].get(m["id"], {}).get("companies")]
        n_all = sum(1 for v in have["labels"].values() if v.get("companies"))
        print(f"{n_all} committed labels name companies; {len(todo)} of them are in today's universe and get relabelled")
    else:
        todo = [m for m in mk if have["labels"].get(m["id"], {}).get("key") != key(m)]
    batches = [todo[i:i + 25] for i in range(0, len(todo), 25)]
    run = ICB.Run("discovery_label", MODEL, None, confirmed=confirm)
    if confirm:
        from worsey.agent.llm import AnthropicLLM
        run = ICB.Run("discovery_label", MODEL, AnthropicLLM(), confirmed=True)
    plan = run.plan(len(batches), in_tok=4000, out_tok=2500)
    print(f"{len(mk)} markets, {len(todo)} to label, {len(batches)} calls")
    if not confirm:
        return
    if not plan["within_caps"]:
        raise SystemExit("estimate exceeds the per-run caps; refusing (SPEC section 6.1)")
    for b in batches:
        items = [{"id": m["id"], "title": m.get("title"), "subtitle": m.get("subtitle"), "rules": (m.get("rules_primary") or "")[:300],
                  "closes": m.get("close_date")} for m in b]
        prompt = PROMPT.format(factors=", ".join(FACTORS), sectors=", ".join(SECTORS), countries=", ".join(COUNTRIES),
                               markets=json.dumps(items, indent=0, ensure_ascii=False))
        k = ICB.key(run.job, {"ids": [m["id"] for m in b]}, prompt, MODEL)
        try:
            resp = run.call(k, model=MODEL, max_tokens=8000, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as err:
            print("stopped:", err)
            break
        by = {m["id"]: m for m in b}
        for it in _parse(_text(resp)):
            if not isinstance(it, dict) or it.get("id") not in by:
                continue
            toks = str(it.get("f") or "").split()
            if len(toks) != len(FACTORS):
                continue
            signs = {f: t[0] for f, t in zip(FACTORS, toks) if t[:1] in "+-" and len(t) > 1}
            sizes = {f: {"S": "small", "M": "medium", "L": "large"}.get(t[1:2].upper(), "small") for f, t in zip(FACTORS, toks) if f in signs}
            have["labels"][it["id"]] = {
                "key": key(by[it["id"]]), "title": by[it["id"]].get("title"), "moves_markets": bool(it.get("moves_markets")),
                "severity": it.get("severity") if it.get("severity") in ("minor", "moderate", "major", "extreme") else "moderate",
                "factor_signs": signs, "factor_sizes": sizes,
                "sectors": [[s_, d_] for s_, d_ in (it.get("sectors") or []) if s_ in SECTORS and d_ in ("+", "-")][:2],
                "companies": [[c.upper(), d_] for c, d_ in (it.get("companies") or []) if isinstance(c, str) and d_ in ("+", "-")][:2],
                "countries": [[c, d_] for c, d_ in (it.get("countries") or []) if c in COUNTRIES and d_ in ("+", "-")][:3],
                "why": str(it.get("why") or "")[:120]}
        OUT.write_text(json.dumps(have, indent=1, ensure_ascii=False), encoding="utf-8")
    if only_countries:   # keep the earlier label, take only the new countries field (as #232 did for companies)
        for m in todo:
            new = have["labels"].get(m["id"], {})
            if m["id"] in before["labels"]:
                have["labels"][m["id"]] = dict(before["labels"][m["id"]], countries=new.get("countries", []))
    have["labelled_with"] = MODEL
    OUT.write_text(json.dumps(have, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run.summary() | {"labelled": len(have["labels"]), "of": len(mk),
                                      "moves_markets": sum(1 for v in have["labels"].values() if v["moves_markets"])}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--polymarket", action="store_true", help="also label the topic-filtered Polymarket feed")
    ap.add_argument("--only-companies", action="store_true", help="relabel only labels that name companies (#232)")
    ap.add_argument("--only-countries", action="store_true", help="relabel only labels whose market names a country (#238)")
    a = ap.parse_args()
    main(a.confirm, a.polymarket, a.only_companies, a.only_countries)
