"""Scorecard: the current engine vs a raw LLM on the whole of question 1 (Part B2; DECISIONS #239).

  python scripts/scorecard_vs_raw.py            # DRY RUN: runs, calls, estimated cost (nothing called)
  python scripts/scorecard_vs_raw.py --confirm  # run both arms, judge, score -> docs/pmrm/results/scorecard_vs_raw_llm.md

PREDECLARED (written before any run; no tuning on results):
  Dates      forecast dates T = 2026-07-01, 2026-08-01, 2026-09-01 (after the model's knowledge, ~June 2026); each scored
             on the 30 days after T. Evidence frozen at T: library events dated before T, prices as of T.
  Books      the 12 test books of #185 ($100k each) plus the example book (PFE, WDC, TSLA, URA, GLD; its USO puts are
             left out: no historical option chains as of T).
  Raw arm    the same model as the chat, no tools: the date, the holdings and that week's news (the library's stories
             from the 7 days before T, the same news Worsey sees), and question 1 verbatim: "find the events most likely
             to move this portfolio, put a probability on each (using prediction markets where relevant), and estimate
             how much the portfolio would gain or lose if each happens", plus "give a 90% range". Up to 8 events.
  Worsey arm the current engine (discovery.watchlist, unchanged) with the library cut off at T (live.set_cutoff) and
             prices as of T, fed what can be reconstructed as of T:
               news       the library's labelled stories from the 7 days before T (the source of the live news feed)
               markets    Kalshi/Polymarket markets open at T that resolved Jul 1 - Oct 23 (the Part II question set),
                          at their price on T, labelled by the live market labeller (scripts/discovery_label.PROMPT)
               proposals  the live proposal call, with the book's own history (the replay, cut off at T)
             NOT reconstructable: markets still open after Oct 23, the release calendar as of T, Polymarket events
             outside the question set. Worsey's list = its three picks (what it shows) and its full list (headline +
             tail risks + the rest it counts).
  Realized   the library's major and extreme events dated in (T, T + 30]; each one's realized book move = the book's
             actual return over the event's window (report day + next session) x $100k.
  Judge      one blinded call per (book, date, arm): for each listed event, did it happen in the 30 days (yes / no /
             unclear), and which realized events it is or is the same developing story as; and which open-at-T market
             (if any, from a shortlist) asks the same question.
Scores, each Worsey vs raw:
  1 Coverage       share of the window's realized event-driven book loss whose event was on the arm's list (top 3; full)
  2 Probabilities  Brier and log score on the arm's events judged yes/no (Worsey: only events with a real probability,
                   i.e. a market tier or a forecast; base-rate-only events are counted separately); plus the pilot (#230)
  3 Moves/ranges   on listed events that happened and match a realized event: the interval score (5/50/95), the 90%
                   coverage and the middle error vs the realized book move; floor = zero + the book's normal 2-day noise
                   (Worsey's 95th = its 5th mirrored around its middle: impact() reports the middle and the 1-in-20)
  4 Traceability   numbers traceable to data (Worsey: grounding check; raw: none), and for events matched to an
                   open-at-T market, the error of the arm's probability vs that market's price on T
Intervals: 90% bootstrap clustered by (book, date).
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey import config  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

DATES = ["2026-07-01", "2026-08-01", "2026-09-01"]
WINDOW = 30
OUT = ROOT / "cache/pmrm/scorecard"
RES = ROOT / "docs/pmrm/results"
MODEL = config.EXPERT_MODEL

RAW = """Today is {T}. You manage this portfolio (about $100,000):
{holdings}

This week's news:
{news}

Find the events most likely to move this portfolio over the next 30 days, put a probability on each (using prediction
markets where relevant), and estimate how much the portfolio would gain or lose if each happens. Give a 90% range.
Up to 8 events. Answer with ONLY a JSON array:
[{{"event": "<one line>", "probability": <0-1>, "book_move_if_it_happens_usd": <number>, "range_90_usd": [<low>, <high>],
   "by_holding_usd": {{"<TICKER>": <number>, ...}}}}]"""

JUDGE = """Below is a list of events someone predicted on {T} for the next 30 days, the major and extreme news events
that actually happened from {T} to {T2}, and some prediction markets open on {T}.

For each predicted event decide:
- "happened": "yes" if it (or something that clearly is it) happened within the 30 days, "no" if it didn't, "unclear"
  if you can't tell from the list (a scheduled data print or a vague trend is usually "unclear")
- "matches": the ids of the actual events that are this event or a direct step of the same developing story ([] if none)
- "market": the id of a market below that asks the same question, or null

Return ONLY a JSON array: [{{"i": <predicted index>, "happened": "yes|no|unclear", "matches": [...], "market": <id or null>}}]

Predicted:
{pred}

Actual events (id | date | event):
{actual}

Markets open on {T} (id | question):
{markets}"""


def books() -> dict:
    from planb_naive_books import BOOKS
    import yaml
    out = {}
    bal = yaml.safe_load((ROOT / "portfolios/balanced.yaml").read_text(encoding="utf-8"))
    pos = {p["ticker"]: float(p["value_usd"]) for p in bal["positions"]}
    tot = sum(pos.values())
    out["Demo book (balanced)"] = {t: 100000 * v / tot for t, v in pos.items()}
    for b, (w, _) in BOOKS.items():
        out[b] = {t: 100000 * x for t, x in w.items()}
    out["Example book"] = {"PFE": 20000 * 100 / 90, "WDC": 15000 * 100 / 90, "TSLA": 20000 * 100 / 90, "URA": 15000 * 100 / 90, "GLD": 20000 * 100 / 90}
    return out


def library():
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    return [r for r in lib if r["kind"] == "unscheduled"]


def news_before(lib, T, days=7):
    lo = (date.fromisoformat(T) - timedelta(days=days)).isoformat()
    return [r for r in lib if lo <= r["date"] < T]


def realized_in(lib, T):
    hi = (date.fromisoformat(T) + timedelta(days=WINDOW)).isoformat()
    return [r for r in lib if T < r["date"] <= hi and r.get("severity") in ("major", "extreme") and r.get("w1")]


def questions(T):
    q = json.loads((ROOT / "cache/pmrm/part2/questions.json").read_text(encoding="utf-8"))["questions"]
    return [x for x in q if x["T"] == T]


def as_market(q) -> dict:
    p = q["market_p_at_T"]
    return {"id": q["id"] if str(q["id"]).startswith(("PM-", "KX")) or q["venue"] == "kalshi" else f"PM-{q['id']}",
            "venue": q["venue"], "title": q.get("title"), "subtitle": q.get("sub"), "rules_primary": q.get("rules"),
            "event_ticker": f"{q['venue']}:{q['event']}", "close_date": q["close"], "bid": p, "ask": p, "last": p, "status": "active"}


def plan_counts():
    n_runs = len(DATES) * len(books())
    n_q = sum(len(questions(T)) for T in DATES)
    return {"runs": n_runs, "raw": n_runs, "propose": n_runs, "relation": n_runs, "forecast_max": n_runs * 3 * 3,
            "label": -(-n_q // 25), "judge": n_runs * 2, "questions": n_q}


def label_markets(run, mk: list[dict]) -> dict:
    import discovery_label as DL
    from planb_b2 import _parse, _text
    path = OUT / "market_labels.json"
    have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [m for m in mk if m["id"] not in have]
    for i in range(0, len(todo), 25):
        b = todo[i:i + 25]
        items = [{"id": m["id"], "title": m.get("title"), "subtitle": m.get("subtitle"), "rules": (m.get("rules_primary") or "")[:300],
                  "closes": m.get("close_date")} for m in b]
        prompt = DL.PROMPT.format(factors=", ".join(DL.FACTORS), sectors=", ".join(DL.SECTORS), countries=", ".join(DL.COUNTRIES),
                                  markets=json.dumps(items, indent=0, ensure_ascii=False))
        resp = run.call(ICB.key("scorecard_label", {"ids": [m["id"] for m in b]}, prompt, MODEL), model=MODEL, max_tokens=8000,
                        messages=[{"role": "user", "content": prompt}])
        for it in _parse(_text(resp)):
            if not isinstance(it, dict) or it.get("id") not in {m["id"] for m in b}:
                continue
            toks = str(it.get("f") or "").split()
            if len(toks) != len(DL.FACTORS):
                continue
            signs = {f: t[0] for f, t in zip(DL.FACTORS, toks) if t[:1] in "+-" and len(t) > 1}
            sizes = {f: {"S": "small", "M": "medium", "L": "large"}.get(t[1:2].upper(), "small") for f, t in zip(DL.FACTORS, toks) if f in signs}
            have[it["id"]] = {"moves_markets": bool(it.get("moves_markets")), "severity": it.get("severity") or "moderate",
                              "factor_signs": signs, "factor_sizes": sizes,
                              "sectors": [list(x[:2]) for x in (it.get("sectors") or []) if isinstance(x, list) and len(x) >= 2][:2],
                              "companies": [], "countries": [list(x[:2]) for x in (it.get("countries") or []) if isinstance(x, list) and len(x) >= 2][:3],
                              "why": str(it.get("why") or "")[:120]}
        path.write_text(json.dumps(have), encoding="utf-8")
    return have


class Capped:
    """Routes the engine's own LLM calls through the per-run caps and the label cache (SPEC 6.1)."""
    def __init__(self, run, tag):
        self.run, self.tag, self.n = run, tag, 0

    def create(self, **kw):
        kw.pop("_source", None)
        self.n += 1
        txt = json.dumps(kw.get("messages"), sort_keys=True, default=str)
        return self.run.call(ICB.key("scorecard_engine", {"tag": self.tag, "n": self.n}, txt, kw.get("model", MODEL)), **kw)


def worsey_arm(run, name, hold, T, lib, labels) -> dict:
    import argparse as ap

    from worsey import discovery as D
    from worsey.agent import tools
    from worsey.agent.session import Session
    from worsey.data.provider import LiveMarketData
    from worsey.planb import live
    live.set_cutoff(T)
    data = LiveMarketData(as_of=date.fromisoformat(T))
    data.event_markets = lambda: []           # nothing from today's market list leaks into the past
    s = Session(data)
    tools.load_portfolio(s, holdings=[{"ticker": t, "value_usd": v} for t, v in hold.items()])
    ctx = D.context(s)
    mk = [as_market(q) for q in questions(T)]
    rows_m = []
    for m in mk:
        lb = labels.get(m["id"])
        if not lb or not lb.get("moves_markets"):
            continue
        r = {"source": "market", "venue": m["venue"], "market_id": m["id"], "event": m["event_ticker"], "title": m["title"],
             "when": m["close_date"], "why_it_moves": lb.get("why"), **D.impact(ctx, lb)}
        rows_m.append(D._with_odds(r, m["last"], "market"))
    rows_n = []
    for e in news_before(lib, T):
        fs = ((e.get("direction") or {}).get("factor_signs")) or {}
        sev = e.get("severity") or "moderate"
        lb = {"severity": sev, "factor_signs": {k: ("+" if v > 0 else "-") for k, v in fs.items() if v},
              "factor_sizes": {k: D.SEV_SIZE.get(sev, "small") for k, v in fs.items() if v}}
        if lb["factor_signs"]:
            rows_n.append(D._with_odds({"source": "news", "title": f"Story develops further: {(e.get('title') or '')[:120]}",
                                        "when": f"reported {e['date']}", "category": e.get("category"), **D.impact(ctx, lb)},
                                       D._followup_rates().get(e.get("category")), "base rate"))
    saved = (D.market_candidates, D.calendar, D.news_candidates, D._universe)
    D.market_candidates = lambda c, h=91: rows_m
    D.calendar = lambda c, h=91: []
    D.news_candidates = lambda c, days=7: rows_n
    D._universe = lambda s_: mk
    try:
        w = D.watchlist(s, Capped(run, f"{name}|{T}"), horizon_days=WINDOW)
    finally:
        D.market_candidates, D.calendar, D.news_candidates, D._universe = saved
        live.set_cutoff(None)
    def ev(r):
        q05 = r.get("bad_case_if_yes_1_in_20_usd")
        mid = r.get("full_impact_usd", r.get("swing_usd"))
        return {"event": r["title"], "probability": r.get("odds"), "tier": r.get("tier"), "source": r.get("source"),
                "move": mid, "q05": q05, "q95": (2 * mid - q05) if (q05 is not None and mid is not None) else None}
    full = w.get("headline", []) + w.get("tail_risks", [])
    return {"picks": [ev(r) for r in w.get("picks", [])], "full": [ev(r) for r in full], "more_count": w.get("more_count"),
            "swing": w.get("normal_2_day_swing_usd")}


def raw_arm(run, name, hold, T, lib) -> list:
    from planb_b2 import _text
    news = "\n".join(f"- {e['date']}: {(e.get('title') or '')[:200]}" for e in news_before(lib, T)) or "- (no notable items)"
    pr = RAW.format(T=T, holdings="\n".join(f"- {t}: ${v:,.0f}" for t, v in hold.items()), news=news)
    resp = run.call(ICB.key("scorecard_raw", {"book": name, "T": T}, pr, MODEL), model=MODEL, max_tokens=4000,
                    messages=[{"role": "user", "content": pr}])
    t = _text(resp)
    try:
        arr = json.loads(t[t.find("["):t.rfind("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    out = []
    for x in arr if isinstance(arr, list) else []:
        if isinstance(x, dict) and x.get("event"):
            rg = x.get("range_90_usd") or [None, None]
            out.append({"event": str(x["event"])[:200], "probability": x.get("probability"), "move": x.get("book_move_if_it_happens_usd"),
                        "q05": rg[0] if len(rg) > 0 else None, "q95": rg[1] if len(rg) > 1 else None})
    return out[:8]


def judge(run, arm_events, T, realized, mk_short) -> list:
    from planb_b2 import _text
    T2 = (date.fromisoformat(T) + timedelta(days=WINDOW)).isoformat()
    pr = JUDGE.format(T=T, T2=T2, pred="\n".join(f"{i}: {e['event']}" for i, e in enumerate(arm_events)),
                      actual="\n".join(f"{r['id']} | {r['date']} | {(r.get('title') or '')[:160]}" for r in realized) or "(none)",
                      markets="\n".join(f"{m['id']} | {(m.get('title') or '')[:120]} {m.get('subtitle') or ''}" for m in mk_short) or "(none)")
    resp = run.call(ICB.key("scorecard_judge", {"T": T, "ev": [e["event"] for e in arm_events]}, pr, MODEL), model=MODEL, max_tokens=3000,
                    messages=[{"role": "user", "content": pr}])
    t = _text(resp)
    try:
        arr = json.loads(t[t.find("["):t.rfind("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    return [x for x in arr if isinstance(x, dict) and isinstance(x.get("i"), int)]


def realized_moves(realized, hold, T):
    """Each realized event's actual book move over its window (report day + next session)."""
    from worsey.data.provider import LiveMarketData
    d = LiveMarketData(as_of=date.fromisoformat(T) + timedelta(days=WINDOW + 10))
    cl = d.closes(list(hold), days=120)
    r = cl.pct_change()
    out = {}
    for e in realized:
        a, b = pd.Timestamp(e["w1"]["start"]), pd.Timestamp(e["w1"]["end"])
        win = r[(r.index > a) & (r.index <= b)]
        if len(win):
            out[e["id"]] = float(sum(v * ((1 + win[t].fillna(0)).prod() - 1) for t, v in hold.items() if t in win))
    return out


def interval_score(lo, mid, hi, y, alpha=0.1):
    lo, hi = min(lo, hi), max(lo, hi)
    return (hi - lo) + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)


def boot(vals_by_cluster: dict, f, B=2000, seed=0):
    keys = list(vals_by_cluster)
    if not keys:
        return [None, None]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = [x for k in rng.choice(keys, len(keys)) for x in vals_by_cluster[k]]
        v = f(pick)
        if v is not None:
            out.append(v)
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))] if out else [None, None]


def main(confirm: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    c = plan_counts()
    from worsey.agent.llm import AnthropicLLM
    llm = AnthropicLLM() if confirm else None
    run = ICB.Run("scorecard", MODEL, llm, confirmed=confirm)
    est = 0.0
    for n, i, o in ((c["raw"], 2000, 1800), (c["propose"], 4500, 2500), (c["relation"], 3500, 1200), (c["forecast_max"], 2500, 300),
                    (c["label"], 4500, 2500), (c["judge"], 3500, 800)):
        est += run.plan(n, in_tok=i, out_tok=o)["est_usd_max"]
    print(json.dumps(c), f"| total estimate up to ${est:.2f} (forecasts at their maximum)")
    if not confirm:
        return
    if est > 10 and "--approved-over-10" not in sys.argv:   # The trainer approved the full run at ~$12 (Oct 25)
        raise SystemExit("estimate over $10: stop and report (task rule)")
    lib = library()
    B = books()
    labels = label_markets(run, [as_market(q) for T in DATES for q in questions(T)])
    out_f = OUT / "runs.json"
    runs = json.loads(out_f.read_text(encoding="utf-8")) if out_f.exists() else {}
    from worsey.planb import probability as PR
    for T in DATES:
        realized = realized_in(lib, T)
        mk = [as_market(q) for q in questions(T)]
        for name, hold in B.items():
            key = f"{name}|{T}"
            if key in runs:
                continue
            w = worsey_arm(run, name, hold, T, lib, labels)
            rw = raw_arm(run, name, hold, T, lib)
            moves = realized_moves(realized, hold, T)
            rec = {"book": name, "T": T, "worsey": w, "raw": rw, "realized": {e["id"]: {"title": e.get("title"), "date": e["date"], "move": moves.get(e["id"])} for e in realized}}
            for arm, evs in (("worsey", w["full"]), ("raw", rw)):
                short = []
                for e in evs:
                    for m in PR.candidates_for(e["event"], mk, k=3):
                        if m not in short:
                            short.append(m)
                rec[f"judge_{arm}"] = judge(run, evs, T, realized, short[:30])
            rec["market_p"] = {m["id"]: m["last"] for m in mk}
            runs[key] = rec
            out_f.write_text(json.dumps(runs, default=float), encoding="utf-8")
            print(f"  {key}: worsey {len(w['full'])} events (picks {len(w['picks'])}), raw {len(rw)}, realized {len(realized)}", flush=True)
    score(runs, run)


def score(runs, run=None):
    clusters = {}
    res = {"runs": len(runs), "dates": DATES, "books": len(books())}
    cov = {"worsey_picks": {}, "worsey_full": {}, "raw": {}}
    prob = {"worsey": {}, "raw": {}}
    rng_ = {"worsey": {}, "raw": {}, "floor": {}}
    mkt_err = {"worsey": {}, "raw": {}}
    base_only = 0
    for key, r in runs.items():
        real = {k: v for k, v in r["realized"].items() if v.get("move") is not None}
        loss = {k: -v["move"] for k, v in real.items() if v["move"] < 0}
        tot = sum(loss.values())
        swing = (r["worsey"].get("swing") or 0.0)
        for arm, evs, jkey in (("worsey_full", r["worsey"]["full"], "judge_worsey"), ("raw", r["raw"], "judge_raw")):
            j = {x["i"]: x for x in r.get(jkey, [])}
            matched = {m for x in j.values() for m in (x.get("matches") or [])}
            cov[arm][key] = [(sum(loss[k] for k in loss if k in matched), tot)] if tot > 0 else []
            if arm == "worsey_full":
                pick_titles = {p["event"] for p in r["worsey"]["picks"]}
                mp = {m for i, x in j.items() if i < len(evs) and evs[i]["event"] in pick_titles for m in (x.get("matches") or [])}
                cov["worsey_picks"][key] = [(sum(loss[k] for k in loss if k in mp), tot)] if tot > 0 else []
            a = "worsey" if arm == "worsey_full" else "raw"
            for i, e in enumerate(evs):
                x = j.get(i)
                if not x:
                    continue
                p = e.get("probability")
                if a == "worsey" and e.get("tier") is None:
                    base_only += 1
                    p = None
                if x.get("happened") in ("yes", "no") and isinstance(p, (int, float)):
                    prob[a].setdefault(key, []).append((float(min(max(p, 0.001), 0.999)), 1 if x["happened"] == "yes" else 0))
                if x.get("happened") == "yes":
                    ys = [real[m]["move"] for m in (x.get("matches") or []) if m in real]
                    if ys and all(isinstance(e.get(k), (int, float)) for k in ("move", "q05", "q95")):
                        y = ys[0]
                        rng_[a].setdefault(key, []).append((interval_score(e["q05"], e["move"], e["q95"], y), int(min(e["q05"], e["q95"]) <= y <= max(e["q05"], e["q95"])), abs(e["move"] - y)))
                        sd = swing / np.sqrt(2) * np.sqrt(1.5) if swing else abs(y)
                        rng_["floor"].setdefault(f"{key}|{a}", []).append((interval_score(-1.645 * sd, 0, 1.645 * sd, y), int(abs(y) <= 1.645 * sd), abs(y)))
                if x.get("market") and x["market"] in r.get("market_p", {}) and isinstance(p, (int, float)):
                    mkt_err[a].setdefault(key, []).append(abs(p - r["market_p"][x["market"]]))
    def share(rows):
        n = sum(x[1] for x in rows)
        return sum(x[0] for x in rows) / n if n else None
    res["coverage"] = {a: {"share": share([x for v in d.values() for x in v]), "ci90": boot(d, share), "windows": sum(1 for v in d.values() if v)} for a, d in cov.items()}
    def brier(rows):
        return float(np.mean([(p - y) ** 2 for p, y in rows])) if rows else None
    def logs(rows):
        return float(np.mean([-(y * np.log(p) + (1 - y) * np.log(1 - p)) for p, y in rows])) if rows else None
    res["probabilities"] = {a: {"brier": brier([x for v in d.values() for x in v]), "log": logs([x for v in d.values() for x in v]),
                                "n": sum(len(v) for v in d.values()), "yes_rate": float(np.mean([y for v in d.values() for _, y in v])) if d else None,
                                "ci90_brier": boot(d, brier)} for a, d in prob.items()}
    res["probabilities"]["worsey_base_rate_only_events_not_scored"] = base_only
    def isc(rows):
        return float(np.mean([x[0] for x in rows])) if rows else None
    res["moves_ranges"] = {a: {"interval_score": isc([x for v in d.values() for x in v]), "coverage_90": float(np.mean([x[1] for v in d.values() for x in v])) if d else None,
                               "median_middle_error": float(np.median([x[2] for v in d.values() for x in v])) if d else None,
                               "n": sum(len(v) for v in d.values()), "ci90_interval_score": boot(d, isc)} for a, d in rng_.items()}
    res["traceability"] = {"numbers_from_data": {"worsey": "every number from code (grounding check, #28/#83/#182)", "raw": "none: no data access"},
                           "probability_error_vs_market": {a: {"mean_abs": float(np.mean([x for v in d.values() for x in v])) if d else None,
                                                               "n": sum(len(v) for v in d.values())} for a, d in mkt_err.items()}}
    if run is not None:
        res["llm"] = run.summary()
    (RES / "scorecard_vs_raw_llm.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--confirm", action="store_true")
    a.add_argument("--score-only", action="store_true")
    a.add_argument("--approved-over-10", action="store_true")
    x = a.parse_args()
    if x.score_only:
        score(json.loads((OUT / "runs.json").read_text(encoding="utf-8")))
    else:
        main(x.confirm)
