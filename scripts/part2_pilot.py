"""Part 2, step 5: the forecaster pilot (docs/event_pipeline/probabilities.md; DECISIONS #230).

  python scripts/part2_pilot.py            # DRY RUN: questions, calls, cost (nothing called)
  python scripts/part2_pilot.py --confirm  # run the arms (capped, cached) -> docs/pmrm/results/part2_pilot.md

Questions: from cache/pmrm/part2/questions.json (resolved Jul 1 - Oct 23, 2026; forecast date T a calendar date; one per
event). A random draw is mostly elections, tweet counts and company KPIs, which the library says nothing about, so the
pilot is the first 30 (in hash order, about half per venue) of the first 400 questions whose TEXT an LLM labels as one
of the library's event kinds other than elections (tariffs, sanctions, conflict, supply shocks, courts, disasters,
health; the elections label caught mostly local primaries, which move no book); outcomes and prices are never shown to
that step: 29 of the first 400. Thinking is off for every arm (v1 spent its token budget thinking and wrote no answer). The
market's price at T is hidden from every arm and kept only as the benchmark.

Evidence, frozen at T (nothing dated T or later):
  base rate      the question's kind (one labelling call per batch picks a library category, or none) -> how often at
                 least one major/extreme event of that kind happened in a window of the question's horizon, 2016-2025
  similar past   the 8 library events dated before T most similar to the question (local bge-small embeddings)
  news           Wikipedia current events from the 14 days before T, the 15 items sharing the most words with it
Arms:
  A  naive LLM: the question, its resolution rules and the date T; no tools
  B  A plus the base rate (what the library's base rate alone adds)
  C  Worsey library forecaster: 3 agents (same model, default temperature), each with the base rate, the similar past
     events and the news in a different order; aggregated as the median in log-odds, clipped to [2%, 98%]. (No Platt
     scaling in the pilot: it needs a held-out set; the backtest fits it.)
  D  the market's price at T (benchmark)
Scores: Brier and log loss per arm; paired bootstrap of C - A and C - D (one question per event, so questions resample).
Every call: dry run first, --confirm, per-run caps, the permanent label cache (SPEC 6.1).
"""

import argparse
import json
import pickle
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey import config  # noqa: E402
from worsey.planb import library as LIB  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

Q = ROOT / "cache/pmrm/part2/questions.json"
EMB = ROOT / "cache/pmrm/part2/lib_emb.pkl"
OUT = ROOT / "docs/pmrm/results"
MODEL = config.EXPERT_MODEL
POOL, N_PILOT = 400, 30
STOP = {"the", "a", "an", "of", "by", "in", "on", "to", "will", "be", "or", "and", "for", "at", "than", "more", "less",
        "before", "after", "is", "2026", "2027", "this", "that", "with", "any", "from", "as"}

ASK = """Today is {T}. Forecast the probability that this question resolves YES.

Question: {q}
Resolution rules: {rules}
It resolves by {close}.
{evidence}
Think briefly, then answer with ONLY a JSON object: {{"p": <probability between 0 and 1>, "why": "<at most 3 short lines>"}}"""

LABEL = """For each forecasting question, pick the event category it is mainly about from this list, or "none" if it is
not about such an event (e.g. elections, sports, company products, data prints):
{cats}
Return ONLY a JSON array: [{{"qid": "...", "category": "<key or none>"}}]

{qs}"""


def _text(resp) -> str:
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


def _p(txt: str) -> float | None:
    """The last "p": <number> in the answer (models sometimes truncate or restate the JSON)."""
    hits = re.findall(r'"p"\s*:\s*"?([0-9]*\.?[0-9]+)', txt)
    p = float(hits[-1]) if hits else None
    return None if p is None or not (0 <= p <= 1) else p


def _tok(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def library():
    from worsey.planb import live
    ev = [e for e in live.pack()["events"]]
    if EMB.exists():
        E = pickle.load(open(EMB, "rb"))
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        txt = [f"{e.get('title') or ''}. {e.get('description') or ''}"[:600] for e in ev]
        E = {"ids": [e["id"] for e in ev], "V": m.encode(txt, batch_size=64, normalize_embeddings=True).astype(np.float32)}
        pickle.dump(E, open(EMB, "wb"))
    return ev, E


def base_rate(ev, cat: str | None, horizon: int) -> float | None:
    if not cat or cat == "none":
        return None
    n = sum(1 for e in ev if e["category"] == cat and e.get("severity") in ("major", "extreme") and "2016" <= e["date"][:4] <= "2025")
    return float(1 - np.exp(-n / 10 * horizon / 365))


def similar(ev, E, qtext: str, T: str, k: int = 8) -> list[dict]:
    from sentence_transformers import SentenceTransformer
    global _M
    try:
        _M
    except NameError:
        _M = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    v = _M.encode([qtext], normalize_embeddings=True)[0]
    s = E["V"] @ v
    ok = np.array([e["date"] < T for e in ev])
    idx = [i for i in np.argsort(-s) if ok[i]][:k]
    return [{"date": ev[i]["date"], "category": ev[i]["category"], "severity": ev[i].get("severity"), "event": (ev[i].get("title") or "")[:140]} for i in idx]


def news(qtext: str, T: str, n: int = 15) -> list[str]:
    from worsey.planb import wikinews as W
    t = date.fromisoformat(T)
    lo = (t - timedelta(days=14)).isoformat()
    months = sorted({(d.year, d.month) for d in (t - timedelta(days=14), t - timedelta(days=1))})
    items = [it for y, m in months for it in W.month_items(y, m) if lo <= it["day"] < T]
    q = _tok(qtext)
    items.sort(key=lambda it: -len(q & _tok(it["text"])))
    return [f"{it['day']}: {it['text'][:220]}" for it in items[:n] if q & _tok(it["text"])]


def evidence(q, br, sim, nw, order) -> str:
    parts = {
        "base": (f"Base rate: in 2016-2025, at least one major or extreme {q['category_label'].replace('_', ' ')} event happened in "
                 f"about {br:.0%} of {q['horizon_days']}-day windows (how often events of that KIND happen, not this question's odds)."
                 if br is not None else ""),
        "sim": "Similar past events (from Worsey's library, all before today):\n" + "\n".join(
            f"- {s['date']} [{s['category']}, {s['severity']}] {s['event']}" for s in sim) if sim else "",
        "news": "News from the last 14 days (Wikipedia current events):\n" + "\n".join(f"- {x}" for x in nw) if nw else "News from the last 14 days: nothing matching found."}
    return "\n\n".join(parts[k] for k in order if parts[k])


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def boot(a, b, y, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b, y = map(np.asarray, (a, b, y))
    d = (a - y) ** 2 - (b - y) ** 2
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return {"mean": float(d.mean()), "ci": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}


def main(confirm: bool):
    from worsey.agent.llm import AnthropicLLM
    D = json.loads(Q.read_text(encoding="utf-8"))
    pool = D["questions"][:POOL]           # hash order, fixed before any forecast
    run = ICB.Run("part2_pilot", MODEL, AnthropicLLM() if confirm else None, confirmed=confirm)
    run.plan(-(-len(pool) // 50), in_tok=3500, out_tok=1500)
    run.plan(N_PILOT * 2, in_tok=700, out_tok=250)
    run.plan(N_PILOT * 3, in_tok=3500, out_tok=350)
    print(f"label {len(pool)} questions, then {N_PILOT} pilot questions x 5 calls")
    if not confirm:
        return
    # the pilot is about the library's event kinds: label the pool's question TEXT (never outcomes or prices), keep those
    cats = "\n".join(f"- {k}: {v}" for k, v in LIB.CATEGORIES.items())
    lab = {}
    for i in range(0, len(pool), 50):
        chunk = pool[i:i + 50]
        lp = LABEL.format(cats=cats, qs="\n".join(f"{q['qid']}: {q['title']} {q.get('sub') or ''}" for q in chunk))
        r = run.call(ICB.key("part2_pilot_label", {"q": [q["qid"] for q in chunk]}, lp, MODEL), model=MODEL, max_tokens=3000,
                     messages=[{"role": "user", "content": lp}])
        try:
            t = _text(r)
            lab.update({x["qid"]: x["category"] for x in json.loads(t[t.find("["):t.rfind("]") + 1])})
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    fit = [q for q in pool if lab.get(q["qid"]) in LIB.CATEGORIES and lab.get(q["qid"]) != "elections_politics"]
    pick = [q for q in fit if q["venue"] == "kalshi"][:N_PILOT // 2] + [q for q in fit if q["venue"] == "polymarket"][:N_PILOT // 2]
    pick += [q for q in fit if q not in pick][:N_PILOT - len(pick)]
    qs = sorted(pick, key=lambda q: q["qid"])
    print(f"{len(fit)} of {len(pool)} labelled questions are about the library's event kinds; pilot: {len(qs)}", flush=True)
    ev, E = library()
    orders = [("base", "sim", "news"), ("news", "sim", "base"), ("sim", "news", "base")]
    rows = []
    for q in qs:
        qtext = f"{q['title']} {q.get('sub') or ''}".strip()
        q["category_label"] = lab.get(q["qid"], "none") if lab.get(q["qid"]) in LIB.CATEGORIES else "none"
        br = base_rate(ev, q["category_label"], q["horizon_days"])
        sim, nw = similar(ev, E, qtext, q["T"]), news(qtext, q["T"])
        base = {"T": q["T"], "q": qtext, "rules": (q.get("rules") or "")[:600], "close": q["close"]}

        def ask(tag, ev_text, temp=None):
            pr = ASK.format(evidence=("\n" + ev_text + "\n") if ev_text else "", **base)
            kw = {"model": MODEL, "max_tokens": 600, "messages": [{"role": "user", "content": pr}], "thinking": {"type": "between_tools"}}
            if temp is not None:
                kw["temperature"] = temp
            resp = run.call(ICB.key("part2_pilot", {"qid": q["qid"], "arm": tag, "v": 2}, pr, MODEL), **kw)
            return _p(_text(resp)), _text(resp)[-400:]
        pA, wA = ask("A", "")
        pB, wB = ask("B", evidence(q, br, [], [], ("base",)))
        agents = [ask(f"C{k}", evidence(q, br, sim, nw, o)) for k, o in enumerate(orders)]
        ps = [p for p, _ in agents if p is not None]
        lg = np.median([np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99))) for p in ps]) if ps else 0.0
        pC = float(np.clip(1 / (1 + np.exp(-lg)), 0.02, 0.98))
        rows.append({"qid": q["qid"], "venue": q["venue"], "T": q["T"], "question": qtext[:160], "y": q["y"],
                     "category": q["category_label"], "base_rate": br, "n_similar": len(sim), "n_news": len(nw),
                     "A": pA, "B": pB, "C": pC, "C_agents": [p for p, _ in agents], "D": q["market_p_at_T"], "why_C": agents[0][1][-200:]})
        print(f"  {q['qid']} y={q['y']} A={pA} B={pB} C={pC:.2f} D={q['market_p_at_T']:.2f} | {qtext[:60]}", flush=True)
    ok = [r for r in rows if None not in (r["A"], r["B"])]
    y = [r["y"] for r in ok]
    res = {"questions": len(rows), "scored": len(ok), "yes_rate": float(np.mean(y)),
           "brier": {a: brier([r[a] for r in ok], y) for a in "ABCD"},
           "logloss": {a: logloss([r[a] for r in ok], y) for a in "ABCD"},
           "C_minus_A": boot([r["C"] for r in ok], [r["A"] for r in ok], y),
           "C_minus_B": boot([r["C"] for r in ok], [r["B"] for r in ok], y),
           "C_minus_D": boot([r["C"] for r in ok], [r["D"] for r in ok], y),
           "B_minus_A": boot([r["B"] for r in ok], [r["A"] for r in ok], y),
           "rows": rows, "llm": run.summary()}
    (OUT / "part2_pilot.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    names = {"A": "A. naive LLM", "B": "B. naive + base rate", "C": "C. Worsey library forecaster (3 agents)", "D": "D. market at T (benchmark)"}
    L = [f"# Part 2: forecaster pilot ({len(rows)} questions)", "", "Generated by `scripts/part2_pilot.py`. Questions: `results/part2_questions.md`. "
         "The market price was hidden from arms A-C.", "",
         f"- {res['scored']} questions scored; YES rate {res['yes_rate']:.0%}; cost ${run.summary().get('est_usd', 0):.2f}", "",
         "| arm | Brier (lower is better) | log loss |", "|---|---|---|",
         *[f"| {names[a]} | {res['brier'][a]:.4f} | {res['logloss'][a]:.4f} |" for a in "ABCD"], "",
         "Paired differences in Brier (negative = the first arm is better), 95% bootstrap intervals:", "",
         *[f"- **{k.replace('_minus_', ' - ')}:** {res[k]['mean']:+.4f} [{res[k]['ci'][0]:+.4f}, {res[k]['ci'][1]:+.4f}]"
           for k in ("C_minus_A", "C_minus_B", "B_minus_A", "C_minus_D")], "",
         "This is a pilot: a Brier gap of about 0.01 needs roughly 250 to be significant (PART2_PROBABILITIES.md).", "",
         "| qid | T | question | resolved | A | B | C (agents) | market |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
        L.append(f"| {r['qid']} | {r['T']} | {r['question'][:80]} | {'YES' if r['y'] else 'NO'} | {f(r['A'])} | {f(r['B'])} | "
                 f"{f(r['C'])} ({', '.join(f(p) for p in r['C_agents'])}) | {f(r['D'])} |")
    (OUT / "part2_pilot.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows",)}, indent=1, default=float))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--confirm", action="store_true")
    main(a.parse_args().confirm)
