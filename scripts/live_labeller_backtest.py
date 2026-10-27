"""Backtest the labelling prompts the product actually runs (the design brief;
DECISIONS #236).

  python scripts/live_labeller_backtest.py            # DRY RUN: calls and cost per labeller (nothing called)
  python scripts/live_labeller_backtest.py --confirm  # label (capped, cached), score, write the results

Events (text only: title, category, region, description; never an outcome):
  main    the 745 major and extreme C0 library events (#196's set)
  clean   every C0 event dated on or after 2026-07-01 (after the model's knowledge), any severity
Labellers, each with its live prompt, byte-identical; only the wrapper changes:
  L1  the market labeller (scripts/discovery_label.PROMPT): each event passed as a market whose YES is "this happens"
      (title "This happens: <title>", rules = its description), 25 per call as live
  L2  the user-event labeller (discovery.PROPOSE + USER_BLOCK, n = 0): the event passed as a user's event; the book is
      the trainer's example book (the prompt needs one); no listed items, no history; today = the event's date; 8 per call
      (live passes up to 4 user events; the count is the only wrapper change)
  L3  the news labeller (planb_grow.PROMPT, used by the prospective log): signs for 7 factors, NO sizes. Not rerun: the
      library's Wikipedia-sourced events were labelled with this exact prompt, so its direction is scored from those
      labels ($0); size can't be scored
  #196  the research direction read (scripts/planc_direction_read.py), on the same events and the same 7 factors
Scored as #196 / #214: the realized move over the report day and the next session, in units of each factor's normal
2-day noise (C0 factor_y / factor_noise); direction = % right on moves over 2x noise where the label has a sign; size =
the realized move per size bucket; 90% bootstrap intervals clustered by event.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import discovery_label as DL  # noqa: E402
from planb_b2 import MODEL, _parse, _text  # noqa: E402
from worsey import config  # noqa: E402
from worsey.planb import live  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

C0 = ROOT / "cache/pmrm/planc/c0_events.pkl"
OUT_DIR = ROOT / "cache/pmrm/live_labeller"
RES = ROOT / "docs/pmrm/results"
F7 = list(live.SIGNED)
SZ = {"S": "small", "M": "medium", "L": "large", "N": "none"}
B1, B2 = 25, 8


def events():
    D = pickle.load(open(C0, "rb"))
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))}
    ev = sorted([r for r in D["events"] if r["severity"] in ("major", "extreme") or r["date"] >= "2026-07-01"], key=lambda r: r["date"])
    return D, lib, ev


def text_of(lib, e):
    r = lib.get(e["id"], {})
    return {"title": (r.get("title") or e["spec"].get("title") or "")[:200], "category": r.get("category"), "region": r.get("region"),
            "description": (r.get("description") or "")[:400]}


def tokens_to_label(f: str) -> dict | None:
    toks = str(f or "").split()
    if len(toks) != len(F7):
        return None
    out = {}
    for fac, t in zip(F7, toks):
        s = 1 if t[:1] == "+" else (-1 if t[:1] == "-" else 0)
        out[fac] = (s, SZ.get(t[1:2].upper(), "small") if s else "none")
    return out


def run_l1(run, lib, ev, confirm):
    path = OUT_DIR / "L1.json"
    have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [e for e in ev if e["id"] not in have]
    for i in range(0, len(todo), B1):
        b = todo[i:i + B1]
        items = [{"id": e["id"], "title": "This happens: " + text_of(lib, e)["title"], "subtitle": None,
                  "rules": (text_of(lib, e)["description"] or "")[:300], "closes": e["date"]} for e in b]
        prompt = DL.PROMPT.format(factors=", ".join(DL.FACTORS), sectors=", ".join(DL.SECTORS), markets=json.dumps(items, indent=0, ensure_ascii=False))
        resp = run.call(ICB.key("live_labeller_L1", {"ids": [e["id"] for e in b]}, prompt, MODEL), model=MODEL, max_tokens=8000,
                        messages=[{"role": "user", "content": prompt}])
        for it in _parse(_text(resp)):
            if isinstance(it, dict) and it.get("id") in {e["id"] for e in b}:
                lb = tokens_to_label(it.get("f"))
                if lb:
                    have[it["id"]] = lb
        path.write_text(json.dumps(have), encoding="utf-8")
    return have


def run_l2(run, lib, ev, confirm):
    from worsey import discovery as D
    path = OUT_DIR / "L2.json"
    have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [e for e in ev if e["id"] not in have]
    if not todo:
        return have
    import argparse as ap

    from worsey.agent import tools
    from worsey.agent.session import Session
    from worsey.cli import _data
    s = Session(_data(ap.Namespace(synthetic=False, quote_tag=None)))
    tools.load_portfolio(s, holdings=[{"ticker": "PFE", "value_usd": 20000}, {"ticker": "WDC", "value_usd": 15000}, {"ticker": "TSLA", "value_usd": 20000},
                                      {"ticker": "URA", "value_usd": 15000}, {"ticker": "GLD", "value_usd": 20000}])
    ctx = D.context(s)
    book = json.dumps(D.exposures(ctx), default=str)
    items = json.dumps({"markets": [], "calendar": [], "news": []})
    for i in range(0, len(todo), B2):
        b = todo[i:i + B2]
        ue = [f"{text_of(lib, e)['title']}" + (f" ({text_of(lib, e)['description'][:240]})" if text_of(lib, e)["description"] else "") for e in b]
        prompt = D.PROPOSE.format(today=b[0]["date"], days=91, n=0, cats="|".join(live.CATEGORIES), factors=", ".join(live.SIGNED),
                                  sectors=", ".join(list(ctx["SP"])), book=book, items=items, fh_field="", history_block="",
                                  user_block=D.USER_BLOCK.format(events=json.dumps(ue, ensure_ascii=False), n=0))
        resp = run.call(ICB.key("live_labeller_L2", {"ids": [e["id"] for e in b]}, prompt, config.EXPERT_MODEL), model=config.EXPERT_MODEL,
                        max_tokens=3500, messages=[{"role": "user", "content": prompt}])
        txt = _text(resp)
        try:
            arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
        except (ValueError, json.JSONDecodeError):
            arr = []
        users = [x for x in arr if isinstance(x, dict) and x.get("user") is True] or [x for x in arr if isinstance(x, dict)]
        for e, it in zip(b, users[:len(b)]):     # the prompt returns the user's events first, in order
            lb = tokens_to_label(it.get("f"))
            if lb:
                have[e["id"]] = lb
        path.write_text(json.dumps(have), encoding="utf-8")
    return have


def label_196(ev):
    dr = json.loads((ROOT / "cache/pmrm/planc/direction_read.json").read_text(encoding="utf-8"))
    out = {}
    for e in ev:
        v = dr.get(e["id"])
        if v:
            out[e["id"]] = {f: (int(v[f][0]), v[f][1] if v[f][0] else "none") for f in F7 if f in v}
    return out


def label_l3(lib, ev):
    out = {}
    for e in ev:
        r = lib.get(e["id"], {})
        fs = ((r.get("direction") or {}).get("factor_signs")) if r.get("origin") == "wikipedia_portal_monthly" else None
        if fs:
            out[e["id"]] = {f: (int(np.sign(fs.get(f, 0))), None) for f in F7}
    return out


def rows_for(D, ev, labels):
    j = {f: D["factors"].index(f) for f in F7}
    rows = []
    for e in ev:
        lb = labels.get(e["id"])
        if not lb:
            continue
        for f in F7:
            if f not in lb or not e["factor_noise"][j[f]] > 0 or not np.isfinite(e["factor_y"][j[f]]):
                continue
            z = float(e["factor_y"][j[f]] / e["factor_noise"][j[f]])
            rows.append({"id": e["id"], "f": f, "sign": lb[f][0], "size": lb[f][1], "z": z, "clean": e["date"] >= "2026-07-01",
                         "big_event": e["severity"] in ("major", "extreme")})
    return rows


def boot_ci(rows, stat, B=1000, seed=0):
    ids = sorted({r["id"] for r in rows})
    if not ids:
        return [None, None]
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = [r for i in rng.choice(ids, len(ids)) for r in by[i]]
        v = stat(s)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))] if vals else [None, None]


def hit(rows):
    s = [r for r in rows if r["sign"] != 0 and abs(r["z"]) > 2]
    return float(np.mean([np.sign(r["z"]) == r["sign"] for r in s])) if s else None


def score(rows) -> dict:
    out = {"rows": len(rows), "events": len({r["id"] for r in rows})}
    big = [r for r in rows if r["sign"] != 0 and abs(r["z"]) > 2]
    out["direction_big_moves"] = {"hit": hit(rows), "n": len(big), "ci90": boot_ci(rows, hit)}
    out["by_factor"] = {f: {"hit": hit([r for r in rows if r["f"] == f]), "n": sum(1 for r in big if r["f"] == f)} for f in F7}
    bins = [(0, 1, "<1x"), (1, 2, "1-2x"), (2, 3, "2-3x"), (3, 99, ">3x")]
    out["by_move_size"] = {}
    for lo, hi, name in bins:
        s = [r for r in rows if r["sign"] != 0 and lo <= abs(r["z"]) < hi]
        out["by_move_size"][name] = {"hit": float(np.mean([np.sign(r["z"]) == r["sign"] for r in s])) if s else None, "n": len(s)}
    out["signed_share"] = float(np.mean([r["sign"] != 0 for r in rows])) if rows else None
    if any(r["size"] for r in rows):
        out["size"] = {}
        for b in ("none", "small", "medium", "large"):
            s = [r for r in rows if r["size"] == b]
            if s:
                out["size"][b] = {"n": len(s), "mean_abs_move": float(np.mean([abs(r["z"]) for r in s])),
                                  "move_in_label_direction": float(np.mean([r["sign"] * r["z"] for r in s])) if b != "none" else None,
                                  "ci90_abs": boot_ci(s, lambda x: float(np.mean([abs(r["z"]) for r in x])) if x else None, B=500)}
        m = [out["size"].get(b, {}).get("mean_abs_move") for b in ("none", "small", "medium", "large")]
        m = [x for x in m if x is not None]
        out["size_monotonic"] = bool(all(a < b for a, b in zip(m, m[1:])))
    return out


def main(confirm: bool):
    D, lib, ev = events()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_clean = sum(e["date"] >= "2026-07-01" for e in ev)
    print(f"{len(ev)} events ({sum(e['severity'] in ('major', 'extreme') for e in ev)} major/extreme; {n_clean} on or after 2026-07-01)")
    from worsey.agent.llm import AnthropicLLM
    llm = AnthropicLLM() if confirm else None
    r1 = ICB.Run("live_labeller_L1", MODEL, llm, confirmed=confirm)
    r2 = ICB.Run("live_labeller_L2", config.EXPERT_MODEL, llm, confirmed=confirm)
    p1 = r1.plan(-(-len(ev) // B1), in_tok=4500, out_tok=2500)
    p2 = r2.plan(-(-len(ev) // B2), in_tok=4200, out_tok=900)
    total = p1["est_usd_max"] + p2["est_usd_max"]
    print(f"L3 and #196: existing labels, $0. Total estimate ${total:.2f}")
    if not confirm:
        return
    if total > 5:
        raise SystemExit("estimate over $5: stopping (task rule)")
    L = {"L1 market labeller": run_l1(r1, lib, ev, confirm), "L2 user-event labeller": run_l2(r2, lib, ev, confirm),
         "L3 news labeller (signs only)": label_l3(lib, ev), "#196 research read": label_196(ev)}
    res = {"events": len(ev), "clean_events": n_clean, "cost": {"L1": r1.summary(), "L2": r2.summary()}, "labellers": {}}
    for name, lab in L.items():
        rows = rows_for(D, ev, lab)
        res["labellers"][name] = {"labelled_events": len(lab),
                                  "main (major/extreme)": score([r for r in rows if r["big_event"]]),
                                  "before Jul 1 2026": score([r for r in rows if not r["clean"]]),
                                  "clean (Jul 1 2026 on)": score([r for r in rows if r["clean"]])}
    try:
        sh = json.loads((RES / "planc_shadow.json").read_text(encoding="utf-8"))
        res["prospective_log"] = {"scored_events": sh.get("scored_events"), "late_events": sh.get("late_events"),
                                  "note": "all scored after their windows had closed (late): a plumbing check, not evidence"}
    except Exception:  # noqa: BLE001
        pass
    (RES / "live_labeller_backtest.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for name, v in res["labellers"].items():
        m, c = v["main (major/extreme)"], v["clean (Jul 1 2026 on)"]
        print(f"{name}: labelled {v['labelled_events']} | main big-move direction {m['direction_big_moves']['hit']} (n {m['direction_big_moves']['n']}) "
              f"| clean {c['direction_big_moves']['hit']} (n {c['direction_big_moves']['n']}) | size {json.dumps({k: round(x['mean_abs_move'], 2) for k, x in (m.get('size') or {}).items()})}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--confirm", action="store_true")
    main(a.parse_args().confirm)
