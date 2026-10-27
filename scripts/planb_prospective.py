"""Prospective (forward) test of model vs raw LLM vs assisted LLM (#3, DECISIONS #173).

  python scripts/planb_prospective.py log            # DRY RUN: new days, calls, cost; nothing is written or called
  python scripts/planb_prospective.py log --confirm  # label new news days and FREEZE all three predictions (capped run)
  python scripts/planb_prospective.py score          # attach outcomes to frozen predictions whose windows have closed

log:   for each day after the last logged day (up to yesterday), code collects that day's relevant Wikipedia Current
       events items; one capped Claude call picks and labels the economically relevant events (planb_grow's prompt);
       then the model, raw-LLM and assisted-LLM predictions (planb_b4_threeway's prompts and bucket mapping) are made
       BEFORE any outcome exists and appended to cache/pmrm/prospective/frozen.jsonl with a UTC timestamp and the
       prompt hashes. The file is append-only; nothing in it is ever edited.
score: for each frozen event whose short window has closed in the factor data, compute the realized factor moves and
       the running scores (R² vs zero, paired differences), written to docs/pmrm/results/prospective.json.
Every LLM call goes through the per-run and project caps (SPEC §6.1, $70 project cap).
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import planb_b4_threeway as B4  # noqa: E402
import planb_grow as G  # noqa: E402
from planb_b2 import MODEL, _parse, _text  # noqa: E402
from worsey.planb import impact as IM  # noqa: E402
from worsey.planb import library as LIB  # noqa: E402
from worsey.planb import wikinews as W  # noqa: E402
from worsey.pmrm.ic import budget as ICB  # noqa: E402

DIR = ROOT / "cache/pmrm/prospective"
FROZEN = DIR / "frozen.jsonl"
START = date(2026, 10, 17)       # the library's monthly growth ends with October 2026's page as of Oct 21


def history():
    """The library, factor space, bucket mapping and model, exactly as B4 builds them."""
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    X = ((ext["anchors"] - ext["anchor_mean"]) / ext["anchor_sd"]).fillna(0)
    lib = json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))
    uns = sorted([r for r in lib if r["kind"] == "unscheduled"], key=lambda r: r["date"])
    fi = [list(X.columns).index(f) for f in B4.FACTORS]
    ss = X.index

    def resp(r):
        a = ss.searchsorted(pd.Timestamp(r["w1"]["start"])) + 1
        b = ss.searchsorted(pd.Timestamp(r["w1"]["end"])) + 1
        return X.iloc[a:b].to_numpy().sum(0)[fi]

    U = np.array([resp(r) for r in uns])
    pre = (pd.to_datetime(pd.Series([r["date"] for r in uns])) < "2016-01-01").to_numpy()
    sd = U[pre].std(0)
    zU = U / sd
    bval = np.zeros((len(B4.BUCKETS), len(B4.FACTORS)))
    for j in range(len(B4.FACTORS)):
        bins = np.searchsorted(B4.CUTS, zU[pre, j])
        for b in range(len(B4.BUCKETS)):
            v = U[pre, j][bins == b]
            bval[b, j] = v.mean() if len(v) else 0.0
    cats = sorted(LIB.CATEGORIES)
    feat = lambda e: np.concatenate([[1.0], IM.direction_features(e), [e["category"] == c for c in cats]])
    Fd = np.array([feat(e) for e in uns])
    B = IM.ridge(Fd, U, 2.0)
    return {"X": X, "uns": uns, "U": U, "zU": zU, "sd": sd, "bval": bval, "B": B, "feat": feat, "fi": fi}


def bucket_of(z):
    return B4.BUCKETS[int(np.searchsorted(B4.CUTS, z))]


def log(confirm: bool):
    from worsey.agent.llm import AnthropicLLM
    DIR.mkdir(parents=True, exist_ok=True)
    frozen = [json.loads(l) for l in FROZEN.read_text(encoding="utf-8").splitlines()] if FROZEN.exists() else []
    done_days = {f["news_day"] for f in frozen} | ({l.strip() for l in (DIR / "days_done.txt").read_text().splitlines()} if (DIR / "days_done.txt").exists() else set())
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    days = [START + timedelta(days=k) for k in range((yesterday - START).days + 1)]
    days = [d for d in days if d.isoformat() not in done_days]
    llm = AnthropicLLM() if confirm else None
    run = ICB.Run("planb_prospective", MODEL, llm, confirmed=confirm)
    run.plan(len(days) * 3, in_tok=6000, out_tok=1500)
    if not confirm or not days:
        print(f"{len(days)} new days: {[d.isoformat() for d in days][:10]}")
        return
    H = history()
    ftxt = "\n".join(f"- {f}: {B4.FACTOR_TEXT[f]}" for f in B4.FACTORS)
    cats = " | ".join(LIB.CATEGORIES)
    dirf = ", ".join(f'"{k}": "{"|".join(v)}"' for k, v in G.DIRECTION_FIELDS.items())
    for d in days:
        items = [it for it in W.month_items(d.year, d.month) if it["day"] == d.isoformat()]
        if not items:
            if (datetime.now(timezone.utc).date() - d).days > 3:   # Wikipedia can lag a day or two: retry recent days (#222)
                with open(DIR / "days_done.txt", "a") as f:
                    f.write(d.isoformat() + "\n")
            continue
        items = G.select_items(items)[:120]
        listing = "\n".join(f"[{k}] {it['day']} | {it['text'][:240]}" for k, it in enumerate(items))
        prompt = G.PROMPT.format(month=d.isoformat(), maxev=4, cats=cats, sectors=", ".join(LIB.SECTORS), dirfields=dirf,
                                 factors=", ".join(G.SIGN_FACTORS), items=listing)
        k = ICB.key("planb_prospective_label", {"day": d.isoformat(), "n": len(items)}, prompt, MODEL)
        try:
            resp = run.call(k, model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": prompt}])
        except ICB.RunCapExceeded as e:
            print("stopped:", e)
            break
        evs = []
        for lab in _parse(_text(resp)):
            try:
                i = int(lab.get("i"))
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(items):
                ev = G.build_event(items[i], lab, d.strftime("%Y-%m"), i)
                if ev:
                    ev["id"] = f"p{d:%Y%m%d}{i:03d}"
                    evs.append(ev)
        if not evs:
            with open(DIR / "days_done.txt", "a") as f:
                f.write(d.isoformat() + "\n")
            continue
        # model prediction and analogs (all library events are earlier than these news days)
        preds = {}
        for e in evs:
            mp = H["feat"](e) @ H["B"]
            sims = np.array([IM.direction_similarity(e, u) for u in H["uns"]])
            top = np.argsort(-sims)[:5]
            preds[e["id"]] = {"model": mp.tolist(), "analogs": [H["uns"][t]["id"] for t in top],
                              "model_buckets": {f: bucket_of(mp[j] / H["sd"][j]) for j, f in enumerate(B4.FACTORS)},
                              "band": "wide" if np.mean(np.abs(H["zU"][top] - H["zU"][top].mean(0))) > 0.8 else "narrow"}

        def evtxt(e, assisted):
            t = {"id": e["id"], "date": e["date_first_report"], "category": e["category"], "region": e.get("region"),
                 "description": (e.get("description") or "")[:500]}
            if assisted:
                p = preds[e["id"]]
                t.update({"model_estimate": p["model_buckets"], "band": p["band"],
                          "similar_earlier_events": [{"date": H["uns"][int(np.where([u["id"] == a for u in H["uns"]])[0][0])]["date"],
                                                      "title": (next(u for u in H["uns"] if u["id"] == a).get("title") or "")[:100],
                                                      "moved": {f: bucket_of(H["zU"][[u["id"] for u in H["uns"]].index(a), j]) for j, f in enumerate(B4.FACTORS)}}
                                                     for a in p["analogs"]]})
            return t

        for mode, tmpl, assisted in (("raw LLM", B4.RAW, False), ("assisted LLM", B4.ASSISTED, True)):
            pr = tmpl.format(factors=ftxt, events=json.dumps([evtxt(e, assisted) for e in evs], indent=0, default=str))
            kk = ICB.key("planb_prospective_pred", {"mode": mode, "ids": [e["id"] for e in evs]}, pr, MODEL)
            try:
                r = run.call(kk, model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": pr}])
            except ICB.RunCapExceeded as e:
                print("stopped:", e)
                return
            byid = {it.get("id"): it.get("pred") or {} for it in _parse(_text(r)) if isinstance(it, dict)}
            for e in evs:
                p = byid.get(e["id"], {})
                preds[e["id"]][mode] = [H["bval"][B4.BUCKETS.index(p.get(f)), j] if p.get(f) in B4.BUCKETS else 0.0
                                        for j, f in enumerate(B4.FACTORS)]
                preds[e["id"]][mode + " buckets"] = p
                preds[e["id"]][mode + " prompt_hash"] = hashlib.sha256(pr.encode()).hexdigest()[:16]
        with open(FROZEN, "a", encoding="utf-8") as f:
            for e in evs:
                f.write(json.dumps({"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "news_day": d.isoformat(),
                                    "event": e, "predictions": preds[e["id"]]}, default=str) + "\n")
        print(f"  {d}: {len(evs)} events frozen", flush=True)
    print(json.dumps(run.summary()))


def score():
    if not FROZEN.exists():
        print("nothing frozen yet")
        return
    H = history()
    X, ss = H["X"], H["X"].index
    rows = [json.loads(l) for l in FROZEN.read_text(encoding="utf-8").splitlines()]
    Y, P = [], {"model": [], "raw LLM": [], "assisted LLM": []}
    for r in rows:
        d = pd.Timestamp(r["event"]["date_first_report"])
        i = ss.searchsorted(d)
        if i + 1 >= len(ss):            # the 2-session window hasn't closed in the factor data yet
            continue
        Y.append(X.iloc[i:i + 2].to_numpy().sum(0)[H["fi"]])
        for m in P:
            P[m].append(r["predictions"].get(m, [0.0] * len(B4.FACTORS)))
    out = {"frozen": len(rows), "scored": len(Y)}
    if Y:
        Y = np.array(Y)
        for m in P:
            Pm = np.array(P[m])
            out[m] = float(1 - ((Y - Pm) ** 2).sum() / (Y ** 2).sum())
    (ROOT / "docs/pmrm/results/prospective.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["log", "score"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    log(a.confirm) if a.step == "log" else score()
