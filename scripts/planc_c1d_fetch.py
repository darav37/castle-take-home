"""Plan C, C1d: download (and disk-cache) the minute bars of the first window session for every C0 event (PLAN_C §2a
group 2). Massive minute aggregates include pre-market (from 04:00 ET) and after-hours; FX and crypto are not on the plan.
No LLM calls.

  python scripts/planc_c1d_fetch.py [--threads 8]
"""

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from worsey.data import massive  # noqa: E402

MIN_TICKERS = ["SPY", "IEF", "HYG", "USO", "GLD", "UUP", "EEM", "XLE"]


def minute_rows(t: str, day: str) -> list[dict]:
    try:
        return massive._get(f"/v2/aggs/ticker/{t}/range/1/minute/{day}/{day}",
                            {"adjusted": "true", "sort": "asc", "limit": 50000}).get("results") or []
    except Exception:  # noqa: BLE001
        return []


def event_days() -> list[str]:
    D = pickle.load(open(ROOT / "cache/pmrm/planc/c0_events.pkl", "rb"))
    ext = pickle.load(open(ROOT / "cache/pmrm/b1/model_ext.pkl", "rb"))
    ss = ext["anchors"].index
    lib = {r["id"]: r for r in json.loads((ROOT / "cache/pmrm/b2/library.json").read_text(encoding="utf-8"))}
    days = set()
    for r in D["events"]:
        a = ss.searchsorted(pd.Timestamp(lib[r["id"]]["w1"]["start"])) + 1
        days.add(str(ss[a].date()))
    return sorted(d for d in days if d >= "2004-01-01")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    jobs = [(t, d) for d in event_days() for t in MIN_TICKERS]
    print(f"{len(jobs)} ticker-days", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for _ in ex.map(lambda j: minute_rows(*j), jobs):
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
