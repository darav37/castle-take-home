"""Freeze the real-event part of Plan C's gate model for the live chat (DECISIONS #213). No LLM calls.

  python scripts/planc_real_event_pack.py      # cache/pmrm/planc/shadow_model.pkl -> src/worsey/planb/real_event.json

The gate model (rung D, #194; frozen for the shadow run, #201) splits every event into "noise" and "a real event". For a
user's "what if", the event is posited as happening for real, so the live scenario uses the real part: per factor, a
push of d x (the LLM's sign) x the factor's normal noise, and a spread of k x its normal noise. d and k were fitted on
all 3,571 events; the rung was fitted before 2023 and beat noise on 2023-26 (98-99% support on 2026; direction on big
moves 63-75%).
"""

import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

M = pickle.load(open(ROOT / "cache/pmrm/planc/shadow_model.pkl", "rb"))
out = {"source": "Plan C rung D (gate + LLM sign), frozen for the shadow run", "model_hash": M["hash"], "events": M["events"],
       "evidence": "fitted before 2023 it beat zero + normal noise on 2023-26 (2026: 98% support, big events 99%); direction on "
                   "big moves 63% (2023-25) and 68-75% (2026); DECISIONS #194, #197, #201",
       "signed_factors": [f for f, m in M["factors"].items() if m["has_sign"]],
       "factors": {f: {"push_in_llm_direction": round(float(m["d"]), 4), "real_event_width": round(float(m["k"]), 4)}
                   for f, m in M["factors"].items()}}
# the move in the LLM's direction by its size call (#196's read on 745 big events vs what happened, in noise units)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = pickle.load(open(ROOT / "cache/pmrm/planc/c0_events.pkl", "rb"))
dr = json.loads((ROOT / "cache/pmrm/planc/direction_read.json").read_text(encoding="utf-8"))
rows = []
for r in D["events"]:
    for j, f in enumerate(D["factors"]):
        v = (dr.get(r["id"]) or {}).get(f)
        if v and v[0] and r["factor_noise"][j] > 0:
            rows.append((v[1], v[0] * r["factor_y"][j] / r["factor_noise"][j], r["split"]))
S = pd.DataFrame(rows, columns=["size", "dz", "split"])
out["size_map"] = {sz: {"move_in_llm_direction": round(float(g.dz.mean()), 4),
                        "spread": round(float(np.sqrt(max((g.dz ** 2).mean() - g.dz.mean() ** 2, 0.0))), 4),
                        "events": int(len(g)), "move_2023_2026": round(float(g[g.split != "train"].dz.mean()), 4)}
                   for sz, g in S.groupby("size") if sz in ("small", "medium", "large")}
out["size_map_note"] = ("the LLM's size call per factor vs the real move in its direction, 745 major and extreme events 2004-2026 "
                        "(DECISIONS #196, #214); pre-2023 may carry hindsight, so the 2023-26 figure is shown beside it")
(ROOT / "src/worsey/planb/real_event.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["size_map"], indent=0))
print(json.dumps(out["factors"], indent=0)[:600])
