"""Freeze the sector layer's numbers for the live chat (DECISIONS #217). No LLM calls.

  python scripts/planc_sector_pack.py     # -> src/worsey/planb/sectors.json

Per sector (step 1, #215): its ETF, and how big its factor's 2-session moves are (beyond the macro factors, in its own
noise): the median, the 90th and the 95th percentile of |move| over 2011-2026. Per sector (step 2a, #216): peer
spillover from a leader's earnings, the fraction of the leader's move (in noise units) its peers move on average, and how
often they go the same way.
"""

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "scripts")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from planc_sectors import SECTORS  # noqa: E402

SF = pickle.load(open(ROOT / "cache/pmrm/planc/sector_factors.pkl", "rb"))
S = SF["S"].loc["2011-01-01":]
sp = json.loads((ROOT / "docs/pmrm/results/planc_spillover.json").read_text(encoding="utf-8"))
lead = sp.get("leaders_by_sector", {})
out = {"source": "sector factors (#215) and earnings spillover (#216)", "sectors": {}}
for name, etf in SECTORS.items():
    x = S[name].dropna()
    two = (x + x.shift(-1)).dropna() / np.sqrt(2)
    a = two.abs()
    out["sectors"][name] = {"etf": etf, "two_day_move_median": round(float(a.median()), 3), "two_day_move_p90": round(float(a.quantile(0.9)), 3),
                            "two_day_move_p95": round(float(a.quantile(0.95)), 3),
                            "residual_daily_sd": round(float(SF["meta"][name]["residual_daily_sd"]), 5),
                            "spillover_fraction": round(float(lead.get(name, {}).get("slope", 0.0)), 3) if lead.get(name, {}).get("slope", 0) > 0 else 0.0,
                            "spillover_same_direction": lead.get(name, {}).get("dir_reporter")}
(ROOT / "src/worsey/planb/sectors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: (v["two_day_move_median"], v["two_day_move_p90"], v["spillover_fraction"]) for k, v in out["sectors"].items()}))
