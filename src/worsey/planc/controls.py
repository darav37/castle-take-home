"""Ordinary control days for Plan C (C1b gate, C1d open baseline; DECISIONS #193).

The library has an event in almost every week (about 5,000 dated entries), so "not within 3 sessions of any event" leaves
only ~40 sessions before 2023 (the rule C1b first used, #192). The rule here: not the first window session of any library
event, and not within 3 sessions of a major or extreme one. Minor and moderate library events move markets about as
much as an ordinary day (#171), so the sessions next to them are ordinary days for this purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def control_sessions(ss: pd.DatetimeIndex, lib: list[dict], n_target: int = 3000, first: int = 600,
                     before: str = "2022-12-20", seed: int = 1729) -> list[int]:
    """Indices into ss (the first session of a 2-session control window), sorted."""
    busy = set()
    for r in lib:
        if not r.get("w1"):
            continue
        i = ss.searchsorted(pd.Timestamp(r["w1"]["start"]))
        busy.add(i + 1)                                  # its first window session
        if r.get("severity") in ("major", "extreme"):
            busy.update(range(i - 3, i + 5))
    cand = [i for i in range(first, ss.searchsorted(pd.Timestamp(before))) if i not in busy]
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(cand, min(n_target, len(cand)), replace=False))
