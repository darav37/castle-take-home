"""Freeze the country layer's numbers for the live chat (DECISIONS #238). No LLM calls.

  python scripts/country_pack.py     # -> src/worsey/planb/countries.json

Per country: its ETF, and how big its country factor's 2-session moves are (the ETF's daily move beyond the 13 macro
factors, in its own noise): the median, the 90th and the 95th percentile of |move|, over all the daily data the live
feed has (2016-2026). The same convention as the sector pack (#217), so a country leg and a sector leg are comparable.
Broad multi-country ETFs (EEM, EFA, VGK) are not countries: the EM factor already carries emerging markets as a whole.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

COUNTRIES = {"Brazil": "EWZ", "Mexico": "EWW", "Argentina": "ARGT", "Chile": "ECH", "Peru": "EPU", "Canada": "EWC",
             "Japan": "EWJ", "China": "MCHI", "Hong Kong": "EWH", "Taiwan": "EWT", "South Korea": "EWY", "India": "INDA",
             "Australia": "EWA", "Singapore": "EWS", "Indonesia": "EIDO", "Malaysia": "EWM", "Thailand": "THD",
             "Germany": "EWG", "United Kingdom": "EWU", "France": "EWQ", "Italy": "EWI", "Spain": "EWP", "Switzerland": "EWL",
             "Netherlands": "EWN", "Sweden": "EWD", "Poland": "EPOL", "Greece": "GREK", "Turkey": "TUR", "Israel": "EIS",
             "Saudi Arabia": "KSA", "South Africa": "EZA"}


def main():
    from worsey.cli import _data
    from worsey.planb import live
    d = _data(argparse.Namespace(synthetic=False, quote_tag=None))
    tick = list(dict.fromkeys(list(COUNTRIES.values()) + [t for legs in live.PROXIES.values() for t, _ in legs]))
    cl = d.closes(tick, days=2700)
    r = cl.pct_change().iloc[1:]
    F = live.factor_returns(r).reindex(columns=live.FACTORS)
    out = {"source": "country factors: each country ETF's daily move beyond the 13 macro factors (#238)",
           "since": str(r.index[0].date()), "countries": {}}
    for name, etf in COUNTRIES.items():
        dd = pd.concat([r[etf].rename("y"), F], axis=1).dropna()
        A = np.column_stack([np.ones(len(dd)), dd[live.FACTORS].fillna(0).to_numpy()])
        c = np.linalg.lstsq(A, dd["y"].to_numpy(), rcond=None)[0]
        e = pd.Series(dd["y"].to_numpy() - A @ c, index=dd.index)
        x = e / e.std()
        two = ((x + x.shift(-1)).dropna() / np.sqrt(2)).abs()
        out["countries"][name] = {"etf": etf, "two_day_move_median": round(float(two.median()), 3),
                                  "two_day_move_p90": round(float(two.quantile(0.9)), 3), "two_day_move_p95": round(float(two.quantile(0.95)), 3),
                                  "residual_daily_sd": round(float(e.std()), 5), "macro_r2": round(float(1 - e.var() / dd["y"].var()), 3)}
    (ROOT / "src/worsey/planb/countries.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["etf"], v["two_day_move_median"], v["two_day_move_p90"], v["residual_daily_sd"], v["macro_r2"]) for k, v in out["countries"].items()}))


if __name__ == "__main__":
    main()
