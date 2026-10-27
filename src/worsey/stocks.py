"""Single-stock risk for a book (docs/pmrm/SINGLE_STOCK_TRACK.md; DECISIONS #204).

The earnings calendar: every company the book holds, directly or through its ETFs (look-through), that reports in the
next weeks, with how much of the book rides on it, what options price for the move (the at-the-money straddle on the
first expiry after the report), and what the stock did on its last reports. Code makes every number.

News base rates (#205): for a kind of company news (an FDA decision, guidance change, analyst upgrade...), how big the
stock's move usually is, whether the headline's direction held, and whether it drifted after, from 60,909 Benzinga items
on S&P 500 names (2019-2026, away from earnings; results/stock_s0b.md), scaled to this stock and this book.

What the research says, and what the expert must say with it (results/stock_s0.md, stock_s2.md):
  - earnings are real events for a stock: the reaction day averages ~3.9x a normal day; most reports move > 2x
  - options price those moves about right: neither buying nor selling every straddle paid after costs (2,718 reports)
  - a stock's own history doesn't say when options are cheap or dear
  - after the report, the surprise calls direction only a little better than a coin flip (54% in 2026), with no drift
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import lookthrough

RESULTS = Path(__file__).resolve().parents[2] / "docs/pmrm/results"


@lru_cache(maxsize=1)
def _s3() -> dict:
    return json.loads((RESULTS / "stock_s3.json").read_text(encoding="utf-8"))


def after_open_single(s, ticker: str, kind: str) -> dict | None:
    """Once the stock has opened on the news (S3, #206): today's opening gap, and the range to the close. The gap sticks
    (the rest of the day is unrelated to it), so the middle is the open and the range is the rest of the day's usual
    swing for this kind of news. None when today's open isn't known (before 09:30, no session, replays)."""
    st = _s3().get(kind)
    gaps_fn = getattr(s.data, "opening_gaps", None)
    if st is None or gaps_fn is None:
        return None
    try:
        g = gaps_fn([ticker]).get(ticker)
        r = s.data.closes([ticker], days=260)[ticker].pct_change().dropna()
    except Exception:  # noqa: BLE001
        return None
    if g is None or r.empty:
        return None
    rest_sd = float(r.tail(252).std()) * st["rms_rest"]
    return {"todays_opening_gap": float(g), "middle_to_close": float(g),
            "range_9_in_10_to_close": [float(g - 1.645 * rest_sd), float(g + 1.645 * rest_sd)],
            "closes_on_the_gaps_side_after_big_gaps": st["day_close_same_sign_as_gap_big"],
            "how": f"after the open the gap usually sticks: the rest of the day is unrelated to it (slope "
                   f"{st['slope_rest_on_gap']:+.2f}, {st['events']:,} past events of this kind), and big gaps close on their "
                   f"side {st['day_close_same_sign_as_gap_big']:.0%} of the time"}


@lru_cache(maxsize=1)
def evidence() -> dict:
    s0 = json.loads((RESULTS / "stock_s0.json").read_text(encoding="utf-8"))
    s2 = json.loads((RESULTS / "stock_s2.json").read_text(encoding="utf-8"))
    hit = s0["surprise_direction"]["eps_surprise_pct"]
    return {
        "reaction_day_size": f"earnings reaction days average {s0['size']['rms_day1_vs_normal_day']:.1f}x a normal day's move; "
                             f"{s0['size']['share_over_2x']:.0%} move more than 2x ({s0['reports']:,} S&P 500 reports, 2011-2026)",
        "options_pricing": f"options price these moves about right: after costs, buying every straddle lost "
                           f"{-s2['test']['buy_every_straddle_pnl_per_premium']:.0%} of premium in 2026 and selling every one "
                           f"{-s2['test']['sell_every_straddle_pnl_per_premium']:.0%} ({s2['priced']:,} reports)",
        "history_vs_options": "a stock's own past moves don't say when its options are cheap or dear (tested on held-out reports)",
        "direction_after": f"after the report, the earnings surprise calls the stock's direction {hit['test']['direction_hit_all']:.0%} "
                           f"of the time in 2026 ({hit['train']['direction_hit_all']:.0%} before 2023), with no drift in the "
                           "following days",
        "after_the_open": f"once the stock has opened on the report, the gap sticks: the rest of the day is unrelated to it, "
                          f"but still swings about {_s3()['Earnings']['rms_rest']:.1f}x a normal day; big gaps close on their side "
                          f"{_s3()['Earnings']['day_close_same_sign_as_gap_big']:.0%} of the time",
    }


def _exposure(s) -> pd.DataFrame:
    vals = s.priced["value"]
    eq = s.priced.index[s.priced["asset_class"] == "equity"]
    hold = {t: h for t in eq if (h := s.data.holdings(t)) is not None and not h.empty}
    eff = lookthrough.effective_exposure(vals[eq], hold)
    eff["weight"] = eff["value"] / float(vals.sum())
    return eff[[not lookthrough.is_bucket(n) for n in eff.index]]


def _implied(s, name: str, report: pd.Timestamp, time: str | None, filled_chain, market_state) -> dict:
    """The move options price for the earnings day itself. With an expiry before the report and one after it, the extra
    variance of the later one is the event's (market_state.event_move_from_term; expected |move| = 0.8 x its sd).
    Without one, the at-the-money straddle on the first expiry after the report, which also prices the ordinary days to
    expiry, so it overstates the event (flagged)."""
    S = float(s.data.prices([name])[name])
    ch = filled_chain(s, name, S)
    ch = ch if "iv_mid" in ch else market_state.fill_iv(ch, S, s.data.today())
    term = market_state.atm_iv(ch, S)
    term["expiry"] = pd.to_datetime(term["expiry"])
    react = report + pd.Timedelta(days=1) if str(time or "") >= "16:00" else report      # the session that reacts
    pre, post = term[term.expiry < react], term[term.expiry >= react]
    if post.empty:
        return {"implied_move": None}
    p1 = post.iloc[0]
    pre = pre[np.isfinite(pre.iv) & (pre["T"] > 1.5 / 365)]
    if not pre.empty and np.isfinite(p1.iv):
        sd = market_state.event_move_from_term(float(pre.iloc[-1].iv), float(p1.iv), float(p1["T"]))
        if sd > 0:
            return {"implied_move": 0.8 * sd, "implied_how": f"event-day move from the {pd.Timestamp(pre.iloc[-1].expiry).date()} and "
                    f"{pd.Timestamp(p1.expiry).date()} expiries (expected size of the move, either way)"}
    if len(post) >= 2 and np.isfinite(post.iloc[1].iv) and np.isfinite(p1.iv) and post.iloc[1]["T"] > p1["T"]:
        # both expiries hold the event: total variance_i = s_d^2 * T_i + event variance, so two of them give both
        p2 = post.iloc[1]
        sd2 = (p2.iv ** 2 * p2["T"] - p1.iv ** 2 * p1["T"]) / (p2["T"] - p1["T"])
        ev = p1.iv ** 2 * p1["T"] - max(sd2, 0.0) * p1["T"]
        if ev > 0:
            return {"implied_move": 0.8 * float(np.sqrt(ev)), "implied_how": f"event-day move from the {pd.Timestamp(p1.expiry).date()} "
                    f"and {pd.Timestamp(p2.expiry).date()} expiries, both after the report (expected size, either way)"}
    im = float(market_state.implied_move(ch, S, p1.expiry))
    return {"implied_move": im, "implied_how": f"at-the-money straddle to {pd.Timestamp(p1.expiry).date()}: includes the ordinary "
            "days to expiry, so it overstates the earnings move itself"}


def earnings_calendar(s, days: int = 45, top: int = 25, min_weight: float = 0.005) -> dict:
    """Names worth at least `min_weight` of the book (after look-through) reporting within `days`."""
    from .agent.tools import filled_chain
    from .options import market_state
    s.require_portfolio()
    if not hasattr(s.data, "earnings"):
        return {"rows": [], "note": "no earnings calendar on this data source"}
    today = pd.Timestamp(s.data.today())
    eff = _exposure(s).head(top)
    eff = eff[eff["weight"] >= min_weight]
    book = float(s.priced["value"].sum())
    rows, skipped = [], []
    for name, r in eff.iterrows():
        try:
            e = s.data.earnings(name)
        except Exception:  # noqa: BLE001
            skipped.append(name)
            continue
        nxt = (e or {}).get("next")
        if not nxt or not nxt.get("date") or (pd.Timestamp(nxt["date"]) - today).days > days:
            continue
        past = [abs(m["move"]) for m in (e.get("past_moves") or [])][:8]
        row = {"ticker": name, "reports": nxt["date"], "time": nxt.get("time"), "confirmed": nxt.get("status") == "confirmed",
               "your_exposure_usd": float(r["value"]), "share_of_book": float(r["weight"]),
               "held_via": {k: float(v) for k, v in (r.get("via") or {}).items()},
               "past_median_move": float(np.median(past)) if past else None, "past_moves": past}
        try:
            row.update(_implied(s, name, pd.Timestamp(nxt["date"]), nxt.get("time"), filled_chain, market_state))
            if row.get("implied_move"):
                row["your_dollars_at_implied_move"] = row["implied_move"] * float(r["value"])
        except Exception:  # noqa: BLE001
            row["implied_move"] = None
        if row.get("implied_move") and row["past_median_move"]:
            row["options_vs_history"] = row["implied_move"] / row["past_median_move"]
            if not 1 / 3 <= row["options_vs_history"] <= 3:
                row["data_check"] = ("the options' number is far from this stock's history; short-dated quotes may be stale or "
                                     "thin, so lean on the history and say so")
        rows.append(row)
    rows.sort(key=lambda x: x["reports"])
    at_stake = sum(x.get("your_dollars_at_implied_move") or 0.0 for x in rows)
    return {"window_days": days, "book_value": book, "rows": rows,
            "total_dollars_at_implied_moves": at_stake,
            "names_checked": int(len(eff)), "names_without_data": skipped,
            "evidence": evidence(),
            "how_to_read": "implied move = the size of move options price for the earnings day itself (see implied_how per "
                           "name); your dollars = your exposure (including through ETFs) x that move, either way",
            "numbers": "every number is code's; quote them, never adjust them"}


NEWS_TYPES = {"FDA": "an FDA decision or filing", "M&A": "a merger or acquisition", "Guidance": "a guidance change",
              "Upgrades": "an analyst upgrade", "Downgrades": "an analyst downgrade", "Legal": "a lawsuit, probe or ruling",
              "Offerings": "a stock or debt offering", "Buybacks": "a buyback", "Management": "a management change",
              "Rumors": "a rumor", "Contracts": "a contract or partnership", "Dividends": "a dividend change",
              "Initiation": "a new analyst initiation"}


@lru_cache(maxsize=1)
def _news_stats() -> dict:
    return json.loads((RESULTS / "stock_s0b.json").read_text(encoding="utf-8"))["by_channel"]


def news_base_rates(s, news_type: str, ticker: str | None = None) -> dict:
    """What news of this kind usually does to a stock's price beyond the market (timing-clean items: published before the
    open or after the close), scaled to `ticker`'s normal daily swing and the book's exposure to it."""
    st = _news_stats().get(news_type)
    if st is None:
        return {"error": f"news_type must be one of {sorted(NEWS_TYPES)}"}
    c = st["clean_timing"]
    out = {"news_type": NEWS_TYPES[news_type], "events_studied": c["events"],
           "typical_move_x_normal_day": c["median_abs_day1"], "big_move_share_over_2x_normal": c["share_over_2x"],
           "normal_day_share_over_2x": 0.046,
           "headline_direction_held": c["direction_hit_signed"], "headline_direction_held_2026": c["direction_hit_2026"],
           "headline_direction_held_on_big_moves": c["direction_hit_signed_moves_over_2x"],
           "after_the_day": ("tends to give some back over the next 5 sessions" if (c["drift_next5_in_headline_direction"] or 0) < -0.08
                             else "no consistent drift over the next 5 sessions"),
           "caveat": ("most of the move happens at the open; by the time a user reads the headline it is usually priced. "
                      "Direction for this news type isn't carried by the headline" if c["direction_hit_signed"] is None else
                      "most of the move happens at the open; by the time a user reads the headline it is usually priced")}
    if ticker:
        t = ticker.upper()
        try:
            r = s.data.closes([t], days=260)[t].pct_change().dropna()
            day = float(r.tail(252).std())
            out["ticker"] = t
            out["normal_daily_swing"] = day
            out["typical_move_pct"] = day * c["median_abs_day1"]
        except Exception:  # noqa: BLE001
            pass
        if getattr(s, "priced", None) is not None:          # a loaded book: its exposure to the name, counting ETFs
            try:
                eff = _exposure(s)
                if t in eff.index:
                    out["your_exposure_usd"] = float(eff.at[t, "value"])
                    if out.get("typical_move_pct"):
                        out["your_dollars_at_typical_move"] = out["your_exposure_usd"] * out["typical_move_pct"]
            except Exception:  # noqa: BLE001
                pass
    if ticker and (ao := after_open_single(s, ticker.upper(), news_type)):
        out["after_the_open"] = ao
    out["numbers"] = "every number is code's, from the S0b and S3 studies (DECISIONS #205, #206); quote them, never adjust them"
    return out
