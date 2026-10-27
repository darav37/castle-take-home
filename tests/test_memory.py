"""Accumulated knowledge (DECISIONS #114): code-written log, code-computed outcomes, gated expiring lessons."""

import time
from datetime import timedelta

import pandas as pd
import pytest

from worsey.agent import memory, tools
from worsey.agent.session import Session
from worsey.data.provider import SyntheticMarketData

FIX = __import__("pathlib").Path(__file__).parent / "fixtures" / "strategies"


@pytest.fixture(autouse=True)
def files(tmp_path, monkeypatch):
    for name in ("LOG", "BOOKS", "LESSONS", "SEED"):
        monkeypatch.setattr(memory, name, tmp_path / f"{name.lower()}.jsonl")


def sess():
    s = Session(SyntheticMarketData(), member="darin")
    s.__dict__["remember"] = True
    return s


def test_a_strategy_turn_is_logged_with_dated_legs_and_spot(monkeypatch):
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(FIX))
    s = sess()
    tools.view_and_strategies(s, underlying="NVDA", direction="up", target_move_pct=8, horizon_days=42, confirmed=True)
    s.log.append({"tool": "view_and_strategies", "input": {}, "ok": True})
    new = memory.record_turn(s, "[darin to Worsey]: NVDA up 8%. I bought the call spread last week")
    kinds = [e["kind"] for e in new]
    assert kinds == ["strategy", "member_said"]
    st = new[0]
    assert st["underlying"] == "NVDA" and st["spot"] and st["candidates"][0]["legs"][0]["expiry"] > st["date"]
    assert "bought the call spread" in new[1]["text"]


def test_expired_option_outcome_is_computed_from_the_close_at_expiry():
    s = sess()
    closes = s.data.closes(["NVDA"], days=400)["NVDA"].dropna()
    d0, d1 = closes.index[-120], closes.index[-60]
    k = round(float(closes.loc[d0]))
    e = {"id": "x1", "member": "darin", "kind": "strategy", "underlying": "NVDA", "date": str(d0.date()), "spot": float(closes.loc[d0]),
         "candidates": [{"id": "long_call", "cost": 500.0,
                         "legs": [{"kind": "call", "qty": 1, "strike": k, "expiry": str(d1.date())}]}]}
    out = memory.outcome(e, s.data)
    c = out["candidates"][0]
    want = 100 * max(float(closes.loc[d1]) - k, 0) - 500.0
    assert c["status"] == "expired" and c["pnl_at_expiry"] == pytest.approx(want)
    assert out["underlying_move"] == pytest.approx(float(closes.iloc[-1]) / float(closes.loc[d0]) - 1, rel=1e-6)


def test_lessons_need_three_real_cited_outcomes_and_expire():
    log = [{"id": f"e{i}", "outcome": {"x": 1}} for i in range(3)] + [{"id": "e9"}]
    kept = memory.accept_lessons([
        {"text": "Short-dated calls into earnings lost money for this group", "kind": "general", "cites": ["e0", "e1", "e2"]},
        {"text": "Too few cites", "cites": ["e0", "e1"]},
        {"text": "Cites without outcomes", "cites": ["e0", "e1", "e9"]},
        {"text": "Vol is cheap this month", "kind": "regime", "cites": ["e0", "e1", "e2"]},
    ], log)
    assert [l["text"][:5] for l in kept] == ["Short", "Vol i"]
    gen, reg = kept
    assert reg["expires"] - reg["created"] == pytest.approx(86400 * memory.TTL_REGIME_D)
    assert gen["expires"] - gen["created"] == pytest.approx(86400 * memory.TTL_GENERAL_D)
    memory.LESSONS.write_text(__import__("json").dumps([dict(gen, expires=time.time() - 1), reg]))
    assert [l["kind"] for l in memory.lessons()] == ["regime"]       # expired lessons drop out


def test_book_history_is_code_state_with_value_then_and_now():
    s = sess()
    tools.load_portfolio(s, holdings=[{"ticker": "SPY", "value_usd": 50000}, {"ticker": "NVDA", "value_usd": 50000}])
    hist = tools.book_history(s)["snapshots"]
    assert hist and hist[0]["value_then"] == pytest.approx(100000) and hist[0]["positions"]["NVDA"] == pytest.approx(50000)


def test_recall_returns_relevant_entries_with_outcomes_first():
    s = sess()
    memory._append(memory.LOG, {"id": "a", "member": "darin", "kind": "event", "event_id": "FED-CUT-DEC", "p": 0.3, "ts": 1})
    memory._append(memory.LOG, {"id": "b", "member": "darin", "kind": "event", "event_id": "TARIFF-25-MAR", "p": 0.4, "ts": 2})
    memory._append(memory.LOG, {"id": "c", "member": "sam", "kind": "event", "event_id": "FED-CUT-DEC", "p": 0.3, "ts": 3})
    out = tools.recall(s, "what about the FED-CUT-DEC market?")
    assert [e["id"] for e in out["episodes"]][:2] == ["a", "b"] and "outcome" in out["episodes"][0]
    assert all(e["member"] == "darin" for e in out["episodes"])      # only this member's history
