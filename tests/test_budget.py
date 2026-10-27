"""Live-call limits: every live call is gated and recorded; nothing here touches the network."""

import json
import time

import pytest

from worsey.agent import budget
from worsey.agent.llm import AnthropicLLM


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "USAGE", tmp_path)
    monkeypatch.setattr(budget, "LEDGER", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr(budget, "STOP", tmp_path / "STOP")
    for k in ("WORSEY_LLM_MAX_PER_TURN", "WORSEY_LLM_MAX_PER_HOUR", "WORSEY_LLM_MAX_PER_DAY", "WORSEY_LLM_MAX_TOKENS_DAY",
              "WORSEY_LLM_MAX_USD_DAY", "WORSEY_LLM_MAX_USD_TOTAL",
              "WORSEY_LLM_DISABLED"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path / "llm_calls.jsonl"


class _FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        return type("R", (), {"to_dict": lambda self_: {"content": [], "usage": {"input_tokens": 100, "output_tokens": 20}}})()


def _client():
    llm = AnthropicLLM.__new__(AnthropicLLM)   # skip the real SDK client
    import threading
    llm.turn_calls, llm._lock = 0, threading.Lock()
    msgs = _FakeMessages()
    llm.client = type("C", (), {"messages": msgs, "beta": type("B", (), {"messages": msgs})()})()
    return llm, msgs


def test_every_live_call_is_recorded_in_the_ledger(ledger):
    llm, msgs = _client()
    llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert msgs.calls == 1 and len(rows) == 1 and rows[0]["in"] == 100 and rows[0]["out"] == 20


def test_per_turn_limit_stops_a_runaway_loop(monkeypatch):
    monkeypatch.setenv("WORSEY_LLM_MAX_PER_TURN", "3")
    llm, msgs = _client()
    llm.start_turn()
    for _ in range(3):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    with pytest.raises(budget.BudgetExceeded):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert msgs.calls == 3            # the 4th call never reached the API
    llm.start_turn()                  # a new turn gets a fresh per-turn count
    llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])


def test_hourly_daily_and_token_limits_survive_a_restart(ledger, monkeypatch):
    now = time.time()
    ledger.write_text("\n".join(json.dumps({"t": now - 60, "model": "m", "ok": True, "in": 10, "out": 1}) for _ in range(5)))
    monkeypatch.setenv("WORSEY_LLM_MAX_PER_HOUR", "5")
    llm, msgs = _client()             # a "restarted" client still sees the 5 calls on disk
    with pytest.raises(budget.BudgetExceeded, match="hourly"):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    monkeypatch.setenv("WORSEY_LLM_MAX_PER_HOUR", "100")
    monkeypatch.setenv("WORSEY_LLM_MAX_USD_DAY", "0.0000001")      # the daily cap is in estimated dollars now
    with pytest.raises(budget.BudgetExceeded, match="today"):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert msgs.calls == 0


def test_kill_switch_env_and_stop_file(ledger, monkeypatch):
    llm, msgs = _client()
    monkeypatch.setenv("WORSEY_LLM_DISABLED", "1")
    with pytest.raises(budget.BudgetExceeded, match="switched off"):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    monkeypatch.delenv("WORSEY_LLM_DISABLED")
    (ledger.parent / "STOP").write_text("")
    with pytest.raises(budget.BudgetExceeded):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert msgs.calls == 0


def test_failed_calls_are_recorded_too(ledger):
    llm, msgs = _client()

    def boom(**kw):
        raise RuntimeError("network")
    msgs.create = boom
    with pytest.raises(RuntimeError):
        llm.create(model="claude-haiku-5-5", max_tokens=10, messages=[])
    assert json.loads(ledger.read_text().splitlines()[0])["ok"] is False


def test_project_total_cap_stops_calls_whatever_the_daily_cap(ledger, monkeypatch):
    old = time.time() - 5 * 86400            # spend from earlier days counts toward the project total
    ledger.write_text("\n".join(json.dumps({"t": old, "model": "claude-sonnet-5-5", "ok": True, "fresh": 1_000_000, "out": 0}) for _ in range(3)))
    monkeypatch.setenv("WORSEY_LLM_MAX_USD_DAY", "1000")
    monkeypatch.setenv("WORSEY_LLM_MAX_USD_TOTAL", "8")       # 3 x $3 = $9 already spent
    llm, msgs = _client()
    with pytest.raises(budget.BudgetExceeded, match="total"):
        llm.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    assert msgs.calls == 0
