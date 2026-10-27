"""In-chat train mode (/fix): trainer-only, active now for the session, permanent only after /approve, golden cases."""

import asyncio
import json
import time

import pytest

from worsey import server
from worsey.agent import llm as llm_mod
from worsey.agent import train_fix
from worsey.agent.llm import FakeLLM, response, text_block
from worsey.data.provider import SyntheticMarketData


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(train_fix, "QUEUE", tmp_path / "queue.jsonl")
    monkeypatch.setattr(train_fix, "GOLDEN", tmp_path / "golden.jsonl")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "train_fix.md").write_text("distil", encoding="utf-8")
    (tmp_path / "prompts" / "options_desk.md").write_text("You are options_desk.", encoding="utf-8")
    (tmp_path / "library" / "strategies").mkdir(parents=True)
    (tmp_path / "library" / "strategies" / "long_call.md").write_text("---\nid: long_call\n---\n## When it fits\n", encoding="utf-8")
    monkeypatch.setattr(train_fix, "ROOT", tmp_path)
    seen = []

    def fn(kw):
        seen.append(kw)
        sys_text = kw["system"] if isinstance(kw.get("system"), str) else ""
        if sys_text == "distil":
            return response(text_block(json.dumps({"rule": "For earnings views, compare to past earnings-day moves",
                                                   "expert": "options_desk", "topic": "earnings", "target": "strategy_file",
                                                   "strategy_id": "long_call", "restatement": "I'll compare to past earnings moves."})))
        return response(text_block("**Buy the call.**"))

    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda: FakeLLM(fn))
    monkeypatch.setattr(server.Runtime, "_data", lambda self: SyntheticMarketData())
    return server.Runtime(replay=None, portfolio=None, mention_only=False), seen, tmp_path


async def _settle(chat, n, timeout=10):
    t0 = time.time()
    while sum(1 for e in chat.history if e["type"] == "done") < n:
        assert time.time() - t0 < timeout
        await asyncio.sleep(0.02)


def test_only_the_trainer_can_fix(setup):
    rt, seen, _ = setup

    async def go():
        c = rt.chat("demo", "sam")
        await c.user_message("sam", "/fix that's wrong")
        await _settle(c, 1)
        return c
    c = asyncio.run(go())
    assert any("Only Worsey's trainer" in e.get("text", "") for e in c.history if e["type"] == "reply")
    assert not train_fix.QUEUE.exists()


def test_fix_is_active_now_reanswers_and_queues_a_permanent_edit(setup):
    rt, seen, _ = setup

    async def go():
        c = rt.chat("demo", "darin")
        await c.user_message("darin", "NVDA rips after earnings, what should I buy?")
        await _settle(c, 1)
        await c.user_message("darin", "/fix the implied earnings move is bigger than his view, compare to past earnings moves",
                             quote="worsey: **Buy the call.**")
        await _settle(c, 3)
        return c
    c = asyncio.run(go())
    got = [e for e in c.history if e["type"] == "reply" and e["text"].startswith("**Got it:**")]
    assert got and "/approve" in got[0]["text"] and got[0]["reply_to"]["user"] == "worsey"
    q = train_fix.pending()
    assert len(q) == 1 and q[0]["target"] == "strategy_file" and q[0]["strategy_id"] == "long_call"
    assert "past earnings" in train_fix.GOLDEN.read_text(encoding="utf-8")
    s = rt.members["darin"].session
    assert train_fix.session_rules(s, "options_desk") == ["For earnings views, compare to past earnings-day moves"]
    reanswer = [kw for kw in seen if isinstance(kw.get("system"), list) and any(
        "[the trainer's correction, apply it:" in str(m.get("content")) for m in kw["messages"])]
    assert reanswer                                                    # the original question ran again with the rule


def test_approve_writes_darins_words_into_the_file_reject_does_not(setup):
    rt, seen, root = setup
    c1 = train_fix.record("darin", "demo", "q", "bad", "options_desk", "compare to past earnings moves",
                          {"rule": "Compare to past earnings moves", "expert": "options_desk", "topic": "earnings",
                           "target": "strategy_file", "strategy_id": "long_call", "restatement": "ok"})
    c2 = train_fix.record("darin", "demo", "q", "bad", "options_desk", "be shorter",
                          {"rule": "Be shorter", "expert": "options_desk", "topic": "voice", "target": "prompt",
                           "strategy_id": None, "restatement": "ok"})
    a = train_fix.decide(c1["id"], approve=True)
    text = (root / "library" / "strategies" / "long_call.md").read_text(encoding="utf-8")
    assert train_fix.SECTION in text and 'the trainer: "compare to past earnings moves"' in text and a["status"] == "approved"
    train_fix.decide(c2["id"], approve=False)
    assert "Be shorter" not in (root / "prompts" / "options_desk.md").read_text(encoding="utf-8")
    assert train_fix.pending() == []
    with pytest.raises(KeyError):
        train_fix.decide(c2["id"], approve=True)                       # already decided


def test_pending_fixes_stay_active_for_the_trainer_across_restarts(setup):
    from worsey.agent.session import Session
    train_fix.record("darin", "demo", "q", "bad", "options_desk", "grab deltas",
                     {"rule": "Pick strikes by delta and compare structures", "expert": "options_desk", "topic": "strikes",
                      "target": "playbook", "strategy_id": None, "restatement": "ok"})
    fresh = Session(SyntheticMarketData(), member="darin")              # a new server: nothing in session memory
    assert train_fix.session_rules(fresh, "options_desk") == ["Pick strikes by delta and compare structures"]
    assert train_fix.session_rules(Session(SyntheticMarketData(), member="sam"), "options_desk") == []
    c = train_fix.pending()[0]
    train_fix.decide(c["id"], approve=False)
    assert train_fix.session_rules(fresh, "options_desk") == []        # rejected: no longer active
