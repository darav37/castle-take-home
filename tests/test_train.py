"""Train mode: the drill → propose_edit → review loop, offline, in a temp tree."""

import pytest

from worsey.agent import train
from worsey.agent.llm import FakeLLM, response, text_block, tool_use
from worsey.agent.session import Session, ToolError
from worsey.data.provider import SyntheticMarketData


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "ALLOWED", [tmp_path / "library", tmp_path / "examples" / "events"])
    monkeypatch.setattr(train, "QUEUE", tmp_path / "library" / "_proposals")
    return tmp_path


def test_propose_edit_is_confined(tree):
    s = Session(SyntheticMarketData())
    with pytest.raises(ToolError):
        train.propose_edit(s, "src/worsey/agent/loop.py", "x", "sneaky")
    with pytest.raises(ToolError):
        train.propose_edit(s, "library/../prompts/router.md", "x", "sneaky")
    assert train.propose_edit(s, "library/playbook/playbook_rules.yaml", "- id: r1\n", "from drill")["queued"]


def test_session_queues_then_review_accepts(tree):
    rule = "- id: view_inside_implied\n  status: approved\n"
    script = [
        response(text_block("First drill: NVDA +6% in 3 weeks. My pick is a call spread. Yours?")),
        response(tool_use("p1", "propose_edit", {"path": "library/playbook/playbook_rules.yaml", "content": rule,
                                                 "reason": "the trainer: spreads when the view is inside the implied move"})),
        response(text_block("Queued 1 rule.")),
    ]
    out = []
    train.run_train(FakeLLM(script), "options", data=SyntheticMarketData(), inputs=["spread, because IV is rich"], out=out.append)
    assert out[-1] == "Queued 1 rule." and len(train.pending()) == 1
    acc = train.review(ask=lambda _: "y", out=lambda *_: None, commit=False)
    assert acc == ["library/playbook/playbook_rules.yaml"]
    assert (tree / "library/playbook/playbook_rules.yaml").read_text() == rule
    assert train.pending() == []


def test_factor_moves_tool():
    s = Session(SyntheticMarketData())
    out = train.factor_moves(s, "2026-06-01")
    assert "SPY" in out["moves"] and "CREDIT" in out["moves"]
