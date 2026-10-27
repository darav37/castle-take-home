"""Replay both Tier-A demo scripts offline: recorded LLM responses + the local build/cache, no network.
Skips where there is no local build (cache/ is gitignored); the trainer's machine has one."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUOTE_TAG = (Path(__file__).resolve().parent / "recordings" / "QUOTE_TAG").read_text().strip()     if (Path(__file__).resolve().parent / "recordings" / "QUOTE_TAG").exists() else "20261016T07"


@pytest.mark.parametrize("name,portfolio", [("demo_main", "balanced"), ("demo_iran", None), ("demo_tariffs", None)])
def test_demo_replays_offline(name, portfolio, monkeypatch):
    from worsey.build import latest
    rec = ROOT / "eval" / "recordings" / f"{name}.jsonl"
    if latest() is None or not rec.exists():
        pytest.skip("no local build or recording")
    from worsey import config
    monkeypatch.setattr(config, "SCENARIO_ENGINE", "v1")     # the recordings were made with the v1 engine (DECISIONS #179)
    monkeypatch.setenv("WORSEY_OFFLINE", "1")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)     # replays must not need keys
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(ROOT / "tests" / "fixtures" / "strategies"))
    from worsey.agent.llm import ReplayLLM
    from worsey.agent.loop import run_turn
    from worsey.agent.session import Session
    from worsey.data.provider import BuiltMarketData
    s = Session(BuiltMarketData(quote_tag=QUOTE_TAG))
    if portfolio:
        from worsey.agent import tools
        tools.load_portfolio(s, name=portfolio)
    llm = ReplayLLM(rec)
    for line in (ROOT / "eval" / "scripts" / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            out = run_turn(llm, s, line)
            assert out["reply"] and out["grounding"]["ok"], (line, out["grounding"])
            #: the card carries the numbers; replies are short, table-free, and hedge figures use the
            # card's convention (your book after the hedge)
            fmt = out["grounding"]["format"]
            assert fmt["ok"] and not fmt.get("enforced"), (line, fmt, out["reply"])
            if "compare_hedges" in out["tools"]:
                from worsey.agent.grounding import hedge_mismatches
                assert hedge_mismatches(out["reply"], s.last_hedges) == [], (line, out["reply"])
