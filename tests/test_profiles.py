"""Member profiles (SPEC §1e-1; DECISIONS #120): guardrails in code, behavior-based sophistication, DM-only view."""

from datetime import date, timedelta

import pytest

from worsey.agent import profiles, tools
from worsey.agent.session import Session
from worsey.data.provider import SyntheticMarketData
from worsey.options import builder
from worsey.options.view import View


@pytest.fixture(autouse=True)
def tmpdir(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "DIR", tmp_path)


def test_only_what_the_member_said_no_psychology_nothing_about_others():
    prof = profiles.load("darin")
    said = "keep it short please, and roast me when I'm wrong. Sam always panics btw"
    added = profiles.accept_lines(prof, [
        {"section": "preferences", "text": "Wants short replies", "quote": "keep it short please"},
        {"section": "opt_ins", "text": "Opted into roasts", "quote": "roast me when I'm wrong"},
        {"section": "voice", "text": "Seems anxious about losses", "quote": "keep it short"},          # psychology
        {"section": "knowledge", "text": "Trades vol for a living", "quote": "I trade vol"},           # never said
        {"section": "voice", "text": "Thinks Sam panics", "quote": "Sam always panics"},               # about another member
    ], said, other_members=["sam", "jo"])
    assert [l["text"] for l in added] == ["Wants short replies", "Opted into roasts"]


def test_sophistication_from_behavior_unlocks_gates():
    prof = profiles.load("darin")
    assert profiles.sophistication(prof)["level"] == "novice"
    profiles.track_behavior(prof, "is NVDA implied vol rich? what's the skew and theta like", ["long_call", "long_put"])
    assert profiles.sophistication(prof)["level"] == "intermediate"
    profiles.track_behavior(prof, "show me a straddle; how does delta hedging work", ["long_straddle"])
    s = profiles.sophistication(prof)
    assert s["level"] == "sophisticated" and s["unlocked_gates"] == ["sophisticated"]
    profiles.track_behavior(prof, "I have a margin account", [])
    assert "sophisticated_with_margin" in profiles.sophistication(prof)["unlocked_gates"]


def test_group_sees_tone_only_dm_sees_depth():
    prof = profiles.load("darin")
    profiles.accept_lines(prof, [{"section": "preferences", "text": "Wants short replies", "quote": "short"},
                                 {"section": "knowledge", "text": "New to options", "quote": "never traded options"}],
                          "short please, I've never traded options", [])
    assert profiles.for_model(prof, "group") == {"preferences": ["Wants short replies"]}
    dm = profiles.for_model(prof, "dm")
    assert dm["knowledge"] == ["New to options"] and dm["sophistication"] == "novice"


def test_builder_gate_opens_from_track_record_not_only_by_name():
    lib = [{"id": "plain", "name": "plain", "direction": ["up"], "vol_view": ["none"], "risk": "defined",
            "legs": [{"type": "call", "side": "long", "strike": "atm"}]},
           {"id": "fancy", "name": "fancy", "direction": ["up"], "vol_view": ["none"], "risk": "defined", "gate": "sophisticated",
            "legs": [{"type": "call", "side": "long", "strike": "atm"}, {"type": "call", "side": "short", "strike": "target"}]}]
    today = date(2026, 10, 14)
    q = builder.FlatQuoter(100.0, today, [today + timedelta(days=d) for d in (21, 42, 63)], lambda T, K: 0.4, step=1.0)
    v = View("NVDA", "up", today + timedelta(days=42), target_move_pct=10)
    locked = builder.propose(v, 100.0, today, q, 0.3, lib)
    assert [c.id for c in locked["candidates"]] == ["plain"] and locked["dropped"][0]["id"] == "fancy"
    opened = builder.propose(v, 100.0, today, q, 0.3, lib, unlocked={"sophisticated"})
    assert "fancy" in [c.id for c in opened["candidates"]]


def test_what_do_you_know_about_me_goes_to_dm_from_the_group():
    s = Session(SyntheticMarketData(), member="darin", channel="group")
    out = tools.my_profile(s)
    assert out["sent_to_dm"] and "What I know about you" in s.__dict__["dm_out"][0]
    s2 = Session(SyntheticMarketData(), member="darin", channel="dm")
    assert "What I know about you" in tools.my_profile(s2)["profile"]
