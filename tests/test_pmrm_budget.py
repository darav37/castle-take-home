"""SPEC §6.1 caps for experiment LLM calls: dry run first, per-run caps, permanent cache. No network."""

import pytest

from worsey.pmrm.ic import budget as B


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        assert kw.pop("_source") == "pmrm"
        self.calls += 1
        return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1000, "output_tokens": 200}}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "CACHE", tmp_path)
    monkeypatch.delenv("WORSEY_PMRM_MAX_CALLS_PER_RUN", raising=False)
    monkeypatch.delenv("WORSEY_PMRM_MAX_USD_PER_RUN", raising=False)


def test_unconfirmed_run_makes_no_call():
    llm = FakeLLM()
    run = B.Run("j", "claude-sonnet-5-5", llm)
    run.plan(10, 2000, 500)
    with pytest.raises(B.RunCapExceeded):
        run.call("k1", model="claude-sonnet-5-5", messages=[])
    assert llm.calls == 0


def test_cache_makes_a_rerun_free_and_the_call_cap_stops(monkeypatch):
    monkeypatch.setenv("WORSEY_PMRM_MAX_CALLS_PER_RUN", "2")
    llm = FakeLLM()
    run = B.Run("j", "claude-sonnet-5-5", llm, confirmed=True)
    run.call("a", model="claude-sonnet-5-5", messages=[])
    run.call("b", model="claude-sonnet-5-5", messages=[])
    with pytest.raises(B.RunCapExceeded):
        run.call("c", model="claude-sonnet-5-5", messages=[])
    again = B.Run("j", "claude-sonnet-5-5", llm, confirmed=True)
    again.call("a", model="claude-sonnet-5-5", messages=[])
    assert llm.calls == 2 and again.hits == 1


def test_plan_over_the_usd_cap_is_flagged():
    p = B.Run("j", "claude-sonnet-5-5").plan(5000, 20000, 4000)
    assert not p["within_caps"]
