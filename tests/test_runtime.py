"""Multi-user runtime (SPEC §1e; DECISIONS #109, #110, #126), offline: fake model with delays, synthetic market."""

import asyncio
import json
import threading
import time

import pytest

from worsey import server
from worsey.agent import llm as llm_mod
from worsey.agent import tools
from worsey.agent.llm import FakeLLM, response, text_block, tool_use
from worsey.data.provider import SyntheticMarketData


def _router_text(kw):
    m = next(m for m in reversed(kw["messages"]) if m["role"] == "user" and isinstance(m["content"], str))
    return m["content"]


def _own(kw):
    """Only the member's own request line (the prompt also carries overheard context from others)."""
    return _router_text(kw).rsplit("to Worsey]:", 1)[-1]


@pytest.fixture
def rt(monkeypatch):
    """A live-mode runtime whose members get a scripted fake model and the synthetic market."""
    seen, starts, lock = [], [], threading.Lock()

    def fn(kw):
        text = _own(kw) if kw["messages"] and isinstance(kw["messages"][-1]["content"], str) else ""
        with lock:
            seen.append(kw)
            starts.append((text, time.time()))
        if "slow" in text:
            time.sleep(0.6)
        elif "wait" in text:
            time.sleep(0.3)
        if "load my book" in text and not any(m["role"] == "user" and isinstance(m["content"], list) for m in kw["messages"][-1:]):
            return response(tool_use("t1", "load_portfolio", {"holdings": [{"ticker": "SPY", "value_usd": 150000}]}))
        if isinstance(kw["messages"][-1]["content"], list):   # a tool result came back: answer from it
            res = json.loads(kw["messages"][-1]["content"][0]["content"])
            return response(text_block(f"**Loaded.** Total ${res.get('total_value', 0):,.0f}."))
        return response(text_block(f"**ok** ({text[-12:]})"))

    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda: FakeLLM(fn))
    monkeypatch.setattr(server.Runtime, "_data", lambda self: SyntheticMarketData())
    r = server.Runtime(replay=None, portfolio=None, mention_only=False)
    r.seen, r.starts = seen, starts
    return r


async def _settle(chat, n_done, timeout=10.0):
    t0 = time.time()
    while sum(1 for e in chat.history if e["type"] == "done") < n_done:
        assert time.time() - t0 < timeout, "timed out waiting for replies"
        await asyncio.sleep(0.02)


def test_two_members_run_in_parallel_and_the_slow_reply_quotes_and_mentions(rt):
    async def go():
        c = rt.chat("demo", "darin")
        t0 = time.time()
        await c.user_message("darin", "this one is slow")
        await c.user_message("sam", "quick one")
        await _settle(c, 2)
        return c, time.time() - t0
    c, took = asyncio.run(go())
    replies = [e for e in c.history if e["type"] == "reply"]
    assert took < 1.4                                       # 0.6s + quick ran side by side, not back to back (1.2s+)
    assert replies[0]["text"].endswith("quick one)") and replies[0]["reply_to"] is None   # answered the latest message
    slow = replies[1]
    assert slow["reply_to"] == {"user": "darin", "text": "this one is slow"}   # out of order: quote + @mention
    assert {e["rid"] for e in replies} == {f"demo-1", f"demo-2"}


def test_a_members_follow_up_waits_for_their_own_request(rt):
    async def go():
        c = rt.chat("demo", "darin")
        await c.user_message("darin", "please wait for me")
        await c.user_message("darin", "actually make it 2 weeks")
        await _settle(c, 2)
        return c
    asyncio.run(go())
    order = [t for t, _ in rt.starts]
    first = next(i for i, t in enumerate(order) if "wait for me" in t)
    second = next(i for i, t in enumerate(order) if "make it 2 weeks" in t)
    assert second > first and rt.starts[second][1] >= rt.starts[first][1] + 0.29   # no cancel: it waited


def test_group_sees_a_book_scaled_to_100k_and_real_dollars_go_to_dm(rt):
    async def go():
        c = rt.chat("demo", "darin")
        await c.user_message("darin", "load my book please")
        await _settle(c, 1)
        return c
    c = asyncio.run(go())
    reply = next(e for e in c.history if e["type"] == "reply")
    assert "$100,000" in reply["text"] and "150,000" not in reply["text"]      # the model only ever saw $100k
    s = rt.members["darin"].session
    assert float(s.priced_real["value"].sum()) == pytest.approx(150000) and float(s.priced["value"].sum()) == pytest.approx(100000)
    dm = rt.chats["dm:darin"].history
    assert dm and "1.50" in dm[0]["text"] and "150,000" in dm[0]["text"]       # code-written, private
    tools.apply_channel(s.__class__(s.data)) if False else None
    s.channel = "dm"
    tools.apply_channel(s)
    assert float(s.priced["value"].sum()) == pytest.approx(150000)             # in the DM: real dollars


def test_each_member_has_their_own_state_and_never_sees_anothers_book(rt):
    async def go():
        c = rt.chat("demo", "darin")
        await c.user_message("sam", "load my book please")
        await _settle(c, 1)
        await c.user_message("darin", "what do I hold?")
        await _settle(c, 2)
    asyncio.run(go())
    assert rt.members["sam"].session is not rt.members["darin"].session
    assert rt.members["darin"].session.portfolio is None
    darin_calls = [kw for kw in rt.seen if "what do I hold" in json.dumps(kw["messages"], default=str)]
    assert darin_calls and all("150000" not in json.dumps(kw["messages"], default=str) for kw in darin_calls)


def test_overheard_context_is_given_but_marked_never_to_quote(rt):
    async def go():
        c = rt.chat("demo", "darin")
        await c.user_message("sam", "I bet CPI comes in hot")
        await _settle(c, 1)
        await c.user_message("darin", "settle it")
        await _settle(c, 2)
    asyncio.run(go())
    router = [kw for kw in rt.seen if isinstance(kw.get("system"), list) and "front desk" in kw["system"][0]["text"]]
    last = [kw for kw in router if "settle it" in _router_text(kw)][-1]
    t = _router_text(last)
    assert "never quote it" in t and "sam: I bet CPI comes in hot" in t and t.rstrip().endswith("[darin to Worsey]: settle it")


def test_self_correction_posts_quoting_the_original(rt):
    def fn(kw):
        if isinstance(kw["messages"][-1]["content"], list):
            return response(text_block("**Fixed it.**"))
        return response(tool_use("c", "correct_previous", {"original_excerpt": "NVDA reports Nov 18",
                                                           "correction": "NVDA reports Nov 19, not Nov 18."}))
    rt.members.clear()
    import worsey.agent.llm as L
    L.AnthropicLLM = lambda: FakeLLM(fn)
    async def go():
        c = rt.chat("demo", "darin")
        await c.user_message("darin", "anything to fix?")
        await _settle(c, 2)
        return c
    c = asyncio.run(go())
    fix = [e for e in c.history if e["type"] == "reply" and e["text"].startswith("**Correction:**")]
    assert fix and fix[0]["reply_to"] == {"user": "worsey", "text": "NVDA reports Nov 18"}


def test_lan_mode_requires_the_access_code(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(server.Runtime, "_data", lambda self: SyntheticMarketData())
    app = server.create_app(replay=None, key="s3cret")
    with TestClient(app) as c:
        assert c.get("/").status_code == 401 and c.get("/config").status_code == 401
        r = c.get("/?key=s3cret")
        assert r.status_code == 200 and c.cookies.get("worsey_key") == "s3cret"
        assert c.get("/config").status_code == 200                    # the cookie carries it from here on
    with TestClient(app) as c2:
        with pytest.raises(Exception):
            with c2.websocket_connect("/ws?chat_id=x&user=darin") as ws:
                ws.receive_text()
