"""The web UI's event stream over a real WebSocket, replaying the s0 script offline (SPEC §1d-0).
Skips where there is no local build (cache/ is gitignored)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ws_streams_message_status_card_reply_done_in_order(monkeypatch):
    from worsey.build import latest
    if latest() is None or not (ROOT / "eval" / "recordings" / "demo_main.jsonl").exists():
        pytest.skip("no local build or recording")
    monkeypatch.setenv("WORSEY_OFFLINE", "1")   # restored on teardown (the server sets it for replay)
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(ROOT / "tests" / "fixtures" / "strategies"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from fastapi.testclient import TestClient
    from worsey.server import create_app
    with TestClient(create_app(replay="s0")) as client, client.websocket_connect("/ws?chat_id=t&user=darin") as ws:
        events = []
        for _ in range(2):  # load portfolio, then the scan
            ws.send_json({"type": "next"})
            while True:
                events.append(ws.receive_json())
                if events[-1]["type"] == "done":
                    break
    kinds = [e["type"] for e in events]
    scan = [e for e in events if e["type"] == "card" and e["card"] == "scan"]
    assert kinds[0] == "message" and events[0]["text"].startswith("@worsey")
    assert scan and scan[0]["payload"]["events"]
    i = kinds.index("card")
    assert "status" in kinds[:i] and kinds.index("reply", i) < len(kinds) - 1 and kinds[-1] == "done"
    assert events[-1]["route"][0]["entry"] == "scan_risks" and events[-1]["sources"]["mode"] == "replay"


def test_plain_message_without_mention_gets_no_answer_in_mention_only_mode(monkeypatch):
    from worsey.build import latest
    if latest() is None:
        pytest.skip("no local build")
    monkeypatch.setenv("WORSEY_OFFLINE", "1")
    monkeypatch.setenv("WORSEY_LIBRARY_DIR", str(ROOT / "tests" / "fixtures" / "strategies"))
    from fastapi.testclient import TestClient
    from worsey.server import create_app
    with TestClient(create_app(replay="s0", mention_only=True)) as client, client.websocket_connect("/ws?chat_id=u&user=sam") as ws:
        ws.send_json({"type": "message", "text": "anyone around?"})
        ws.send_json({"type": "message", "text": "lunch?"})
        first, second = ws.receive_json(), ws.receive_json()   # no status/reply between: worsey stayed quiet
    assert [first["text"], second["text"]] == ["anyone around?", "lunch?"] and first["user"] == "sam"


def test_hedge_after_reload_reprices_the_event_for_the_new_book(monkeypatch):
    # "about 60k" reloads the book, which clears the scan; the hedge tool must re-price the named market itself
    from worsey.build import latest
    if latest() is None:
        pytest.skip("no local build")
    monkeypatch.setenv("WORSEY_OFFLINE", "1")
    from worsey.agent import tools
    from worsey.agent.session import Session
    from worsey.data.provider import BuiltMarketData
    tag = (ROOT / "eval" / "recordings" / "QUOTE_TAG").read_text().strip()
    s = Session(BuiltMarketData(quote_tag=tag))
    tools.load_portfolio(s, holdings=[{"ticker": "IWM", "value_usd": 100000}])
    eid = next(c["event_id"] for c in tools.find_event(s, "Iran flare up")["candidates"] if c["priced"])
    before = tools.compare_hedges_tool(s, event_id=eid)["loss_if_yes"]
    tools.load_portfolio(s, holdings=[{"ticker": "IWM", "value_usd": 60000}])
    after = tools.compare_hedges_tool(s, event_id=eid)["loss_if_yes"]
    assert after == pytest.approx(before * 0.6, rel=1e-6)
