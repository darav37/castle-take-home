"""LLM clients: live (Anthropic SDK), record, replay, and a scripted fake for tests.

All return plain dicts (response.to_dict()), and the loop works on dicts, so recordings
replay byte-for-byte offline (SPEC §11a: record once, replay for CI and rehearsal).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..config import EXPERT_MODEL, ROUTER_MODEL  # noqa: F401  (models live in config; env-overridable)


def _key(kwargs: dict) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:24]


class AnthropicLLM:
    """Live calls. Reads ANTHROPIC_API_KEY from the environment (.env via dotenv in the CLI).
    Every call passes the budget gate first and is written to the usage ledger after (agent/budget.py)."""

    def __init__(self):
        import threading

        import anthropic
        self.client = anthropic.Anthropic(max_retries=1, timeout=120.0)   # no retry storms, no hung calls
        self.turn_calls, self._lock = 0, threading.Lock()

    def start_turn(self) -> None:
        with self._lock:
            self.turn_calls = 0

    def create(self, **kw) -> dict:
        from . import budget
        source = kw.pop("_source", None)   # ledger tag, never sent to the API
        with self._lock:   # experts run in parallel threads: count and gate atomically
            budget.gate(self.turn_calls)
            self.turn_calls += 1
        resp = None
        try:
            if kw.get("model", "").startswith("claude-sonnet-5-5"):
                # server-side refusal fallback, routed by category (Claude API only)
                resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw).to_dict()
            else:
                resp = self.client.messages.create(**kw).to_dict()
            return resp
        finally:
            budget.record(kw.get("model", ""), resp, ok=resp is not None, source=source)


class RecordingLLM:
    def __init__(self, inner, path: Path):
        self.inner, self.path = inner, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_turn(self) -> None:
        getattr(self.inner, "start_turn", lambda: None)()

    def timeout_check(self, role: str, timed_out: bool) -> bool:
        """Wall-clock caps are nondeterministic; record each decision so replays take the same path."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"timeout": timed_out, "role": role}) + "\n")
        return timed_out

    def create(self, **kw) -> dict:
        resp = self.inner.create(**kw)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": _key(kw), "response": resp}) + "\n")
        return resp


class ReplayLLM:
    """Serves recorded responses by request hash; a miss means the conversation diverged."""

    def __init__(self, path: Path):
        self.book, self.timeouts = {}, {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "timeout" in r:
                self.timeouts.setdefault(r.get("role"), []).append(r["timeout"])
            else:
                self.book.setdefault(r["key"], []).append(r["response"])

    def timeout_check(self, role: str, timed_out: bool) -> bool:
        """Replays reuse the recorded cap decisions, not the (faster) replay clock."""
        q = self.timeouts.get(role)
        return q.pop(0) if q else False

    def create(self, **kw) -> dict:
        k = _key(kw)
        if not self.book.get(k):
            raise KeyError(f"replay miss {k}: the conversation or prompts changed since recording")
        return self.book[k].pop(0)


class FakeLLM:
    """Scripted responses for tests: a list of dicts, or a function(kwargs) -> dict."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def create(self, **kw) -> dict:
        self.calls.append(kw)
        return self.script(kw) if callable(self.script) else self.script.pop(0)


def text_block(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def response(*blocks, stop: str | None = None) -> dict:
    stop = stop or ("tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn")
    return {"role": "assistant", "content": list(blocks), "stop_reason": stop}
