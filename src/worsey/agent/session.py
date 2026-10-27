"""Session state (SPEC §1): what "that one" and "make it 2 weeks" resolve against."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..data.provider import MarketData
from ..portfolio import Portfolio

# the event pipeline's cards; on the live Plan B chat they're replaced by text messages (the options and ETF desks keep theirs)
TEXT_ONLY_CARDS = {"risks", "scan", "scenario", "stress", "hedge", "sources", "upcoming", "earnings", "meta", "template"}

@dataclass
class Session:
    data: MarketData
    portfolio: Portfolio | None = None
    priced: pd.DataFrame | None = None
    last_scan: dict | None = None
    active_event: str | None = None
    view: object | None = None             # options.view.View
    last_strategies: dict | None = None
    last_hedges: dict | None = None
    history: list = field(default_factory=list)      # router messages, append-only
    turn_results: list = field(default_factory=list)  # every tool result this turn (for grounding)
    turn_replies: list = field(default_factory=list)  # expert-written replies this turn (DECISIONS #100)
    replies_fresh: bool = False                       # True when the turn's last tool produced turn_replies
    member: str = "me"                                # who is asking this turn (user-level state key, SPEC §1e)
    channel: str = "dm"                               # "group": the book is scaled to $100k before anything sees it (#110)
    priced_real: pd.DataFrame | None = None           # the member's real book; `priced` is what tools and experts see
    expert_sessions: dict = field(default_factory=dict)  # (expert, member) → warm transcript + handoff (DECISIONS #108)
    log: list = field(default_factory=list)           # tool calls, for agent tests
    routing: list = field(default_factory=list)       # request → entry tool → experts run/skipped → why (SPEC §3.4h)
    turn_t0: float = 0.0                              # perf_counter at the start of the turn (latency budget, U14)
    timings: list = field(default_factory=list)       # per LLM call / tool: {kind, name, start, s}
    emit: object = None                               # UI hook: emit(kind, payload) for status/card events (SPEC §1d-0)

    def send(self, kind: str, payload) -> None:
        if kind == "card" and isinstance(payload, dict) and payload.get("type") in TEXT_ONLY_CARDS:
            from .. import config
            if config.SCENARIO_ENGINE == "planb":   # the live chat is text only for events; replays (v1) keep their cards
                return
        if self.emit is not None:
            try:
                self.emit(kind, payload)
            except Exception:  # noqa: BLE001 — a UI hiccup must never break a turn
                pass

    def require_portfolio(self):
        if self.portfolio is None or self.priced is None:
            raise ToolError("no portfolio loaded; call load_portfolio first")


class ToolError(RuntimeError):
    """A tool failure the model should see (and explain), not a crash."""
