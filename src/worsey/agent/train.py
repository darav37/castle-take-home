"""Train mode (SPEC §6a options track, §3.1c events track). trainer-only, local only.

Nothing is weight-trained: Worsey interviews the trainer, runs live drills, and proposes edits to
its own knowledge files. Proposals queue in library/_proposals/; `worsey review` shows each
diff and commits only what the trainer accepts. Code enforces that edits land only under
library/ or examples/events/ (train mode can't touch code or prompts).
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import time
from datetime import date
from pathlib import Path

import pandas as pd

from ..scenario import impact
from . import loop, tools
from .session import Session, ToolError

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = [ROOT / "library", ROOT / "examples" / "events"]
QUEUE = ROOT / "library" / "_proposals"


def _allowed(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if not any(p.is_relative_to(a.resolve()) for a in ALLOWED) or p.is_relative_to(QUEUE.resolve()):
        raise ToolError(f"train mode may only edit library/ or examples/events/, not {rel}")
    return p


def propose_edit(s: Session, path: str, content: str, reason: str) -> dict:
    """Queue a full-file proposal (new or replacement) for the trainer's review."""
    _allowed(path)
    QUEUE.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")[-40:]
    f = QUEUE / f"{int(time.time())}_{slug}.json"
    f.write_text(json.dumps({"path": path, "content": content, "reason": reason, "date": date.today().isoformat()}, indent=1),
                 encoding="utf-8")
    return {"queued": f.name, "path": path}


FACTOR_PROXIES = ["SPY", "TLT", "UUP", "USO", "GLD", "HYG", "IEF", "VIXY"]


def factor_moves(s: Session, date_: str, before: int = 1, after: int = 1, extra: list | None = None) -> dict:
    """Realized factor moves around a date: the code side of every worked example (SPEC §3.1c)."""
    closes = s.data.closes(FACTOR_PROXIES + list(extra or []), days=6000)
    mv = impact.window_moves(closes, pd.Timestamp(date_), before, after)
    return {"date": date_, "window": f"close {before}d before → close {after}d after", "moves": mv.to_dict()}


tools.TOOLS["propose_edit"] = (propose_edit, "Queue a proposed new/replacement file under library/ or examples/events/ for the trainer to approve. Full file content, plus the reason in one line.",
                               tools._obj({"path": {"type": "string", "minLength": 1}, "content": {"type": "string", "minLength": 1},
                                           "reason": {"type": "string", "minLength": 1}}, ["path", "content", "reason"]))
tools.TOOLS["factor_moves"] = (lambda s, date, before=1, after=1, extra=None: factor_moves(s, date, before, after, extra),
                               "Realized factor-proxy moves (SPY, TLT, UUP, USO, GLD, HYG, IEF, VIXY, + extra tickers) around a date.",
                               tools._obj({"date": {"type": "string", "minLength": 10}, "before": {"type": "integer", "minimum": 1, "maximum": 5},
                                           "after": {"type": "integer", "minimum": 1, "maximum": 5},
                                           "extra": {"type": "array", "items": {"type": "string"}}}, ["date"]))
tools.GROUPS["train_options"] = ["set_view", "market_state", "propose_strategies", "propose_edit"]
tools.GROUPS["train_events"] = ["factor_moves", "propose_edit"]
loop.ROLES["train_options"] = {"model": loop.EXPERT_MODEL, "prompt": "train_options.md", "max_tokens": 16000, "effort": "high"}
loop.ROLES["train_events"] = {"model": loop.EXPERT_MODEL, "prompt": "train_events.md", "max_tokens": 16000, "effort": "high"}


def run_train(llm, track: str, data=None, inputs=None, out=print) -> Session:
    """Interactive session. `inputs` (a list) replaces the keyboard for tests/replays."""
    from ..data.provider import LiveMarketData
    s = Session(data or LiveMarketData())
    role = f"train_{track}"
    out(f"[train mode: {track}. Proposals queue for `worsey review`. Type 'done' to finish.]")
    msg = "Start the session: pick the topic from the plan and the gaps, then ask your first question."
    while True:
        s.turn_results = []
        s.history.append({"role": "user", "content": msg})
        out(loop._step_loop(llm, s, role, s.history))
        if inputs is not None:
            if not inputs:
                break
            msg = inputs.pop(0)
        else:
            try:
                msg = input("darin> ").strip()
            except EOFError:
                break
        if msg in ("done", "exit", "quit"):
            break
    return s


def pending() -> list[Path]:
    return sorted(QUEUE.glob("*.json")) if QUEUE.exists() else []


def review(ask=input, out=print, commit: bool = True) -> list[str]:
    """Show each proposal's diff; accept (y), reject (n) or skip (s). Accepted files are committed."""
    accepted = []
    for f in pending():
        prop = json.loads(f.read_text(encoding="utf-8"))
        target = _allowed(prop["path"])
        old = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        out(f"\n=== {prop['path']}  ({prop['reason']})")
        out("\n".join(difflib.unified_diff(old, prop["content"].splitlines(), "current", "proposed", lineterm="")))
        a = ask("accept? [y/n/s] ").strip().lower()
        if a == "y":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(prop["content"].rstrip("\n") + "\n", encoding="utf-8")
            accepted.append(prop["path"])
            f.unlink()
        elif a == "n":
            f.unlink()
    if commit and accepted:
        subprocess.run(["git", "add", *accepted], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", "library: train-mode edits approved by the trainer\n\n" + "\n".join(accepted)], cwd=ROOT, check=True)
    return accepted
