"""In-chat train mode: `/fix <correction>` on any Worsey reply.

Hybrid:
- **Now:** the distilled rule is active for the trainer's current session, for that expert, and Worsey re-answers the
  original question with it applied.
- **Permanent only with approval:** each fix queues a proposed edit (a strategy file, the playbook, or a prompt).
  `/approve <id>` (or `worsey corrections approve <id>`) writes the trainer's words into that file under a marked
  "Corrections from the trainer" section; `/reject <id>` drops it.
- **Every fix becomes a golden case** (eval/golden/corrections.jsonl): question, bad reply, fix, rule.
Only config.TRAINERS can train.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from ..data.cache import CACHE_ROOT

ROOT = Path(__file__).resolve().parents[3]
QUEUE = CACHE_ROOT / "train" / "corrections.jsonl"
GOLDEN = ROOT / "eval" / "golden" / "corrections.jsonl"
EXPERTS = ("router", "options_desk", "event_analyst", "equity_analyst", "etf_desk", "recent_markets_analyst")
SECTION = "## Corrections from the trainer (approved in train mode)"


def _rows() -> list[dict]:
    if not QUEUE.exists():
        return []
    return [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_all(rows: list[dict]) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")


def distil(llm, model: str, question: str, reply: str, expert: str, fix: str) -> dict:
    """One call turns the trainer's fix into a rule + scope + where it belongs. Code validates the fields."""
    system = (ROOT / "prompts" / "train_fix.md").read_text(encoding="utf-8")
    resp = llm.create(model=model, max_tokens=600, system=system, output_config={"effort": "medium"},
                      messages=[{"role": "user", "content": f"Question: {question[:800]}\n\nWorsey's reply ({expert}): {reply[:1500]}"
                                                             f"\n\nthe trainer's /fix: {fix[:1000]}"}])
    raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    rule = str(d.get("rule") or fix).strip()[:300]
    exp = d.get("expert") if d.get("expert") in EXPERTS else (expert if expert in EXPERTS else "router")
    target = d.get("target") if d.get("target") in ("strategy_file", "playbook", "prompt") else "prompt"
    sid = d.get("strategy_id") if target == "strategy_file" else None
    if target == "strategy_file" and not (sid and (ROOT / "library" / "strategies" / f"{sid}.md").exists()):
        target, sid = "playbook" if exp == "options_desk" else "prompt", None
    return {"rule": rule, "expert": exp, "topic": str(d.get("topic") or "")[:80], "target": target, "strategy_id": sid,
            "restatement": str(d.get("restatement") or rule)[:300]}


def record(trainer: str, chat: str, question: str, reply: str, expert: str, fix: str, d: dict) -> dict:
    """Queue the proposed permanent edit and save the golden case."""
    c = {"id": uuid.uuid4().hex[:6], "ts": time.time(), "trainer": trainer, "chat": chat, "question": question,
         "reply": reply, "fix": fix, **d, "status": "pending"}
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(c, default=str) + "\n")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": c["id"], "question": question, "bad_reply": reply, "fix": fix, "rule": d["rule"],
                            "expert": d["expert"], "topic": d["topic"]}, default=str) + "\n")
    return c


def pending() -> list[dict]:
    return [r for r in _rows() if r["status"] == "pending"]


def target_file(c: dict) -> Path:
    if c["target"] == "strategy_file" and c.get("strategy_id"):
        return ROOT / "library" / "strategies" / f"{c['strategy_id']}.md"
    if c["target"] == "playbook":
        return ROOT / "library" / "playbook" / "corrections.md"
    return ROOT / "prompts" / ("router.md" if c["expert"] == "router" else f"{c['expert']}.md")


def decide(cid: str, approve: bool) -> dict:
    """Approve: append the trainer's fix (his words) and the distilled rule to the target file. Reject: drop it."""
    rows = _rows()
    c = next((r for r in rows if r["id"] == cid), None)
    if c is None or c["status"] != "pending":
        raise KeyError(f"no pending correction {cid}")
    if approve:
        f = target_file(c)
        f.parent.mkdir(parents=True, exist_ok=True)
        text = f.read_text(encoding="utf-8") if f.exists() else ("# The trainer's playbook corrections\n" if c["target"] == "playbook" else "")
        if SECTION not in text:
            text = text.rstrip() + f"\n\n{SECTION}\n"
        day = time.strftime("%Y-%m-%d", time.localtime(c["ts"]))
        text = text.rstrip() + f"\n- ({day}, {c['topic'] or c['expert']}) {c['rule']}  — the trainer: \"{c['fix']}\"\n"
        f.write_text(text, encoding="utf-8")
        c["applied_to"] = str(f.relative_to(ROOT))
    c["status"] = "approved" if approve else "rejected"
    c["decided"] = time.time()
    _write_all(rows)
    return c


def session_rules(s, expert: str) -> list[str]:
    """Rules active for the trainer: their pending fixes (kept until approved or rejected, across server restarts) plus
    any made this session, for this expert or the router. Approved ones live in the files themselves."""
    from .. import config
    rows = list(s.__dict__.get("session_corrections", []))
    if getattr(s, "member", None) in config.TRAINERS:
        have = {r["id"] for r in rows}
        rows += [r for r in pending() if r["id"] not in have and r.get("trainer") == s.member]
    return [r["rule"] for r in rows if r["expert"] in (expert, "all")]


def render_pending() -> str:
    rows = pending()
    if not rows:
        return "**No pending corrections.**"
    lines = [f"- `{r['id']}` → {r['target']}{' ' + r['strategy_id'] if r.get('strategy_id') else ''} ({r['expert']}): {r['rule']}"
             for r in rows[-15:]]
    return "**Pending corrections** (`/approve <id>` writes it into the file, `/reject <id>` drops it):\n" + "\n".join(lines)
