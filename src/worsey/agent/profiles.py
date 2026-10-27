"""Member profiles (SPEC §1e-1; DECISIONS #111, #113, #120): one context file per member, so Worsey speaks to each
person the way they've asked to be spoken to, and unlocks advanced structures from their actual behavior.

Guardrails, enforced in code:
1. Only what the member said: every line proposed by the background pass must quote the member's own message (checked
   as a substring); lines naming another member are dropped; no inferred-psychology words.
2. The book and trades come from code (portfolio, book history); the profile never holds a P&L or a position.
3. The member can see it ("what do you know about me", answered in DM) and remove any line.
4. Written off the hot path: one cheap call after each live turn.
5. Sections: preferences, voice, opt_ins, knowledge; plus `behavior` (code-tracked structures and concepts explored).
6. Use: in the group only `preferences` and `voice` (tone) reach the model; experts see the rest in DMs.
7. Sophistication is computed by code rules on behavior and replaces the "asked by name" gate (#104) for gated structures.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from ..data.cache import CACHE_ROOT

DIR = CACHE_ROOT / "profiles"
SECTIONS = ("preferences", "voice", "opt_ins", "knowledge")
TONE_SECTIONS = ("preferences", "voice")
PSYCH = re.compile(r"\b(anxious|nervous|insecure|greedy|fearful|emotional|impulsive|reckless|depressed|stressed|scared|"
                   r"overconfident|seems|probably feels|personality)\b", re.I)
MARGIN = re.compile(r"\b(i have|i've got|got|with) (a )?margin( account)?\b|\bmargin account\b", re.I)
CONCEPTS = {
    "implied vs realized vol": r"\b(implied vol|realized vol|iv\b|rv\b|vol (is )?(rich|cheap|expensive))",
    "skew": r"\bskew\b|risk reversal|\brisky\b",
    "term structure": r"term structure|calendar|front month|back month",
    "theta": r"\btheta\b|time decay",
    "delta hedging": r"delta[- ]hedg|hedge (the|my) delta|\bdelta\b",
    "gamma": r"\bgamma\b",
    "vega": r"\bvega\b",
    "event vol crush": r"vol crush|iv crush|earnings vol",
    "spreads": r"\bspread\b",
    "straddles": r"straddle|strangle",
}
STRUCTURE_GROUPS = {"calls": {"long_call", "call_debit_spread"}, "puts": {"long_put", "put_debit_spread"},
                    "straddles": {"long_straddle", "long_strangle"}}


def _path(member: str):
    return DIR / f"{re.sub(r'[^a-z0-9_-]', '_', member.lower())}.json"


def load(member: str) -> dict:
    p = _path(member)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"member": member, "lines": [], "behavior": {"structures": [], "concepts": []}, "updated": None}


def save(prof: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    prof["updated"] = time.time()
    _path(prof["member"]).write_text(json.dumps(prof, indent=1), encoding="utf-8")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def accept_lines(prof: dict, proposed: list, said: str, other_members: list[str]) -> list[dict]:
    """Code-side guardrails on the background pass's proposals. Returns the lines added."""
    added, have = [], {_norm(l["text"]) for l in prof["lines"]}
    for p in (proposed or [])[:3]:
        sec, text, quote = p.get("section"), str(p.get("text", "")).strip(), str(p.get("quote", "")).strip()
        if sec not in SECTIONS or not text or len(text) > 160 or not quote:
            continue
        if _norm(quote) not in _norm(said):                       # only what the member actually said
            continue
        if PSYCH.search(text):                                    # no inferred psychology
            continue
        if any(re.search(rf"\b{re.escape(o)}\b", text + " " + quote, re.I) for o in other_members):   # nothing about others
            continue
        if _norm(text) in have:
            continue
        line = {"id": uuid.uuid4().hex[:6], "section": sec, "text": text, "quote": quote[:160], "ts": time.time()}
        prof["lines"].append(line)
        have.add(_norm(text))
        added.append(line)
    return added


def track_behavior(prof: dict, said: str, structures: list[str]) -> None:
    """Code-tracked: structures the member asked about or was shown, concepts their own words touched, margin stated."""
    b = prof.setdefault("behavior", {"structures": [], "concepts": []})
    b["structures"] = sorted(set(b["structures"]) | {s for s in structures if s})
    b["concepts"] = sorted(set(b["concepts"]) | {c for c, rx in CONCEPTS.items() if re.search(rx, said, re.I)})
    if MARGIN.search(said):
        b["margin_stated"] = True


def sophistication(prof: dict) -> dict:
    """Code rules on behavior (#120). sophisticated = has worked with calls, puts and straddles
    and touched ≥ 4 concepts; intermediate = ≥ 3 structures or ≥ 3 concepts."""
    b = prof.get("behavior", {})
    st, cc = set(b.get("structures", [])), set(b.get("concepts", []))
    groups = {g for g, ids in STRUCTURE_GROUPS.items() if st & ids}
    if groups == set(STRUCTURE_GROUPS) and len(cc) >= 4:
        level = "sophisticated"
    elif len(st) >= 3 or len(cc) >= 3:
        level = "intermediate"
    else:
        level = "novice"
    unlocked = set()
    if level == "sophisticated":
        unlocked.add("sophisticated")
        if b.get("margin_stated"):
            unlocked.add("sophisticated_with_margin")
            if "delta hedging" in cc:
                unlocked.add("delta_hedged_with_education_and_margin")
    return {"level": level, "structure_groups": sorted(groups), "concepts": sorted(cc), "margin_stated": bool(b.get("margin_stated")),
            "unlocked_gates": sorted(unlocked)}


def for_model(prof: dict, channel: str) -> dict:
    """What the model may see: tone lines in the group; everything (plus sophistication) in a DM."""
    secs = TONE_SECTIONS if channel == "group" else SECTIONS
    out = {s: [l["text"] for l in prof["lines"] if l["section"] == s] for s in secs}
    out = {k: v for k, v in out.items() if v}
    if channel != "group":
        out["sophistication"] = sophistication(prof)["level"]
    return out


def render(prof: dict) -> str:
    """The member's own view of their profile, with line ids so they can remove any."""
    soph = sophistication(prof)
    rows = [f"- [{l['id']}] ({l['section']}) {l['text']}  (you said: \"{l['quote']}\")" for l in prof["lines"]]
    return ("**What I know about you** (only what you've told me; say \"forget <id>\" to remove a line):\n"
            + ("\n".join(rows) if rows else "- nothing yet")
            + f"\n\nLevel, from what you've done here: {soph['level']}. Structures explored: "
            + (", ".join(soph["structure_groups"]) or "none yet") + ". Positions and trades come from your book, not from this file.")


UPDATE_PROMPT = None


def update_after_turn(llm, member: str, said: str, reply: str, structures: list[str], other_members: list[str],
                      model: str) -> list[dict]:
    """The background pass: one cheap call proposes lines; code accepts only the ones that pass the guardrails."""
    from pathlib import Path
    global UPDATE_PROMPT
    UPDATE_PROMPT = UPDATE_PROMPT or (Path(__file__).resolve().parents[3] / "prompts" / "profile_update.md").read_text(encoding="utf-8")
    prof = load(member)
    track_behavior(prof, said, structures)
    added = []
    try:
        resp = llm.create(model=model, max_tokens=400, system=UPDATE_PROMPT, output_config={"effort": "low"},
                          messages=[{"role": "user", "content": f"Member: {member}\nTheir message: {said[:800]}\n"
                                                                 f"Worsey's reply: {reply[:600]}"}])
        raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
        m = re.search(r"\{.*\}", raw, re.S)
        added = accept_lines(prof, (json.loads(m.group(0)) if m else {}).get("lines", []), said, other_members)
    except Exception:  # noqa: BLE001 — the background pass must never break anything
        pass
    save(prof)
    return added
