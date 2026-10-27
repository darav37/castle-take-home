"""Group-chat web UI backend (SPEC §1d, §1d-0, §1e). FastAPI, one WebSocket per chat.

  worsey serve [--portfolio balanced] [--mention-only] [--cached-data]     # live
  worsey serve --replay --script s0|iran|tariffs                           # recorded demos, zero tokens

Multi-user runtime (Option 3, DECISIONS #109, #110, #126):
- **State is split.** Each member has their own state (portfolio, scans, views, warm expert sessions, router history),
  shared by their group chat and their DM with Worsey. The chat keeps only its transcript.
- **Concurrency.** Different members' requests run in parallel (up to LIVE_CONCURRENCY live turns). A member's
  follow-up waits for their own running request, then goes to the warm expert session: no cancel-and-restart.
- **Replies post when ready.** A reply that isn't answering the latest message quotes the request and @mentions its
  sender. Every status/card/reply carries the request id. No ack messages: status text only.
- **Privacy by construction.** In a group chat, tools and experts see the member's book scaled to $100k
  (tools.apply_channel), so real dollars can't reach a group reply. The DM uses real dollars. After a group answer the
  member gets a code-written DM with their scale factor (no extra model call). "post it here" opts a member in for that chat.
- **Overheard context.** The router reads the last ~20 messages from the last ~2 hours, never to be quoted.
- **Corrections.** Worsey's own corrections post after the turn, quoting the original. In --mention-only mode, an
  unaddressed message with a material factual error about an instrument may get one short line (once per 2h).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "web"
SCRIPTS = {"s0": ("demo_main", "balanced", "darin"), "iran": ("demo_iran", None, "sam"),
           "tariffs": ("demo_tariffs", None, "jo")}
MEMBERS = {"darin": "#2f6fdb", "sam": "#c2410c", "jo": "#0f8a5f"}
MENTION = re.compile(r"@worsey\b", re.I)
POST_HERE = re.compile(r"\bpost it here\b|\bshow (my )?(real )?(numbers|dollars) here\b", re.I)
INSTRUMENT_CLAIM = re.compile(r"\b(tracks?|can only lose|can't lose|guaranteed|always|never|decay|leveraged|3x|2x|short put|"
                              r"covered call|futures|roll|expires?|assigned|margin)\b", re.I)
MAX_MESSAGE_CHARS = 1000      # live mode: longer messages are refused, not sent to the model
USER_MSGS_PER_MIN = 6         # live mode: per-person rate limit
LIVE_CONCURRENCY = 3          # live turns in parallel across all chats (each still capped per turn by budget.py)
OVERHEARD_N, OVERHEARD_S = 20, 7200
FACT_CHECK_EVERY_S = 7200


class Member:
    """One person's user-level state, shared by their group chat(s) and their DM (SPEC §1e)."""

    def __init__(self, user: str, session, llm):
        self.user, self.session, self.llm = user, session, llm
        self.lock = asyncio.Lock()           # their follow-up waits for their own running request
        self.opted_in: set[str] = set()      # chats where they said "post it here"
        self.scale_note_sent = False
        self.book_checked: str | None = None   # date of the last daily position check (DECISIONS #145)


class Runtime:
    """App-wide: members, chats, the live-turn semaphore."""

    def __init__(self, replay: str | None, portfolio: str | None, mention_only: bool):
        self.replay, self.portfolio, self.mention_only = replay, portfolio, mention_only
        self.members: dict[str, Member] = {}
        self.chats: dict[str, Chat] = {}
        self.sem = asyncio.Semaphore(1 if replay else LIVE_CONCURRENCY)
        self.replay_llm = None
        if replay:
            from . import config
            config.SCENARIO_ENGINE = "v1"   # the recorded demos were made with the v1 engine (DECISIONS #179)
            from .agent.llm import ReplayLLM
            os.environ["WORSEY_OFFLINE"] = "1"
            os.environ.setdefault("WORSEY_LIBRARY_DIR", str(ROOT / "tests" / "fixtures" / "strategies"))
            self.replay_llm = ReplayLLM(ROOT / "eval" / "recordings" / f"{SCRIPTS[replay][0]}.jsonl")

    def _data(self):
        from .data.provider import BuiltMarketData
        if not self.replay and os.environ.get("WORSEY_OFFLINE") != "1":
            return BuiltMarketData(live=True)   # live data: real date, hourly quotes, fresh closes
        tag_f = ROOT / "eval" / "recordings" / "QUOTE_TAG"
        return BuiltMarketData(quote_tag=tag_f.read_text().strip() if tag_f.exists() else None)

    def member(self, user: str) -> Member:
        if user in self.members:
            return self.members[user]
        from .agent import tools
        from .agent.session import Session
        s = Session(self._data(), member=user)
        s.__dict__["remember"] = not self.replay   # live chat writes the episodic log and book snapshots
        if self.replay:
            m = Member(user, s, self.replay_llm)
            _, pf, script_user = SCRIPTS[self.replay]
            if pf and user == script_user:   # the recordings were made with this portfolio pre-loaded (CLI default)
                tools.load_portfolio(s, name=pf)
        else:
            from .agent.llm import AnthropicLLM
            m = Member(user, s, AnthropicLLM())   # per member, so the per-turn call limit is per request
            if self.portfolio:
                s.channel = "group"
                out = tools.compact(tools.load_portfolio(s, name=self.portfolio))
                s.__dict__.setdefault("_all_results", []).append({"load_portfolio": out})
                s.history += [{"role": "user", "content": "[system: the member's demo portfolio is already loaded] "
                               + json.dumps(out, default=str)},
                              {"role": "assistant", "content": "Got it: the portfolio is loaded."}]
        self.members[user] = m
        return m

    def chat(self, chat_id: str, user: str) -> "Chat":
        cid = f"dm:{user}" if chat_id == "dm" else chat_id
        if cid not in self.chats:
            self.chats[cid] = Chat(cid, self)
        return self.chats[cid]


class Chat:
    """One chat: sockets, the event history, the transcript, and (in replay) the script cursor."""

    def __init__(self, chat_id: str, rt: Runtime):
        self.id, self.rt = chat_id, rt
        self.kind = "dm" if chat_id.startswith("dm:") else "group"
        self.sockets, self.history, self.transcript, self.recent = {}, [], [], {}
        self.seq, self.last_fact_check = 0, 0.0
        self.script, self.cursor, self.script_user = [], 0, None
        if rt.replay:
            name, _, user = SCRIPTS[rt.replay]
            self.script = [l.strip() for l in (ROOT / "eval" / "scripts" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
                           if l.strip()]
            self.script_user = user

    async def broadcast(self, event: dict):
        self.history.append(event)
        dead = []
        for ws in list(self.sockets):
            try:
                await ws.send_text(json.dumps(event, default=str))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.sockets.pop(ws, None)

    # ------------------------------------------------------------------ messages in

    async def user_message(self, user: str, text: str, quote: str | None = None):
        if text.startswith("/") and not self.rt.replay:
            if await self.command(user, text, quote):
                return
        self.seq += 1
        seq = self.seq
        self.transcript.append({"seq": seq, "user": user, "text": text, "ts": time.time()})
        await self.broadcast({"type": "message", "user": user, "text": text, "quote": quote, "ts": time.time(), "seq": seq})
        addressed = bool(MENTION.search(text) or (quote and quote.startswith("worsey:")) or self.kind == "dm")
        if self.rt.mention_only and not addressed:
            if not self.rt.replay:
                asyncio.create_task(self.maybe_fact_check(user, text, seq))
            return
        if not self.rt.replay:
            refusal = self.live_refusal(user, text)
            if refusal:
                await self.broadcast({"type": "reply", "rid": f"{self.id}-{seq}", "text": refusal})
                await self.broadcast({"type": "done", "rid": f"{self.id}-{seq}", "timings": {}, "route": [], "sources": {}})
                return
        if POST_HERE.search(text):
            self.rt.member(user).opted_in.add(self.id)
        job = self.handle(user, MENTION.sub("", text).strip(), seq)
        if self.rt.replay:
            await job                         # demos step one scripted line at a time
        else:
            asyncio.create_task(job)          # live: other messages keep flowing while this one runs

    # ------------------------------------------------------------------ in-chat train mode (/fix)

    async def command(self, user: str, text: str, quote: str | None) -> bool:
        """/fix <correction>, /review, /approve <id>, /reject <id>: trainers only (config.TRAINERS). True if handled."""
        from . import config
        from .agent import train_fix
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd not in ("/fix", "/review", "/approve", "/reject"):
            return False
        self.seq += 1
        seq = self.seq
        await self.broadcast({"type": "message", "user": user, "text": text, "quote": quote, "ts": time.time(), "seq": seq})
        rid = f"{self.id}-{seq}"
        if user not in config.TRAINERS:
            await self.say(rid, "Only Worsey's trainer can use that.")
            return True
        if cmd == "/review":
            await self.rt.chat("dm", user).broadcast({"type": "reply", "rid": f"{rid}-dm", "text": train_fix.render_pending()})
            await self.say(rid, "Pending corrections are in your DM." if self.kind == "group" else train_fix.render_pending())
            return True
        if cmd in ("/approve", "/reject"):
            try:
                c = train_fix.decide(arg.strip(), approve=cmd == "/approve")
                msg = (f"**Approved `{c['id']}`:** written into `{c['applied_to']}`." if c["status"] == "approved"
                       else f"**Rejected `{c['id']}`.**")
            except KeyError as e:
                msg = str(e).strip("'")
            await self.say(rid, msg)
            return True
        await self.fix(user, arg.strip(), quote, seq, rid)
        return True

    async def say(self, rid: str, text: str):
        await self.broadcast({"type": "reply", "rid": rid, "text": text})
        await self.broadcast({"type": "done", "rid": rid, "timings": {}, "route": [], "sources": {}})

    def _target(self, quote: str | None):
        """The Worsey reply being corrected: the quoted one, else the latest; plus its question and route."""
        replies = [e for e in self.history if e["type"] == "reply" and e.get("rid") and not str(e["rid"]).endswith(("-fix", "-dm"))
                   and not e["text"].startswith(("**Got it:**", "**Correction:**"))]
        target = None
        if quote and quote.startswith("worsey:"):
            q = re.sub(r"\W+", " ", quote[7:]).strip().lower()[:50]
            target = next((e for e in reversed(replies) if re.sub(r"\W+", " ", e["text"]).strip().lower().startswith(q[:30])), None)
        target = target or (replies[-1] if replies else None)
        if target is None:
            return None, None, None
        seq = int(str(target["rid"]).rsplit("-", 1)[-1]) if str(target["rid"]).rsplit("-", 1)[-1].isdigit() else None
        question = next((m["text"] for m in self.transcript if m["seq"] == seq), "")
        done = next((e for e in self.history if e["type"] == "done" and e.get("rid") == target["rid"]), {})
        experts = [p["expert"] for r in done.get("route", []) for p in r.get("plan", []) if p.get("run")] + \
                  [r["entry"] for r in done.get("route", []) if r.get("entry") in ("options_desk", "event_analyst", "equity_analyst",
                                                                                   "etf_desk", "recent_markets_analyst")]
        return target, question, (experts[0] if experts else "router")

    async def fix(self, user: str, fix: str, quote: str | None, seq: int, rid: str):
        from .agent import train_fix
        from .agent.loop import ROLES
        if not fix:
            await self.say(rid, "Add what's wrong after /fix, e.g. `/fix for earnings views compare to past earnings moves`.")
            return
        target, question, expert = self._target(quote)
        if target is None:
            await self.say(rid, "I couldn't find a reply of mine to correct here.")
            return
        m = self.rt.member(user)
        loop = asyncio.get_running_loop()
        await self.broadcast({"type": "status", "rid": rid, "text": "learning from your fix…"})
        d = await loop.run_in_executor(None, train_fix.distil, m.llm, ROLES["router"]["model"], question, target["text"], expert, fix)
        c = train_fix.record(user, self.id, question, target["text"], expert, fix, d)
        m.session.__dict__.setdefault("session_corrections", []).append(c)
        await self.broadcast({"type": "reply", "rid": rid, "reply_to": {"user": "worsey", "text": target["text"][:160]},
                              "text": f"**Got it:** {d['restatement']}\nActive now for {d['expert']}; `/approve {c['id']}` makes it "
                                      f"permanent ({d['target']}{' ' + d['strategy_id'] if d.get('strategy_id') else ''})."})
        await self.broadcast({"type": "done", "rid": rid, "timings": {}, "route": [], "sources": {}})
        if question:   # re-answer the original question with the rule applied
            self.seq += 1
            self.transcript.append({"seq": self.seq, "user": user, "text": question, "ts": time.time()})
            await self.handle(user, f"{question}\n[the trainer's correction, apply it: {d['rule']}]", self.seq)

    def live_refusal(self, user: str, text: str) -> str | None:
        """Cheap checks before any model call."""
        if len(text) > MAX_MESSAGE_CHARS:
            return f"That message is too long for me (max {MAX_MESSAGE_CHARS} characters)."
        now = time.time()
        recent = [t for t in self.recent.get(user, []) if t > now - 60]
        if len(recent) >= USER_MSGS_PER_MIN:
            return "Slow down a little: I take at most 6 questions a minute from each person."
        self.recent[user] = recent + [now]
        return None

    def overheard(self, seq: int) -> str:
        """The last ~20 messages from the last ~2h before this one, as context the router must never quote (#126)."""
        now = time.time()
        rows = [m for m in self.transcript if m["seq"] < seq and now - m["ts"] <= OVERHEARD_S][-OVERHEARD_N:]
        if not rows:
            return ""
        lines = "\n".join(f"{m['user']}: {m['text'][:300]}" for m in rows)
        return ("[recent group chat, for context only: never quote it, never attribute it, never use what one member "
                f"says about another]\n{lines}\n\n")

    # ------------------------------------------------------------------ one request

    async def handle(self, user: str, text: str, seq: int):
        from .agent import tools
        from .agent.loop import run_turn
        m = self.rt.member(user)
        rid = f"{self.id}-{seq}"
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def emit(kind, payload):
            ev = ({"type": "card", "card": payload["type"], "payload": payload["payload"]} if kind == "card"
                  else {"type": kind, **payload})
            ev["rid"] = rid
            loop.call_soon_threadsafe(q.put_nowait, ev)

        async with m.lock:                    # a follow-up mid-request waits, then goes to the warm session
            async with self.rt.sem:
                s = m.session
                s.member, s.emit = user, emit
                s.__dict__["chat_id"] = self.id
                s.channel = "dm" if (self.kind == "dm" or self.id in m.opted_in or self.rt.replay) else "group"
                tools.apply_channel(s)
                tone = ""
                if not self.rt.replay:   # member profile (#120): tone in the group, depth in DMs; gates from behavior
                    from .agent import profiles
                    prof = profiles.load(user)
                    s.__dict__["profile_view"] = profiles.for_model(prof, s.channel)
                    s.__dict__["unlocked_gates"] = set(profiles.sophistication(prof)["unlocked_gates"])
                    from . import config as _cfg
                    level = profiles.sophistication(prof)["level"]
                    s.__dict__["lingo"] = ("trader" if user in _cfg.TRAINERS or level == "sophisticated"
                                           else "mixed" if level == "intermediate" else "plain")   # voice.md
                    lines = [x for k in profiles.TONE_SECTIONS for x in s.__dict__["profile_view"].get(k, [])]
                    tone = f"[how {user} likes to be talked to: {'; '.join(lines)}]\n" if lines else ""
                    tone += f"[lingo for {user}: {s.__dict__['lingo']}]\n"
                from .agent.train_fix import session_rules
                fixes = session_rules(s, "router")
                tone += f"[trainer corrections active: {'; '.join(fixes)}]\n" if fixes and not self.rt.replay else ""
                prompt = text if self.rt.replay else (self.overheard(seq) + tone + (
                    "[group chat: amounts are per $100k of your book; real dollars go to DM]\n" if s.channel == "group" else "")
                    + f"[{user} to Worsey]: {text}")
                start = len(s.routing)
                fut = loop.run_in_executor(None, run_turn, m.llm, s, prompt)
                while not fut.done() or not q.empty():
                    try:
                        await self.broadcast(await asyncio.wait_for(q.get(), timeout=0.05))
                    except asyncio.TimeoutError:
                        continue
                try:
                    out = fut.result()
                except Exception as e:  # noqa: BLE001 — a turn failure becomes a message, never a dead chat
                    from .agent.budget import BudgetExceeded
                    msg = str(e) if isinstance(e, BudgetExceeded) else f"Sorry, that one failed on my side ({type(e).__name__})."
                    await self.broadcast({"type": "reply", "rid": rid, "text": msg})
                    await self.broadcast({"type": "done", "rid": rid, "timings": {}, "route": [], "sources": {}})
                    return
        # replies post when ready; one that isn't answering the latest message quotes the request and @mentions its sender
        reply_to = {"user": user, "text": text[:160]} if self.seq > seq and self.kind == "group" else None
        await self.broadcast({"type": "reply", "rid": rid, "text": out["reply"], "grounding": out["grounding"],
                              "reply_to": reply_to})
        await self.broadcast({"type": "done", "rid": rid, "timings": out["latency"], "tools": out["tools"],
                              "route": s.routing[start:], "sources": self.sources(s), "usage": self.usage()})
        for c in s.__dict__.pop("corrections", []) or []:   # Worsey's own corrections, quoting the original (#126)
            await self.broadcast({"type": "reply", "rid": f"{rid}-fix", "text": f"**Correction:** {c['text']}",
                                  "reply_to": {"user": "worsey", "text": c["quote"]}})
            await self.broadcast({"type": "done", "rid": f"{rid}-fix", "timings": {}, "route": [], "sources": {}})
        await self.scale_note(m, s)
        await self.position_check(m, s)
        for msg in s.__dict__.pop("dm_out", []) or []:   # e.g. "what do you know about me" asked in the group
            await self.rt.chat("dm", user).broadcast({"type": "reply", "rid": f"{rid}-dm", "text": msg})
        if not self.rt.replay:
            asyncio.get_running_loop().run_in_executor(None, self.profile_pass, m, text, out["reply"], s, start)

    def profile_pass(self, m: Member, said: str, reply: str, s, start: int):
        """Off the hot path (#120): structures this turn touched, then one cheap call proposes profile lines."""
        from .agent import profiles
        from .agent.loop import ROLES
        structures = []
        for x in s.log[-12:]:
            inp = x.get("input") or {}
            structures += list(inp.get("include") or [])
        if s.last_strategies:
            structures += [c.id for c in s.last_strategies.get("candidates", [])]
        others = [u for u in set(MEMBERS) | set(self.rt.members) if u != m.user]
        profiles.update_after_turn(m.llm, m.user, said, reply, structures, others, ROLES["router"]["model"])

    async def position_check(self, m: Member, s):
        """Once a day, after the member's first message: drift and staleness on their paper positions, as a code-written
        DM (DECISIONS #145). Positions are private, so it never posts in a group."""
        from .agent import tools
        from .options import book
        today = s.data.today().isoformat()
        if self.rt.replay or m.book_checked == today or not book.open_positions(m.user):
            return
        m.book_checked = today
        try:
            status = await asyncio.get_running_loop().run_in_executor(None, tools.book_status, s, m.user)
        except Exception:  # noqa: BLE001  (a failed check never breaks the chat)
            return
        note = book.daily_note(status)
        if note:
            await self.rt.chat("dm", m.user).broadcast({"type": "reply", "rid": f"dm-book-{m.user}-{today}", "text": note})

    async def scale_note(self, m: Member, s):
        """After a group answer on a scaled book: a code-written DM with the member's scale factor (no model call)."""
        if s.channel != "group" or m.scale_note_sent or s.priced_real is None or self.rt.replay:
            return
        real = float(s.priced_real["value"].sum())
        if abs(real - 100_000) < 1:
            return
        m.scale_note_sent = True
        dm = self.rt.chat("dm", m.user)
        f = real / 100_000
        await dm.broadcast({"type": "reply", "rid": f"dm-scale-{m.user}", "text":
                            f"**Your real numbers, privately.** In the group I show figures per $100k so your book stays "
                            f"yours. Your book is ${real:,.0f}, so multiply group dollar figures by {f:.2f}. Ask me here for "
                            f"anything in real dollars, or say \"post it here\" in the group to share them there."})

    async def maybe_fact_check(self, user: str, text: str, seq: int):
        """--mention-only: one short correction of a material factual error about an instrument, at most once per 2h."""
        if not INSTRUMENT_CLAIM.search(text) or time.time() - self.last_fact_check < FACT_CHECK_EVERY_S:
            return
        from .agent.loop import ROLES
        m = self.rt.member(user)
        system = (ROOT / "prompts" / "fact_check.md").read_text(encoding="utf-8")
        try:
            resp = await asyncio.get_running_loop().run_in_executor(None, lambda: m.llm.create(
                model=ROLES["router"]["model"], max_tokens=300, system=system,
                messages=[{"role": "user", "content": text[:600]}], output_config={"effort": "low"}))
            raw = "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
            verdict = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception:  # noqa: BLE001
            return
        if verdict.get("correct") and verdict.get("line") and not re.search(r"\d", verdict["line"]):
            self.last_fact_check = time.time()
            await self.broadcast({"type": "reply", "rid": f"{self.id}-{seq}-fc", "text": verdict["line"],
                                  "reply_to": {"user": user, "text": text[:160]}})
            await self.broadcast({"type": "done", "rid": f"{self.id}-{seq}-fc", "timings": {}, "route": [], "sources": {}})

    # ------------------------------------------------------------------ misc

    def usage(self) -> dict | None:
        if self.rt.replay:
            return None
        from .agent import budget
        u = budget.usage()
        return {"calls_day": u["calls_day"], "max_day": u["limits"]["per_day"], "spend": u["spend_usd"]}

    def sources(self, s) -> dict:
        d = s.data
        b = getattr(d, "build", None) or {}
        mf = b.get("manifest", {})
        live = getattr(d, "live", False)
        return {"build": mf.get("built_at_utc"), "as_of": str(d.today()) if live else mf.get("as_of"),
                "quotes": "live, refreshed hourly" if live else getattr(d, "quote_tag", None),
                "markets": "Kalshi (public API)", "prices_options_holdings": "Massive",
                "mode": "replay" if self.rt.replay else "live", "channel": s.channel}

    async def next_scripted(self):
        if self.cursor >= len(self.script):
            await self.broadcast({"type": "script_end"})
            return
        line = self.script[self.cursor]
        self.cursor += 1
        await self.user_message(self.script_user, f"@worsey {line}")


def access_key() -> str:
    """The access code for --lan: WORSEY_ACCESS_KEY if set, else one generated once and kept in cache/access_key."""
    import secrets
    from .data.cache import CACHE_ROOT
    if os.environ.get("WORSEY_ACCESS_KEY"):
        return os.environ["WORSEY_ACCESS_KEY"]
    f = CACHE_ROOT / "access_key"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_urlsafe(9), encoding="utf-8")
    return f.read_text(encoding="utf-8").strip()


def create_app(replay: str | None = None, portfolio: str | None = None, mention_only: bool = False,
               key: str | None = None) -> FastAPI:
    app = FastAPI(title="Worsey")
    rt = Runtime(replay, portfolio, mention_only)
    app.state.rt = rt

    if key:   # --lan: every request needs the access code (query ?key= once, then a cookie)
        from fastapi import Request
        from fastapi.responses import PlainTextResponse
        import hmac

        @app.middleware("http")
        async def require_key(request: Request, call_next):
            given = request.query_params.get("key") or request.cookies.get("worsey_key") or ""
            if not hmac.compare_digest(given, key):
                return PlainTextResponse("Worsey: add ?key=<access code> to the link.", status_code=401)
            resp = await call_next(request)
            if request.query_params.get("key"):
                resp.set_cookie("worsey_key", key, httponly=True, samesite="strict", max_age=30 * 86400)
            return resp

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/config")
    async def config(chat_id: str = "demo", user: str = "darin"):
        c = rt.chat(chat_id, user)
        live = None
        if not replay:
            from .agent import budget
            u = budget.usage()
            live = {"calls_day": u["calls_day"], "max_day": u["limits"]["per_day"], "spend": u["spend_usd"],
                    "data": "live" if os.environ.get("WORSEY_OFFLINE") != "1" else "cached build"}
        return JSONResponse({"members": MEMBERS, "replay": replay, "live": live, "mention_only": mention_only,
                             "kind": c.kind, "script": c.script, "script_user": c.script_user, "cursor": c.cursor})

    @app.get("/charts/{name}")
    async def charts(name: str):
        p = ROOT / "out" / "charts" / Path(name).name
        return FileResponse(p) if p.exists() else JSONResponse({"error": "no chart"}, 404)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, chat_id: str = "demo", user: str = "darin"):
        if key:
            import hmac
            given = websocket.query_params.get("key") or websocket.cookies.get("worsey_key") or ""
            if not hmac.compare_digest(given, key):
                await websocket.close(code=4401)
                return
        await websocket.accept()
        c = rt.chat(chat_id, user)
        if c.kind == "dm" and c.id != f"dm:{user}":
            await websocket.close()
            return
        c.sockets[websocket] = user
        for ev in c.history:  # late joiners see the chat so far
            await websocket.send_text(json.dumps(ev, default=str))
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                if msg.get("type") == "next" and rt.replay:
                    await c.next_scripted()
                elif msg.get("type") == "message" and msg.get("text", "").strip():
                    await c.user_message(user, msg["text"].strip(), msg.get("quote"))
        except WebSocketDisconnect:
            c.sockets.pop(websocket, None)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; this just picks the outward interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def serve(replay: str | None, port: int = 8000, portfolio: str | None = None, cached_data: bool = False,
          mention_only: bool = False, lan: bool = False):
    import uvicorn
    if not replay and cached_data:
        # --cached-data: market data only from the cached build (no Massive/Kalshi calls); Claude is still live
        os.environ["WORSEY_OFFLINE"] = "1"
    if not replay:
        os.environ["WORSEY_LIVE_QUOTES"] = "1"   # live chat: the live_quote tool (always fetches, even on cached data)
    key = access_key() if lan else None
    host = "0.0.0.0" if lan else "127.0.0.1"
    if lan:
        print(f"[worsey] on your network: http://{lan_ip()}:{port}/?user=darin&key={key}", flush=True)
        print("[worsey] (Tailscale: same link with your PC's Tailscale IP or name)", flush=True)
    uvicorn.run(create_app(replay, portfolio, mention_only, key=key), host=host, port=port, log_level="warning")
