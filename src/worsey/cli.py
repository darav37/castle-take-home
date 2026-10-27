"""Terminal entry points.

  worsey chat [--portfolio balanced] [--synthetic] [--record F | --replay F]
  worsey scan [--portfolio balanced] [--horizon 90] [--synthetic]     # no LLM: the ranked table
  worsey train options|events [--record F | --replay F]               # trainer-only (SPEC §6a, §3.1c)
  worsey review                                                       # approve pending train-mode edits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _data(args):
    """Synthetic if asked; else the latest `worsey build` (events/β cached, quotes refreshed hourly); else live."""
    from .build import latest
    from .data.provider import BuiltMarketData, LiveMarketData, SyntheticMarketData
    if args.synthetic:
        return SyntheticMarketData()
    return BuiltMarketData(quote_tag=getattr(args, "quote_tag", None)) if latest() else LiveMarketData()


def _llm(args):
    from .agent.llm import AnthropicLLM, RecordingLLM, ReplayLLM
    if getattr(args, "replay", None):
        from . import config
        config.SCENARIO_ENGINE = "v1"   # recordings were made with the v1 engine (DECISIONS #179)
        return ReplayLLM(Path(args.replay))
    live = AnthropicLLM()
    return RecordingLLM(live, Path(args.record)) if getattr(args, "record", None) else live


def cmd_scan(args):
    from .agent import tools
    from .agent.session import Session
    s = Session(_data(args))
    tools.load_portfolio(s, name=args.portfolio)
    out = tools.compact(tools.scan_events(s, horizon_days=args.horizon, n=args.n))
    if out.get("synthetic_data"):
        print("(synthetic data — plumbing check only)")
    print(f"{'event':<34}{'p':>5}{'if YES':>9}{'if NO':>9}{'σ':>7}  flags")
    shown_divider = False
    for e in out["events"]:
        if not e.get("above_divider", True) and not shown_divider:
            print("----- weak evidence below this line -----")
            shown_divider = True
        print(f"{e['event_id'][:33]:<34}{e['p']:>5.0%}{e['pnl_yes']:>+9,.0f}{e['pnl_no']:>+9,.0f}{e['sigma']:>7,.0f}  "
              f"{e.get('beta_tier')}/{e.get('prior_tier')} ev {e.get('evidence', 0):.1f}  {','.join(e['flags'])}")
    for e in out.get("drift", []):
        print(f"{e['event_id'][:33]:<34}{e['p']:>5.0%}  resolves after horizon: drift σ {e['sigma_drift']:,.0f}  {','.join(e['flags'])}")
    if out.get("joint"):
        j = out["joint"]
        print(f"joint 5% VaR: {j['var']:+,.0f}   ES: {j['es']:+,.0f}   worst combo: {j['worst_combo']}")


def cmd_chat(args):
    from .agent import tools
    from .agent.loop import run_turn
    from .agent.session import Session
    s = Session(_data(args))
    llm = _llm(args)
    if args.portfolio:
        tools.load_portfolio(s, name=args.portfolio)
    lines = Path(args.script).read_text(encoding="utf-8").splitlines() if args.script else None
    transcript = []
    while True:
        if lines is not None:
            if not lines:
                break
            msg = lines.pop(0).strip()
            if not msg:
                continue
            print(f"> {msg}")
        else:
            try:
                msg = input("> ").strip()
            except EOFError:
                break
            if msg in ("exit", "quit"):
                break
        out = run_turn(llm, s, msg)
        print(out["reply"])
        lat = out["latency"]
        print(f"  [first card {lat['first_card_s']}s · total {lat['total_s']}s · router {lat['router_llm_s']}s · experts {lat['expert_llm_s']}s · tools {out['tools']}]")
        if not out["grounding"]["ok"]:
            print(f"  [grounding: unverified numbers {out['grounding']['ungrounded']}]")
        if out["grounding"].get("stripped"):
            print(f"  [grounding: stripped unsupported claims {out['grounding']['stripped']}]")
        transcript.append({"user": msg, **out, "routing": s.routing[-1:] if s.routing else []})
    if args.transcript:
        Path(args.transcript).write_text(json.dumps(transcript, indent=1), encoding="utf-8")


def cmd_build(args):
    from .agent.llm import AnthropicLLM
    from .build import add_names, run
    if args.names_only:
        add_names(AnthropicLLM())
        return
    from .data.provider import LiveMarketData
    run(LiveMarketData(), max_markets=args.max_markets, llm=AnthropicLLM())


def cmd_vol_history(args):
    from datetime import date
    from . import config
    from .data.provider import LiveMarketData
    from .options import vol_history
    tickers = [t.upper() for t in args.tickers] or config.VOL_HISTORY_TICKERS
    print(json.dumps(vol_history.update(tickers, LiveMarketData(), date.today(), days=args.days), indent=1))


def cmd_etf_panel(args):
    from .etfmatch import load_panel
    p = load_panel()
    print(f"[etfmatch] panel: {p['closes'].shape[0]} days × {p['closes'].shape[1]} tickers")


def cmd_corrections(args):
    from .agent import train_fix
    if args.action in ("approve", "reject") and args.id:
        c = train_fix.decide(args.id, approve=args.action == "approve")
        print(f"{c['status']}: {c['id']}" + (f" → {c['applied_to']}" if c.get("applied_to") else ""))
        return
    for r in train_fix.pending():
        print(f"{r['id']}  {r['target']:<13} {r['expert']:<15} {r['rule']}")
    if not train_fix.pending():
        print("no pending corrections")


def cmd_digest(args):
    from . import markets
    from .agent.llm import AnthropicLLM
    from .agent.loop import ROLES
    from .data.provider import BuiltMarketData
    d = markets.write_digest(AnthropicLLM(), BuiltMarketData(live=True), ROLES["recent_markets_analyst"]["model"])
    print(d["text"])


def cmd_reflect(args):
    from .agent import memory
    from .agent.llm import AnthropicLLM
    from .agent.loop import ROLES
    from .data.provider import BuiltMarketData
    kept = memory.reflect(AnthropicLLM(), BuiltMarketData(live=True), ROLES["router"]["model"])
    print(f"[reflect] {len(kept)} lesson(s) passed the gates" + "".join(f"\n  - ({l['kind']}) {l['text']}" for l in kept))


def cmd_usage(args):
    from .agent import budget
    u = budget.usage()
    sp = u["spend_usd"]
    print(f"estimated spend (assumed list prices): ${sp['hour']:.2f} last hour · ${sp['day']:.2f} today · "
          f"${sp['all']:.2f} since {sp['since']}  |  calls: {u['calls_hour']} last hour, {u['calls_day']} today, {u['calls_all']} all")


def cmd_serve(args):
    from .server import serve
    if args.replay and not args.script:
        raise SystemExit("--replay needs --script s0|iran|tariffs")
    if not args.replay:   # the forecaster's local embedder takes ~9 s to load: warm it in the background, off the first turn
        import threading

        def _warm():
            try:
                from .planb import forecaster
                forecaster._model()
                forecaster._library()
            except Exception:  # noqa: BLE001 — warm-up only
                pass
        threading.Thread(target=_warm, daemon=True).start()
    serve(args.script if args.replay else None, port=args.port, portfolio=args.portfolio or None, cached_data=args.cached_data,
          mention_only=args.mention_only, lan=args.lan)


def cmd_train(args):
    from .agent.train import run_train
    run_train(_llm(args), args.track)


def cmd_review(args):
    from .agent.train import review
    review()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; replies use σ, −, etc.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="worsey")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chat", "scan"):
        p = sub.add_parser(name)
        p.add_argument("--portfolio", default="balanced")
        p.add_argument("--synthetic", action="store_true", help="fake data, for plumbing checks only")
    sc = sub.choices["scan"]
    sc.add_argument("--horizon", type=int, default=90)
    sc.add_argument("--n", type=int, default=6)
    ch = sub.choices["chat"]
    ch.add_argument("--script", help="run lines from a file instead of the keyboard")
    ch.add_argument("--transcript", help="save the conversation as JSON")
    ch.add_argument("--quote-tag", dest="quote_tag", help="pin the live-quote cache bucket (replays: reuse the recording's tag)")
    tr = sub.add_parser("train")
    tr.add_argument("track", choices=["options", "events"])
    for p in (ch, tr):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--record")
        g.add_argument("--replay")
    sub.add_parser("review")
    sub.add_parser("usage", help="estimated spend and call counts on the Anthropic key (from the usage ledger)")
    co = sub.add_parser("corrections", help="in-chat /fix corrections: list pending, approve or reject")
    co.add_argument("action", nargs="?", choices=["list", "approve", "reject"], default="list")
    co.add_argument("id", nargs="?")
    sub.add_parser("digest", help="write this week's tape digest every expert reads (recent_markets_analyst, #115)")
    sub.add_parser("reflect", help="propose advisory lessons from the episodic log (code gates them; the trainer promotes)")
    sub.add_parser("etf-panel", help="build/extend the ~3.5y daily price panel for ETF matching (Massive grouped bars)")
    sv = sub.add_parser("serve", help="group-chat web UI (SPEC §1d)")
    sv.add_argument("--replay", action="store_true", help="replay a recorded script offline (zero tokens)")
    sv.add_argument("--script", choices=["s0", "iran", "tariffs"])
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--portfolio", default=None, help="live mode: pre-load a demo portfolio (e.g. balanced)")
    sv.add_argument("--mention-only", action="store_true", help="answer only messages with @worsey (group-chat etiquette)")
    sv.add_argument("--lan", action="store_true", help="reachable from your phone on the same network (or Tailscale); "
                                                       "requires the access code printed at startup")
    sv.add_argument("--cached-data", action="store_true",
                    help="live mode: market data only from the cached build (default: live Massive/Kalshi, cached on disk)")
    b = sub.add_parser("build", help="offline Kalshi universe scan + cached β (run before chat)")
    b.add_argument("--max-markets", type=int, default=None)
    b.add_argument("--names-only", action="store_true", help="only add short display names to the latest build")
    vh = sub.add_parser("vol-history", help="backfill/append the implied-vol history the playbook signals use")
    vh.add_argument("tickers", nargs="*", help="default: config.VOL_HISTORY_TICKERS")
    vh.add_argument("--days", type=int, default=365)
    args = ap.parse_args(argv)
    {"chat": cmd_chat, "scan": cmd_scan, "train": cmd_train, "review": cmd_review, "build": cmd_build, "serve": cmd_serve, "usage": cmd_usage, "etf-panel": cmd_etf_panel, "reflect": cmd_reflect, "digest": cmd_digest, "corrections": cmd_corrections,
     "vol-history": cmd_vol_history}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
