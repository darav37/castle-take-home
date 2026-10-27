The trainer, an options trader, just corrected one of Worsey's replies with /fix. Turn his correction
into something Worsey can apply. You see the original question, Worsey's reply, which expert wrote it, and the trainer's fix.

1. `rule`: one line, in the trainer's terms, that would have produced the right answer. Keep his meaning exactly; don't
   soften or generalize beyond what he said. If he gave a number threshold, keep it verbatim.
2. `expert`: which expert the rule is for (options_desk, event_analyst, equity_analyst, etf_desk, recent_markets_analyst,
   or router for routing/voice).
3. `topic`: 1–4 keywords for when it applies (e.g. "earnings, long calls").
4. `target`: where it belongs permanently: "strategy_file" (about one strategy; give its id), "playbook" (a general
   options rule), or "prompt" (behavior, voice, routing, or a non-options expert).
5. `strategy_id`: only for target "strategy_file".
6. `restatement`: one short sentence Worsey says back to confirm what it learned.

Reply with JSON only: {"rule": "...", "expert": "...", "topic": "...", "target": "...", "strategy_id": null, "restatement": "..."}
