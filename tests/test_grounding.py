"""Grounding fails closed: derived numbers must come from code."""

from worsey.agent import grounding


def test_multiples_and_spelled_multiples_are_checked():
    res = [{"cost": 760.0, "loss": 122.0, "cost_to_loss_multiple": 6.23}]
    assert grounding.check("Costs 6x the loss.", res)["ok"]
    for bad in ("It's about 9x bigger.", "roughly nine times the risk", "costs twice as much", "a 3× multiple",
                "4 times the size"):
        g = grounding.check(bad, res)
        assert not g["ok"], bad


def test_no_fuzzy_match_and_display_rounding_still_works():
    res = [{"a": 4132.0, "p": 0.1823, "x": 480.0}]
    assert grounding.check("about $4.1k at 18%", res)["ok"]
    assert not grounding.check("about $500 less", res)["ok"]   # used to pass on a ±5% match
    assert grounding.check("3 hedges over 2 weeks", res)["ok"]   # plain small counts stay exempt


def test_strip_drops_rows_bullets_and_sentences():
    reply = ("**Short answer.** Costs $760.\n\n| Hedge | Cost |\n|---|---|\n| SPY | $760 |\n| QQQ | $581 |\n\n"
             "- Put spreads cost 9x the loss. They pay whatever the cause.\n- Do nothing is fine.")
    out = grounding.strip(reply, ["$581", "9x"])
    assert "$581" not in out and "9x" not in out
    assert "| SPY | $760 |" in out and "They pay whatever the cause." in out and "Do nothing is fine." in out


def test_format_check_rejects_tables_and_long_replies_and_enforce_fixes_them():
    from worsey.agent.grounding import enforce_format, format_check
    ok = "**Gas is your biggest risk.**\nIt costs $662 if YES.\nWant hedges?"
    assert format_check(ok)["ok"]
    table = ok + "\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert format_check(table)["problems"] == ["markdown table"]
    long = "\n".join(f"line {i}" for i in range(8))
    assert not format_check(long)["ok"]
    assert format_check(enforce_format(table))["ok"] and format_check(enforce_format(long))["ok"]


def test_hedge_figures_must_use_the_card_convention():
    from worsey.agent.grounding import hedge_mismatches
    h = {"loss_if_yes": -992.0, "pnl_if_no": 142.0,
         "rows": [{"name": "Do nothing", "cost": 0.0, "residual_if_yes": -992.0, "book_if_no": 142.0},
                  {"name": "SPY put spread", "cost": 566.0, "residual_if_yes": -1200.0, "book_if_no": -268.0,
                   "cost_per_dollar_covered": 2.99, "notional_usd": 150200.0, "hedge_payoff_if_yes": -208.0}]}
    assert hedge_mismatches("The SPY spread costs $566 and leaves you at -$1.2k if CPI runs hot.", h) == []
    assert hedge_mismatches("The SPY spread loses $208 if YES.", h) == ["$208"]   # the leg's own P&L: other convention
    assert hedge_mismatches("Your book is $60k.", h) == []                         # not a hedge number at all


def test_format_check_reads_short_lines_strictly():
    from worsey.agent.grounding import enforce_format, format_check
    assert format_check("**Headline.**\n- a row\nWant more?")["problems"] == ["bullet list"]
    long = "**Headline.**\n" + "A sentence that keeps going. " * 12 + "\n" + "More detail here. " * 20 + "\nWant more?"
    assert not format_check(long)["ok"]
    fixed = enforce_format(long)
    assert format_check(fixed)["ok"] and fixed.startswith("**Headline.**") and fixed.endswith("Want more?")
