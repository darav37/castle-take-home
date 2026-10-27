# Part II of question 1: a probability on every event

**Current through DECISIONS #234 (Oct 25, 2026).** The design and build plan are in `docs/event_pipeline/probabilities.md`; results are in `docs/pmrm/results/part2_*.md`.

**The question:** put a probability on each event, using prediction markets where relevant, and show how events are matched to markets and when nothing fits.

## The shape of the answer

Every event in the universe (the user's, Castle's, a market, a calendar item, a news story) goes down three tiers and stops at the first that applies. Every probability carries its tier label.

| Tier | When | Where the probability comes from | Who makes the number |
|---|---|---|---|
| **1** | A listed market *is* the event | The market's price, corrected for the longshot bias | Code |
| **2a** | Same thing, different strike or deadline ("CPI 0.35%" when 0.3% and 0.4% are listed) | Interpolated from the sibling markets | Code |
| **2b** | A different but logically related event ("strike near Tehran" vs "US strikes Iran") | Bounds from the logical relation (implies, implied by, can't both happen) | The LLM labels the relation; code computes the bounds |
| **3** | Nothing listed is close | The forecaster: agents reading Worsey's library and the dated news | LLM agents, aggregated by code |

**"Nothing fits" is an explicit outcome.** If no market passes tier 1 or 2, the event says "no market lists this". It then gets a forecast (tier 3) or, in today's chat, a base rate labelled "context only, not this event's odds".

## 1. Before Part 2 (earlier sessions)

- **Use market levels, not changes:**
  - daily changes in prediction-market odds explained under 1% of portfolio moves (#153)
  - for releases, Kalshi's implied surprise tracked the bond reaction far better than naive estimates: 0.52–0.54 vs 0.12–0.34 (#172)
- **v1 matching code** (`src/worsey/scenario/matching.py`, METHODOLOGY §6):
  - scores a market against the event field by field: variable, threshold, deadline, jurisdiction, resolution source
  - decides exact, derived strike, derived deadline or no fit, with a logged reason
  - ladder interpolation: monotone, never extrapolated
  - deadline rescaling: constant hazard for "happens by date" events, meeting counts for Fed-type events
  - it was never wired into Plan B, and its ~15 hand-labelled test pairs were never done
- **Strike ladders on releases (B5):** across 102 releases, the published number landed inside the Kalshi ladder's own bracket 95–100% of the time. Ladders are trustworthy.
- **Plan B chat (#182):** when the user asks about an event, the expert judges whether a retrieved market *really is* that event, and uses its odds only when it is.

## 2. In the chat today (the event universe, #218–225, #231)

- **Odds on the watchlist:**
  - Kalshi's built universe (222 markets, quotes refreshed hourly) plus a topic-filtered Polymarket feed (about 300 markets), each labelled once by the LLM with what a YES would do to markets
  - the same question on both venues is kept once
- **The priced-in maths (#218):** once the market's odds are p, part of the move is already in prices.
  - if YES, the book moves full × (1 − p)
  - if NO, −full × p
  - the watchlist ranks events by the risk left, |full| × √(p(1 − p))
  - outcomes above 85% or below 15% are flagged "mostly priced"
- **Matching in discovery (#223, #225):** when the LLM proposes an event, or labels one of the user's, it may link a market *only if that market's YES is exactly the event*. The market's odds then become the event's odds, and that market leaves Castle's list so it isn't counted twice.
- **No match:** the event shows a base rate, meaning how often a big event of that kind happened in the horizon (2016–2025). It's labelled "not this event's odds" and never used for the priced-in split, because a base rate is about a *kind* of event and overstates one specific event.
- **Coverage test (#224):**
  - of the 31 big 2026 events, an exact market was open beforehand for only 5, and a market on the same story for 14 more
  - for the trainer's example book, 12% of 2026's event-driven loss was covered by an exact market or an already-running news story, 72% counting same-story markets
  - that's why Part 2 is mostly about tiers 2 and 3

## 3. Part 2 itself: plan (#226) and steps 1–5 (#227–230)

### Step 1, tier 1: is a market's price a fair probability? ($0)

- **Data:** 88,256 snapshots of 19,792 resolved markets, 2023 to June 2026 (Kalshi 11,246 markets, Polymarket 8,546). Every interval is clustered by event.
- **Method:** on the 1st and 15th of each month, every market still open contributes its price and its eventual outcome.
- **The first version was wrong.** I priced markets "7 days before close", but a market closes when it resolves, and "by date" markets resolving YES close early. So the snapshot date depended on the outcome, and it showed the *opposite* bias. Calendar dates fixed it, and every later step uses them.
- **Result: the textbook longshot bias, on both venues.**
  - prices of 2–35% overstate YES: 3% resolves about 1.2% of the time, 7% about 3.7%, 14% about 10%
  - favourites at 90–98% are understated: 92% resolves about 95%
  - between 35% and 90%, the price is fair
  - the same signs hold in the held-out year (Jul 2025 – Jun 2026)
- **Correction:** corrected log-odds = −0.10 + 1.17 × the price's log-odds.

  | Market price | 5% | 10% | 20% | 50% | 90% |
  |---|---|---|---|---|---|
  | Corrected | 2.8% | 6.5% | 15% | 47.5% | 92% |

  - out of sample it lowers log loss by 1.7% (Brier 0.0818 → 0.0811); the step-shaped isotonic alternative does slightly worse
  - it's applied only in the bins where the bias held out of sample (0–35% and 90–98%), and raw and corrected odds are always shown together
  - the parameters are committed in `src/worsey/planb/longshot.json`
- **Why it matters for Castle:** a "5% tail risk" on Kalshi is closer to 3%. On a big-impact event, that roughly halves the risk left.

### Steps 2–3, tiers 2a and 2b (`src/worsey/planb/probability.py`, $0.12)

- **2a, code only:**
  - strike ladders, from "above k" markets or from exact buckets summed into a survival curve: made monotone and interpolated, never extrapolated
  - deadline rungs ("by Nov 1 / Dec 1 / Jan 1"): kept non-decreasing and interpolated; past the last rung only by constant hazard, within 3× the time
  - examples: CPI above 0.35% → 15.3% (between the 0.3% and 0.4% markets); gas above $5.50 → 12.5%; Hormuz normal by Feb 15 → 12.7% (between the Jan 1 and Apr 1 rungs)
- **2b, the LLM labels and code bounds:** one call per event labels each candidate market as same, other strike, other deadline, implies, implied by, can't both happen, correlated, or unrelated.
  - code turns the logical relations into bounds and intersects them
  - if the interval is narrow (≤ 10 points), Worsey reports its midpoint, labelled "derived from markets X, Y"
  - otherwise the event goes to tier 3, and that forecast is clipped to the bounds
- **Test** (`docs/pmrm/part2_pairs.json`): 12 events against 35 live Kalshi markets, covering every relation type.
  - 34 of 35 labels right. The miss is arguably correct: the model spotted that the Netanyahu market also resolves YES on an *announcement*, so it isn't exactly the event.
  - all 4 bounds checks held
- **Caveat:** the build assistant labelled the pairs; they're for

### Step 4, the question set for testing a forecaster ($0)

- **The set:** 1,409 questions from Kalshi and Polymarket markets that resolved Jul 1 – Oct 23, 2026, after the model's knowledge. One question per event.
- **Forecast dates:** a calendar date (Jul 1, Jul 15, Aug 1, Aug 15, Sep 1), never "N days before close".
- **The market's price on that date** is hidden from the forecasters and kept only as the benchmark.
- **Finding:** only about 7% are geopolitical or macro events. The rest are local primaries, tweet counts and company KPIs.

### Step 5, the tier 3 pilot (29 event questions, about $3.20)

- **Arms:**
  - A: a naive LLM (question, rules, date)
  - B: A plus the library base rate
  - C: Worsey's forecaster, three agents each given the base rate, the 8 most similar past library events and the Wikipedia news from the 14 days before the date, in different orders, combined by the median in log-odds
  - D: the hidden market price

| Arm | Brier (lower is better) |
|---|---|
| A. naive LLM | 0.167 |
| B. naive + base rate | 0.172 |
| **C. Worsey library forecaster** | **0.139** |
| D. market price (hidden from the forecasters) | 0.143 |

- **What it shows:**
  - **C beats A** by 0.028, with a 95% interval of [−0.060, −0.0001]. That's just significant on 29 questions.
  - **C ties the market**: a difference of −0.004, with a wide interval.
  - **The base rate alone adds nothing.** The gain comes from the similar past events and the dated news.
- **Caveats:**
  - it's small
  - the market benchmark is noisy on this set (for example, a 95% hurricane-name market that resolved NO)
  - there's no calibration step yet (that needs the backtest's held-out set)
  - the web-search agents in the design are untested (they can only be scored prospectively, because web search can't be date-restricted)

## 3b. Step 7: in the chat, end to end (#233, about $0.25 per scan)

- **Every event on the watchlist carries its tier:** mkt (a market is the event, its price longshot-corrected, raw price on hover), derived, bounded, or forecast; anything else keeps the labelled base rate.
- **Tier 1** is code; **tiers 2a/2b** cost one batched relation call per scan; **tier 3** runs the pilot's three-agent forecaster on the **top 3** unpriced events by impact only.
- **The priced-in split** uses the raw market price (what asset prices carry); for a forecast it's labelled "assumes markets share this forecast".
- **The trainer's book, live:** the 5-year at 5.25% shows 15.9% (market 21%); gas above $5.80 6.4% (market 10%); the AI bubble burst 6.1% (market 9.3%); two tail risks with no market get forecasts: a private-credit or regional-bank failure 4% (agents 4–6%), wider China export controls 22% (agents 20–25%).
- **#234, what the user sees:** the user's events plus Castle's three picks (by risk left, among events with a real probability, no two with the same move), as text messages sent as the scan goes: the book at a glance (~20 s), the user's events with an honest verdict, "researching the odds", then the final message with every probability and its tier (~90 s). One owner for odds: a proposal's market link is only a hint; the relation call decides. The forecaster runs on every unpriced user event and the top 3 unpriced proposals, with related markets as evidence.

## 4. What's left

| Item | State |
|---|---|
| Backtest, about 250 event questions (calibration fitted, clustered intervals) | Passed the pilot gate. Costs ~$30–40; the cap is now $90 (about $24 left). The trainer's call. |
| Prospective test with search agents (open markets, scored as they resolve) | Not started (~$30) |
| Step 7: every watchlist event shows its probability and its tier | **Done (#233)**; forecaster limited to the top 3 events; scan latency to fix |
| The trainer's review of the 35 pairs and the 29 pilot questions | Open |

## The one-paragraph version

Worsey uses a prediction market's price when a market really is the event, corrected for a longshot bias measured on about 20,000 resolved markets (5% really means about 3%). It derives a probability from nearby markets when only the strike or deadline differs, and bounds it by logic when the event is related but different. That match is decided field by field, and "no fit" is an explicit, logged outcome. When nothing fits, a small forecaster reading Worsey's own event library and the dated news beat a naive LLM and tied the market on a 29-question pilot. The backtest that would make that claim solid is designed and costed.
