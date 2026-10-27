# Part 2 of question 1: a probability on every event

> **Design document and build plan** (Oct 24–25, with a status block). For how the code works today, read [ARCHITECTURE.md](../ARCHITECTURE.md), section 4.

**The question:** put a probability on every event, using prediction markets where they fit, and show how events are matched to markets and when nothing fits.

**Status, Oct 25, 2026 (DECISIONS #226–233):** steps 1–5 and 7 done (the chat shows every event's probability with its tier, #233; one owner for odds, and the forecaster on the user's events plus the top 3 unpriced proposals, #234); the backtest (step 6) is not run (budget).
- **Step 1, tier 1 (#227):** the longshot bias is real on both venues (prices of 2–35% overstate YES, 90–98% understate it); a two-parameter logit correction improves out-of-sample log loss by 1.7%; applied only where the bias holds out of sample. `results/part2_longshot.md`, `planb/longshot.json`.
- **Steps 2–3, tiers 2a and 2b (#228):** `planb/probability.py`; relation labels 34 of 35 right on the test pairs (`part2_pairs.json`, for the trainer to review); bounds checks hold. `results/part2_relations.md`.
- **Step 4, questions (#229):** 1,409 resolved questions; only ~7% are about the library's event kinds. `results/part2_questions.md`.
- **Step 5, pilot (#230):** on 29 event questions, Brier: naive 0.167, naive + base rate 0.172, **Worsey library forecaster 0.139**, market 0.143. C − A = −0.028 [−0.060, −0.0001]. `results/part2_pilot.md`.
- **Method finding:** never measure "N days before close": close depends on the outcome. Calendar snapshots throughout.

## Where we stand

- **Markets:** a market's odds are used when a market really is the event (#218). Kalshi has 222 built markets and Polymarket about 300 topical ones (#224).
- **Everything else:** gets a labelled base rate shown as "context only, not this event's odds" (#223).
- **The coverage test (#224):** an exact market existed beforehand for only 5 of 31 big 2026 events, and a market on the same story for 14 more. For the trainer's book, 12% of 2026 event-driven loss was covered strictly, 72% counting same-story markets.
- **So:** most events that matter fall into tiers 2 and 3 below. Part 2 is mostly about those.

## The pipeline: three tiers, one event E at a time

Each event in the universe (user's, Castle's, markets, calendar, news, proposals) goes down the tiers and stops at the first that applies. Every probability is labelled with its tier.

### Tier 1. E is a listed market: use the market's odds

- **Test for "is":** same variable, threshold, deadline, resolution source and jurisdiction. Use the v1 field scoring (`scenario/matching.py`, METHODOLOGY §6).
- **Longshot correction:** prediction markets overprice low-probability outcomes (Snowberg & Wolfers 2010).
  - Measure it; don't assume it. Fit a calibration curve (market price → realized frequency) on resolved Kalshi and Polymarket markets that resolved **before Jul 1, 2026**, binned or isotonic.
  - Apply it only where the data shows a significant bias, probably below about 10%.
  - Show the raw and corrected odds side by side.
  - Cost: code only, $0.
- The priced-in maths stays as in #218.

### Tier 2. E is close to listed markets: derive the odds from them

This tier covers two different problems.

**2a. Same variable, different strike or deadline (code, no LLM).**
- **Strike ladders:** "CPI prints at 0.4%" when only "above 0.3%" and "above 0.5%" are listed. Use monotone interpolation of the ladder's survival curve, never extrapolate, and report the bracket. This is built for releases in B5 (`results/b5_ladders.md`, 102 releases; the published number sat inside the ladder's own bracket 95–100% of the time) and in v1 matching.
- **Deadlines:** "by Dec 31" when "by Nov 30" is listed. Use a constant hazard for "by date" events and per-meeting probabilities for Fed-type events, as in v1 matching.
- **Work:** wire v1's ladder and deadline derivations into Plan B's discovery path. They aren't wired today (UNIVERSE.md).

**2b. A different but related event (LLM classifies the relationship, code computes).**
Example: "US air strike on a base outside Tehran" vs a listed "US strike on Iran" or "Hormuz closed".
- **The LLM doesn't adjust odds.** Adjusting them would be the LLM producing a number, which breaks #182. Instead, for each candidate market M, it classifies the relationship and gives one short reason. The candidates come from the existing retrieval: keyword plus the same-move folding in #225.
  - **E ⊂ M** (E implies M, e.g. Tehran strike ⊂ any strike on Iran) → P(E) ≤ P(M)
  - **M ⊂ E** → P(E) ≥ P(M)
  - **mutually exclusive** → P(E) ≤ 1 − P(M)
  - **correlated, no logical link**: no bound; M is passed to tier 3 as evidence
  - **unrelated**: drop M
- **Code** intersects the bounds from all related markets into an interval [lo, hi].
  - If the interval is narrow (hi − lo ≤ 0.10), report its midpoint, labelled "derived from markets X, Y".
  - Otherwise send E to tier 3, with the bounds and the markets as inputs. Code clips the tier-3 forecast to [lo, hi].
- **"Close enough", defined:** tier 1 = exact on all five fields. Tier 2a = the same variable with only the strike or deadline differing. Tier 2b = the LLM finds a logical relation with at least one market. Anything else is tier 3.
- **Test:** the ~15 hand-labelled event-to-market pairs in the project notes (never done; still needed), extended to about 30 so the 2b relation cases are covered. Score the relation labels for agreement, and check that bounds never exclude the market-implied truth.
- **Cost:** one LLM call per event covers all of its candidate markets, about $0.01. Under $1 to test.

### Tier 3. Nothing listed is close: the forecaster

The design follows Halawi et al. 2024 and Bridgewater's AIA Forecaster (2025): several research agents, aggregated, then calibrated. Worsey's addition is its own labelled event library as structured evidence.

- **Agents, 5 per event:**
  - **3 library agents (no web search).** Each gets the event in the schema, plus:
    - the reference class: the library's base rate for that kind of event within the horizon, and the most similar past events, with whether and when the next step followed
    - the news up to the forecast date (prospective log, Wikipedia current events by date)
    - any tier-2 bounds and related markets
  - **2 search agents.** The same evidence, plus web search (capped, `WEB_SEARCH_MAX_USES`), told to research and conclude.
  - Each agent returns a probability and three lines of reasoning. They're the same model at nonzero temperature, with different evidence orderings.
- **Aggregation (code):** the median of the 5 in log-odds. Then Platt scaling, fitted on a held-out split of the test set. Clip to the tier-2 bounds if any, and cap at [2%, 98%].
- **Shown as:** "Worsey forecast, no market lists this", with the 5 agents' spread as its uncertainty.
- **Priced-in maths:** the #218 split assumes asset prices already carry the market's p. For a tier-3 event we don't observe that p. So the card shows forecast × impact as the expected move and labels the priced-in split "if markets share this forecast".

## The test: beat a naive LLM on Brier, and show the gap to the market

**The claim for Castle:** a small, honest forecaster that measurably beats a naive LLM, with its distance from the market reported plainly. Expected result: it beats the naive LLM and loses to the market.

**Ground truth:** resolved Kalshi and Polymarket markets, treated as if unlisted. The market is hidden from every arm, and its price at the forecast date is kept as a benchmark.

**Leakage (Paleka et al. 2025, "Pitfalls in Evaluating LM Forecasters"):**
- The model's knowledge runs to about June 2026. So the backtest uses only questions that **opened before and resolved between Jul 1 and Oct 23, 2026**, forecast as of a date T before resolution (14 days before, or the market's midpoint if it's shorter).
- **Evidence is frozen at T:** library events dated ≤ T, Wikipedia current events ≤ T, the prospective log ≤ T.
- **Web search can't be date-restricted**, so the 2 search agents leak in any backtest. They're scored only **prospectively**: open markets forecast now, scored as they resolve.
- One question per event, so ladder rungs don't count as independent.
- Drop questions whose answer is mechanically implied by an asset price on date T (the reflexive filter, as in the build).

**Arms (backtest):**
| Arm | What it gets |
|---|---|
| A. Naive LLM | the question and date T only, no tools (the same model) |
| B. Naive + base rate | A plus the library base rate (isolates how much the library alone adds) |
| C. Worsey library forecaster | 3 library agents, aggregated and calibrated |
| D. Market | the hidden market's price at T (the benchmark, not a competitor) |

**Prospective arm:** C + the 2 search agents (the full tier 3) vs A on currently open markets, scored as they resolve. Report whatever has resolved, plus the shadow log.

**Metrics:**
- Brier (the headline) and log score
- calibration curve
- a paired bootstrap, clustered by event, of C − A and C − D, with 95% intervals

**Size:**
- about 250 independent questions in the backtest
- a Brier gap of about 0.01 needs roughly this many to be significant
- below about 150, report the gap but don't claim it

**Optional, ForecastBench:** if its public question sets include enough questions resolving after Jul 1, 2026, run A vs C on 100 of them. Skip it if not; the market questions are the main test.

## Budget (assumed list prices in `agent/budget.py`: Sonnet $3/$15 per M tokens, $0.01 per search)

| Step | Calls | Estimate |
|---|---|---|
| Tier 1 longshot fit | none | $0 |
| Tier 2b relation labels + test | ~100 | ~$1 |
| Pilot: 30 questions, arms A–C, with a dry run | ~150 | ~$5 |
| Backtest: 250 questions × (A + B + 3 library agents) | ~1,250 | ~$30–40 (library agent ~$0.06 with the shared context cached; naive ~$0.01) |
| Prospective: 40 open questions × 5 agents (2 with search, ~$0.30 each) | ~200 | ~$30 |
| Reruns, prompt fixes, the chat integration | | ~$10 |
| **Total** | | **~$75–85** |

- **Spend now:** about $70 of the $80 cap.
- **Proposal:** raise the total cap to **$160**, gated:
  1. run the pilot ($5)
  2. go on to the backtest only if C beats A on the pilot or the gap is close
  3. run the prospective arm last
- **Cheaper version (~$25):** 150 questions, no search agents, no prospective arm. It probably can't show significance, but it shows the method.
- Every LLM step follows SPEC §6.1: dry run first, `--confirm`, per-run caps, a permanent label cache.

## Build order (for the build assistant)

1. **Tier 1:** the longshot calibration fit on markets resolved before Jul 1, 2026. Report in `results/part2_longshot.md`.
2. **Tier 2a:** wire v1 ladder and deadline derivations into discovery.
3. **Tier 2b:** the relation classifier, the bounds in code, and the labelled pairs.
4. **The question set:** resolved markets from Jul 1 to Oct 23, 2026, one per event, the reflexive filter, evidence frozen at T. The trainer approves the list.
5. **Pilot (30 questions), then stop for review.**
6. **Backtest (250), then the prospective arm.** Report in `results/part2_forecaster.md`.
7. **The chat:** every event in "what could hit me" shows its probability and its tier label.

Stop after each step for the trainer's review.

## Out of scope (kept for later)

SDE and hazard-rate models of standardized events, and how they interact (a world model). These are the trainer's longer-term direction, not this take-home.
