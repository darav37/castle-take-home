# Walkthrough guide: how Worsey's scenario analysis works, from first principles

> Written Oct 24 (through #231). For the current map of the code, read [ARCHITECTURE.md](../ARCHITECTURE.md).

How the event pipeline works, from first principles (written Oct 24; current through DECISIONS #231). Read in this order; each step names the code, the evidence and the key numbers.

## 0. The question and the architecture
- **Castle's question 1:** what events could hit this portfolio, how likely are they, and what would each do to it (counting ETF look-through, matching prediction markets, "already priced in").
- **The rule that shapes everything:** the LLM identifies, structures and explains; code makes every number (#182, after LLM adjustments made results worse, #180). The one exception is black-swan mode, where the LLM writes an explicit shock shown as an editable assumption (#208).
- **Flow of a turn:** router (`src/worsey/agent/loop.py`) → event_analyst expert (`prompts/event_analyst_planb.md`) → tools (`src/worsey/agent/tools.py`) → engine (`src/worsey/planb/live.py`) → cards (`src/worsey/web/app.js`).

**The flow of one question (start here):**
1. **"What could hit me?"**
   - kinds of events: every big past event replayed through the book
   - upcoming Kalshi events, each labelled with what YES does (discovery, #218)
2. **"What if X?"**
   - the LLM structures X: category, severity, a direction and size (small / medium / large) for 7 factors, and any sector or company legs
3. **Factor moves (code):** the measured move for each size call (#214); the factors not set get their typical co-move.
4. **× loadings (code):** macro loadings × factor moves + sector loading × sector move + the company-headline leg.
5. **P&L (code):** options fully repriced; ranges from factor draws plus each holding's own noise; a verdict (noise / small / meaningful vs the book's normal 2-day swing). The analyst can also say "not market-moving" (#219).
6. **Odds (code):** if it happens, × (1 − p); if not, −full × p.
7. **Then:** the hedge verdict, and the LLM writes the words.

**Three different things, often confused:**
- **odds p:** will it happen? Prediction markets, step 6.
- **gate g:** given it happened, is it a real market mover or noise? Plan C. In shadow only, for real news (#201). In a user's "what if" the event is assumed real, so the gate is effectively fully open.
- **size calls:** if real, how big? Step 3, live (#214), measured from the gate work's real-move data.

**Which plan is live?**
- **The chat runs the Plan B engine** (`planb/live.py`, since #179): the factor map, the event library, the scan, the ranges, the cards.
- **Pieces from Plan C's research are wired into it:**
  - "what if" sizing: the real-event part of the gate model and the measured size calls (#213–214)
  - the sector layer (#215–217)
- **Also built this week, beside Plan C:** single-stock tools, options repricing, black-swan mode and the hedge verdict.
- **The Plan C forecaster itself** (the gate deciding whether a news event is real or noise, and which way) is **not live**. It runs as a frozen shadow on the prospective log every morning (#201) and replaces the live estimate only if it holds there (PLAN_C §7).
- **Plan B's original estimate** is still shown as "history's average for events like this".
- **The v1 engine** remains only so the recorded demos replay.

## 1. Layer 1: macro factors (how events reach holdings)
- **What:** 13 named factors (US stocks, 2y and 10y yields, dollar, oil, gold, credit, size, value, EM vs US, healthcare, farm commodities, utilities vs energy).
- **How it works:** each holding's loadings come from a regression over the last 504 days; each holding's move = Σ loading × factor move + its own noise.
- **Code:** `PROXIES`, `factor_returns()`, `loadings()` in `live.py`.
- **Evidence:** 62.7% of a typical ETF's daily moves explained held out (#163); for unusual books, factors explain 77% of event moves vs 37% for market beta alone (#178). Results: `docs/pmrm/results/b1_factors.md`.
- **Limits:** linear; two-year lookback; the 2y and 10y factors overlap (TLT can get the wrong sign in sharp steepenings, #214).

## 2. Layer 2: sectors (#215–217)
- **What:** 19 sector factors (each sector ETF's move beyond the macro factors). Each holding gets its best one, kept if it adds more than 2%.
- **Evidence:** median S&P 500 stock R² 24% → 33% held out (2021–26); Boeing 4% → 34%, KeyCorp 50% → 75%.
- **Spillover:** peers move a fraction of a reporting leader's move (regional banks 25%, semis 12%), same direction about 58–68%; the EPS surprise itself doesn't carry (#216).
- **Code:** `sector_loadings()`, `sector_legs()` in `live.py`; `planb/sectors.json`.
- **Results:** `planc_sectors.md`, `planc_spillover.md`.

## 3. Layer 3: single stocks (#202–206)
- **Earnings:** reaction days average 3.9× a normal day; the surprise calls direction 60% → 54% (fading); no drift after. Options price earnings moves about right; a stock's own history doesn't find cheap options (#203).
- **Company news** (60,909 Benzinga events): analyst rating changes hold direction 77–80%; FDA a coin flip; guidance the biggest movers.
- **After the open:** a stock's gap on news sticks (#206).
- **Code:** `src/worsey/stocks.py` (`earnings_calendar`, `news_base_rates`).
- **Results:** `stock_s0.md`, `stock_s2.md`, `stock_s0b.md`, `stock_s3.md`.

## 4. The event engine: what an event does to the factors
- **History** (Plan B): 3,876 dated events since 2003, each with an LLM schema written without seeing outcomes; code measured each event's factor moves over the report day and the next session.
- **Key finding:** most events are noise. Effects concentrate in big events and a few factors (#162, #171).
- **"What if" mode** (#213–214): the LLM gives a direction and size (small, medium, large) for 7 factors; code uses the measured move for that size call (large = 2.4× normal noise, 2.0× on 2023–26), fills the other factors with their typical co-move, and shows history's average beside it.
- **Black-swan mode** (#208): the LLM writes the shock and code maps it. Gut check on 46 anonymized extreme events: skill 33–39%; anchoring on dissimilar past events hurt.
- **Code:** `price()`, `real_event_draws()`, `stress_scenario()` in `live.py`; `planb/real_event.json`.

## 4b. The event universe: which events, how likely, what's priced (#218, #221–224; full write-up in `docs/event_pipeline/universe.md`)
- **Two streams, matched:**
  - **A, what's coming:**
    - prediction markets (Kalshi build + topic-filtered Polymarket, each labelled once by the LLM: what a YES does)
    - the calendar (Fed, CPI, jobs, GDP, held-name earnings, EIA, WASDE, the election)
    - the last week's labelled news stories
  - **B, what would move this book:** one LLM call reads the book's biggest exposures and proposes 6 events, linked to an A item only when that item is the catalyst.
- **One engine prices everything:** the "what if" engine (directions and sizes → factors → positions, sector and company legs, options repriced).
- **Odds:**
  - a market's odds when it is the event: if YES the book moves full × (1 − p), if NO −full × p; risk left = full × √(p(1 − p))
  - otherwise a base rate, shown as context only, never as this event's odds
- **The match:**
  - headline = coming and ≥ half the book's normal 2-day swing
  - tail risks = big, but no catalyst
  - noise for you = coming, but small
- **The user's events and Castle's (#225):** a named worry plus "anything else" is priced first, with its verdict; Castle's list is everything else, with other routes to the same move folded under the user's event.
- **Coverage test (31 big 2026 events):** an exact market existed beforehand for only 5, and a same-story market for 14 more. Markets price the story more than the step, which is why stream B and the news stream exist.
- **In the chat:** "what could hit me" shows the risk card and the watchlist card.
- **Code:** `src/worsey/discovery.py`, `data/polymarket.py`, `scripts/discovery_label.py`, `scripts/universe_coverage.py`.
- **Results:** `results/universe_coverage.md`.

## 4c. Part II: a probability on every event (#226–230; full rundown in `docs/event_pipeline/part2_summary.md`)
- **Three tiers, first that applies:** (1) a market is the event: its price, longshot-corrected; (2a) same variable, other strike or deadline: interpolated from sibling markets; (2b) related event: the LLM labels the logical relation, code makes bounds; (3) nothing fits: the forecaster.
- **Longshot bias, measured (#227):** 19,792 resolved markets; prices of 2–35% overstate YES (7% → 3.7%), 90–98% understate it; logit correction, out-of-sample log loss −1.7%.
- **Matching (#228):** relation labels 34 of 35 right; bounds hold.
- **Forecaster pilot (#230):** 29 event questions, Brier: naive LLM 0.167, Worsey 0.139, market 0.143.
- **In the chat (#233):** every watchlist event shows its probability with its tier (mkt / derived / bounded / forecast); the forecaster runs on the top 3 unpriced events.
- **Code:** `src/worsey/planb/probability.py`; scripts `part2_longshot.py`, `part2_relations.py`, `part2_questions.py`, `part2_pilot.py`.

## 5. Ranges, options, hedges
- **Ranges:** draws in factor space (past events' factor patterns, so oil and stocks move together) plus each holding's own noise. The 92-event test: Worsey's "9 in 10" held 90% of real outcomes vs 73–77% for the naive LLM (#185–186).
- **Options** (#209): full repricing (spot, IV through the measured spot-vol response, time), with an exact spot/vol/time split. Taylor Greeks fail on big moves.
- **Hedges** (#199–200): real option prices on 4,727 book-events: always hedging costs about $130–150 per $100k per event and pays 3 times in 10, so the card defaults to "probably not worth hedging"; extreme events and a user's bad-case weight flip it.
- **Code:** `options/scenario.py`; `hedge_verdict()` in `live.py`.

## 6. What was tried and failed (the research story)
- **v1 and the PMRM:** daily prediction-market odds explained under 1% of portfolio moves (#153); parked.
- **Plan C:**
  - text embeddings lost to noise (#197)
  - macro events' targeted sectors: no edge on held-out years (#211)
  - the real-event gate + LLM direction is the best forecaster (2026: −0.8% vs noise, 98% support) and runs daily as a shadow on the prospective log (#194, #201)
- **Commodities** (#212): WASDE moves corn 4.6× a normal hour; EIA gas 1.65×, crude 1.34×; hurricane landfalls barely move markets (forecasts price them).

## 7. The comparison
- **Flow map:** [flow_map.html](flow_map.html), every LLM and code step; six traced cases (a normal and a weird book × "what could hit me" / a tariff escalation / aliens) in `docs/pmrm/results/six_cases.json`.
- **The trainer's example book vs the naive LLM:** `docs/pmrm/results/side_by_side_example.md`.
- **One-page summary:** `docs/results/RESULTS_SUMMARY.md` (through #231).

## Known gaps to state plainly
- The forecaster backtest is pending budget; the forecaster in the chat runs only on the top 3 unpriced events, and a scan now takes about 3 minutes.
- Data is frozen at Oct 16 for research (the shadow run uses fresh data).
- Directions after a report and for macro-targeted sectors are near a coin flip.
- The TLT steepening quirk.
- Survivorship (today's S&P 500).
