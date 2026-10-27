# Glossary

Every term you'll see in Worsey's cards, docs and code, in plain words. Current through DECISIONS #238 (Oct 25, 2026). Grouped in the order of the pipeline; the index at the end is alphabetical.

## The book and its exposures

- **Book.** The user's portfolio: stocks, ETFs and options. The example book: PFE $20k, WDC $15k, TSLA $20k, URA $15k, GLD $20k, 2 USO $157 puts.
- **ETF look-through.** Seeing through an ETF to what it holds: someone who owns QQQ also owns NVDA. `lookthrough.effective_exposure()`.
- **Factor.** One of 13 named market-wide drivers that most assets move with:
  - US stocks
  - 2-year yields, 10-year yields
  - the dollar
  - oil, gold
  - credit (high-yield minus Treasuries)
  - size (small vs large caps)
  - value vs growth
  - Asia/EM vs US
  - healthcare
  - farm commodities
  - utilities vs energy

  Each is measured from ETFs (e.g. oil = USO, credit = HYG − IEF).
- **Loading (beta).** How much a holding moves per unit of a factor's move. It comes from regressing the holding's daily returns on the factors over the last 504 trading days (about 2 years), refit daily.
- **Sector factor / sector layer.** A US sector ETF's move beyond what the 13 factors explain (e.g. semis = SMH beyond the macro factors). Each holding gets at most one: the sector that best explains what the factors miss, kept only if it adds more than 2% explained variance on held-out data. There are 19 sectors.
- **Country factor / country layer (#238).** The same idea for 31 country ETFs (EWZ = Brazil, etc.). Country ETFs map to their own country, foreign stocks to their home country.
- **Own noise (residual).** The part of a holding's daily move the factors don't explain. It isn't thrown away: it goes into the ranges and the bad case.
- **R² (explained share).** The share of a holding's daily moves the factors explain:
  - about 63% for a typical ETF held out
  - about a third for a typical stock (24% → 33% with sectors)
  - more for a diversified book
- **Coverage.** How much of the whole book's risk the factors explain. Under about 75%, the card warns that the factors only partly cover this book.
- **Normal 2-day swing.** The book's typical move over two sessions: its daily standard deviation × √2. The yardstick for "is this event big for this book?"

## Events and labels

- **Event.** Anything that could happen and move markets: a market's YES, a Fed meeting, a news story developing, an event the LLM proposes, or one the user names.
- **Label.** The structured description of what an event does to markets:
  - a **direction** (+ or −) and a **size** (small, medium or large) for each of 7 factors
  - optional **legs** (sector, country, company)
  - a category and severity

  Labels are LLM judgments. Every number is computed by code.
- **Size call.** The LLM's small, medium or large on a factor. Code turns it into the move that past events with that call actually produced. Live labeller backtest: large ≈ 2.5–2.7× normal noise, small ≈ 0.85–0.9×.
- **Leg.** An extra move on top of the factors, for holdings in a sector, country or company the event directly hits. Only the largest leg counts per holding, never the sum (`combine_legs()`, #232). Company legs only for events about that company.
- **Co-move.** For factors the label leaves blank, code fills in their typical move alongside the signed factors, from past events.
- **Event library.** 3,876 dated past events since 2003:
  - every scheduled release (Fed, CPI, jobs, GDP)
  - plus economically relevant items from Wikipedia's daily current-events pages

  Each was labelled blind (no outcomes), and code measured what markets did.
- **Blind labelling.** The LLM labels an event from its text only, never seeing what happened. That's how the labels were tested.
- **Labeller backtest (#236).** The live labelling prompts, run on 745 big past events. Direction right on big moves: 74% (markets), 73% (user events). Size calls ordered.

## The universe (Part I: which events)

- **Universe.** Every candidate event for this book and this question, before probabilities and ranking.
- **What's coming.** Listed upcoming items:
  - prediction markets (Kalshi, Polymarket)
  - the calendar (Fed, CPI, earnings, EIA, WASDE, elections)
  - the last 7 days of labelled news

  Priced on arrival.
- **Past event replay (risk card).** Every kind of past event run through this book: which kinds hurt it most (e.g. financial stress, bad case −$4.5k). Shown as its own card (`live.scan()`). The top 4 kinds also go into the proposal call.
- **Exposures.** The book's biggest factor, sector and name exposures in dollars per normal day (`exposures()`).
- **Proposal call.** One LLM call (`propose()`). It reads your events, the history (top 4 kinds), the exposures and what's coming. It labels your events and invents 6 book-driven events, at least 2 of them current versions of the historical kinds. It links each to its catalyst where one exists.
- **Your events (user events).** Events the user names ("I'm worried about an oil spike"). Always priced and shown first.
- **Catalyst link.** A proposal's pointer to the upcoming market, date or story that would trigger it. Used for timing only, never for odds (#234).
- **Fold.** Anything that is really one of your events (same market, or same main factor move) goes under it as "related", so Castle's picks are about something else (`same_move()`).
- **Absorb (remove duplicates).** A proposal that *is* a listed market or story replaces it, so it isn't shown twice.

## Probabilities (Part II: how likely)

- **Tier.** Where an event's probability came from. Checked in order; the first that applies wins:
  - **Tier 1, exact market:** a market is the event. Its price, longshot-corrected.
  - **Tier 2a, derived:** same variable, different strike or date. Read off the market's own ladder of strikes or deadlines.
  - **Tier 2b, bounded:** a logically related market sets a ceiling or floor (if the event implies the market, P(event) ≤ P(market)). Used if the bounds are within 10 points.
  - **Tier 3, forecast:** nothing fits. The forecaster runs (your events, plus the 3 biggest of Castle's).
  - **Base rate:** everyone else. How often big events of that kind happened. Context only, not this event's odds.
- **Longshot correction.** Prediction markets overprice unlikely outcomes and underprice near-certain ones. Fitted on 19.8k resolved markets:
  - a 7% price really happens about 3.7% of the time
  - logit(p) = −0.10 + 1.17·logit(price)
  - applied only in the price ranges where the bias was significant
- **Relation call.** One LLM call (`classify_many()`) that labels how each event relates to its candidate markets: same, other strike, other deadline, implies, implied by, exclusive, correlated or unrelated. No numbers.
- **"Nothing fits".** No candidate is the same event, none can be derived, and the bounds are wider than 10 points.
- **Forecaster.** 3 LLM agents, each seeing:
  - the base rate
  - the 8 most similar past events
  - 14 days of news
  - related markets

  Combined by the median in log-odds, clipped to [2%, 98%] and to any bounds (`forecaster.forecast()`).
- **Brier score.** A forecast's mean squared error against what happened (0 = perfect; lower is better). Pilot, 29 questions: Worsey 0.139, naive LLM 0.167, market 0.143.

## Impact (Part III: how big) and the numbers on the card

- **Full impact.** The book's move if the event happens as a total surprise: label → factor moves → loadings → holdings, plus legs, options repriced.
- **Priced in.** Today's prices already carry p × the event:
  - **if YES:** the book moves full × (1 − p)
  - **if NO:** it moves −full × p

  The split uses the raw market price, since that's what asset prices carry.
- **Risk left.** |full| × √(p(1 − p)): how much the book can still swing when the event resolves. Largest for coin flips, small for near-certain events. Used to rank the "three to watch".
- **Pick three.** Castle's three events beyond yours: the top 3 by risk left, among events with a real probability (tier 1–3), with no two the same move (`choose_three()`).
- **Size screen / impact threshold.** An event counts as big for this book if its full impact is at least half the book's normal 2-day swing (about $1.1k for the example book).
- **Verdict.** The size label on an event, compared with the normal 2-day swing:
  - **noise:** under 0.5×
  - **small:** 0.5–1×
  - **meaningful:** over 1×
- **Middle.** The central estimate of the move.
- **Ranges:**
  - **likely range:** the middle half of outcomes (25th–75th percentile)
  - **9-in-10 range:** the 5th–95th percentile

  Both come from drawing factor moves from the real spread of past events, plus each holding's own noise. Tested: Worsey's 9-in-10 held 90% of real outcomes vs 73–77% for the naive LLM (#185–186).
- **Bad case.** A tail outcome, always stated with its odds:
  - **1 in 20 (5th percentile):** on the watchlist and scenario cards
  - **1 in 10:** on the risk card's replay of past kinds

  On the watchlist, options are held at their middle value in the bad case.
- **Options repricing.** A full Black–Scholes reprice at the new spot, a new implied vol and the scenario's time. Not Greeks, because Taylor Greeks fail on big moves.
- **Spot-vol response (beta).** How much an underlier's implied vol has historically moved per spot move, fitted separately for up and down moves. Behaves roughly like sticky strike near the money.
- **Shared noise (#235).** A stock and options on it use the same random draw in each scenario, so a put hedges the stock's own swings.
- **Hedge verdict.** Whether hedging an event is worth it. A separate turn, only when the user asks. Default: probably not, since always hedging cost about $130–150 per $100k per event and paid 3 times in 10.
- **Black-swan mode.** For events bigger than anything in the library: the LLM writes an explicit shock, shown as an editable assumption.

## Chat and architecture

- **Router.** The first LLM call in a turn. It picks what kind of question this is and pulls out your events (`run_turn()` → `scan_risks`).
- **Expert / event_analyst.** The LLM that writes the reply from code's numbers (`run_expert()`).
- **Tool call.** The LLM asks the program to run a named function; the program runs it and returns the result.
- **Grounding check.** Every number in a reply must match a number from a tool result. Otherwise one rewrite, then the sentence is deleted (`grounding.check()`).
- **Card.** The structured box in the chat UI (risk card, watchlist), drawn from code's data.
- **Messages 1–3.** Code-written messages sent while the scan runs: your book at a glance, your events, the forecasts. The LLM writes only the last message.
- **Rule #182.** The LLM identifies, structures and explains. Code makes every number.

## Research history and testing

- **v1 / PMRM.** The first engine: daily prediction-market odds changes vs portfolio moves. Explained under 1% (#153). Parked; kept for the recorded demos.
- **Plan B.** The current engine: factors + the event library + labels. Live since #179.
- **Plan C.** The research layer: the real-event gate, sectors, single stocks, the forecaster. Pieces are wired into Plan B.
- **Gate.** Plan C's model of "is this news a real market mover or noise?" Shadow only (#201).
- **Shadow run / prospective log.** A daily 07:30 job that logs the news and frozen forecasts, scored later. Evidence that can't leak.
- **Held out / out of sample.** Tested on data the model never saw while being fit.
- **Leakage.** The model knowing the answer: future data in a backtest, or an LLM remembering events from training (knowledge to about June 2026, so the clean tests use events after July 1, 2026).
- **Naive LLM.** The same Claude model with only the holdings text: no tools, no data. The baseline.
- **CRPS.** A score for a whole forecast distribution, not just a point (lower is better). Used for the gate.
- **DECISIONS #N.** The numbered entry in `docs/DECISIONS.md` recording the decision and its evidence.

## Index

absorb · bad case · base rate · black-swan mode · blind labelling · Brier · card · catalyst link · co-move · country layer · coverage · CRPS · ETF look-through · event · event library · expert · exposures · factor · fold · forecaster · full impact · gate · grounding check · hedge verdict · held out · label · labeller backtest · leakage · leg · loading · longshot correction · messages 1–3 · middle · naive LLM · normal 2-day swing · nothing fits · options repricing · own noise · past event replay · pick three · Plan B / Plan C · PMRM · priced in · proposal call · R² · ranges · relation call · risk left · router · rule #182 · sector layer · shadow run · shared noise · size call · size screen · spot-vol response · tier · tool call · universe · verdict · what's coming · your events
