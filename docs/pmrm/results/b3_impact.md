# Plan B B3: event → factor impact model, held-out tests

Run 2026-10-20 by `scripts/planb_b3.py` (no LLM calls; uses the B1 model and the B2 library). Model code: `src/worsey/planb/impact.py`. Raw numbers: `b3.json`. BUILD_PLAN B3, PLAN_B §2, §4, §5.3.

**Stop here for the trainer** (BUILD_PLAN B3).

## In plain words

We tested whether the event library can predict what an event does to markets, on events the model never saw.

**1. Scheduled releases (Fed, CPI, jobs, GDP, PCE): a small, real effect, not yet useful for portfolios.**
- **Setup:** trained on 2003–2020, tested on 2021–2026 (310 releases in the Kalshi era).
- **Factor level:** given how much interest rates jumped on the release, the model predicts a little of how the market, dollar, oil, credit and style factors moved that day: **3.6% better than knowing only the release type**. That beats chance, measured by shuffling which release got which surprise.
- **Portfolio level:** it helps the demo balanced book (+1.1%) and the energy basket (+5.2%), both beating chance. It hurts most other portfolios.
- **Why:** the stock–bond relationship flipped after 2021. Before, a jump in yields on a release day usually meant good growth news and rising stocks; since 2021 it usually means inflation fear and falling stocks. Training on 2003–2020 learns the old sign. Retraining each year on the previous 8 years didn't fix it (1.8%).
- **Using the data alone** (actual minus a naive expectation) predicts nothing. Our free "expected" value is too crude; Kalshi's strike ladders will give a far better one (B5).

**2. Unscheduled events (tariffs, wars, disasters, bank failures…): no predictive skill yet.**
- **Setup:** each of the 194 events was predicted from its 5 most similar *other* events (same category, overlapping sectors, similar severity and region).
- **Direction:** the analogs get it right **about half the time, a coin flip**. Their predictions are *worse* than predicting no move, and no better than five random events.
- **Size alone:** analogs don't predict how big the move will be either. Grouping by severity helps by only about 1%.
- **The one thing that works:** when the analogs disagree, the real outcome is further from the prediction (correlation 0.25). So the spread of past analogs is an honest, if wide, uncertainty band.

**Why, and what fixes it:**
1. **The event descriptions have no direction.** "Trade and tariffs" mixes new tariffs with truces; "sanctions" mixes new sanctions with relief; "central bank" mixes surprise cuts with surprise hikes. Averaging analogs from a category cancels the direction out. The fix is a signed field (escalation vs relief, risk-on vs risk-off) plus the LLM comparison step planned for B4 (is this event bigger or smaller, more or less surprising, than each analog?).
2. **194 events is small** for 10 categories and very noisy 1–5 day windows, where everything else that happened those days is in the move too. The full library should be built systematically (GDELT) with thousands of events.
3. **Big, unusual events (COVID, SVB, 2008) dominate.** They are exactly where analogs are scarcest.

**What it means for "what events could hit my portfolio, and what's the damage?"** Today the library honestly supports **ranges, not point forecasts**: "events like this have moved a book like yours by −X% to +Y%". For scheduled releases, the model's small factor-level skill is real but needs better surprise measures and regime awareness before it beats a simple average at the portfolio level.

## (a) Scheduled releases, held out by time

**Setup:**
- **Data:** 1,306 releases from late 2003. Train: 996 (to 2020). Test: 310 (2021-01 to 2026-10): CPI 69, FOMC 46, GDP 68, jobs 70, PCE 69.
- **Response:** the release-day factor move (9 anchored, in standardized units, plus 9 latent).
- **Model:** reduced-rank ridge on type-specific surprises (shared low-rank structure). Rank chosen by blocked time-CV inside training.
- **Baseline:** each type's training-mean response.
- **Chance:** test surprises shuffled within type (200 permutations).

| Surprise measure | Rank | Non-rates factors R² (vs type mean) | Chance 95th pct | Latent factors R² | Rates factors R² |
|---|---|---|---|---|---|
| Market-based (Δ2y, Δ10y, Δ5y breakeven) | 3 | **+3.6%** | −8.9% | +20.5% | +99.9% (mechanical: the surprise is a rates move) |
| Data surprise (actual − naive expectation) | 1 | +0.2% | +0.1% | +0.3% | +0.1% |
| Market-based, rolling 8-year refit (PLAN_B §4) | 3 | +1.8% | — | — | — |

**By release type** (market factor, R² vs type mean): GDP +12%, jobs +7%, CPI −23%, FOMC −38%, PCE −14%. Inflation and Fed days are where the regime flip bites.

**Portfolios** (release-day move predicted through each portfolio's factor loadings, fitted 2011–2020 daily; R² vs type mean):

| Portfolio | Via anchored factors | Chance 95th | p | Direct regression | Rolling 8-year |
|---|---|---|---|---|---|
| Demo `balanced` book | **+1.1%** | −1.2% | 0.05 | +0.8% | +1.0% |
| Energy bucket | **+5.2%** | −3.4% | 0.05 | +5.1% | +3.6% |
| Study A (XOM, DAL, MSFT) | −2.0% | −1.3% | 0.14 | −10% | −4.9% |
| Demo `tech_heavy` | −9.7% | | 0.67 | −1.7% | −10.5% |
| Demo `policy_exposed` | −12.6% | | 0.52 | +5.4% | −8.7% |
| Rate-sensitive | −10.4% | | 0.67 | −4.0% | −6.1% |
| Travel and consumer | −8.4% | | 0.71 | −3.4% | −9.9% |
| Mega-cap tech | −8.5% | | 0.67 | −0.8% | −7.9% |
| Defensives | −5.4% | | 0.62 | −9.1% | −5.4% |
| Placebo | −8.7% | | 0.86 | −2.5% | −11.6% |

**Bridge check** (how well the data surprise lines up with the rates reaction): correlation CPI +0.11, jobs +0.17, PCE −0.03, GDP −0.23 (41). The naive expectations are too crude. **B5 will measure surprises against the Kalshi-implied mean** (dense ladders since 2022, B0 §4).

## (b) Unscheduled events, leave one event out

**Setup:**
- **Data:** 194 events. The short window is 1 session (time known) or 2 sessions (time unknown); the long window is 5 sessions.
- **Prediction:** the similarity-weighted mean of the 5 most similar other events' realized factor responses. Similarity = category 0.40, sector overlap 0.25, severity 0.15, US vs non-US 0.10, anticipation 0.10 (fixed before scoring).
- **Baselines:** zero (no information) and the category mean.
- **Chance:** 5 random other events (200 draws).

| Window | Predictor | Non-rates R² vs zero | Latent R² vs zero | Market direction right |
|---|---|---|---|---|
| Short | Analog (all other events) | −16.2% (random-analog 95th: −14.0%, p = 0.16) | −17.9% | 46% |
| Short | Analog (earlier events only, hindsight-free) | −10.9% | −16.7% | 48% |
| Short | Category mean | −2.9% | −4.2% | 38% |
| 5 sessions | Analog (all other events) | −15.0% (random 95th: −14.8%, p = 0.08) | −18.9% | 51% |
| 5 sessions | Category mean | −4.1% | −4.6% | 44% |

- **Portfolios** (short window, analog via factors, R² vs zero): −5% to −25% for every portfolio.
- **By severity** (short window, analog): extreme −1.5% (27 events; mean |market move| 2.2 sd), major −20% (118; 1.2 sd), moderate −35% (45; 1.0 sd).
- **Uncertainty band:** the analogs' spread correlates **+0.25** with the realized error (short window; +0.24 over 5 sessions). The band is informative.

**Exploratory** (added after the direction result; labelled as such):
- **Analogs' |move| as a size forecast:** −13% vs the all-event mean |move| (no skill).
- **Severity-bucket mean |move|:** +0.4% on non-rates factors, +1.4% on the market factor (negligible).

## Implications and options

1. **Add direction to the schema, then run B4 as planned** (LLM, capped; about 10–20 calls, under $3):
   - Label each event's direction (escalation/relief, risk-off/risk-on) and its main channel.
   - Retrieve analogs within direction.
   - Add the §5.3 comparison step (the LLM says bigger/smaller and more/less surprising for each analog; code turns that into weights).
   - Re-score with the same leave-one-out and chance tests. This is the most likely fix for the coin-flip direction.
2. **Grow the library systematically.** Use GDELT event counts and tone around each date to add hundreds to thousands of events, including the many that barely moved markets, which corrects the lean toward big events. That gives the analog step enough neighbours.
3. **Scheduled releases: move to B5's surprise measures.** Kalshi-implied means since 2022, regime-aware fitting (post-2021 stock–bond sign), and the PLAN_B §4 head-to-head with the current engine.
4. **Product framing now:** present event damage as a historical range from analogs (with the band), not a point estimate. Say plainly that unscheduled-event direction isn't predicted yet.

Your idea of keeping the big events and their market responses (2008, COVID, SVB…) as reference memory for the LLM fits option 1: the comparison step shows the LLM the analogs' real responses, which is that memory, used under measurement.

## Follow-up, Oct 20: event direction

**What was added:**
- **Direction labels (Claude, 10 batched calls, about $0.59; SPEC §6.1 caps; dry run first):** for each of the 194 events,
  - (A) the direction of the event itself on six channels: escalation/de-escalation, growth shock −/+, inflation shock up/down, oil supply disruption/increase, policy tighter/looser, risk-off/risk-on
  - (B) separately, the LLM's expected sign for market, 2y and 10y yields, dollar, oil, gold and credit
- **Countries:** each event's region now maps to its country ETF (Brazil → EWZ, China → FXI, Spain → EWP, Israel → EIS, and so on; whole-word matching). Its beyond-factor move is stored with the targeted sectors and tickers. 81 of 194 events have a country ETF; for example, the Madrid bombings show EWP −2.6% beyond factors, and the 2005 yuan revaluation FXI +4.1%.

**Results** (short window, leave-one-out; chance = the same scoring with the labels shuffled across events, 300 times):

| Predictor | Non-rates R² vs zero | Chance 95th | p | Market direction right |
|---|---|---|---|---|
| Code-learned direction → factors (ridge on the six channels) | **+2.9%** | −2.5% | 0.003 | 57% (chance 95th 57%) |
| LLM factor signs × average move size | **+4.0%** | −9.2% | 0.003 | **63%** when it calls a sign (chance 95th 58%) |
| Analogs within the same direction | −12.9% | | | 56% |
| Analogs, no direction (the B3 result above) | −16.2% | −14.0% | 0.16 | 46% |

- **LLM sign hit rate by factor:** market 63%, 2y yield 67%, 10y 63%, dollar 60%, oil 69%, gold 58%, credit 59%.
- **Over 5 sessions** both effects fade: direction model −1.6% (still above the −2.5% chance 95th, p = 0.01); LLM signs +1.1%, market 61%.

**Reading:**
- **Direction was the missing piece.** Once the event's direction is known, code learns a small but real map to factor moves. It's the first unscheduled-event result that beats both "no move" and chance.
- **The LLM's own signs score higher (+4.0%, 63% on direction), but can't be taken at face value on historical events:** it may remember how markets reacted (hindsight, PLAN_B §5.5). The clean score needs events after its training cutoff or prospective ones.
- **Analog averaging still adds noise.** With 194 events, the five nearest neighbours are too few and too varied. Growing the library is the fix.

**Next: grow the library systematically.** Plan: Wikipedia's daily current-events pages 2004–2026 give, free and by code, thousands of dated items by section. Claude structures and direction-labels them in batches (capped runs), with code checks as before.


## Re-scored on the grown library (Oct 21)

**Library:** 3,487 unscheduled events.
- 194 hand-picked (B2)
- 3,293 chosen systematically from Wikipedia's Current events portal, Jan 2004 to about Dec 2023 (240 of 274 months; the last 34 months finish after the daily LLM limit resets)
- 143 Wikipedia picks dropped as near-duplicates of hand-picked events
- every event direction-labelled

Same leave-one-out tests as above; the direction model is now fitted on all events with exact leave-one-out (`scripts/planb_b3.py`, `scripts/planb_b3_subsets.py`).

**Overall** (short window):

| Predictor | All events | Hand-picked 194 | Wikipedia 3,293 |
|---|---|---|---|
| Code-learned direction → factors, non-rates R² | +0.4% (p = 0.005) | +3.3% | +0.1% |
| LLM factor signs, R² / market direction | −6.7% / 52% | +5.2% / 63% | −7.9% / 52% |
| Analogs (all others) | −20% (no better than random analogs) | | |
| Analogs within direction | −17% | | |

**Where the skill is** (short window; chance from 200 label shuffles):

| Subset | n | Mean abs market move (sd) | Direction model R² | Market direction | LLM signs R² / direction |
|---|---|---|---|---|---|
| **Extreme severity** | 45 | 2.65 | **+5.8%** (p = 0.005) | 60% | **+10.1%** / 68% |
| Major | 632 | 1.35 | +1.2% (p = 0.005) | 54% | −5.1% / 58% |
| Moderate | 1,620 | 0.92 | −0.4% (n.s.) | 51% | −12.6% / 48% |
| Minor | 1,190 | 0.92 | −0.4% (n.s.) | 54% | −3.4% / 54% |
| **Financial stress** | 420 | 1.47 | **+1.6%** (p = 0.005; 5-day +2.0%) | 57% | −3.3% / 53% |
| **Health / pandemic** | 150 | 1.12 | **+2.0%** (p = 0.005) | 56% | −4.5% / 57% |
| **Central bank (unscheduled)** | 171 | 1.28 | +0.8% (p = 0.015) | 61% | −2.9% / **67%** |
| Supply shock | 354 | 1.02 | +0.7% (p = 0.005) | 47% | −2.7% / 50% |
| Conflict / war | 390 | 0.96 | −0.1% | 50% | −21% / 48% |
| Elections / politics | 579 | 0.87 | −0.2% | 50% | −12% / 51% |
| Sanctions | 293 | 0.88 | −0.5% | 51% | −13% / 51% |
| Trade / tariffs | 184 | 0.98 | −0.3% | 57% | −3.3% / 60% |
| Disasters | 377 | 0.86 | −0.6% | 51% | −4.6% / 47% |
| Regulatory / court | 569 | 0.96 | −0.3% | 53% | −3.1% / 46% |

For the LLM signs, R² below zero means worse than predicting no move, even where it beats the shuffled-label chance rate. 5-session windows give the same pattern (extreme: direction model +6.2%, LLM +10.0% / 73%).

**Reading:**
1. **Measurable event effects are concentrated in big events.** For the 45 extreme events (2008 failures, COVID, SVB-type shocks), direction predicts about 6% of the broad factor moves (the LLM's signs about 10%, with the hindsight caveat). For major events it's about 1%. For moderate and minor events, nothing beyond noise: their 1–2 day market moves are the size of an ordinary day.
2. **By category, the learnable ones are financial stress, pandemics and surprise central-bank actions.** Wars, sanctions, tariffs, elections and disasters show no consistent broad-factor effect in the first days. Their effects, where they exist, are likely in specific sectors and countries (measured per event as beyond-factor moves, not yet tested), or slower than 5 sessions.
3. **The earlier +2.9% on 194 events was real but reflected a library tilted to big events.** The systematic library is the honest denominator.
4. **The LLM's factor signs look good on famous events and poor on ordinary ones.** That fits hindsight (it remembers COVID's market reaction) as much as skill. The clean test is post-cutoff events (B4).

**What it means for the product:** for "what could hit my portfolio?", **severity and category are the first filter.**
- **Extreme and major financial-stress, pandemic and central-bank events:** have a measurable, direction-dependent effect on the book through its factor exposures.
- **Most other headlines:** have no measurable broad-market effect. The honest answer for them is "historically, events like this haven't moved books like yours beyond normal daily noise", plus any targeted sector or country effect.
