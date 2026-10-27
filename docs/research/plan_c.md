# Plan C: a learned event → market-move forecaster on embeddings

**Status:** proposal (Oct 22, design review; DECISIONS #181). Nothing built yet. Plan B stays live in the chat until Plan C beats it on the predeclared test (§7).

**One line:** represent everything known about an event at the moment it happens (the news text, what markets were doing beforehand, the regime each factor is in) as embeddings. Let a model learn the mapping from that to the distribution of factor and ETF moves, and let the LLM build the input and explain the output, never adjust the numbers.

---

## 1. Why Plan C (what A and B taught us)

| Lesson | Source | What Plan C does about it |
|---|---|---|
| Daily odds wiggles carry almost no portfolio signal | P0c, #153 | Not used. Prediction markets only give the probability and a "how surprising" input where one exists |
| The factor map is strong (13 named factors, 62.7% held out) | #163 | **Kept** as the map from factor moves to any book |
| Event effects are real but small, and concentrated in big events and a few factors | #162, #164, #171 | The output is built to say "noise" by default (§4) |
| Hand labels (category, severity, direction) are what made events learnable at all | #160 | Replaced or extended by learned text representations that carry far more than five labels |
| The LLM is good at structuring an event, bad at sizing it | #170, #180 | The LLM writes the input and explains the output. It does not touch the numbers between the two |
| Over two sessions a typical event is about 3% of the move; in the first 2 hours of a big event it is 80–87% | #176 | Timed events are scored on the 2-hour window as well as the close |
| The product promises ranges and a bad case, but we scored point forecasts | #180 review | Outputs are distributions, scored on calibration (CRPS, range coverage) |

: "I want to harness what makes LLMs so good at learning into this sort of forecaster of event → market moves mappings. I just want it to learn the mappings, and yes most of them will be 'the answer is noise, outside of maybe 1 or 2 factors'."

---

## 2. The event, as the model sees it

Each event is a set of **tokens** of different kinds, all as of the first-report time (nothing after it):

| Token type | What it holds | How it becomes a vector |
|---|---|---|
| **Text** | The library description; the first news reports (headlines and lead paragraphs within the first hours); Wikipedia's Current events line | A frozen pretrained text-embedding model (open-weight, run locally), one vector per chunk |
| **Market state** | Each of the 13 factors' moves over the previous 1, 5 and 20 sessions; realized vol; the pre-event intraday move when the time is known | A small learned encoder (an MLP over the numbers) → one token per factor |
| **Regime** | Per factor: vol level, trend, its rolling correlation with the market; global: stock–bond correlation (#165), VIX level and term structure, rate level, credit spread level | Same encoder style → regime tokens |
| **Calendar and context** | Time of day and session (pre-market, intraday, after close, weekend); nearby scheduled releases (#157's confound flag); days to the next FOMC | Small embedding tables |
| **Structure from the LLM** | The Plan B schema: category, severity, region, direction, targeted sectors, countries and stocks | Embedding tables, plus the targets' tickers as tokens |
| **Prediction market (optional)** | Kalshi or Polymarket odds just before, when a market really is the event; the release surprise from strike ladders (#172) | Numeric token |

### 2a. Feature catalogue (DECISIONS #188)

All features are as of the first report, except group 2, which is explicitly "after the first report" and feeds the first-hours and close horizons. Every source is free code and data; LLM spend stays near zero.

| # | Group | Features | Source | Build |
|---|---|---|---|---|
| 1 | **Surprise / anticipation** | prediction-market odds before and the jump at the first report; options-implied move before (SPY, oil, gold, Treasuries); news-volume build-up in the prior days; fear gauges the day before (VIX, VIX term structure, OVX, GVZ) | Kalshi / Polymarket; Massive options (history depth to check); Massive news counts (2022+), GDELT counts (2015+); FRED | C1c |
| 2 | **First reaction (after the first report)** | first 5–15 minutes' move and volume surge; for off-hours events, overnight index and oil futures, Asian and European markets, FX, crypto, the Kalshi odds move | Massive minute bars and futures; FX | C1d |
| 3 | **Positioning** | CFTC net positioning (oil, gold, Treasuries, currencies) and its change; put/call; short interest; ETF flows where history allows | CFTC (free weekly); CBOE; Massive | C1c |
| 4 | **Size and reach** | quantities in the text ($, %, bp, barrels/day); the affected country's share of world GDP, oil output, chips and grain exports; reversible vs permanent | regex on text; EIA / World Bank | C1a (quantities), later (shares) |
| 4b | **The LLM's directional read** (#189) | per factor and per target ETF: the direction (and a coarse size) the LLM expects, from the event text alone, without seeing outcomes; an input feature only, gated, never a number in the output | the schema LLM call (budget: about $0.5 per 1,000 events with batching) or a local open model | C1c |
| 5 | **Story position** | first shock / escalation / retaliation / de-escalation; days since the last similar event; count of similar events in the past 30 days; novelty (text similarity to recent events); days to the next FOMC | library labels; code; embeddings (C2) | C1a (counts, days), C2 (novelty) |
| 6 | **Market state before** | per factor: 1-, 5- and 20-day moves, 20-day realized vol, drawdown from the 60-day high; curve shapes (oil futures, VIX term structure, yield curve), credit spread; regime labels (#165) | research anchors; FRED | C1a |
| 7 | **Instrument history** | per target ETF: its past response to events of the same category and region (a learned prior), its region exposure | the C0 dataset (train split only) | C1a |
| 8 | **Calendar and liquidity** | day of week, time known / time bucket, holiday-adjacent, options-expiry week, month-end, earnings season, a macro release in the window | calendars; library flags | C1a |

### 2b. Cleaner targets and a real-event gate (DECISIONS #188)

- **Split factors that mix two forces,** since events push the parts in opposite directions (tested on held-out events; kept only if direction and CRPS improve):
  - 10-year rates → real yield (FRED DFII10) + inflation expectations (T10YIE)
  - oil → demand (oil's co-movement with global stocks and copper) + supply (the rest)
  - dollar → haven (yen, franc) + broad
  - gold → its real-rate part + the rest
- **The gate is a two-part model per target, not a hard threshold:** move = normal noise + (with probability g) an event effect. Fitted against thousands of random ordinary days as controls, so "bigger than an ordinary day" is learned, not assumed. It's trained first on the 2-hour windows of timed events, where 80–87% of the move is the event.
- **Gate quality is scored:** when it opens, how often the target really moves beyond noise, and how many real moves it misses.

**Why tokens rather than one flat feature vector:** the model can attend across them. "Oil supply shock" text plus "oil already up 8% in 5 days" plus "high-vol regime" should map differently from the same text in a calm market. That interaction is exactly what attention learns and what ridge on labels can't.

**This is the "harness what makes LLMs good" part.** The text encoder brings everything a language model knows about what an event *is*: which events are alike, which sectors a sanction touches. The fusion model learns what those meanings did to markets, conditioned on the state they landed in.

---

## 3. The model

**Default (C2–C4): frozen encoders + a small fusion transformer.**

```
text chunks ──► frozen text embedder ──┐
market state ─► small MLP encoder ─────┤
regime ───────► small MLP encoder ─────┼──► fusion transformer (2–4 layers, few heads) ──► output heads (§4)
calendar ─────► embedding tables ──────┤
LLM schema ───► embedding tables ──────┘
```

- **Small on purpose.** We have thousands of labelled events, not millions. Parameters stay in the low hundreds of thousands, with heavy dropout and weight decay, and early stopping on a time-ordered validation set.
- **Frozen text embedder.** Fine-tuning it on ~4k events would overfit. Freezing it means we borrow its knowledge without spending our data on it.

**Stretch (C5): fine-tune an open LLM end to end.**
- Serialize the market state and regime as text alongside the news.
- Attach a quantile head and train with LoRA (low-rank adapters, so only a small set of new weights is trained).
- Only worth trying after pretraining (§5) shows the data can support it. Needs a GPU (Colab is enough for a small model).

**Not this:** asking a chat LLM for the numbers. That is the raw-LLM row in every test so far, and it lost.

---

## 4. What it predicts

**Targets (more targets, not more factors):**

1. **The 13 named factors** (B1), for the book mapping as today.
2. **The residual move of each of ~100 panel ETFs beyond the factors** (sectors, industries, 26 countries, commodities). This is where targeted effects land (#164: big events hit their targets 1.2–1.6×). It also gives the model ~100 labels per event instead of 8, the cheapest way to get more learning signal out of a small event set.
3. **Optional, for single-name books:** the targeted stocks' residuals, when the event names them.

**Two horizons:** the first 2 hours (timed events) and the close of the second session (all events). T+5 later.

**The shape of each output: "noise unless it isn't".** Per target, per horizon, the model outputs:
- **a gate:** the probability that the event moves this target beyond normal noise
- **if it does:** a distribution for the move (quantiles at 5, 25, 50, 75, 95%)
- **if it doesn't:** the target's normal noise over the window, known from history, not learned

Most events will come out "gate low on everything, maybe high on one or two". That is the right answer and the model is built to say it cheaply. It is a learned version of the finding in #162.

**Loss:** CRPS on the full mixture (it scores the whole distribution, so overconfident ranges are punished as much as wrong middles), summed across targets, with each target standardized by its own noise so oil doesn't drown out credit.

**To any book (code, unchanged in spirit from Plan B):**

```
book move  =  L_book · f  +  Σ w_i · e_i        f = factor draws, e_i = residual draws for the held ETFs
```

Code samples from the predicted distributions, keeping their correlation through the fusion model's joint output (or a factor-level copula fitted on residuals), and reports the middle, the range and the bad case.

---

## 5. Data

| Set | Size | Labels | Use |
|---|---|---|---|
| **Curated library** (B2, Wikipedia growth) | 3,876 events | Full window moves, LLM schema, regime, timing for 232 | Fine-tuning and the main test |
| **Scheduled releases** 1976–2026 | ~1,300 since 2003, more in the long history | Surprise from yields / Kalshi ladders | Fine-tuning; checks that the model learns rate-surprise channels |
| **Headline-days (weak labels)** | ~100k headline-days, 2015+ | That day's (or that hour's) factor and ETF moves | **Pretraining.** Most headlines are noise, so the labels are noisy, but there are many of them. That's how language models learn too |
| **Prospective log** (#173) | Grows daily | Frozen predictions made before outcomes | The cleanest test |

**Text sources:**
- GDELT (open; headlines, URLs, themes and tone from 2015)
- Wikipedia Current events (already parsed)
- The library's cited sources

Full article text has licensing limits. Headlines plus lead paragraphs and the Wikipedia line are the default. Whether full text adds anything is itself a test (C3).

**Compute:**
- Embeddings with a local open model cost $0 of the LLM budget.
- The fusion model trains on CPU or Colab.
- LLM spend is only for schema labelling of new events, under the existing caps (SPEC §6.1; $57.47 of $70 spent as of #180).

---

## 6. Leakage and hindsight rules (non-negotiable)

1. **As-of everything.**
   - Text, market state and regime use only what existed at the first-report time.
   - News published after the market moved is excluded. The timing work in #177 gives first-report times for 2022+.
   - When the time is unknown, inputs stop at the prior close.
2. **The text embedder can remember history.** A model trained on the web knows "Lehman" meant a crash. So:
   - the headline score is on events after the embedder's training cutoff
   - we pick an embedder with a documented cutoff
   - we report results on events before the cutoff separately, as possibly flattered (same rule as #170)
3. **Anonymized check.** Re-run the test with names, places and dates masked (#176's blinding). A gap between masked and unmasked results measures how much the score relies on the model recognizing the event.
4. **Time-ordered splits only.** Train before, validate after, test after that. No random splits.
5. **The LLM schema labels** are made without seeing outcomes (as in B2).

---

## 7. The test (predeclared)

**The ladder:** each rung must beat the one below to matter.

| Rung | Model | What it isolates |
|---|---|---|
| 0 | Predict zero (noise only) | The floor |
| 1 | Ridge on labels: today's engine (B3) | The current product |
| 2 | Nearest past events by text embedding (kNN), averaging their realized moves | Does learned similarity beat hand labels? |
| 3 | Ridge on embeddings + market state + regime | Is it the representation or the model? |
| 4 | Fusion transformer, gated distributional output (§3–4) | Does attention over heterogeneous tokens add anything? |
| 5 | Rung 4 pretrained on headline-days, then fine-tuned | Does weak-label scale make depth pay? |
| (5b) | LoRA-tuned open LLM with a quantile head | Only if 5 beats 4 |

**Test sets:**
- 2026 events (post-cutoff for the LLM; check against the embedder's cutoff too)
- big-event subset reported separately
- the prospective log as it grows
- the 2016–23 sample, marked as hindsight-exposed

**Metrics:**
1. **CRPS** relative to rung 0 (the headline)
2. **Range coverage:** does the 90% range hold the real move about 90% of the time, per horizon?
3. **Gate quality:** when the model says "this factor moves", how often does it, and how often does it miss real moves?
4. **Per-factor standardized R²** (comparable with #170 and #180)
5. **Book-level** coverage and CRPS on the 11 books of #178

**Uncertainty:** paired bootstrap over events for each rung against the rung below; chance from shuffling predictions across events.

**The gate:**
- Plan C replaces Plan B's estimate in the chat only if the best rung beats rung 1 on CRPS with ≥ 90% bootstrap support on 2026 events **and** holds on the prospective log.
- A deeper rung ships only if it beats the shallower one the same way. Otherwise the simpler model ships.

**Expectation (written down so we can be wrong in public):**
- Rung 3 gets most of the gain.
- The distributional output and the 2-hour window matter more than depth.
- Rung 4 over rung 3 is small unless rung 5's pretraining works.

### 7a. Frozen at C0 (Oct 22, DECISIONS #187; `results/planc_c0.md`)

| Item | Frozen as |
|---|---|
| Splits | train < 2023-01-01 (2,995 events); validation 2023–2025 (484); test 2026+ (92) plus the prospective log |
| Prediction | 19 quantiles (5%…95%) per event and target |
| Targets | 13 factors; the 96 research ETFs' moves beyond the factors; the 12 portfolios of #185 |
| Horizon | the library's short window (to the second-session close); the 2-hour horizon is added in C1 for timed events |
| Main score | CRPS from the quantiles, standardized by the target's normal noise over the window, averaged over targets (`src/worsey/planc/harness.py`) |
| Reference rung | the best of rungs 0, 0b and 1 on validation = **rung 0, zero + normal noise** (it beat today's engine and the noise + premium rung on validation and test, for every target kind) |
| Pass bar | beat the reference on test CRPS with ≥ 90% paired bootstrap support over events and hold on the prospective log; a deeper rung must beat a shallower one the same way |
| At-the-open rungs (#193) | features known by 09:45 on the window's first session are a different information set: their bar is "ordinary day given the open" (slope × opening gap + leftover noise, fitted on ordinary days), not noise |
| Control days (#193) | `worsey/planc/controls.py`: not a library event's first session, not within 3 sessions of a major or extreme one (773 before 2023) |

---

## 8. Where the LLM sits

```
news / user question ─► LLM: identify the event, pull first reports, fill the schema, pick targets
                     ─► code: build tokens as of the event time ─► model ─► distributions
                     ─► code: book mapping, ranges, bad case, hedges
                     ─► LLM: explain which tokens drove it (gate highs, nearest past events), in words
```

- **No step where the LLM changes a number.** That step is what lost in #180.
- **The LLM can flag that the event is out of distribution** (nothing like it in the library). Code then widens the range by a rule, not by the LLM's say-so.

---

## 9. Do we need more factors?

**For the map from factors to books: no.** 13 named factors explain 62.7% held out (#163), and adding factors gives diminishing returns.

**For what events do: the answer is more *targets*, not more factors.**
- Event effects are sparse and often land off the factor axes: a sanction on one country, a ruling on one industry. A factor model built on everyday variance will smear those effects across factors or lose them in the residual.
- Predicting the ~100 ETF residuals (§4) catches them without redefining the factor space.

**One experiment worth running (C4):** let the model learn its own low-rank **event-response basis** over the ~100 ETF moves. These are the few directions events actually push markets along (maybe "risk-off", "oil supply", "hawkish rates", "China"). Compare them with the 13 named factors.
- **If they line up:** good, the named factors are also the event factors.
- **If not:** the gap is a finding in itself. Events move markets along directions that everyday variance doesn't emphasize, and the product should report exposure to those directions too.

---

## 10. Build order

| Step | What | Cost |
|---|---|---|
| **C0** ✅ | Freeze the test (§7a), splits and metrics; harness; score rungs 0, 0b, 1 (done Oct 22, #187: rung 0 is the reference) | $0 |
| **C1** | (C1a ✅ #190, C1b ✅ #191–192, C1d ✅ #193, C1c ✅ #194: gate + LLM sign leads, 98% support on 2026) Assemble the features of §2a in stages: C1a (market state, regime, calendar, quantities, story counts, instrument history), C1b (factor splits and the gate, §2b), C1c (surprise and positioning), C1d (first reaction and off-hours markets); leakage checks (§6) | $0 (no LLM) |
| **C2** ✅ | Rungs 2–3: embedding kNN and ridge on embeddings + state (#197: both lose to noise; text in the gate adds nothing; the gate + LLM direction read leads) | $0 |
| **Shadow** ▶ | The leading rung (gate + LLM sign) frozen and scored daily on the prospective log (#201) | $0 |
| **C3** | Text-source ablation: description only vs headlines vs lead paragraphs | $0 |
| **C4** | Rung 4: fusion transformer, gated distributional heads, ETF residual targets; learned event-response basis vs the 13 factors | $0 (CPU or Colab) |
| **C5** | Headline-day pretraining (GDELT 2015+), then rung 5; rung 5b only if 5 beats 4 | $0 data; GPU time |
| **C6** | If the gate passes: swap the estimate behind `event_evidence` / `price_scenario`; the prospective log scores both in parallel for two weeks first | Existing caps |

---

## 11. Open questions

1. **Text embedder:** a local open-weight model (free, documented cutoff; recommended) or an API embedding model (stronger, costs budget, cutoff less clear)?
2. **News text:** headlines + lead paragraphs only (recommended), or try for full articles where licensing allows?
3. **Horizons:** 2 hours and second-session close to start, T+5 later. Agree?
4. **Gate threshold:** is "beats today's engine on CRPS with ≥ 90% bootstrap support, and holds on the prospective log" the right bar to swap the chat over?
5. **Priority vs the open Plan B items** (#180 fix, demo re-recording): Plan C research before or after?
