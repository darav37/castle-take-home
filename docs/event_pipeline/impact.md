# PMRM Plan B: train on realized surprises, test on prediction-market resolutions

> **Design document**, written Oct 20 before the engine was built; it explains the reasoning. For how the code works today, read [ARCHITECTURE.md](../ARCHITECTURE.md), section 3.

**Status:** draft v0.2, 20 October 2026. **The main direction for Option 1 (DECISIONS #152)**, with P0c's naive daily-covariance test kept as the baseline it must beat. Build order: `BUILD_PLAN.md` checkpoints B0–B5. Nothing here changes `SPEC.md` until the trainer approves it.

## 1. The idea in one paragraph

The small-move model asks a short, noisy Kalshi history (2022 onward) to do two jobs: say how likely an event is, and say how much it would move a portfolio. Plan B splits them. **The prediction market supplies the probability and the timing. The impact comes from a factor model trained on realized surprises**, the actual outcome of an economic release minus what was expected, which exist for decades. The prediction market's own resolutions then become an **out-of-sample test**: does the surprise-trained model predict how stocks move when a Kalshi contract resolves?

## 2. The model

Same architecture as the PMRM: stocks load on a few shared factors. What changes is what drives the factors.

**Stage 1, stocks to factors (rich data).** Each stock's daily return loads on a small set of factors estimated on years of daily data:

```text
r[i,t] = L[i,:] f[t] + e[i,t]
```

The factors can be **anchored** to tradable proxies (2-year yield, 10-year yield, dollar, oil, equity index), **latent** (as in the PMRM, here fitted by reduced-rank regression on the surprises, so they are chosen to explain event responses), or a **hybrid**: an anchored core plus one or two latent factors fitted only to what the core leaves unexplained. Which one is used is decided by the test in §4, not by taste.

- *Latent* finds response directions no proxy captures (a duration or growth-vs-value dimension, a crowding or theme factor), and is the most predictive per parameter.
- *Anchored* is easier to explain ("long rate-sensitive names"), stays stable across refits, and lets events without surprise history be mapped by analog onto named factors.
- *Hedging is not a differentiator.* Hedges are built on the impact vector B = L S (how the book moves per unit of each surprise), which does not depend on how the factors are rotated or named (§3a).

**Stage 2, surprises to factors (the event link).** On an announcement day, the factors move in proportion to the surprise:

```text
f[t] = S s[t] + u[t]        s[t] = vector of that day's surprises (CPI, payrolls, Fed, GDP, ...)
r[i,t] = L[i,:] S s[t] + noise    =>    impact matrix B = L S  (stocks x event types, low rank)
```

Fitting B directly with a rank-k constraint (reduced-rank regression) is the same idea in one step: a few factors carry every event type's effect to every stock. Low rank is what lets thin events borrow strength from rich ones, as in the PMRM.

**Training data: realized surprises, decades long.**

| Event type | Surprise measure | History | Source (free first) |
|---|---|---|---|
| Fed decisions | Change in the 2-year Treasury yield on FOMC day (a standard market-based surprise); fed funds futures surprise where available | 1990s onward | FRED (DGS2); futures via Massive if history is available |
| CPI | Actual minus expectation. Expectation from the Cleveland Fed inflation nowcast (free); survey consensus if a source is available | Nowcast from 2013; survey longer | FRED / Cleveland Fed; ALFRED for first-release vintages |
| Payrolls | Actual minus expectation; or, without consensus data, the release-day 2-year yield change as the market's read of the surprise | 1990s onward | FRED (PAYEMS, DGS2) |
| GDP | Advance estimate minus nowcast (Atlanta Fed GDPNow, NY Fed Staff Nowcast) | 2011 onward | Atlanta Fed, NY Fed |

**Market-based surprises solve the consensus-data problem.** Survey consensus series are usually paid (Bloomberg). The release-day move in the 2-year yield is a widely used proxy for "how surprising was this release", it is free, and it goes back decades. The model can be trained entirely on free data, with survey consensus as an upgrade.

### Where prediction markets enter

The impact model (§2, §5) is trained and tested on realized events and broad asset returns, with no prediction markets. Markets enter in two places:

1. **Anticipation, in training.** A realized response depends on how surprising the event was. Where a prediction market existed at the time (Kalshi since 2022; older venues such as PredictIt, Betfair and the Iowa Electronic Markets), the surprise is exactly outcome − p. Elsewhere it comes from consensus or market-based measures (macro), or is unknown and the response is labelled a lower bound.
2. **The live model.** With B the impact of the event fully happening vs not (learned from the library and scaled to a full surprise):

```text
P&L if YES ≈ B (1 - p)        P&L if NO ≈ -B p          (resolution)
P&L ≈ B Δp                                               (a probability move; needs Phase R's H1)
several events: responses combine in factor space; joint outcomes from market probabilities and their dependence
```

The probability-move line connects Plan B to the small-move question: B·Δp can be scored on the same held-out days as P0c's naive covariance baseline, with no daily covariance estimated at all.

## 3. The bridge: from a surprise model to a prediction-market resolution

A Kalshi contract is a yes/no question about the same variable: "CPI ≥ 0.3%". Two ways to connect it to the surprise model:

**(a) Binary surprise.** The contract's resolution surprise is outcome − p_before. The impact of YES vs NO comes from the surprise model plus the distribution of the actual number:

```text
move if YES = E[ B s | actual >= k ]  -  E[ B s ]      move if NO = E[ B s | actual < k ]  -  E[ B s ]
```

The distribution of the actual number comes from the expectation and its historical surprise volatility.

**(b) Prediction-market-implied expectation.** A Kalshi strike ladder ("CPI ≥ 0.1, 0.2, 0.3, 0.4 ...") gives the market's whole distribution for the release, and so its mean. Then the surprise is actual − PM-implied mean, directly comparable to actual − consensus.

Route (b) gives a question that is interesting in its own right: **which expectation better measures what was priced in, prediction markets or economists?** Whichever surprise better explains announcement-day stock moves is the better measure of what the market expected. That is a result Castle (and Kalshi) would care about.

## 3a. Hedging from the impact vector

Include the hedge instruments (TLT and Treasury futures, sector ETFs, SPY options as repriced positions, and the Kalshi contract itself) as rows of the model, so each has its own row of B. The hedge for a book's exposure to an event is the instrument weights that cancel the book's row of B for that event, subject to cost. This is the same calculation for latent, anchored or hybrid factors, and it is stable when latent factors rotate between refits. The Kalshi contract pays exactly on the event, so it has no basis risk to the binary outcome and is always a candidate. Basis risk for every other instrument is measured, not assumed: the hedge's held-out offset of the book's announcement-day P&L (§4).

## 4. The test

**Train** Stage 1 and Stage 2 on realized surprises up to 2021 (and on rolling windows thereafter, never using a release to predict itself).

**Test** on every Kalshi-era release (2022 onward) that has a resolving contract or strike ladder: about 38 FOMC meetings, 58 CPI prints, 58 jobs reports and 55 GDP releases. For each release, predict the portfolio's announcement-day move four ways and score against what happened:

| Model | Prediction from |
|---|---|
| 0 | No event information (unconditional) |
| 1 | Surprise model, consensus or nowcast surprise |
| 2 | Surprise model, PM-implied surprise (route b) |
| 3 | Surprise model, binary contract resolution (route a) |
| 4 | Current engine: B_post × (outcome − p) from daily prediction-market history |

Metrics as in the PMRM harness: held-out R² of the announcement-day portfolio move, log score, and calibrated tests against measured chance rates.

**What the comparisons answer:**
- 1 vs 0: does the surprise-trained model predict announcement moves at all? (It should: this is well established in the macro-finance literature.)
- 2 vs 1: are prediction markets a better "priced-in" baseline than consensus?
- 3 vs 2: how much is lost by seeing only the yes/no outcome of one strike instead of the whole number?
- 3 vs 4: does training on decades of realized surprises beat training on Kalshi's own short history? This is the head-to-head that decides the engine.

**Factor choice, the same way:** models 1 to 3 are each run with anchored, latent and hybrid factors on the same held-out releases. If latent wins by a wide margin, the proxy list is missing something, and the latent loadings show what. For each, report the **held-out hedge offset**: the share of the book's announcement-day P&L that the §3a hedge, built only from training data, cancels on test releases.

## 5. The event library: pricing new and bespoke events

Sections 2–4 cover scheduled macro releases, which have long surprise histories. Most events a user asks about do not: a tariff on a new country, a strike at a port, a Taiwan Strait incident, a new pandemic. The event library gives every such event a data-backed impact estimate and a hedge.

### 5.1 What it is

A database of a few thousand dated historical events across every category that moves markets: trade actions and tariffs, sanctions, wars and conflicts, elections, regulatory and court rulings, natural disasters, pandemics, strikes and supply shocks, central-bank surprises outside scheduled meetings, plus the scheduled releases of §2. For each event:

- **Structured description** (fixed schema): type, actors, region, sectors and commodities targeted, severity, whether it was scheduled, how anticipated it was (prediction-market price at the time where one exists), and the source documents.
- **Timing:** the first credible public report, not when the underlying thing happened. Leaks and slow-building stories are dated by first report, with the window starting there.
- **Realized response:** the cross-asset move over a short window from that first report (one and five sessions), expressed in the broad factor space of §5.2, plus the raw moves of a fixed panel of assets.

**Sources, free first:** GDELT (systematic global event coverage, which limits selection bias), the Federal Register, USTR trade actions, OFAC sanctions, EM-DAT disasters, ACLED conflicts, Wikipedia's dated event pages as a cross-check, and Massive news for recent events. Commercial event-tagged news data exists as an upgrade.

### 5.2 A broader factor space

A handful of macro factors cannot express most of these events. The library uses a broad cross-asset universe for Stage 1:

- about 50–60 industry groups
- commodities separately: crude, natural gas, grains, metals
- major currencies and country equity indices
- rates

The factors are fitted on that universe (anchored, latent or hybrid as decided by §4), and each event's response is stored as a vector in that factor space. Factor space is far less noisy than per-stock responses, and it is what makes two events comparable. Any portfolio's exposure to an event then follows from its loadings, and its hedge from §3a.

### 5.3 Predicting the impact of a new event

1. **Structure it.** The LLM maps the user's description into the library schema, citing sources. Ambiguous fields are asked about, not guessed.
2. **Retrieve analogs.** Code finds the nearest historical events by schema fields and by an embedding of the description.
3. **Compare.** The LLM is shown the analogs *with their realized responses* and, for each, chooses from fixed buckets: is the new event bigger or smaller, more or less surprising, broader or narrower? This is comparative judgment over real data, not recall from memory.
4. **Compute.** Code turns the buckets into weights and scales on the analogs' realized factor responses. The predicted impact vector is the weighted blend. The **uncertainty band comes from how much the analogs disagree**, so a novel event with scattered analogs gets wide bands automatically.
5. **Fallback.** Where analogs are thin or disagree wildly, the LLM proposes a direction on named factors and a size bucket (the current engine's tier 3). It is blended with the analog estimate in proportion to each one's **measured** skill (§5.5).

Probability for an unpriced event comes from a Kalshi market if one exists, else a base rate from the library (how often similar events occurred) or the user; never from the LLM. The hedge is §3a on the predicted impact vector.

**Order of trust:** a structural mechanism where the event is written on a traded underlying; then close analogs with realized responses; then LLM decomposition weighted by its measured skill.

### 5.4 Pitfalls designed around

1. **Confounding:** an event window also contains everything else that happened. Factor-space responses, short windows and pooling across analogs reduce it; macro release days inside a window are flagged.
2. **Anticipation:** a partly expected event moves markets less. Where no prediction market existed at the time, the realized response is a lower bound on full impact, and is labelled as such.
3. **Selection:** memorable events are the ones that moved markets, so a hand-picked library overstates typical impact. Systematic sources (GDELT) and including events that did not move markets correct for it.
4. **LLM hindsight:** the LLM may remember what happened after a historical event, which flatters any historical test of its judgment (§5.5).

### 5.5 How it is tested

- **Leave one event out:** hide an event, run §5.3 on its description alone, and score the predicted factor response against what happened, against a no-information baseline and against the LLM-only fallback.
- **Hindsight control:** the LLM sees only the structured description, never the outcome, and the headline skill estimate uses events after the model's training cutoff, plus prospective scoring of new events as they occur. Historical scores are reported but marked as possibly flattered.
- **Skill weights** for the blend in §5.3 step 5 come only from these held-out scores.

### 5.6 Scope for the take-home

A prototype of about 100–200 hand-curated events across categories, with the broad factor space, the retrieval-and-compare step, and one leave-one-out chart (analog predictions vs LLM-only vs realized). That demonstrates the idea; the full GDELT-scale library is a product build.

## 6. Scope and what stays elsewhere

- **Covered:** scheduled macro releases with a measurable number (Fed, CPI, payrolls, GDP, PCE). This is most of the volume in Kalshi's economics categories.
- **Contracts on traded underlyings** (oil above $X): priced structurally as barrier options on the underlying, with stocks linked through their oil beta (`PHASE_R.md` §8, question 3).
- **Elections and policy or geopolitical shocks** (House control, tariffs, Iran): no long surprise history. Priced through the event library (§5): analogs with realized responses, with LLM decomposition as a skill-weighted fallback.
- **Small daily moves:** if Plan B's B is good and the mixture model holds (`PHASE_R.md` H1), the daily P&L for a small odds move follows as B × Δp, with no daily-covariance estimation at all. That test comes free once B exists.

## 7. The LLM's role

Unchanged in spirit: it labels and selects, code computes. Here it maps each Kalshi contract to its underlying release and variable (which series, which strike, which date), flags releases distorted by special circumstances (shutdown-delayed data, revisions), and proposes analogs for event types without surprise history. Every number comes from code.

## 8. Data gaps to check first (a P0-style audit, no LLM calls)

1. FRED and ALFRED access from the trainer's machine (DGS2, CPIAUCSL first releases, PAYEMS first releases).
2. Cleveland Fed nowcast and GDPNow archives: downloadable, and how far back.
3. Fed funds or SOFR futures history through Massive.
4. Kalshi strike ladders per release since 2022: how many have enough strikes to give a usable implied mean.
5. Survey consensus: any free source with history (otherwise market-based surprises only).
6. Event library sources: GDELT access and volume, Federal Register, USTR, OFAC, EM-DAT and ACLED downloads; how far back each goes; and daily prices for the broad factor universe (industry ETFs, commodity futures, currencies, country indices).

## 9. Open questions for the trainer

1. ~~Anchored or latent factors?~~ Decided Oct 20: test anchored, latent and hybrid head to head (§4); hedging is built on the impact vector, so it does not favour either.
2. Train on market-based surprises only (free, long), or pay for consensus data?
3. Should Plan B start now in parallel with P0c, or only after P0c reports?
4. Event library: start with the hand-curated prototype (100–200 events) for the take-home, and which categories to prioritize?
