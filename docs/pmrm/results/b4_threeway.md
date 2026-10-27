# Plan B B4: model alone vs raw LLM vs LLM assisted by the model

Run 2026-10-21. Script: `scripts/planb_b4_threeway.py`. Raw numbers, including every event's predictions: `b4_threeway.json`. DECISIONS #167, #170.

**Cost:** 60 Claude calls, about $1.95 (30 raw, 30 assisted, batches of 10 events), under the per-run and project caps.

## In plain words

We asked three predictors the same question about the same news events: "how will these 8 market factors move in the next one to two sessions?" The factors are the stock market, the 10-year yield, the dollar, oil, gold, credit, defensives vs energy producers, and Asia/EM vs the US.

1. **The model alone:** our factor, direction and category model, trained only on events before each one.
2. **The raw LLM:** reads the news item and answers from its own knowledge.
3. **The assisted LLM:** the same LLM, also shown the model's estimate and how uncertain it is, the market regime at the time, and the five most similar earlier events with what markets actually did after them.

**The headline test** uses the 92 events from 2026, after the LLM's training cutoff, so it can't remember how markets reacted. **The LLM with the model's context is the best of the three.** It explains +2.5% of the factor moves (beats chance). The raw LLM is **worse than predicting no move at all** (−5.7%), and the model alone is about zero.
- **Context vs no context:** on the same events, the context adds +8.2 points over the raw LLM (90% range +4.3 to +12.8), a clear, robust gain.
- **Assisted vs model:** the assisted LLM also beats the model alone by +2.7 points, probably real (90% of resamples) but not certain with 92 events.

**On older events (2016–23)** the raw LLM looks much better (+5.3%), consistent with it remembering what happened. The assisted LLM still beats it (+8.8%, difference +3.5 points, 90% range +0.7 to +6.4).

**What it means:**
- **Raw LLM guesses about market impact are unreliable on genuinely new events.** That's the approach Worsey's live v1 engine leans on (its LLM priors), so this is direct evidence against it.
- **The data and the LLM are complementary.** The model supplies calibrated, regime-aware evidence (analog events with real outcomes, an estimate, a band). The LLM judges how the new event differs. Together they beat either alone.
- **Where each helps:**
  - Without context, the LLM wildly over-calls credit moves; the context fixes most of that.
  - Both LLMs read oil well.
  - The assisted LLM is best on defensives vs producers and Asia/EM.
  - The model is best on the overall market.

## Setup

- **Events:** the unscheduled library (3,876 events, complete through October 2026).
- **Test sets:**
  - **post-cutoff:** every event from 2026-01-01 (92), the headline (assumed LLM training cutoff about end-2025)
  - **historical:** a fixed random sample of 200 events from 2016–2023 (seed 1729), marked as possibly flattered by hindsight
- **Responses:** each event's short-window move of 8 named factors (B1 13-factor space): market, 10y yield, dollar, oil, gold, credit, defensives vs producers, Asia/EM.
- **Predictors:**
  - **Model:** ridge on direction labels + category, refitted monthly on events strictly before the test month.
  - **Raw LLM:** description, date, category and region only.
  - **Assisted LLM:** the same plus the model's estimate per factor (as a bucket), the analog band (narrow/wide), the regime labels at the event date, and the 5 most similar strictly-earlier events with their realized moves (as buckets).
- **LLM answers:** fixed buckets (strong down, down, flat, up, strong up), cut at −1.5, −0.5, +0.5, +1.5 sd of pre-2016 event responses. Code maps each bucket to the mean pre-2016 response in it.
- **Score:** R² vs zero pooled over the 8 factors; chance from shuffling predictions across events (200 times); paired bootstrap over events for differences between predictors (2,000 resamples).

## Results

| Test set | Events | Model | Raw LLM | **Assisted LLM** | Chance 95th (model / raw / assisted) |
|---|---|---|---|---|---|
| **Post-cutoff (2026)** | 92 | −0.2% | −5.7% | **+2.5%** | +0.3% / −8.3% / −2.1% |
| Historical 2016–23 (hindsight risk) | 200 | −0.1% | +5.3% | **+8.8%** | −0.1% / −4.6% / −3.4% |

**Paired differences** (R² points, same events; 90% bootstrap range; share of resamples above zero):

| | Post-cutoff | Historical |
|---|---|---|
| **Assisted − raw** | **+8.2** (+4.3 to +12.8; 100%) | **+3.5** (+0.7 to +6.4; 98%) |
| Assisted − model | +2.7 (−0.6 to +5.8; 90%) | +8.9 (+2.8 to +15.4; 100%) |
| Raw − model | **−5.5** (−11.2 to −0.5; 3%) | +5.3 (−0.6 to +11.6; 92%) |

**By factor, post-cutoff** (R² vs zero; assisted / raw / model):

| Factor | Assisted | Raw | Model |
|---|---|---|---|
| Market | −8% | −39% | **+6%** |
| 10y yield | −6% | −24% | −5% |
| Dollar | −4% | −13% | **+4%** |
| Oil | **+11%** | **+12%** | +4% |
| Gold | −1% | −5% | −2% |
| Credit | −48% | **−384%** | −54% |
| Defensives vs producers | **+10%** | +6% | 0% |
| Asia/EM | **+7%** | −1% | +1% |

Direction hit rates are about 51–54% for all three: the gains come from sizing (knowing when to call "flat" and when to call a big move), not from calling up or down more often.

## Caveats

1. **92 events is a modest sample.** The assisted-vs-raw gain is robust; assisted-vs-model is suggestive.
2. **The training cutoff is an assumption** (about end-2025). If the LLM saw early-2026 news, the post-cutoff test is slightly flattered for both LLMs equally.
3. **Credit is badly predicted by everyone.** Credit moves on news days are small and noisy, so any non-flat call is costly. A "default to flat" rule for credit is an easy improvement.
4. **The assisted prompt's evidence comes from the same model it's compared with.** That's the point (LLM plus model vs either alone), but the two aren't independent predictors.

## What it means for the product

The answer to "what if there's another Iran flare-up?" should be produced by the assisted LLM:
- **Code first:** retrieves analogs, measures their real moves, computes the model's regime-aware estimate and band.
- **Then the LLM:** judges the new event against that evidence.
- **Code at the end:** turns the LLM's buckets into numbers for the book.

That's exactly the PLAN_B §5.3 loop. This result says it beats both the raw LLM (what v1 effectively uses) and the model alone.

## Follow-up (Oct 21): is "a few percent" any good, and was the raw LLM fairly prompted?

The trainer: "We're saying our best model is only a few percent better than pure chance… was pure LLM in the right frame of mind when testing?… I just think we should be able to make it better and vastly improve accuracy."

### 1. The ceiling: how much of an event-window move is the event at all?

**Method:** the variance of each factor's move over event windows divided by its variance over same-length windows on all days since 2004. The share an event could explain is about 1 − 1/ratio.

| Events | Variance vs normal days (8 factors) | Max explainable share |
|---|---|---|
| All unscheduled (3,876) | 1.0–1.1× | **≈ 3%** |
| Moderate / minor (3,093) | 0.8–0.9× | ≈ 0% |
| **Major (733)** | 1.3–1.8× | **≈ 32%** |
| **Extreme (50)** | 3–8× | **≈ 80%** |
| Financial stress | up to 2.7× (credit) | ≈ 41% |
| Unscheduled central bank | about 1.5× | ≈ 26% |
| CPI / FOMC / jobs 2021+ | 1.2–2.2× | 14–23% |
| Wars, sanctions, tariffs | ≤ 1× | ≈ 0% on broad factors |

**Reading:**
- **Typical events:** across all events, the assisted LLM's +2.5% is most of what's achievable (≈ 3%).
- **Big events:** the attributable share is 30–80% and we capture about 7%. That's the headroom, and where improvement has to come from.

### 2. A fair raw LLM, and the best-prompted assisted LLM

**Prompts:**
- **Raw LLM, well prompted:** the regime at the time, the previous week's factor moves, historical base rates of each bucket per factor, and an instruction to reason briefly before answering. No model evidence.
- **Assisted LLM, well prompted:** the same, plus the model's evidence.

**Cost:** 60 more calls, about $2.6.

| | Post-cutoff (2026), all 92 | Post-cutoff, major/extreme (31) | Historical, all 200 (hindsight) | Historical, major/extreme (48) |
|---|---|---|---|---|
| Model | −0.2% | 0.0% | −0.1% | +2.9% |
| Raw LLM (minimal prompt) | −5.7% | −16.0% | +5.3% | +11.3% |
| Raw LLM (well prompted) | −1.2% | −2.3% | **+13.4%** | +19.8% |
| **Assisted LLM** | **+2.5%** | **+6.8%** | +8.8% | +18.2% |
| Assisted LLM (well prompted) | +1.4% | +3.0% | +11.5% | **+24.1%** |

**Paired differences, post-cutoff (R² points; 90% range; share of resamples above 0):**
- well-prompted raw − minimal raw: **+4.5** (+2.1 to +7.5; 100%). The trainer was right that the raw prompt mattered.
- **assisted − well-prompted raw: +3.7** (+0.2 to +7.7; 96%)
- **well-prompted assisted − well-prompted raw: +2.6** (−0.3 to +5.7; 93%). At equal prompting, the model's evidence still adds.
- well-prompted assisted − assisted: −1.1 (−3.5 to +0.9). The heavier prompt doesn't help once the LLM has real analogs.

**Hindsight is visible:** asked to reason first, the raw LLM on 2016–23 events scores +13.4%, better than the assisted LLM, because it can recall what happened. After its cutoff, that advantage disappears (−1.2%). Historical LLM scores can't be used as evidence of skill.

### 3. Verdict

- **Fair comparison:** the model's evidence adds about 2.5–3.7 points over a well-prompted LLM on events the LLM can't remember. Robust against the minimal prompt, likely (93–96%) against the good one.
- **Big events:** on the 31 major/extreme 2026 events, the assisted LLM explains +6.8% against a ceiling of 30–80%. Good relative to the alternatives, far from the ceiling.
- **Where to push for big accuracy gains:**
  1. precise timing (intraday windows around the first report)
  2. surprise scaling with prediction-market probabilities (B5)
  3. magnitude labels (size of the tariff, barrels offline, size of the failure)
  4. more extreme-event history from 1926+ industry data
  5. scoring and tuning on big events, not all events
