# Plan B B5 (part 1): prediction-market-implied surprises for scheduled releases

Run 2026-10-21. No LLM calls.
**Scripts:** `scripts/planb_b5_ladders.py` (ladders, surprises, one-stage tests) and `scripts/planb_b5_twostage.py` (the forward chain).
**Data:** `cache/pmrm/b5/ladders.csv`. **Raw numbers:** `b5_ladders.json`. DECISIONS #172.

## In plain words

**What we built.** For every Kalshi CPI, jobs and GDP release since 2021, we read the ladder of "above X" contracts at 4 pm the afternoon before (for example, CPI above 0.2%: 60%, above 0.3%: 25%…). From that we built the market's expected number and its uncertainty, then compared it with the number actually published that morning (from FRED's release archive).
- **Releases:** 102 in all: 46 CPI, 42 jobs, 14 GDP.
- **Check:** the published number sat inside the ladder's own settled bracket 95–100% of the time, so the data lines up.

**Finding 1: Kalshi measures "what was priced in" much better than our free alternative.** The Kalshi surprise (actual minus Kalshi's expectation) lines up with the bond market's release-day reaction far better than a naive surprise (actual minus recent average):

| Release | Kalshi surprise vs 2y yield move | Naive surprise vs 2y yield move |
|---|---|---|
| CPI | **0.52** | 0.34 |
| Jobs | **0.54** | 0.12 |
| GDP (14 events) | −0.13 | −0.40 |

On its own, the Kalshi surprise predicts **18–21%** of the 2-year yield's release-day move (leave one release out). The naive surprise predicts nothing. This is the "prediction markets vs economists" question in PLAN_B §3, answered in Kalshi's favour against a free baseline. A paid consensus feed wasn't tested.

**Finding 2: a surprise doesn't tell you how stocks react; the rates reaction does.**

| Predictor of the release-day 49-industry cross-section (98 releases) | R² vs type mean | Chance 95th |
|---|---|---|
| Kalshi surprise directly | −3.0% | −0.0% |
| Naive surprise directly | −17.7% | −13.3% |
| 2y yield move (the market's reaction) | **+3.1%** | +1.1% |
| **Long-history regime model given the actual rates reaction** | **+4.7%** | |
| Forward chain: Kalshi surprise → predicted rates → regime model | +0.3% | +0.8% |

The chain breaks at its first link. The headline surprise explains only 18% of the 2y move and 7% of the 10y. The rest of a release-day rate move comes from details, revisions and the expected Fed reaction. Kalshi's history (2021+) is too short to learn that link well per release type.

**What it means for the product:**
- **Probability and size of the surprise:** from Kalshi ("there's a 25% chance CPI prints 0.4% or more; the market expects 0.25%").
- **Damage:** comes through the rates reaction ("a hot print that lifts the 2-year by 10 bp tends to do X to these industries in today's regime"). The long-history regime model handles that well (+4.7%).
- **Rates-reaction range:** comes from the stage-1 relationship plus its wide uncertainty, stated honestly rather than as a point estimate.

## Method notes

- **Quote time:** the last 16:00 ET endpoint before the release morning; carried book quotes (P0 rules); a strike is kept if it has a mid. The ladder is made monotone (P(>k) non-increasing). Mass sits at strike midpoints, with tails half a step outside.
- **Actual value:** the ALFRED vintage on the release day.
  - CPI: MoM % (rounded to 0.1, as published)
  - jobs: change in payrolls
  - GDP: annualized quarterly growth (rounded to 0.1)
- **Responses:**
  - Ken French 49 industries and the long factors on the release day, standardized on 1976–2020 release responses (two-stage) or within the sample (one-stage)
  - the industry data ends 2026-08, so 98 of 102 releases are scored
- **Two-stage:**
  - stage 1 is per release type, leave one release out
  - stage 2 is the regime kernel from `planb_longrel.py` (bandwidth 1, ridge 10, chosen by leave-one-era-out), trained 1976–2020 on realized yield changes
  - chance: stage-1 predictions shuffled across releases (100 times)

## Next

- **Use Kalshi for probability and expected-vs-actual in the release answer,** and the regime model for the rates-reaction → industry step.
- **Part 2 (unscheduled events):** match library events to Kalshi markets that existed beforehand and scale responses by surprise (outcome − p). This needs light, capped LLM matching.

## Kalshi-weighted damage distribution (Oct 21, #174)

The trainer: "[Kalshi] should be weighted in such a way that it improves accuracy in terms of surprise and the move itself."

**What it does** (`src/worsey/planb/release_scenarios.py`, validated by `scripts/planb_b5_distribution.py`):
- **Outcomes:** for an upcoming release, every possible print comes from the Kalshi ladder with its probability.
- **Rates:** each print → surprise vs Kalshi's expectation → a range of release-day rates reactions (stage 1, with its real residual spread).
- **Industries and book:** → industry moves through the 50-year regime model at today's regime → the book's P&L.
- **Result:** a probability-weighted damage distribution with tails, plus "if it prints X" rows.

**Backtest on 98 Kalshi-era releases × 7 baskets (each built from information before the release; stage 1 leave-one-out):**

| Check | Target | Result |
|---|---|---|
| Real move inside the 5–95% range | 90% | **93%** |
| Real move inside the 25–75% range | 50% | **51%** |
| Centre of the distribution vs zero (R²) | | −0.8% (expected: the print isn't known in advance) |

**Bug fixed during validation:** the first version estimated a basket's own noise as if its industries were independent (coverage only 82% / 34%). Measuring each basket's spread directly on past release days fixed the calibration.

**Reading:** the value of Kalshi here is a calibrated, probability-weighted range of damage for each release, not a point forecast of the move.
