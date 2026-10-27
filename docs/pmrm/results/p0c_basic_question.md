# PMRM P0c: does event covariance predict portfolio P&L on held-out days?

Run 2026-10-20 by `scripts/pmrm_p0c.py` (no LLM calls). Data from 2022-01-03 to 2026-10-16. Raw numbers: `p0c.json` (carried quotes) and `p0c_strict.json` (strict). BUILD_PLAN P0c, DECISIONS #151.

**The trainer's ruling (Oct 20): no usable signal.** The small-move idea is reported as a negative result. This model becomes the baseline Plan B must beat, and P1–P5 are not built. DECISIONS #155.

## In plain words

**The question.** When prediction-market odds move during a day (Fed-cut odds up 4 points, recession odds down 2), does knowing that move help predict how a stock portfolio did that same day? If it did, Worsey could answer "if these odds move, what happens to my book?" from the daily co-movement of odds and prices.

**How it was tested.**
- **Events:** 15 of them: the Fed, CPI, jobs, recession, GDP, US–Iran, House and Senate markets (each rolled from contract to contract), plus 7 long-lived single markets.
- **Portfolios:** 8: Delta, Exxon and Microsoft; the demo book; and baskets of banks and utilities, energy, travel, big tech, defensives, and stocks that should have nothing to do with any of these events.
- **Learning:** each model learned only from the previous 120 trading days, then predicted the next days it had never seen, rolling forward through 2022–2026 (815 test days per portfolio).
- **Comparison:** each was compared with a model that ignores the odds entirely.
- **Chance baseline:** the whole test was rerun 50 times with the odds slid out of sync with prices by random amounts. That shows how much "skill" pure luck produces.

**What we found.**
- **There is a real link, but it is tiny.** The odds explain **less than 1% of a portfolio's daily moves** in every case tried; usually close to 0%.
  - **Real:** more combinations beat the luck baseline than luck alone would produce (30 vs about 11).
  - **Tiny:** knowing the odds move improves a forecast of tomorrow's portfolio move about as much as knowing the weather would.
- **Most of what little there is is just "the market went up or down".** Once the S&P 500's own move is taken out, almost nothing is left.
- **The control group behaved.** The baskets that shouldn't care about these events showed no link, as expected.
- **The strict data reading** (only odds that printed exactly at 4 pm) gives the same answer.

**What it means.**
- **Daily odds moves aren't a useful way to measure what an event does to a portfolio, at least with the 2–4 years of prediction-market history that exist.** On a typical day, odds wiggle for reasons that have little to do with stock prices, and the real information arrives in a few moments (the release, the decision) that daily data blur.
- **That's why the work now moves to Plan B.** Plan B learns event impacts from decades of actual events and their price reactions, uses prediction markets for the *probability* of an event, and must beat this test's result to count.

## Verdict (technical): statistically detectable, economically negligible

- **Something real is there.**
  - **Significant cells:** 30 of 272 cells (event or "all jointly", × portfolio × raw/beyond-market) beat chance on log score at p ≤ 0.05.
  - **Measured null:** the same count on shifted data averages 10.8, with a 95th percentile of 17.5 and a maximum of 24 over 50 replicates. So the excess isn't multiple-testing noise.
  - **Where it comes from:** Fed, CPI and the 2026 recession contract, on cyclical and tech portfolios.
- **It's tiny.**
  - **Held-out R²:** below 1% in every cell (maximum 0.86%); the median cell is slightly negative.
  - **Joint R²:** with all events at once, the best is 0.73% (travel and consumer bucket, shrunk covariance).
  - **Log-score gains:** 0.002 to 0.010 nats per row at best.
- **Most of it is broad-market exposure.**
  - **Raw:** 19 of 136 cells are significant.
  - **Net of SPY:** 11 of 136, against about 5.4 expected by chance. That's 1 of 16 for the joint model (energy bucket, shrunk covariance).
- **Sanity checks behave.** The placebo and defensive portfolios get negative joint gains and R² under every model.
- **Strict quotes say the same** (§4): 30 of 208 cells beat chance (null mean 8.2, 95th percentile 18.2), and the largest held-out R² is 0.69%.

**For the trainer's gate.** The core premise (daily odds moves carry held-out information about portfolio P&L) is true only in the weakest sense. The information is real but explains under 1% of daily P&L variance, mostly through the market, and less than the cost of estimating it in most cells. That's below any economically useful threshold for a small-move engine. It supports DECISIONS #152 (Plan B as the main direction), with P0c as the naive baseline Plan B must beat (B5). The joint shrunk-covariance model is that baseline: about +0.2% to +0.7% held-out R² on the cyclical portfolios.

## 1. Setup

**Events (15):**
- **Stitched front-contract series (8):** Fed, CPI, payrolls, recession, GDP, US–Iran agreement, House, Senate.
- **Single markets from the current build with ≥ 200 clean rows (7):**
  - `KXRECSSNBER-26`
  - `TESLACEOCHANGE-26` (411)
  - `TESLAOPTIMUS-26DEC31` (390)
  - `KXRETIREMM-26` (252)
  - `KXSAVEACT-27-JAN04` (240)
  - `KXMJSCHEDULE-27` (226)
  - `KXGOVTCUTS-26-250` (210)

**Portfolios (8):**
- Study A (XOM, DAL, MSFT, equal-dollar)
- the demo `balanced` book
- one equal-dollar portfolio per Study B bucket: rate-sensitive, energy, travel and consumer, mega-cap tech, defensives, placebo

**Panel:** 935 eligible daily intervals at 16:00 ET, 815 scored per portfolio.

**Fitting:**
- **Window:** rolling 120 rows, refit every 20, predictions frozen before each scored row, raw scoring, no tuning on test rows.
- **Event moves:** an event's move enters a prediction only when the event is valid and interior at the start of the day, |dp| ≤ 0.05, and |z| ≤ 3 against its training sd. Otherwise it is unobserved that day (never zero).

**Models, each against M0** (the unconditional Gaussian from the training window):
- **One event at a time:** M0's Gaussian conditioned on the event through the training correlation: mean m0 + ρ·σ_y/σ_x·(x − x̄), variance σ_y²(1 − ρ²).
  - **Fix made before the final run:** a first version used the residual variance from event-quoted days only. It could "win" because those days are calmer, not through covariance, so it was replaced.
- **All events jointly:**
  - **Shrunk covariance:** Gaussian conditioning on every event observed that day; pairwise moments with missing values; Ledoit-Wolf-style shrinkage toward the diagonal; PSD.
  - **Ridge (check):** penalty by 5-fold blocked cross-validation inside the training window only; missing moves set to the training mean.
- **Beyond the market:** the same, on portfolio return minus β·SPY, with β from the training window.

**Chance:**
- **Shifts:** each event series is circularly shifted by its own random 20+ days against the portfolios, 50 times. p = (1 + #null ≥ observed) / 51, so the smallest attainable p is 0.02.
- **Multiplicity:** the number of significant cells has its own measured null (each shifted replicate scored against the others).

## 2. Lead table: all events jointly × portfolio

**Raw portfolio P&L: all events jointly**

| Portfolio | Model | Held-out R² | R² null 95th | Gain (nats/row) | Gain null 95th | p (gain) |
|---|---|---|---|---|---|---|
| study_a (XOM,DAL,MSFT) | shrunk cov | +0.16% | +0.42% | -0.0021 | +0.0014 | 0.37 |
| study_a (XOM,DAL,MSFT) | ridge | -0.38% | +0.29% | -0.0017 | +0.0006 | 0.39 |
| balanced book | shrunk cov | +0.15% | +0.36% | +0.0029 | +0.0013 | 0.04 |
| balanced book | ridge | -0.73% | +0.17% | +0.0000 | +0.0005 | 0.10 |
| bucket: rate_sensitive | shrunk cov | +0.27% | +0.43% | +0.0016 | +0.0024 | 0.14 |
| bucket: rate_sensitive | ridge | -0.13% | +0.17% | -0.0020 | +0.0013 | 0.47 |
| bucket: energy | shrunk cov | +0.34% | +0.37% | +0.0020 | +0.0011 | 0.06 |
| bucket: energy | ridge | -0.23% | +0.14% | +0.0002 | +0.0006 | 0.08 |
| bucket: travel_consumer | shrunk cov | +0.73% | +0.18% | +0.0021 | +0.0006 | 0.02 |
| bucket: travel_consumer | ridge | +0.59% | +0.38% | +0.0052 | +0.0015 | 0.02 |
| bucket: megacap_tech | shrunk cov | +0.29% | +0.24% | +0.0030 | +0.0011 | 0.04 |
| bucket: megacap_tech | ridge | +0.55% | +0.30% | +0.0060 | +0.0012 | 0.02 |
| bucket: defensives | shrunk cov | -0.55% | +0.41% | -0.0031 | +0.0010 | 0.57 |
| bucket: defensives | ridge | -0.60% | +0.22% | -0.0016 | +0.0003 | 0.35 |
| bucket: placebo | shrunk cov | -0.51% | +0.38% | -0.0022 | +0.0017 | 0.63 |
| bucket: placebo | ridge | -0.32% | +0.33% | -0.0008 | +0.0010 | 0.39 |

**Beyond the market (net of β×SPY): all events jointly**

| Portfolio | Model | Held-out R² | R² null 95th | Gain (nats/row) | Gain null 95th | p (gain) |
|---|---|---|---|---|---|---|
| study_a (XOM,DAL,MSFT) | shrunk cov | -0.14% | +0.28% | -0.0020 | +0.0006 | 0.45 |
| study_a (XOM,DAL,MSFT) | ridge | +0.15% | +0.16% | -0.0002 | -0.0000 | 0.10 |
| balanced book | shrunk cov | -0.70% | +0.29% | -0.0016 | +0.0004 | 0.39 |
| balanced book | ridge | +0.10% | +0.16% | +0.0015 | +0.0002 | 0.06 |
| bucket: rate_sensitive | shrunk cov | -0.02% | +0.34% | -0.0009 | +0.0013 | 0.25 |
| bucket: rate_sensitive | ridge | -0.50% | +0.13% | -0.0025 | +0.0004 | 0.51 |
| bucket: energy | shrunk cov | +0.38% | +0.14% | +0.0015 | +0.0008 | 0.04 |
| bucket: energy | ridge | -0.31% | +0.24% | -0.0014 | +0.0004 | 0.33 |
| bucket: travel_consumer | shrunk cov | +0.26% | +0.22% | +0.0004 | +0.0007 | 0.08 |
| bucket: travel_consumer | ridge | -0.08% | +0.19% | -0.0006 | +0.0001 | 0.18 |
| bucket: megacap_tech | shrunk cov | -0.35% | +0.35% | -0.0004 | +0.0015 | 0.22 |
| bucket: megacap_tech | ridge | -0.36% | +0.22% | -0.0019 | -0.0000 | 0.49 |
| bucket: defensives | shrunk cov | -0.47% | +0.56% | -0.0046 | +0.0010 | 0.67 |
| bucket: defensives | ridge | -0.32% | +0.23% | -0.0024 | -0.0001 | 0.41 |
| bucket: placebo | shrunk cov | -0.28% | +0.32% | -0.0022 | +0.0014 | 0.59 |
| bucket: placebo | ridge | -0.02% | +0.44% | -0.0003 | +0.0013 | 0.25 |

## 3. One event at a time: where the significant cells are

| Event | Raw: portfolios with p ≤ 0.05 (of 8) | Raw: best R² | Beyond market: p ≤ 0.05 (of 8) | Beyond market: best R² |
|---|---|---|---|---|
| Fed (front series) | 4 | 0.35% | 2 | 0.41% |
| CPI (front series) | 3 | 0.10% | 1 | 0.32% |
| KXRECSSNBER-26 (single) | 3 | 0.79% | 2 | 0.86% |
| Recession (front series) | 2 | −0.19% | 0 | −0.12% |
| US–Iran agreement | 1 | 0.09% | 0 | 0.18% |
| KXMJSCHEDULE-27 | 1 | 0.46% | 1 | 0.54% |
| House, Senate, GDP, payrolls | 0 | ≤ 0.62% | 0 | ≤ 0.35% |
| Other single markets | 0 | ≤ 0.28% | 0–1 each | ≤ 0.43% |

Every cell, with its R² null, gain null, p and block t, is in `p0c.json`.

**Two oddities, recorded.**
- **The recession front series wins on log score (2 portfolios) while its held-out R² is negative.** Its gain comes from the variance term on rare large-move days, not from predicting the mean.
- **The Tesla and other political single markets are long-lived but unrelated to the portfolios.** They behave like the placebos, as they should.

## 4. Strict-quote sensitivity

Strict reading (a candle at exactly 16:00 at both ends; `p0c_strict.json`). 11 events qualify (US–Iran, House, Senate and 3 of the singles drop below 200 clean rows or out of the panel); 208 cells.

- **Significant cells:** 30 of 208 have p ≤ 0.05 on log score. The measured null averages 8.2 (95th percentile 18.2, maximum 27). The excess is similar to the carried run.
- **R²:** at most 0.69%, median −0.06%.
- **The joint model** wins on log score for 9 of 16 raw portfolios (gains 0.002–0.007 nats per row), but its held-out R² is ≤ 0 for 5 of those. The gain comes from the variance term (sharper intervals on quiet days), not from predicting the mean.
- **Beyond the market:** the joint model is significant for 3 of 16 (Study A, rate-sensitive, energy), with R² up to 0.51%.

Same conclusion under both readings: real but negligible.

## 5. Gate options

1. **Treat as "no usable signal"** for the small-move premise. Report the negative result, keep P0c's joint model as the naive baseline for Plan B (B5), and don't start P1–P5. Recommended, given #152.
2. **Treat as "mixed"** and narrow the small-move work to Fed, CPI and recession on cyclical and tech portfolios. Expect gains of a few thousandths of a nat per row.
