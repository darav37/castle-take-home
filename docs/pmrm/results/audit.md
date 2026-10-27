# PMRM stage 0: data and power audit (P0)

Run 2026-10-18/19 by `scripts/pmrm_audit.py` (no LLM calls). Data through the 2026-10-16 close. Raw numbers are in `audit.json`. Spec v1.6, §0.1, §0.1a, §2, §3, §6.

## Verdict

1. **Study A as specified fails the stage 0 gate: `INSUFFICIENT_DATA`.**
   - The seed universe has **29 complete rows** under the carried-quote reading and **5** under the strict reading. It needs ≥ 200 (120 training + 40 validation + 40 test).
   - The bottleneck is event_C (exactly 25 bp in December). It spends most of its life below the 5% interior floor, leaving only 38 clean days.
2. **No three-event substitute passes.** Kalshi has no long-lived oil contract. The WTI yearly-high series has usable quotes mostly from 2026, and weekly Brent from April 2026. Every universe with an oil *event* has ≤ 59 rows.
3. **Proposed substitute (for the trainer to approve or reject):** two events plus oil as a context variable.
   - **Events:** recession and Fed, each as a front-contract series.
   - **Oil:** the USO total return as a §6 J1 context variable, not an event.
   - **Stocks:** XOM, DAL, MSFT.
   - **Size:** d = 6, keeping all three themes.
   - **Rows:** **244** under the carried reading (84 test rows) and **139** under the strict reading.
   - **Gate:** it passes only under the carried reading.
4. **The planned primary test (M2c vs M2) is underpowered at 244 rows.**
   - At 80% power it detects only gains of about **0.14 nats per row** in the scenario where cleaning helps at all.
   - That scenario is cleaning shared venue glitches, which gains 0.03–0.09 nats per row. Power for those gains at 244 rows is 24–56%.
   - The test also over-rejects at this size: 15% empirical size against a nominal 5%.
   - With the proposed substitute, the expected outcome is **INCONCLUSIVE**. The detectable effect becomes about 0.08 nats at 500 rows and 0.05 at 1,000.

## 1. Rules applied

| Rule (§2, §6) | Applied as |
|---|---|
| Endpoint | 16:00 America/New_York on full XNYS sessions (from SPY daily bars); quote at or before the endpoint, never after |
| Interval | Consecutive sessions 23–25 h apart. Weekends, holidays and the 13:00 early closes are excluded (the early-close list is hand-entered for 2023–2026: no exchange-calendar package is installed) |
| Quote | Midpoint; 0 ≤ bid ≤ ask ≤ 1; spread ≤ 0.04 |
| Interior | p_start in [0.05, 0.95] for every event in the row; conditioning event \|dp\| ≤ 0.05 |
| Life | The interval containing a market's close or settlement, and every later interval, are excluded |
| Jump day | \|dp\| ≥ 3 pp (the J2 flag rule) and ≥ 5 pp (the conditioning cap); independent episodes = jump days more than 5 sessions apart |
| Stale | Book unchanged (same mid) and zero volume since the previous endpoint |
| Total return | Massive `adjusted=true` adjusts for splits only. Cash dividends are added on the ex-date (largest dividend contribution: XOM 0.9%). |

**Finding: §6's 900-second quote-age rule can't be checked on Kalshi's hourly candles.** Kalshi returns a candle only for an hour with a trade or a quote change. So the 16:00 book is either:
- **strict:** a candle ending exactly at 16:00 (age 0), or
- **carried:** the last earlier candle, a standing quote that hasn't changed (its age is reported).

The audit reports both readings. Choosing one is the trainer's call. Two other routes would also settle it: minute candles, or the prospective live collection.

## 2. Study A seeds

| | event_A | event_B | event_C |
|---|---|---|---|
| Contract | `KXWTIMAX-26DEC31-T150` | `KXRECSSNBER-26` | `KXFEDDECISION-26DEC-C25` |
| Rule (verbatim gist) | WTI front-month settle above $150 by Dec 31, 2026 (ICE) | two consecutive quarters of negative GDP growth in 2025 or 2026 (BEA) | Fed cuts 25 bp on Dec 9, 2026 |
| Listed / closes | 2026-03-04 / 2026-12-31 | 2025-07-14 / 2027-01-31 | 2025-09-29 / 2026-12-09 |
| Intervals in life | 125 | 249 | 206 |
| Valid, carried / strict | 111 / 83 | 249 / 234 | 63 / 38 |
| Clean (interior), carried / strict | 111 / 83 | 234 / 219 | **38 / 20** |
| Stale share | 0% | 7.6% | 14.3% |
| Jump days ≥ 3 pp / ≥ 5 pp | 20 / 9 | 18 / 3 | 3 / 1 |
| Independent jump episodes | 7 | 9 | 2 |
| Excess kurtosis of dp | 12.5 | 3.9 | 17.3 |

**Threshold caveats (§2: never substitute an almost-matching market silently):**
- **event_A is not literally "a record":** WTI's record settle is $145.29 (intraday $147.27), and Kalshi lists no strike there. T150 is the nearest strike above the record.
- **event_B is not an NBER recession:** it resolves on the two-negative-quarters GDP rule.

**Pairwise overlap of clean days (carried; strict in brackets):**

| | A | B | C |
|---|---|---|---|
| A | 111 (83) | 96 (71) | 31 (7) |
| B | | 234 (219) | 36 (18) |
| C | | | 38 (20) |

**Complete-case rows** (all events clean, |dp_B| ≤ 0.05, all stocks valid): **29 carried (2026-03-27 to 09-09), 5 strict.**

Stocks XOM, DAL and MSFT have valid total returns on every eligible interval (254 in the Study A window).

## 3. Substitutes considered (same themes, recurring series, window 2022-01-03 to 2026-10-16)

Rolling variable: the front unresolved event. Within it, the market whose mid is nearest 0.5 on the first session it becomes front. It is re-picked only when that market resolves (e.g. a touched strike), using information at that time only. An interval counts only if both endpoints are on the same market. Conditioning event: recession.

| Universe | Rows carried | Rows strict |
|---|---|---|
| oil (WTI yearly high) + recession + Fed | 59 | 36 |
| weekly Brent + recession + Fed | 42 | 38 |
| recession + Fed + CPI | 131 | 89 |
| recession + Fed + payrolls | 155 | 94 |
| **recession + Fed (+ USO context)** | **244** | **139** |

## 4. Study B pools (proposed; frozen only on the trainer's approval)

**Event families.** Window 2022-01-03 to 2026-10-16. "Clean" = valid and interior.

| Category | Series | Clean carried / strict | Stale | Jumps ≥ 3 pp (episodes) | First / last clean | Status |
|---|---|---|---|---|---|---|
| Fed decisions | KXFEDDECISION | 398 / 294 | 2.5% | 150 (32) | 2023-04 / 2026-10 | in |
| CPI | KXCPI | 473 / 283 | 14% | 151 (55) | 2022-01 / 2026-10 | in |
| Payrolls | KXPAYROLLS | 409 / 205 | 12% | 64 (36) | 2023-03 / 2026-10 | in |
| Growth / recession | KXRECSSNBER | 463 / 239 | 10% | 47 (21) | 2022-07 / 2026-10 | in |
| Growth / recession | KXGDP | 431 / 164 | 22% | 87 (41) | 2022-01 / 2026-10 | in |
| Energy | KXWTIMAX (yearly high) | 143 / 96 | 24% | 21 (11) | 2023-02 / 2026-10 | thin |
| Energy | KXBRENTW (weekly) | 77 / 75 | 0% | 74 (3) | 2026-04 / 2026-10 | too short |
| Energy | KXWTIMAX (monthly) | 4 / 3 | | | 2026-07 / | out |
| Trade / geopolitics | KXUSAIRANAGREEMENT | 227 / 212 | 0.4% | 47 (13) | 2025-06 / 2026-10 | in |
| Trade / geopolitics | KXHORMUZNORM | 92 / 92 | 0% | 41 (7) | 2026-03 / 2026-10 | thin |
| Trade / geopolitics | KXEFFTARIFF | 24 / 9 | 37% | 5 (3) | 2026-04 / 2026-07 | out |
| Elections | CONTROLH (House 2026) | 366 / 287 | 11% | 15 (10) | 2024-09 / 2026-10 | in |
| Elections | CONTROLS (Senate 2026) | 365 / 277 | 19% | 12 (9) | 2024-09 / 2026-10 | in |

**Stock pool.** All have total-return coverage on all 935 eligible intervals. UAL, AMZN and ORLY pay no dividends.
- **Rate-sensitive:** JPM, BAC, DHI, LEN, NEE, DUK
- **Energy:** XOM, CVX, COP, OXY
- **Travel and consumer:** DAL, UAL, MAR, HD
- **Mega-cap tech:** MSFT, AAPL, NVDA, GOOGL, AMZN
- **Defensives:** PG, KO, JNJ, WMT
- **Placebo:** ROL, WM, CHD, VRSK, ORLY

**Placebo caveat for the trainer.** No US stock has zero market beta, so placebos *will* show a response through the shared market factor. The placebo test should be stated as "no response beyond what the market factor implies". Otherwise every model "fails" it.

Pairwise overlaps of every family pair are in `audit.json` (`study_b.overlap_carried`, `overlap_strict`).

## 5. Power gate (§6): the planned primary, simulated

**Simulated:** M2c vs matched-k M2.
- **Fitting:** rolling 120-row training window, refit every 20 rows; the first 160 rows are training plus validation; k matched.
- **Score:** conditioning event B; equal-dollar portfolio of the three stocks; Gaussian negative log score on raw test rows within the support gate (|x| ≤ 3 training sd).
- **Test:** one-sided 5% day-block (5-day) percentile-t bootstrap of the mean paired difference.
- **Replications:** 200 per cell.

**Data assumptions from the audit:**
- t innovations with 6.3 df (from the median excess kurtosis of the seed stocks)
- factor lag-1 autocorrelation 0.035
- one true factor

**J2 labelling assumptions:**
- J2 flags entries with |z| ≥ 2.5
- it labels true contamination as idiosyncratic with 85% recall
- it wrongly labels 15% of real large moves

**Approximation:** cleaned models fit on pairwise moments with masked entries removed, then the §4.2 raw variance top-up. The real build fits the masked likelihood directly.

**Two contamination mechanisms:**
- **idiosyncratic:** one contract's bad print or own news
- **shared:** a venue-wide bad print that moves every event the same day

| Rows (test rows) | Mechanism | Rate, jump size | Mean gain, nats/row | Power |
|---|---|---|---|---|
| 244 (84) | none | (empirical size) | −0.000 | **0.15** |
| 244 (84) | idiosyncratic | 5–20%, 4–6 sd | ≈ 0.000 | 0.10–0.16 (≈ size) |
| 244 (84) | shared | 3%, 4 sd | 0.026 | 0.24 |
| 244 (84) | shared | 5%, 6 sd | 0.087 | 0.44 |
| 244 (84) | shared | 10%, 6 sd | 0.090 | 0.56 |
| 500 (340) | none | (empirical size) | 0.000 | 0.09 |
| 500 (340) | shared | 3% / 5% / 10% | 0.030 / 0.085 / 0.099 | 0.52 / 0.94 / 1.00 |
| 1000 (840) | none | (empirical size) | 0.000 | 0.055 |
| 1000 (840) | shared | 3% / 5% / 10% | 0.027 / 0.085 / 0.098 | 0.82 / 1.00 / 1.00 |

**Smallest effect detected with 80% power** = 2.49 × the bootstrap SE of the mean difference. It depends on the noise level:

| Rows | When contamination is shared | When it is idiosyncratic or absent |
|---|---|---|
| 244 | about 0.14 nats/row | about 0.002–0.011 nats/row |
| 500 | 0.08 | |
| 1,000 | 0.05 | |

The idiosyncratic case has the small minimum detectable effect, but its true gain is ≈ 0 there.

**Findings:**
1. **The variance top-up cancels the gain from cleaning idiosyncratic jumps on the conditioning event.** Restoring the event's raw variance attenuates the conditional slope exactly as in M2 (§4.2 says so). The cross-covariances were unbiased to begin with. So J2 can only win the *primary* comparison when contamination is shared across variables (it creates a fake factor), or through effects this simulation leaves out:
   - stock-side cleaning when the conditioning event isn't the contaminated variable
   - non-conditioning events in larger universes
2. **The test is anti-conservative with few test rows.** Empirical size is 15% at 84 test rows and 9% at 340, against 5% nominal; it reaches about 5% only near 840. P1's `score.py` needs a test calibrated by simulation, or more and longer blocks, before any Study A verdict.
3. **At 244 rows a realistic J2 gain is detected less than half the time.** An INCONCLUSIVE outcome is the expected result, not evidence against J2.

## 6. Options for the trainer (P0 decision)

1. **Approve the substitute:** recession plus Fed as front contracts, with USO as a context variable. Use the carried quote reading. Run Study A knowing it is likely INCONCLUSIVE, as a full end-to-end mechanical run. Study B's pooled primary (many universes, shared days) is where power can come from.
2. **Strict quotes:** 139 rows. That's `INSUFFICIENT_DATA` with this data. Pursue minute candles, or wait for prospective collection.
3. **Prospective-only Study A:** keep the seed contracts and collect forward.
4. **Change the primary:** e.g. stage 2a (M1 vs M0) as the gate, or condition on a higher-variance event family (Fed or CPI). This needs a logged spec change before any fitting.

Either way, the Study B pools above need approval (with the placebo caveat), and the quote-age reading needs a ruling.
