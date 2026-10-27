# Plan C, C0: the frozen test and the baselines

Run 2026-10-22. Scripts: `scripts/planc_c0.py` (build, score), harness `src/worsey/planc/harness.py` (tests `tests/test_planc_harness.py`). Raw numbers: `planc_c0.json`; the dataset is `cache/pmrm/planc/c0_events.pkl`. No LLM calls. DECISIONS #187.

## In plain words

**What C0 does.** It fixes, before any Plan C model exists, exactly how every rung will be scored, and scores the baselines a learned model has to beat.

**The dataset (reused by every later step):** 3,571 events.
- **Splits:** train before 2023 (2,995 events), validation 2023–2025 (484), test 2026 (92).
- **Targets per event:**
  - **the 13 factors' moves**
  - **the 96 research ETFs' moves beyond the factors:** where sector, country and commodity-specific effects land
  - **the 12 portfolios of #185**

**The baselines on test (2026).** Lower score is better; it's standardized by each target's normal noise.

| Target | Rung 0: zero + normal noise | Rung 0b: noise + event premium | Rung 1: today's engine |
|---|---|---|---|
| 13 factors | **0.607** (90% inside the 90% range) | 0.616 (92%) | 0.630 (86%) |
| 96 ETF residuals | **0.713** (85%) | 0.716 (88%) | 0.720 (78%) |
| 12 portfolios | **0.577** (91%) | 0.589 (93%) | 0.587 (90%) |

- **Rung 0 wins everywhere,** on validation (2023–25) as well as test, and on big events too.
- **The engine is 1–4% worse than plain noise,** and loses in 95–100% of resamples on most comparisons.
- **The event premium makes things worse.** On average an event window isn't bigger than a normal day (#171), so widening the range only costs.

**What it means for Plan C.** The bar is plain normal noise for each target, not today's engine. Plan C has to learn when an event pushes a target beyond its normal noise, and which way. The gated "noise unless it isn't" output (PLAN_C §4) is built for exactly that: its default is rung 0, and it only departs when the inputs justify it.

## Frozen test (also in PLAN_C §7a)

| Item | Frozen as |
|---|---|
| Splits | train < 2023-01-01; validation 2023–2025; test 2026+ plus the prospective log |
| Prediction format | 19 quantiles (5%, 10%, …, 95%) per event and target |
| Horizon | the library's short window (to the second-session close); the 2-hour first reaction added in C1 for timed events |
| Targets | 13 factors; 96 ETFs' moves beyond the factors; 12 portfolios |
| Main score | CRPS from the 19 quantiles, standardized by the target's normal noise over the window, averaged over targets |
| Also reported | 90% and 50% coverage, below-5% rate, R² of the median vs zero; big events separately; by target kind |
| Reference rung | the best of rungs 0, 0b and 1 on validation: **rung 0** (zero + normal noise) |
| Pass bar | a rung beats the reference on test CRPS with ≥ 90% paired bootstrap support over events, holds on the prospective log, and a deeper rung must beat a shallower one the same way |
