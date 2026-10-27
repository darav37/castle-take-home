# Are the shipped ranges calibrated?

Run 2026-10-22 ( §3). Script: `scripts/planb_range_calibration.py`. Raw numbers: `range_calibration.json`. No LLM calls. DECISIONS #183.

## In plain words

**The test.** For every one-off event we rebuilt the range the chat would have shown and checked where the real move landed:
- **Ranges:** made with the same function `price()` uses (`live.scenario_draws`).
- **No hindsight:** the event model was refitted only on earlier events, and each book's exposures came from the two years before the event.
- **Events:** all 92 events of 2026, plus 1,681 events from 2016–2025 scored with rolling yearly refits.
- **Books:** the 11 books of #178, plus the 13 factors one at a time.

**1. On real books, the end-of-day ranges are calibrated.**

| | Middle half (target 50%) | 9-in-10 range (target 90%) | Below the 1-in-20 bad case (target 5%) | Above the top 5% (target 5%) | Bias of the middle |
|---|---|---|---|---|---|
| 2026 (92 events × 11 books) | 50% (47–53) | 90% (87–92) | 3.6% (2.3–4.9) | 6.6% (4.7–8.5) | +0.08 sd |
| 2016–2025 (1,681 events) | 51% (50–52) | 91% (91–92) | 3.7% (3.3–4.1) | 5.2% (4.7–5.6) | +0.05 sd |

90% error bars come from a bootstrap over events. The middle isn't biased, and misses are close to even on both sides. Real moves land a little more often above the range than below it, so the bad case is slightly conservative.

**2. But the ranges are no sharper than "nothing happens, plus normal noise."** CRPS scores a whole range against the outcome, rewarding both calibration and tightness. Against a predict-zero range with the book's normal day-to-day noise, the shipped ranges score 2026 −2.2% and 2016–2025 +0.2%, about even. That's the honest reading of #171 and #180: for most events the best forecast is "the usual noise". The exception is extreme events in 2016–2025, where the shipped ranges beat the baseline by +11%.

**3. Where the ranges miss:**
- **Extreme events are too narrow.** In 2016–2025 only 82% landed inside the 9-in-10 range (28 events), with misses on both sides. 2026 has only 4 extreme events, too few to check.
- **Some factors miss in 2026.**
  - Oil (9-in-10 67%), gold (66%) and EM vs US (74%) were too narrow, with misses on both sides. Those factors had an unusually volatile 2026 and were fine over 2016–2025 (86–94%).
  - Credit was too wide in 2026 (100%).
  - Value vs growth was too narrow in both periods (76–78%).
- **First-hours ranges run slightly wide.** For 2022–2025 timed events, 61–67% of first reactions landed inside the middle half (target 50%), while the 9-in-10 range was about right (89–96%). The 2026 check has only 11 timed events.

**What it means:**
- The ranges the chat shows are honest. A user reading "9 in 10 between −$X and +$Y" gets that about 9 times in 10, across ordinary and unusual books.
- They aren't more informative than "expect normal noise" except for the biggest events, which matches every earlier finding: the product's value is honest, book-specific ranges and the "most events don't matter" default, not sharp forecasts.

## Proposed fixes (not applied; the task forbids tuning on the scored events)

| Miss | Proposed fix | Where it was found | Where it can be tested |
|---|---|---|---|
| Extreme too narrow (82% in 9-in-10) | widen extreme-event spread by about ×1.25 (the normal-equivalent of 82% → 90%) | 2016–2025 | 2026 has 4 extreme events, too few; needs the prospective log or pre-2016 extreme events |
| First hours too wide (middle half ~63%) | narrow the first-hours spread by about ×0.75 | 2022–2025 timed events | 2026 timed events (11): too few; the prospective log's timed events |
| Value vs growth too narrow in both periods | widen that factor's spread by about ×1.3 | 2016–2025 | 2026 (value 9-in-10 was 78%) |
| Oil, gold, EM in 2026 | none yet: fine over 2016–2025, so it looks like a 2026 volatility regime | 2026 | the prospective log; a regime-scaled spread (residual size by VIX) is a candidate |

## Method notes

- **Draws:** `live.scenario_draws` with the event's stored schema labels (the library's labels, not a new LLM call):
  - middle = the engine's estimate through the book's exposures
  - spread = residual factor vectors of earlier events of the same severity plus the book's own noise over ~1.5 sessions, recentered on the middle
- **Book exposures:** OLS of the book's daily returns on the 13 standardized factor proxies over the 504 sessions before the event; own noise = that regression's residual sd.
- **Real move:** the book's compounded total return over the library's event window.
- **Factors:** the factor's real event-window move (`U`) against the factor's draws (no own-noise term).
- **First hours:** timed events from #177 (`cache/pmrm/bigtest/windows.json`); the real first reaction in factor-sd units; the first-reaction scaling estimated only from timed events before each cutoff (2023+).
- **CRPS:** from the draws (sample formula) vs a normal with mean zero and sd = the book's daily sd × √(window sessions).
- **Error bars:** the 11 books share each event, so the bootstrap resamples events, not rows.
