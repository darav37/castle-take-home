# Single-stock track, S3: after the open, does the gap stick?

Run 2026-10-23. Script: `scripts/stock_s3.py`. Raw numbers: `stock_s3.json`. No LLM calls, $0. DECISIONS #206.

## In plain words

**The question.** A user reads company news at 9:45, after the stock has opened. Does the opening gap tell them where it closes?

**The data:**
- the 25,915 earnings reports (S0)
- the company-news events published outside market hours (S0b)
- each reacting session split into the gap (previous close → open) and the rest of the day (open → close), in units of the stock's normal daily swing
- ordinary days of the same stocks as the baseline

| | Gap (RMS) | Rest of day (RMS) | Share of the day's move in the gap | Rest of day vs gap (slope) | Big gaps that give back half | Big gaps closing on their side |
|---|---|---|---|---|---|---|
| Earnings | 2.43 | 1.85 | 63% | −0.01 | 18% | 94% |
| Guidance | 1.63 | 1.36 | 59% | +0.01 | 13% | 98% |
| M&A | 1.37 | 0.96 | 67% | +0.02 | 12% | 97% |
| Downgrades | 1.26 | 1.02 | 60% | +0.07 | 19% | 97% |
| Upgrades | 1.12 | 1.00 | 55% | +0.02 | 19% | 94% |
| FDA | 1.08 | 0.91 | 58% | +0.10 | 15% | 97% |
| Initiations | 0.56 | 0.88 | 28% | −0.10 | 29% | 88% |
| (ordinary day) | 0.53 | 0.88 | 26% | +0.05 | 22% | 94% |

**What it shows:**
1. **The gap sticks.** The rest of the day is unrelated to the gap for every kind of news (slopes within ±0.16). Big gaps extend or fade about half and half, rarely give back half (12–19%), and the day closes on the gap's side 94–98% of the time.
2. **Most of the news is in the gap:** 55–67% of the day's move for earnings, M&A, guidance and rating changes. Small news (initiations, contracts, dividends) looks like an ordinary day.
3. **Earnings days stay volatile after the open:** the rest of the day swings about 1.9× a normal day, so the range to the close is still wide.

**In the product.** `news_base_rates` adds `after_the_open` when today's open is known (live data):
- today's gap
- the middle to the close = the open
- the 9-in-10 range from the rest-of-day swing for that kind of news
- how often big gaps close on their side

The earnings evidence says the same. The same rule as the book-level after-the-open range (#193, #200), now for single names.
