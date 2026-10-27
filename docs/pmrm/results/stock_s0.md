# Single-stock track, S0: earnings as corporate events

Run 2026-10-23. Script: `scripts/stock_s0.py` (`build`, `audit`). Raw numbers: `stock_s0.json`. No LLM calls, $0. DECISIONS #202.

## In plain words

**The data:**
- Benzinga earnings via Massive for the current S&P 500 plus the demo names: **25,915 reports, 493 companies, July 2011 to Oct 2026**
- 94% have an EPS estimate (so a surprise) and 95% a revenue surprise
- report times are almost all known: 58% before the open, 42% after the close

**The reaction:** each stock's move beyond the 13 factors on the first session that can react, in units of its own normal daily noise.

**1. Earnings are real events by default, unlike most macro headlines.**

| | Earnings reaction day | An ordinary day |
|---|---|---|
| Typical size (RMS, × normal noise) | **3.9×** | 1× |
| Moves bigger than 2× normal | **55%** | 4.6% |
| Bigger than 3× | 39% | 0.3% |
| Bigger than 5× | 18% | ~0% |

For a holder of a single stock, its own earnings are almost always the biggest scheduled risk on the calendar. That's the opposite of macro headlines, where about 90% of events are noise (#162, #192).

**2. The surprise calls direction, but less and less.**

| Sign of the surprise vs the move | before 2023 | 2023–25 | 2026 |
|---|---|---|---|
| EPS surprise, all reports | 60% | 58% | 54% |
| EPS surprise, moves > 2× | 66% | 61% | 58% |
| EPS and revenue agree | 63% | 61% | 55% |

- The rank correlation between surprise and move falls from 0.26 to 0.18 in 2026.
- Markets increasingly trade the guidance and the call, not the headline number. A text read of the release and call would be the next lever (S0b).

**3. The size of the surprise doesn't predict the size of the move.** Every surprise quintile averages about 3× normal noise. A big move comes with or without a big beat or miss.

**4. No drift afterwards.** Over the next 5 sessions the move continues by +0.01× noise on average (+0.02× after moves above 3×; 50% continue, 50% reverse). In 2026, big moves gave back slightly (−0.12×). There's nothing for a user to chase after the report.

**5. Headlines are plentiful.** Benzinga has about 1,000 headlines for NVDA, AAPL or TSLA in 2–4 months, and for MRNA about 330 a year. Labelling every headline isn't affordable under the budget; S0b must filter first (big moves, importance, channels).

## What it means for the product

- **Before earnings, the useful number is size, not direction.** The question is "how big is this stock's earnings move usually, and is that more or less than options price?" (S2): this stock's past reactions vs the implied move.
- **After the report, direction is about a coin flip plus a little** (55–58% on 2026), and there's no drift to chase. Worsey should say that plainly.
- **For a book,** the stocks reporting in the next weeks, weighted by position size and counting ETF look-through, are the single-name risk calendar ("NVDA reports Nov 19: its moves average ±7% and options price ±6%; that's 10% of your book counting QQQ").

## Next

- **S2:** the implied move before earnings (option history from about 2015) vs realized; did buying or selling the straddle pay, net of costs, on held-out reports?
- **S0b:** non-earnings corporate news (trial readouts, M&A, guidance, legal), filtered to news that coincided with large residual moves, then LLM-labelled by type, under a dry-run cap.
- **Product row:** the earnings calendar for the user's book, with typical move vs implied move.
