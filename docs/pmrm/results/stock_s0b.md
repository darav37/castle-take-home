# Single-stock track, S0b: what company news does to a stock

Run 2026-10-23. Script: `scripts/stock_s0b.py` (`fetch`, `build`, `audit`). Raw numbers: `stock_s0b.json`. No LLM calls, $0. DECISIONS #205.

## In plain words

**The data:**
- Benzinga's tagged news, 2019 to Oct 2026, in 13 channels (FDA, M&A, guidance, upgrades, downgrades, legal, offerings, buybacks, management, rumors, contracts, dividends, initiations)
- kept: items naming one or two S&P 500 companies, away from earnings (not within a session of a report): **60,909 events**
- events are **selected by the news, not by big moves**, so there's no hindsight in which events count
- direction comes from the channel (upgrade, downgrade) or title keywords fixed before scoring (approval vs rejection, guidance raised vs cut, dividend raised vs cut, buyback, equity offering, initiation at buy vs sell)

**Timing matters.** An item published during the session sits in a session that also holds the hours before it, and analysts often write after a move. So the main numbers use only items published **before the open or after the close**, where the whole reacting session follows the news.

| News type (clean timing) | Events | Typical move (RMS, × normal day) | Headline direction held | …2026 |
|---|---|---|---|---|
| Guidance change | 1,850 | **2.65×** | 65% | 70% |
| M&A | 4,261 | 2.22× | (channel doesn't say target or buyer) | |
| Buyback | 688 | 2.14× | 73% | 60% |
| **Analyst downgrade** | 5,059 | 1.89× | **80%** | 77% |
| FDA | 2,217 | 1.81× | **51%** | 55% |
| **Analyst upgrade** | 5,533 | 1.75× | **77%** | 75% |
| Rumor | 1,752 | 1.66× | | |
| Offering | 951 | 1.65× | 66% | 62% |
| Management change | 2,866 | 1.60× | | |
| Legal | 2,404 | 1.28× | | |
| Dividend change | 2,070 | 1.24× | 66% | 57% |
| Initiation | 5,733 | 1.14× | 67% | 55% |
| Contract | 3,180 | 1.12× | | |
| (an ordinary day) | | 1.0× | | |

**What it shows:**
1. **Company news moves stocks less than earnings do** (1.1–2.7× a normal day vs 3.9× for earnings, #202). Guidance changes, M&A and buybacks are the biggest.
2. **Analyst rating changes are the one reliable direction call:** pre-open upgrades and downgrades go their way 77–80% of the time (about 90% on big moves), steady from 2019 through 2026. But they're priced at the open: by the time a user reads them, the move has happened.
3. **FDA headlines are a coin flip on direction (51%).** Approvals are usually expected and priced in; the surprises go both ways.
4. **Guidance changes partly reverse:** the next 5 sessions give back about 0.1–0.16× normal noise.
5. **Otherwise no drift:** after the news day, nothing consistent to chase.

## In the product

- **`news_base_rates(news_type, ticker)`** (Plan B experts and router): the typical size of a move for that kind of news, in this stock's own percent and the user's dollars (look-through), how often the headline's direction held (overall, 2026, big moves), drift after, and the caveat that the move is mostly priced at the open.
- **The LLM only classifies the news type;** code gives every number.

## Method notes

- **Fetch:** 3,705 channel × 10-day windows (14 hit the 1,000-item page limit, so those few are undercounted).
- **Reacting session:** before 16:00 ET → that session; after → the next.
- **Reaction:** the move beyond the 13 factors (betas over the 252 sessions before), ÷ the stock's residual daily sd.
- **Dedup:** one event per channel, company and reacting session.
