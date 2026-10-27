# Single-stock track, S2: do options price earnings moves fairly?

Run 2026-10-23. Script: `scripts/stock_s2.py` (`build`, `score`). Raw numbers: `stock_s2.json`. No LLM calls, $0. DECISIONS #203.

## In plain words

**The test.** For a sample of earnings reports from 2015–2026 (every 2026 report, 1,500 from 2023–25, 1,000 from 2015–22), 2,718 had prices for both legs:
- **The straddle:** the call and put nearest the money on the first expiry after the report, bought at the close before the reaction and sold at the reaction-day close, at real option closes with a 3% spread each way.
- **Implied move:** the straddle's price ÷ the stock price, i.e. what options charge.
- **Realized move:** the stock's actual move on the reaction day.
- **History:** the median move of the stock's previous 8 reports, known beforehand.

| | 2015–22 | 2023–25 | 2026 |
|---|---|---|---|
| Median implied move | 4.8% | 5.4% | 6.4% |
| Median realized move | 2.9% | 3.7% | 4.1% |
| Realized bigger than implied | 28% | 34% | 34% |
| Buy every straddle (P&L per $ of premium, after costs) | −11% | −7% | −6% |
| Sell every straddle | −2% | −6% | −7% |

**What it shows:**
1. **Options price earnings moves about right.** After a realistic spread, neither buying nor selling every straddle pays. The typical realized move is about two-thirds of the implied one, but that partly reflects the straddle running to the first expiry after the report: its price includes more than the one-day move.
2. **A stock's own history doesn't say when options are cheap.** The rule "buy when past moves are big relative to the implied move" (threshold chosen on 2015–22) breaks even on 2026 (+0.2%, 50% support) and loses on 2023–25 (−11%). "Sell when they're small" loses on both (−4% and −6%, 0–2% support).
3. **Implied moves have grown:** 4.8% → 6.4%, and realized moves with them.

**What it means for the product:**
- **Before a report, the options' implied move is the best single number for "how big could this be?"** Worsey should show it, with the stock's own history as context ("options price ±6%; its last 8 reports moved a median 4%"), and say plainly that history alone doesn't find cheap or dear options.
- **For a user with conviction on direction,** the trade-off is the premium vs a move about the size options already price. The options desk's existing tools apply (Option 4).
- **For a holder,** the same "honest about noise" logic as macro hedging (#199): protection through earnings costs about what the move is worth, so it's a choice about how much the user minds a big single-name move, not an edge.

## Method notes

- **Expiry:** the first Friday-listed expiry on or after the reaction day (Massive contract lists); strike nearest spot common to calls and puts.
- **Costs:** each leg's close ± 3% + $0.01 (buy), mirrored for selling.
- **Missing:** 31% of sampled reports lacked a contract list or a close on both days.
- **Thresholds:** chosen on 2015–22 (at least 50 reports per side).
