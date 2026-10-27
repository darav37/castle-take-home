# One book, three questions: Worsey before and after the fixes, and the naive LLM

Run 2026-10-23 on data as of 2026-10-16 (the frozen build). DECISIONS #207–210. Cost about $1.40 for all runs.

**The book (about $100k):** PFE $20,000, WDC $15,000, TSLA $20,000, URA (uranium ETF) $15,000, GLD $20,000, and 2 USO put contracts, strike $157, about 3 months to expiry (USO at $157.31; about $1,996 of premium).

**The three answers compared:**
- **Worsey before:** the chat as it was before #207–209. It couldn't hold options, so the puts went in as their delta-equivalent short in USO.
- **Worsey after:** with the per-position split (#207), black-swan mode (#208) and real option positions (#209).
- **Naive LLM:** the same Claude model with only the holdings text. No tools, no data.

## Summary

| Question | Worsey before | Worsey after | Naive LLM |
|---|---|---|---|
| What could hit me? | Book totals only ("the card doesn't split that by holding") | Each risk split by holding (WDC −$2.4k, TSLA −$2.4k, URA −$1.4k; puts +$780 in the financial-stress bad case) | 5 scenarios × 6 positions, rich narrative; numbers are guesses (WDC −40% in a growth scare) |
| What if Hormuz closes? | −$79 "basically noise" (history averages it down) | Both modes: history −$46; black-swan stress (oil +40%, stocks −5%) −$5.2k, by holding, puts −$1.4k (loss capped near the $2k premium) | Two cases (−$3k / −$10k) by holding, put mechanics right; sizes unverified |
| When do my stocks report? | Real dates, look-through, options-implied dollars | Same, and it says direction after a report is close to a coin flip | Guessed dates (WDC "Oct 28–31", actually Oct 22), sizes from memory |

**Reading it:**
- After the fixes, Worsey answers each question by position and in total, with every number from code, including the options (fully repriced) and the black-swan case (the LLM's shock is shown as an assumption, and code maps it).
- The naive LLM still writes the richer story. Its numbers are unchecked, and its dates are guesses.
- **A single example can't show the measured advantage:** calibrated ranges (90% of real outcomes inside "9 in 10", against 73–77% for the naive LLM over 92 events × 12 books, #185–186). Present the two together.

## Q1. What could hit my portfolio? Tell me what it would do to each position and to the whole portfolio.

### Worsey, before

**Biggest hit: financial stress, with a 1 in 10 bad case of -$4.5k on your $76.2k book.**

The card doesn't split that by holding, so I can't give per-position numbers yet. The worst past example is 2020-03-11, the COVID pandemic declaration, which would cost -$13.3k today. A surprise central-bank move has a bad case of -$2.5k. The 13 factors explain about 73% of your risk, so these numbers are partial.

Want me to price financial stress as a scenario to get the split by position?

### Worsey, after

**Financial stress is the big one: about -$4.5k in the bad case (1 in 10) on your $92k book.**

WDC, TSLA and URA carry most of it, at -$2.4k, -$2.4k and -$1.4k. Your USO put adds +$780, so it hedges a little. The worst past example, 2020-03-11, would cost -$13.1k today. The factors only partly cover this book, so some risk sits outside them.

Want me to price a financial-stress shock in detail?

### Naive LLM

# Portfolio risk map

I can't see live prices or news after my training data, so these are rough scenario estimates built from typical sensitivities, not forecasts. Please verify the numbers against your broker.

## How the portfolio is built

| Position | Size | Main driver | Rough beta to S&P |
|---|---|---|---|
| PFE | $20k | Policy, pipeline, defensive | ~0.5 |
| WDC | $15k | AI/data-center storage, memory cycle | ~1.7 |
| TSLA | $20k | Growth sentiment, EV demand, Musk/valuation | ~2 |
| URA | $15k | Uranium price, nuclear/AI-power theme | ~1.3 |
| GLD | $20k | Real yields, dollar, safe-haven flows | ~0 to 0.1 |
| 2 USO puts | $2k premium | Oil falling | negative delta |

- **Concentration:** TSLA, WDC and URA are 50% of the portfolio, and they tend to fall together in a risk-off or "AI trade unwinds" move.
- **Net beta:** about 1.0 on equity-like exposure, but with much higher volatility than the S&P. A typical 1-month 1σ move is roughly ±8–9% (~$8–9k).
- **The puts:** 200 USO shares of notional is about $31k, with delta around −0.45 (about −$14k of oil exposure). Max loss is the $2k premium (2% of the portfolio), and time decay is roughly $10–15/day now, accelerating later. **You hold no oil-sensitive assets, so this is a directional bet on oil falling, not a hedge.**
- **USO price:** $157 is far from where I last saw USO trade. I've taken it as given, but if the price or premium differs, rescale the put figures.

## Scenarios (approximate P&L over a few weeks)

| Scenario | PFE | WDC | TSLA | URA | GLD | USO puts | **Total** |
|---|---|---|---|---|---|---|---|
| **A. Broad selloff / AI unwind** (S&P −15%, oil −15%) | −7% / −$1.4k | −35% / −$5.3k | −30% / −$6.0k | −25% / −$3.8k | +3% / +$0.6k | +$3.0k | **≈ −$12.8k (−12.8%)** |
| **B. Oil spike +25%** (Mideast/supply shock, S&P −7%) | −$0.6k | −$1.8k | −$2.4k | +$0.3k | +$1.2k | −$1.9k (nearly worthless) | **≈ −$5.2k** |
| **C. Rates/dollar shock** (hawkish Fed, S&P −8%, gold −12%) | −$0.6k | −$2.7k | −$3.6k | −$1.8k | −$2.4k | +$0.7k | **≈ −$10.4k** |
| **D. AI-capex scare** (WDC −40%, TSLA −20%, URA −20%) | +$0.4k | −$6.0k | −$4.0k | −$3.0k | +$0.6k | +$0.6k | **≈ −$11.4k** |
| **E. Oil glut, equities fine** (oil −25%) | +$0.2k | +$0.3k | +$0.6k | −$0.5k | −$0.2k | +$6.0k | **≈ +$6.5k** |
| **F. Broad rally** (S&P +10%, oil +5%) | +$0.8k | +$2.7k | +$3.6k | +$1.8k | 0 | −$0.7k | **≈ +$8.2k** |

## Single-name risks

- **PFE (−15% ≈ −$3k):** drug-pricing or tariff policy, patent expirations, trial failures, weak guidance. It is the most defensive holding but not immune.
- **WDC (±30–40% ≈ ±$4.5–6k):** hyperscaler capex cuts, HDD/NAND pricing turns, competition from Seagate, and a stock that has likely run hard, so expectations are high.
- **TSLA (−35% ≈ −$7k):** deliveries and margins, robotaxi/Optimus disappointments, loss of EV incentives, Musk headlines. A single stock here can move ±10% on earnings.
- **URA (−25% ≈ −$3.75k, or +30% ≈ +$4.5k):** uranium spot and term prices, Sprott-type physical buying fading, Kazakhstan supply, and nuclear policy. Its link to the AI-power theme makes it correlate with WDC and TSLA more than you'd expect.
- **GLD (−15% ≈ −$3k):** a drop in gold after a big run, rising real yields, a stronger dollar, or slower central-bank buying. Gold falling alongside stocks (scenario C) removes your main diversifier.
- **USO puts:**
  - Oil rising, or just sitting flat, loses premium. A flat oil price still costs you about $1–2k by expiry.
  - USO tracks front-month futures, so contango/roll can make it diverge from spot oil.
  - Implied volatility changes move the option price independently of oil.

## Key takeaways

1. **The big risk is correlated growth/risk-off drawdown:** a −10% to −13% portfolio hit (about $10–13k) is plausible in a bad month, and the puts offset only about $3k of it.
2. **The puts hurt in the wrong scenario:** an oil spike from geopolitics tends to hit stocks and the puts together (scenario B).
3. **Gold is your only real offset,** and it fails in a rates/dollar shock.
4. **Worst realistic single-month case:** a combined AI unwind plus rates shock could reach −15% to −20% (−$15–20k), given the weight in TSLA, WDC and URA.

## Dates to check (verify these)

- Q3 earnings for TSLA, WDC and PFE, roughly late October to early November
- FOMC meeting, around Oct 27–28
- US midterm elections, Nov 3
- OPEC+ meetings
- Your put expiry, around mid-January 2027

If you tell me your goal (hedge vs. speculation) and risk tolerance, I can suggest ways to rebalance or hedge, such as sizing down TSLA/WDC, using index puts instead of oil puts, or setting stop levels.

## Q2. What if Iran closes the Strait of Hormuz? What happens to each of my positions and to my total?

### Worsey, before

**If Hormuz closes, code's middle is about -$79 on your $76.2k book, basically noise.**

By position: USO short -$55, URA -$16, WDC -$13, PFE -$12, TSLA +$6, GLD +$11. History says wars like this rarely move markets within days, and code expects little change in oil. The 9-in-10 range is -$3.8k to +$3.1k, and the worst nearby past event (Abqaiq attack, 2019) would cost -$819. There's about a 43% chance of losing more than $380. Partly covered (73%), so treat these as partial. Want to see what a hedge would cost?

### Worsey, after

**A real Hormuz closure would likely cost about -$5.2k on your $92k book, under my assumption (+40% oil, -5% stocks). Past events say almost nothing.**

Past cousins were much smaller and moved the book only about -$46 in the middle. My shock is an assumption, not tested on history. TSLA is -$2.2k, WDC -$1.4k and URA -$1.1k. GLD gains +$1.0k and PFE is flat at -$88. Your oil put loses -$1.4k despite the oil spike. Half that shock is -$3.0k, and 1.5x is -$7.2k.

Want to change the shock size, or look at what would cushion it?

### Naive LLM

# Hormuz closure scenario: rough estimates

These are illustrative ranges based on typical betas and past oil shocks, not forecasts. I can't see live market data, so the figures are only as good as the inputs you gave me.

## The main point
Your only oil-linked position, the USO put, **loses** if oil spikes. Your equities are risk-on assets that would likely fall. GLD is your only real offset. In effect you're **net short oil** going into a supply shock.

## Position by position

| Position | Value | Mechanism |
|---|---|---|
| **PFE** | $20k | Defensive, low beta. It would fall with the market, but less than most. |
| **WDC** | $15k | High-beta tech and AI-cycle stock, so it is hit hard in risk-off. Qatar supplies roughly a third of the world's helium, which matters for helium-sealed HDDs and chipmaking. Contracts and stockpiles probably cushion this at first. |
| **TSLA** | $20k | High beta. An "oil is expensive, so EVs" story helps, but risk-off and consumer-demand fears likely dominate at first. |
| **URA** | $15k | Mixed. Energy-security demand for nuclear helps, but miners are volatile and get sold in liquidity crunches. This is the widest range. |
| **GLD** | $20k | Classic safe haven, usually up. It can dip briefly if investors are forced to sell everything for cash. |
| **2 USO puts** | ~$2k | Oil up means USO up, so the puts go out of the money. Your loss is capped at the premium paid (~$1,996, about 2% of the portfolio). |

## Scenarios

| | **A: Brief or partial disruption** (oil +20%, S&P −3%) | **B: Sustained closure** (oil +60–100%, S&P −10–15%) | **C: Severe tail** (oil $150+, stagflation, S&P −20%+) |
|---|---|---|---|
| PFE | −2% (−$400) | −7% (−$1,400) | −10% (−$2,000) |
| WDC | −8% (−$1,200) | −22% (−$3,300) | −35% (−$5,250) |
| TSLA | −6% (−$1,200) | −18% (−$3,600) | −30% (−$6,000) |
| URA | +3% (+$450) | +5% (+$750) | −10% (−$1,500) |
| GLD | +4% (+$800) | +8% (+$1,600) | +3% (+$600) |
| USO puts | −$1,500 | −$1,996 | −$1,996 |
| **Total** | **≈ −$3,050 (−3%)** | **≈ −$7,950 (−8%)** | **≈ −$16,150 (−16%)** |

URA in B could plausibly range from −10% to +20%.

## Caveats
- **The put isn't a hedge.** It profits if oil falls, for example if a closure is quickly resolved and the risk premium collapses. In that case your equities would likely rally too, so it adds to your gains rather than offsetting losses.
- **Volatility cushions the put slightly.** A vol spike would keep it from going to zero immediately, but a big move up in USO still takes it close to worthless.
- **Path matters.** A spike followed by a fast reversal could leave you better off than the table suggests.
- **USO tracks futures, not spot.** Roll yield and backwardation can make it diverge from headline crude prices.
- **Check your USO inputs.** $157 is higher than I'd expect for USO, and I can't verify it. Please confirm the price, strike, and premium, because the put's behavior depends on them.

## If you want to hedge this risk
You could add exposure that benefits from higher oil: USO or XLE calls, energy producers, or a call spread. You could also swap the put for something that doesn't bet against oil. I can sketch sizing for that if you tell me how much downside in scenario B you're comfortable with.

## Q3. When do my stocks report earnings, and what's at stake for me?

### Worsey, before

**TSLA, WDC and PFE carry most of your earnings risk, and options price about $2,967 of total swing across all reports, either direction.**

TSLA reports Oct 21 ($20k held, $146 implied, but that quote looks thin, so trust history). WDC reports Oct 22 ($15k, $548). PFE reports Nov 3 ($20k, $861). The URA names are small.

Want me to look at hedging TSLA or WDC?

### Worsey, after

**Three reports hit in the next two weeks: TSLA Oct 21, WDC Oct 22, PFE Nov 3, with about $3.0k at stake at options-implied moves.**

TSLA ($20k) is priced for only 0.7%, but its history says 4.5%, so trust the history. WDC ($15k) is priced near 3.7% and has moved about 8.4% in the past. PFE ($20k) is priced at 4.3% against a usual 1.5%. Direction after a report is close to a coin flip.

Want me to show a hedge for TSLA or WDC?

### Naive LLM

I can't see live calendars, so these dates are estimates based on past reporting patterns. Confirm them on your broker or the companies' investor relations pages.

## Earnings timing (Q3 reports, estimated)

| Holding | Size | Likely report window | Notes |
|---|---|---|---|
| **TSLA** | $20k (20%) | ~Oct 21-23 | Usually the first of your three to report, possibly within a week |
| **WDC** | $15k (15%) | ~late Oct (around Oct 29-30) | Fiscal Q1 report |
| **PFE** | $20k (20%) | ~Oct 28 to early Nov | Has varied between late Oct and the first week of Nov |
| **URA** | $15k | No earnings | Holds miners and producers, e.g. Cameco and others. Cameco usually reports around early Nov, and Kazatomprom issues operating updates around then |
| **GLD** | $20k | No earnings | Driven by real yields, the dollar, and the Fed |
| **USO puts** | ~$2k | No earnings | Driven by crude prices, OPEC+, and inventories |

## What's at stake

About 55% of your portfolio ($55k) sits in stocks reporting within roughly 2-3 weeks. As a rough guide, typical earnings-day moves are:

- **TSLA:** ±6-8%, or about $1,200-1,600 on your position
- **WDC:** ±8-12%, or about $1,200-1,800
- **PFE:** ±3-4%, or about $600-800

If they all move against you, a 3-5% hit to the total portfolio ($3-5k) is plausible, and the tails can be larger.

**What to watch:**
- **TSLA:** auto gross margins, the timing and credibility of robotaxi/FSD, capex guidance, and the post-EV-tax-credit demand picture. Valuation rests on the story more than current earnings, so guidance and commentary often matter more than the numbers.
- **WDC:** hard drive demand from AI and data-center customers, pricing, gross margin, and guidance. The stock has run hard on AI storage enthusiasm, so expectations are high, and a good report can still sell off. Customer concentration is also a risk.
- **PFE:** whether guidance holds, how fast cost cuts are showing up, the post-COVID revenue decline, upcoming patent expirations, obesity-drug pipeline progress, and drug-pricing and tariff policy. It's the most defensive of the three.

## Other points

- **No earnings risk:** 45% of the portfolio (URA, GLD, USO puts) has none, which gives some diversification. URA does carry indirect earnings risk through its holdings' reports in early November.
- **The USO puts** are at-the-money with about 3 months left, so they're a bet that oil drops, and they lose value to time decay if it doesn't. They don't hedge your equity earnings risk. If you want protection around earnings, that would need separate hedges or smaller position sizes.
- **Check the USO price.** $157 is far above where I last saw it trade, so confirm the quote and your contract details.

I can sketch what a rough scenario table would look like, or how much a hedge around TSLA and WDC earnings would cost, if you'd like.
