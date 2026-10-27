# Results summary: what we tested, what held up

Brief. Every number comes from a held-out test (the model never saw the events it was scored on); details and error bars are in `docs/pmrm/results/` and DECISIONS. Last updated 2026-10-25 (through #231). Moved here from `docs/results/RESULTS_SUMMARY.md` on Oct 25.

## The question (take-home Q1)

"What could hit my portfolio, and how could I hedge it?", using prediction markets where needed.

## The arc in four lines

1. **Daily prediction-market odds don't predict portfolios.** Held-out R² < 1%, mostly market exposure (P0c, #153).
2. **Real events, measured over decades, do move factors, but mostly only big events** (#162, #171). A typical event is about 3% of its window's move; extreme events about 80%; the first 2 hours after a big event's first report about 80–87% (#176).
3. **A 13-factor map of any portfolio works** (62.7% of daily moves held out, #163). For unusual portfolios it's essential: a perfect market forecast explains 37% of their event moves, the factors 77% (#178).
4. **Against just asking an LLM, the system wins on honest risk ranges, not on prediction** (#185, #186).

## The headline comparison (#185, #186)

92 news events of 2026, after the LLM's training cutoff; 12 portfolios of $100k (SPY, 60/40, tech, banks, energy, agriculture, gold miners, Latin America, utilities + REITs, Treasuries, commodities, a demo book); the same Claude model as the naive arm.

| | Worsey (system) | Naive LLM | "Nothing happens" + normal noise |
|---|---|---|---|
| Real moves inside the "90%" range, portfolios | **90%** | 77% | |
| Real moves inside the "90%" range, single positions | **90%** | 73% | |
| Interval score, portfolios ($, lower is better) | $752 | $781 | **$721** |
| Interval score, summed over positions ($) | $803 | $839 | **$769** |
| System better than the naive LLM | $29 (98% support) | | |
| …summed over positions | $37 (100% support) | | |
| Direction of the move right | 54% | 53% | |

- **Biggest gap on unusual portfolios:** gold miners, Latin America, commodities, energy (summed positions: $53 better, 100% support). The LLM's "90%" ranges held 49% of real moves on gold, 53% on commodities, 58% on gold miners.
- **Where the naive LLM is better:** steady holdings (SPY, Treasuries, high yield, real estate, banks), where our ranges are too wide.
- **What no approach does:** predict which way news moves a holding (about a coin flip), or beat "normal noise" on average.

## Supporting results

| Test | Result | Ref |
|---|---|---|
| Are our ranges calibrated? (1,773 events × 11 portfolios) | Yes: 50% inside the middle half, 90–91% inside the 9-in-10 range, 3.6–3.7% below the bad case | #183 |
| LLM adjusting the engine's numbers | Made it worse: −7.3% vs engine +1.8%. Now the LLM never touches numbers | #180, #182 |
| LLM with our evidence vs raw LLM (factor calls, post-cutoff) | Assisted +2.5% vs raw −5.7%; blinded big events +5% vs −20% | #170, #176 |
| Kalshi for scheduled releases | Kalshi surprise tracks the rates reaction (0.52–0.54 vs 0.12–0.34 naive); Kalshi-weighted damage range calibrated 93% / 51% | #172, #174 |
| Does the first move stick? | On average yes (market close ≈ 1.17× first move), but about 1 in 4 big events gives back half by the close | #177 |
| Skill beyond noise at the factor/ETF level | Only big events, a few factors: US stocks +3.7% (+9% in 2026), utilities vs energy +2.3% (+6.5%), dollar +11% (2026); none on targeted holdings | #184 |
| Hedging (real option prices, 4,727 book-events) | Always hedging costs about $130–150 per $100k per event and pays about 3 times in 10, so the default verdict is "probably not worth hedging"; extreme events and a user's bad-case weight flip it | #199–200 |

## Added since #186

| Test | Result | Ref |
|---|---|---|
| Plan C: a learned event → move model | Text embeddings lost to noise (#197). The real-event gate + the LLM's direction is the best forecaster: 2026 −0.8% vs noise, 98% support (#194); it runs daily as a frozen shadow on the prospective log (#201) | #187–201 |
| Single stocks | Real effects: earnings days move 3.9× a normal day; analyst rating changes get the direction right 77–80% | #202–206 |
| Per-position "what could hit me" | Every risk row shows which holdings drive it | #207 |
| Black-swan mode (46 anonymized extreme events) | The raw LLM's shock calls have 33–39% skill; anchoring on dissimilar past events hurt, so the shock is shown as an editable assumption | #208 |
| Options | Full repricing (spot, IV via the measured spot-vol response, time) with an exact spot/vol/time split. Taylor Greeks fail on big moves: the trainer's oil puts at oil +40%, Greeks +$1.8k, truth −$1.5k | #209 |
| Targeted sectors of macro events | No edge on held-out years | #211 |
| Commodity releases and weather | WASDE moves corn 4.6× a normal hour; EIA gas 1.65×, crude 1.34×; hurricane landfalls barely move markets (forecasts price them) | #212 |
| Live labelling prompts (745 major and extreme events, text only) | Direction on big moves: market labeller 74%, user-event labeller 73% (research prompt 75%); size calls ordered (small 0.90×, medium 1.55×, large 2.70× noise); news labeller weaker (65%, factors undefined in its prompt) | #236 |
| "What if" size calls | The LLM picks direction and small/medium/large per factor; measured: large = 2.4× normal noise (2.0× on 2023–26) | #213–214 |
| Sector layer (19 sectors, one per holding) | Stock R² 24% → 33% | #215–217 |
| "Noise is an answer" | Every scenario gets a noise / small / meaningful verdict for the book, and the event itself is graded ("a real event, but your holdings are mostly out of its path") | #219–220 |
| The event universe (part I) | Markets + calendar + news (what's coming) matched with LLM proposals from the book's exposures; an exact market existed beforehand for only 5 of 31 big 2026 events (14 more on the same story) | #221–225, #231 |
| Longshot bias (part II, tier 1) | 19,792 resolved Kalshi + Polymarket markets: prices of 2–35% overstate YES (7% → 3.7%), 90–98% understate it; the correction cuts out-of-sample log loss 1.7% | #227 |
| Matching (part II, tiers 2a/2b) | Relation labels 34 of 35 right; bounds checks hold; ladders and deadline rungs interpolated, never extrapolated | #228 |
| Forecaster when no market fits (part II, tier 3, pilot) | 29 event questions, Brier: naive LLM 0.167, Worsey 0.139, market 0.143; Worsey − naive = −0.028 [−0.060, −0.0001] | #229–230 |

Part II in full: `docs/event_pipeline/part2_summary.md`.

## The takeaway

- **The LLM alone is confidently wrong about risk.** Its ranges miss about 1 in 4 times, and about half the time on unusual holdings.
- **Worsey gives each position an honest range** (90% means 90%) from a factor map of the actual portfolio, and says "this is noise" when history says so.
- **Nobody predicts direction from headlines**, so Worsey doesn't pretend to. The LLM identifies the event and explains; code produces every number.
- **Probabilities are honest too:** market odds corrected for a measured longshot bias; derived from nearby markets when only the strike or deadline differs; "no market fits" said plainly, with a small forecaster that beat a naive LLM on a pilot.
- **Next:** the forecaster backtest (~250 questions, budget pending) and probabilities with tier labels in the chat.

## Spend

LLM API: about $72.90 of the $80 project cap (all research and chat through #231).
