# The fair hedge test: when is protection worth the premium?

Run 2026-10-23. Script: `scripts/planc_hedge.py` (`build`, `naive`, `score`; the design was fixed before scoring, DECISIONS #195). Raw numbers: `planc_hedge.json`. Cost: $0.81 (the naive LLM arm, 92 calls). DECISIONS #199.

: "The powerful thing about 'being honest about noise' is going to be the hedge component. Telling the user when they ACTUALLY should spend premium to hedge and when (probably most of the time) it is not worth it."

## In plain words

**The test.** For every news event of 2023–2026 (576 events) and each of the 12 portfolios ($100k each), 4,727 book-events in all:
- **The hedge:** a protective put about 3% out of the money on the liquid ETF most correlated with the portfolio (SPY, QQQ, IWM, EWZ, USO, TLT, GDX...), sized to the portfolio's beta. Every approach uses this same put, so only the decision to buy it is scored.
- **Real option prices** from Massive, plus a 3% spread each way. Bought at the close before the news, sold when the event window closes 1–2 sessions later. That timing favours hedging, since the protection is in place before the news.
- **Approaches:**
  - never hedge
  - always hedge
  - hedge when options look cheap
  - our system: hedge when our event forecast is wider than what the option already prices
  - the naive LLM: Claude with no tools, told the event, the book and the exact put with its premium (2026 only)

**1. Hedging every event is expensive.** On average:

| | 2023–25 | 2026 |
|---|---|---|
| Premium per hedge | $948 | $1,243 |
| Net result of the hedge (what it returned minus what it cost) | **−$131** | **−$150** |
| Share of hedges that paid for themselves | 29% | 30% |
| Average of the worst 5% of outcomes, no hedge → always hedge | −$3,603 → −$2,660 | −$5,068 → −$3,522 |

- About $59 of each $131–150 is the bid-ask spread; the rest is time decay and the premium sellers earn.
- **When a portfolio does take a loss beyond its normal bad case** (2.6% of book-events, average loss −$4,332), the hedge returns +$1,428, covering about a third of it.
- **Hedging everything only pays if you weigh a loss beyond your normal bad case 3–6× more than everyday P&L** (break-even 6.4× on 2023–25, 3.0× on 2026).

**2. Every disciplined rule learned "almost never hedge".**
- Tuned on 2023–25 (weighting losses beyond the normal bad case 1×), the system rule hedged 1% of book-events and the "cheap options" rule 0%. On 2026 both hedged nothing.
- The naive LLM, told that protection usually costs more than it saves, hedged 0.3% of the time (15 of 1,104). 10 of those 15 paid; the gain is +$2 per book-event with 82% support, too few cases to call.
- **Never hedging beats always hedging** on 2023–25 and 2026 at a 1× weight (always is worse by $99–111 per book-event, 0–4% support). At a 3× weight it's even on 2026 and still worse on 2023–25.

**3. Where protection comes closest to paying:**

| Group | Net hedge result | Hedge paid | Loss beyond normal bad case |
|---|---|---|---|
| **Extreme events (12)** | **−$32** | **48%** | 9% |
| Major events | −$114 | 32% | 3% |
| Moderate and minor | −$143 to −$146 | 27–28% | 2% |
| Options cheapest vs realized vol (top fifth) | −$60 | 34% | |
| SPY, TLT, EEM, XLU puts | −$54 to −$80 | | |
| GDX, XLE, EWZ, XLF puts | −$208 to −$313 | | |

- Our forecast's width does pick out the risky book-events: the widest fifth sees a loss beyond the normal bad case 6% of the time vs 1–2%. But those are also the books with the most expensive protection (gold miners, Brazil), so hedging them still loses.
- **Extreme events only** (picked after seeing the table above, so exploratory): roughly even on 2023–25 (8 extreme events) and +$18 per book-event on 2026 (4 events, 84% support; the hedges returned +$239 each). Too few events to rely on; it goes on the prospective log as a rule to track.

**What it means for the product:**
- **The default answer is "don't hedge, and here's why":** "your normal 2-day range is ±$X; protection for this event costs $Y, and puts like this have paid for themselves about 3 times in 10."
- **The exceptions:**
  - extreme events, where hedging is close to break-even and the bad case is several times likelier
  - users who weigh a bad-case loss much more than everyday P&L (for example, someone who must not lose more than $Z this week). That weight is the user's call, and the product should ask for it, not assume it.
- **Cheaper protection matters more than better timing.** SPY and Treasury puts cost about $60–80 per event; sector and country ETF puts $200–300. When a user does hedge, the hedge comparator should prefer the cheapest instrument that covers most of the risk.
- **This is the measurable value of "honest about noise".** On 2026 it saves about $150 per $100k per event versus reflexive hedging, while keeping the option to hedge when an event is truly extreme.

## Method notes

- **Instrument:** the most correlated of 13 liquid ETFs over the 252 sessions before (correlation > 0.2); beta-weighted notional.
- **Option:** the first monthly expiry at least 14 days after the window, strike nearest 97% of spot. Daily closes ± 3% of premium + $0.01; positions need a close on both days.
- **Implied vol:** backed out from the put's price (Black-Scholes, r = 4%).
- **System signal:** ρ = R × realized/implied, where R is the book's forecast sd ÷ its normal sd from the C1c gate's factor quantiles (rung A, the best of A/B/C on validation; D is within 0.1%), weighted by each factor's share of the book's normal variance.
- **Thresholds:** chosen on 2023–25 at λ = 1 (system 1.7, cheap options 1.55).
- **Utility:** P&L − λ × max(0, normal bad case − P&L), with the normal bad case = −1.645 × the book's normal sd over the window. Paired bootstrap over events (2,000 resamples).
- **Naive prompt:** in the script; it says protection usually costs more than it saves, which is fair (it's true) and pushes the LLM toward the same default.
