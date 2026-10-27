# Matching an event to a prediction market, and deciding nothing fits

**The question:** prediction markets rarely ask exactly the question you care about (say, tariffs of 25% or more by March 31), so this page shows how events are matched to markets and when nothing fits.

Current through DECISIONS #234. Code: `src/worsey/planb/probability.py` (matching, derivations, bounds) and `src/worsey/discovery.py` (`tier_probabilities`, `forecast_probabilities`, `_set_prob`, `choose_three`). Written from the code (Oct 25).

## The one-line version

> Code finds the candidate markets. One LLM call says only how each one relates to the event, logically, with no numbers. Code turns that into a probability: the market's price if it's the same event, derived from the ladder if it's a different strike or date, bounds if one logically implies the other. If nothing is the same or logically related, we say no market prices it.

## Which events go through matching

- **Listed markets** (Kalshi, Polymarket) don't need matching: they *are* markets. They get tier 1 directly, the longshot-corrected price.
- **Proposals and your own events** go through the steps below. These are the events nobody lists exactly.

## Step 1. Find candidate markets (code) — `candidates_for()`

For each event, take 6 candidates from the ~150 priced markets in this book's universe:
- the **top 4 by meaning**: a small local embedding model (bge-small) on market titles, compared with the event's title
- the **top 2 by shared words** in the title

Two more things go in:
- **The proposal's own catalyst link, if it named a market.** It's checked like any other candidate, never taken as the answer.
- **Up to 12 of this week's news stories**, so a proposal that *is* a listed story can be absorbed rather than shown twice.

*Why both meaning and words:* word overlap alone missed obvious matches ("AI bubble bursts" vs "AI bubble burst in 2026?"), fixed in #234.

## Step 2. Classify the relationship (one LLM call for all events) — `classify_many()`

The prompt says: "Logic only: do NOT estimate any probability." For each event–market pair, the LLM picks one:

| Relation | Meaning | Example |
|---|---|---|
| `same` | the market's YES is exactly the event: variable, threshold, deadline and resolution all match | "AI bubble bursts in 2026" vs "AI bubble burst in 2026?" |
| `other_strike` | same variable, different threshold (it gives the event's threshold) | "CPI ≥ 0.4%" vs a market on "CPI above 0.3%" |
| `other_deadline` | same event, different deadline (it gives the event's date) | "…by Dec 31" vs "…by Nov 30" |
| `E_implies_M` | if the event happens, the market's YES must happen | "US strikes a base near Tehran" → "US strikes Iran" |
| `M_implies_E` | if the market's YES happens, the event must happen | the reverse |
| `exclusive` | both can't happen | "Hormuz deal signed" vs "no Hormuz deal" |
| `correlated` | related, but no logical implication | "Saudi pipeline struck" vs "oil above $120" |
| `unrelated` | | |

The examples are illustrations, not logged outputs.

## Step 3. Turn the relationship into a probability (code) — `tier_from()`

Tiers are checked in order. The first that applies wins.

1. **Tier 1, exact market.** Any candidate is `same` → its price, **longshot-corrected**.
2. **Tier 2a, derived from a ladder or deadline.**
   - **Other strike** → `derive_strike()`: interpolate the market's own ladder of "above k" contracts, or of its exact buckets summed into a survival curve, at the event's threshold. Never extrapolated beyond the listed strikes.
   - **Other deadline** → `derive_deadline()`: interpolate between the market's deadline rungs. Probability can't fall for a later date. With a single rung, use a constant hazard, only within 3× the time; further than that, no answer.
   - The derived number is longshot-corrected too.
3. **Tier 2b, bounded by related markets** → `bounds()`:
   - event implies market: P(event) ≤ P(market)
   - market implies event: P(event) ≥ P(market)
   - exclusive: P(event) ≤ 1 − P(market)

   Code intersects all of them. If the interval is **10 points wide or less**, its midpoint is used. If the quotes contradict each other, the overlap is reported.
4. **Otherwise, nothing fits.** No candidate is the same event, none derives, and the bounds are wider than 10 points. The event goes to tier 3 carrying its bounds and up to 3 related markets as evidence.

## Step 4. When nothing fits — `forecast_probabilities()`

- **Who gets a forecast:**
  - every one of **your** events still without a probability (you asked about it)
  - plus the **3 largest** of Castle's unpriced proposals that pass the size screen (at least half the book's normal 2-day swing)
- **The forecaster:**
  - 3 agents, each seeing the library base rate, the 8 most similar past events, the last 14 days of news and the related markets, in different orders
  - aggregated by the median in log-odds
  - clipped to [2%, 98%] and to any tier-2b bounds
- **Shown as** "Worsey forecast — no market lists this". The priced-in split carries "assumes markets share this forecast".
- **Everything else** keeps a **labelled base rate**: how often big events of that kind happened. It's context only, not the event's odds, and it isn't eligible for the "three to watch".

## The longshot correction (tiers 1 and 2a) — `corrected()`

- **Fitted on** 88k calendar snapshots of 19.8k resolved Kalshi and Polymarket markets, 2023 to June 2026 (#227).
- **Both venues overprice longshots and underprice near-certainties.** Prices of 2–35% overstate YES; 90–98% understate it. The bias held in the held-out year.
- **Correction:** logit(p) = −0.10 + 1.17 · logit(price), applied **only in the bins where the bias was significant**. Raw elsewhere. Out-of-sample log loss improves 1.7%.
- **Examples from the live check (#233):** the 5-year yield at 5.25% moved from a 21% price to 15.9%; gas above $5.80 moved from 10% to 6.4%.
- **Which number feeds what:**
  - the **if-yes / if-no split uses the raw price**, because that's what asset prices carry
  - **risk left uses the corrected probability**

## Evidence and honest limits

- **The relation labels:** 34 of 35 right on test pairs (#228), and the bounds held. *These pairs were labelled by the build assistant; review a few before quoting the number.*
- **Coverage (#224):** for the 31 big 2026 events, an exact market existed beforehand only 5 times, and a market on the same story 14 more times. That's why tiers 2 and 3 matter: markets usually price the story, not the specific shock.
- **Not tested end to end:** whether the tier-2b midpoints are calibrated, and whether the retrieval ever misses the right market. The forecaster's test so far is the 29-question pilot (Brier 0.139 vs naive 0.167, market 0.143).

## Say it in the room

> "Matching is mostly code. We retrieve candidate markets by meaning and by words. One LLM call labels the logical relationship for every pair: same event, different strike, different deadline, implies, implied by, exclusive, correlated. It never gives a number. Code does the rest: an exact match uses the corrected market price, a different strike or date is read off the market's own ladder, and a logical relationship gives bounds. 'Nothing fits' has a definition: no same event, no derivation, and bounds wider than ten points. Then we say no market prices it, and only the biggest of those go to the forecaster."
