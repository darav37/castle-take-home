# One owner for odds, three picks, and messages as the scan goes (DECISIONS #234)

Task: the design brief, as agreed with the trainer on Oct 25: assessment first; the output is the user's events plus three Castle picks, as text messages in succession; the pipeline otherwise unchanged.

## What changed

- **One owner for odds:** a proposal's market link is now the catalyst and a hint for the relation check, never its odds.
  - Every proposed and user event gets its probability from the tier step (one batched relation call).
  - Its candidates: the linked market, the top markets by meaning (local embedder on market titles) plus the top 2 by title words, and the week's news stories.
- **Dedupe in code:** a proposal that the relation call finds IS a listed market (tier 1) or a listed news story absorbs it (`absorbs`). Same-move overlap alone doesn't dedupe, because it would merge distinct events such as a pipeline strike and gas above $5.80.
- **Topical vs tail in code:** topical = tier 1, 2a or 2b, or a calendar or news catalyst; tail = the rest.
- **Forecaster:**
  - every unpriced user event, always (the user asked; the noise verdict says so honestly)
  - plus the largest 3 unpriced proposals above the size screen (`WORSEY_FORECAST_MAX_PER_SCAN`)
  - related markets (correlated, broader, narrower) go to the agents as labelled evidence (PART2 tier 3; untested in the pilot)
- **Castle's three (`choose_three`):** ranked by risk left, |move| × √(p(1 − p)), among events with a real probability (a market tier or a forecast). A pick is skipped if it's the same move as one already chosen. The rest are counted ("I looked at N more").
- **Messages (`scan_messages.py`):**
  1. your book at a glance
  2. your events, with an honest verdict
  3. "researching the odds", naming the events
  4. the analyst's last message: your events' probabilities, the three to watch, what's noise, how many more

  The watchlist card is no longer sent; the risk card stays.
- **Retrieval fix:** word overlap over titles + rules missed obvious markets ("AI bubble bursts" never retrieved "AI bubble burst in 2026?"). Retrieval by meaning fixed it.

## Live checks

| Question | Messages arrive at | Final | Cost |
|---|---|---|---|
| "I am worried about an oil spike, and also anything else…" | 21 s (book), 49 s (your events), 73 s (researching) | 95 s | $0.29 |
| "What could hit my portfolio?" | 22 s (book), 71 s (researching) | 92 s | $0.27 |

Before this change, the same scan took 175 s and showed one card with up to 13 events.

- **Your event:** an oil spike costs −$1,715 if it happens (USO puts −$710, then TSLA and WDC).
  - The Worsey forecast was 72–80% across the two runs; it's a vague event over 91 days.
  - The gas above $5.80 and Iran-nuke markets are folded under it.
- **The three (plain scan):**
  - the 5-year yield at 5.25%: −$3,530; 15.9%, corrected from 21%
  - an AI bust: −$3,646; Worsey forecast 9%
  - a Hormuz deal: +$2,449; 26.4%, corrected from 32%
  - 42 more were looked at
- **Finding:** the relation call is strict.
  - "AI bubble bursts: chip and storage names crater" vs Polymarket's "AI bubble burst in 2026?" is *correlated*, not the same (different resolution and deadline), so it goes to the forecaster.
  - The forecaster now gets that market as evidence. Across two runs, the forecast was 22% without it and 9% with it.
