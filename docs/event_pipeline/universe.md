# The event universe: which events could hit this book

> **Design document** (Oct 24, with a status block). For how the code works today, read [ARCHITECTURE.md](../ARCHITECTURE.md), sections 2 and 5.

**The question:** find the geopolitical and macro events most likely to move a portfolio. Part 2 (probabilities, matching to prediction markets, "already priced") runs in the same pipeline.

**Status, Oct 24, 2026:** built and live in "what could hit me" (DECISIONS #221 design, #222 news-feed fix, #223 build, #224 Polymarket and the coverage test). Code: `src/worsey/discovery.py`.

## The design

Two streams, matched:

```
 Stream A: what's actually coming (topical)       Stream B: what would move THIS book most (impactful)
 ─────────────────────────────────────────        ──────────────────────────────────────────────────
 A1 the calendar (Fed, CPI, jobs, GDP,            B1 the book's biggest exposures (factor $ per normal
    earnings, EIA, WASDE, US election)               day, sector $, top names incl. through ETFs, options),
                                                     and its history: the past-event replay's worst kinds (#231)
 A2 the news feed (last 7 days' labelled          B2 one LLM call proposes the 6 events that would push
    stories, "the story develops further")           those exposures hardest, linked to an A item when
 A3 prediction markets (Kalshi + Polymarket)         that is the catalyst; at least two tail risks
              │                                                   │
              └──────────────► MATCH ◄────────────────────────────┘
```

**Every candidate, from either stream, goes through the same code:**
- **impact on the book:** the "what if" engine. Directions and sizes (from a market's label, a story's label or a proposal) become factor moves, then each position's move through its loadings, plus sector and held-company legs, with options repriced.
- **probability:**
  - a market's odds when a market really is the event; the priced-in maths then applies (if YES: full × (1 − p); if NO: −full × p; risk left: |full| × √(p(1 − p)))
  - otherwise a labelled **base rate**: how often a big event of that kind happened within the horizon (2016–2025), or how often a story was followed by another big event within 30 days. It's context only, never used for the priced-in split, and it's labelled "not this event's odds".

**The match** (impactful = full impact ≥ half the book's normal two-day swing, the same rule as the noise verdict, #219):

| | Would move this book | Wouldn't move this book |
|---|---|---|
| **Topical / coming** | **Headline:** ranked by risk left (markets) or size | **Noise for you:** said plainly |
| **No live catalyst** | **Tail risks:** proposals linked to nothing | ignored |

## The user's events and Castle's (#225)

A run's universe has two sources. When the user names a worry and asks for anything else ("I'm worried about an oil spike, and anything else my book cares about"):
- **The user's events** (`user_events` on `scan_risks`) are labelled in the same proposal call, priced by the same engine, and always shown first with a verdict (noise / small / meaningful). A market supplies the odds only if its YES is exactly that event.
- **Castle's events** are everything else. The proposal call is told not to repeat the user's events or other routes to the same move, and code folds any market, story or proposal whose main move matches (same factor and direction, other than the broad market) under the user's event as "related".
- **Example:** the oil spike, −$1.9k and "small for your book", with gas above $5.80, WTI at $120 and Saudi airspace folded under it. Beyond it: the AI-bubble selloff (−$4.2k, 9.7%) and a hawkish Fed hold (−$2.1k).

## What the user sees (#234)

The user's events (always, with an honest noise verdict) plus Castle's three picks, chosen by risk left among events with a real probability (no two with the same main move), sent as messages as the scan goes: the book at a glance, the user's events, "researching the odds", then the final message with each probability and its tier. Everything else is counted ("I looked at N more"). A proposal's market link is the catalyst only; the tier step owns every probability, and duplicates are removed in code when the relation call finds that a proposal IS a listed market or story.

## The pieces

### A3. Prediction markets
- **Kalshi:** the built universe (`worsey build`, Oct 16), with quotes refreshed hourly in the live chat:

  | Stage | Markets left |
  |---|---|
  | All open Kalshi markets | 145,214 |
  | In-scope categories (sports and entertainment dropped) | 47,876 |
  | Resolve within 12 months | 14,656 |
  | At least 60 days of history | 6,912 |
  | Liquidity threshold | 1,248 |
  | Reflexive price-level markets dropped | 1,120 |
  | Top by volume, round-robin, plus the standing calendar | 327 |
  | Usable price history | **222** |

- **Polymarket (#224):** its open list is mostly sports, so it's read through topic tags (politics, geopolitics, economy, finance, world, business, tech, Fed rates, trade war). It takes the top 50 events per tag by 24-hour volume, markets with at least $50k of volume, and the 3 most-traded rungs per ladder. That gives about 300 markets, normalized to the Kalshi format and cached per hour (`data/polymarket.topical`).
- **Labels:** `scripts/discovery_label.py [--polymarket]`. The LLM labels each market once: what a YES does to markets. The labels are committed (`planb/market_labels.json`): 533 labelled, 310 move markets. A new market is unlabelled until the script reruns (cheap, and only new ones are labelled).
  - **Country legs (#238):** a label names countries only when the event is about that country or directly hits it (a war, a coup, sanctions, an election, a crisis), each with its direction, from the 31 in `planb/countries.json`. A holding's country loading (country ETFs, foreign stocks by home country, others only if they pass a held-out test) carries the move; the largest of sector, company and country leg is kept.
  - **Company legs (#232):** a label names companies only when the event is about that company (its drug, earnings, export licence, CEO); broad events reach companies through the factors and sectors. 31 of 533 labels name companies. A holding with both a sector leg and a company leg from one event gets the larger of the two, never the sum (`discovery.combine_legs`).
- **Duplicates:** one market per event; the same question on both venues (titles overlap by ≥ 50%, closing within 7 days) is kept once.

### A1. The calendar (`discovery.calendar`)
- **Fed, CPI, jobs, GDP:**
  - the next date comes from the market universe (Kalshi lists each meeting and print)
  - the impact is this book replayed through past releases of that kind (the typical swing and the 1-in-10 bad case)
- **Held-name earnings, including through ETFs:** look-through value × the name's median earnings-day move.
- **EIA and WASDE:** the book's oil or farm exposure per normal day × how big those release days usually are (#212).
- **The US election (Nov 3, 2026):** this book replayed through past big political events.
- **Not built:** OPEC meetings (no reliable date source) and other central banks.

### A2. The news feed (`discovery.news_candidates`)
- **Source:** the daily prospective log (`scripts/planb_prospective.py`, 07:30). It's Wikipedia's current events, from which the LLM picks and labels the economically relevant stories. Since #222 the current month is re-fetched daily.
- **Candidates:** the last 7 days' stories, each as "the story develops further", sized by its labelled severity and factor signs.

### B. Proposals (`discovery.propose`)
- **Input:** one LLM call (about $0.03) gets the book's exposures, the list of coming items (markets, calendar, news) and, since #231, **the book's own history**.
- **The book's history (#231):** the risk card's replay (`live.scan`, code only) goes into the proposal call:
  - **What goes in:** the 4 one-off categories with the worst 1-in-10 bad case. Each comes with how often such events happen, what history says, the worst past example (date, title, this book's dollars) and the 2 holdings that carry most of the bad case. Releases are left out, because they're already in the calendar.
  - **Where it comes from:** in "what could hit me", the card's own rows. When the watchlist is called on its own, the replay is computed there (code, free).
  - **The instruction:** unless the user's events already cover them, at least 2 proposals must be specific, current versions of the top categories, each with its catalyst. The numbers are context only, and code prices every proposal (#182).
  - **`from_history`:** each proposal names the category it instantiates, or null. Code checks it against the category list and drops anything else. The card shows "from history: <kind> (bad case −$X)", where the dollar figure is the replay's, not the LLM's.
  - **The risk card stays as it is,** and so does the analyst prompt.
- **Output:** 6 events in the label format.
- **Links:** a proposal is linked to a coming item only when that item is its catalyst:
  - a market only if its YES is this exact event (that market's odds are then used)
  - a calendar item if the event happens at that release
  - a news story if the event is that story developing further
- **Tail risks:** unlinked proposals.
- **When it runs:** only with a live LLM, so replays and tests are unchanged.

## The trainer's example book (Oct 24)

154 markets, 9 calendar items, 16 news candidates and 6 proposals were priced.

**Headline:**

| Event | Book move if it happens | Odds | Where they come from | Note |
|---|---|---|---|---|
| China–Taiwan blockade | −$3.0k | — | proposal, from a live news story | |
| 5-year yield at 5.25% | −$3.5k | 21% | Kalshi | |
| US–Iran Hormuz agreement | +$2.8k | 34% | Polymarket | the USO puts gain $1.4k |
| AI bubble bursts | −$4.3k | 9.7% | Polymarket | |
| Gas above $5.80 | −$4.2k | 10% | Kalshi | |
| Fed on Oct 28 | ±$1.1k | — | calendar | |
| Strike on the Saudi pipeline | −$1.6k | 58% | Polymarket | |

**Noise for you:** EIA, GDP, PFE earnings, the election.

## Coverage test (`scripts/universe_coverage.py`, `results/universe_coverage.md`)

**Question:** for the 31 big 2026 events in the library, would the universe have held each one before it happened?

| Check | Events |
|---|---|
| An exact market open on the day (e.g. the Hormuz weekly-transit markets before the IRGC closure; the ECB September decision; Musk at $1T) | **5** |
| A market on the same developing story | 14 more (some links are thin) |
| The story already in the news in the 14 days before (a lower bound: library titles are short) | 4 |

**Example book (stocks only):** the share of its 2026 event-driven loss that was covered is 12% strictly, and 72% counting same-story markets.

**What it means:** markets rarely list the exact shock in advance. That's why the universe needs stream B (the book's exposures → proposed events, priced even with no market) and the news stream. Prediction markets mostly price the *story*; the proposals name the *step*.

**Not backtested:** proposals. That would need the LLM to propose for past dates without hindsight.

## What's measured vs what's design

- **Measured:**
  - the factor and sector maps, and event effects by size
  - the priced-in mechanics for releases (#172)
  - daily odds changes don't predict portfolio moves (#153), so levels are used
  - coverage (above)
- **Design, not measured:**
  - whether the proposals pick the events that later matter
  - the base rates as probabilities for a specific event (that's why they're shown as context only)
