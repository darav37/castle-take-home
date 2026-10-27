# How Worsey works

This is the map of the code: what happens when someone asks a question, what each part does, and why it is built that way. It is current through decision #242 (Oct 27, 2026). The `#N` references point to [decisions.md](decisions.md), where each decision has its reason and the alternatives considered; the tests behind the numbers are in [results/](results/RESULTS_SUMMARY.md).

**The rule behind everything: the LLM never produces a number.** Claude finds events, labels them, classifies how an event relates to a market, routes questions and explains results. Code computes every probability, dollar move and range, and a grounding check rejects any reply whose numbers don't come from a tool result (#7, #28, #182). The rule came from evidence: when the analyst was allowed to adjust the engine's factor moves, results got worse (#180).

## 1. The shape of the system

```
browser (src/worsey/web/app.js)
   │  websocket: one group chat, several members, private DMs
server.py  (Runtime: members, chats, history; one Session per member)
   │
agent/loop.py  run_turn()
   ├─ router (Claude): picks an entry tool or hands the question to an expert
   ├─ entry tools: scan_risks · analyze_event · compare_hedges   ← code does the work, then an expert writes the reply
   ├─ experts (Claude, own prompt + tools): event_analyst · options_desk · equity_analyst · etf_desk ·
   │                                        recent_markets_analyst · natural_world_analyst
   └─ grounding.check(): every number in the reply must appear in a tool result
   │
engines (code, no LLM):
   planb/live.py        factor map, scenarios, ranges, options repricing, hedging verdict
   discovery.py         the event universe, pricing every candidate, grouping, Castle's picks
   planb/probability.py probability tiers: markets, derived, bounded
   planb/forecaster.py  tier-3 forecast when no market fits (3 LLM agents, aggregated in code)
   scan_messages.py     the messages the user reads, written in code
   │
data/  provider.py (LiveMarketData / SyntheticMarketData) → Massive, Kalshi, Polymarket, Wikipedia current events,
       every call cached on disk (data/cache.py)
```

The shipped data the engine needs lives in `src/worsey/planb/*.json`: `pack.json` (the event library and the factor model), `real_event.json` (measured sizes for "small / medium / large" moves), `market_labels.json` (labels for prediction markets), `longshot.json` (the longshot correction), `sectors.json`, `countries.json`, `domicile.json`. Each was built by a script in `scripts/` from research data; the chat only reads them.

## 2. One request, end to end: "what could hit my portfolio?"

About 1.5–3 minutes and $0.20–0.30 of Claude calls. Each stage sends a message as soon as it finishes, so the user sees progress.

1. **Routing.** The router (Claude, `prompts/router.md`) picks the `scan_risks` entry tool (`agent/loop.py: ENTRY`, Plan B wording in `ENTRY_PLANB`). If the user names worries ("I'm worried about China blockading Taiwan"), they're passed along as `user_events`.
2. **The book's history, in code** (`planb/live.scan`). Every kind of big past event in the library (3,876 events, 10 categories) and every scheduled release is replayed through today's factor exposures of this book. This gives the kinds of events that hurt this book most and their 1-in-10 bad case. **Message 1** (`scan_messages.book`): the normal two-day swing, the biggest exposures, and those kinds of past events.
3. **The event universe** (`discovery.watchlist`). First `discovery.context()` computes everything the pricing needs once: loadings, sector and country loadings, options, the normal swing. Then two streams:
   - **Stream A, what's coming:** prediction markets closing within the horizon (Kalshi and Polymarket, each with a stored label), the calendar (Fed, CPI, jobs, GDP, held names' earnings, EIA and WASDE releases), and the last week's labelled news stories.
   - **Stream B, what would move this book most:** one LLM call (`discovery.propose`) gets the book's biggest exposures and its own history (the kinds of past events from step 2, #231) and proposes about six specific events, in the same label format the markets use. The same call labels the user's own events.
   - Every candidate is priced by the same code (`discovery.impact`, section 3).
4. **The user's events come first.** Anything that is the same event, or another route to the same main move (`discovery.same_move`), is folded under the user's event, so Castle's suggestions are everything else (#225).
5. **Probabilities** (section 4). One batched LLM call classifies how each proposed or user event relates to candidate markets; code turns that into a probability tier. Events with no market get a forecast: always for the user's events, and for the three biggest unpriced proposals that clear the size screen (half the normal swing). **Message 2** says which events are being researched.
6. **Grouping and Castle's picks.** Code ranks by **risk left**, |move| × √(p(1−p)): an event that is likely and big, but not already certain, has the most risk still to come. `discovery.choose_three` picks three, among events with a real probability (a base rate doesn't count), never two versions of the same move (#234).
7. **One message per event** (`scan_messages.event`), for the user's events and the three picks: what it is, why this book, the probability and where it comes from, and if it happens, each holding's average dollar move with its 9-in-10 range, then the book's, with a verdict (noise, small or meaningful for this book) (#242).
8. **The closing message.** The `event_analyst` expert (`prompts/event_analyst_planb.md`) writes two to four lines: which event matters most and why, what is noise. It may quote only numbers already in the messages, and the grounding check enforces that.

The other entry tools reuse the same engine: `analyze_event` ("what if oil spikes 30%?"): the analyst structures the event, code prices it; `compare_hedges`: the hedge table in code, then the event and options experts in parallel.

## 3. How big: an event's dollar move (Part III)

**The factor map** (`planb/live.loadings`, #156, #163). Every holding is described by its exposure to 13 named factors: US stocks, 2-year yields, 10-year yields, the dollar, oil, gold, credit, small vs large caps, value vs growth, emerging markets vs US, healthcare, farm commodities, utilities vs energy producers. Each factor is a live ETF proxy (`PROXIES`, e.g. SPY; SHY and IEF sign-flipped for yields; HYG − IEF for credit), in units of its 2011–2020 daily standard deviation. Loadings are an OLS fit on the last 504 sessions; what's left is each holding's own noise.
- **Why factors:** for an unusual book, a market-only view explains 37% of its event moves; the factor map explains 77% (#178).

**Layers on top of the macro factors**, each fitted on what the macro factors leave over:
- **Sector** (`sector_loadings`, `sectors.json`): one sector factor per stock, kept only if it explains more than 2% more of the stock's daily moves. The median stock's R² rises from 24% to 33% (#215–217).
- **Country** (`country_loadings`, `countries.json`, `domicile.json`): country ETFs map to their own country, foreign stocks to their home country (#238).
- **Company** (`discovery.impact`, `live.company_legs`): only for company-specific events (#232).
- Per holding, the largest of its sector, company and country legs counts, never the sum (`discovery.combine_legs`), because a broad event's company leg is mostly the sector's move again.

**The event's label → the middle of the scenario** (`live.real_event_draws`, #213, #214). An LLM label gives the event a severity, a direction per factor, and a size per factor: small, medium or large. Each size maps to a measured move: past events the labeller called "large" moved that factor about 2.4× its normal two-day noise, "medium" about 0.7×, "small" about 0.1× (`real_event.json`). Factors the label doesn't sign get their typical co-move given the signed ones, from past major and extreme events.

**The spread.** The scenario's shape comes from resampling past events of the same severity: their leftover factor moves, rescaled to the label's widths (`live.resid_pool`). Each holding gets its own noise on top, drawn jointly with the noise of any option on it, so a protective put offsets the stock's own drops within each draw (`live.shared_noise`, #235). 4,000 draws.
- Drawing from past events of the same category instead was tested and was worse on held-out years, so it isn't shipped (#239; switch `WORSEY_DRAW_POOL`).

**Options** (`live.option_setup`, `live.option_pnl`, #209). Fully repriced in every draw: spot, implied vol through a measured spot–vol response, and time. Taylor-expansion Greeks fail on big moves: on the example book's oil puts at oil +40%, Greeks said +$1.8k and full repricing −$1.5k.

**What the user sees** (#242). Every holding's dollars in every draw, so each holding gets its average and 9-in-10 range, and the holdings add up to the book draw by draw. The verdict compares the book's average to its normal two-day swing: under half is noise, under one swing is small, more is meaningful (#219).

**Why trust the ranges.** On 12 portfolios × 92 events of 2026, the real move landed inside the 9-in-10 range 90% of the time, versus 77% for a raw LLM's range (#185, #186; rerun on the current code with identical results). The live "what if" path scored 93% on 1,956 held-out book-events (#239). Nobody predicts the direction of a move from a headline (about a coin flip), and predict-zero plus normal noise scores better than both on the interval score; the ranges are honest, not sharp.

## 4. How likely: a probability on every event (Part II)

`planb/probability.py`, wired in by `discovery.tier_probabilities` and `discovery.forecast_probabilities` (#226–234). The LLM never writes a probability here: it labels relations and reads thresholds and deadlines from the event's words.

| Tier | When | How (code) |
|---|---|---|
| **1, market** | a listed market is the event | its price, longshot-corrected where the bias was measured (`corrected`, `longshot.json`: logit fit on 19,792 resolved markets; prices of 2–35% overstate YES, so a 7% price resolves YES 3.7% of the time, #227) |
| **2a, derived** | same variable, different strike or deadline | strike ladders made monotone and interpolated, never extrapolated (`derive_strike`); deadline rungs interpolated, constant hazard past the last rung within 3× the time (`derive_deadline`) (#228) |
| **2b, bounded** | a related market | relations → bounds (E implies M: P(E) ≤ P(M); M implies E: ≥; exclusive: ≤ 1 − P(M)), intersected (`bounds`); the midpoint if the bounds are within 10 points, else tier 3 inside them |
| **3, forecast** | nothing listed is close | `forecaster.forecast`: three agents (same model) each see the library base rate, the 8 most similar past events, the last 14 days of news and up to 3 related markets, in different orders; median in log-odds, clipped to [2%, 98%] and to any tier-2b bounds |
| base rate | only a kind of event fits | how often a big event of that kind happened in a window this long; context only, never used as the event's probability |

**Matching** (`candidates_for`, `classify_many`, `tier_from`): candidates come from embedding similarity plus shared title words; one batched LLM call labels each event–market relation; code picks the tier. A proposal's own market link is only a hint (#234). Relation labels were 34 of 35 right on a test set (#228).

**Priced in** (`discovery._set_prob`): if yes = move × (1 − p), if no = −move × p, from the raw market price, because that is what asset prices carry. Risk left uses the corrected probability.

**Evidence:** the forecaster pilot (29 questions): Brier 0.139, vs 0.167 for a raw LLM and 0.143 for the hidden market price (#230). The sample is small and the larger backtest wasn't run.

## 5. Where events come from (Part I)

- **The event library** (`pack.json`, built by `scripts/planb_pack.py`): 3,876 past events, mostly from Wikipedia's Current events portal, each labelled with a category, a severity and the factor moves it caused (#157, #161). It feeds the replay, the base rates, the spreads and the forecaster's similar events.
- **Market labels** (`market_labels.json`, built by `scripts/discovery_label.py`): each listed market's label (does it move markets, which factors, which sectors), made once with the same prompt the live labeller uses. The live labelling prompts were backtested on 745 past events: about 74% right on the direction of big moves (#236).
- **The news feed** (`planb/wikinews.py`, `scripts/planb_prospective.py`): the daily labelled stories behind "the story develops further" candidates.
- **The honest limit:** an exact market existed beforehand for only 5 of 31 big 2026 events (#224), and in a 39-run test a raw LLM named about twice as many of the events that later happened (#240). The universe's strength is that every event in it is priced and traceable, not that it's complete.

## 6. The agent layer

- **The loop** (`agent/loop.py`) runs on the raw Messages API, no framework (#27). The router sees the entry tools and the experts; each expert is the same loop with its own prompt (`prompts/`), tools and a fresh message list. Experts get at most 4 tool rounds and 45 seconds (`config.EXPERT_MAX_ROUNDS`, `EXPERT_TIMEOUT_S`), and may consult one other expert, one hop only (#112).
- **Who writes the reply** (#100): on judgment turns the expert writes it; code checks it, gives the expert one rewrite if needed, and only then shows it.
- **Grounding** (`agent/grounding.py`, #28, #83): every number in a reply must match a tool result from the conversation or the user's own words, allowing display rounding ($4.1k, 12%). It fails closed: an unsupported sentence or table row is dropped. A format check keeps replies short.
- **Spending caps** (`agent/budget.py`): every live call passes `gate()` first. Limits per turn (25 calls), hour, day ($25 by default) and project, all env-overridable. Research scripts add their own per-run caps and a dry run (`pmrm/ic/budget.py`).
- **Memory and profiles** (`agent/memory.py`, `agent/profiles.py`): what a member said or did, never inferred psychology; advisory lessons expire and are never promoted without the trainer's approval (#111, #114, #120).

## 7. Data

- **`data/provider.py`**: one interface for the engines. `LiveMarketData` uses Massive (prices, option chains, ETF holdings, earnings, news) and Kalshi; `SyntheticMarketData` is a fake world for tests and plumbing, never for results. Polymarket (`data/polymarket.py`) joins the universe through topic tags (#224).
- **`data/cache.py`**: every external call is stored on disk, keyed by source, path and parameters, never the key. `WORSEY_OFFLINE=1` forbids the network, so tests run with no keys.
- **ETF look-through** (`lookthrough.py`): a fund's holdings count toward each named company's exposure (#61).

## 8. The chat UI

`server.py` serves `web/app.js` over a websocket: one group chat with several members, plus private DMs, with privacy enforced by construction (#109, #110). In the group chat the book is scaled to $100k before anything sees it. On the live path the event pipeline sends text messages only (`agent/session.py: TEXT_ONLY_CARDS`); the options desk and ETF desk keep their cards. Recorded demo replays belong to the first engine and keep their cards.

## 9. Configuration

`src/worsey/config.py` holds the tunable defaults with comments. The switches that matter:

| Variable | Default | What it does |
|---|---|---|
| `WORSEY_SCENARIO_ENGINE` | `planb` | the engine; `v1` is the first engine, kept for the recorded replays |
| `WORSEY_ROUTER_MODEL`, `WORSEY_EXPERT_MODEL` | `claude-sonnet-5-5` | models (#29, #80) |
| `WORSEY_LLM_MAX_USD_DAY`, `WORSEY_LLM_MAX_PER_TURN`, … | $25, 25 calls | spending caps |
| `WORSEY_FORECAST_MAX_PER_SCAN` | 3 | Castle's unpriced proposals forecast per scan |
| `WORSEY_DRAW_POOL` | `severity` | `category` turns on the tested-and-rejected category pool |
| `WORSEY_OFFLINE`, `WORSEY_CACHE_DIR` | off, `cache/` | network off; where the disk cache lives |

## 10. Product code vs research code

- **Product:** everything above, plus the explorations (options desk, ETF desk, natural-world analyst, train mode), described in [explorations.md](explorations.md).
- **Research:** `src/worsey/pmrm/` and `src/worsey/planc/` hold the harnesses behind the tests; `scripts/` holds the scripts that produced each number in the results (each names its decision). `live.set_cutoff(date)` hides everything on or after a date, so the shipped engine can be scored on events it never saw. What was tried and dropped: [research_history.md](research_history.md).
- **Tests:** `tests/` (about 470, no keys, no network). `tests/test_planb_live.py` covers the engine and the messages, including that each holding's average adds up to the book's and that options are repriced in every draw.

## 11. Known limits

- Directions after a headline are near a coin flip; the value is in honest ranges and in saying "this is noise for you".
- The forecaster rests on a 29-question pilot.
- The universe misses many events a raw LLM names (#240).
- The news labeller is the weakest labeller (65% on big-move direction).
- Liquidity isn't graded yet: a thin or stale market is used like a liquid one. The design is in [future_work.md](future_work.md).
- Loadings use today's holdings over 504 sessions, so survivorship and regime changes aren't modelled.
