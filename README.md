# Worsey

A chat agent for event-driven risk on a portfolio, built for the Castle take-home. Give it a book; it finds the events that could move it, puts a probability on each, and says how much each could move the book, in dollars, with an honest range. A **worse Percy**: built in a few days on a dummy portfolio with real prices.

**One rule runs through all of it: the LLM never produces a number.** The model finds, labels and explains; code computes every probability, move and range, and a grounding check rejects any reply whose numbers don't match a tool result.

**To understand the code, start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):** one request followed end to end through the code, what each part does and why, with links to the decisions and tests behind it.

Author: Darin Avila.

## Question 1: what could hit my portfolio

Three parts, each with its evidence. The event pipeline is the headline; everything else is an exploration (see the end).

**Part I. Which events.** A run's universe is the user's own events plus Castle's: listed prediction markets (Kalshi, Polymarket), the economic and earnings calendar, the news, and LLM proposals aimed at the book (including the kinds of events that hurt it in the past). Castle's picks are the three with the most risk left, excluding anything the user asked about.
- Prediction markets rarely list the exact shock in advance: an exact market existed beforehand for only 5 of 31 big 2026 events (19 of 31 counting same-story markets). A market list alone isn't enough.
- → [docs/event_pipeline/universe.md](docs/event_pipeline/universe.md)

**Part II. How likely.** Every event gets a probability with its source, in four tiers: a market (corrected for the measured longshot bias), a number derived from related markets (ladders, deadlines), bounds from markets, or a forecaster when no market fits. "No market fits" is said plainly.
- Longshot bias, measured on 19,792 resolved markets: a 7% price resolves YES 3.7% of the time. The correction cuts out-of-sample log loss 1.7%.
- Event-to-market relations: 34 of 35 labelled right.
- Forecaster pilot, 29 questions, Brier (lower is better): raw LLM 0.167, Worsey 0.139, market 0.143. Small sample; the larger backtest wasn't run.
- → [docs/event_pipeline/probabilities.md](docs/event_pipeline/probabilities.md)

**Part III. How big.** A 13-factor map of the portfolio, plus a sector layer and a country layer; an LLM label gives each event's directions and sizes; code turns them into dollars per holding, options fully repriced, with a 9-in-10 range and a verdict (noise, small or meaningful).
- Against asking an LLM directly, on 12 portfolios × 92 events of 2026: the real move lands inside Worsey's "90%" range 90% of the time, versus 77% for the raw LLM's range. It scores better on the interval score, but **plain "nothing happens plus normal noise" scores better than both**, and nobody predicts the direction of a move from a headline (about a coin flip).
- → [docs/event_pipeline/impact.md](docs/event_pipeline/impact.md), [docs/results/by_the_numbers.md](docs/results/by_the_numbers.md)

The flow chart of the whole pipeline: [docs/event_pipeline/flow_map.html](docs/event_pipeline/flow_map.html) (open in a browser). Terms: [glossary](docs/event_pipeline/glossary.md).

## What's real, what's thin, what isn't built

- **Real:** the event pipeline above, the tests behind it, and the options and hedge pricing.
- **Thin:** the forecaster (small pilot), Part I against a raw LLM (in a 39-run test as of three 2026 dates the raw LLM named about twice as many of the events that happened; Worsey's edge is that its universe is priced and traceable, see `docs/results/by_the_numbers.md`), the news labeller (65% direction, weakest of the three).
- **Tested and dropped:** drawing a scenario's spread by event category (worse on the live path), learned event-to-move models (lost to noise); see [docs/research_history.md](docs/research_history.md).
- **Designed, not built:** thin-market handling (a quote's liquidity grade) and combining two venues that list the same event; see [docs/future_work.md](docs/future_work.md).

## Run it

```
pip install -e ".[dev]"
python -m pytest                      # ~470 tests; no keys, no network
python scripts/check_submission.py    # links, secrets and structure
worsey scan --synthetic               # the scan on a fake world (plumbing check)
```

### Chat with it on live data

1. Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` and `MASSIVE_API_KEY` (Massive supplies prices, option chains and ETF holdings; Kalshi and Polymarket need no key).
2. `worsey serve --portfolio balanced`, then open http://127.0.0.1:8000/?user=me and ask "what could hit my portfolio?".

Market data is fetched on first use and cached under `cache/` (not in the repo). On an empty cache the first "what could hit me" takes about 3 minutes; later ones are faster. Each event comes back as its own message: its probability and where it comes from, and if it happens, each holding's average dollar move and 9-in-10 range, then the book's. A full turn costs about $0.20–0.30 of Claude calls. Live calls are capped per turn, hour, day and project (`src/worsey/agent/budget.py`; set `WORSEY_LLM_MAX_USD_DAY` lower to be safe). Example output: [docs/pmrm/results/text_only_messages.md](docs/pmrm/results/text_only_messages.md).

Tested from a fresh clone with an empty cache (Oct 27, 2026). The recorded-replay server (`worsey serve --replay`) belongs to the first engine and only runs with the data it was recorded on, so it doesn't run from this repo.

## Where things are

| | |
|---|---|
| How the code works, end to end | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| The pipeline, part by part | [docs/event_pipeline/](docs/event_pipeline) |
| Every test, with the headline table first | [docs/results/](docs/results), raw numbers in [docs/pmrm/results/](docs/pmrm/results) |
| The main decisions, with reasons and alternatives | [docs/decisions.md](docs/decisions.md) |
| What was tried and dropped | [docs/research_history.md](docs/research_history.md) |
| The other features, honestly | [docs/explorations.md](docs/explorations.md) |
| Module map | [src/worsey/README.md](src/worsey/README.md) |
| Scripts that reproduce the quoted numbers | [scripts/](scripts) |
| Strategy library, prompts, portfolios | `library/`, `prompts/`, `portfolios/` |
