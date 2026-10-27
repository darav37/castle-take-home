# Future expansions

Ideas for after the take-home, collected during the walkthrough (Oct 24–25). Each item says what it is, why it matters, what the evidence says so far, and how we'd test it. Nothing here is built. The first item is the one the trainer would try first.

## 1. Train the labelling LLM

**What it is today.** Every event is labelled zero-shot by a frontier model: a direction and a small, medium or large bucket for 7 factors, plus sector and company legs. The labeller sees the event's text and the label format, but no past events, examples or outcomes (`scripts/discovery_label.py`, `discovery.PROPOSE`). The 3,876-event library is used only after labelling, in code: to turn buckets into measured moves (#214), fill co-moves, and replay history through the book.

**Why improve it.** The label is the one LLM judgment that every number depends on. Its known weak spots:
- direction after earnings reports and for macro events aimed at specific sectors: close to a coin flip
- black-swan shocks: 33–39% skill
- arbitrary company legs: partly fixed in #232

**Options, cheapest first:**
1. **Worked examples (retrieval).** For each new event, put the most similar past events in the prompt, with their labels and what markets actually did. Similarity can come from category, severity and the labels themselves, not text embeddings (#197: embeddings lost to noise).
2. **Calibration on top.** Keep the zero-shot label, then learn a small correction from history: when the labeller says "oil large" for a category, what did oil actually do? Code only, no model change.
3. **Fine-tune a model.** Train it to produce labels that match the realized factor moves of the 3,876 library events (label → realized direction and bucket per factor). It needs enough examples per category, and a strict time split.

**How to test (the same test for all three).** Use held-out library events, after the labeller's knowledge cutoff, to avoid leakage. Score per factor:
- sign accuracy
- bucket accuracy
- the book-level error of the priced move

Compare each option with today's zero-shot labeller.

**Caution from the evidence.** LLM adjustments layered on the engine made results worse (#180), and embedding retrieval lost to noise (#197). Each option has to beat zero-shot on held-out events before it goes live.

## 2. Short-history stocks (recent IPOs)

Loadings need 504 trading days, so a recent listing (e.g. SPCX, listed June 2026) can drop out of the factor map without a warning. The labeller also may not know a company is now listed.

**Options:**
- fit on the history that exists and flag it as low-confidence
- fall back to the sector ETF's loadings
- give the labeller the list of currently listed tickers

## 3. Scan latency

A full "what could hit me" scan takes about 175 s since #233 (relation call, forecasts, reply), up from about 50 s. Code-written messages now stream as results arrive (#234). Next steps: overlap the LLM rounds, and cache forecasts per event per day.

## 4. Options in the bad case

The watchlist's 1-in-20 bad case holds options at their middle-case value ("options at middle" on the flow map). Drawing options across the bad-case scenarios, as the scenario pricer does, would show how puts and calls behave in the tail.

## 5. More event sources

- **Discovery:** Kalshi and topic-filtered Polymarket today. Not covered: news threads with no market, and other sources in PLAN_B §5.1 (GDELT, Federal Register, USTR, OFAC, ACLED).
- **Calendar:** OPEC meetings, other central banks and elections beyond the US.

## 6. Sector betas that adapt faster

Sector loadings refit daily on a 504-day window. A shorter or exponentially weighted window would react faster to regime changes. It would need the held-out R² test from #215 to show it's worth it.

## 7. A world model of standardized events (long term)

The trainer's direction beyond the take-home: SDE and hazard-rate models of standardized events, and how they interact (Cox processes for event intensity, Hawkes processes for escalation). It would give a structural forecaster for events with a quantifiable underlying, beside the LLM forecaster. Background: Lando (1998), Duffie & Singleton (1999), Mohler et al. (2011).

## 8. An alternative architecture: LLM-first, code verifies

**Today the pipeline is code-first:** fixed steps (universe, labels, tiers, factor pricing) with LLM calls at chosen points.

**The alternative is draft and critique:**
- one strong model answers the whole question in one go (events, probabilities, moves)
- expert sub-agents for probabilities and for moves critique the draft, with access to the data

**The key constraint:** the critics verify every number by calling the existing modules (the factor engine, the market matcher, the forecaster), never by writing their own. The modules become verification tools for the draft.

**Why consider it:** the pipeline already makes 5+ LLM calls per scan. A draft might surface events or framings the structured steps miss, while every number stays checked.

**How to test it:** the same scorecard as the raw-LLM comparison (coverage of events that hit the book, Brier, range coverage, traceability). The raw-LLM baseline is the draft with no critics; its ranges held 73–77% of outcomes vs 90% for the pipeline (#185–186), so the critics must close that gap.
