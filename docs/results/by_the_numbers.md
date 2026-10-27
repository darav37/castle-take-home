# By the numbers: every decision against a raw LLM

**Current through DECISIONS #240 (Oct 25, 2026). Part B1: compiled from existing results, $0.** Each row is one claim with its metric, n, interval and source. Where the raw LLM ties or wins, the row says so. **Stale** marks a result measured on an earlier engine that has changed since (size calls #214, sectors #215–217, company legs #232, shared noise #235, countries #238). The B2 scorecard (top section) reruns the core comparison on the current engine.

## Scorecard (B2, all 39 runs: 13 books × 3 dates of 2026, as of each date)

Details and caveats in `docs/pmrm/results/scorecard_vs_raw_llm.md`.
- **Part I, coverage: Worsey loses.** It named 16% of the events that happened (0.07–0.29); the raw LLM named 34% (0.19–0.51). The universe is traceable and priced, not more complete.
- **Part III, ranges: Worsey wins on the raw LLM, loses to the floor.** Interval score $9,814 vs raw $31,696; predict-zero plus noise $6,427. The raw LLM's 9-in-10 range held 38%.
- **Part II, probabilities: not comparable here** (different question sets; Worsey's are mostly rare events). See the pilot rows below.
- **Traceability:** Worsey's probabilities sit within 0.09 of the market on average (n 114).

Why the scorecard's range coverage (100% vs 38%) differs from the 90% vs 77% below: the scorecard has only 9 Worsey and 13 raw-LLM book-windows with a scored move (30-day windows, 3 dates), so 100% is 9 of 9. The 90% vs 77% comes from 1,104 book-events (12 books × 92 events of 2026) and is the number to quote. It was rerun on the current code on Oct 25 (#239) with identical results; the live chat's own "what if" path scores 93% on 1,956 book-events. *Stale* rows below were measured on an earlier engine and not rerun.

"Raw LLM" = the same Claude model the chat uses, with no tools and no Worsey data, given the same date, holdings and news.

## Part I: which events could hit the book

| Claim | Metric | Worsey | Raw LLM / baseline | n | 90% interval | Source |
|---|---|---|---|---|---|---|
| Prediction markets rarely list the exact shock in advance, so a market list alone isn't enough | Big 2026 events with an exact market open beforehand | 5 of 31 (16%) | – | 31 events | 8–30% | #224, `results/universe_coverage.md` |
| …but they often price the story | Same-story market open beforehand | 14 more (19 of 31 in all, 61%) | – | 31 | 47–74% (any market) | #224 |
| The universe covers most of the example book's event losses only with same-story markets | Share of 2026 event-driven loss covered | 12% strictly (exact market or story already in the news); 72% counting same-story markets | – | 16 loss events, $24,183 | – | #224 |
| Feeding the book's own history into the proposal call aims the proposals at what hurt this book | Proposals in the replay's top-3 kinds (example book) | 1 → 3 | – | 1 book, 1 run each | – (a single run) | #231, `results/replay_feeds_proposals.md` |

**Not measured:** whether the proposals pick the events that later matter (that needs the LLM to propose for past dates without hindsight). The B2 scorecard measures coverage against the raw LLM: Worsey loses it (see the Scorecard section).

## Part II: how likely

| Claim | Metric | Worsey | Raw LLM / baseline | n | Interval | Source |
|---|---|---|---|---|---|---|
| Market prices overstate longshots; correcting them helps | Out-of-sample log loss | 0.2649 | raw market price 0.2694 (−1.7%) | 62,360 test snapshots (of 88,256; 19,792 markets) | per-bin event-clustered 95% CIs in the source | #227, `results/part2_longshot.md` |
| e.g. a 7% market | Realized YES rate at a 7% price | 3.7% | price 7% | ~12k snapshots in the 5–10% bin | 95% CI of the gap −3.8 to −2.6 points | #227 |
| Event-to-market relations are labelled correctly | Agreement with hand labels | 34 of 35 (97%) | – | 35 pairs, 12 events | 88–99% | #228, `results/part2_relations.md`. **Caveat: the hand labels were written by the build assistant, not by an independent person.** |
| When no market lists an event, the library forecaster beats the raw LLM | Brier (lower is better) | **0.139** | raw LLM 0.167; raw + base rate 0.172 | 29 event questions | Worsey − raw = −0.028, **95%** CI −0.060 to −0.0001 (barely excludes 0; n = 29, one question per event) | #230, `results/part2_pilot.md` |
| …and ties the market | Brier | 0.139 | hidden market price 0.143 | 29 | Worsey − market = −0.004, 95% CI −0.108 to +0.096 | #230 |
| The library's base rate alone adds nothing (arm B vs arm A) | Brier, raw + base rate vs raw | 0.172 vs 0.167 | – | 29 | +0.005, 95% CI −0.031 to +0.035 | #230 |

**What each arm of the forecaster pilot saw (#230; checked against `scripts/part2_pilot.py` and `src/worsey/planb/forecaster.py`).** 29 questions that resolved between Jul 1 and Oct 23, 2026, about the kinds of event in the library (not elections), picked by an LLM that saw only the question text, never outcomes or prices. The market's price on the forecast date was hidden from every arm and kept only as the benchmark.

| Arm | What it gets |
|---|---|
| **A, raw LLM** | The question, its resolution rules and the forecast date. No tools, no base rate, no market price. |
| **B** | A, plus the library base rate for the question's event category: how often a major or extreme event of that kind happened in a window of the question's length, 2016–2025. |
| **C, Worsey** | The same question, rules and date, plus (1) the base rate; (2) the 8 library events most similar to the question (embedding similarity on title and description, only events dated before the forecast date); (3) the 15 Wikipedia current-events items from the 14 days before the date that share the most words with the question. Three agents (same model) each see this evidence in a different order. Result: the median in log-odds, clipped to [2%, 98%]. Related markets: none in the pilot (the live chat adds up to 3 related markets as evidence; that part is untested). |
| D, market (benchmark) | Its price on the forecast date. |

- **The similar events show date, category, severity and title only.** The measured factor moves are not shown to the forecaster.
- **What the pilot can't say:** which ingredient produced C's gain. Only C has the similar events, the news and the 3-agent median, and no arm isolates any of them. B vs A shows the base rate alone adds nothing.
- **The headline is thin:** n = 29, and the 95% interval on C − A (−0.060 to −0.0001) only just excludes zero. The larger backtest (~250 questions) has not been run.

## Part III: how big

| Claim | Metric | Worsey | Raw LLM / baseline | n | Interval | Source |
|---|---|---|---|---|---|---|
| The live labels get the direction of big moves right | % right on factor moves over 2× noise (market labeller / user-event labeller) | 74% / 73% | research prompt 75% | 341 / 423 big moves, 745 events | 70–79% / 68–77% | #236, `results/labeller_backtest.md` |
| Their size calls separate | Realized move per size call (× noise), market labeller | small 0.90, medium 1.55, large 2.70 (ordered) | – | 2,053 / 448 / 80 calls | – | #236 |
| No sign of leakage, but too few events to prove it | % right on moves over 1× noise, after the model's cutoff | 70% | 71% before the cutoff | 20 moves, 32 events | 53–86% | #236 |
| **The live chat's "what if" ranges are calibrated too** | Real move inside the 9-in-10 range, live user-event labels, held out by year | **93%** (2.0% below the 5th percentile) | – (no raw-LLM arm in this test) | 1,956 book-events: 163 major and extreme events of 2023–26 × 12 books | – | #239, `results/draws_by_category.md` |
| **Worsey's ranges are honest; the raw LLM's are overconfident** *(rerun on the current code, Oct 25: identical)* | Real move inside the "9 in 10" range, portfolios / positions | **90% / 90%** | 77% / 73% | 12 books × 92 events of 2026 | – | #185–186, `results/naive_vs_system*.md` |
| …and score better overall *(rerun on the current code, Oct 25: identical)* | Interval score, $ per book-event (lower is better) | $752 (books), $803 (positions) | $781, $839 | 12 × 92 | Worsey better by $29 (+$7 to +$51), 98% support; by $37 (+$17 to +$57), 100% support | #185–186 |
| **Neither predicts the middle** *(rerun on the current code, Oct 25: identical)* | Median error of the middle; R² vs zero | $903; −1.2% | $943; −3.8% | 12 × 92 | – | #185 |
| **Predict-zero plus normal noise beats both** *(rerun on the current code, Oct 25: identical)* | Interval score | Worsey $752 / $803 | **zero + noise $721 / $769** | 12 × 92 | – | #185–186 |
| **The chat's factor calls had no edge over the raw LLM** *(stale; before size calls)* | R² vs zero on 8 factors | shipped chat −7.3% | raw LLM −7.4%; engine alone +1.8% | 31 major/extreme 2026 events | chat − raw: −11 to +13 points (coin flip) | #180, `results/product_vs_raw.md` |
| One sector factor per stock explains a third more of its moves | Median stock R², held out 2021–26 | 33.3% | macro factors only 24.2% | 456 S&P 500 stocks | 65% of stocks better, 9% worse | #215, `results/planc_sectors.md` |
| A home-country factor explains much more of a foreign stock's moves | Median held-out gain in R² | +13.7 points (to ~44%) | macro only ~30% | 41 foreign stocks/ADRs | 36 of 41 pass the 2% rule | #238, `results/country_layer.md` |
| …and fixes a confidently wrong "no effect" | EWZ on "Brazil invades Peru" ($30k) | −$934 (−3.1%) | before the layer −$60 (−0.2%) | 1 case | – | #238 |
| Options must be fully repriced, not approximated with Greeks | Example book's USO puts at oil +40% | full reprice −$1.5k | Greeks +$1.8k (wrong sign) | 1 case | – | #209 |
| Hedging every event loses money; "don't hedge by default" is right *(stale engine; the decision is robust)* | Net result of an always-on protective put, per $100k book-event | – | always hedge −$131 (2023–25), −$150 (2026); never hedge better by $99–111 | 4,727 book-events | 0–4% support for always hedging | #199–200, `results/planc_hedge.md` |
| The raw LLM, told protection usually costs more, also almost never hedges | Share of book-events hedged; gain | – | 0.3% hedged (15 of 1,104), +$2 per book-event | 1,104 | 82% support (too few to call) | #199 |

## Traceability

| Claim | Metric | Worsey | Raw LLM | n | Source |
|---|---|---|---|---|---|
| Worsey's facts come from data; the raw LLM guesses *(stale)* | Earnings dates for the example book | real dates from the calendar | guessed (WDC "Oct 28–31"; actually Oct 22) | 1 book | #210, `results/side_by_side_example.md` |
| Every number in Worsey's replies comes from code | Grounding check before each reply | enforced (one rewrite, then fail closed) | none | every reply | #28, #83, #182 |

## Reading it

- **Where Worsey wins clearly:** honest ranges (90% means 90%), probabilities when no market exists (vs the raw LLM), corrected market odds, sector and country structure, options repricing, and traceable facts.
- **Where it ties or loses:** nobody predicts the middle of a move from a headline, and predict-zero plus normal noise beats both arms on the interval score. The chat's factor calls tied the raw LLM in #180 (before size calls); B2 re-tests that on the current engine.
- **What B2 adds:** the four-score comparison (coverage, probabilities, moves and ranges, traceability) rerun on the current engine, replacing the stale rows.
