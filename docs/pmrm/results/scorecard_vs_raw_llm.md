# Scorecard: Worsey vs a raw LLM on question 1 (B2, all 39 runs)

13 books × 3 dates (2026-07-01, 08-01, 09-01), each scored on the next 30 days; every date is rebuilt as of that day, with no hindsight. Design predeclared in the docstring of `scripts/scorecard_vs_raw.py`. "Raw LLM" = the same Claude model with no tools and no Worsey data, given the date, the holdings and the news. A blinded LLM judge matches each arm's named events to what actually happened. LLM spend about $9 for the run.

## Results

| Score | Worsey | Raw LLM | Floor (predict zero + normal noise) | n | Read |
|---|---|---|---|---|---|
| **Part I, coverage:** share of realized events the list named | 16% (90% interval 0.07–0.29) | **34%** (0.19–0.51) | – | 28 book-windows with a realized event | **the raw LLM named more of what happened.** The intervals overlap, but the gap is not small |
| …Castle's 3 picks only | 8% (0.02–0.17) | – | – | 28 | picks rank by risk left, not by chance of happening |
| **Part II, probabilities:** Brier (lower is better) | 0.038 (0.020–0.057) | 0.177 (0.125–0.245) | – | 113 vs 68 questions | **not a like-for-like comparison:** Worsey's questions mostly ask about rare events (yes rate 3.5% vs 19%), which flatters a low Brier |
| **Part III, ranges:** interval score per book (lower is better) | $9,814 | $31,696 | **$6,427** | 9 / 13 / 22 | Worsey's ranges are far better than the raw LLM's (raw 9-in-10 coverage 38% vs Worsey 100%), but predict-zero plus noise still wins, as in #185 |
| Median error of the middle | $709 | $3,327 | $496 | | |
| **Traceability:** mean \|p − market price on the date\| | 0.091 (n 114) | 0.26 (n 4) | – | | every Worsey number comes from code; the raw LLM mostly names no market-checkable probability |

32 Worsey events with only a base rate (context, not a forecast) are left out of the probability score.

## What it says

- **Part I is where Worsey loses.** Given the same date and news, the raw LLM names about twice as many of the events that then happened. Worsey's list is built from listed markets, the calendar, the news and proposals aimed at the book, and it is not better at guessing what will happen. This matches the earlier finding that markets rarely list the exact shock in advance (#224). The honest claim for Part I is that the universe is *traceable and priced*, not that it is more complete.
- **Part III holds.** The ranges are honest where the raw LLM's are not, and the floor still beats both. Few book-windows have a scored move (9, 13 and 22), so the intervals are wide.
- **Part II can't be called from this test.** The two arms were scored on different question sets. The like-for-like test is the 29-question forecaster pilot (#230).

## Caveats

- The 13 books share three dates and the same news, so the runs aren't independent.
- The judge is an LLM, matching named events to realized ones.
- Only windows with a realized event enter the coverage score (28 of 39); only windows with a scored move enter the range score.
- Reproduce: `python scripts/scorecard_vs_raw.py --confirm --approved-over-10` (resumable; ~$9), then `--score-only`.
