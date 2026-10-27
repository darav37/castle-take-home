# Labeller backtest: are the live labels right?

**Current through DECISIONS #236 (Oct 25, 2026).** Full results: `docs/pmrm/results/live_labeller_backtest.md`.

## Say it out loud

> We backtested the exact labelling prompts the product runs on 745 past major and extreme events. From the headline alone, they call the direction of big factor moves right about three times in four (74% for the market labeller, 73% for the user-event labeller), as well as our research prompt (75%). Their size calls separate cleanly: a 'large' call means a move about three times a 'small' one.

## Direction (right on factor moves over 2× normal noise)

| | market labeller (L1) | user-event labeller (L2) | research read (#196) | news labeller (L3) |
|---|---|---|---|---|
| **All 7 factors** | **74%** (70%–79%) | **73%** (68%–77%) | **75%** (71%–80%) | **65%** (60%–70%) |
| US stocks | 77% | 79% | 78% | 75% |
| 2-year yields | 71% | 71% | 66% | 57% |
| 10-year yields | 60% | 62% | 64% | 64% |
| dollar | 68% | 74% | 80% | 69% |
| oil | 78% | 77% | 83% | 75% |
| gold | 80% | 62% | 71% | 63% |
| credit | 83% | 83% | 86% | 53% |

By how big the real move was (all signed calls):

| real move | < 1× | 1–2× | 2–3× | > 3× |
|---|---|---|---|---|
| market labeller | 56% | 69% | 70% | 81% |
| user-event labeller | 54% | 65% | 70% | 77% |

## Size (the real move per size call, × normal noise)

| size call | market labeller | user-event labeller |
|---|---|---|
| none | 0.77 | 0.80 |
| small | 0.90 | 0.84 |
| medium | 1.55 | 1.30 |
| large | 2.70 | 2.53 |

## Leakage

On the 32 events after the model's training cutoff (Jul 1 – Sep 30, 2026), the market labeller gets 70% of moves over 1× noise right (n 20, interval 53%–86%) vs 71% before it: in line, so no sign of remembered outcomes, but too few events to prove it.

## One weak spot

The news labeller (which signs the watchlist's news stories) is worse (65%; credit 53%): its prompt never defines the factors. Fix: add the one-line definitions the other two prompts carry.

## What this does and doesn't show

- **Does:** the labels the product turns into numbers (direction and size per factor) are as good as our best research prompt, on 745 past events.
- **Doesn't:** it says nothing about choosing which events matter (that's the universe and the forecaster), and it's conditioned on the move being big, which isn't known in advance (that's what the size calls and the noise verdict handle).
