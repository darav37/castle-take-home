# Drawing the scenario spread by category: tested, not shipped (DECISIONS #239)

Task: the design brief. Script: `scripts/draws_by_category.py` (`history`, `whatif`, `report`); raw numbers `draws_by_category.json`. No LLM calls, $0. Code: `live.resid_pool()`, switch `WORSEY_DRAW_POOL` (default `severity`).

## In plain words

- **The idea:** a scenario's spread and co-movement come from resampling past events' misses (their factor moves minus the model's centre). Today the pool is every past event of the same severity. The change draws from past events of the same **category and severity** (at least 20 of them; else the category's major + extreme events; else the severity pool), so an oil shock's spread would look like past oil shocks'.
- **On the path the chat uses (what-if and the watchlist), it's worse.** Held out by year on 2023–26 (163 major and extreme events × 12 books = 1,956 book-events, with the live user-event labels): the interval score worsens by **+0.19** normal-noise units (90% interval +0.08 to +0.32; the new pool is better in 0.1% of resamples). Coverage stays fine (93.4% → 92.4%).
- **On the history-average path (the range tests #183, #185, #186), it's roughly a wash, slightly better against the raw LLM.** Worsey's lead over the naive LLM grows from $29 to $34 per book-event (books) and from $37 to $43 (summed over positions). Calibration barely moves: 9-in-10 coverage 2026 89.8% → 88.7%, 2016–25 91.1% → 90.0%. Its score against predict-zero is a hair worse (CRPS skill 2016–25 +0.002 → -0.000).
- **Decision, by the predeclared rule:** not shipped. The rule requires "no worse on interval score at 90% support"; the live path is worse with 99.9% support. The code stays behind a switch (`WORSEY_DRAW_POOL=category`) for a better version later (last section).

## The what-if path, by category (interval score in noise units; lower is better)

| category | book-events | coverage now → new | score now → new | new − now (90% interval) |
|---|---|---|---|---|
| central_bank_unscheduled | 24 | 88% → 88% | 5.48 → 5.69 | +0.21 (-0.06 to +0.47) |
| conflict_war | 624 | 96% → 96% | 4.33 → 4.33 | -0.01 (-0.10 to +0.08) |
| disaster | 228 | 95% → 96% | 3.53 → 3.69 | +0.16 (+0.07 to +0.26) |
| elections_politics | 168 | 94% → 94% | 4.15 → 4.35 | +0.20 (+0.11 to +0.28) |
| financial_stress | 72 | 90% → 89% | 5.35 → 5.72 | +0.37 (-0.28 to +0.94) |
| health_pandemic | 24 | 79% → 96% | 4.06 → 3.71 | -0.35 (-1.07 to +0.38) |
| regulatory_court | 108 | 95% → 96% | 3.79 → 3.89 | +0.10 (-0.18 to +0.27) |
| sanctions | 108 | 94% → 96% | 4.08 → 4.13 | +0.05 (-0.17 to +0.25) |
| supply_shock | 252 | 94% → 94% | 6.53 → 6.55 | +0.02 (-0.07 to +0.09) |
| trade_tariffs | 348 | 89% → 80% | 6.17 → 6.92 | +0.75 (+0.19 to +1.42) |

- **Trade tariffs** get much worse (coverage 89% → 80%): 2025's tariff shocks don't look like earlier tariff events, so a pool of past tariff events gives the wrong co-movement out of sample.
- **Conflict and supply shocks**, where the gain was expected, don't change: their category pools look like the severity pool.
- **Pandemics** improve (coverage 79% → 96%), on only 24 book-events.

## Pool sizes

- Every minor, moderate and major (category, severity) cell has at least 20 past events (30 of 40 cells; smallest 34). No extreme cell does, so extreme events use their category's major + extreme pool (41–137 events).
- Pools used in the what-if test: conflict_war, major 576; trade_tariffs, major 312; supply_shock, major 228; disaster, major 204; elections_politics, major 168; regulatory_court, major 108; sanctions, major 108; financial_stress, major 60; conflict_war, major or extreme 48; trade_tariffs, major or extreme 36; health_pandemic, major 24; disaster, major or extreme 24; central_bank_unscheduled, major 24; supply_shock, major or extreme 24; financial_stress, major or extreme 12.
- The watchlist's market candidates have no category in their labels, so they'd stay on the severity pool even if this shipped.

## Why it fails, and what might work

- **Small pools overfit the past.** A scenario's whole spread is resampled from 34–137 past misses instead of ~730. Their correlations are noisy, and a category's next events (tariffs in 2025) can behave unlike its past ones.
- **In the what-if path the widths come from the label**, so the pool only shapes the correlation, and the label's directions already carry most of the category's pattern. The extra category correlation mostly adds noise.
- **A shrunk version** (draw from the category pool with probability n/(n + k), else the severity pool, with k chosen on 2016–22 and tested on 2023–26) is the natural next try. So is matching misses by label similarity (past events whose direction labels look like this one's) instead of by category.
