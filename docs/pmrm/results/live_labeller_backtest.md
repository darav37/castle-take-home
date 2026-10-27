# Backtest of the labelling prompts the product runs (DECISIONS #236)

Script: `scripts/live_labeller_backtest.py`. Raw numbers: `live_labeller_backtest.json`. Task: the design brief.
Cost: $5.68 (L1 $2.09, L2 $3.59; the dry run said $4.09, L2's outputs ran longer). L3 and #196 were scored from labels that already existed.

## In plain words

- **The two labellers that price events in the chat are as good as the research prompt.** On the 745 major and extreme library events, from the event's text alone, the market labeller (L1) calls the direction of big factor moves (over 2× normal noise) right **74%** of the time (90% interval 70%–79%), the user-event labeller (L2) **73%** (68%–77%); #196's research read scores 75% on the same events. The intervals overlap.
- **Their size calls separate.** The realized move grows with the bucket for both, in order (none < small < medium < large): L1 0.90 / 1.55 / 2.70× normal noise for small / medium / large, L2 0.84 / 1.30 / 2.53×. A 'large' call means a move about 3× a 'small' one.
- **The news labeller (L3) is clearly worse:** 65% (60%–70%), and only 53% on credit. Its prompt never defines the factors ("credit" can be read as spreads or as bonds), and its credit signs agree with L1's defined credit only 71% of the time. It has no sizes, so size can't be scored.
- **The post-cutoff (clean) set is too small to settle anything:** 32 events from Jul 1 to Sep 30, 2026, with only 4–5 signed moves over 2× per labeller. On moves over 1× noise, L1 gets 70% right (n 20, interval 53%–86%) vs 71% before the cutoff: no sign that remembered outcomes inflate the main numbers, but the interval is wide.
- **The prospective log** has 15 scored stories, all scored after their windows closed: a plumbing check, not evidence.

## Direction on big moves (over 2× normal noise), 745 major and extreme events

| labeller | right | 90% interval | moves scored | events labelled | share of factors given a sign |
|---|---|---|---|---|---|
| L1 market labeller | **74%** | 70%–79% | 341 | 720 | 52% |
| L2 user-event labeller | **73%** | 68%–77% | 423 | 745 | 71% |
| L3 news labeller (signs only) | **65%** | 60%–70% | 318 | 606 | 74% |
| #196 research read | **75%** | 71%–80% | 391 | 743 | 58% |

### By factor (right on big moves; n in brackets)

| factor | L1 market | L2 user event | L3 news | #196 research |
|---|---|---|---|---|
| US stocks | 77% (66) | 79% (75) | 75% (57) | 78% (74) |
| 2-year yields | 71% (45) | 71% (56) | 57% (51) | 66% (61) |
| 10-year yields | 60% (48) | 62% (64) | 64% (45) | 64% (53) |
| dollar | 68% (38) | 74% (43) | 69% (32) | 80% (40) |
| oil | 78% (50) | 77% (61) | 75% (40) | 83% (52) |
| gold | 80% (40) | 62% (60) | 63% (46) | 71% (52) |
| credit | 83% (54) | 83% (64) | 53% (47) | 86% (59) |

### By the real move's size (all signed calls)

| real move (× noise) | L1 | L2 | L3 | #196 |
|---|---|---|---|---|
| <1x | 56% (1632) | 54% (2396) | 52% (2083) | 55% (1888) |
| 1-2x | 69% (608) | 65% (829) | 61% (696) | 69% (699) |
| 2-3x | 70% (198) | 70% (258) | 62% (202) | 72% (232) |
| >3x | 81% (143) | 77% (165) | 70% (116) | 81% (159) |

## Size calls: the realized move per bucket (× normal 2-day noise)

| bucket | L1: mean move (n) | L1: move in its direction | L2: mean move (n) | L2: move in its direction | #196: mean move (n) | #196: move in its direction |
|---|---|---|---|---|---|---|
| none | 0.77 (2410) | – | 0.80 (1515) | – | 0.78 (2171) | – |
| small | 0.90 (2053) | +0.21 | 0.84 (2743) | +0.13 | 0.86 (2276) | +0.15 |
| medium | 1.55 (448) | +1.03 | 1.30 (800) | +0.74 | 1.33 (542) | +0.85 |
| large | 2.70 (80) | +2.61 | 2.53 (105) | +2.28 | 2.81 (160) | +2.55 |

Ordering monotonic: L1 True, L2 True, #196 True.

## Before vs after the model's knowledge cutoff (Jul 1, 2026)

| labeller | before: right on big moves (n) | before: right on moves > 1× (n) | clean: right on moves > 1× (n, 90% interval) |
|---|---|---|---|
| L1 market labeller | 74% (337) | 71% (936) | 70% (20, 53%–86%) |
| L2 user-event labeller | 73% (420) | 68% (1238) | 62% (26, 42%–83%) |
| L3 news labeller (signs only) | 65% (314) | 62% (999) | 52% (27, 32%–74%) |
| #196 research read | 75% (387) | 71% (1077) | 77% (13, 50%–100%) |

#196's clean set is 12 events: its research read covered major and extreme events only.

## What to change (not changed in this task)

- **L3, the news labeller:** add the factor definitions L1 and L2 already carry ("rates = Treasury yields, + = up; credit = high-yield minus Treasuries, + = spreads tighten") to `planb_grow.PROMPT`. That's the smallest change likely to close most of the gap; credit (53%) and 2-year yields (57%) are its worst factors. In the chat its labels give news candidates their signs (sizes come from severity), so they move numbers in the watchlist.
- **L1 and L2:** no change needed. L2's gold (62% vs 71% for #196) is its weak spot; L2 also gives a sign more often (71% of factors vs 52–58%), which adds calls on small moves without hurting big-move accuracy.

## How it was run

- **Events:** text only (title, category, region, description), never an outcome. Main: the 745 major and extreme C0 events (#196's set). Clean: the 32 C0 events dated Jul 1 – Sep 30, 2026 (any severity).
- **L1:** `discovery_label.PROMPT`, unchanged; each event passed as a market titled "This happens: <title>", its description as the rules, 25 per call.
- **L2:** `discovery.PROPOSE` + `USER_BLOCK`, unchanged; the event passed as a user's event with n = 0 proposals, the trainer's example book as the book, no listed items, no history, today = the event's date, 8 per call (live passes up to 4).
- **L3:** labels already in the library from `planb_grow.PROMPT` (the prompt the prospective log uses), Wikipedia-sourced events only; signs for 7 factors, no sizes.
- **Scoring (as #196 / #214):** realized move over the report day and the next session ÷ the factor's normal 2-day noise; direction = % right where the label has a sign; 90% bootstrap intervals clustered by event.
- **Book-level check:** skipped (optional in the task).
