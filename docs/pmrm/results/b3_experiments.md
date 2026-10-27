# Plan B experiments on the grown library (Oct 21)

: "I want to try a bunch of different things now that we have such a rich dataset." No LLM calls in any of these.

**Data:** the 3,487-event library (`b2_library.md`, `b3_impact.md`), the B1 factor model, and the extended named factors below.
**Raw numbers:** `b1.json` (`extended`) and `b3_targets.json`.
**Scripts:** `scripts/planb_b1_extended.py`, `scripts/planb_b3_targets.py`.
**Rules:** every test is leave-one-out or held out, and every result is compared with a measured chance rate (shuffled labels or targets).

## In plain words

1. **A better named-factor model.** Four factors the statistical model found on its own are now named factors: Asia/emerging markets, healthcare, agriculture, and defensives vs commodity producers. The named model now explains **62.7%** of daily moves on unseen years, up from 58.2%. That beats the old hybrid, and every factor has a plain name a user can understand.
2. **Events hit specific factors, not "the market".** Once each factor is scored on its own, several event types that looked like noise show real, predictable effects:
   - **Tariffs:** the market, healthcare and credit.
   - **Sanctions:** defensives vs commodity producers.
   - **Supply shocks:** oil.
   - **Big events generally:** credit, oil, healthcare, the market, and defensives vs producers, about 4% each.
   - **Wars and disasters:** still almost nothing on any broad factor.
3. **Big events hit their targets harder.** The sectors and countries an extreme event targets move 1.6× more than the other sectors and countries that day; for major events 1.24×. Elections move their own country's market 1.3× more; foreign central-bank surprises move their country 1.26× more. Tariffs, sanctions, wars and disasters show no extra move in the sectors the labels name.
4. **The direction of a targeted move is predictable, a little.** From past events of the same type and direction on the same target: **+2.9% of the beyond-factor move explained** (beats chance, p = 0.01), once small groups are pulled toward zero.

**For the product:** damage to a book now has three measurable parts, each tested:
- the book's exposure to the 13 named factors
- how each event type moves those specific factors
- the extra hit on the specific sectors and countries big events target

Effects are small and concentrated in big events, but they're real, and each comes with a plain-language name.

## 1. Extended named factors (B1 revision)

Common scored set: 91 panel ETFs (the new anchor ETFs EEM, XLV, DBA, XLU and XLE are excluded for every version). Test window 2021-01 to 2026-10.

**New factors:**
- asia_em = EEM − SPY
- healthcare = XLV
- agriculture = DBA
- defensive_vs_producers = XLU − XLE

| Version | Test R² |
|---|---|
| Latent (9) | 58.2% |
| Named 9 | 58.2% |
| Named 9 + 3 latent (hybrid) | 61.7% |
| **Named 13** | **62.7%** |
| Named 13 + 3 latent | 64.4% |

**Test R² lost when each named factor is dropped (named 13):** market 8.4 pts, value 3.3, **asia_em 2.8**, 10y yield 2.6, gold 2.2, dollar 2.2, size 1.8, oil 1.8, 2y yield 1.7, **healthcare 1.0**, agriculture 0.4, defensives vs producers 0.4, credit 0.1 (credit is almost redundant once the others are in).

## 2. Which factors each event type moves

**Method:**
- **Response:** the event's short-window move on each of the 13 named factors.
- **Model:** leave-one-out ridge on the direction labels plus category.
- **Score:** R² vs zero, factor by factor.
- **Chance:** labels shuffled across events, 100 times.
- **Shown:** factors with p ≤ 0.05 and R² > 0.

| Events | n | Factors with real skill (LOO R²) |
|---|---|---|
| All | 3,487 | credit +1.4%, market +0.9%, healthcare +0.8%, oil +0.3%, defensives +0.3% |
| **Extreme or major** | 677 | credit +4.6%, healthcare +4.4%, market +4.3%, oil +4.1%, defensives vs producers +3.6%, agriculture +2.0%, dollar +0.8% |
| Financial stress | 420 | credit +4.4%, healthcare +4.0%, market +2.9%, agriculture +2.0%, oil +1.6%, dollar +1.4% |
| Health / pandemic | 150 | market +3.8%, credit +3.6%, healthcare +3.3%, oil +2.7%, defensives +2.3%, 2y yield +1.5% |
| **Trade / tariffs** | 184 | healthcare +3.6%, market +3.2%, credit +1.8% |
| **Sanctions** | 293 | defensives vs producers +2.6% |
| **Supply shock** | 354 | oil +2.5%, defensives +1.7%, credit +1.1% |
| Central bank (unscheduled) | 171 | market +2.0%, dollar +1.4% |
| Conflict / war | 390 | value +1.6% |
| Elections / politics | 579 | 2y yield +0.9% |
| Regulatory / court | 569 | agriculture +1.0%, credit +0.6% |
| Disaster | 377 | none |

## 3. Targeted sectors and countries

For each event, every sector and country ETF's beyond-factor move (realized minus its 13-factor prediction, loadings fitted 2011–2020) over the short window. "Targeted" = the sector ETFs of the event's labelled sectors, plus its region's country ETF.

**(a) Do targets move more?** Mean |beyond-factor move| of targeted ETFs ÷ that of the other ETFs in the same window, averaged over events. Chance: targets shuffled within the event, 100 times.

| Events | n | Ratio | Chance 95th | p |
|---|---|---|---|---|
| All | 3,202 | 1.00 | 1.05 | 0.88 |
| **Extreme** | 39 | **1.61** | 1.31 | 0.01 |
| **Major** | 597 | **1.24** | 1.08 | 0.01 |
| **Elections / politics** | 426 | **1.29** | 1.10 | 0.01 |
| **Central bank (unscheduled)** | 165 | **1.26** | 1.14 | 0.01 |
| Financial stress | 394 | 1.09 | 1.09 | 0.03 |
| Conflict, sanctions, tariffs, disasters, supply, regulatory, health | 148–541 each | 0.86–0.97 | about 1.1 | n.s. |

**(b) Is the direction predictable?**
- **Method:** each targeted ETF's beyond-factor move predicted by the leave-one-out mean of other events with the same category, risk direction and target. 6,803 event-target pairs; chance by shuffling the moves.

| Variant | R² vs zero | Chance 95th | p |
|---|---|---|---|
| Plain group mean | −19.2% | −8.2% | 0.93 |
| **Shrunk** (sum / (count − 1 + 5)) | **+2.9%** | −1.9% | 0.01 |
| Shrunk (sum / (count − 1 + 20)) | +2.6% | −0.5% | 0.01 |

Sign right 56% of the time. The shrinkage constants were fixed before scoring; both pass.

## 4. What changed in the picture

- **Named factors:** extend from 9 to 13. Simpler to explain and better out of sample than the old hybrid.
- **Tariffs, sanctions and supply shocks are not noise.** They move specific factors (market, healthcare and credit for tariffs; defensives vs producers for sanctions; oil for supply shocks), which the pooled test hid.
- **Wars and disasters remain the weakest category:** almost no consistent effect on any factor or target in the first days. Their impact is probably event-specific (which oil facility, which country) or slower than 5 sessions. B4's comparison step and longer windows are the next things to try there.
- **The LLM's sector labels for tariffs, sanctions and wars don't pick out the ETFs that move.** The country labels for elections and central banks do. A better "who is targeted" mapping (industry-level, companies with revenue exposure) is a candidate improvement.

## Next experiments queued

1. **Longer windows (10 and 20 sessions)** for wars, sanctions and disasters: slower effects?
2. **Target mapping from holdings data** (companies with revenue in the affected country or industry) instead of the LLM's sector list.
3. **Scheduled releases on the 13 factors,** with a regime switch (pre/post-2021).
4. **Finish the last 34 months** of the library (2024–2026, about $3–4), then rerun everything.
