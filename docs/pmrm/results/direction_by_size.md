# Direction by size: is it a coin flip everywhere?

Run 2026-10-22. Script: `scripts/planc_direction_by_size.py`. Raw numbers: `direction_by_size.json`. No LLM calls (existing runs only). DECISIONS #189.

The trainer: "I can't for the life of me understand why we can't predict direction for a given factor on a given event. If it's any sort of meaningful event it should be obvious no? Why the coin flip?"

## In plain words

**The test.** For every forecast we already have, did it get the direction right? Broken down by how big the real move turned out to be, in units of that holding's normal noise. If direction is only a coin flip because most moves are small, the hit rate should climb as moves get bigger.

| Real move size (× normal noise) | < 0.5 | 0.5–1 | 1–2 | 2–3 | > 3 |
|---|---|---|---|---|---|
| **Naive LLM, 19 positions (2026)** | 46% | 52% | **57%** | **62%** | **75%** (n = 12) |
| **Naive LLM, 12 portfolios (2026)** | 48% | 50% | **62%** | **73%** | 50% (n = 8) |
| Engine, 13 factors (2023–25) | 51% | 51% | 50% | 56% | 54% |
| Engine, 13 factors (2026) | 49% | 51% | 47% | 57% | 50% |
| Engine, 12 portfolios (2023–25) | 52% | 53% | 51% | 49% | 44% |
| Engine, 12 portfolios (2026) | 49% | 55% | 60% | 55% | 44% |
| *Best possible with a perfectly known effect of 0.1 / 0.3 / 0.5 / 1 / 2× noise* | *54%* | *62%* | *69%* | *84%* | *98%* |

90% error bars are about ±2–5 points for the small bins, ±6–13 for 2–3×, and ±10–29 for above 3× (few cases).

**What it shows:**
1. **Your intuition is right for the LLM.** When a holding really moves (1–3× its normal noise), the naive LLM calls the direction 57–73% of the time, and 75% above 3×. On small moves it's a coin flip (46–52%), because those moves are mostly noise. Averaged over everything, it looked like 53%.
2. **The engine has no sense of direction even on big moves:** 44–57% at every size. Its direction comes from a regression on a few hand labels (category, severity, six direction channels), which can't tell "oil supply disruption in a tight market" from "oil supply disruption already priced in". The LLM reads the event and knows what it means.
3. **So the earlier conclusion, "nobody predicts direction", was too broad.** The LLM has real directional knowledge for meaningful moves. That knowledge gets drowned out when it also commits to a direction on the many events that don't matter, and its ranges are overconfident (#185, #186).

**Caveat:** this is conditioned on the real move's size, which you don't know in advance. It shows the LLM's direction is right when a move turns out to be real, not that we can tell beforehand which events will be real. That's the gate's job (PLAN_C §2b).

**What it means for Plan C:**
- **Direction knowledge lives in the text, not in our labels.** That's the core bet of Plan C (embeddings of the event text) and of using the LLM as an input builder. An LLM-derived direction per factor, as a feature (code still makes every number), is now a strong candidate input. It's added to the feature catalogue (§2a, group 4b).
- **The gate decides when to trust it:** the gate predicts "this event will move this target beyond noise", and direction features matter only when it's open. That combines the LLM's directional sense on real events with honest noise ranges everywhere else, which no arm does today.
- **#180 is consistent with this.** Letting the LLM override every factor bucket hurt, because most of those calls were on noise. The fix is gating, not discarding the LLM's direction.
