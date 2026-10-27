# The shipped chat vs an untrained LLM, on 2026 events

Run 2026-10-21. Script: `scripts/planb_product_vs_raw.py` (run, score). Raw numbers: `product_vs_raw.json`; runs in `cache/pmrm/product_vs_raw/runs.json`. DECISIONS #180.

**Cost:** $1.46 (product loop 124 calls, $1.16; untrained LLM 31 calls, $0.30), capped at $3 per run.

The trainer: "is what we have better than just having an untrained LLM answer this question?" … "Yes please that would be a great comparison."

## In plain words

**The test.** We took all 31 major and extreme events of 2026, all after the LLM's training cutoff, and asked about each one two ways:
- **The shipped product:** the exact chat path (Plan B expert prompt, `event_evidence`, its bucket judgments, `price_scenario`) on the demo book. The engine was put in cutoff mode: the event model was refitted on pre-2026 events, and the evidence came only from earlier events, so nothing it saw knew the outcome.
- **An untrained LLM:** the same model, given the event, the 8 factors and what the buckets mean, asked to reason and answer.

We also scored **the engine alone** (no LLM judgment), using the product's own description of each event.

**Result: on these 31 events the shipped product did not beat the untrained LLM.**

| On the 8 factors | R² vs zero | Market direction |
|---|---|---|
| Shipped product | −7.3% | 55% |
| Untrained LLM | −7.4% | 48% |
| Engine alone | **+1.8%** | 48% |

The product vs the untrained LLM is a coin flip: the 90% range of the difference is −11 to +13 points, and the product wins in 51% of resamples. The engine alone is the only one above zero.

**On real books** (each book's actual return over the event window):

| Book | Product | Untrained | Engine alone |
|---|---|---|---|
| Balanced (demo) | −24.9% | −9.3% | **+7.1%** |
| Energy producers | −19.1% | −25.3% | **+5.7%** |
| Gold miners | **+3.9%** | +2.9% | −0.9% |
| Latin America | −14.9% | −1.7% | 0.0% |
| Long Treasuries | −52.3% | +6.1% | **+12.4%** |
| Agriculture / weather | −12.5% | −0.5% | +3.0% |

**Why.**
- **Both LLMs call directions that don't hold.** Each one moves off "flat" on 84–94 of 248 factor calls. The real moves were non-flat 129 times, but their direction wasn't predictable. Stocks (product −115%, untrained −80%) and credit (−169%, −407%) are where the calls go most wrong.
- **The product's judgment helps on oil.** It scored +32%, against +18% for the untrained LLM and +7% for the engine alone. It also leads on utilities vs energy producers and gold miners.
- **The engine never calls a bucket** (its estimates are small) and comes out slightly ahead, consistent with "most events don't move markets" (#162, #171).
- **The product's expert rated these "major" library events lower** (18 moderate, 7 minor, 6 major). That matches #175: 2026's "major" events moved the market less than ordinary ones.

**How this squares with earlier results.** The research harness found the assisted LLM ahead of the raw LLM after the cutoff (#170: +2.5% vs −5.7% on 92 events; #176 blinded: +5% vs −20% on these 31 big events). The shipped loop is a port of that setup with a new prompt, tools and output, and on this test it loses the edge. Two differences stand out:
- the shipped expert writes a full reply and is told to judge the factors itself, while the research prompt asked only for buckets beside the evidence
- the untrained baseline here is a fair, well-framed prompt, not the plain one (#171 showed framing alone closes most of the gap)

**What it means.**
- The shipped product isn't yet better than an untrained LLM at calling factor moves on new events. Its edge on the numbers comes from the engine: book-specific exposures, calibrated ranges, and defaulting to "little effect".
- The LLM's judgment adds value on specific channels (oil, energy vs utilities). It costs value on the broad market, credit and rates.

**Options:**
1. **Lean on the engine.** Price with the engine's estimate for market, rates and credit, and use the expert's judgment only for oil, gold and utilities vs producers. Those are the factors where both this test and #170 found the LLM ahead. This is a rule from prior evidence, but it should be confirmed on fresh events (the daily prospective log) before calling it validated.
2. **Shrink the judgment.** Blend the expert's buckets halfway toward the engine, and test prospectively.
3. **Test on more events first:** all 92 events of 2026 (about $3–4) to narrow the ranges.

Tuning on these 31 events and re-scoring on the same 31 would be overfitting, so any change needs fresh events.

## Update, Oct 22 (#182): numbers only from code

The trainer ruled that the LLM no longer adjusts any number. The shipped path now prices every factor with the engine's estimate, and the LLM only identifies the event, fills the schema, matches a market and explains. Re-scored on the same 31 events (no new LLM calls; the product's own schema from the run above):

| On the 8 factors | R² vs zero |
|---|---|
| **Shipped product (#182)** | **+1.8%** (identical to the engine alone, checked) |
| Product as in #180 (LLM buckets) | −7.3% |
| Untrained LLM | −7.4% |

Shipped vs untrained: +8.8 pts (90% −3 to +25; ahead in 88% of resamples). On books the shipped product is the engine: balanced +7.1%, energy +5.7%, Treasuries +12.4%, agriculture +3.0%, Latin America 0.0%, gold miners −0.9%.

