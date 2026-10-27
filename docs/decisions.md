# Decision log

Newest at the bottom. Each entry gives the decision, the reason, and the alternatives considered.

## 2026-10-14 — Planning

This is the curated log: the decisions that define the design, the tests and the main pivots, with their original numbers (so `#N` references elsewhere resolve; gaps are entries kept in the research notes).

1. **Options 1 + 4, bridged.**
   - **Decision:** build Options 1 and 4 as one agent ("Worsey", a mini Percy) on a dummy multi-asset portfolio with real prices.
   - **Why:**
     - Option 1 is the stated first ask for the role.
     - Option 4 is where the trainer's trading edge shows.
     - Bridging them shows a product, not two answers.
   - **Alternatives:** Option 1 alone (rejected: leaves the trainer's options strength unused); a CLI report with no agent (rejected: doesn't match how Castle's product works).

2. **Prediction-market probabilities as factors.**
   - **Decision:** estimate an event β for each position by regressing returns on Δp.
   - **Priced-in adjustment:** P&L if YES = β(1−p), P&L if NO = −βp.
   - **Ranking:** by the standard deviation of resolution P&L, |β|√(p(1−p)).
   - **Why:** this answers the question of how much today's prices already carry with a number instead of hand-waving.

3. **Event-first, not asset-first.**
   - **Decision:** build a positions × events β matrix.
   - **Why:** most asset variance is market-wide or idiosyncratic, so assets shouldn't be decomposed into events.
   - **ETF look-through** is used for concentration, mapping name- and sector-specific events, and explanation. Macro β is estimated at the ETF level.

4. **Three β tiers.**
   1. Empirical, pooled across Kalshi recurring series.
   2. Two-stage: event → macro factor shocks, then asset → factor loadings.
   3. Claude prior (sign and size bucket), used as shrinkage.
   - **Why:** most events lack usable market history. Tier 2 shrinks the LLM's job to a few macro numbers.

7. **Claude chooses, code computes.** The LLM parses, matches, routes and explains. Every number comes from code, enforced by a grounding check on replies.

16. **Finding: the long-call bar is the breakeven, not the implied move.**
    - **What:** pricer tests show an ATM call breaks even near half the straddle (≈ 0.4·σ·√T). A +5% view against a ±9% implied move is mildly profitable on an ATM call; a +3% view loses.
    - **Decision:** the "teaching moment" is triggered by E[P&L] under the user's own distribution being negative, computed in code, not by comparing the view to the implied move.
    - **Alternatives:** the spec's original "view < implied move" heuristic (rejected: gives the wrong answer for views between the breakeven and the implied move).

19. **Finding: tier-1 β has little power for small effects.**
    - **What:** on synthetic data, β = 0.05 with 1% daily noise over 400 days gives t ≈ 1.5; β ≈ 0.2 is recovered cleanly.
    - **So:** tier 1 is trusted only for large, jumpy markets and pooled recurring series; most events rely on tiers 2–3. Shrinkage (k = 4) keeps a t = 1.5 estimate at about a third of its size.

22. **Matching rules (code, after Claude parses both sides into the schema).**
    - Variable and jurisdiction must agree; comparator direction must agree (the complement at the same threshold is allowed, flagged).
    - A threshold mismatch is derived only if a strike ladder spans it (no extrapolation); ladders are made monotone by isotonic regression first.
    - A deadline mismatch over 3 days is rescaled by constant hazard only if T′/T ≤ 3×; beyond that it's no fit.
    - **Alternatives:** letting Claude decide the outcome directly (rejected: not reproducible or auditable).

27. **Agent loop as built.** A manual loop on the Messages API (~100 lines, no framework), working on plain dicts so record/replay is byte-exact. Router tools: load_portfolio, exposures, compare_hedges, chart, plus the two experts as tools. Experts get a fresh context with only the session slice they need, and return JSON. Validation errors go back to the model with one retry per tool per turn. History is append-only.

28. **Grounding check.** Every number in a reply must match a tool-result number this turn (allowing $4.1k-style rounding to ~2 significant figures, fractions shown as %, bp), or the user's own words. Integers ≤ 10 and years are exempt. A failing reply gets one rewrite; if it still fails, the CLI prints the unverified numbers next to the reply instead of hiding them.

29. **Models.** Router `claude-haiku-5-5` (low effort), experts and train mode `claude-sonnet-5-5`, per the §11a tiering. Env-overridable.

31. **Train mode guardrail in code.** `propose_edit` can only queue files under `library/` or `examples/events/` (path-resolved, so `..` tricks fail). Nothing lands until `worsey review`, which shows the diff and git-commits only accepted files.

45. **The scenario engine is the centerpiece, with its own methodology doc.** `docs/research/methodology_v1.md` is authoritative. The options engine is the secondary centerpiece, organized around the interview and worked-example format.

61. **ETF holdings + look-through per C11 (fix 3).** Holdings = the latest `effective_date` snapshot only, weights as published. Each ETF is truncated to its top holdings covering ≥ 90% of weight; the residual goes to an "other holdings (ETF)" bucket and cash/futures sleeves to "cash (via ETFs)", so no name is ever overstated. Only equity ETFs are looked through.
    - **Alternatives:** renormalizing the truncated holdings (rejected by the trainer: overstates every kept name); looking through bond ETFs (rejected: 1,340 bond lines say nothing about name concentration).

80. **Router back to Sonnet, by the trainer's rule on the pooled eval (F23).** Haiku was faster but missed the same §0 demo turn in all three runs. The eval now pools 3 repeats, since one run of 30 flips on a single case. Model stays a config value.

83. **Grounding fails closed; derived numbers come from code.** The check now parses multiples ("9x", "9×", "9 times", spelled-out "nine times", "twice", "double") and percentages, with no small-number exemption for multiples, and the ±5% fuzzy match is gone (display rounding only). Tool outputs carry the derived figures a reply needs: hedge `cost_to_loss_multiple`, `cost_vs_cheapest_multiple`, `cost_pct_of_book`; event `pnl_yes_pct_of_book`, `sigma_pct_of_book`, `sigma_vs_top_multiple`; strategy `cost_vs_first`, `cost_minus_first`. A failing reply gets one rewrite; then every sentence, bullet or table row still carrying an unsupported number is stripped and the CLI shows what was stripped.

99. **Live chat with hard limits on live calls.** Castle's keys have their own limits; ours sit in front of them, in code, on every live call path (chat, server, build, train). See SPEC §1d-0 "Live chat".
    - **Limits:**
      - per turn: 15 calls
      - per rolling hour: 120 calls
      - per UTC day: 400 calls and 3M tokens
      - one live turn at a time
      - 6 questions per person per minute
      - 1,000-character messages
      - a kill switch (env var or STOP file)
      - SDK: 1 retry, 120s timeout
    - **Ledger:** every call goes to an on-disk ledger, so a restart can't reset the daily limits.
    - **Market data:** live chat uses the cached build by default, so Massive/Kalshi get no calls unless `--live-data`.
    - **Why these numbers:** a demo turn uses 1–5 calls today (up to about 8 with experts back), so 15 per turn stops only a loop. 400 a day is ample for a day of testing.
    - **Alternative:** rely on Castle's key limits alone. Rejected: those protect Castle's bill, not our share of it, and they fail loudly mid-demo.
    - **Finding:** a pre-loaded portfolio was invisible to the agent ("I don't have your portfolio yet"). Code now adds a note to the conversation saying it is loaded, with its contents.

100. **Expert ownership.** See SPEC §1b "Amended Oct 17".
    - **Hedge requests run both experts.** A broad macro or portfolio hedge request runs `event_analyst` and `options_desk` in parallel; the router presents their two answers as two options, unless the user says they want specifically an options hedge or an event hedge.
    - **Experts write the replies.** Same voice and format rules; grounding and format checks run on what they write. The router writes only simple turns, and on hedge turns it presents the two expert answers side by side without rewriting them.
    - **`equity_analyst` is Tier A.** It must be there; its knowledge is to be "context trained".
    - **The router's view parsing (Oct 16, #89) goes back to `options_desk`,** because the 60s budget (#98) removes the reason for it.
    - **Open, for the build:**
      - the equity analyst's knowledge sources (see the finding below)
      - each expert's prompt, caps and output shape
      - re-recording the three scripts
    - **Finding, equity research for training:** most sell-side equity research is paywalled and licensed, not something we can collect at scale. Freely available, citable sources:
      - SEC EDGAR filings (10-K/10-Q: segment and geographic revenue, risk factors, customer concentration; 8-Ks for events)
      - Massive, already on our plan (Benzinga news, income statements, earnings dates and history)
      - company IR pages and earnings-call transcripts (where public)

      The proposal is a cited company profile per name, built at `worsey build` from filings and Massive, plus the trainer's train-mode reviews.

109. **Replies post when ready; no cancel-and-restart; two state scopes.** Replaces §1e's seq-ordered hold and cancel-and-restart. See SPEC §1e.
    - **Ordering:** a reply posts as soon as it's ready. If it isn't answering the latest message, it quotes the request and @mentions the sender, naming the point when the message had several ("on the Iran point, @darin …"). Only acks are ordered. Why: the 20s hold was designed for the old 15s budget (#98); at 60s it makes fast answers wait, and people in group chats reply out of order anyway.
    - **Edits:** no cancellation. A follow-up mid-request waits for the original, then goes to the warm expert (#108). Another member's change to someone's request becomes their own request.
    - **State:** chat and user only. A separate thread scope was proposed and dropped by the trainer as too much complexity; warm sessions are keyed by member instead of thread.

110. **Privacy enforced by construction.** See SPEC §1e.
    - **Rules:** group replies show normalized numbers only (per $100k, %); actual dollars and holdings go to DM even when the asker asked, unless they say "post it here" or opt in per chat. An expert's context holds only the asker's book; group questions get members' slices as percentages computed in code. Grounding fails any dollar figure, or another member's holding, in a group reply.
    - **Why:** a prompt rule can be broken by the model; data the model never sees can't leak. And asking in the group isn't consent to broadcast your book.
    - **Alternative:** the Oct 15 rule (post the asker's dollars if they asked). Rejected for the consent reason.

111. **Member profiles: a context file per member.** Worsey learns each member's preferences, humor and knowledge, and knows their book and trades over time, and speaks to them through that. Design in progress in design review. See SPEC §1e-1.

112. **No mesh: experts ask for other lanes, code makes the call.** An expert returns `needs: [{expert, question}]`; the orchestrator dispatches it one hop, in parallel, within budget, answered by the warm session (#108). See SPEC §1b.
    - **Why:** the essay's case for meshes needs long-running work, roles that emerge on their own, and a root that fills up. Worsey has a 60s turn, four roles designed on purpose, and a code root. A mesh would bring the costs the essay names (duplicated work, communication paths that grow with team size, loops) under a 25-call cap, and lose §3.4h's testable coverage.
    - **Alternative:** direct expert-to-expert calls. Revisit if autopilot or proactive monitors ever run for hours.

114. **Experts accumulate knowledge over time without changing their principles.** Each expert must grow with how the market and the portfolios evolve, and never forget its core grounding principles.
    - **Design (agreed Oct 17):** four layers by precedence: principles (fixed; the trainer only) > episodic log (code-written; outcomes computed from prices) > lessons (cited, minimum count, expiring, advisory) > member profiles and warm sessions. Book history is a code tool. Only the trainer promotes a lesson into the library. See SPEC §1b.
    - **Why:** an expert that rewrites its own prompt drifts and learns from noise (one expensive put spread becomes "puts are bad"). A grounded log plus gated lessons gives the "years together" feel while staying auditable.
    - **Alternative:** experts edit their own prompts or knowledge files freely. Rejected for drift and ungrounded numbers.

120. **Member profile guardrails.** Only what the member said or did, no inferred psychology; book and trades from code; visible and editable in DM; written by a background pass after each turn; tone-only use in the group; sophistication from code rules on behavior replaces the #104 "asked by name" gate. See SPEC §1e-1.
    - **Why:** inferred traits are creepy and leak when quoted in a group; ungrounded P&L breaks the numbers rule; a behavior-based gate is safer than "asked by name".
    - **Alternative:** a free-form profile the model writes. Rejected for the reasons above.

146. **Build and test a successor to the event engine: the prediction-market response model (PMRM), spec `docs/research/pmrm_spec.md` v1.6.** The trainer's first-principles framework, reviewed with ChatGPT, Gemini and Claude across v1.2 to v1.6.
    - **What it is:** one joint low-rank factor model of daily prediction-market probability changes and stock returns. It answers one question: given an event's observed small odds move over a day, what is the conditional distribution of portfolio P&L? The LLM's role ("intelligence constraints", IC) is five jobs: J1 universe selection, J2 measurement cleaning and news attribution (the primary job), J3 priors, J4 Type II embedding, J5 factor changes and naming. The LLM picks labels, buckets and analogs; code computes every number.
    - **Why:** the current engine's own finding (METHODOLOGY §4.7) is that daily single-market β is mostly noise (about 1% of events reach book-level |t| ≥ 2), so its ranking is largely its tier-3 prior. The PMRM puts the LLM where it adds information the price panel lacks (what drove each large move), and makes every layer earn its place on a held-out proper score against matched ablations.
    - **Scope:** small moves only. Resolution impact ("what if X happens") is explicitly unsupported by the PMRM and stays with the current engine, which also keeps feeding the hedge comparator.
    - **Experiment program:** Study A, the toy universe (oil-record, recession and exactly-25bp Fed contracts; XOM, DAL, MSFT) end to end; then Study B, varying the universe (category pools, placebo stocks, stratified random universes from 2×2 up, a pooled primary with day-block uncertainty). One predeclared primary per study: cleaned M2c vs raw M2. The current engine enters as comparator `V1_current`, a direct head-to-head in the same units (dollars per unit of probability).
    - **Cost control:** every PMRM LLM call goes through `agent/budget.py` plus per-run caps (300 calls, $5), a dry-run estimate and `--confirm`, and a permanent label cache (SPEC §6.1).
    - **Finding:** Kalshi and Massive are not reachable from Claude's cloud sandbox (proxy refuses both), so the build runs on the trainer's machine.
    - **Alternatives considered:** keep refining the current engine (rejected for now: its core estimate is noise-dominated and its LLM prior is untested); replace it outright (rejected: the PMRM is unvalidated and does not support resolution scenarios, which the hedge flow needs); an eight-world terminal outcome table (rejected in v1.2: mixes resolution with daily covariance).
    - **Docs:** `docs/pmrm/` holds SPEC.md, BUILD_PLAN.md, SIMPLE.md, the v1.5 white paper and explainer PDFs (the model is unchanged in v1.6) and their sources; `scripts/pmrm_power_sim.py` is the reference power simulation.

151. **PMRM: answer the basic question first; new gating checkpoint P0c.** Before any factor model, J2 cleaning, priors or Study B work: does conditioning on prediction-market odds moves predict portfolio P&L better than ignoring them, on held-out days? One event at a time, all events jointly (shrunk covariance and ridge), and beyond-the-market, across the Study B event families and the stock-bucket portfolios, against measured chance rates (circular shifts). BUILD_PLAN P0c.
    - **Why:** P0b showed that up-weighting news days does not help, which is a secondary question. The primary premise, that daily event covariance carries held-out information about portfolio P&L, was never tested; P0b's 222-market table is in-sample. Rough bounds from P0b's panel correlations (recession about 0.01–0.04, Fed about −0.1) suggest single-event gains near or below the cost of estimating one slope on 120 rows, so a negative result is plausible and must be known before building on the premise.
    - **Gate:** real signal → P1/P2 proceed; none → report the negative result and pivot to release windows (Phase R3 / tier 1b); mixed → narrow the universe.
    - **Alternatives considered:** proceed to P1/P2 with stage 2a inside Study A (rejected: Study A's single 244-row panel is too small to answer the basic question; P0c uses every family and portfolio at once).

152. **Plan B is the main direction for Option 1; P0c stays as the naive baseline.** BUILD_PLAN checkpoints B0–B5, PLAN_B v0.2.
    - **What:** build the impact model from realized events and broad asset returns, without prediction markets: the LLM finds, labels and categorizes events into a fixed schema (code assigns windows and computes every number); a broad-universe factor model denoises returns; event → factor responses are learned and tested on held-out events and periods. Prediction markets enter at the end: the current probability for the event in question, anticipation (outcome − p) for recent training events, and the out-of-sample tests on Kalshi-era resolutions and on daily moves (B·Δp vs P0c's baseline).
    - **Why:** P0b found no news-day concentration and little daily signal; the identifying information arrives at resolution, and realized-event histories are decades long where Kalshi's is from 2022.
    - **Order:** P0c first, then B0 to B5. The small-move PMRM engine (P1 to P5) proceeds only if P0c finds real signal.
    - **Alternatives considered:** continue the small-move PMRM as the main path (rejected for now, pending P0c); LLM priors as the impact model (rejected: last resort only, weighted by measured skill).

153. **PMRM P0c: the basic question, answered (Oct 20; `docs/pmrm/results/p0c_basic_question.md`).** Findings; the gate is the trainer's.
    - **Signal is statistically detectable, economically negligible:**
      - **Excess over chance:** 30 of 272 cells (15 events or "all jointly" × 8 portfolios × raw/beyond-market) beat chance on held-out log score. The measured null of that count averages 10.8, 95th percentile 17.5, maximum 24.
      - **Size:** held-out R² < 1% in every cell (max 0.86%, median slightly negative); the best joint R² is 0.73%; gains are 0.002–0.010 nats/row.
      - **Mostly market exposure:** 19 of 136 cells significant raw vs 11 of 136 net of SPY (≈ 5.4 expected by chance); the joint model beyond the market wins in 1 of 16.
      - **Where:** Fed, CPI and the 2026 recession contract × cyclical and tech portfolios.
      - **Sanity:** placebo and defensive portfolios are negative under every model.
      - **Strict quotes:** same conclusion (30 of 208 cells vs null 8.2 / 95th 18.2; max R² 0.69%).
    - **Method fix during the build:** the first single-event model took its variance only from event-quoted days, which are calmer, so it could win without any covariance. It now conditions M0's own Gaussian through the training correlation.
    - **Recommendation:** treat it as "no usable signal" for the small-move premise. Keep the joint shrunk-covariance model as the naive baseline Plan B must beat (B5); don't start P1–P5. Consistent with #152.

156. **Plan B B1: the broad factor model (Oct 20; `docs/pmrm/results/b1_factors.md`).** No LLM calls.
    - **Panel and periods:** 96 ETFs with history from 2011; train 2011–18, validation 2019–20, test 2021–26-10. Scored cross-asset (factors from half the assets predict the other half), so no version grades itself.
    - **Test R²:** hybrid (9 named factors + 5 latent) **60.8%** > anchored (9 named) 57.2% > latent (9) 54.2%. Latent carries the equity structure; the named factors carry rates, FX and commodities.
    - **Stable:**
      - named-factor loadings correlate 0.8–0.98 across 3-year refits (credit and value drift most)
      - latent subspace similarity 0.73–0.90
      - 2003–10 loadings predict 2021–26 as well as 2011–20 loadings (k ≤ 5)
    - **Stock layer (501 names: current S&P 500, pools, books), test R² median:** S&P 500 latent 32% / anchored 25%; demo books up to 94%; placebo bucket 20% / −4%.
    - **Survivorship caveat:** the S&P 500 layer uses current members.
    - **Code:** `src/worsey/planb/factors.py` (tests `tests/test_planb_factors.py`); model saved to `cache/pmrm/b1/model.pkl`.
    - **Finding:** the 2003 check at k = 9 with 19-asset halves is a small-sample artifact (7.8% vs 32%); at k ≤ 5 the two fits are equal.

157. **Plan B B2: event library v0 (Oct 20; `docs/pmrm/results/b2_library.md`).**
    - **Scheduled:** 1,525 releases since 2000 (1,306 from late 2003 with factor responses). FOMC from federalreserve.gov historical pages; CPI, jobs and PCE first-publication dates and GDP vintages from ALFRED (CPI vintages include February seasonal-factor revisions, so first-publication dates are used). Matches P0b on 2022–26 and fixes P0b's shutdown dates.
    - **Surprises:** release-day ΔDGS2, ΔDGS10, Δ5y breakeven; naive data surprises; GDP advance − GDPNow (41).
    - **Unscheduled:** 194 events, 2004–2026, 10 categories × about 20. Claude with web search: 10 calls, about $4.73 estimated, against a $3.90 dry-run estimate. Within the SPEC §6.1 caps (300 calls, $5); tagged `source=pmrm`. Responses are cached, so a rerun costs $0.
    - **The model ran out of searches (6 per category):** 82 events cited (75 with working links); 112 from model knowledge, kept and flagged.
    - **Code checks every date against Wikipedia's current-events pages:** 173 of 194 match within 2 days.
    - **Bugs fixed during the run:**
      - JSON extraction tripped on "[]" in prose
      - the Wikipedia section slice was empty
      - events with no stated time were measured on the report day even when they broke after the close (e.g. the Hamas attack); the short window now spans 2 sessions when the time is unknown (179 of 194)
    - **Responses:** factor (latent and anchored), a 30-ETF panel, and targeted sectors/tickers beyond-factor, over a short window and 5 sessions; 55 of 194 windows contain a macro release (flagged).
    - **Known weaknesses:**
      - selection leans to big events (118 major, 27 extreme, 4 minor)
      - "first report" for forecast events (hurricanes) gives lower-bound responses
      - no prediction-market anticipation yet (B5)
    - **New infrastructure:** `src/worsey/pmrm/ic/budget.py` (SPEC §6.1 per-run caps, dry run, permanent cache; tests `tests/test_pmrm_budget.py`); `source` tag in the LLM ledger.
    - **Stop for the trainer:** review the sample before B3 (BUILD_PLAN B2 step 4).

159. **Plan B B3: impact model, held-out tests (Oct 20; `docs/pmrm/results/b3_impact.md`).** Findings, mostly negative.
    - **(a) Scheduled releases, test 2021–26 (310):**
      - **Market-based surprises:** predict non-rates factor moves +3.6% vs the type mean (chance 95th −8.9%); latent +20%; rates +99.9% (mechanical).
      - **Portfolios:** balanced book +1.1% and energy +5.2% (both p = 0.05); the other 8 negative.
      - **By type:** CPI, FOMC and PCE negative, consistent with the post-2021 flip in the stock–bond sign.
      - **Rolling 8-year refit:** doesn't fix it (+1.8%).
      - **Naive data surprises:** no skill; their correlation with the rates reaction is 0.1–0.2. B5 should use Kalshi-implied means.
    - **(b) Unscheduled events, leave-one-out (194):**
      - **Analog prediction:** direction 46–52% (coin flip); R² vs zero −15% to −16%, not better than random analogs (p 0.08–0.16); every portfolio negative.
      - **Size:** analogs don't predict size (exploratory); severity buckets add about 1%.
      - **Band:** the analogs' spread correlates +0.25 with the error (the uncertainty band is informative).
    - **Diagnosis:** the schema has no direction field (escalations and relief events share a category, so analog averages cancel), plus 194 events is small and the windows are noisy.
    - **Options for the trainer:** add direction and run B4's comparison step (capped); grow the library systematically (GDELT); B5 surprise measures for releases; present damage as an analog range for now.

161. **Library growth from Wikipedia's Current events portal.**
    - **Candidates (code, free, cached):** `src/worsey/planb/wikinews.py` parses every monthly portal page 2004–2026, keeping relevant news items with their topic and news-source links.
    - **Selection (one capped Claude call per month, `scripts/planb_grow.py`):** events picked by news prominence and category coverage, never by market reaction. The model returns only compact labels; title, description, date and sources come from the Wikipedia item.
    - **Cost:** dry run 274 calls, about $25.48; actual about $0.058/month (about $16 total), in successive runs each under the $5 per-run cap.
    - **Merge:** near-duplicates of curated events dropped (same category, ±2 days); every event tagged with its `origin`.
    - **Scale fixes in B3:** precomputed similarity matrices; exact LOO ridge via the hat matrix.
    - **Finding:** the taxonomy lacks a corporate category (mergers, corporate actions); next schema version.
    - **Results:** to follow in `results/b2_library.md` and `b3_impact.md` when the runs finish.

163. **Named factors extended from 9 to 13.** Added asia_em (EEM − SPY), healthcare (XLV), agriculture (DBA) and defensive_vs_producers (XLU − XLE), the four patterns the latent model found that the named set missed.
    - **Common scored set (91 ETFs):** named 13 test R² **62.7%** vs named 9 58.2%, latent 58.2%, and the old hybrid (named 9 + 3 latent) 61.7%. Named 13 + 3 latent: 64.4%.
    - **Factor value:** asia_em is the most valuable addition (2.8 pts); credit is nearly redundant (0.1 pt).
    - **Choice:** the named-13 model becomes Plan B's default factor space. It is interpretable and beats the hybrid. `scripts/planb_b1_extended.py`, `cache/pmrm/b1/model_ext.pkl`, `results/b3_experiments.md` §1.

165. **Regime model: labels, regime-aware outputs and recency weighting.** No LLM calls.
    - **Diagnosis (data):** on release days the S&P–2y yield correlation went from +0.2–0.3 (2003–2020) to −0.14 (2021–26): CPI +0.20 → −0.42, FOMC +0.23 → −0.37, jobs unchanged +0.20. The real-yield part turned negative (−0.29) while the breakeven part stayed positive, and rate surprises doubled in size. Yearly stock–bond correlation: +0.35 to +0.6 in 2009–2020, −0.4 to +0.1 since 2021.
    - **Labelling:** every library event gets its regime state, known the day before (`src/worsey/planb/regime.py`): stock–bond correlation, 5y breakeven, fed funds 6-month change, VIX, Baa − 10y spread (FRED keeps only about 3 years of the HY OAS), 10y − 2y slope. Plus labels (together/apart, inflation high/normal, hiking/cutting/hold, VIX stressed, credit stressed, curve inverted). `cache/pmrm/b2/regimes.json`.
    - **Releases (test 2021–26):**
      - continuous regime interactions extrapolate badly (−37% to −133%)
      - stock–bond interaction alone fixes CPI/FOMC signs but hurts others
      - **split together/apart, shrunk:** CPI −0.19 → +0.03, FOMC −0.38 → −0.10, overall +1.5%
      - **regime-similarity weighting:** best overall +2.2% (beats chance), but its bandwidth was chosen on a same-regime validation period, so CPI and FOMC stay wrong
      - recency weighting: +1.4% → +1.9% (half-life 8 y)
    - **Unscheduled (time-ordered 2016–23):** direction + category +2.4% on extreme/major; regime features and recency add nothing.
    - **Lesson:** only 153 of 984 training releases are in the "apart" regime (vs 186 of 322 test). Regime models, and even their tuning, need history spanning many regimes.
    - **Proposed:** a 1976–2026 release library from Fama-French daily data (49 industries and factors, 1926+, free for research), FRED yields (2y 1976, 10y 1962), ALFRED release dates and FOMC from 1994. Validate settings on a different regime from training.

170. **B4 three-way result: the LLM with the model's context beats both the raw LLM and the model (Oct 21; `results/b4_threeway.md`).** 60 calls, about $1.95.
    - **Post-cutoff (2026, 92 events; the LLM can't remember outcomes):** assisted LLM **+2.5%** R² vs zero (beats chance), raw LLM **−5.7%** (worse than no move), model −0.2%.
      - paired: assisted − raw **+8.2 pts** (90% +4.3 to +12.8)
      - assisted − model +2.7 (−0.6 to +5.8)
      - raw − model −5.5 (−11.2 to −0.5)
    - **Historical 2016–23 (200, hindsight risk):** raw +5.3%, assisted +8.8% (difference +3.5, 90% +0.7 to +6.4).
    - **By factor:** the raw LLM over-calls credit (−384%), which the context mostly fixes; both LLMs read oil (+11–12%); the assisted LLM leads on defensives vs producers and Asia/EM; the model leads on the market.
    - **Implications:**
      - raw LLM priors (what v1 relies on) are unreliable on genuinely new events
      - the product should run the PLAN_B §5.3 loop (code evidence → LLM judgment → code numbers)
      - next improvements: a default-flat rule for credit, and more post-cutoff events (prospective logging)
    - **Library completed (same run):** 3,876 unscheduled events through October 2026 (growth total about $24); re-scored, conclusions unchanged (extreme events: direction model +4.9%; extreme/major per-factor 3–4%).
    - **Spend to date:** $42.76 of the $70 project cap.

171. **The ceiling, and a fair raw-LLM baseline.**
    - **Ceiling:** event windows are about as volatile as normal days on average (1.0–1.1×), so only ≈ 3% of an average event-window move is attributable to events. For major events ≈ 32%, extreme ≈ 80%, financial stress ≈ 41%, releases 14–23%; wars, sanctions and tariffs ≈ 0% on broad factors. +2.5% across all events is most of the achievable; big events are where the headroom is.
    - **Fair baseline (60 calls, about $2.6):**
      - post-cutoff, a well-prompted raw LLM (regime, recent markets, base rates, reasoning) improves −5.7% → −1.2%
      - the assisted LLM (+2.5%) still beats it by +3.7 pts (96%); at equal prompting the evidence adds +2.6 (93%)
      - on 31 major/extreme 2026 events: assisted +6.8%, well-prompted raw −2.3%, model 0%
    - **Hindsight confirmed:** historical well-prompted raw +13.4% (beats assisted), but post-cutoff −1.2%. Only post-cutoff scores count as skill evidence.
    - **Plan to improve big-event accuracy:** intraday timing, surprise scaling (B5), magnitude labels, pre-2004 extreme events from 1926+ data, big-event-focused tuning.
    - **Spend to date:** about $45.4 of $70.

175. **Size awareness for big events.**
    - **Model (forward test 2016–26):**
      - severity-specific scaling of the base model helps extreme events (+6.4% → **+10.8%**)
      - adding many severity × direction terms overfits (worse everywhere)
      - severity-scaled ranges are calibrated (80% ranges cover 83–87%; spread ≈ 3.8σ extreme vs 2.0σ major vs 1.5σ overall)
    - **Size-aware assisted LLM** (severity-conditional base rates, severity-scaled model estimate; 10 calls, $0.59): 2026 overall +0.9% vs plain assisted +2.5% (n.s.).
    - **Key finding:** the 2026 post-cutoff set has almost no market-moving events. "Major" events moved the market 0.58σ on average, less than moderate/minor (0.77σ); only 4 "extreme". So big-event skill can't yet be tested after the cutoff, and severity labels assigned before outcomes are noisy for "major".
    - **Proposed:**
      - a blinded historical test of big events: descriptions anonymized (names, places, dates generalized) so the LLM can't recall outcomes; about 800 major/extreme events, about $3–4
      - prospective logging accumulates real post-cutoff big events over time
      - timing (intraday) and magnitude labels remain the main levers

178. **The naive baseline vs the factor model, on real books.** No LLM calls.
    - **Naive baseline:** book market beta × the average market move of earlier events with the same severity and risk-sentiment label (a stress test). Scored on 662 blinded big events × 11 ETF books, split before scoring into market-like (SPY, 60/40, tech, banks) and unusual (energy, agriculture/weather, gold miners, Latin America, utilities + REITs, long Treasuries, commodities).
    - **The factor map is essential for unusual books:** with perfect foresight, the market alone explains 85% of market-like books' event moves but only 37% of unusual books'; the 8 factors explain 77% (+40 pts, 90% +37 to +44). Gold miners 3% → 67%, TLT 3% → 85%; in 2026, unusual books 0% → 65%.
    - **As a forecaster, the code model alone ≈ naive** (unusual +2.7% vs +1.6%, range includes 0): label-based forecasts of non-market factor moves are weak.
    - **The assisted LLM uses the map:** it beats naive in every group (historical unusual +13.3% vs +1.6%; extreme unusual +27% vs +3%). In 2026 on unusual books: assisted +4.1%, naive −3.0%, raw LLM −24.6% (vs naive, 90% −0.4 to +14.2: borderline).
    - **Coverage gap:** the 13 factors explain 60–85% of unusual books' daily moves (agriculture/weather 60%), which is the trainer's El Niño point. Next: a low-coverage flag, niche factors (natural gas, softs, fertilizer, reinsurance), and a slow-event track for forecast-driven events.
    - **Decision:** keep the factor model as the map. The product's forecaster is the assisted loop over it; the naive stress test is the floor every predictor must beat.

179. **Plan B replaces the v1 engine in Worsey's chat.**
    - **What the chat now does** (`src/worsey/planb/live.py`; config `SCENARIO_ENGINE = "planb"`, env `WORSEY_SCENARIO_ENGINE`):
      - **"What could hit me?" (scan_risks):** code replays every kind of big past event (10 categories, major/extreme) and each scheduled release (2021–26, today's stock–bond regime) through the book's current factor exposures. Each row gives the typical hit, the bad case (worst 1 in 10) and how often such events happen. Releases show the next date Kalshi lists (and, for the Fed, the most likely outcome). One-off events carry no odds in the scan. New `risks` card.
      - **One event (analyze_event):** the assisted loop (#170, #176, #178).
        - event_analyst structures the event (category, severity, direction channels, region, targets) and calls `event_evidence`: the model's estimate per factor as a bucket, this severity's base rates, the 5 closest past events and what each would do to this book today, the regime, factor coverage, and candidate Kalshi markets.
        - It then picks a bucket per factor and calls `price_scenario`; code turns the buckets into numbers.
        - Output: first-hours and end-of-day ranges (#177), the bad case (1 in 20), and the result by position and by factor. Odds appear only if the expert links a market that really is the event. New `scenario` card.
      - **Hedges (compare_hedges):** the same comparator, priced against the scenario's middle case and its bad case (the card's "middle / bad case" toggle).
    - **Runtime data:** `src/worsey/planb/pack.json` (2.2 MB, built by `scripts/planb_pack.py` from the research caches), so the app never reads `cache/pmrm/`. It holds 13 factors with live ETF proxies, the direction + category model with severity scales, bucket values and base rates, 3,876 events, 322 releases, reversion stats and the latest regime.
      - Live proxies for rates are Treasury ETFs (SHY, IEF), which correlate 0.85 / 0.96 with the research's FRED yield changes; the other proxies are the research instruments themselves.
      - Loadings are OLS on each holding's last 504 sessions.
    - **Ranges are centered on the judged middle:** the spread of past events of that severity plus the book's own daily noise, recentered so the card, the reply and the hedge table share one middle case. A first live run showed a mismatch (reply −$387 vs hedge middle −$532); fixed and tested.
    - **Kept:** the v1 engine stays in the code. Recorded demos and replays pin `v1` (server `--replay`, `chat --replay`, `eval/test_demo_replay.py`), and the v1 prompt and tool wording are unchanged, so the three demos still replay byte-for-byte. Agent tests default to v1 (`tests/conftest.py`); `tests/test_planb_live.py` (7 tests) covers Plan B. 434 tests pass.
    - **Prompts:** new `prompts/event_analyst_planb.md` (draft for the trainer's review); the router prompt is unchanged, and the entry-tool descriptions have Plan B wording under planb.
    - **Not carried over (yet):** the scan no longer runs equity_analyst / natural_world_analyst notes; the joint-tail copula and theme collapse don't apply to category rows; no prediction-contract hedge row unless a market is linked.
    - **Live test** (3 questions on the balanced book, $0.20):
      - scan → financial stress (bad case 1 in 10 ≈ −$3.4k on $100k)
      - "what if Iran attacks Gulf oil infrastructure?" → end-of-day middle with a likely range; stocks drag, XLE and GLD offset
      - hedge → no option pays for itself on the middle case; the SPY put spread trims the bad case
      - grounding passed on all three
    - **Spend to date:** $56.01 of $70.
    - **Alternatives considered:** a per-market v1-style scan with Plan B damage (needs one LLM call per market to structure it; too costly per turn); re-recording the demos now (costs budget; deferred, see the project notes).

180. **The shipped chat vs an untrained LLM on 2026 events: no edge yet.** $1.46 (155 calls, capped at $3 per run).
    - **Test:** all 31 major/extreme 2026 events, three predictors:
      - **the exact chat path:** Plan B prompt, `event_evidence`, buckets, `price_scenario`; the engine in a new cutoff mode (`live.set_cutoff`: model refitted on pre-2026 events, evidence only from earlier events)
      - **an untrained LLM:** same model, a fair well-framed prompt
      - **the engine alone**
    - **8 factors:** product −7.3%, untrained −7.4% (difference 90% −11 to +13, a coin flip), engine alone **+1.8%**. Market direction 55% / 48% / 48%.
    - **Real books:** the engine alone is best on 4 of 6 (balanced +7.1%, energy +5.7%, Treasuries +12.4%, agriculture +3.0%). The product is best on gold miners (+3.9%) and worst on Treasuries (−52%).
    - **Why:** both LLMs call stock and credit moves whose direction doesn't hold (stocks: product −115%, untrained −80%). The product's judgment helps on oil (+32% vs +18% vs +7%). The engine never leaves flat.
    - **Reconciling with #170 / #176:** the research harness found the assisted LLM ahead after the cutoff; the shipped port (new prompt, full reply, fair raw baseline) loses that edge on this test. The product's measured value today is the engine (book-specific exposures, calibrated ranges, "little effect" by default), not the LLM's factor calls.
    - **Not changed:** no tuning on these 31 events.
    - **Options for the trainer:**
      1. engine estimate for market, rates and credit, with the expert judging only oil, gold and utilities vs producers (supported by #170's by-factor result and this test)
      2. shrink the judgment halfway to the engine
      3. extend the test to all 92 2026 events first
      Whatever is chosen gets validated on fresh events via the prospective log.
    - **Spend to date:** $57.47 of $70.

182. **Numbers only from code in the "what if X?" path.** Follows #180 (the expert's bucket overrides tied an untrained LLM, and the engine alone was best) and the project notes rule 6.
    - **Removed:** the bucket override. `price_scenario` takes only an optional `market_id`, and `live.price()` uses the engine's estimate for every factor. A test checks that the tool schema has no way to pass a number.
    - **The LLM keeps:** identifying the event, filling the schema (category, severity, region, direction, targets), matching a Kalshi market, and explaining the result. `prompts/event_analyst_planb.md` was rewritten to match.
    - **Buckets survive only as words describing code's estimate** (`code_estimate`: e.g. "oil: a moderate rise").
    - **Added to the scenario output (all code):**
      - the worst and best of the 5 nearest past events on this book
      - tail probabilities P(loss > X) at round thresholds, from the same draws as the ranges
      - `reply_facts` (middle, 9-in-10 range, worst nearest past event, one tail probability), which the reply template uses
      - the card shows the worst nearest event and a tail line
    - **One shared distribution function** (`live.scenario_draws`), used by the chat and by the calibration test.
    - **Check:** `scripts/planb_product_vs_raw.py score` with the shipped path on the 31 events of #180 reproduces the engine alone exactly (factors +1.8%; books +7.1% balanced, +12.4% Treasuries, +5.7% energy). Against the untrained LLM: +8.8 pts (90% −3 to +25; 88% bootstrap support, just short of the 90% bar). The #180 bucket version is kept in the table for reference.

183. **The shipped ranges are calibrated but not sharper than normal noise (Oct 22; `results/range_calibration.md`).** No LLM calls.
    - **Test:** every 2026 event (92) and 2016–2025 with rolling yearly refits (1,681), × 11 books and per factor, with the shipped distribution function and stored schema labels.
    - **Books, end of day:**
      - 2026: 50% inside the middle half, 90% inside the 9-in-10, 3.6% below the bad case, 6.6% above the top; bias +0.08 sd
      - 2016–2025: 51%, 91%, 3.7%, 5.2%; bias +0.05 sd
    - **CRPS vs a predict-zero range with the book's normal noise:** 2026 −2.2%, 2016–2025 +0.2% (about even); extreme events 2016–2025 +11%.
    - **Misses:**
      - extreme events too narrow (82% inside the 9-in-10, 2016–2025)
      - value vs growth too narrow in both periods
      - oil, gold and EM too narrow in 2026 only (fine historically)
      - credit too wide in 2026
      - first-hours ranges slightly wide (middle half ~63%)
    - **Proposed, not applied:** widen extreme ×1.25, narrow first hours ×0.75, widen value ×1.3. Each needs a held-out test the current data can't give (2026 has 4 extreme and 11 timed events), so the prospective log is the validation set.
    - **Reading:** the chat's ranges are honest and book-specific; for most events they're about as informative as "expect normal noise", which is the right answer for most events (#171).

185. **The core test: system vs just asking the LLM, on 12 portfolios × 92 events of 2026.** $2.48 (92 calls).
    - **Design (predeclared in the script):**
      - **naive arm:** the chat's model, no tools, given the date, the news and the holdings; returns an expected dollar change, a 90% range and a hedge
      - **system arm:** the shipped engine in cutoff mode
      - **scoring:** weighted interval score on 5/50/95%, coverage, middle error; bar = system better with ≥ 90% support
    - **Result: passes.** Interval score system $752 vs naive $781 (better by $29, 90% +$7 to +$51, 98% support). The 90% range holds 90% (system) vs 77% (naive).
      - unusual books: $45 better (99% support), with the naive LLM's range holding only 74%
      - market-like books, the demo book and big events: not significant
    - **Neither predicts direction** (53–54%).
    - **Neither beats predict-zero plus the book's normal noise** ($721). The system's ranges are calibrated but slightly too wide on plain books; the naive LLM's are too narrow.
    - **Hedges: not valid.** The naive LLM declined to hedge 71% of the time; 126 of its named tickers had no prices; the system's minimum-variance "hedge" could sell the book itself. A fair hedge test is proposed.
    - **Reading:** the modules beat just giving it to the LLM on the risk numbers, through honest, portfolio-specific ranges (biggest gap on gold miners, LatAm, commodities, energy), not through predicting direction.
    - **Proposed:**
      1. book-noise floor plus event premium for range width (tighten plain books)
      2. a fair hedge-decision test
      3. show the "normal range vs event range" framing in the product
    - **Spend to date:** $59.98 of $70.

186. **The core test, position by position.** $1.37 (92 calls).
    - **Design (predeclared):** as #185, but the naive LLM and the engine each predict all 19 holdings (expected % move, 90% range). Per portfolio, the dollar interval score is summed over its positions, so errors can't cancel.
    - **Result: passes, more clearly.** Summed position score system $803 vs naive $839 (better by $37, 90% +$17 to +$57, 100% support). 90% ranges hold 90% vs 73%.
      - unusual portfolios: $53 better (100% support)
      - demo book: $20 better (94% support)
      - market-like portfolios: 75% support (not significant)
      - big events: not significant
    - **By holding:**
      - the system's range is better on 13 of 19 (Chile, Mexico, Brazil, commodities, gold, gold miners, Nasdaq, energy, oil producers, agribusiness…)
      - the naive LLM's ranges miss badly (gold 49% inside, commodities 53%, gold miners 58%)
      - the naive LLM is better on 6 steady holdings (SPY, Treasuries, high-yield, real estate, banks), where the system is too wide (99–100% inside)
      - the system is too narrow on gold (66%) and gold miners (72%)
    - **Middles: no difference** (sum of absolute errors $1,470 vs $1,465 vs predict-zero $1,453); direction about a coin flip.
    - **Neither beats zero plus each holding's normal noise** ($769).
    - **Proposed:** per-holding range = own normal noise widened by the event premium (tightens steady holdings); widen gold and gold miners. Validate on the prospective log.
    - **Spend to date:** $61.36 of $70.

187. **Plan C C0: the test is frozen, and the bar is plain normal noise.** No LLM calls.
    - **Built:**
      - `src/worsey/planc/harness.py`: 19-quantile predictions, standardized quantile CRPS, coverage, paired bootstrap over events (3 tests)
      - `scripts/planc_c0.py`: a dataset of 3,571 events (train < 2023: 2,995; validation 2023–25: 484; test 2026: 92), each with 13 factors, 96 ETF moves beyond the factors, and 12 portfolios
    - **Baselines on test (CRPS standardized by each target's normal noise, lower is better):**
      | | factors | ETF residuals | portfolios |
      |---|---|---|---|
      | rung 0, zero + normal noise | **0.607** | **0.713** | **0.577** |
      | rung 0b, noise + severity premium | 0.616 | 0.716 | 0.589 |
      | rung 1, today's engine | 0.630 | 0.720 | 0.587 |

      Rung 0 also wins on validation and on big events. The engine loses to it in 95–100% of resamples on most comparisons. The premium widens ranges that don't need widening (event windows ≈ normal days on average, #171).
    - **Decision:** the reference rung (the bar Plan C must beat) is rung 0, zero + each target's normal noise, not today's engine. Pass = beat it on 2026 test CRPS with ≥ 90% support and hold on the prospective log.
    - **Implication for the product (consistent with #185–186):** today's engine is slightly worse than plain noise as a forecast. Its value is the honest portfolio map and ranges, not event-specific numbers.

189. **Direction by size: the LLM calls real moves; the engine doesn't (Oct 22; `results/direction_by_size.md`).** No LLM calls.
    - **Diagnostic (#188):** direction hit rate by the real move's size in noise units, on existing forecasts.
    - **Naive LLM (2026):**
      - positions: 46% (< 0.5×), 52%, **57% (1–2×), 62% (2–3×), 75% (> 3×, n = 12)**
      - portfolios: **62% (1–2×), 73% (2–3×)**
    - **Engine:** 44–57% at every size, including 2–3× and above 3×.
    - **Reading:**
      - the LLM has real directional knowledge when a move is real, drowned out by the many small or noise moves
      - the engine's label regression has none, even on big moves
      - "nobody predicts direction" (#185, #186) was too broad
    - **Caveat:** conditioned on the realized size, which isn't known in advance; the gate (PLAN_C §2b) has to predict which events are real.
    - **Plan C consequence:** add an LLM-derived per-factor direction as an input feature (group 4b; code still makes every number), used through the gate. This reconciles #180: un-gated LLM overrides hurt because most calls were on noise.

194. **Plan C C1c: the gate plus the LLM's sign is the best forecast so far, 98% support on 2026; positioning and the vol premium hurt.** No LLM calls, $0.
    - **Built:**
      - `scripts/planc_c1c.py`: CFTC positioning (12 futures markets, 1986–2026, released-date as-of), implied-vs-realized vol, and B2's LLM factor signs inside the gate
      - real part N(d·sign·s0, (k·s0)²), every event N(e·sign·s0, s0²)
      - fitted with an exact gradient (hours → minutes)
    - **Finding:** B2's signs (7 factors, all 3,876 events) already show #189's pattern: 51% on small moves, 58% at 1–2×, 60% at 2–3×, 70% above 3×.
    - **Result (standardized CRPS, 13 factors):**
      - **gate + sign: 2026 0.6019 vs noise 0.6066 (−0.8%, 98% support)**
      - big events 0.5990 (−1.3%, 99%)
      - the 7 signed factors 0.5798 (−2.1%, 100%)
      - 2023–25 0.5419 (100%)
      - the gate alone: 2026 0.6032 (95%)
      - adding positioning and premium makes it worse (0–17% support vs the gate)
    - **Direction on moves > 2× noise:** 63% on 2023–25 big events, 68–75% on 2026 (small n). That's the LLM's own hit rate; the gate decides when to lean on it.
    - **Learned push (noise units):** 10-year 0.90, US stocks 0.79, 2-year 0.76, credit 0.49, oil 0.36; dollar and gold about 0.
    - **Decision:**
      - gate + sign is the leading Plan C rung
      - positioning and premium are dropped from the gate
      - nothing ships until it holds on the prospective log
      - rung D was added after the first run (B overfit); it is the natural variant but was chosen after seeing results
    - **Not built:** prediction-market jumps (7 events have markets), CBOE put/call (stops 2019), GDELT build-up (C2).

199. **The fair hedge test: protection rarely pays over an event window; "don't hedge" is the right default, with extreme events the exception to track (`results/planc_hedge.md`).** $0.81 (naive arm).
    - **Data:** 4,727 book-events (576 events of 2023–26 × 12 books), real option closes.
    - **Always hedging:** net −$131 (2023–25) / −$150 (2026) per $100k book per event on $948 / $1,243 of premium.
      - the hedge pays 29–30% of the time
      - it cuts the worst-5% outcome from −$5,068 to −$3,522 (2026)
      - about $59 is spread
      - break-even only if losses beyond the normal bad case weigh 3–6× everyday P&L
    - **Rules tuned on 2023–25 (λ = 1):** system and "cheap options" hedge 0–1%. On 2026 they're the same as never hedging. The naive LLM hedged 0.3% (10 of 15 paid; +$2, 82%, too few to call).
    - **Where it comes close:**
      - extreme events: −$32, paid 48%, book loss beyond the normal bad case 9% vs 2%
      - the cheapest options: −$60
      - SPY, TLT, EEM, XLU puts: −$54 to −$80, vs −$208 to −$313 for GDX, XLE, EWZ, XLF
    - **Exploratory (chosen after seeing results):** hedging extreme events only is about even on 2023–25 (8 events) and +$18 per book-event on 2026 (4 events, 84%). Logged as a rule to track on the prospective log.
    - **Decisions:**
      - the hedge card defaults to "don't hedge", with the cost and how often such protection has paid
      - hedge for extreme events, or when the user says a bad-case loss matters much more to them (a user-set weight, not assumed)
      - prefer the cheapest instrument that covers most of the risk

200. **Product: "don't hedge by default", the after-the-open range, and an $80 project cap.**
    - **Project cap:** `WORSEY_LLM_MAX_USD_TOTAL` 70 → 80 (`agent/budget.py`).
    - **Hedge verdict (from #199), code only:**
      - `live.hedge_verdict` → `should_you_hedge` on every Plan B hedge table
      - default "skip by default", with the book's normal 2-day bad case, the cheapest hedge's cost, how often such puts paid for themselves (by severity), the chance of a loss beyond the normal bad case, and how much a hedge covered when it happened
      - "worth considering" for extreme events, or when the user's bad-case weight ≥ the measured break-even (3)
      - the router can pass `bad_case_weight` (1–10) from what the user said; Plan B schema only, so the v1 demos replay byte-identical
      - both hedge experts are told to lead with it
      - the hedge card shows it first
      - the evidence is read from `results/planc_hedge.json` (`product_evidence`), scaled to the book
    - **After the open (from #193), code only:**
      - `price_scenario(already_reported_before_open=true)`, for an event that really happened before today's US open
      - adds `after_the_open`: the book's opening move (today's 09:30 opens vs the last close, `massive.opening_price_today`, live and cached per minute) × 1.05, with 90% of a 2-day range still ahead (the ordinary-day rule's median over the 12 test books)
      - before the open or in replays it says the estimate stands
      - the flag is a boolean, so the LLM still can't pass a number (#182)
    - **Tests:** 439 pass (2 new); the demo replays are unchanged.
    - **Fix after the live test ($0.37, 2 questions):** the verdict's "normal 2-day bad case" used the book's noise beyond the factors ($190 a day on the demo book) instead of its whole daily swing ($828). Now `coverage.book_daily_sd_usd`, used by the verdict and the after-the-open range. Demo book: −$441 → −$1,927. In the live test both hedge experts led with "skip it" and code's numbers.

207. **"What could hit me" is now split by position.** $0.15 (live test).
    - **Found by the side-by-side on the trainer's example book** (PFE, WDC, TSLA, URA, GLD, oil puts vs the naive LLM): the scan gave only book totals, and the expert said it couldn't split by holding.
    - **Built (`live.scan`):** every risk (event categories and releases) gets `by_position`. Each holding gets:
      - its typical move (the same replay of past events through its own factor exposures)
      - its own 1-in-10 bad and good case (factor moves plus its own noise, residual sd × √1.5, as #186 scored)
      - its part of the book's bad case (the average of the book's worst 1 in 10 past events, split by holding; the parts add up exactly)
      - the worst past example also gets a per-position split
    - **Card:** a "by position" table under each risk. **Prompt:** say which holdings carry each risk.
    - **Example:** on the trainer's book the financial-stress bad case is WDC 35%, TSLA 34%, URA 21%, PFE 19%, GLD 2%, offset 11% by the short oil.
    - **Also found (next):**
      - unprecedented events are averaged down to history's typical analog (Hormuz closure → −$79): a model-assisted black-swan mode is next (#208)
      - option positions can't be held or repriced: full repricing with a measured spot–vol response is planned (#209)
    - **Tests:** 442 pass.

208. **Black-swan mode: the LLM writes an explicit factor shock, code maps it through every holding; anchoring on other kinds of past events hurts, so the LLM reasons from its own read.** $0.42 (gut check $0.27, live test $0.15).
    - **Why:** the historical estimate averages unprecedented events toward smaller cousins (Hormuz closure on the trainer's book: −$79 to −$235).
    - **Built:** `live.stress_scenario` → the `stress_scenario` tool (event_analyst, Plan B):
      - the analyst's plain-unit shock (% moves, bp for yields) with a why per factor
      - factors not set are filled with their conditional co-move over past big event windows (labelled)
      - flags beyond the biggest move in 3,876 past event windows
      - per-position and book P&L at 0.5×, 1× and 1.5×, with each holding's own noise
      - feeds the hedge table, with the verdict "worth considering" (extreme)
      - also: plain-unit analog moves in `event_evidence`; a `stress` card
    - **Rule change (#182):** this is the one mode where the LLM writes numbers, and only the shock, shown as an editable assumption. Everything after it is code's.
    - **Finding:** code can't detect "no close precedent" from schema similarity (median best match 0.975, since labels are coarse). The mode is triggered by the expert's judgment or the user's words.
    - **Gut check (46 anonymized extreme events, each with its own category hidden):**
      - raw LLM: skill vs no move 39% (2023–26: 33%), direction on big moves 85% (74%), size slope 1.14
      - anchored on 5 other-kind analogs: 32% (3%), 79% (57%), slope 1.19 (0.61)
    - **Decision:** the analyst writes the shock from its own understanding, with no anchoring on dissimilar past events; analogs are context only. Tool description and prompt updated.
    - **Live example:** both modes were used (historical −$235; stress, oil +40% and stocks −5%: −$8.3k).
    - **Tests:** 443 pass.

209. **Options in the book, fully repriced in every scenario with a measured spot–vol response.**
    - **Design:**
      - the event engine supplies each underlying's moves
      - code fully reprices each option (Black-Scholes at the shocked spot, the shocked IV and 2 sessions less time)
      - the expert explains
      - repricing is code, not the options expert's judgment (#180's lesson)
      - full repricing captures delta, gamma, vega, vanna, volga and theta at once; a Taylor expansion of today's Greeks fails on event-size moves
    - **Built:**
      - `options/scenario.py`: `OptionPosition`, vectorized BS
      - `spot_vol_response`: ΔIV per unit spot return, down and up separately, from cached IV history (12 underlyings), else the CBOE vol index on FRED (VIX, VXN, OVX, GVZ), else SPY's
      - `iv_now`: this contract's quote, else the nearest expiry's at-the-money IV, else realized vol
      - `scenario_pnl` and `greek_split`: an exact split by full repricing one input at a time (spot, then vol at the new spot, then time), plus the Taylor Greeks with their remainder for reference
    - **Book:** portfolios and chat holdings accept `{"ticker": underlying, "option": {kind, strike, expiry, contracts}}`; options sit beside the linear book (no factor betas applied to a premium).
    - **Engines:**
      - `price()`: each option repriced on the same factor draws as the book (plus its underlying's own noise); the book's range, bad case and tails include them
      - `stress_scenario()`: repriced at 0.5×, 1× and 1.5× the shock
      - `scan()`: every past event replayed through each option's repricing, so per-position and book figures include them
      - cards and prompt updated
    - **The trainer's book (2 USO 157 puts, Jan 2027):**
      - IV 34.8% (the contract's quote); oil-VIX response down +0.24, up +0.56
      - stress oil +40%, stocks −5%: puts −$1,434 of $1,985 premium (1.5×: −$1,520), book −$5.2k (was −$8.3k with the delta stand-in)
      - exact split at +40%: spot −$1,895, vol +$462, time −$21; the Taylor Greeks sum to +$1,769, the wrong sign
      - live test ($0.15): the expert ran both modes; puts −$1.4k
    - **Tests:** 445 pass (2 new); the demo replays are unchanged.

211. **Targeted sectors and countries: given a macro event, we don't beat noise on the ETFs it targets, on size or direction.** $0.
    - **Data:** 290,042 event × ETF rows (96 ETFs' moves beyond the factors), with targeting at the first report: LLM-named (1,940), sector match (24,418), country match (1,849), and the ETF's history on earlier same-category events.
    - **LLM-named ETFs on 2023–26:** move 1.1–1.3× other ETFs; direction 52% (big moves 60%, n = 48). Before 2023 much better, but hindsight is possible.
    - **Gate model (fitted before 2023):** loses to noise on 2023–25 and 2026 for all and for targeted rows (support 0–22%); gate AUC about 0.5. The crisis years taught it 9× tails that are too wide later.
    - **Simplest rule tuned on 2023–25:** no widening and a 0.1 tilt; slightly worse on 2026 (4%).
    - **Reading:** macro targeting cuts both ways; two-session windows dilute; most events are small. Targeted ETFs have fatter tails than noise (81–89% coverage), not bigger typical moves.
    - **Contrast:** single-stock company news does carry signal (#202, #205).
    - **Next to test:** 2-hour ETF windows for timed events; the product keeps "targeted sector: normal range, fatter tails".

213. **"What if" scenarios now assume a real market mover, with the analyst's directions and the real-event model's sizes; company headlines can be added as legs.**
    - **Found (live, a USO + NVDA book, "another Iran attack tomorrow but a positive AI headline for NVDA"):**
      - the scenario was history's average of every moderate conflict event, most of which did nothing: oil +0.08%, book +$107
      - the NVDA headline couldn't be priced at all
    - **Built:**
      - `event_evidence` takes `factor_signs` ('+', '-', '0' for US stocks, 2y, 10y, dollar, oil, gold, credit): a direction judgment, no numbers
      - `live.real_event_draws`: for each signed factor, middle = push × sign × the factor's normal 2-session noise, spread = real-event width × that noise
      - the push and width come from Plan C's gate model, real part (rung D frozen for the shadow run, `planb/real_event.json` from `scripts/planc_real_event_pack.py`): oil push 1.31, US stocks 0.73, 2y 0.71, 10y 0.70, credit 0.58, dollar 0.22, gold 0.12
      - factors signed 0, or never signed: normal noise; co-movement from past events of this severity, rescaled
      - history's average stays beside it as context
      - `price_scenario(company_news=[{ticker, news_type, direction}])`: each leg's typical move for that news type (the timing-clean S0b study, #205) in the stock's own noise beyond the factors, in the user's direction, on the look-through exposure, with its own spread; shown per leg and per position (of which from the event, of which from the headline)
      - the card shows the legs, the mode and history's average
    - **Live rerun ($0.15):** oil +4.8%, US stocks −1.1%, 10y −4.7bp; USO +$2.4k; NVDA event −$710 and headline +$451 (+0.9%), net −$260; book +$2.2k (9 in 10: −$5.6k to +$9.1k); history's average +$173.
    - **Evidence:** each block is tested (the real-event rung beat noise on 2026 with 98–99% support and called 63–75% of big-move directions; the headline sizes come from 60,909 events); the combination is arithmetic on them, gut-checked by examples rather than backtested.
    - **Tests:** 447 pass.

214. **Scenario sizes: the analyst gives a size with each direction, mapped to the measured move for that size call; unsigned factors get their typical co-move; company headlines can be "big".** About $0.75 (gut checks).
    - **Why (gut checks after #213):**
      - every signed factor got the same average push: a surprise 50bp emergency cut moved 2y yields only −6bp and oil −4.8%
      - XLE barely fell with oil −5% (energy stocks' own factor was left at 0)
      - a "big" drug rejection got the typical FDA headline's move (PFE −0.9%)
    - **Built:**
      - `factor_sizes` (small, medium or large per signed factor) mapped to the measured move in the LLM's direction for that size call (#196's read vs outcomes, 745 big events): large 2.44× noise (2023–26: 2.05×, n = 24), medium 0.74× (0.25×), small 0.14× (0.09×), each with its spread
      - factors not signed are filled with their conditional co-move (correlations over past big event windows)
      - company legs take size "big" (the news type's top-10% move; `stock_s0b.json` now stores the 90th percentile)
    - **Gut checks after:**
      - Fed cut: 2y −21bp, gold +1.6%, stocks −1.1%, oil −0.5%
      - OPEC increase: oil −9%, USO −$2.7k, XLE −3.2%
      - FDA rejection ("big"): PFE −2.3%
      - Iran + NVDA headline: oil +2.7%, NVDA +1.5% net
    - **Known quirk:** the 2y and 10y factors overlap, so in a sharp curve steepening a long bond (TLT) can get the wrong sign (−0.3% in the Fed case). To fix with a duration-based mapping for bond ETFs.
    - **Tests:** 447 pass.

215. **Sector layer, step 1: one sector factor per name on top of the 13 macro factors explains a third more of a typical stock's daily moves, held out.** $0.
    - **Built:** 19 sector factors (each sector ETF's move beyond the macro factors) and a held-out R² test (fit 2019–20, test 2021–Oct 2026).
    - **Result:**
      - median S&P 500 stock R² 24.2% (macro) → 33.3% with each name's best one sector (picked on the fit years, used only if it adds more than 2 points); 65% better, 9% worse; every GICS sector ≥ 0
      - all 19 sector factors for every name: 31.0%, with more names worse (overfit)
      - ETFs: median unchanged (countries and commodities get no sector), sector ETFs gain; gold miners worse
    - **Decision:** add sectors to the live model as a second stage. The macro loadings stay exactly as tested; each holding gets its best sector and a loading on it, fitted on what the macro factors leave. Used for sector headline legs and sector risk rows (steps 2–3).

216. **Sector layer, step 2: earnings spillover to sector peers is real but partial; the reporter's move carries it, not its EPS surprise (`results/planc_spillover.md`).** $0.
    - **Data:** 18,446 reports (2011–26); peers = same best sector (#215), excluding the reporter; each peer's move beyond the macro factors, averaged and scaled to the group's ordinary day.
    - **Result:**
      - peers move 1.21× an ordinary day (9% > 2× vs 4.6%)
      - direction from the EPS surprise 51%; from the reporter's own move 58% (63% after big moves), steady through 2026
      - peers move about 0.07 of the reporter's move: regional banks 0.25 (68%), transports 0.16, energy 0.13, semis 0.12 (63%), materials 0.10; software, biotech, insurers about 0
    - **Sector-tagged Benzinga news** is too sparse to build on; user scenarios supply the direction, and size comes from the sector factors.

217. **Sector layer, step 3: sectors in the live chat: sector legs in scenarios, peer spillover from company legs, and sector risks in the scan.** $0.15 (live test).
    - **Built:**
      - `planb/sectors.json` (`scripts/planc_sector_pack.py`): each sector's 2-session move (median, 90th, 95th percentile) beyond the macro factors, and its spillover fraction
      - `live.sector_loadings`: each holding's best sector and its loading (second stage on what the macro factors leave; kept if it adds more than 2%; macro loadings unchanged)
      - `price_scenario(sector_news=[{sector, direction, size}])`: typical = the median, big = the 90th percentile; through every holding's sector loading, with spread
      - company legs on a sector name add peer spillover
      - positions show the event, headline and sector parts
      - `scan.sector_risks`: a 1-in-20 two-session move in each sector the book leans on
      - cards and prompt updated
    - **Real-data check (NVDA, AMD, SMH, KRE, PFE):** sectors assigned (semis × 3, regional banks); "big positive semis" reaches NVDA, AMD and SMH only; scan: a bad semis day −$1.8k, regional banks −$348.
    - **Live test** ("US eases chip export curbs and NVDA announces a big AI deal"): NVDA +4.5% (event +$78, headline +$806, sector +$455), AMD +3.2%, SMH +2.9%, KRE and PFE flat; book +$2.9k (9 in 10: −$2.8k to +$7.7k; history's average +$436).
    - **Tests:** 447 pass.

219. **Noise is an answer: the analyst can say an event isn't market-moving, and every scenario carries code's verdict.** $0.15 (live test).
    - **Built:**
      - `event_evidence(market_moving=false)`: code then uses only the book's normal range (no real-event sizes), mode "not a market event"
      - every scenario gets a `verdict`: the middle vs the book's normal 2-session swing; < 0.5 → noise, < 1 → small, otherwise meaningful
      - the prompt tells the expert to lead with it and say "noise" plainly
      - the card shows it
    - **Live:** "What if Mitch McConnell resigns early?" on the trainer's book → "For your book this is noise: code's middle is about +$24, against a normal two-day swing of ±$2.2k" (and the Kalshi odds, 2.05%).
    - **Tests:** 449 pass.

221. **The event universe: two streams matched.** Design agreed; no code in this entry.
    - **Stream A, what's coming:** the calendar, a live news feed, prediction markets (Kalshi, Polymarket).
    - **Stream B, what would move this book most:** its biggest factor, sector and single-name exposures, and the events that would push them hardest (LLM proposals plus the replay of past event types).
    - **The match:** every candidate priced by the same code (the "what if" engine, a market's odds or a labelled base rate, the priced-in maths), then grouped:
      - topical and impactful = the headline list
      - impactful but not topical = tail risks
      - topical but harmless = "noise for you"
    - **State:**
      - prediction markets built (the v1 funnel, 145,214 → 222; live labelling and pricing #218)
      - v1 matching built (rules scoring, ladders, deadlines, "no fit")
      - calendars live but separate
      - the news feed built for research only
      - exposure-driven proposals designed in v1 but not in Plan B
      - the match not built; coverage never measured
    - **Plan, in order:** coverage test, stream B proposals, one calendar, news feed as candidates, the match, refreshed universe + Polymarket.

223. **The event universe built: both streams, priced by one engine, matched into a watchlist.** About $0.03 per scan (one proposal call).
    - **Stream A (`discovery.py`):**
      - markets: the labelled Kalshi universe (#218), closing within 91 days
      - calendar: next Fed/CPI/jobs/GDP (dates from the market universe; this book replayed through past releases), held-name earnings (look-through value × median earnings-day move), EIA and WASDE (the book's oil and farm exposure × how big those days usually are, #212), the US election date (replay of past big political events). OPEC not built (no reliable date source).
      - news: the last 7 days of the prospective log as "the story develops further" candidates, sized by the story's labelled severity and signs
    - **Stream B:** one LLM call gets the book's biggest exposures (factor $ per normal day, sector $, top look-through names, options) and proposes 6 events in the label format (signs + sizes per factor, sector and company legs), each linked to a market, calendar item or news story only when that's its catalyst; at least two unlinked tail risks. Code prices every one.
    - **Odds:** a market's odds when one is the event (priced-in split and risk left as #218). Otherwise a base rate (how often a big event of that kind happened within the horizon, 2016–2025; or how often a story was followed by another big one within 30 days), shown as context only, never used for the priced-in split, and labelled "not this event's odds".
    - **The match:** impactful = full impact ≥ half the book's normal two-day swing (the noise rule, #219). Headline = topical and impactful, ranked by risk left (markets) or size; tail risks = impactful proposals with no catalyst; noise for you = topical but small.
    - **Dates:** "coming" is judged from the wall clock when the data is live (prices lag; data date Oct 16, today Oct 24), so past earnings drop out.
    - **In the chat:** "what could hit me" now shows the risk card plus the watchlist card (replacing the upcoming card); the expert leads with the headline, then tail risks, then says what's noise. Proposals run only with a live LLM, so replays and tests are unchanged.
    - **The trainer's book (Oct 24):** 57 markets, 9 calendar items, 16 news candidates, 6 proposals. Headline: Taiwan blockade (−$4.3k, from a news story), Hormuz escalation, gas above $5.80 (−$4.2k, 10%), Fed Oct 28, Iran deal (+$2.6k, 9.6%: the USO puts gain $1.4k). Tail risk: AI capex bust (−$4.2k). Noise for you: EIA, GDP, PFE earnings, the election.
    - **Alternatives:** proposals without a link step (would double-count markets); base rates as odds (they're about a kind of event, so overstated for one specific event).

224. **Polymarket joins the universe, and the coverage test: markets rarely list the exact shock in advance (Oct 24).** $1.01 ($0.88 labelling, $0.13 judging).
    - **Polymarket:** read through topic tags (politics, geopolitics, economy, finance, world, business, tech, Fed rates, trade war), the top 50 events per tag by 24-hour volume, markets with at least $50k of volume, the 3 most-traded rungs per ladder: about 300 markets, normalized to the Kalshi format, cached per hour. Labelled with the same prompt (`discovery_label.py --polymarket`): 533 labelled in all, 310 move markets. The same question on both venues is kept once. Live data only; replays stay Kalshi-only.
    - **Effect on the trainer's book:** 57 → 154 priced markets. Proposals now link to Polymarket odds: Hormuz agreement 34% (+$2.8k), AI bubble bursts 9.7% (−$4.3k), strike on the Saudi pipeline 58% (−$1.6k).
    - **Coverage (31 big 2026 library events):**
      - an exact market open on the day: 5 (Hormuz weekly transits before the closure, the ECB September decision, Musk at $1T, the Starmer exit market, the Supreme Court tariff vote count)
      - a market on the same developing story: 14 more (some links are thin)
      - the story already in the news in the 14 days before: 4 (a lower bound: short titles)
      - The trainer's book: 12% of its 2026 event-driven loss was covered strictly, 72% counting same-story markets
    - **Why it matters:** markets price the story more often than the step. That's the case for stream B (book-driven proposals, priced without a market) and the news stream.
    - **Finding:** Kalshi's open-market list is cached for good, but the live chat refreshes each built market's quotes hourly, so the odds are current; only the list itself dates from the Oct 16 build. A full rebuild was not run; Polymarket adds the fresh markets.
    - **Alternatives:** a light Kalshi re-scan (150+ calls for a week's new markets; Polymarket covers the gap more cheaply); counting same-story markets as covered (too loose to headline).

225. **A run's universe = the user's events + Castle's, and Castle's excludes the user's.** About $0.15 per such turn.
    - **Router:** "I'm worried about X, and anything else my book cares about" goes to `scan_risks` with `user_events` (their words, up to 4; Plan B schema only). One named event and nothing else still goes to `analyze_event`.
    - **One LLM call** (the proposal call) first labels the user's events in the same format (directions and sizes, legs, a market link only if its YES is exactly that event), then proposes 6 others that are not those events, versions of them, or other routes to the same move. Markets that share words with a user event are added to its candidate list wherever they rank.
    - **Code:** prices the user's events like every candidate, always shows them first with a verdict (noise / small / meaningful vs the normal 2-day swing), and folds under each one any market, story or proposal that is the same event or another route to its main move (`same_move`: the largest-sized factors share a factor and direction other than the broad market). The other groups are therefore everything else. Opposite moves (an oil spike vs a Hormuz deal) stay in Castle's list.
    - **Live check:** the oil spike −$1.9k ("small for your book"; TSLA −$1.0k, USO puts −$710, GLD +$322), folding gas above $5.80 (10%), Iran nuke (4%), WTI at $120 (5%), Saudi airspace closed (20%). Beyond it: AI bubble bursts −$4.2k (9.7%), hawkish Fed hold −$2.1k, 5-year at 5.25% −$3.5k (21%), Hormuz deal +$2.4k (34%). The reply: "Bigger risks are elsewhere."
    - **Alternatives:** a separate labelling call for the user's events (an extra call; the joint call also stops proposals from repeating them); folding by keyword (misses "Saudi airspace" as an oil-spike route).

226. **Part 2 of question 1: a probability on every event, in three tiers, and a forecaster tested against a naive LLM.** Plan only; no code in this entry.
    - **Tier 1, a listed market:** use its odds. A longshot correction is measured on markets resolved before Jul 1, 2026, never assumed.
    - **Tier 2, close to listed markets:**
      - (a) a different strike or deadline is derived in code (v1 ladders and hazard, not yet wired into Plan B)
      - (b) for a different but related event, the LLM classifies the logical relation only (subset, superset, exclusive, correlated, unrelated); code turns that into bounds. The LLM never adjusts odds (#182).
    - **Tier 3, nothing close:** a forecaster with 3 library agents (library base rate, similar past events, news frozen at the forecast date) and 2 web-search agents; the median in log-odds, Platt-calibrated, clipped to any tier-2 bounds.
    - **Test:** resolved markets treated as unlisted, from questions resolving Jul 1–Oct 23, 2026 (after the model's knowledge cutoff), evidence frozen at T. Arms: naive LLM, naive + base rate, the Worsey library forecaster, and the hidden market price as the benchmark. Brier with a paired bootstrap clustered by event, ~250 questions. Search agents only prospectively (web search can't be date-restricted).
    - **The claim for Castle:** beats the naive LLM on Brier, with the gap to the market reported plainly.
    - **Budget:** about $75–85 more. Proposed: raise the total cap to $160, gated on a $5 pilot. The cheaper version (~$25: 150 questions, no search agents) shows the method but probably not significance.
    - **Alternatives:**
      - an LLM "event market adjustment" that outputs adjusted odds (breaks #182; replaced by relation → bounds)
      - backtesting the search agents (leaks)
      - an SDE or hazard-rate forecaster

227. **Tier 1: the longshot bias is real, measured, and corrected only where it holds out of sample (PART2 step 1).** $0.
    - **Sample:** 88,256 snapshots of 19,792 resolved markets (Kalshi 53,100 snapshots of 11,246 markets; Polymarket 35,156 of 8,546), closing 2023-01-01 to 2026-06-30. Kalshi in-scope series without daily/weekly/intraday ones, volume ≥ 500; Polymarket topic tags, volume ≥ $10k; at most 4 markets per event; every interval clustered by event.
    - **Finding (method):** pricing markets "7 days before close" is biased, because close is when a market resolves, which comes early for "by date" markets that resolve YES. It showed the *opposite* bias (YES underpriced). Scheduled end dates don't fix it (Polymarket's end date is often the event's; Kalshi moves close_time on early closes). **Calendar snapshots** (1st and 15th of each month, every market still open) don't depend on the outcome.
    - **Finding (bias):** both venues show the textbook pattern. Prices of 2–35% overstate YES (7% → 3.7% realized; 14% → 10%); favourites at 90–98% are understated (92% → 95%). Same signs in the held-out year.
    - **Correction:** P = sigmoid(a + b·logit(price)), a = −0.10 [−0.16, −0.02], b = 1.17 [1.13, 1.21]. Out of sample (learned before Jul 2025, scored Jul 2025–Jun 2026): log loss 0.2694 → 0.2649 (−1.7%), Brier 0.0818 → 0.0811; the step-shaped isotonic map is slightly worse (0.2663). Applied only in bins significant in the full sample and again in the test year (0–35%, 90–98%); raw elsewhere; raw and corrected shown side by side. Parameters committed in `planb/longshot.json`.
    - **Alternatives:** isotonic (worse out of sample, step-shaped); no correction (leaves a 2× overstatement at 5%).

228. **Tiers 2a and 2b built; relation labels 34 of 35 right (PART2 steps 2–3).** $0.12.
    - **`planb/probability.py`:** tier 1 (price + longshot correction), 2a (strike ladders from "above k" markets or exact buckets summed into a survival curve, monotone, never extrapolated; deadline rungs made non-decreasing and interpolated, constant hazard only within 3× the time), 2b (one LLM call per event labels each candidate market's relation; code turns implies / implied by / exclusive into bounds and intersects them; narrow ≤ 0.10 → the midpoint, else tier 3 inside the bounds). The LLM never gives a probability.
    - **Test (`part2_pairs.json`, 12 events × 35 Kalshi pairs, labelled by the build assistant for the trainer to review):** 34 of 35 agree; the miss is arguably right (the Netanyahu market also resolves on an announcement, so it isn't exactly the event). All 4 bounds checks hold. Tier 2a examples: CPI above 0.35% → 15.3% (between the 0.3% and 0.4% rungs); gas above $5.50 → 12.5%; Hormuz normal by Feb 15 → 12.7% (between the Jan 1 and Apr 1 rungs).
    - **Not wired into the chat yet** (step 7 of the plan).

229. **The question set: 1,409 resolved questions, one per event (PART2 step 4).** $0.
    - Markets resolved 2026-07-01 to 10-23 (after the model's knowledge). Forecast date T is a calendar date (Jul 1, Jul 15, Aug 1, Aug 15, Sep 1), picked per market by a hash among the dates when it was open and at least 14 days before close; never "N days before close" (#227). Reflexive price-level markets dropped; market price at T in [2%, 98%]; YES rate 40%; median horizon 29 days.
    - **Finding:** a random draw is mostly local primaries, tweet counts and company KPIs. Only 29 of the first 400 are about the library's event kinds other than elections (the "elections" label caught 120, mostly local races). The pilot uses those 29; the backtest needs a bigger labelled pool (about 10 label calls for ~100 event questions).

230. **Pilot: the library forecaster beats the naive LLM and ties the market on 29 event questions (PART2 step 5).** About $3.20 in all (including two broken runs).
    - **Arms:** A naive LLM (question, rules, date); B + library base rate; C three library agents (base rate, 8 most similar past events by local embeddings, Wikipedia news from the 14 days before T, in different orders), median in log-odds; D the hidden market price at T. Thinking off for every arm.
    - **Brier:** A 0.167, B 0.172, **C 0.139**, D 0.143. **C − A = −0.028 [−0.060, −0.0001]**; C − D = −0.004 [−0.108, +0.096]; B − A = +0.005 (the base rate alone adds nothing: the gain comes from the similar events and dated news).
    - **Caveats:** 29 questions; the market benchmark is noisy on this set (e.g. a 95% hurricane-name market that resolved NO); no Platt scaling yet (needs the backtest's held-out split); the search agents are untested (prospective only).
    - **Findings (method):** the model spent a 400-token budget thinking and wrote no answer (fixed: thinking off via `"between_tools"`); models restate or truncate JSON (fixed: read the last "p"); the chat's 300 calls/hour limit stops a pilot midway (raised for that run only).
    - **Gate:** passed (C beats A). The backtest (~250 questions, ~$30–40) doesn't fit the remaining ~$7 under the $80 cap: the trainer's call.

231. **The past-event replay feeds the proposal call.**
    - **Problem:** `live.scan()` (the risk card) is the one data-only view of what has historically hurt this book, but it fed nothing. Stream B proposed from exposures alone, and the card and the watchlist could disagree without anything reconciling them.
    - **Change:**
      - `watchlist(..., replay=None)` takes the scan's rows; `loop._scan_risks_planb` passes the card's `table["risks"]`. Called on its own, the watchlist computes the replay itself (code, free; only when proposals run; the session's stored card scan is left as it was).
      - `discovery.history()` puts the top 4 one-off categories by bad case into the prompt, from code. Each has its bad case, how often, what history says, the worst past example and the 2 holdings carrying most of the bad case. Releases are left out: they're in the calendar.
      - The instruction: unless the user's events cover them, at least 2 proposals must be current versions of the top categories, with their catalyst. Don't repeat the numbers.
      - Each proposal returns `from_history` (a category or null). Code validates it against `live.CATEGORIES` and drops anything else. The card shows "from history: <kind> (bad case −$X)", with the replay's number.
      - With no replay rows, the prompt is byte-identical to before.
      - The risk card and the analyst prompt are unchanged. #182 holds: the LLM still gives only directions, sizes, legs and links.
    - **Live comparison (`docs/pmrm/results/replay_feeds_proposals.md`), the trainer's book:**
      - proposals in a top-3 replay category (financial stress −$4.5k, supply shock −$2.6k, disaster −$2.5k): **1 → 3** of those shown; 4 claim a top-3 category as `from_history`
      - new: Saudi East–West pipeline strike (Polymarket-linked, from supply shock), a regional-bank and credit contagion tail (−$3.4k), China rare-earth export controls tail (−$2.1k)
      - dropped: the China–Taiwan blockade tail (−$4.3k, war isn't in this book's top-3 replay) and the December Fed-hike proposal; tail risks 1 → 2
      - "natural disaster" (3rd) was not instantiated
    - **Normal and weird books, after:** not run. Worsey's hourly LLM gate (300 calls per rolling hour) was full from the concurrent part-2 session; the script redoes only the failed books.
    - **Cost:** $0.20 (4 calls).
    - **Finding:** the history pulls proposals towards the replay's categories, at the expense of the categories outside it. Here the biggest tail before (Taiwan, −$4.3k) dropped out. The instruction says "at least 2", so this is the LLM's choice, not a rule. Worth watching on the other books.
    - **Alternatives:**
      - leave the replay as its own card only (keeps the disagreement)
      - replace the risk card with the watchlist

232. **Company legs only for company-specific events; with a sector leg on the same holding, the larger of the two.** $0.35.
    - **Problem:** for broad events the LLM listed famous names ("AI bubble burst" → NVDA−, MSFT−), and `impact()` added a full company-news day on top of the factor and sector moves: a double count, on arbitrary names.
    - **Rule (both prompts):** companies only when the event is about that company (its drug, earnings, export licence, CEO); never for broad, macro or sector events. Usually empty.
    - **Code (`discovery.combine_legs`):** a holding with both legs from one event gets the larger of the two, never the sum. The task proposed skipping the company leg; the trainer agreed to "larger" instead, because an event aimed at a company (an export ban naming NVDA, with a semis leg) should keep its own move. Rows record `company_legs_skipped` / `company_legs_over_sector`.
    - **Relabel:** labels naming companies 116 → 31 of 533. The relabel re-drew whole labels (factor sizes moved for unrelated reasons), so only the new `companies` field was kept; the rest is the earlier label's.
    - **Effect:** The trainer's book nearly unchanged (Musk net worth market +$482 → +$173: its TSLA leg is gone); normal book: nothing over $100 (largest $76); weird book: "Pandemic in 2026" +$153 → −$309 (MRNA leg dropped). On a mega-cap tech book, the OpenAI markets halve (≈ +$870 → +$460) and "AI bubble burst" loses its double count (−$4,594 → −$4,183). The larger-of-two rule fired 0 times in these books after the relabel: a backstop.
    - **For the trainer to judge:** the pandemic market losing MRNA (a direct beneficiary, arguably); "Altman out as OpenAI CEO" keeping MSFT− and NVDA−.
    - **Finding:** sector loadings are already rolling (504 sessions, refit daily); not tested against a shorter or weighted window.
    - **Alternatives:** drop company legs entirely (loses real company events: FDA, CEO exits, deals); keep them but halve them (still arbitrary names); skip whenever a sector leg exists (loses the company's own move on events aimed at it).

233. **Part II end to end in the chat: every event's probability carries a tier; the forecaster runs on the top 3 unpriced events; project cap $90.** About $0.25 per scan.
    - **Tier 1 (code):** market odds are shown longshot-corrected (#227), with the raw price kept (`odds_raw`). The priced-in split uses the raw price (what asset prices carry); the risk left uses the corrected probability. E.g. the 5-year at 5.25%: 21% → 15.9%; gas above $5.80: 10% → 6.4%.
    - **Tiers 2a/2b:** one batched relation call (`probability.classify_many`) for every proposed or user event without a market, against its top 6 keyword-matched markets; code resolves tier 1 / 2a / 2b (`tier_from`) with no further calls.
    - **Tier 3 (`planb/forecaster.py`):** the pilot's three-agent forecaster (#230) for the top 3 still-unpriced events by impact, in parallel, clipped to any 2b bounds; shown as "Worsey forecast", with the agents' spread; the priced-in split is labelled "assumes markets share this forecast". The rest keep the labelled base rate.
    - **Card and expert:** odds tagged mkt / derived / bounded / forecast (hover: raw price, label, agents' spread); the expert is told to say each probability's tier.
    - **Live check:** 8 headline events at tier 1 (corrected); the two tail risks forecast at 4% (private-credit or regional-bank failure; agents 4–6%) and 22% (China export controls; agents 20–25%). Scan cost $0.25.
    - **Finding:** that scan took 175 s (about 50 s before). The embedder load is 9 s (now warmed at server start); the rest is the extra LLM rounds (relation call, then forecasts, then the reply). Not optimized tonight.
    - **Budget:** `WORSEY_LLM_MAX_USD_TOTAL` default 80 → 90.
    - **Alternatives:** forecast every unpriced event (3 calls each; too slow and costly per scan); per-event relation calls (more calls against the 25-per-turn cap).

234. **One owner for odds; the output is the user's events plus three Castle picks, sent as messages as the scan goes.** About $0.28 per scan; results in `docs/pmrm/results/proposal_roles.md`.
    - **Odds:** a proposal's market link is the catalyst and a hint, never its odds; every proposed and user event goes through the tier step (one batched relation call). Candidate markets are now retrieved by meaning (the local embedder on market titles) plus title words; word overlap over titles + rules missed obvious matches.
    - **Dedupe and grouping in code:** a proposal the relation call finds IS a listed market or news story absorbs it; topical = tiers 1/2a/2b or a calendar/news catalyst. Same-move overlap alone doesn't dedupe (it would merge distinct events).
    - **Forecaster:** every unpriced user event, always; plus the top 3 unpriced proposals above the size screen. Related markets are passed as labelled evidence.
    - **Picks:** Castle's three by risk left (|move| x sqrt(p(1 - p)); the trainer preferred it to total move once explained), among events with a real probability, no two with the same main move; the rest counted.
    - **Messages:** book at a glance (code), your events (code), "researching the odds" (code), then the analyst's last message. The watchlist card is no longer sent.
    - **Live:** final message at 92-95 s (175 s before), first at ~21 s; $0.27-0.29 per scan.
    - **Alternatives:** a fixed universe of 7 (it changes nothing: pricing in code is free and LLM work is bounded by what's shown); ranking by total move (a 0.5% doomsday event would always top the list); dedupe on same move (merges distinct events).

235. **An option's underlier moves exactly like the stock: the same legs and one shared own-noise draw.** $0; results in `docs/pmrm/results/option_underlier_legs.md`.
    - **Sanity check:** the gap was real in the watchlist (`discovery.impact`) and the one-event path (`live.price`): stocks got sector and company legs, options on them factors only. The black-swan path and the history replay are factor-only for stocks too, so they were already consistent and are unchanged.
    - **Legs:** an option's underlier gets exactly what the stock gets in that path (watchlist: factors + the larger of sector and company leg, #232; one event: factors + the analyst's sector legs + user-named company legs, on the same spread draws).
    - **Shared noise:** the book's own noise and each held option underlier's own noise are drawn jointly from their measured residual covariance (`live.shared_noise`), so a stock and its options share one draw per scenario and the book's own-noise spread is unchanged. Options on names not held keep an independent draw; books without options on held names get identical draws to before.
    - **Effect:** WDC calls under "AI bubble burst": underlier −7.57% → −8.43% (= the stock), P&L −$1,728 → −$1,887. A protective WDC put on a WDC + GLD book: 9-in-10 range ±$2,855 → about ±$1,100 (the put now hedges WDC's own swings). The trainer's USO puts unchanged.
    - **Not done:** an ETF option doesn't get a company leg through the ETF's look-through weight (the ETF holding itself doesn't in the watchlist either).
    - **Alternatives:** legs only (leaves single-stock hedges unable to offset a stock's own swings); per-name independent noise (would change the book's own-noise spread, which includes the names' correlations).

236. **The live labelling prompts, backtested: as good as the research prompt; the news labeller needs factor definitions .** $5.68 (dry run $4.09); results in `docs/pmrm/results/live_labeller_backtest.md` and `docs/results/labeller_backtest.md`.
    - **Direction on big moves (745 major/extreme events, text only):** market labeller 74% (70%–79%), user-event labeller 73% (68%–77%), research read #196 75%; news labeller 65%.
    - **Size:** ordered for both live labellers (L1 small/medium/large 0.90/1.55/2.70× noise; L2 0.84/1.30/2.53×).
    - **Clean set:** 32 events after Jul 1, 2026; too small to settle (L1 70% on moves over 1×, n 20, vs 71% before).
    - **Finding:** the news labeller's prompt (`planb_grow.PROMPT`, used by the prospective log) never defines the factors; its credit signs agree with L1's defined credit 71% of the time and score 53% on big credit moves. Proposed (not done): add the factor definitions L1 and L2 carry.
    - **Method:** prompts byte-identical; wrappers only (an event as a market's YES for L1; as a user's event with n = 0 for L2, 8 per call); L3 and #196 scored from existing labels.

238. **A country layer: a country's own shock reaches its holdings.** $0.12 (one label check, the relabel); results in `docs/pmrm/results/country_layer.md`.
    - **Problem:** "Brazil invades Peru" left EWZ at −0.2%: no country factors, and labels had nowhere to say "Brazil".
    - **Built:** 31 country factors (each country ETF beyond the 13 macro factors; `planb/countries.json`). Each holding gets its own country slot beside its sector: country ETFs map to their own country; foreign stocks to their home country (`planb/domicile.json`, 225 names; kept above 2%); anything else to the best fit only if it adds more than 2% held out beyond the macro factors and its sector.
    - **Legs:** `countries` in labels (only countries the event is about or hits), `country_news` for the analyst; the largest of sector, company and country leg per holding, never the sum (`combine_legs`, #232); option underliers follow (#235).
    - **Effect:** EWZ on "Brazil invades Peru" −$60 → −$934 (−3.1%); foreign stocks' held-out explained share +13.7% (median); the trainer's, the normal and the weird book unchanged on current labels. 26 labels relabelled (countries field only).
    - **Alternatives:** the EM factor scaled by region (moves every EM holding on a single country's shock, and misses developed countries); regional factors in the macro layer (changes the validated 13-factor map; too coarse: Mexico isn't Brazil).

239. **Drawing the scenario spread from past events of the same category: tested, not shipped .** $0; results in `docs/pmrm/results/draws_by_category.md`.
    - **Idea:** resample a scenario's misses from past events of the same category and severity (≥ 20; else the category's major + extreme events; else the severity pool), so its co-movement matches its kind.
    - **What-if path (what the chat uses), held out by year 2023–26, 1,956 book-events:** interval score worse by +0.19 noise units (90% interval +0.08 to +0.32); worst on trade tariffs (coverage 89% → 80%); no change on conflict and supply shocks.
    - **History path (the range tests):** lead over the naive LLM $29 → $34 (books), $37 → $43 (positions); coverage within about a point; CRPS against predict-zero a hair worse.
    - **Decision:** not shipped (fails the predeclared rule on the live path). Code kept behind `WORSEY_DRAW_POOL=category`; every scenario now reports its `draw_pool`.
    - **Alternatives:** per-factor independent draws (rejected: drops co-movement and understates the book's tail); a shrunk category pool, or misses matched by label similarity (next steps).

240. **Scorecard vs a raw LLM (B2): all 39 runs; Worsey loses on coverage, wins on ranges against the raw LLM, loses to the floor.** Spend about $9; results in `docs/pmrm/results/scorecard_vs_raw_llm.md`.
    - **Design:** 13 books × 3 dates, rebuilt as of each date, scored over the next 30 days; blinded LLM judge.
    - **Part I:** the raw LLM named 34% of realized events, Worsey 16% (Castle's 3 picks 8%). The universe is traceable and priced, not more complete (consistent with #224).
    - **Part III:** interval score $9,814 (Worsey) vs $31,696 (raw) vs $6,427 (predict-zero plus noise); raw 9-in-10 coverage 38%.
    - **Part II:** Brier 0.038 vs 0.177, not comparable (different question sets).
    - **Finding:** an as-of-date run is slow because of the price API's rate limit (hours per new book), not LLM calls.
    - **Alternatives:** none run; the design was predeclared.

242. **Text-only event messages in the live chat: one message per event with its probability, each holding's average move and 9-in-10 range, and the book's; no event cards.**
    - **What changed:** `discovery.impact()` now computes every holding's dollars in every draw (the draw's factor moves, the event's legs, the holding's own noise shared with its options per #235, options repriced in each draw), so each holding gets its own average and 9-in-10 range and the holdings add up to the book draw by draw; `by_position` lists every holding (no top-4 cut). `scan_messages.event()` writes each event's message in code. On the Plan B path the event pipeline's cards are dropped (`agent/session.py: TEXT_ONLY_CARDS`); the options and ETF desks keep theirs; v1 replays keep all cards.
    - **"Expected" = the average of the draws if the event happens** (not weighted by the probability, which is shown beside it). `full_impact_usd` (the middle) still drives ranking and the priced-in split, unchanged.
    - **Dropped from the task:** the worst-past-event line and the relabel it needed; the single-event "what if X" path still writes its own reply (now without the scenario card).
    - **Live runs** (XOM / NVDA / TLT): "what could hit my portfolio?" 113 s, $0.29; with "I'm worried about China blockading Taiwan" 91 s, $0.29 (the one saved in `docs/pmrm/results/text_only_messages.md`); from a fresh clone of the submission repo, empty cache, 176 s, $0.21. Grounding passed in all three.
    - **Alternatives:** keep the middle instead of the average (rejected: "expected" means the average); book range only (rejected: per-holding ranges fit on one line each).
