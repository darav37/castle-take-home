# Prediction-market portfolio response model

**Implementation brief for the build assistant | Version 1.7 | 19 October 2026**
**Status:** research specification, not a validated model.  
**Companions (in `docs/pmrm/`):** `whitepaper_v1.5.pdf`, `explainer_v1.5.pdf` and `SIMPLE.md` describe the model; v1.6 changes only the experiment program and operations, not the model. `BUILD_PLAN.md` is the build order for the build assistant. This specification is normative for implementation; neither document reports a real-data backtest.

**v1.1 clarification:** the same joint model supports prediction-market positions and mixed stock/event portfolios. Section 5.1 defines event-to-event conditioning, contract units, and full portfolio covariance. The original stock-only experiment remains the primary benchmark unless the evaluation scope is changed before test inspection.

**v1.3 change (IC redefined):** IC is no longer only a correlation-interval penalty. It is the set of five jobs the LLM does around the empirical model (section 4.3): **J1** selecting the event universe for a portfolio, **J2** cleaning outliers and labelling observations for high-frequency identification, **J3** injecting ranges and priors, **J4** embedding Type II events, and **J5** proposing factor additions or removals and naming solved factors. **J2 is the primary IC contribution at this stage**; it is tested first and on its own (stage 3). Every job keeps the v1.2 rule: the LLM selects, labels and reasons; code computes every number, and each job must earn its place in a matched ablation.

**v1.4 changes (review):** J2 is split into **J2a** measurement cleaning and **J2b** news attribution; real company-specific news is kept as portfolio risk. The fit has two steps: loadings and cross-covariances from cleaned data, each variable's own variance from raw data (section 4.2). J2 *proposes evidence-supported attributions*; it does not establish causes. Communality priors are reworded. J5 gains a check for technical (venue-wide) factors. J4's test gains a closest-analog baseline. News-dependent responses are added as a later experiment gated on power.

**v1.5 changes (final review before implementation):** the variance top-up is an *experimental estimator* with its own test and a shared-risk check, and the no-treatment path returns the raw model by construction. J2 labelling is one request per flagged entry, so blinding holds. Only **documented invalid observations** are excluded unconditionally; rule-change repricing and thin-market moves are real prices whose exclusion is a hypothesis tested on validation. Power figures are tied to a shipped, reproducible simulation with stated assumptions. The experiment runs as a staged sequence with **one predeclared primary comparison** (section 7). Specification work stops here: the next changes should come from the data audit and first benchmark results.

**v1.6 changes (experiment program and operations; the model is unchanged):** the experiment runs as two studies. **Study A** is the full study on the toy universe, the simplest complete example. **Study B** varies the universe from there: event and stock pools by category, placebo stocks, stratified random universes from 2×2 upward, a pooled primary comparison and a cross-universe stability check (section 0.1). The number of factors is chosen per universe on validation, from 0 to an identification ceiling, with parallel analysis and BIC as diagnostics (section 6). Every LLM call goes through hard per-run caps on top of Worsey's existing budget gate, with a dry-run cost estimate and a permanent label cache (section 6.1). The current Worsey engine enters the harness as a comparator, `V1_current` (section 7). Kalshi and Massive are not reachable from Claude's cloud sandbox, so the build runs where the keys and network are.

**v1.7 changes (after the P0 audit; Study A's universe and the quote-age reading still await the trainer's ruling, see `results/audit.md` §6):** (1) **Calendar first.** Scheduled releases (FOMC decisions and minutes, CPI, the jobs report, GDP, company earnings) are tagged by code from published calendars before any LLM call; the LLM labels only flagged moves the calendar does not explain (section 4.3, J2). This is cheaper, deterministic, and gives a ground-truth sample to measure the labeller's accuracy. (2) **The thesis is signal concentration.** Daily comovement is diluted: the event-stock link is carried mostly by news days and averaged with noise days. The P0 power simulation modeled only masking contamination, under which the variance top-up cancels most of the gain on the conditioning event; it did not model news-day concentration, which is what news-day up-weighting (T4, T5) targets. A new checkpoint P0b simulates it, and re-analyses the current engine's significance table by news day, interior band and category against the chance rate, before the primary comparison is frozen.

## 0. Assignment and boundaries

Build a reproducible experiment inside the user's existing data-testing environment. Inspect the repository's instructions, price/probability adapters, conventions, and testing tools first. Reuse them. Do not build a frontend, agent orchestration platform, new market-data integration, trading system, or graph database. The user already has ingestion and agents.

The deliverable is a comparison of empirical conditional-response estimators, plus a transparent audit trail. It must be possible to conclude that IC adds no value, that a factor model is unnecessary, or that data are insufficient.

**V1 question:** using a model frozen before an interval, what distribution of same-interval stock returns is associated with a modest observed change in one event's quoted probability?

This is conditional reconstruction, not a forecast of the event change, not a causal intervention, and not a tradable strategy backtest. The conditioning change becomes known at the interval's end. Freeze the response function beforehand; do not label the resulting conditional estimate an ex-ante forecast.

### What V1 includes

- A synchronized daily panel of probability changes and total stock returns.
- A regularized full-covariance benchmark and low-rank-plus-diagonal factor models.
- IC universe selection (J1), feeding the human-reviewed seed universe.
- IC measurement cleaning (J2a) and news attribution (J2b) on training data, with a two-step fit that keeps real idiosyncratic risk: the primary IC experiment.
- Optional IC soft priors on rotation-invariant quantities: total same-interval correlations, communalities and conditional-response signs (J3).
- Gaussian conditioning, hypothetical portfolio P&L, walk-forward evaluation.
- Numerical correctness, provenance, data-quality exclusions, and reproducible reports.

### What V1 excludes

- Binary resolution scenarios, probabilities bumped to 0 or 1, and an eight-world outcome table.
- Causal claims or shock-source attribution from correlation alone.
- Separate event-to-factor and factor-to-stock causal propagation stages.
- Factors whose economic meaning is assumed rather than tested. Semantic names from J5 are display labels computed from code-measured evidence; they never feed back into fitting unless a separately tested anchored model is promoted (section 12).
- Post-fit impact overrides and automatic pairwise edge propagation.
- Synthetic historical IC presented as genuinely point-in-time intelligence.
- Type II/unlisted events in the primary empirical score. J4 (section 4.3) and section 13 define their embedding and its separate leave-one-out test.

### 0.1 Stages of the initial experiment

| Stage | Work | Evaluation |
|---|---|---|
| 0. Data and power audit | Common overlap, jump-day counts, quote staleness, independent episodes, and the smallest response the planned test can detect at the available row counts. | Gate: if the planned comparisons cannot detect a plausible effect, change the universe (e.g. longer-lived recurring contracts) before any fitting. |
| 1. Exposures, universe and matching (J1) | Resolve holdings, let J1 propose events and context variables, verify scenario-to-contract definitions. Freeze a human-reviewed seed universe. | Evaluate this pipeline independently of numerical model scores. |
| 2a. M0 → M1 | Does event conditioning help at all? | Same verified mappings, holdings, eligible rows and chronological splits. |
| 2b. M1 → M2 | Does the factor structure help? | Same. |
| 3a. M2 → M2a (add J2a) | Does evidence-based measurement cleaning help? | Identical scored test rows, scored **uncleaned**. |
| 3b. M2a → M2c (add J2b and the variance step) | Does news attribution add value beyond cleaning? The variance step is also tested on its own (section 4.2). | Same. |
| 4. M2c → M3 (add J3) | Do numerical IC priors improve the remaining errors? | Same inputs and comparable tuning budgets. |

**One primary comparison, predeclared:** M2c against matched-k M2, i.e. the full J2 contribution to the factor model. Every other step in the sequence is explanatory: it shows where any improvement comes from. Stage 2a is also a gate: if conditioning does not beat M0, later stages are reported as exploratory only.

### 0.1a Experiment program: Study A, then Study B

**Study A: the toy universe, end to end.** The seed universe (section 2: events A, B, C; XOM, DAL, MSFT), run through every stage above. It is the simplest complete example and the first thing to finish. Its primary comparison is M2c vs matched-k M2 on that universe. If the seed contracts fail the stage 0 gate, Study A uses the nearest feasible substitutes (same three themes: an oil, a growth and a Fed-path contract, preferring recurring series) chosen and documented before any model is fitted.

**Study B: varying the universe.** Run after Study A, with the same code and the same frozen settings.

1. **Pools (built from the stage 0 audit, frozen before fitting).**
   - *Event pool, by category:* Fed decisions, CPI, payrolls, growth/recession, energy and commodities, trade and geopolitics, elections. Recurring series first; only contracts that pass the audit.
   - *Stock pool, by expected exposure:* rate-sensitive (banks, homebuilders, utilities), energy, travel and consumer, mega-cap tech, defensives.
   - *Placebo stocks:* names with no plausible link to any pooled event. Every model should predict about zero response for them. A cleaning or prior layer that "finds" a link there is a red flag, reported as a failure.
2. **Random universes.** Stratified random draws with a fixed seed: 20 universes each at 2×2, 3×3 and 5×5 (events × stocks), and 10×10 if the audit shows enough overlapping history. Each draw takes events and stocks across categories in proportion to the pools. Random draws stop anyone from picking universes that flatter the model.
3. **Designed universes, in addition:** *matched* (e.g. energy events × energy stocks), *mismatched* (energy events × tech stocks), and *placebo-only*.
4. **One pooled primary.** Study B's predeclared primary is M2c vs matched-k M2 **averaged over all randomly drawn universes**. Universes share days, so uncertainty comes from a block bootstrap that resamples **days** (all universes' scores for a day move together), never from treating universes as independent. Breakdowns by size, category mix and designed universe are explanatory.
5. **What each axis tests.** Size mainly tests the factor layer (M2 vs M1 should improve as d grows; at 2×2 a full covariance is adequate). Category mix mainly tests where IC helps.
6. **Cross-universe stability.** For each (event, stock) pair that appears in several universes, report the spread of its fitted conditional slope across universes. Large spreads mean the factor model is unstable and are reported as such.
7. **Data limits.** Complete-case overlap shrinks as d grows. Larger universes use the masked likelihood (section 4.2) for raw models too, with missing entries never treated as zero; M1 uses the EM covariance. If 10×10 universes cannot be built from overlapping history, Study B stops at the largest feasible size and says so.

J2 labels are per variable per interval, so they are shared across every universe containing that variable: labelling cost scales with the pools, not with the number of universes.
| 5. Type II embedding (J4) | Leave-one-listed-event-out embedding test (section 13). | Separate score; never mixed into the stage 2-4 headline. |

J5 (factor changes and naming) is not an accuracy stage: factor-count changes go through the section 12 refit test, and names are display labels with stability diagnostics.

IC remains a planned part of the initial experiment. Freeze prospective labels and priors before their test outcomes become available, even if stages 3-5 are implemented later. J1 assistance in stage 1 does not by itself enable numerical IC penalties. Setting lambda=0 removes the J3 penalty; it does not remove the factor structure or turn M2 into M1. Do not select historical events because they subsequently produced large losses.

### 0.2 Upstream backend pipeline

Reuse existing data adapters and agents. These are backend responsibilities even when the frontend collects the request.

1. **Resolve exposures.** Use dated holdings and ETF constituent weights. Aggregate direct and indirect holdings without counting both an ETF and its decomposed holdings as separate economic exposure. Retain cash, unmapped exposure and coverage denominators; do not redistribute missing weights.
2. **Discover events (IC job J1, section 4.3).** A portfolio-only request needs proposed risks linked to holdings, with evidence and timestamps. Business, geographic and supply-chain relevance are discovery evidence, not measured numerical sensitivity.
3. **Normalize the scenario.** Record actor, action, affected entities, threshold/operator/unit, deadline/timezone, path versus endpoint, announcement versus implementation, and resolution criteria. Material ambiguity requires clarification or abstention.
4. **Retrieve candidate contracts.** Text/vector retrieval proposes candidates. Read their full rules and versions; titles or similarity scores cannot establish equivalence.
5. **Verify the relationship.** Return one of `EXACT_EQUIVALENT`, `SCENARIO_IMPLIES_CONTRACT`, `CONTRACT_IMPLIES_SCENARIO`, `PROXY`, `NO_MATCH`, `AMBIGUOUS`. Combine deterministic field checks with LLM interpretation and retain evidence. Human-review the initial seed universe.
6. **Check model eligibility separately.** Semantic equivalence does not establish adequate quote quality, common history or small-move support. Only verified equivalent definitions with usable data enter automatic analysis of the original scenario.

Logical implication can establish a probability bound if the definitions support it; it does not establish equal probabilities or sensitivities. Analyze an imperfect proxy explicitly as that proxy. Do not transfer its probability or response coefficient to the bespoke scenario, average near-matches, or sum overlapping contracts. A no-match result routes to the separately labeled Type II workflow or unavailable status.

Required contrast cases include: at least 25% versus exactly 25%; announced by March 31 versus in force by March 31; tariffs on any country versus China; occurrence by a date versus on that date; any Fed cut versus exactly 25 bp. Matching confidence is not event probability.

### 0.3 Output contract and priced-in interpretation

Return three separate fields: (1) quoted probability of the verified contract with quote timestamp and definition; (2) conditional P&L for a specified small probability change and interval, with model version and uncertainty; (3) resolution-impact support status, initially `NOT_SUPPORTED_BY_SMALL_MOVE_MODEL`. Include match status, proxy differences, exposure coverage and rejection reasons.

This engine does not estimate the probability already embedded in equity prices. Historical comovement around a prevailing state does not solve terminal valuation. Do not extrapolate to `x=1-p`, apply an additional `1-p` discount, or label a small-move response as the loss if the event occurs. Unsupported large moves and boundary violations must remain flagged rather than silently clipped into plausible-looking results.

## 1. Audit decisions that supersede the discussion

| Earlier ambiguity | Binding decision |
|---|---|
| Daily data described as immediate repricing | V1 is a close-to-close daily response model. Intraday repricing requires a separately aligned and validated dataset. |
| Eight terminal worlds mixed with daily changes | The outcome table is out of scope; do not combine it mechanically with covariance. |
| EC and IC as two effects | One fitted covariance. EC supplies observations; IC acts on what enters the fit (J1 universe, J2 cleaning) and optionally regularizes the fit (J3). IC never adjusts outputs after fitting. |
| IC = correlation priors only (v1.2) | IC has five jobs (section 4.3). Cleaning and HF identification labels (J2) are the primary contribution; correlation-style priors (J3) come after and must beat a cleaned no-prior model. |
| Cleaning removes "contamination" | Only documented invalid observations are excluded unconditionally. Rule-change repricing, thin-market moves and company-specific news are real prices or real risk: excluding them from the loading estimate is a hypothesis tested on validation, and they stay in each variable's own variance. |
| "No identifiable news" means bad data | It does not. Thin liquidity or a rule clarification can produce real, tradable repricing. |
| LLM supplies numbers for priors | The LLM chooses from fixed categorical buckets; code maps buckets to numerical intervals, weights and scales. News timestamps are copied from cited sources, never generated. |
| No direct causal effect implies zero correlation | False in general. Priors must address total observed comovement, including common causes. |
| Latent factors named oil/growth/rates | V1 factors are anonymous. Names require anchors or restrictions that constitute a later model. |
| Recession implies a 25 bp cut | The proposed contract is **exactly** 25 bp, not any cut. Increased 50 bp probability can reduce its probability. Do not hard-code a positive sign. |
| A probability shift has one universal impact | Conditional association averages over news sources represented in the sample. Source-specific interventions need additional variables/identification. |
| Gaussian intervals represent all uncertainty | They are plug-in process intervals; parameter, IC, measurement, and structural uncertainty are separate. |
| LLM prior = Bayesian posterior | V1 uses prior-like penalized likelihood, not full Bayesian inference. |
| Eight-way sample or many daily ticks validates outcome dependence | Neither identifies outcome calibration across independent episodes. |
| Small panel is guaranteed to be backtestable | Audit common overlap and independent episodes first. Fail clearly if insufficient. |
| The graph is a new computation engine | It is an optional view of loadings; conditioning uses the covariance matrix. |

## 2. Universe, time, and notation

### Seed universe

- `event_A`: oil record high by a specified deadline; exact contract rules govern instrument, threshold, path, and resolution source.
- `event_B`: US recession by a specified deadline; do not infer the contractual definition from the title.
- `event_C`: exactly 25 bp decrease at the specified December Fed meeting; preserve venue rounding rules.
- Stocks: `XOM`, `DAL`, `MSFT`.
- Example portfolio: USD 10,000 of each at each scored interval's start. This is an explicitly rebalanced analytical exposure, not a self-financing investment performance series.

The 2026 listings discussed are candidate identifiers, not guaranteed liquid histories. Resolve exact venue IDs and rules through existing adapters. Never substitute an almost-matching market silently. If sufficient common history does not exist, report the gap and offer a documented replacement universe or prospective-only collection.

### Definitions

- `t`: end of a synchronized observation interval; `t-1` is its preceding eligible scheduled endpoint, not an arbitrary previous surviving row.
- `p[j,t]`: event j quote midpoint in probability units, 0 to 1. A price-based proxy, not a claim of true probability.
- `dp[j,t] = p[j,t] - p[j,t-1]`: absolute change; `0.01 = 1 percentage point`.
- `TR[i,t]`: validated total-return price/index for stock i.
- `r[i,t] = TR[i,t] / TR[i,t-1] - 1`: arithmetic total return.
- `z[t] = [dp[A,t], dp[B,t], dp[C,t], r[XOM,t], r[DAL,t], r[MSFT,t]]` in that exact stored order.
- `d = 6`: number of observed variables. General implementations may support other d.
- `n`: number of usable training intervals.
- `mu`: d-vector of training means; `s`: d-vector of training standard deviations using `ddof=0`; `D = diag(s)`.
- `u[t] = D^{-1}(z[t] - mu)`: standardized observations.
- `k`: latent-factor count, chosen per universe on validation (section 6); rank 0 (diagonal, no shared structure) is always a candidate.
- `h`: signed dollar exposures at interval start, ordered like stock returns.

### Interval policy

Default endpoints are 16:00 America/New_York on full US equity trading sessions. Store UTC instants and the exchange calendar/version. For each event use the most recent valid quote **at or before** the endpoint. Never take the nearest quote from the future. Stock endpoint availability and closing-price timestamps must match the declared convention.

For the initial core daily sample, require a 23-25 hour elapsed interval, accommodating daylight saving changes. Exclude multi-day weekend/holiday gaps and shortened sessions under a predetermined rule. Keep them in a separately labeled evaluation cohort if desired; never treat a Friday-to-Monday interval as identical to a normal day without disclosure. Missing a scheduled endpoint invalidates its associated interval; do not bridge across missing data automatically.

Daily data with incompatible vendor timestamp conventions are not a synchronized daily panel. Report the mismatch and pause the comparison until a common convention is defined.

## 3. Data contracts

Use the project's native storage format. The following are logical schemas, not a demand to install a new database. IDs and times must be stable; JSON metadata must be machine-readable.

### 3.0 Upstream audit records

Store these logical records in the project's existing storage system:

- `exposure_snapshot`: as-of timestamp, source holdings IDs, underlying exposures, ETF weights and dates, sources, currencies, cash/unmapped amounts and coverage denominators.
- `event_candidate`: candidate ID, portfolio/request ID, normalized proposition, relevant holdings, relevance rationale, evidence references and information cutoff.
- `match_decision`: candidate ID, retrieved contract IDs/ranks, rule hashes, field agreements/mismatches, relationship class, unresolved material fields, evidence, model/prompt version, reviewer and decision timestamp.
- `analysis_eligibility`: match decision ID, semantic status, quote-quality/history/support checks, accepted model instrument ID or rejection reason.

Keep these records distinct from the J3 `prior_set` records (section 3.5). A relevance or matching judgment must never implicitly create a numerical correlation prior. Retain as-of rule and exposure versions to prevent hindsight leakage.

### 3.1 `event_registry`

Required fields: `event_id`, `venue`, `venue_market_id`, `title`, `rules_text`, `rules_hash`, `outcome_definition`, `event_family_id`, `episode_id`, `listed_at_utc`, `deadline_utc`, `resolution_source`, `status`, `rules_available_at_utc`.

Optional/audit fields: `resolved_at_utc`, `realized_outcome`, `parent_event_id`, `related_contract_ids`, `logical_constraints`, `rule_change_history`.

The realized outcome is never a fitting feature. Rules changed after an endpoint cannot be treated as known before it. Nested, identical, or complementary markets should be deduplicated or explicitly represented; Gaussian forecasts of event changes are not guaranteed to enforce logical bounds.

### 3.2 `event_quotes`

Required: `event_id`, `quote_time_utc`, `bid`, `ask`, `source`, `available_at_utc`, `retrieved_at_utc`. Optional: `trade_price`, `volume`, `depth`, `quote_method`, `is_historical_reconstruction`.

Require `0 <= bid <= ask <= 1`. Keep spread and age; do not convert unknown bid/ask quality into zero spread. If only last trades are available, set `quote_method=last_trade`, flag lower evidence quality, and run a separately labeled experiment rather than silently mixing with midquotes.

### 3.3 `asset_prices`

Required: `asset_id`, `price_time_utc`, `total_return_index`, `source`, `adjustment_method`, `available_at_utc`, `retrieved_at_utc`, `currency`. Keep corporate-action/revision provenance. Do not use future stock composition or dividends as predictive features. Vendor adjustments must be checked to produce the intended interval total return.

### 3.4 `design_rows`

Required: `row_id`, `start_utc`, `end_utc`, six ordered values of z, three `p_start` values, three `p_end` values, endpoint spreads/ages, event IDs and rules hashes, episode IDs, `elapsed_hours`, `quality_flags`, `eligibility_reason`, `source_snapshot_hash`.

No implicit forward-fill. Use complete valid rows in V1 for the raw models (M0, M1, M2). The cleaned models (M1c, M2c, M3) may additionally mask individual entries named by J2 treatments, either in both fit steps (`mask`) or in the loading step only (`mask_loading`); a masked entry is missing, never zero, and the row stays in the fit through its observed entries (section 4.2). Store masks as a separate `row_mask` record keyed by `row_id`, `variable_id`, `step` (`loading|both`) and the `interval_label` that caused it; never overwrite raw values. If an entire variable has effectively zero training variance, fail that universe/fold rather than inventing a standard deviation. Set an explicit numerical floor in configuration and report which variable failed it.

Exclude intervals containing a contract's formal resolution or settlement and subsequent intervals, using status/timestamp records rather than the sign of the realized stock return. A touch-threshold contract that has already become certain is not a source of ongoing probability-change variation. Record the resulting scope/selection limitation; do not use knowledge of the eventual outcome to remove otherwise eligible earlier rows.

### 3.5 IC records

Every LLM output is a versioned record. **Common metadata on all IC records:** `ic_record_id`, `ic_job` (J1-J5), `version`, `created_at_utc`, `information_cutoff_utc`, `model_provider_version`, `prompt_hash`, `evidence_snapshot_hash`, `historical_hindsight_risk`, `approval_status`, `reviewer`. Records are immutable; a revision is a new version that affects only later fits under the predeclared schedule.

**J1 `universe_proposal`:** portfolio/request ID, exposure snapshot ID, proposed event candidates (each becomes an `event_candidate`, section 3.0) with channel rationale and evidence, proposed **context variables** (observed series such as SPY, a sector ETF or a futures return that carry outside drivers into the model) with rationale, and deliberate exclusions with reasons.

**J2 `interval_label`:** `row_id`, `variable_id`, `measurement_label` (J2a: `INVALID_OBSERVATION|RULE_CHANGE|MICROSTRUCTURE|CLEAN`), `invalid_documentation` (required for `INVALID_OBSERVATION`: venue or vendor notice, or the failed code quality check), `attribution_label` (J2b, section 4.3 taxonomy), optional `attribution_subtype` (event-family specific, e.g. `supply|demand`; used only by the later news-dependent experiment), `confidence_bucket` (`low|medium|high`), `evidence_refs` (each with its source publication timestamp), `news_time_utc` (copied from a cited source; null if none), `blind` (must be `true` for scored experiments: see the blinding rule), `abstained` (bool).

**J3 `prior_set`:** each prior has `quantity` (`correlation|communality|response_sign`), `left_variable`, `right_variable` (null for communality), `bucket` (from the fixed menus in section 4.3), the code-mapped `interval_lower`/`interval_upper`, `weight_level`, `rationale`, `evidence_refs`, `known_counterarguments`, `applicability`, `expires_at_utc`. Constraints: bounds within the quantity's range, lower <= upper, nonnegative finite weight; no self pairs for pairwise quantities; canonical sorted pair IDs; duplicate conflicts surfaced. The agent may abstain. Empty prior sets are valid. An omitted quantity is unconstrained, not zero.

**J4 `type2_embedding`:** the normalized Type II proposition (section 0.2 fields), `analogs` (1-3 entries, each `variable_id`, `direction` `+1|-1`, `scale_bucket`), `communality_bucket`, rationale, evidence, and the source of any probability level used for support checks (`market|base_rate|user`; never the LLM).

**J5 `factor_proposal`:** `proposal_type` (`add_factor|remove_factor|label_factor`), the model snapshot it refers to, the code-computed evidence it read (section 12), rationale, and for labels the proposed `label_text` and `label_confidence_bucket`.

Example J3 prior, format only, not an economic assertion:

```json
{
  "quantity": "correlation",
  "left_variable": "dp_event_A",
  "right_variable": "r_MSFT",
  "bucket": "weak",
  "interval_lower": -0.15,
  "interval_upper": 0.15,
  "weight_level": "standard",
  "rationale": "Illustrative broad weak-correlation prior; requires evidence before use.",
  "evidence_refs": [],
  "known_counterarguments": ["Shared growth or risk news can induce comovement."],
  "applicability": "toy_fixture_only"
}
```

### 3.6 `model_snapshot`

Persist `model_id`, model family, ordered variable IDs, training window and row hash, IC ID, mu/s, k, L, psi, standardized covariance C, original covariance Sigma, chosen hyperparameters, optimizer diagnostics, seed, software versions, refit schedule, input assumptions, conditional-slope support statistics, maximum endpoint data availability, and snapshot creation time. Arrays can be saved separately with hashes referenced in metadata.

### 3.7 `scenario_result` and `evaluation_record`

Result fields: `model_id`, conditioning event ID, x, p_start, `query_type`, `reference_change`, `response_variable_order`, `response_mean`, `response_covariance`, portfolio h, `contract_positions`, `exposure_vector_g`, P&L mean/sd/intervals, `increment_vs_no_move`, support flags, `uncertainty_scope`, `event_probability_forecasts_advisory_only`.

`contract_positions` records `event_id`, venue/instrument ID, YES/NO side, signed contract count, dollar payout per contract, currency, midpoint source, and complementarity assumption if used. Store both raw positions and the mapped effective YES exposures. Do not infer contract counts from invested dollars without the actual entry/current price and the intended position convention.

Evaluation adds: interval IDs, pre-interval model freeze time, conditioning-observation time, target observation time, realized r/P&L, per-model losses, eligibility, and explicit `mode=conditional_reconstruction`.

## 4. Statistical model and IC objective

### 4.1 Joint latent-factor model

```text
u[t] = L F[t] + epsilon[t]
F[t] ~ Normal(0, I_k)
epsilon[t] ~ Normal(0, diag(psi))
F[t] independent of epsilon[t]
C = L L^T + diag(psi)
Sigma = D C D
```

Assume a locally stable marginal distribution inside each training window. Serial independence is a working fitting approximation, not a fact to assert; evaluation uses block/episode dependence corrections. Residual cross-covariances are zero in V1. A full unrestricted residual covariance would undermine the compression being tested.

Require every `psi[j] >= psi_floor > 0`. Standardization does not force the fitted diagonal of C to equal 1. When forming correlations always normalize by the **fitted** diagonal:

```text
rho[a,b] = C[a,b] / sqrt(C[a,a] * C[b,b])
```

The graph has k latent nodes and d observation nodes with loadings as edges. Edges are descriptive model dependencies, not causal arrows. Two observed nodes can correlate through common factors. Correlations are not multiplied along stock-stock paths.

### 4.2 Objective

Let `o[t]` be the set of unmasked variables in row t (all d variables for raw models) and `w[t] >= 0` a row weight (1 unless a J2 treatment sets otherwise). Fit L and psi by minimizing:

```text
NLL_t(C) = 0.5 * (logdet(C[o_t,o_t]) + u[t,o_t]^T solve(C[o_t,o_t], u[t,o_t]))
J(L,psi) = sum_t w[t] * NLL_t(C) / sum_t w[t] + lambda_ic * P(C)
P(C) = sum_q w_q * dist(q(C), [lo_q, hi_q])^2 / sum_q w_q
dist(v,[lo,hi]) = max(lo-v, 0, v-hi)
```

This is the **loading step**: masks named `mask` or `mask_loading` and loading-step weights apply. With no masks and unit weights it reduces exactly to v1.2: `0.5 * (logdet(C) + trace(solve(C,S)))` with `S = (1/n) sum_t u[t] u[t]^T`. That reduction is a required test. Training moments mu and s are computed from unmasked entries only. M1c (cleaned full covariance) uses the Gaussian EM covariance estimate under the same masks and weights, then the same shrinkage rule as M1.

**Variance step (cleaned models with `mask_loading` entries; an experimental estimator).** Cleaning should change how variables move together, not hide real risk. After the loading step, convert to original units, `Sigma_clean = D_clean C D_clean`, and let `v_raw[j]` be variable j's variance on the same training rows with only `mask` (both-step) entries removed, unweighted. Then:

```text
Sigma[a,b] = Sigma_clean[a,b]                              (a != b)
Sigma[j,j] = max(Sigma_clean[j,j], v_raw[j])
```

Cross-covariances, and so the event-to-stock mapping, come from cleaned data; each variable's own variance is topped up to its raw level, so company-specific stock risk and held-contract mark-to-market noise stay in every P&L range. Adding a nonnegative diagonal keeps Sigma positive definite. For the conditioning event, the top-up means the slope `Sigma[r,A]/Sigma[A,A]` is attenuated by A's measurement noise. That is correct for this estimand: the model conditions on the *quoted* change, which includes that noise at test time. For a factor model the top-up is equivalent to raising psi only.

**Why it is experimental.** The top-up keeps Sigma valid and restores each variable's own variance, but it does not guarantee correct *portfolio* risk. News removed from the loading step can be shared: an MSFT earnings day can also move peers, and that shared part lives in the off-diagonals the top-up never restores. So the construction is tested on its own:

- every cleaned model is also fitted without the top-up (`*_notopup`, cleaned variances throughout) and scored on the same rows;
- a shared-risk check compares training-window portfolio variance `h^T Sigma h` with the raw sample value `h^T S_raw h`, and stock-stock off-diagonals cleaned vs raw; a shortfall above `shared_risk_tolerance` is flagged `shared_risk_understated`;
- test-set coverage is checked on company-specific and held-contract thin-market days (section 8).

**No-treatment path.** Raw models (M0, M1, M2) and treatment `T0` never enter the cleaning pipeline: they return the raw model object itself (identical snapshot hash). The equivalence is enforced by construction, not by assuming the top-up reduces to the identity, which is not guaranteed for regularized or constrained fits.

Each penalized quantity q(C) must be **rotation-invariant** (loadings are not identified, section 12):

- `correlation`: `rho[a,b] = C[a,b] / sqrt(C[a,a] C[b,b])`.
- `communality`: `h[j] = 1 - psi[j] / C[j,j]`, the share of variable j's fitted variance carried by the common factors. A low communality says a variable's daily moves are mostly **its own**, not explained by the shared factors. That can be noise or genuine idiosyncratic content; it is not by itself a statement that the data are untrustworthy.
- `response_sign`: the sign of the conditional slope `C[r,a] / C[a,a]`, implemented as an interval on `rho[a,r]` of `[0, 1]` or `[-1, 0]`.

If total positive prior weight is zero, define P=0. The omitted Gaussian constant does not affect optimization. This is **average** negative log likelihood plus a fixed-strength penalty. Call it prior-like regularization or generalized MAP-style fitting, not a posterior with calibrated Bayesian uncertainty. Averaging means the effective penalty does not automatically weaken as n grows; select lambda on chronological validation data. Never interchange sum and average likelihood without adjusting lambda.

Use `psi = psi_floor + softplus(theta)` or a bounded optimizer. Use Cholesky-based log determinants and solves, never explicit inverse formation. Use deterministic multiple starts and preserve convergence/failure diagnostics. A failed fit returns a flagged failure, not a silently substituted successful model. Test optimization gradients or finite differences where appropriate.

At lambda=0, the IC and no-IC paths must reduce to the same objective and agree numerically within optimizer tolerances. The latter is the matched ablation.

### 4.3 Intelligence constraints (IC): the LLM's five jobs

**Why IC exists.** The empirical covariance is the starting evidence, not a decomposition of causes. For small moves it mixes shared responses to information, which the model should learn; measurement error (thin books, stale quotes, contract mechanics), which it should not; real company-specific news, which is irrelevant to the events but is genuine portfolio risk; and drivers from outside the observed universe. Market efficiency does not separate these, and the same small move can have more than one interpretation (oil-record odds can rise on a supply disruption or on strong demand). The prediction-market side is much noisier than the stock side. IC treats the LLM's evidence-backed judgment as real information that is absent from the price panel, applied to telling these apart, rather than as a black box to be kept away from the data. Code then tests whether combining the two produces better conditional responses.

#### Rules for every job

1. **The LLM selects, labels and reasons; code computes.** LLM outputs are categorical: selections, labels, buckets, analog choices, names. Code maps every category to a number through fixed tables in configuration. News timestamps are copied from cited sources, never generated.
2. **Evidence-bound.** Every non-abstaining output cites evidence published before its information cutoff. Uncited outputs are treated as abstentions (`UNKNOWN`).
3. **Pre-fit only.** IC changes what enters the fit (J1, J2a, and J2b in the loading step only), how the fit is penalized (J3), or how a new variable is appended (J4). It never edits fitted parameters or outputs after fitting.
4. **One ablation per job.** Each job is tested against the same model with that job switched off, holding mappings, rows and tuning budgets fixed, so one job's gain is never credited to another.
5. **Hindsight.** A present-day LLM may know what happened after a historical date. Historical outputs carry `historical_hindsight_risk`; the clean test freezes outputs prospectively (section 8).
6. **Confidence is not probability.** Confidence buckets may only be used as predeclared treatment conditions (e.g. "treat low-confidence `RULE_CHANGE` as `CLEAN`").

| Party | Responsibility |
|---|---|
| Code | As-of alignment, flagging, applying treatments, masks and weights, standardization, covariance and factor fitting, conditional inference, factor-score and proxy diagnostics, scoring. |
| LLM | J1-J5 below. |
| Human / configuration | Approve the seed universe, label taxonomy, treatment grid, bucket tables, weight levels, proxy library, validation budgets and success criteria. |

#### J1. Selecting the event universe for a portfolio

- **Input:** the exposure snapshot (look-through weights, sectors, countries, supply-chain notes), the event registry as of the cutoff, and budgets `max_events` and `max_context_variables`.
- **Output (`universe_proposal`):** ranked event candidates, each with its channel to the holdings and evidence; **context variables** (observed series such as SPY, a sector ETF or an oil future) that carry outside drivers into the model so they are not misattributed to the events; and deliberate exclusions with reasons. A qualitative expected sign may be recorded but is not used numerically unless separately entered through J3.
- **Code:** runs matching verification (section 0.2), eligibility and the data audit; enforces the budgets, since complete-case overlap shrinks as d grows. Context variables enter z as observed variables. Headline queries integrate them out rather than conditioning on them, and they add to P&L only if held.
- **Guardrails:** never select events because they later produced large losses. Freeze the universe with its cutoff and hold it fixed across stages 2-5.
- **Evaluation:** separately from the numerical engine (section 9.1): judged recall against human review, false inclusions and exposure coverage.

#### J2. Measurement cleaning and news attribution (primary)

Daily prediction-market changes carry most of their information on a few days and noise on the rest. For each flagged observation, J2 **proposes an evidence-supported attribution** of what moved that variable. It does not establish causes. The attribution is used for two different purposes, kept apart because they act on different parts of the model:

- **J2a, measurement quality.** Is the observed number valid, and if so, is it the kind of price change the model should learn from? Only a **documented invalid observation** (a data error, an outage print, a vendor glitch, a quote failing a code quality check) is measurement error and is excluded unconditionally. Rule-change repricing and thin-market moves are real, tradable prices that may carry little information about the proposition; "no identifiable news" does not establish measurement error. Excluding them from the loading step is a **hypothesis**, chosen or rejected on validation.
- **J2b, news attribution.** Given a faithful measurement, what kind of news moved it? Event-specific, broad, sector or company-specific news is *real variation*. It is labelled, never deleted from the risk model. Company-specific stock news is excluded only when estimating how events map to stocks (the loadings), and is kept in each stock's own variance (two-step fit, section 4.2).

The distinction matters most for company-specific news: an MSFT earnings day is irrelevant to the events but is genuine portfolio risk, and removing it from the variance would make every P&L range too narrow.

**Flagging (code).** Event entries with `|dp| >= flag_abs_event_change` or `|u| >= flag_abs_standardized`; stock entries with `|u| >= flag_abs_standardized` or a scheduled company event (e.g. earnings) inside the interval. Unflagged entries are not labelled and are kept.

**Calendar first (code, before any LLM call).** Scheduled releases come from published calendars, not the LLM: FOMC decisions and minutes (Federal Reserve calendar), CPI and the Employment Situation (BLS release schedule), GDP and PCE (BEA), and company earnings dates (existing `massive.earnings`). Code tags every interval containing a release, then labels flagged entries on those intervals by a fixed, configured mapping:

- an event entry on the release that resolves or directly informs its own proposition (the FOMC decision for a Fed contract, the CPI print for a CPI contract, the jobs report for a payrolls contract) → `OWN_NEWS`;
- an event entry on any other scheduled macro release (CPI for a Fed contract, payrolls for a recession contract) → `COMMON_NEWS`, with its `calendar_tag` recorded;
- a stock entry on its own earnings interval → `IDIOSYNCRATIC`.

Calendar labels carry `source=calendar` and need no LLM call. The LLM labels only flagged entries the calendar does not explain, which are the genuinely ambiguous off-calendar jumps. Two uses of the calendar beyond cost: a **labeller accuracy check** (the LLM also labels, blind, a predeclared random sample of calendar-explained entries; its agreement with the calendar is reported in the J2 audit, never used to tune it), and the **news-day treatment** T5 (below). The mapping table is configuration, frozen before fitting.

Each flagged entry the calendar does not explain gets **its own request**, containing only that variable's permitted information; one request returns both its J2a and J2b labels. Requests may be submitted together operationally (concurrently or through a batch API), but no request ever contains another variable's same-interval move.

**Labeller input.** The variable's definition (contract rules or company), interval start and end, the size and direction of **that variable's own** move, timestamped news from existing adapters, the contract's rule-change history, and quote-quality statistics (spread, age, volume, trade count).

**Blinding rule.** The labeller must not see same-interval values of the other panel variables or any fitted model output. A labeller that sees them can, even unintentionally, discard days that contradict the correlation it expects and so manufacture the dependence the fit then "finds". Attributions describe a variable's own move, from evidence alone.

**Label taxonomy.**

| Part | Variable type | Label | Meaning | Treatment |
|---|---|---|---|---|
| J2a | Either | `INVALID_OBSERVATION` | Documented invalid value: data error, outage print, vendor glitch, quote failing a code quality check | mask in both steps, unconditionally (requires documentation) |
| J2a | Event change | `RULE_CHANGE` | Rule clarification, roll, listing, ladder restructure, settlement mechanics: real repricing, not news about the proposition | hypothesis; grid: loading step only |
| J2a | Event change | `MICROSTRUCTURE` | No identifiable news, plus quote-quality evidence of thin trading (wide spread, thin volume, single large print, stale catch-up) | hypothesis; grid: loading step only |
| J2a | Either | `CLEAN` | No measurement problem found | keep |
| J2b | Event change | `OWN_NEWS` | News about the event's own proposition (e.g. an OPEC announcement for an oil-record market) | keep; identified day; grid: up-weight |
| J2b | Event change | `COMMON_NEWS` | Broad macro or risk news moving many markets | keep: the comovement the model should learn |
| J2b | Stock return | `IDIOSYNCRATIC` | Company-specific news: earnings, guidance, M&A, litigation, management, product | grid: mask in the loading step only |
| J2b | Stock return | `SECTOR` / `MACRO` | Industry or broad news | keep |
| J2b | Either | `UNKNOWN` | Insufficient evidence, or abstention | keep |

**Treatments (code, from the configuration grid; the LLM never chooses a treatment).** `keep`; `mask` (missing in both steps: reserved for documented invalid observations); `mask_loading` (missing in the loading step only; the entry still counts in the variance step); `weight` (a fixed row weight in the loading step, used to up-weight identified rows). The treatment is chosen on validation from the predeclared grid (section 6), with ties going to the least cleaning.

**Mark-to-market.** Rule-change repricing and thin-market moves are real prices, and for a contract the portfolio **holds** they are real P&L. They therefore use `mask_loading`, never `mask`, so the variance step keeps them. Only `INVALID_OBSERVATION`, with its documentation, is masked in both steps.

**High-frequency identification.**

- *Level 1, daily (always available).* `OWN_NEWS` rows are the days where the event's own news is the main story. Code reports (i) the response slope estimated on `OWN_NEWS` rows only, beside the full-sample slope, as a diagnostic, and (ii) the grid option to up-weight these rows by a fixed level. Identification still rests on assumptions: other news on the same day, and information the event's news carries about other variables, can both bias these slopes.
- *Level 2, intraday (only if intraday quotes and prices pass a separate alignment audit).* For `OWN_NEWS` labels with a sourced `news_time_utc`, code builds a narrow window around the news time from intraday event quotes and stock prices, using the same at-or-before rules as section 2, during regular stock trading hours only. Narrow windows reduce, but do not remove, contamination by competing news (event-window studies in the style of Kuttner and Gürkaynak-Sack-Swanson face the same issue). In V1 they give a separate code-computed slope reported beside the daily model; their disagreement with the daily slope is a diagnostic. Any use as a prior goes through J3 with its own ablation.

**Guardrails.**

- **Only training rows are cleaned.** Test rows are always scored raw and complete. Labels for test rows may be produced afterwards, for stratified reporting only. Cleaning must not be able to win by deleting hard test days.
- Evidence must be published before the fit that uses the row. Going forward, label each new interval within one trading day of its end and freeze it.
- Audit labels: re-label a sample with a second run and report agreement; human-review all `INVALID_OBSERVATION` and `RULE_CHANGE` labels plus a stratified random sample, and report precision per label.
- No relabelling after seeing model results. A new taxonomy, flag rule or treatment grid is a new experiment version.
- Report the share of training entries masked per variable and per step; flag any variable losing more than `max_masked_share` of its entries in the loading step.

**Evaluation (stage 3).** The predeclared primary comparison is M2c (J2a + J2b + variance step) against matched-k M2, on identical raw test rows, with paired score differences and block uncertainty, stratified by post-hoc test-row label. Stages 3a (M2 → M2a, J2a alone) and 3b (M2a → M2c) explain where any gain comes from, together with the `*_notopup` variance-step test and the M1c-vs-M1 replication. With few independent episodes, every extra comparison adds multiple-testing risk, so only the primary result is a headline.

**Later experiment, gated on power: news-dependent responses.** The same move in A can mean different things: oil-record odds can rise on a supply disruption or on strong demand, and the two can imply different equity responses. J2b labels make this testable: estimate the response to A separately by attribution type (and, for event families where it matters, by a finer event-specific sub-type such as supply vs demand) and test whether it improves conditional scores. Splitting the sample divides already thin data, so this runs only once the stage 0 power check shows it can detect a difference.

#### J3. Injecting ranges and priors

- **Input:** variable definitions, J1 channel rationales, and the J2 label summary per variable (e.g. "event C: 40% of flagged moves were `MICROSTRUCTURE`"). **J3 never sees fitted or sample correlations** from the training window, so priors cannot echo the data back.
- **Output (`prior_set`):** a sparse set of buckets on rotation-invariant quantities (section 4.2), chosen from fixed menus:

| Quantity | Bucket | Interval |
|---|---|---|
| correlation | `strong_negative` | [-1, -0.5] |
| correlation | `moderate_negative` | [-0.5, -0.15] |
| correlation | `weak` | [-0.15, 0.15] |
| correlation | `moderate_positive` | [0.15, 0.5] |
| correlation | `strong_positive` | [0.5, 1] |
| response_sign | `positive` / `negative` | [0, 1] / [-1, 0] on the correlation |
| communality | `noise_dominated` | [0, 0.2] |
| communality | `mixed` | [0.2, 0.5] |
| communality | `signal_dominated` | [0.5, 1] |

  Weight levels are `light` 0.5, `standard` 1, `strong` 2, relative within P; lambda is chosen on validation. Bucket bounds are starting settings, frozen before test inspection.
- **Communality priors say how much of a series should be shared.** A `noise_dominated` (low-communality) prior on an event, typically supported by J2a evidence of thin quotes, lets the fit assign most of its daily variance to its own residual instead of forcing it onto shared factors. A low communality means "mostly its own moves", which can be real; it does not mean the series is discarded.
- **Guardrails:** no direct effect does not mean zero correlation (common causes create total comovement); conflicts are reported, not resolved by editing the matrix; freeze priors prospectively.
- **Evaluation (stage 4):** M3 vs matched-k M2c, so priors are tested on top of cleaning.

#### J4. Embedding Type II (unlisted) events

- **Input:** the normalized Type II proposition (section 0.2 fields), the fitted model's variable list with definitions (not its loadings), and evidence.
- **Output (`type2_embedding`):** 1-3 analogs from the fitted universe, each with a direction (+1 or -1) and a `scale_bucket` in {0.25, 0.5, 1, 2}, plus a `communality_bucket`.
- **Code:** builds the new standardized loading row as `l_star = sum_i direction_i * scale_i * L[analog_i, :]`, sets `psi_star` so that the communality equals the bucket midpoint (`psi_star = |l_star|^2 (1 - h_star) / h_star`), and sets the raw scale `s_star` to the primary analog's s. This is the section 13 extension with `b_star = s_star * l_star` and `tau_star^2 = s_star^2 * psi_star`. Because the analogs' rows come from the same fitted L, the construction is rotation-consistent: no factor needs a name. If `l_star` is numerically zero (analogs cancel), the embedding is rejected.
- **Probability level:** any level q used for support checks comes from a market, a code-computed base rate or the user, **never the LLM**.
- **Evaluation (stage 5):** leave one listed event out (section 13), against a **closest-analog baseline**: borrow the single nearest analog's responses unchanged. A single analog at scale 1 with its own communality reproduces that analog exactly, so this baseline is J4 with its scale, multi-analog and communality choices switched off; the comparison tests whether those choices add anything.

#### J5. Adding or removing factors, and naming solved factors

- **Add or remove.** The LLM may propose k+1 or k-1 with a reason (regime change, a new event family, a dormant family). A proposal only triggers the section 12 refit test; it never changes k by itself. Data-driven triggers remain.
- **Naming.** After each refit, code picks a display rotation (varimax, then a sign convention) and computes, for each rotated factor: its top-loading variables; the correlation of its estimated factor scores over the training window with a fixed **proxy library** (e.g. oil futures, 10-year yield change, SPY, the dollar index, VIX, sector ETFs); and the dates of its largest scores with their J2 labels. The LLM proposes a name **only from these code-computed facts**, with a confidence bucket, or abstains (`unnamed`). A factor whose best proxy has `|corr| < factor_label_min_proxy_abs_corr` stays `unnamed`.
- **Technical-factor check.** Factors organize shared variation; they do not by themselves separate economic signal from shared technical noise. A venue-wide liquidity or outage shock moves many contracts together and can appear as a factor. Code flags a factor as `possible_technical` when its loadings are concentrated on event series with little stock loading, no proxy clears the naming threshold, and its largest-score days are mostly J2a `MICROSTRUCTURE`, `RULE_CHANGE` or `INVALID_OBSERVATION`. A flagged factor is reported and its stability tracked; removing it goes through the section 12 refit test like any other change.
- **Rules for names:** display-only and versioned per snapshot. A name carries across a refit only if the principal angle to the previously named factor is at most `factor_label_max_principal_angle_deg` and its proxy correlation still clears the threshold; otherwise the factor is renamed or becomes `unnamed`. A name never enters fitting, priors or conditioning. Making names part of the model is an anchored model (section 12), tested separately.

#### Call budget

| Job | Calls |
|---|---|
| J1 | One per portfolio universe build, plus an optional critique call. |
| J2 | One per flagged entry the calendar does not explain (one variable, one interval) over the training history, then one per new such entry, prospectively, plus the fixed calendar accuracy sample. Requests may be submitted together operationally but never share content (blinding rule). |
| J3 | One per universe build. |
| J4 | One per Type II request. |
| J5 | One per refit for names; proposals as needed. |

No call is made per fitting iteration, per pair, or per scenario query. A new IC output creates a new version and affects only subsequent fits under the predeclared schedule.

## 5. Conditional inference and P&L

For conditioning variable A, observed change x, stock block r:

```text
m_r(x) = mu_r + Sigma[r,A] / Sigma[A,A] * (x - mu_A)
V_r = Sigma[r,r] - outer(Sigma[r,A], Sigma[A,r]) / Sigma[A,A]
mean_pnl = h^T m_r(x)
var_pnl = h^T V_r h
increment_vs_no_move = h^T (m_r(x) - m_r(0))
                      = h^T Sigma[r,A] / Sigma[A,A] * x
```

For simultaneous conditioning on event subset J:

```text
m_r(x_J) = mu_r + Sigma[r,J] solve(Sigma[J,J], x_J - mu_J)
V_r = Sigma[r,r] - Sigma[r,J] solve(Sigma[J,J], Sigma[J,r])
```

V1 headline evaluation conditions on one event at a time. Hold all other observations unobserved; integrate them out. Do not condition on B/C at zero unless the user explicitly requests it. Do not predict B/C, feed those predictions back in, and add a second impact. Three event-conditioned evaluations on one day share the same realized returns and are not independent experiments.

`m_r(x)` is the conditional total-return mean. The increment versus x=0 is a model comparison; its causal counterfactual is unobserved and must not be scored as if it were the realized full return. Score `m_r(x)` against r and the portfolio conditional mean against `h^T r`.

No additional priced-in multiplier is applied. The empirical response already relates changes in quoted probabilities to returns around the prevailing state. Multiplying it again by `1-p` would add an unjustified adjustment. V1 does not identify equity-implied event probabilities or terminal scenario values.

Return 50%, 80%, and 95% central Gaussian prediction intervals. Label them `process_only_plugin`. They exclude parameter/IC/measurement uncertainty. The Gaussian V_r is independent of x; it is not a tail-stress uncertainty model. Parameter uncertainty can later be estimated with training-block resampling and full refitting, explicitly labeled separately.

### Support checks

- Reject impossible `p_start+x` outside [0,1]. Reject 0/1 destination resolution requests in V1.
- Require chosen event start probability inside configured interior bounds.
- Reject/flag x outside the predeclared absolute bump cap or training-support gate. Do not clip x into eligibility.
- Report extrapolation and counts of comparable training moves. Do not fabricate high confidence when there are few matches.
- V1's Gaussian distribution has unbounded tails and does not preserve every logical event constraint. Event-change predictions are diagnostics only. Stock/P&L estimates are locally approximate; no claim of a coherent binary-outcome model.

The user provides a finite bump and a time interval; no derivative is interpreted as the causal consequence of all possible sources of that bump.

### 5.1 Prediction-market holdings and mixed portfolios

The existing vector z already contains event-price changes. No new estimator or LLM call is required to obtain event-to-event conditional responses:

```text
E[dp_B | dp_A=x] = mu_B + Sigma[B,A]/Sigma[A,A] * (x-mu_A)
```

To include all positions, condition the full vector, not only its stock block:

```text
m_z(x) = mu + Sigma[:,A]/Sigma[A,A] * (x-mu_A)
V_z = Sigma - outer(Sigma[:,A], Sigma[A,:])/Sigma[A,A]
m_z[A] = x
V_z[A,:] = V_z[:,A] = 0  # mathematically exact under the conditioning scenario
```

For $1 YES contracts, n[j] is signed contract count. For payout c[j] dollars, define exposure e[j]=c[j]*n[j]. Under exact YES/NO complementarity, use e[j]=c[j]*(n_yes[j]-n_no[j]). The price change of a NO contract is then -c[j]*dp[j]. If venue NO quotes are not exact complements, map them to their own observed price series and refit, or reject that position as unsupported; do not silently assert complementarity.

```text
g = [e_A, e_B, e_C, h_XOM, h_DAL, h_MSFT]
pnl = g^T z
conditional_mean_pnl = g^T m_z(x)
conditional_var_pnl = g^T V_z g
increment_vs_no_move = g^T (m_z(x)-m_z(0))
```

Use contract-count/payout exposure for event changes and start-of-interval dollar exposure for stock returns. Do not multiply contract count by its current price again. Conditional on A's exact midpoint-price bump, the held A position's P&L is deterministic; B/C and stocks remain uncertain. Event-stock cross-covariances must be included: do not sum separately computed event and stock variances as if independent. The full V_z is positive semidefinite and singular because conditioned coordinates have zero variance; require positive definiteness only on genuinely unconditioned nonsingular blocks, not on V_z itself.

Event responses are computed jointly. Do not propagate A to B to C recursively or feed predicted B/C into another conditioning pass. A held contract must map to an observed variable (or a documented exact payoff mapping); otherwise it is not supported by the Type I model.

If conditioning fully determines all held positions, variance can be zero. Return a deterministic P&L and degenerate interval rather than adding fictitious variance. A Gaussian log score requires positive variance; mark deterministic queries non-distributional and assess arithmetic consistency separately.

Outputs are midpoint mark-to-market estimates. Fees, spreads, slippage, collateral economics, currency conversion and executable liquidation values are not modeled. Use a common currency or require an explicit FX extension.

**Probability and logical support:** expected remaining event levels are p_start[j]+m_z[j]. Flag levels outside [0,1] or violated known nesting/complementarity constraints, even when the conditioning bump itself is feasible. Do not clip individual predictions or renormalize them after conditioning: that changes the model without recomputing its covariance. If a material held-event prediction violates these constraints, return `UNSUPPORTED_EVENT_PORTFOLIO` for mixed-P&L publication, preserving raw diagnostic results. Even accepted means do not make Gaussian event tails bounded. `event_probability_forecasts_advisory_only` means locally approximate and not a coherent resolution model, not that event holdings cannot be valued.

**Backtest extension:** freeze actual contract counts and stock dollars at interval start; realized mixed P&L is g^T z from synchronized midpoint changes and total stock returns. Use the same full-vector conditioning and holdings for every comparator. Predeclare mixed portfolios as secondary targets, or revise the primary experiment before opening final-test results. Gains from a held A contract are already known once x is supplied; do not count that arithmetic as predictive skill. Report unconditioned holdings' P&L separately. The unconditional comparator for this mixed test must include the same known A-position gain and leave only other positions unconditional; otherwise the model gets an unfair advantage from the supplied x.

## 6. Initial experiment configuration

These are conservative **starting research settings**, not empirically certified thresholds. Adjust after a data-availability audit, before opening final test outcomes. Log all changes.

```yaml
spec_version: '1.7'
mode: conditional_reconstruction
variables: [dp_event_A, dp_event_B, dp_event_C, r_XOM, r_DAL, r_MSFT]
calendar: XNYS
cutoff_timezone: America/New_York
cutoff_local_time: '16:00:00'
elapsed_hours_min: 23
elapsed_hours_max: 25
quote_method: midpoint
max_quote_age_seconds: 900
max_bid_ask_spread: 0.04
min_start_probability: 0.05
max_start_probability: 0.95
max_abs_conditioning_change: 0.05
max_abs_training_standardized_conditioning_change: 3.0
scale_floor: 1.0e-8
psi_floor: 1.0e-4
factor_k_candidates: auto            # 0..k_max per universe, chosen on validation (see below)
factor_k_max_rows_per_param: 5      # k_max also limited so training rows >= 5 x free parameters
lambda_ic_candidates: [0.0, 0.01, 0.1, 1.0]
ic_pair_weights: equal_nonzero
training_window_rows: 120
initial_validation_rows: 40
final_test_rows_min: 40
refit_every_eligible_rows: 20
optimizer_starts: 5
seed: 1729
primary_conditioning_event: event_B
primary_portfolio_usd: {XOM: 10000, DAL: 10000, MSFT: 10000}
primary_metric: gaussian_negative_log_score_portfolio
primary_comparison: M2c_vs_M2_matched_k      # predeclared; all other stages explanatory
secondary_metrics: [mae, rmse, coverage_50, coverage_80, coverage_95, width_80, interval_score_80]

# IC (section 4.3)
ic_j1:
  max_events: 3
  max_context_variables: 0        # 0 keeps the six-variable primary universe; a J1 context variable (e.g. SPY) is a predeclared variant, frozen before test inspection
ic_j2:
  flag_abs_event_change: 0.03
  flag_abs_standardized: 2.5
  flag_scheduled_company_events: true
  blind_labelling: true
  label_lag_trading_days: 1        # prospective labels frozen within one day of interval end
  max_masked_share: 0.30
  variance_step: raw_topup         # section 4.2; cleaned models only
  treatment_grid:                  # chosen on validation; ties go to the least cleaning
    T0: {}                         # = raw (M1 / M2)
    T1: {INVALID_OBSERVATION: mask}                                     # J2a: documented invalid only (always on in cleaned models)
    T2: T1 + {RULE_CHANGE: mask_loading, MICROSTRUCTURE: mask_loading}   # J2a: exclusion hypotheses
    T3: J2a choice + {IDIOSYNCRATIC: mask_loading}                       # J2b
    T4: T3 + {OWN_NEWS: {weight: 2.0}}                                   # J2b
    T5: T4 + {COMMON_NEWS_SCHEDULED: {weight: 2.0}}                      # news-day concentration: all calendar-tagged macro release days up-weighted
  calendar_first: true             # code tags scheduled releases before any LLM call (section 4.3, J2)
  calendar_sources: [fomc, bls_cpi, bls_employment, bea_gdp, bea_pce, earnings]
  calendar_accuracy_sample: 50     # calendar-explained entries also labelled blind by the LLM, for the J2 audit
  shared_risk_tolerance: 0.10      # max relative shortfall of h'Sigma h vs h'S_raw h before flagging
  hf_level2_window_minutes: [-15, 45]   # used only if intraday data pass a separate audit
ic_j3:
  quantities: [correlation, response_sign, communality]
  weight_levels: {light: 0.5, standard: 1.0, strong: 2.0}
  max_priors: 8
ic_j4:
  scale_buckets: [0.25, 0.5, 1.0, 2.0]
  max_analogs: 3
ic_j5:
  display_rotation: varimax
  proxy_library: [SPY, TLT, USO, UUP, VIXY]
  factor_label_min_proxy_abs_corr: 0.5
  factor_label_max_principal_angle_deg: 20
  technical_factor_min_event_loading_share: 0.8   # share of squared loading on event series
  technical_factor_min_j2a_share_top_days: 0.5
```

**Power gate (stage 0).** Before fitting, simulate **the actual planned primary comparison**: the same estimators, refit schedule, score, block-bootstrap test and significance level, at the real row counts and with the audit's estimates of fat tails and serial dependence. Report the smallest effect it detects with reasonable power. If the plausible effect is below that, change the universe (e.g. longer-lived recurring contracts) rather than run an uninformative test.

A reference simulation ships with this spec (`power_sim.py`). Its assumptions are deliberately simple and stated in the file: a bivariate Gaussian (or Student-t, 3 degrees of freedom) relationship between the conditioning event and portfolio P&L, independent rows, a 120-row rolling window refit every 20 rows, sample-moment M1 against M0, mean paired Gaussian log-score difference, one-sided z-test at 1.96, 1,000 replications. Under those assumptions only, a true correlation of 0.10 was detected about 1% of the time even with 1,000 test rows, and 0.20 about 60% of the time. These figures illustrate why the gate exists; they are not a property of the planned test, which must be simulated directly.

Start-probability interior filters apply to all included events when building a core stable panel. The conditioning-change cap and standardized support gate apply to the chosen query event; do not screen on realized stock returns or on unobserved same-interval B/C changes. Training rows may include large non-resolution changes; do not silently winsorize. Record their influence and use separately specified robust models only as later experiments.

The minimum counts are feasibility gates, not proof of statistical power. If the seed contracts cannot supply 200 usable rows plus enough initial endpoints, report `INSUFFICIENT_DATA`; do not loosen all filters, shorten training arbitrarily, stitch different event definitions, or invent observations. A smaller prototype can run as a labeled mechanical demo with no validity claim.

**Choosing the number of factors.** For each universe, fit k = 0, 1, ..., k_max and keep the k with the best validation score (ties to the smaller k). k_max is the smaller of (a) the identification ceiling, the largest k with `(d - k)^2 >= d + k` (3 for d = 6, 14 for d = 20), and (b) the largest k whose free parameters `d*k + d - k*(k-1)/2` stay below `training_window_rows / factor_k_max_rows_per_param`. Report two diagnostics beside the choice, never overriding it: **parallel analysis** (keep factors whose eigenvalues exceed those of the same data with each column independently shuffled) and **BIC**. A large disagreement is flagged for review. Expect k to grow with how varied the universe is, not with d alone.

### 6.1 LLM call budget (hard caps)

Every LLM call in the experiment (J1 universe proposals, J2 labels, J3 priors, J4 embeddings, J5 names, audits) obeys all of the following. None of them is optional.

1. **Worsey's existing gate applies.** Each call goes through `worsey.agent.budget.gate()` before sending and `record()` after (per-hour, per-day and estimated-USD-per-day limits, the persistent ledger, the `cache/usage/STOP` kill switch and `WORSEY_LLM_DISABLED=1`). Experiment calls are tagged `source=pmrm` in the ledger.
2. **Per-run caps on top.** `WORSEY_PMRM_MAX_CALLS_PER_RUN` (default 300) and `WORSEY_PMRM_MAX_USD_PER_RUN` (default $5, estimated with the ledger's price table). A run that would exceed either stops cleanly; everything already labelled stays cached.
3. **Dry run first.** Every command that can call the LLM first prints the number of calls it would make, the model, and the estimated cost, and calls nothing unless run with `--confirm`. A run whose estimate exceeds the per-run cap refuses even with `--confirm`.
4. **Permanent label cache.** Each output is stored under `cache/pmrm/` keyed by (variable, interval, evidence-snapshot hash, prompt hash, model). A rerun with the same inputs makes zero calls. A prompt or taxonomy change is a new key, and so a new, deliberately approved run.
5. **No hidden loops.** No LLM call per fitting iteration, per scenario query, per universe, or per retry beyond one retry on a transport error. Study B reuses Study A's labels for every shared variable.
6. **Prospective labelling** (the daily frozen log) runs at most once per trading day, capped at `WORSEY_PMRM_MAX_CALLS_PER_DAY_PROSPECTIVE` (default 40).
7. **Sizes to expect** (estimates; the dry run is authoritative): Study A is about 6 variables × 250 days × ~10% flagged ≈ 150 J2 labels. A Study B pool of ~20 events and ~30 stocks is ≈ 1,250 labels over the same history. Each label is one call.

## 7. Walk-forward evaluation algorithm

1. Freeze the universe, units, quality rules, primary query/portfolio, metrics, candidate grid, J2 flag rules, label taxonomy and treatment grid, J3 bucket tables, and treatment of prior changes. Save a preregistration config/hash.
2. Build eligible nonoverlapping intervals. Produce the common-overlap report, counts by episode, quote age/spread, probability level, horizon to deadline, gaps, and source provenance. End-of-interval quote quality may determine availability but cannot depend on stock-return outcomes.
3. Partition chronologically: initial training block, following validation block, then a sealed final test block. Group related contracts/news episodes; if a boundary cuts an identified shared episode, move the boundary or exclude that episode according to a rule fixed before scoring. No random row split.
4. On validation, simulate the specified refit schedule. For each candidate use only preceding data; precompute the frozen conditional response function before each scored interval. Score conditional responses using observed x after the interval ends. Pick k, the J2 treatment and lambda using only validation scores, with tie-breaks toward lower complexity, less cleaning and lower lambda. Keep fixed-k ablations for each IC job as well as selected-model results. Validation and test rows are scored raw; only training rows are cleaned.
5. Seal settings. Refit on the latest eligible training window before the first test interval, using validation history now available. Never select settings from final-test outcomes.
6. Step through test intervals chronologically. If a scheduled refit is due, fit using only rows with `end_utc <= interval.start_utc` and available by then. Previous test rows may enter later rolling fits; this is prequential evaluation, not a frozen holdout, and must be labeled as such. They cannot trigger adaptive hyperparameter/model changes.
7. At interval start persist each model/function version. At interval end provide x, apply its support policy, save predictions, and then attach realized returns for scoring. Retrospective replay must distinguish simulated as-of timestamps from actual file creation timestamps.
8. Compare models on identical scored intervals. Log model-specific convergence failures and both common-success and coverage/availability results. Do not let an unreliable model look better by selectively dropping its failures.
9. Produce paired loss differences, block/episode uncertainty intervals, and stratified diagnostics. Freeze the report. Further tuning becomes a new experiment requiring fresh validation and/or prospective evidence.

One-event-at-a-time hypotheses A and C are secondary if B is primary. If exploring all three, account for shared dates and multiple comparisons rather than reporting the best without qualification.

### Comparator models

- `M0_unconditional`: training mean and return covariance, ignoring x. Optional zero-mean reference for immediate-return intuition.
- `V1_current`: Worsey's current engine (`docs/research/methodology_v1.md` v1.0), entered as a comparator on the same rows. Its prediction for portfolio P&L given the conditioning change x is mean `B_post * x`, where `B_post` is its empirical-Bayes book sensitivity to that event estimated with data available before the interval, and variance equal to the training-window variance of the residual `y - B_post * x`. Both engines are in dollars per unit of probability, so this is a direct head-to-head on small moves. It is explanatory, not part of the primary comparison.
- `M1_shrink_cov`: shrink the full standardized sample covariance toward `alpha I`, `alpha = trace(S)/d`; estimate shrinkage with Ledoit-Wolf or use a predeclared validation-tuned shrinkage grid. Transform back with the same D and use the same conditioning formulas.
- `M2_factor`: k-factor model, raw training data, lambda=0.
- `M2a_factor_j2a`: M2 with the chosen J2a treatment (T1 or T2), then the variance step if any `mask_loading` entries exist.
- `M2c_factor_clean`: M2 with the chosen J2a + J2b treatment (T3 or T4), then the variance step. **Primary comparison: M2c vs matched-k M2.**
- `M1c_shrink_cov_clean`: M1 with the same treatment as M2c (EM covariance under loading-step masks and weights, same shrinkage rule), then the variance step; a replication on the full-covariance benchmark.
- `*_notopup`: each cleaned model fitted without the variance step; tests the top-up construction itself.
- `M3_factor_clean_priors`: same k, settings and cleaned inputs as M2c, plus J3 penalties with lambda>0.
- `M2b_bayes_hier` (continuation C1, Study B only; section 16): the Bayesian hierarchical factor model. Not part of Study A or of any predeclared primary comparison until the trainer promotes it.

Sequence (section 0.1): M0 → M1 → M2 → M2a → M2c → M3, each step against the previous at matched k. M2c vs M2 is the single predeclared primary comparison; the steps around it explain where any gain comes from. Compare at matched k even if selected models choose different k.

The full-covariance benchmark retains EC directly; the factor model tests whether compression helps; J2 tests whether removing measurement error and attributing news helps the same estimator without hiding real risk; J3 tests whether priors help on top of that. Do not confound an IC job with an extra information feed, different scored rows, a larger tuning budget, or different preprocessing beyond the job itself. Context variables proposed by J1 are part of the frozen universe for every model, not an IC gain.

## 8. Metrics and promotion criteria

For realized portfolio P&L y, predicted mean m and variance v>0:

```text
negative_log_score = 0.5 * (log(2*pi*v) + (y-m)^2 / v)
```

Lower is better. This proper score is the primary distributional measure. Report MAE/RMSE, sign accuracy as descriptive only, Gaussian coverage at 50/80/95%, interval width, and interval score. For central (1-alpha) bounds [l,u]:

```text
interval_score = (u-l) + 2/alpha * (l-y) * 1(y<l)
                         + 2/alpha * (y-u) * 1(y>u)
```

Compute uncertainty on paired model loss differences with time blocks/episode clusters. If clustering across episodes, report the number of independent clusters. Block bootstrap over dates preserves all same-day query/portfolio results together. A bootstrap with very few episodes cannot manufacture evidence of generalization.

Stratify by event, sign/magnitude of x, starting probability, time to expiry, quote quality, calendar regime, and episode. Predeclare major cuts; treat additional slices as exploratory. Show uncertainty in the scores, not only point improvements. Report excluded intervals and extreme jumps separately; absence from the headline score is not evidence of safety on those intervals.

Promote an added layer only if it improves the prespecified held-out score by an economically meaningful threshold agreed before test inspection, without unacceptable deterioration in calibration, availability, or key subgroups; improvements should survive multiple independent episodes and prospective review. Do not invent a universal minimum percentage improvement. If uncertainty spans no improvement, classify as inconclusive.

**IC-specific conditions:**

- Historical LLM outputs have hindsight risk even when prompts restrict the date. The risk differs by job: J2 labels describe what drove a past day from cited evidence and carry less risk than J3 priors, which can absorb what the LLM knows happened afterwards. Timestamp and freeze new labels and priors now and evaluate their future performance. Historical results are exploratory evidence, not a substitute for prospective validation.
- The primary result is M2c vs M2. Each IC job's place is explained by its own step (J2a: M2 → M2a; J2b: M2a → M2c; J3: M2c → M3). A gain from J2 does not license J3, or the reverse.
- The variance step is kept only if the topped-up models are at least as well calibrated as `*_notopup` and the shared-risk check does not flag; otherwise report the cleaned models with that limitation.
- Check calibration specifically on company-specific stock days and held-contract thin-book days in the test set. If cleaned models are too narrow there, the variance step has failed.
- J2 must also pass its label audit: agreement on a re-labelled sample and human-reviewed precision per label. A score gain driven by labels that fail the audit is reported as unexplained, not as support for J2.

## 9. Required correctness and integration tests

1. **Units:** 1 percentage point is 0.01; converting probability units rescales coefficients but preserves final P&L.
2. **Alignment:** quotes after the endpoint never enter; stale/crossed quotes and missing endpoints are flagged; no interval bridging.
3. **Leakage:** changing future data cannot change any previously frozen model or response function. Changing held-out stock returns must not change predictions or eligibility.
4. **Moments:** mu/s use only training rows and documented ddof. Reject near-zero variance.
5. **Covariance:** C and Sigma symmetric positive definite; correct fitted-diagonal correlation normalization. V_r symmetric positive semidefinite within tolerance; materially negative eigenvalues are errors, not silently clipped.
6. **IC parity:** lambda=0 and empty prior sets match no-IC fitting. Omitted pairs are not forced to zero. Conflicting priors remain soft and produce a reported violation, not an invalid covariance.
7. **Conditioning:** direct matrix conditioning equals the scalar formula; multi-event ordering changes do not change results; unobserved events are integrated out, not fixed to zero.
8. **No double-counting:** the same covariance produces identical results regardless of how an optional graph is displayed. Do not add mediated effects after conditioning.
9. **Non-transitivity fixture:** covariance for variables [A,X,Y] is [[1,0.4,0],[0.4,1,0.4],[0,0.4,1]]. It is positive definite; conditioning on A alone gives E[Y|A=x]=0 despite both adjacent correlations being 0.4.
10. **Rotation:** L and LQ for orthogonal Q produce identical C, Sigma, scenario responses, and scores. This also tests why individual anonymous factors cannot be compared by names alone.
11. **Portfolio:** direct holdings plus ETF-derived holdings sum once; signed h handled; interval-start exposure used. V1 may omit ETFs if no point-in-time constituent adapter exists.
12. **Unsupported requests:** impossible probability endpoints, resolution requests, out-of-support bumps, and failed fits return explicit statuses.
13. **Mixed units and covariance:** event P&L uses contract count times payout times probability change; equities use dollar exposure times return. Check exact complementary NO sign, payout scaling, event-stock cross terms, and reduction to stock-only results when contract holdings are zero.
14. **Conditioned positions:** the conditioned event's mean change is x and its covariance row/column is zero. Accept singular full conditional covariance and handle fully deterministic portfolios without an invalid Gaussian score.
15. **Event-level support:** infeasible conditional levels and violated registered logical constraints are flagged, never clipped. A material held-event violation suppresses the mixed portfolio estimate. Mixed benchmarks all receive the conditioned position's known gain.
16. **Masked likelihood:** with no masks and unit weights the section 4.2 objective equals the v1.2 objective exactly. Masking an entry equals dropping that variable from that row's likelihood term; masked entries never enter as zero; raw values are never overwritten.
17. **J2 blinding and test integrity:** the labeller's prompt for a row contains no other panel variable's same-interval value and no model output (assert on the stored prompt). Cleaning never changes which test rows are scored or their values; changing any J2 label cannot change test-row values.
18. **J2 treatment ownership:** treatments come only from the configuration grid; an LLM output that names a treatment is rejected.
19. **J3 invariance and isolation:** every penalized quantity is unchanged under L -> LQ. The J3 prompt contains no fitted or sample correlations from the training window (assert on the stored prompt). Bucket-to-interval mapping comes from configuration only.
20. **J4 construction:** the embedded row reproduces `Cov(u_old, u_star) = L l_star` and the bucket communality; the extended covariance is positive semidefinite; embedding with a single analog at scale 1 and the analog's own communality reproduces that analog's conditional responses; cancelling analogs are rejected; no LLM-supplied probability level is accepted; the closest-analog baseline equals J4 with one analog, scale 1 and the analog's own communality.
21. **J5 isolation:** factor names never enter fitting, priors or conditioning (outputs identical with and without names); a name is dropped when the principal-angle or proxy-correlation check fails at refit; an add/remove proposal alone never changes k.
22. **Variance step:** cross-covariances equal the loading-step values; each diagonal equals `max(clean, raw)`; Sigma stays positive definite; raw models and `T0` bypass the cleaning pipeline and return the raw model object (identical snapshot hash); `mask_loading` entries change the loading step but not `v_raw`, while `mask` entries change both; the shared-risk check fires on a synthetic panel where masked days carry a common shock across stocks.
24. **J2 request isolation:** each stored J2 request contains exactly one variable's interval information; no request contains another variable's same-interval value.
25. **Invalid-observation documentation:** an `INVALID_OBSERVATION` label without `invalid_documentation` is rejected and treated as `CLEAN`.
23. **Technical-factor flag:** on a synthetic panel with an injected venue-wide event-only shock on microstructure-labelled days, the flag fires; on a panel without it, it does not.

### Golden numerical fixture (synthetic, not an economic estimate)

```text
mu = zeros(6)
s = [0.02, 0.015, 0.02, 0.02, 0.025, 0.018]
L = [[ 0.4,  0.0],
     [ 0.3,  0.2],
     [-0.2,  0.3],
     [ 0.5,  0.1],
     [-0.4,  0.2],
     [ 0.1, -0.5]]
psi = 1 - row_sum(L*L)
x = 0.02  # A increases 2 percentage points
h = [10000, 10000, 10000]
```

Expected conditional stock means: `[0.004, -0.004, 0.00072]`.
Expected P&L mean: `7.20` dollars.
Expected conditional portfolio variance: `104248.16` dollars squared.
Expected conditional portfolio standard deviation: approximately `322.875` dollars.
These expected values must be checked independently against the formulas, not copied into the implementation. The small net mean illustrates offsetting positions; it is not a forecast for these stocks.

Mixed-portfolio extension of the same fixture: add 100 A YES, 200 B YES and 100 C YES contracts, each paying $1. Then E[dp_B|dp_A=0.02]=0.0018 and E[dp_C|dp_A=0.02]=-0.0016. The conditional event-position mean is $2.20; total mixed mean is $9.40. With all other positions zero, 100 A YES contracts yield exactly $2.00 and zero conditional process variance. With 100 A NO contracts and exact complementarity, this becomes -$2.00.

### 9.1 Upstream and staged-experiment validation

Add tests for the contrast cases in section 0.2; proxy/no-match/ambiguous abstention; semantic matches with insufficient data; ETF aggregation without duplication; dated exposure/rule availability; and separation of probability, small-move P&L and unsupported resolution impact. Assert that matched comparisons (M2/M2c, M1/M1c, M2c/M3) use identical mappings, scored rows, holdings and factor count.

Build a human-reviewed challenge set containing exact matches, tempting near-matches and no-match cases. Group related templates/event families when splitting development and held-out cases. Report candidate retrieval recall, accepted-match precision, false exact matches and abstention/coverage, plus field-level errors. Choose acceptance rules on development cases only. The same LLM must not both produce and certify its own ground-truth labels.

Evaluate event discovery separately for relevant risks found, judged omissions and exposure coverage; do not claim exhaustive recall over every possible world event. Hold mappings fixed during numerical engine comparisons. Better retrieval must not masquerade as an IC improvement, and correct arithmetic must not certify correct matching.

## 10. Suggested interfaces and run outputs

Names are conceptual; adapt to the existing project rather than creating redundant abstractions.

```python
resolve_exposures(holdings, as_of, adapters) -> ExposureSnapshot
discover_events(exposures, info_cutoff) -> EventCandidates
parse_scenario(request, info_cutoff) -> NormalizedScenario
retrieve_contracts(scenario, registry_as_of) -> CandidateContracts
verify_contract_match(scenario, candidates, rules_as_of) -> MatchDecision
check_analysis_eligibility(match, data_audit, config) -> AnalysisEligibility
audit_data(registry, quotes, prices, config) -> DataAudit
build_design_panel(registry, quotes, prices, config) -> DesignPanel
propose_universe(exposures, registry_as_of, budgets, info_cutoff) -> UniverseProposal        # J1
flag_observations(panel, config) -> FlaggedEntries
label_observations(flagged, evidence_as_of, blind=True) -> IntervalLabels                    # J2
apply_treatment(panel, labels, treatment_id, config) -> RowMasks  # per step: loading|both; code only
build_hf_windows(labels, intraday_quotes, intraday_prices, config) -> HFWindows              # optional, level 2
propose_priors(definitions, j1_rationales, label_summary, info_cutoff) -> PriorSet           # J3
validate_ic_prior_set(priors, variable_order, info_cutoff) -> PriorAudit
embed_type2(proposition, snapshot, analog_proposal, config) -> ExtendedSnapshot              # J4
describe_factors(snapshot, proxy_returns, labels) -> FactorEvidence                          # code, for J5
propose_factor_changes(factor_evidence, info_cutoff) -> FactorProposal                       # J5
fit_covariance_model(train_rows, model_config, masks=None, weights=None, priors=None) -> ModelSnapshot
apply_variance_step(snapshot, train_rows, masks) -> ModelSnapshot                                 # cleaned models only
condition_model(snapshot, conditioning_changes, p_start, holdings) -> ScenarioResult
walk_forward(panel, split_plan, candidate_configs, prior_versions) -> EvaluationRun
score_and_compare(run) -> EvaluationReport
export_graph(snapshot) -> OptionalGraphView
```

Required run outputs: config and hashes; data-quality and power report; exclusion ledger; IC record provenance per job; J2 label audit (agreement, precision per label, masked share per variable and step); calibration on company-specific and held-contract thin-book test days; technical-factor flags; frozen model snapshots; per-interval predictions/actuals; paired-score table and uncertainty; coverage/width diagnostics; convergence and stability diagnostics; decision report (`SUPPORTED`, `INCONCLUSIVE`, `NO_VALUE_ADD`, or `INSUFFICIENT_DATA`). A numerical test passing does not imply empirical support.

Suggested implementation order:

1. Inspect environment. Run the stage 0 data and power audit; change the universe if the gate fails.
2. Implement exposure/universe (J1)/matching records and validate a fixed seed universe; build the separate matching challenge set.
3. Freeze the experiment configuration and start the prospective label and prior logs before test outcomes. Implement alignment, masked fitting, conditioning and correctness fixtures.
4. Run stage 2: M0, M1 and raw factor M2 under chronological evaluation.
5. Run stage 3: J2 labelling and audit; then M2a (J2a) and M2c (J2a + J2b + variance step) in sequence, the predeclared primary comparison M2c vs matched-k M2, the `*_notopup` and shared-risk checks, and the M1c replication. Preserve identical mappings and scored rows.
6. Run stage 4: M3 against matched M2c.
7. Run stage 5: the J4 leave-one-event-out test. Produce J5 factor evidence and names for display.
8. Report upstream and numerical evaluations separately; recommend keeping, simplifying or rejecting each added layer and each IC job.

## 11. Scaling the universe

With d variables and k factors, the model stores O(dk+d) parameters instead of O(d^2) covariance entries. Rank-k Gaussian likelihood operations can use Woodbury identities and determinant lemmas in roughly O(d k^2 + k^3) per covariance evaluation, plus data-dependent sufficient-statistic or streaming work. Do not claim overall fitting cost independent of sample count n. Avoid materializing the full d-by-d covariance where unnecessary; compute queried blocks from loading rows.

Maintain a versioned shared model where histories support it, then slice stock/event blocks for portfolios. Holding selection alone need not trigger refitting the statistical universe. Loading rows are reusable across users; holdings are separate. ETF decomposition requires point-in-time constituent weights, cash and unmapped-exposure accounting, and no duplicate ETF-plus-underlying exposure.

Short histories and asynchronous listings are the practical scaling bottleneck. V1 complete-case fitting will shrink overlap as d grows. Later use missing-data factor likelihood/EM or a hierarchical model with availability masks; never treat missing as zero. Define event families, maturity, and baseline probability effects before pooling recurring contracts. Do not concatenate a December decision and a January decision as one unchanged instrument.

The agent proposes a sparse subset of high-value priors, grouped by mechanism or event family. Templates are candidate assumptions, not sector-wide facts. Do not make O(d^2) LLM calls or instantiate dense zero priors. J2 labelling scales with the number of flagged entries, not with d^2; labels on a shared variable (e.g. an S&P constituent or a widely used event) are reusable across portfolios. Scaling uncertainty and quality checks matter more than graph visualization.

## 12. Adding, removing, and interpreting latent factors

Because L is rotationally non-unique, a column is not intrinsically an oil or growth factor. Track the common covariance LL^T, its subspace, and conditional responses. Align columns across windows with Procrustes/sign conventions only for display; this does not identify economic meaning.

**Add a factor:** persistent held-out residual dependence or conditional-response errors motivate testing k+1. Fit that candidate from scratch/multiple starts, using only development data and the same IC policy. Promote only if it improves predictive scores/calibration over repeated windows and remains stable enough to justify complexity.

**Remove a factor:** test k-1 with a full refit and identical evaluation; remove only if loss of performance is negligible under a predeclared criterion or the smaller model improves performance. Do not delete an arbitrary column because its narrative seems obsolete; another rotation mixes that column into others.

Changing k changes the admissible covariance family and may change all loadings. Preserve previous model versions. Never adapt k in response to the current held-out shock before scoring it. New semantic factors require a separately tested anchored model, for example macro proxies or identifiable loading restrictions, and revised IC semantics. V1 anonymous factors remain valid without economic labels.

**IC role (J5, section 4.3).** The LLM may propose k+1 or k-1 with a reason; the proposal triggers the tests above and nothing else. The LLM may also name factors, but only after code has fixed a display rotation and computed the evidence (top loadings, factor-score correlations with the proxy library, dates of the largest scores with their J2 labels). Code also flags `possible_technical` factors (section 4.3, J5). Names are display labels: they are versioned, dropped when the subspace or proxy evidence moves past the configured thresholds at a refit, and never enter fitting, priors or conditioning. Promoting a name into the model means building the anchored model above and testing it as a new model.

## 13. Type II events: an explicit future extension

An unlisted event has no market quote history. Give it an exact definition, horizon, resolution source and evidence set. Any probability level q comes from a market, a code-computed base rate or the user, never the LLM, and is used only for support checks. IC may propose this object and plausible response channels, but **q alone does not identify how stock prices move when q changes**.

Three evidence tiers:

1. **Prior-only scenario:** report qualitative channels and wide assumption-driven ranges. Do not include it in the empirical Type I model score or imply market-implied pricing.
2. **Analog-embedded scenario (J4, the default path):** the LLM picks 1-3 listed analogs with directions, scale buckets and a communality bucket; code builds the new row from the analogs' fitted loading rows (section 4.3). The result is conditional on the analog choice and buckets, which are disclosed with every output; it is not a measured sensitivity to the new event.
3. **Prospectively tracked series:** collect frozen snapshots of a probability series for the event (a newly listed market, or a user- or base-rate-driven series) at regular endpoints, alongside actual asset returns. Later evaluate whether this new series adds out-of-sample information beyond the existing observed variables, and compare its fitted row with the J4 embedding.

### Testing J4: leave one listed event out

For each listed event E in the universe: refit the model without E; run J4 blind to E's data (it sees E's definition and evidence only); embed E; then score the embedded model's conditional responses, given E's actual changes, against realized returns on the test rows. Compare with (a) the model refitted with E included, an upper benchmark, (b) the closest-analog baseline (E's nearest analog's responses, borrowed unchanged), and (c) M0. J4 earns its extra choices only if it beats (b). Report this score separately from the stage 2-4 headline. It measures how much of a listed event's real behaviour the analog method recovers, which is the only empirical check available for events that have no data.

### Optional coherent covariance extension

Given the fitted standardized old model u=LF+epsilon, define an unlisted event's probability change in **raw probability units** as:

```text
dp_star = mu_star + b_star^T F + eta_star
Var(eta_star) = tau_star^2 > 0
eta_star independent of F and old epsilon
Var(dp_star) = b_star^T b_star + tau_star^2
Cov(z_old, dp_star) = D L b_star
```

This adds a positive-semidefinite-compatible new row/column automatically. Condition using the same covariance formulas. However, mu_star, b_star, and tau_star are assumptions without data; q's level does not supply them. The magnitude of b_star/tau_star determines the stock sensitivity and must be disclosed. Because factors are anonymous, an agent should not assign economically meaningful b_star coordinates directly. Obtain desired observable relationships from evidence and solve for a compatible extension, or use an independently justified semantic factor model. Check feasibility: an assumed relationship vector may not lie in the old factor span.

If the event introduces a truly new driver, simply adding a factor loading only on dp_star does not create a stock effect. Existing stock loadings onto that new factor also require evidence or explicit uncertain priors. All such impacts remain provisional until tested.

Avoid circularity: an agent revising q after seeing the same-period stock move can mechanically create apparent event-stock dependence. For an incremental-information claim, freeze q before the target return period and compare with an autoregressive/market-informed baseline, or audit and control the inputs with a declared timing design. A same-interval subjective-q reconstruction can still be studied but must be labeled as such. Never feed the system's own generated portfolio impact back into q and count the loop as new evidence.

Never equate an agent's probability with what is already priced into equities. No resolution P&L should be inferred by extending this local model to q=1.

## 14. Definition of done

- Exposure/discovery/matching audit records and independent validation results are included; unsupported matches abstain.
- The stage 0 data and power audit is reported before any model result.
- Each IC job is evaluated on its own terms: J1 in the upstream evaluation; M2c vs M2 as the single predeclared primary comparison, with the J2 label audit, the explanatory sequence, and the variance-step tests; J3 on top of J2; J4 by leave-one-event-out; J5 by stability diagnostics only. Mappings are held fixed across numerical comparisons.
- The mathematical fixtures and alignment/leakage tests pass, including tests 16-25.
- All assumptions, units, label taxonomy, treatment grid, bucket tables, prior semantics, and exclusions are explicit.
- The experiment is reproducible through the existing environment.
- Comparison models use the same information and the same raw scored intervals; every IC job has a matched ablation.
- The report distinguishes conditional reconstruction, prospective forecasts, and unsupported resolution/causal claims.
- Every IC record and model version has a provenance trail; post-fit LLM overrides are absent; factor names are display-only.
- The final conclusion may be inconclusive or no value added. No fabricated empirical performance is acceptable.

**Principles retained for later development.**

1. Adding context variables does not automatically remove confounding.
2. A mathematically valid Type II embedding is not an empirically validated sensitivity.
3. Better conditional predictions do not by themselves establish causation.

## 15. Methodological references

These references support standard tools, not the empirical usefulness of this proposed system. The IC jobs, the interval penalty and the acceptance workflow are design proposals. References 7-10 support the identification ideas behind J2 and masked fitting.

1. C. Shalizi, *More PCA; and, Factor Analysis* (2008). Factor-analysis background and identification. https://www.stat.cmu.edu/~cshalizi/350/2008/lectures/12/lecture-12.pdf
2. O. Ledoit and M. Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* (2004). Shrinkage-covariance benchmark. https://www.ledoit.net/Well-conditioned2004.pdf
3. T. Gneiting and A. E. Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation* (2007). Distributional scoring. https://doi.org/10.1198/016214506000001437
4. D. H. Bailey et al., *The Probability of Backtest Overfitting* (2015). Model-search and backtest-selection risk. https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
5. J. Pearl, *Causal Inference in Statistics: An Overview* (2009). Conditioning and intervention are different estimands. https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf
6. J. Wolfers and E. Zitzewitz, *Interpreting Prediction Market Prices as Probabilities* (2006). Price/probability interpretation. https://www.nber.org/papers/w12200
7. C. Romer and D. Romer, *A New Measure of Monetary Shocks: Derivation and Implications* (American Economic Review, 2004). Narrative identification: classifying what drove each observation from the documentary record.
8. K. Kuttner, *Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market* (Journal of Monetary Economics, 2001). Event-window identification using market-implied surprises.
9. R. Gürkaynak, B. Sack and E. Swanson, *Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements* (International Journal of Central Banking, 2005). Narrow-window, high-frequency identification and its factor structure.
10. E. Snowberg, J. Wolfers and E. Zitzewitz, *How Prediction Markets Can Save Event Studies* (2011). Using prediction-market price changes to scale asset responses in event windows. https://www.brookings.edu/articles/how-prediction-markets-can-save-event-studies

## 16. Continuations (after Study A; not part of the current build)

Continuations extend the v1.6 build without changing Study A. Each starts only after the Study A report and the trainer's approval, uses the same harness, rows and scoring, and enters as an additional comparator, never as a replacement for a predeclared primary.

### C1. Bayesian hierarchical factor model (`M2b_bayes_hier`, Study B)

**Why.** v1.6 fits loadings by penalized maximum likelihood, so its P&L ranges ignore parameter uncertainty (they are labelled `process_only_plugin`). With ~120 training rows and thin event series, that uncertainty is likely a large share of the real error. Separately, the binding constraint is data per contract: every December-meeting Fed contract is a new instrument with a short history. A hierarchical model shares strength across contracts in the same family.

**Model.** The same factor structure as M2 (section 4.1), now Bayesian and hierarchical:

```text
u[t] = L F[t] + eps[t],   F[t] ~ N(0, I_k),   eps[t] ~ N(0, diag(psi))
L[j,:] ~ N(mu_family(j), Tau_family(j))      event series: loadings drawn around their family mean
                                              (Fed decisions, CPI, payrolls, ...; families from the contract registry)
L[i,:] ~ N(mu_group(i), Tau_group(i))        stocks: around their exposure-group mean (rate-sensitive, energy, ...)
mu_*, Tau_*, psi ~ weakly informative hyperpriors (fixed in configuration before fitting)
```

- **Output:** the posterior predictive distribution of portfolio P&L given x, integrating over loading uncertainty. Report it beside the plug-in Gaussian so the two can be compared on calibration.
- **Masked entries** (J2 treatments, section 4.2) are missing data in the likelihood: same masks, same variance-step logic applied per posterior draw.
- **J3 priors** become genuine priors on the same rotation-invariant quantities (correlation, response sign, communality), entered as soft constraints on the posterior. They are tested as an ablation against the hyperpriors alone, exactly as M3 is against M2c.
- **Identification:** loadings are not identified under rotation. Either constrain L (lower-triangular with a positive diagonal, a fixed variable order recorded in the snapshot) or summarize only rotation-invariant quantities (Sigma, conditional slopes, communalities). Never report a raw loading posterior.
- **Computation:** NUTS (e.g. NumPyro) for universes up to d of about 20. For Study B at scale, a Laplace or variational approximation is allowed only after its predictive calibration has been checked against NUTS on a predeclared subset of universes and refit dates. Computation runs offline in `worsey build`-style batch jobs; it makes **no LLM calls**.

**Evaluation.** Same rows, same proper score. M2b against matched-k M2 (does Bayesian uncertainty plus pooling help?), and M2b with and without family pooling (is the gain from pooling or from uncertainty?). Two failure checks:

- **Prior dominance:** report, per contract, how far the posterior moved from its family prior. If posteriors sit on their priors (the current engine's empirical-Bayes failure, METHODOLOGY §4.7), the gain is the prior, not the data, and is reported as such.
- **Coverage:** the posterior predictive should cover better than the plug-in model, not just more widely; the proper score, not the coverage, decides.

### C2. Hierarchical α in Phase R

Phase R's α (`PHASE_R.md` §1, R1.4: whether stocks respond linearly to the probability or to the latent news) is estimated hierarchically by event family, α_family ~ N(α_0, τ_α²), on top of C1. This partially pools thin families toward the overall α and gives each family its own interval. It is the natural home for the nonlinear terms of Phase R (state-dependent scale, regimes, κ by family): a Bayesian nonlinear mixed model, built only after C1 has shown that the hierarchical machinery is calibrated.

