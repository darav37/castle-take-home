# Worsey — Event & Scenario Analysis Methodology

**Status:** v1.0 (2026-10-15). This is the authoritative reference for how Worsey finds, prices and combines the events that could hit a portfolio. `SPEC.md` §3 points here, and wherever the two disagree on methodology, this document wins.

**Oct 18:** a successor methodology for small probability moves, the prediction-market response model (PMRM), is specified in `docs/research/pmrm_spec.md` and is being built and tested beside this one (DECISIONS #146). This document stays authoritative for the live engine, resolution scenarios and the hedge comparator until the PMRM results are in and the trainer decides.

**Oct 21:** the small-move PMRM is parked (P0c: no usable held-out signal, #153, #155). The research track is now Plan B (#152). **Overview of the current state: `docs/event_pipeline/overview.md`.** This document still describes the engine the chat runs today.

**Oct 21 (DECISIONS #179): the chat now runs Plan B** (`src/worsey/planb/live.py`, method in `docs/event_pipeline/overview.md`). This document describes the v1 engine, which stays in the code for the recorded demos (`SCENARIO_ENGINE = "v1"`).

**Scope:** the scenario engine (`event_analyst`, `scan_risks`, `analyze_event`, `merge_scenarios`) and its evaluation. Options and the hedge comparator are covered here only where they consume scenario output.

**Status tags:**
- **[Built]** — in the take-home build
- **[Tier B]** — built if time allows
- **[Design]** — documented, not built

---

## 0. The question and our answer

The brief asks: *what could hit my portfolio?* That question is open-ended on purpose, so our first methodological choice is to define it.

> **Definition.** Over a stated horizon H, the events that *could hit* a portfolio are those with the largest **standard deviation of portfolio P&L from the event's surprise**. For a binary event that is |B|·√(p(1−p)), where B is the portfolio's dollar sensitivity to the event's probability and p is the market-implied probability.

**Three ideas carry the whole methodology:**

1. **Prediction-market probabilities are factors.** A portfolio has a measurable sensitivity (β) to each event's probability, just as it has a β to the S&P.
2. **Only the surprise moves you.** Prices already embed the market's expectation. When an event resolves, the portfolio moves by β × (outcome − p), not by β × outcome.
3. **Claude chooses, code computes.** The LLM finds events, matches them to markets, proposes mechanisms and explains. Every number comes from data and code.

---

## 1. Definitions and notation

| Symbol | Meaning |
|---|---|
| i | a position (stock, ETF, option) |
| Vᵢ | current dollar value of position i |
| m | an event / market |
| pₘ | market-implied probability of event m (§7) |
| Oₘ ∈ {0, 1} | resolution outcome |
| xₘ = logit(pₘ) | log-odds, used for dependence modelling (§8, §11) |
| βᵢₘ | return of position i per +1.00 change in pₘ |
| Bₘ = Σᵢ Vᵢ βᵢₘ | portfolio dollar sensitivity to event m |
| H | analysis horizon (default 3 months) |
| f | macro factor vector (§4.2) |
| Lᵢ | position i's loadings on f |
| sₑ | an event's shock map: factor moves per unit of surprise |

**Event schema** (canonical; used for discovery, matching and the library):
```json
{"variable": "US average tariff rate on Chinese imports",
 "comparator": ">=", "threshold": 25, "unit": "%",
 "deadline": "2027-03-31", "jurisdiction": "US",
 "resolution_source": "USTR / Federal Register",
 "type": "trade_policy", "why_it_matters": "..."}
```

---

## 2. Core model: probabilities as factors [Built]

### 2.1 Return model
For position i and event m, over trading days t:
```
r_i,t = a_i + β_im · Δp_m,t + ε_i,t
```

**Why Δp and not Δlogit(p) for β:** resolution moves p to exactly 0 or 1, so a β estimated in probability units maps directly onto resolution P&L (§2.2). Logit is the better scale for *dependence* between markets, because it stabilizes variance near 0 and 1, so §8 and §11 use logit.

This choice is open for the trainer's review (SPEC §12). The alternative, logit β with a delta-method mapping to resolution, is noted in §15.

**Why total β, not market-residual β:** the user's question is "what happens to my portfolio", including whatever the event does to the S&P. So β is estimated without a market control. SPY's own β to the event is reported alongside, so the reader can see how much of the effect is broad-market.

### 2.2 Resolution P&L (the priced-in adjustment)
```
P&L(YES) ≈ B · (1 − p)
P&L(NO)  ≈ −B · p
E[P&L]   = p·B(1−p) + (1−p)·(−Bp) = 0
```
The expected P&L is zero under the market's own probability, because the expectation is already in prices. That is the precise answer to the question of how much of these events today's prices already carry.

**Caveat, stated in the README:** p is a market price, not a physical probability. It carries risk premia, fees and a longshot bias (§7). The identity above holds under the market's p, not necessarily under the true probability.

**Cross-asset split.** Because β is per position, P&L(YES) decomposes by asset class:
- equities
- rates
- credit
- commodities
- crypto

Positions with opposite-signed β show the *natural hedges already in the book*.

### 2.3 Ranking metric
**Events that resolve inside the horizon:**
```
σ_m = |B_m| · √(p_m (1 − p_m))
```
This is the standard deviation of resolution P&L, i.e. the std of a Bernoulli surprise scaled by sensitivity. It ranks a 95%-likely event with a big β *below* a 50/50 event with a moderate β, which is correct: the first one is mostly priced.

**Events that resolve after the horizon.** Within H the event can't resolve, but its probability can move. The relevant risk is drift:
```
σ_m^drift = |B_m| · σ(Δp_m over H)
```
σ(Δp over H) is estimated from the market's own history, or from its stitched series (§5.1), scaled by √(H / Δt).

The report labels these rows separately: "won't resolve before your horizon; risk is the odds moving."

**What B is in the ranking (Oct 16).** σ and σ^drift use the *posterior* B from §4.7 (empirical Bayes toward the structural prior), not the raw regression estimate. Rows are listed in two groups: **above the divider** (strong evidence: |B_post|/sd ≥ 2, or a tier-2 mechanism with material exposure), then **"weak evidence"**, each group ordered by σ.

### 2.4 Option positions [Tier B]
Option positions aren't linear in the underlying, so their scenario P&L is computed by **repricing**, not via β:
1. Shock the underlying by βᵤ·(O − p).
2. Shock implied vol by the event's vol map; for event-dated options this includes a post-event **vol crush** to the next-expiry ATM level.
3. Reprice with Black-Scholes-Merton.

Aggregated portfolio greeks are reported.

---

## 3. Exposure profile and ETF look-through [Built]

**Look-through:**
```
w_name = w_direct + Σ_etf w_etf · h_name,etf
```
- **h** are ETF holdings weights from Massive: a current snapshot, truncated to the top holdings covering ≥ 90% of weight.
- Cash and futures sleeves are treated as cash.

**What look-through is used for** (and what it is *not*):
- **Concentration:** "you're 14% NVDA, not 6%".
- **Name- and sector-specific events:** mapping, e.g., chip export controls → NVDA, AMD and every ETF that holds them.
- **Explanation:** "which holdings drive this".
- **Not for macro β.** For macro events, β is estimated on the position itself (the ETF's own return series). Regressing ~100 constituents separately only adds noise.

**Exposure profile** (computed in code and passed to the experts):
- name, sector and country weights
- factor loadings (§4.2)
- weather loadings (§9)
- top concentrations
- upcoming name catalysts (earnings dates from `equity_analyst`)

**Stated assumption:** current weights applied to historical returns, i.e. no position history.

---

## 4. Estimating β: three tiers [Built]

Every β carries a tier tag, a standard error, and a confidence label shown to the user.

| Tier | Source | When used |
|---|---|---|
| 1 | Empirical, from prediction-market history | enough history and jumps, or a recurring series |
| 2 | Event → macro shocks × asset factor loadings | known event type, no usable market history |
| 3 | LLM prior (sign + size bucket) | last resort, and as shrinkage |

### 4.1 Tier 1 — empirical
**(a) Single-market regression:**
- Use daily data aligned to the equity close. The market's price is the **last trade or mid before 16:00 ET**; stale prices (no trade or quote update that day) are masked.
- **Minimum history:** 60 trading days, and at least 5 days with |Δp| ≥ 5pp. Most information arrives on jump days, so without jumps β is just noise.
- **Standard errors:** HAC (Newey-West, 2 lags).
- **Shrinkage toward zero:**
  ```
  β_shrunk = β̂ · t² / (t² + k)        default k = 4
  ```
  At |t| = 2 this keeps half of β̂. The value of k is tuned in a sensitivity table (§16).
- **Energy-related markets** (e.g. a Henry Hub price market) include the weather anomaly as a control, so the weather channel isn't double-counted (§9).

**(b) Pooled recurring series (the main source).** Kalshi re-lists the same question for every Fed meeting, CPI print and payrolls release. Each past resolution r gives one observation:
```
r_i,window(r) = β_i · (O_r − p_prev,r) + ε_r
```
- **The window** runs from the close before the release to the close after. Intraday windows can be used where prices exist.
- **p_prev** is the market price just before the window.
- **Pool across all past resolutions of the series** for a single β with a clean, release-timed identification. This is the same high-frequency identification macro economists use around Fed announcements.

### 4.2 Tier 2 — two-stage factor mapping
**Stage B: asset → factors (estimated offline in `worsey build`).**
```
r_i,t = α_i + L_i · f_t + ε_i,t
```
- Rolling 2-year daily window, ridge-regularized, with HAC standard errors.
- **Factor set f** (small, all tradable proxies available in Massive):

| Factor | Proxy |
|---|---|
| Equity | SPY |
| Rates | TLT (or Δ10y) |
| USD | UUP |
| Oil | USO |
| Gold | GLD |
| Credit | HYG − IEF |
| Vol | VIXY |
| Sector (as needed) | SMH, XLE, XLF, XLU, ITA, KWEB, … |

**Stage A: event → factor shocks (the shock map sₑ).** sₑ is the vector of factor moves *per unit of surprise* (surprise = O − p). It comes from one of two sources:
- **A type template**, fitted across historical instances of the event type (§4.4).
- **A novel shock map** proposed by Claude, anchored to retrieved analogs whose realized factor moves code computes and shows alongside (§4.5).

**Combine:**
```
β_i = L_i · s_e
Var(β_i) ≈ L_i Σ_s L_iᵀ + s_eᵀ Σ_L s_e
```
- Σ_s is the shock-map uncertainty (template residual dispersion, or a wide default for novel maps).
- Σ_L is the loading uncertainty.

### 4.3 Tier 3 — LLM prior and blending
- Claude returns only a **sign and a size bucket**. Code maps buckets to numbers:

| Bucket | |β| (return per +1.00 Δp) |
|---|---|
| small | 0.01 |
| medium | 0.03 |
| large | 0.08 |

  Prior std = 100% of the bucket value. The prior is deliberately wide.
- **Blending by inverse variance** (applied whenever an empirical or template estimate exists):
  ```
  β_post = (β_prior/s²_prior + β̂/se²) / (1/s²_prior + 1/se²)
  ```
- With no other source, the prior stands alone, flagged **low confidence**.

### 4.4 Event-type templates (new types and training)
For an event type with instances k = 1..K (3 to 8 historical dates approved by the trainer):
```
Δf_k = s · u_k + ε_k
```
- **u_k ∈ [−1, 1]** is the trainer's surprise score: roughly, how much the news moved the odds.
- **Δf_k** is the factor-move vector over the instance window, computed by code.
- **Fit s by ridge regression,** shrinking toward a prior.
- **Fewer than 3 instances (Oct 16):** residual noise is unidentified, so it is set to the moves' own size; the template is shown with wide bands (b ± 2se) and labelled low confidence.
- **Leave-one-out:** predict each instance from the others and report the error per factor.
- **Units:** s is "factor move per unit surprise", the same units as a tier-1 β. That makes templates and empirical estimates directly blendable.
- **As built today (unfitted, unpriced template; Oct 17):** until surprise scores are approved, the template is the book's P&L on each analog date. **"Typical" = the median episode** (the mean of the two middle ones when the count is even); its by-position split and the hedge's typical moves come from the same episode(s), so they add up to the median exactly. The **average** is shown separately. The stress case is the worst episode. Why median: in the trade-policy template one outlier (Liberation Day, −$13.6k on `tech_heavy`) pulls the mean to −$4.1k, about 1.8× the median episode (−$2.3k).

### 4.5 Worked-example library (few-shot for novel events)
- Lives in `examples/events/*.yaml`. Each entry has the event, the schema, the type, the date window, a surprise level, the trainer's reasoning, and **factor moves computed by code**.
- **For a novel event:**
  1. Retrieve the 2–3 nearest examples by type, then by embedding.
  2. Claude proposes sₑ by comparing the new event to each one (bigger or smaller, more or less surprising).
  3. Code prices the result.
  4. The output shows the proposal next to the analogs' realized moves, flagged `novel_unreviewed` and queued for train mode.
- **Double-counting caveat:** analog moves are already surprise-scaled. The (1 − p) adjustment applies on top, so analogs must be scaled to a unit surprise before use.

### 4.6 Tier selection
1. A usable recurring series → tier 1(b).
2. Else, a single market passing the §4.1 filters → tier 1(a), blended with tier 2 if a template exists.
3. Else, an event type with an approved template → tier 2.
4. Else → a novel tier-2 shock map, plus a tier-3 prior blend, flagged.

### 4.7 Ranking: empirical Bayes toward the structural prior (Oct 16) [Built]

**Why.** On real data (`balanced`, ~270 trading days, 98 ranked rows), only **1% of events have a book-level |t| ≥ 2**, and the cross-sectional signal variance of the raw estimates is ≈ 0: the dispersion of B̂ across markets is fully explained by sampling error. Ranking raw estimates by σ therefore ranks noise (winner's curse: a Venezuela-leader market came out #1 for a US equity book). This is the honest headline result for tier 1(a) and is reported in the README.

**Estimate.** For every candidate market:
- **B̂, se:** pooled tier 1(b) where the market belongs to a recurring series with ≥ 8 past releases (Fed decision by outcome, CPI and payrolls "above X"): the book's return over each release window (close before → release-day close) regressed on surprise = outcome − p at the prior day's 16:00 ET, errors clustered by release. Otherwise tier 1(a): the book's daily return on Δp (HAC).
- **Structural prior B_prior, s_p:** Σᵢ Vᵢ·(Lᵢ·s) with factor loadings L (§4.2). The shock map s is the approved tier-2 template when one exists; otherwise Claude's tier-3 sign + size bucket per factor (stocks, long Treasuries, dollar, oil, gold, credit, vol), numbers from the bucket table (§4.3), asked once per market in `worsey build` and cached. Prior sd per factor = 100% of its bucket; s_p aggregates them through the book's factor exposures.

**Shrink.** With residuals r = B̂ − B_prior across all candidates:
```
τ² = max(0, mean(r² − se² − s_p²))          (method of moments: signal net of sampling and prior noise)
v  = τ² + s_p²
B_post = B_prior + r · v / (v + se²)
sd_post = √(v · se² / (v + se²))
```
The trainer's specification is the s_p = 0 case. Including the prior's own uncertainty keeps the evidence test meaningful when τ² ≈ 0 (otherwise sd_post → 0 and every row would look "strong"). Per-position betas shrink with the same weight, so the asset-class split adds up to B_post.

**Divider.** A row ranks above it only if |B_post| / sd_post ≥ 2, or it has a tier-2 mechanism (approved template) whose σ ≥ 0.25% of the book. Everything else is labelled "weak evidence". With no approved templates yet, only strong-evidence rows (typically the pooled Fed/CPI/payrolls events) appear above it.

**Flags.** Rows that reached the history stage only through the standing-calendar exemption carry `standing_calendar_only` (the exemption is from the volume cap only; history and staleness filters still apply).

---

## 5. Discovery: finding the events [Built]

### 5.1 Stitched constant-maturity series
Individual contracts are short-lived and rarely overlap, so the event universe is organized as **stitched series**, much like continuous futures:
- "Fed cuts at the next meeting" rolls to the next meeting after each one.
- "CPI YoY ≥ x at the next print" rolls to the next print.

Series give the history that tier-1 β (§4.1), drift risk (§2.3) and the dependence model (§8, §11) all need. Each series records its roll dates and the contract it maps to on any date.

### 5.2 Three sources of candidates
1. **Bottom-up market scan** (offline in `worsey build`, refreshed live only for p):
   - **Universe:** all Kalshi and Polymarket markets in macro, economics, politics, geopolitics and weather.
   - **Filters:** resolves within 12 months, liquidity above a threshold, at least 60 days of history (or a stitched series).
   - For each market, compute B and σ; keep the top ~50 for the portfolio.
   - This finds exposures nobody would think to ask about.
2. **Exposure-driven top-down.** `event_analyst` and `equity_analyst` read the exposure profile and propose events in the schema. Examples:
   - semis → export controls
   - China ADRs → tariffs
   - IWM → the Fed path
   - single names → earnings
3. **The standing calendar,** always checked: FOMC, CPI, payrolls, elections, and major scheduled policy deadlines within H.

**Merge and dedupe** by schema similarity. Top-down events with no market are kept as **unpriced**.

**As built in `worsey build` (Oct 16):**
- Series-level category filter: Sports and Entertainment series excluded; in scope: Economics, Financials, Politics, Elections, World, Companies, Commodities, Climate and Weather, Science and Technology, AI. **Crypto is out**: its markets are price levels/returns of crypto assets.
- **Reflexive price-level markets are excluded** ("S&P 500 above 6845 on Dec 31", "WTI max above $115", Treasury-yield levels): they are the asset itself, so their β to a book is mechanical and would top every ranking.
- The volume cap is **round-robin across categories**, and the **standing calendar** (Fed, CPI, payrolls, recession, tariff-policy series) bypasses the cap; both still face the history, liquidity and tier-1 filters.
- Pointed questions match plain words against the whole built universe (not just the top rows) and show the candidates with match scores (§6 step 1).

---

## 6. Matching events to markets [Built]

1. **Retrieve** candidate markets by keyword (v1) or embedding ([Tier B]) over titles and *rules text*.
2. **Parse:** Claude parses each candidate's rules into the schema.
3. **Score:** code scores mismatch field by field:
   - variable: same / proxy / different
   - threshold distance
   - deadline gap
   - resolution source
   - jurisdiction
4. **Decide** (every decision is logged with a reason string):
   - **Exact.** Use the market directly.
   - **Derived: strike ladder.** If the ladder brackets the threshold, interpolate the survival function S(k) = P(X ≥ k) monotonically between adjacent strikes (linear in k, by default). Never extrapolate beyond the ladder.
   - **Derived: time.** If the deadline differs:
     - For hazard-like events (e.g. "invasion by date"), assume a constant hazard: `p′ = 1 − (1 − p)^(T′/T)`.
     - For **meeting- or release-driven events** (e.g. "Fed cuts by December"), don't use calendar time. Count the decision dates inside each window and work from per-meeting probabilities.
   - **No fit.** Keep the event as unpriced and give the reason, e.g. "closest market resolves on tariff *announcement*, not *effective rate*".
5. **Labels.** Matching labels (~15 pairs, hand-labelled by the trainer) test this step (§14).

---

## 7. Probabilities [Built]

- **Price:** use the mid; if the spread is > 10pp, use the last trade, flag it, and mask stale quotes.
- **Cross-venue:** when both Kalshi and Polymarket list the event, report both, combine weighted by liquidity, and **flag a gap above 5pp** ("venues disagree").
- **Confidence label** comes from the spread, volume, staleness, and whether p is derived (§6).
- **Caveats stated, not corrected:**
  - the longshot bias (small p tends to be overpriced)
  - fees
  - risk premia, since p is a market price and not a physical probability
- **Truth-in-odds rule:** these caveats surface automatically whenever a user looks at a low-probability contract.

---

## 8. Dependence and joint scenarios [Built: copula; Tier B: graph-based]

Per-event P&Ls are **never summed**: events are correlated (Fed cut, recession and soft CPI move together).

### 8.1 Themes
- Events whose stitched series have highly correlated Δlogit are grouped into **themes**, shown as one row with members.
- Within a theme, βs are estimated **jointly** (a multivariate regression on all members' Δp), so a shared channel such as SPY isn't double-counted.
- With the event graph (§11), themes become graph communities.

**Membership rule** (ruling, Oct 16). Markets form one theme if either:
- they belong to the same series or underlying variable (e.g. a CPI strike ladder, or CPI markets for adjacent prints), or
- |corr(Δlogit)| ≥ 0.7 on ≥ 30 overlapping days.

**Display.**
- A theme shows as **one row** in the scan card, headed by its most liquid member.
- The row expands to list its members, each with its own p and P&L.

**Ranking.**
- The theme is ranked on its **joint** σ: the std of portfolio P&L across the members' joint outcomes from the copula (§8.2), computed with the jointly estimated βs.
- Members' σs are never summed.
- If the joint model is unavailable, the theme ranks on its single largest member's σ, flagged.

### 8.2 Joint resolution model (latent Gaussian copula)
- Each event m has a latent Zₘ ~ N(0, 1) with Oₘ = 1{Zₘ < Φ⁻¹(pₘ)}, so the marginals match market probabilities exactly.
- **Latent correlation R:**
  - v1 uses the Δlogit correlation of the stitched series, shrunk toward the identity (Ledoit-Wolf).
  - With §11 built, R comes from the graphical-lasso estimate.
  - Either way, this is a proxy for outcome correlation, stated as an assumption.
  - **Too little overlap (< 30 days for a pair; Oct 16):** the pair gets a tail-conservative fallback ρ = 0.5·sign(Bᵢ·Bⱼ): **+0.5 for same-sign exposures**, −0.5 for opposite-sign, never 0. Both choices raise the variance of portfolio P&L. Filled pairs are labelled in the output (`dependence.fallback_pairs`).
- **Logical constraints are built into the structure, not estimated:**
  - **Strike ladders on the same variable share one latent Z** with different thresholds, so nested probabilities are monotone by construction.
    - **What counts as a ladder (Oct 16):** markets in one event that share ONE numeric variable (same `greater`/`less` strike type, parsed distinct thresholds) with prices roughly monotone across thresholds (breaks ≤ 3pp). "Which countries…" or "will A / will B" events are NOT ladders: their markets are separate events, correlated normally.
  - **Mutually exclusive outcomes** use a single categorical draw.
  - **Nested deadlines** share a latent variable with time-ordered thresholds.
- **Simulation** (N = 50,000 draws):
  ```
  P&L = Σ_m B_m (O_m − p_m)  (+ option repricing, + weather factor draws from §9)
  ```
- **Reported:**
  - the P&L distribution
  - VaR and expected shortfall at 5%
  - each event's **Euler contribution to ES**
  - the **most likely bad combination**: the most frequent joint outcome among tail draws

### 8.3 Conditional scenarios
- **Resolution conditioning:** "What if Iran escalation resolves YES?" Condition the copula on Z_A < Φ⁻¹(p_A) and report every other event's conditional probability and the conditional portfolio P&L.
- **Odds-move conditioning** ("what if the odds of A jump 20pp?") is done in logit space with Gaussian conditioning:
  ```
  E[x_B | x_A = a] = μ_B + Σ_BA Σ_AA⁻¹ (a − μ_A)
  ```
  The result is mapped back to p, then to P&L.
- **Multi-event questions** ("X *and* Y") condition on both, and report whether the two events tend to co-occur.

---

## 9. The weather channel (summary; full spec in SPEC §3.4) [Tier B]

- Weather is a **continuous factor**, not a binary event. `meteorologist` supplies the distribution of the degree-day anomaly over H from ECMWF/GFS ensembles.
- **Weather loadings** (estimated offline): each asset's β to the weekly HDD/CDD anomaly, e.g. UNG, VST, CEG, NRG, XLU, and the ag ETFs in season.
- **Priced-in logic:** the forecast is public and mostly priced, so P&L = L × (realized anomaly − currently forecast anomaly), and risk is σ = |L| · σ_ens.
- **Same ranking unit** as §2.3, so `merge_scenarios` interleaves weather rows with event rows.
- **No double-counting:** energy-event β uses weather as a control (§4.1); in the joint simulation, weather is drawn jointly with correlated event Δp.

---

## 10. Bespoke events (no market, no template) [Built]

Example: *"9.0 earthquake in Santiago tomorrow; I hold an ADR headquartered there."* `event_analyst` runs a **fixed recipe**:

1. **Exposure map.**
   - `equity_analyst` establishes the company's in-country operations, revenue and assets, with citations.
   - Look-through finds hidden country exposure (ECH, copper miners, local banks and utilities inside ETFs).
2. **Channels.** Claude lists them, and each is checked against the factor set:
   - direct operations
   - country index and FX
   - commodity supply (copper)
   - insurers and reinsurers
3. **Analogs.** Code computes realized moves for analog windows, e.g. Chile's Feb 2010 M8.8 earthquake, and Tōhoku 2011 for scale.
4. **Shock map.** Claude scales the analogs; code prices position-level P&L with **wide bands**.
5. **Probability.** The market has effectively no pricing for it (p ≈ 0), so the move if it happens ≈ the full shock.
   - **A probability comes only from:** a market, a code-computed base rate from a public catalog, or the user.
   - **Never an LLM-guessed p.**
   - With no p, show P&L at user-chosen probabilities.
6. **Hedges.** Hand off to `compare_hedges`:
   - OTM puts (with a tail-skew note)
   - cross-hedges
   - sizing
   - a "do nothing" row
7. **Flag** the result `novel_unreviewed` and queue it for train mode.

---

## 11. The event graph [Tier B: small version; Design: full]

A network over **stitched series nodes**, plus factor nodes (SPY, TLT, USO, …) linking down to positions. Every edge is labelled honestly by type.

| Edge type | Meaning | Estimation | Label shown to users |
|---|---|---|---|
| Logical | exact constraints (ladders, nested deadlines, exclusivity, conditional markets) | rules | "logically implies" |
| Conditional dependence | linked after controlling for all other nodes | graphical lasso on daily Δlogit (logit clipped to p ∈ [0.01, 0.99]) | "moves with" |
| Impulse response | reaction to an identified shock | event windows around scheduled releases: Δx_B = γ_AB · surprise_A + ε | "reprices when A surprises" |
| Lead-lag | information flow on unscheduled news | Granger / VAR on liquid, staleness-masked intraday data | "tends to move after" |
| Mechanism | hypothesized causal chain | proposed by Claude, approved by the trainer in train mode, weighted or pruned by data | "hypothesized mechanism" |

**Price data alone gives dependence and information flow, not causality.** Event-window impulse responses around scheduled releases are the closest thing to causal identification available, and they are labelled as such.

**Uses on the backend:**
1. **Joint scenarios:** the sparse precision matrix replaces the shrunk correlation in §8.2, and logit-space conditioning (§8.3) becomes graph propagation.
2. **Testing the user's causal story** (the Iran/IWM trace): show the estimated strength of each link along the path (Iran → oil → CPI → Fed → small caps) and where the chain breaks.
3. **Discovery:** surface upstream nodes of what the user is exposed to.
4. **Themes:** themes = graph communities.
5. **Truth-in-odds flags:** flag logical edges violated across venues.

**Scope of the small version:**
- logical edges
- graphical lasso on 30–50 macro and political stitched series
- impulse responses for FOMC, CPI and payrolls

Lead-lag and the mechanism layer are [Design].

---

## 12. From scenario to hedge [Built]

A **scenario object** is either an event (from §2–§10) or a user view:
```json
{"kind": "event|view", "id": "...", "loss_if_yes": -2990, "horizon": "2027-03-31",
 "drivers": [{"position": "QQQ", "contrib": -1170}, ...], "confidence": "tier1|tier2|tier3|novel"}
```
`compare_hedges` turns it into 3–5 candidates, always including a **"do nothing"** row. Each candidate reports:
- cost
- payoff if YES and if NO
- coverage ratio
- cost per $ covered
- residual loss
- basis risk

---

## 13. Output contract

**One event row** (from `event_analyst` → `merge_scenarios` → the card):
```json
{"event": "...", "schema": {...}, "market": {"venue": "kalshi", "ticker": "...",
 "match": "exact|ladder|time|none", "reason": "..."}, "p": 0.35,
 "p_confidence": "high|med|low", "B": -4600, "beta_tier": "1a|1b|2|3|novel",
 "beta_se": ..., "pnl_yes": -2990, "pnl_no": 1610,
 "pnl_yes_by_class": {"equity": -3445, "rates": 292.5, "gold": 260, "credit": -97.5},
 "sigma": 2194, "resolves_in_horizon": true, "drivers": [...],
 "theme": "trade_policy", "flags": ["novel_unreviewed", "venues_disagree", "correlated_with:..."]}
```

**What the user sees:**
- The table and cards: p, P&L if YES/NO, a match badge and a confidence label.
- Prose from the experts, written only from these fields. Every number in a reply must appear in a tool result (the grounding check).

---

## 14. Validation and evaluation

All evaluation is **point-in-time**:
- β, loadings, templates and matches use only data available before each event's resolution
- holdings are fixed at a snapshot
- no look-ahead

1. **Resolution backtest (headline).** 60–100 resolved markets across both venues and all categories.
   - Predict the move on resolution day as B·(O − p_prev), and compare it to the realized move.
   - **Metrics:**
     - sign hit-rate
     - correlation
     - MAE vs a zero-move baseline
     - **coverage**: the share of realized moves inside the ±1σ and ±2σ predicted bands
   - Report each by category and by β tier.
2. **The LLM prior's value.** On the same set, compare tier-3 priors, tier-2 maps and tier-1 β against realized moves. This measures how much weight the LLM deserves.
3. **Templates: leave-one-out** error by factor and by event type (§4.4).
4. **Matching:** agreement with the trainer's ~15 labels.
5. **Look-through:** effective NVDA weight for a QQQ holder vs the issuer's published weight.
6. **Joint model:**
   - On held-out windows, check that the copula's predicted co-resolution frequency matches realized co-resolution for correlated pairs.
   - [Tier B] For the graph, predict Δx_B given A's jump on held-out event windows.
7. **Weather:** see SPEC §3.4d (calibration of ensemble vs market vs blend; sanity of UNG's HDD loading).

**Weak results reported honestly are fine.** The README shows them.

---

## 15. Known limitations and failure modes

- **Linear in p.** Misses convexity in the event → price mapping. Logit-β with a delta-method mapping is the alternative, deferred.
- **Jump scarcity.** Information arrives on a few days, so single-market β is noisy, hence shrinkage, pooling and tiers.
- **Confounding.** On a jump day other news also hits. The fixes are event-window identification (§4.1b, §11) and honest labels.
- **Non-stationarity.** Regimes change (e.g. the stock–bond correlation sign flipped in 2022), so loadings and graphs use rolling windows and are re-estimated nightly.
- **Holdings snapshot.** Current weights are applied to history.
- **Market p ≠ physical probability.** Risk premia, fees and the longshot bias.
- **The copula is a proxy.** Δlogit correlation stands in for outcome correlation.
- **Thin and illiquid markets** create fake lead-lag and stale prices, so they are masked and filtered.
- **LLM steps can be wrong** (matching, shock maps). They are logged, flagged, reviewable in train mode, and never produce numbers.

---

## 16. Default parameters (tuned in a sensitivity table)

| Parameter | Default |
|---|---|
| Horizon H | 3 months |
| Min market history (tier 1a) | 60 trading days |
| Jump threshold / min jump days | 5pp / 5 days |
| Shrinkage k | 4 |
| HAC lags | 2 |
| Loading window | 2 years, rolling, ridge |
| Look-through coverage | top holdings ≥ 90% weight |
| Spread threshold for using last trade | 10pp |
| Cross-venue disagreement flag | 5pp |
| Logit clip | p ∈ [0.01, 0.99] |
| Copula draws | 50,000 |
| Fallback correlation (overlap < 30 days) | ±0.5 by exposure sign, labelled |
| Ladder monotonicity tolerance | 3pp |
| Tail level for VaR/ES | 5% |
| Tier-3 buckets (|β|) | 0.01 / 0.03 / 0.08, prior std = 100% |
| Template instances | 3–8 |
| Bottom-up shortlist | top 50 by σ |
| Pooled series minimum | 8 releases |
| Divider | \|B_post\|/sd ≥ 2, or tier-2 mechanism with σ ≥ 0.25% of book |
| Unpriced-template fallback | keyword match score < 0.20 → "no strong market" + type template |

---

## 17. Worked example (illustrative numbers, not real estimates)

**Portfolio:** $100k, the `balanced` demo.

**Event:** "US average tariff on Chinese imports ≥ 25% by 2027-03-31", matched **exact** to a Kalshi market at **p = 0.35**. β is tier 2 (trade-policy template), blended with tier 1(a).

| Position | Vᵢ | βᵢ | Vᵢβᵢ |
|---|---|---|---|
| SPY | 30,000 | −0.06 | −1,800 |
| QQQ | 20,000 | −0.09 | −1,800 |
| NVDA | 10,000 | −0.15 | −1,500 |
| TLT | 15,000 | +0.03 | +450 |
| GLD | 10,000 | +0.04 | +400 |
| XLE | 10,000 | −0.02 | −200 |
| HYG | 5,000 | −0.03 | −150 |
| **B** | | | **−4,600** |

**Resolution P&L:**
- **P&L(YES)** = −4,600 × 0.65 = **−$2,990**
- **P&L(NO)** = +4,600 × 0.35 = **+$1,610**
- **Expected** = 0.35 × (−2,990) + 0.65 × 1,610 = 0 ✓
- **σ** = 4,600 × √(0.35 × 0.65) ≈ **$2,194**

**By asset class, if YES** (Vᵢβᵢ × 0.65):

| Class | Positions | P&L(YES) |
|---|---|---|
| Equities | SPY, QQQ, NVDA, XLE | −$3,445 |
| Rates | TLT | +$292.50 |
| Gold | GLD | +$260 |
| Credit | HYG | −$97.50 |
| **Total** | | **−$2,990** ✓ |

**Reading it:**
- About two-thirds of the bad outcome is already priced in.
- TLT and GLD are acting as natural hedges.
- NVDA is a third of the equity damage at a tenth of the book. Look-through raises that further, because QQQ also holds NVDA.

**Hand-off:** "how would I hedge that?" → `compare_hedges` with this scenario object (§12).
