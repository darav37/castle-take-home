# Plan B B1: the broad factor model

Run 2026-10-20 by `scripts/planb_b1.py` (no LLM calls). Model code: `src/worsey/planb/factors.py` (tests: `tests/test_planb_factors.py`). Raw numbers: `b1.json`. Saved model for B3: `cache/pmrm/b1/model.pkl`. BUILD_PLAN B1, PLAN_B §2, §5.2, DECISIONS #155.

## In plain words

We built the "coordinate system" Plan B uses to describe what an event does to markets. About 100 ETFs (US sectors and industries, commodities, currencies, countries, bonds) move together in a handful of recurring patterns: the overall market, interest rates, the dollar, oil, credit, small vs large, value vs growth, and a few more.

- **How much it captures.** On days it never saw (2021–2026), the best version explains **61% of the typical ETF's daily move** from those few patterns.
- **Which version wins.** The best version is a **hybrid**: named factors you can explain (market, 2- and 10-year yields, dollar, oil, gold, credit, size, value) plus 5 statistical factors that pick up what the named ones miss. It beats the named-only (57%) and statistical-only (54%) versions.
- **Stability.** The patterns are stable: loadings fitted on 2003–2010 work as well on 2021–2026 as loadings fitted on 2011–2020.
- **Individual stocks.** Every stock (the 435 current S&P 500 members with enough history, the Study B pools, the demo books) is now written as its exposure to these patterns plus its own part.
  - **Typical S&P 500 stock:** the patterns explain about a third of its daily move.
  - **Diversified books:** up to 94% (the demo `balanced` book).
  - **Why it matters:** an event's effect splits into "through the factors" (any book) and "specific to the stocks it hits" (B2 measures that part directly).

## 1. Setup

- **Panel:** 96 ETFs from the B0 universe with daily history from 2011 and ≥ 97% coverage. They exclude the 9 used to build the named factors (SPY, UUP, USO, GLD, HYG, IEF, IWM, VTV, VUG). Daily total returns (split-adjusted closes plus cash dividends) on consecutive XNYS sessions.
- **Periods:** train 2011–2018 (2,012 days), validation 2019–2020 (505 days, used only to choose the number of factors), test 2021-01 to 2026-10-16 (1,454 days).
- **Versions (PLAN_B §2):**
  - **latent:** principal factors of the standardized returns; each day's factor values estimated by GLS from that day's assets.
  - **anchored:** 9 observed factors, with each asset's loadings by regression in the training window. The factors are market (SPY), 2-year and 10-year Treasury yield changes (FRED DGS2, DGS10), dollar (UUP), oil (USO), gold (GLD), credit (HYG − IEF), size (IWM − SPY) and value (VTV − VUG).
  - **hybrid:** the anchored core plus latent factors of what it leaves unexplained.
- **Scoring (no version grades itself):** on each test day, latent factors are estimated from a random half of the assets and used to predict the other half (both ways). Anchored factors are observed returns of proxies excluded from the scored set. Score = pooled R² of standardized returns.

## 2. Results

**Number of factors (validation, cross-asset R²):**
- **Latent:** k = 1: 53%, 3: 59%, 5: 63%, 8: 65.5%, 9: 65.8%, 11: 66%, then falls (13: 55%, 15: 39%: overfitting). Chosen: **k = 9** (ties within 0.2 points go to the smaller k).
- **Hybrid:** anchored only 65.3%; + 1 latent 67.5%; + 5 latent 69.7%; + 6 latent 69.9%; then falls. Chosen: **k2 = 5**.

**Test R² (2021-01 to 2026-10, 1,454 days):**

| Version | Factors | Test R² |
|---|---|---|
| Latent | 9 | 54.2% |
| Anchored | 9 named | 57.2% |
| **Hybrid** | 9 named + 5 latent | **60.8%** |

**By asset group** (test R²; latent | anchored). The latent factors carry the equity structure; the named ones carry bonds, currencies and commodities:

| Group | n | Latent | Anchored |
|---|---|---|---|
| US sectors | 9 | 65% | 65% |
| US industries | 39 | 60% | 56% |
| Country equity | 21 | 52% | 44% |
| Rates and credit | 8 | 64% | 80% |
| Currencies | 7 | 43% | 62% |
| Commodities | 10 | 24% | 39% |

## 3. Stability

**Rolling 3-year refits vs the 2011–2020 fit.** Latent subspace similarity = mean squared cosine of the principal angles (1 = same, rotation-free). Anchored loadings are correlated across assets, per factor.

| Window | Latent (9) | Latent (top 3) | Anchored loadings: market / 2y / 10y / dollar / oil / credit / value |
|---|---|---|---|
| 2011–13 | 0.87 | 0.94 | 0.97 / 0.91 / 0.98 / 0.96 / 0.93 / 0.87 / 0.85 |
| 2014–16 | 0.90 | 0.95 | 0.97 / 0.89 / 0.95 / 0.97 / 0.97 / 0.70 / 0.88 |
| 2017–19 | 0.89 | 0.95 | 0.96 / 0.89 / 0.96 / 0.96 / 0.97 / 0.67 / 0.94 |
| 2020–22 | 0.89 | 0.87 | 0.95 / 0.87 / 0.96 / 0.95 / 0.98 / 0.81 / 0.93 |
| 2023–26 | 0.73 | 0.82 | 0.92 / 0.81 / 0.88 / 0.95 / 0.93 / 0.55 / 0.73 |

The named factors' loadings are very stable (0.8–0.98), except credit and value. Those drift, most in 2023–26, the most out-of-sample window.

**2003 check (38 ETFs with history from 2003)**: loadings fitted on 2003–2010 vs 2011–2020, both scored on 2021–2026.

| k | 2003–10 loadings | 2011–20 loadings | Subspace similarity |
|---|---|---|---|
| 1 | 39.0% | 38.2% | 1.00 |
| 3 | 50.3% | 49.0% | 0.85 |
| 5 | 51.9% | 53.6% | 0.80 |
| 9 | 32.3% | 7.8% | 0.76 |

At 1–5 factors the structure is as good fitted on 2003–2010 as on 2011–2020. At 9 factors, estimating them from 19-asset halves is too noisy (a small-sample artifact of this check, not instability). **Long training histories (2003+) are usable for B3.**

## 4. Stocks and books in the factor space

Each name's loadings come from regression on the factor series over 2011–2020, scored on 2021–2026 (time-series R² using that day's factor values, which come from ETFs only).

| Group | Names | Latent test R² (median) | Anchored test R² (median) |
|---|---|---|---|
| Current S&P 500 members | 435 | 31.7% | 25.3% |
| Study B pools | 28 | 42.4% | 34.7% |
| Later-listed ETFs | 27 | 60.1% | 59.4% |
| Demo books | 11 | 69.2% | 86.3% |

**Pool buckets** (latent | anchored):
- energy 58% | 46%
- big tech 46% | 52%
- rate-sensitive 48% | 36%
- travel and consumer 43% | 36%
- defensives 28% | 16%
- placebo 20% | −4%

**Demo books:** `balanced` 78% | 94%; `policy_exposed` 25% | 18%.

**Caveats:**
- **Survivorship:** the S&P 500 layer uses today's members (SPY holdings). Names delisted or acquired since 2011 are missing, which slightly flatters stock-level fit.
- **Excluded names:** 9 of the 510 requested names lacked enough history.

## 5. What B3 uses

- **Factor space:** the hybrid (9 named + 5 latent), with the latent-only and anchored-only versions kept for the head-to-head (PLAN_B §4).
- **Saved to `cache/pmrm/b1/model.pkl`:**
  - factor series from 2003 for every version
  - ETF loadings
  - every stock's loadings and standardization
- **Event responses (B2) are measured in this space.** An event's effect on any portfolio is the portfolio's loadings × the event's factor response, plus the measured beyond-factor move of the names it targets.
