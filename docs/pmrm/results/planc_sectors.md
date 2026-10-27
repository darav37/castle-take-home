# Sector layer, step 1: sector factors on top of the macro factors

Run 2026-10-24. Script: `scripts/planc_sectors.py`. Raw numbers: `planc_sectors.json`. No LLM calls, $0. DECISIONS #215.

: "sectors move together... I almost think we should be looking at headlines and looking at how each sector moves and then create a factor model for that, or at least just add factors on top."

## In plain words

**The idea.** The 13 factors are macro: stocks, rates, the dollar, oil, gold, credit, size, value, EM, healthcare, farm commodities, utilities vs energy. Sector co-movement (semis, regional banks, defense, homebuilders…) was being lumped into each stock's own noise.

**The sector factors.** 19 sector ETFs, each one's daily move beyond the 13 macro factors:

| Group | Sectors |
|---|---|
| Tech | semis, software, internet |
| Financials | regional banks, insurers |
| Healthcare | biotech, medtech |
| Energy and materials | energy stocks, materials, metals and mining |
| Industrial | industrials, defense, homebuilders, transports |
| Consumer | consumer discretionary, retail, staples |
| Rate-sensitive | utilities, real estate |

**The test (held out, as for the macro factors):**
- loadings fitted on 2019–2020 and frozen
- the share of each name's daily moves explained (R²) on 2021 to Oct 2026, days the model never saw
- 456 current S&P 500 stocks and the 77 research ETFs that aren't themselves the sector ETFs

| | Macro only | + all 19 sector factors | **+ each name's best one sector** |
|---|---|---|---|
| Median stock R² (2021–26) | 24.2% | 31.0% | **33.3%** |
| Stocks better / worse | | 69% better | **65% better / 9% worse** |
| Median ETF R² | 61.0% | 61.4% | 62.3% |

- **"Best one sector":** the sector factor that adds most on the fit years, used only if it adds more than 2 points there; 75% of stocks get one.
- **By GICS sector** (median stock R², macro → macro + best sector): real estate 24% → 36%, utilities 29% → 52%, staples 7% → 20%, materials 28% → 37%, industrials 26% → 35%, financials 42% → 47%, tech 30% → 34%. Energy, healthcare, communications and consumer discretionary gain 0–3 points.
- **Examples:** Boeing 4% → 34% (defense), KeyCorp 50% → 75% (regional banks), AMD 46% → 59% and NVDA 62% → 74% (semis), XOM 67% → 80% (energy stocks), Home Depot 18% → 23%.
- **ETFs:** most of the 96 are countries and commodities, which rightly get no sector (21% get one). Gold miners lose (64% → 58%): the fit-year pick didn't hold.
- **Using all 19 sector factors for every name overfits:** some names get worse on the held-out years. One sector per name is the right size.

**What it means:**
- **There's a real missing dimension:** a third more of a typical stock's daily moves is explained once its sector is in.
- That's the base for sector headlines: a semis headline is a shock to the semis factor, reaching every holding through its semis loading (directly, or through ETFs like QQQ).

**In the live model (two-stage, so nothing tested changes):**
- each holding keeps its 13 macro loadings exactly as before
- separately, its best sector (picked on its own 2-year window, if it adds more than 2 points) and its loading on that sector's factor, fitted on what the macro factors leave unexplained
- used by sector headline legs and the sector risk rows (step 3)

## Method notes

- **Sector factor:** the ETF's return minus its fitted macro part (loadings on the fit years), ÷ its residual sd there.
- **Sector ETFs themselves** are excluded from the medians, since they explain themselves (SMH 75% → 100%).
- **A stock that dominates its sector ETF** (NVDA about 20% of SMH) gains mechanically; the median over 456 stocks is robust to a few of those.
