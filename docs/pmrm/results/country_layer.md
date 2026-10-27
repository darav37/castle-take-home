# The country layer (DECISIONS #238)

Task: the design brief (sanity check approved by the trainer: countries get their own slot; home country for foreign stocks, best fit only for the rest). Code: `live.country_loadings()`, `live.country_moves()`, `discovery.impact()` / `combine_legs()`, `live.price(country_news=…)`; packs `planb/countries.json` (`scripts/country_pack.py`) and `planb/domicile.json`.

## In plain words

- **The live case is fixed.** "Brazil invades Peru" on a book with $30k of EWZ: the label now carries Brazil − and Peru − (severity major), and EWZ moves **−$934** (−3.1%) instead of −$60 (−0.2%). In the one-event path with `country_news`, EWZ moves −3.1% instead of −0.2%.
- **Foreign stocks are better explained.** With their home country's factor, the median held-out share of daily moves explained rises by **13.7%** (from a median 30% with the macro factors alone), for 41 foreign stocks and ADRs; 36 pass the 2% rule. The sector layer added about 9 points.
- **Nothing else moves.** The trainer's example book and the normal book hold nothing with a country loading, so no watchlist candidate changed; the weird book's EWZ maps to Brazil, and no current label is about Brazil.

## How it works

- **Country factors:** each of 31 country ETFs' daily move beyond the 13 macro factors (Brazil EWZ, Mexico EWW, Argentina ARGT, Chile ECH, Peru EPU, Canada EWC, Japan EWJ, China MCHI, Taiwan EWT, South Korea EWY, India INDA, the UK EWU, Germany EWG, France EWQ, Israel EIS, Saudi Arabia KSA, Turkey TUR, …; 2016–2026). Typical (median) two-day move about 0.55× its own noise, large (90th percentile) about 1.5×. Broad ETFs (EEM, EFA, VGK) are not countries.
- **Who gets a country (its own slot, beside the sector):**
  - a country ETF: its own country
  - a foreign stock or ADR: its home country (`domicile.json`, 225 names), kept if it adds more than 2% of the stock's daily moves
  - anything else: the best-fitting country, kept only if it adds more than 2% **held out**, beyond the macro factors **and** its sector (so a US oil stock doesn't pick up "Canada"). In the checks, AAPL, XOM, LVS, WYNN and SPY get no country.
- **Legs:** labels carry `countries` only for the countries an event is about or directly hits, each with its direction (the same rule as company legs). The watchlist uses the typical move, or the large end for a major or extreme event; the analyst passes `country_news` (typical or big) in the one-event path. With a sector, company and country leg on one holding, the largest is kept, never the sum. An option's underlier gets the same move as its stock (#235).

## Who maps where (live data)

| holding | country | how | share of daily moves added |
|---|---|---|---|
| EWZ | Brazil | country ETF | 64.5% |
| EPU | Peru | country ETF | 32.7% |
| VALE | Brazil | home country | 9.7% |
| ITUB | Brazil | home country | 52.9% |
| UBS | Switzerland | home country | 3.6% |
| BHP | Australia | home country | 5.0% |
| TSM | Taiwan | home country | 6.1% |
| BABA | China | home country | 34.4% |
| AAPL | – | none | – |
| XOM | – | none | – |
| YUMC | China | home country | 16.0% |
| LVS | – | none | – |
| WYNN | – | none | – |
| SPY | – | none | – |
| GLD | – | none | – |
| URA | – | none | – |

## Held-out test, foreign stocks (fit on the first 70% of two years, scored on the rest)

- Home-country rule (as built): median gain 13.7%; 36 of 41 pass the 2% rule.
- Best-fit rule (step 1, for comparison): 39 of 44 kept, but 4 picked the wrong country (UBS → Germany, BHP → Peru, UMC → China, LVS → Hong Kong); hence home country for foreign stocks.
- Some gain is mechanical: big names sit inside their own country ETF (ITUB, VALE, PBR in EWZ; BABA in MCHI), as with NVDA in SMH.

## Labels relabelled (26 now carry a country leg; $0.10; only the countries field changed)

- Bank of Japan increases interest rates by 50+ bps after the October 2026 meeting
- France declares state of emergency by December 31, 2026?
- France declares state of emergency by October 31, 2026?
- France declares state of siege (martial law) by December 31, 2026?
- France declares state of siege (martial law) by November 15, 2026?
- France declares state of siege (martial law) by October 31, 2026?
- Friedrich Merz out as Chancellor of Germany before December 31, 2026?
- Friedrich Merz out as Chancellor of Germany before October 31, 2026?
- Israeli airspace closed by December 31, 2026?
- Israeli airspace closed by October 31, 2026?
- Kim Jong Un out as Supreme Leader of North Korea by December 31, 2026?
- Macron out by December 31, 2026?
- Military action against Saudi Oil Pipeline (East-West) by November 30?
- Military action against Saudi Oil Pipeline (East-West) by October 31?
- Saudi-Houthi ceasefire agreement announced by December 31?
- Saudi-Houthi ceasefire agreement announced by October 31?
- U.S. agrees to a new trade deal with "India" before 2027?
- U.S. agrees to a new trade deal with "South Korea" before 2027?
- US-Canada diplomatic agreement to lower tariffs by October 31?
- Will Bank of Canada hike rates by 25bps at their October 2026 meeting?
- Will China invade Taiwan by end of 2026?
- Will Kim Jong Un be the next leader out before 2027?
- Will Saudi airspace be closed by December 31?
- Will Saudi airspace be closed by October 31?
- Will the Bank of Canada increase the target for the overnight rate by 50+ bps at
- Xi Jinping out before 2027?

Iran and Russia have no tradable country ETF, so they aren't countries in the pack: Iran markets keep their factor and sector legs only.

## Limits

- Broad multi-country ETFs (EEM, EFA, VGK) get no country leg; they still carry the EM factor.
- The home-country table is curated (225 names); an ADR outside it falls to best fit, which can pick the wrong country (the held-out test limits the damage).
- Country legs in the watchlist size by the label's severity; sector legs don't (they use the typical move). A small inconsistency, kept because a country leg is reserved for events about that country.
