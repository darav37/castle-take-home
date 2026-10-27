# Commodity releases and weather: WASDE, EIA inventories, Gulf hurricanes

Run 2026-10-23. Script: `scripts/commod_releases.py` (`fetch`, `score`). Raw numbers: `commod_releases.json`. No LLM calls, $0. DECISIONS #212.

: "have we trained up on weather events and WASDE reports and stuff like this?" Not before this.

## In plain words

**The data (all free):**
- **USDA WASDE:** 238 release dates from USDA's archive (ESMIS), 2008 on, released at 12:00 ET
- **EIA Weekly Petroleum Status Report:** Wednesdays 10:30 ET, and **EIA Weekly Natural Gas Storage**, Thursdays 10:30 ET, with the weekly figures from EIA's public history files
- **Gulf hurricanes:** NOAA's track database (HURDAT2), US Gulf landfalls at hurricane strength, 2007–2024
- the commodity ETFs' minute bars

**Size:** the move from 5 minutes before the release to 60 minutes after, compared with the same clock window the session before. That control turns "bigger than normal" into a measurement, not an assumption.

| Release | ETF | First hour vs a normal hour (RMS) | Moves > 2× normal (a normal day) | To the close vs normal |
|---|---|---|---|---|
| **WASDE** | Corn (CORN) | **4.6×** | **56%** (7%) | 3.2× |
| **WASDE** | Soybeans (SOYB) | **4.0×** | 44% (5%) | 2.7× |
| WASDE | Wheat (WEAT) | 2.1× | 27% (3%) | 2.2× |
| WASDE | Farm commodities (DBA) | 1.9× | 21% (4%) | 1.5× |
| EIA gas storage | Natural gas (UNG) | 1.65× | 19% (6%) | 1.3× |
| EIA crude stocks | Oil (USO) | 1.34× | 13% (6%) | 1.3× |

**Direction (EIA only).** Analysts' consensus isn't free, so the surprise here is the week's stock change minus the average change in the same week over the previous 5 years. A bigger build than the seasonal norm should push the price down.
- crude: right 53% of the time (63% on big moves), rank correlation −0.13 (the right sign)
- gas: 52% (65% on big moves), correlation −0.08
- 2026 falls to 34–37% on small samples
- with a seasonal proxy instead of the real consensus, direction is weak; the real surprise is what traders react to

**Hurricanes (25 Gulf landfalls, 2 sessions after landfall):**
- typical move about normal everywhere (RMS 1.06× for natural gas, 1.17× oil, 1.24× energy stocks, 0.67× homebuilders)
- no consistent direction (natural gas up 60% of the time, oil 40%)
- slightly fatter tails for insurers (1.5×) and gasoline (1.4×)
- **the storm is forecast days ahead, so landfall itself is mostly priced.** The tradeable information is in the forecast track, not the landfall

**What it means:**
- **WASDE days are real events for grains:** corn moves several times a normal hour, most of the time. That's the clearest scheduled single-asset event we've measured outside earnings.
- **EIA days are modestly bigger than normal,** with direction depending on the consensus surprise we don't have.
- **Hurricane landfalls don't add much** beyond what the forecasts already priced.

**In the product:** the natural-world expert's release calendar now carries "first hour vs a normal hour" for EIA petroleum (USO 1.3×), EIA gas (UNG 1.6×) and WASDE (corn 4.6×, soybeans 4.0×, wheat 2.1×).

**Next, if wanted:**
- **WASDE direction:** USDA's month-over-month revision to ending stocks as a free surprise proxy (trade estimates aren't free).
- **EIA direction:** the real consensus (paid), or Kalshi markets where they exist.
- **Hurricanes:** forecast-track changes (NOAA advisories) instead of landfall.

## Method notes

- **Thin ETFs** (CORN, SOYB, WEAT): the last trade that morning before the release and the last trade within 60 minutes after it, with the same rule for the control window. The CORN control shows 7% of windows above 2×, about what normal noise gives, so the rule isn't biased.
- **WASDE time:** the archive stamps "12:00Z", but releases are at 12:00 ET; dates kept, time set to noon ET.
- **EIA dates:** week ending Friday → Wednesday (petroleum) or Thursday (gas), one day later after a Monday holiday. Government-shutdown delays aren't modelled.
- **Hurricanes:** HURDAT2 landfall records within the US Gulf coast box (25–31.5°N, 81.5–98°W) at ≥ 64 kt; first Gulf landfall per storm.
