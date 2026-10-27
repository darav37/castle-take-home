# Plan B B2: event library v0

Run 2026-10-20 by `scripts/planb_b2.py` (calendar, events, responses) and `scripts/planb_b2_report.py` (this report's tables). Library code: `src/worsey/planb/library.py`. Data: `cache/pmrm/b2/` (`scheduled.csv`, `unscheduled.json`, `library.json`). BUILD_PLAN B2, PLAN_B §5.

**Stop here for the trainer:** BUILD_PLAN B2 step 4. Review the sample of event records below before B3 fits anything on them.

## In plain words

We built Plan B's event library: a list of dated events, each with how markets moved right after it.

- **1,306 scheduled economic releases since late 2003:** every Fed decision, CPI, jobs report, GDP and PCE release. For each we have how surprising it was, measured by how much the 2-year and 10-year Treasury yields and inflation expectations moved that day, plus the data surprise for CPI, jobs and PCE.
- **194 unscheduled events, 2004–2026,** about 20 in each of 10 categories: tariffs and trade, sanctions, wars and conflicts, elections and politics, regulation and court rulings, natural disasters, supply shocks, surprise central-bank moves, pandemics and health, and financial stress (bank failures, debt crises).
- **For every event:** the measured move of the B1 factors (market, rates, dollar, oil, credit and so on), of a fixed panel of 30 ETFs, and of the specific sectors or stocks it targeted beyond what the factors explain. Measured over a short window and over 5 sessions.

**Who did what:**
- **Claude (10 calls with web search, about $4.73):** found the events, dated them by their first public report, and filled in the fixed description schema with sources. It never saw or wrote any market move.
- **Code:**
  - checked every source link
  - checked every date against Wikipedia's dated news pages
  - assigned each event's market window
  - computed every number

**Checks on quality:**
- **Wikipedia date match:** 173 of 194 events (89%) match an entry on the stated day or within two days.
- **Sourcing:** 82 have cited sources (75 with working links). The other 112 come from the model's own knowledge, because it ran out of searches (6 per category). They are kept and flagged, and their dates are checked against Wikipedia.

**Known weaknesses for the review:**
1. **Selection leans to big events.** Severity: 118 major, 27 extreme, 45 moderate, 4 minor, despite the instruction to include low-impact ones. That will overstate typical impact (PLAN_B §5.4). B3 reports results by severity, and the full library should use systematic sources (GDELT) to correct it.
2. **The exact time is rarely known (15 of 194).** Without a time, the short window covers the report day's session and the next, so after-close news is captured (fixed during this run: the Hamas attack, reported after Friday's close, was first measured on the wrong day).
3. **"First report" is sometimes the event itself for things forecast in advance** (hurricane landfalls). The anticipation field marks them (`widely_expected`), and their responses are lower bounds.
4. **55 of 194 short windows also contain a scheduled macro release.** They are flagged (confounding, PLAN_B §5.4).
5. **Anticipation from prediction markets isn't recorded yet.** The model named a market for only 7 events. That comes in B5, from Kalshi ladders for releases since 2022.

## 1. Unscheduled events
| Category | Events | Cited sources | Working link | Wikipedia date match | Years | Macro release in short window |
|---|---|---|---|---|---|---|
| central_bank_unscheduled | 20 | 13 | 13 | 20 | 2005–2023 | 8 |
| conflict_war | 20 | 6 | 5 | 20 | 2004–2025 | 6 |
| disaster | 20 | 6 | 5 | 20 | 2004–2025 | 4 |
| elections_politics | 19 | 6 | 6 | 18 | 2004–2025 | 3 |
| financial_stress | 19 | 6 | 6 | 15 | 2005–2025 | 7 |
| health_pandemic | 20 | 6 | 6 | 17 | 2004–2025 | 5 |
| regulatory_court | 19 | 6 | 6 | 16 | 2004–2025 | 5 |
| sanctions | 18 | 7 | 7 | 14 | 2004–2026 | 6 |
| supply_shock | 19 | 6 | 6 | 16 | 2006–2025 | 6 |
| trade_tariffs | 20 | 20 | 15 | 17 | 2004–2026 | 5 |
| **All** | **194** | 82 | 75 | 173 | 2004–2026 | 55 |

**By period:** 2004–2010: 43, 2011–2015: 40, 2016–2020: 43, 2021–2026: 68

**Severity:** major: 118, moderate: 45, extreme: 27, minor: 4. **Anticipation:** partly_expected: 84, surprise: 73, widely_expected: 37. **Targeted tickers given:** 18. **Prediction market named:** 7.


## 2. Scheduled releases (from late 2003, when the factor series start)

Calendar since 2000: 247 FOMC meetings (federalreserve.gov), plus CPI, jobs and PCE first-publication dates and every GDP vintage (ALFRED). On 2022–2026 it matches P0b's calendar (FOMC 38/38, GDP 55/55, PCE 56/56), and it has the correct shutdown-delayed 2025 dates where the Kalshi close times had the original schedule.

Surprise measures:
- **Market-based (primary):** release-day changes in DGS2, DGS10 and the 5-year breakeven.
- **Data surprises (secondary):**
  - CPI, jobs and PCE: first release minus the trailing average of 12 first releases (naive).
  - GDP: the advance estimate minus the last GDPNow vintage before it (41 releases since 2016).

| Release | Records | From | sd of release-day 2y change (bp) | sd of SPY 1-day move | Data surprise available |
|---|---|---|---|---|---|
| cpi | 277 | 2003-09-16 | 6.6 | 1.22% | 277 |
| fomc | 199 | 2003-09-15 | 8.0 | 1.65% | 0 |
| gdp | 276 | 2003-09-26 | 4.7 | 1.05% | 41 |
| jobs | 277 | 2003-10-03 | 8.3 | 1.14% | 277 |
| pce | 277 | 2003-09-29 | 5.3 | 1.18% | 277 |


## 3. Review sample for the trainer (2 random events per category, seed 1729)

For each event, check: is it the right event, is the first-report date right, is the window right, and are the sectors right?

**ev0121 · Fed Sunday emergency cut to zero and restart of QE** (central_bank_unscheduled)
- First report: 2020-03-15 → short window 2020-03-13 close to 2020-03-17 close
- Region: United States · severity extreme · anticipation partly_expected · sectors broad_market · tickers -
- On a Sunday, the Fed cut the federal funds target range to 0–0.25% and announced at least $700 billion of asset purchases. It also announced coordinated action on dollar swap lines with five other central banks, two days before its scheduled meeting.
- Sources: [Federal Reserve Board](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm)
- Date check: Wikipedia 2020-03-15 (matched: canada, emergency, reserve, sunday)
- Measured short-window move: SPY -6.13%, TLT -0.62%, USO -17.96%, GLD +0.20%; beyond-factor: SPY +0.74%

**ev0019 · Fed unscheduled FOMC statement and discount rate cut** (central_bank_unscheduled)
- First report: 2007-08-17 → short window 2007-08-16 close to 2007-08-20 close
- Region: United States · severity major · anticipation surprise · sectors banks, financials, broad_market · tickers -
- Between scheduled meetings, the Federal Reserve cut the primary credit (discount) rate by 50 bp to 5.75% and lengthened loan terms. The FOMC also issued an unscheduled statement saying downside risks to growth had increased appreciably.
- Sources: [Federal Reserve Board](https://www.federalreserve.gov/newsevents/pressreleases/monetary20070817c.htm); [VoxEU/CEPR](https://cepr.org/voxeu/columns/federal-reserve-policy-actions-august-2007-answers-more-questions) (link failed); [CNBC](https://www.cnbc.com/2007/08/17/fed-cuts-discount-lending-rate-in-surprise-move.html)
- Date check: Wikipedia 2007-08-17 (matched: discount, federal, reserve)
- Measured short-window move: SPY +1.79%, TLT -0.34%, USO -0.07%, GLD +0.68%; beyond-factor: SPY -0.07%

**ev0162 · Hamas-led attack on southern Israel** (conflict_war)
- First report: 2023-10-06 → short window 2023-10-05 close to 2023-10-09 close (macro release in window)
- Region: Israel / Gaza · severity extreme · anticipation surprise · sectors defense, energy · tickers -
- Hamas fired a rocket barrage and sent gunmen across the Gaza border early on 7 October Israel time, which was late on 6 October ET. About 1,200 people were killed and hostages taken, and Israel declared war.
- Sources: none: LLM knowledge only
- Date check: Wikipedia 2023-10-06 (matched: attack, israel, palestinian)
- Measured short-window move: SPY +1.84%, TLT +1.11%, USO +4.24%, GLD +2.36%; beyond-factor: XLE +1.31%, ITA +3.95%

**ev0025 · Russia-Georgia war begins at Tskhinvali** (conflict_war)
- First report: 2008-08-07 → short window 2008-08-06 close to 2008-08-08 close
- Region: Georgia / South Ossetia · severity major · anticipation partly_expected · sectors energy, transport_shipping · tickers -
- On the evening of 7 August South Ossetia and Georgia accused each other of heavy artillery barrages on Tskhinvali. The fighting grew into a five-day war when Russian forces entered Georgia.
- Sources: [Congressional Research Service (EveryCRSReport)](https://www.everycrsreport.com/reports/RL34618.html); [ICRC Casebook (Human Rights Watch report)](https://casebook.icrc.org/case-study/georgiarussia-human-rights-watchs-report-conflict-south-ossetia) (link failed)
- Date check: Wikipedia 2008-08-07 (matched: georgia, ossetia, russia, south, tskhinvali)
- Measured short-window move: SPY +0.34%, TLT +1.91%, USO -2.87%, GLD -2.55%; beyond-factor: XLE -2.63%

**ev0145 · Hurricane Ian Florida landfall** (disaster)
- First report: 2022-09-28 → short window 2022-09-27 close to 2022-09-29 close (macro release in window)
- Region: US (Southwest Florida) · severity major · anticipation widely_expected · sectors insurance, real_estate, utilities, homebuilders · tickers -
- Ian made landfall near Cayo Costa, Florida, on 28 Sept 2022 as a Category 4 hurricane, causing a major storm surge and flooding. Not verified by search.
- Sources: none: LLM knowledge only
- Date check: Wikipedia 2022-09-28 (matched: florida, hurricane, landfall, state)
- Measured short-window move: SPY -0.16%, TLT +2.81%, USO +3.65%, GLD +2.06%

**ev0023 · Sichuan (Wenchuan) earthquake** (disaster)
- First report: 2008-05-12 → short window 2008-05-09 close to 2008-05-13 close
- Region: Sichuan, China · severity extreme · anticipation surprise · sectors industrials, materials, insurance, real_estate · tickers -
- A magnitude ~7.9–8.0 earthquake struck Wenchuan County, Sichuan, at 14:28 China time (06:28 UTC, about 02:28 US Eastern) on 12 May 2008. It killed about 69,000 people, with about 18,000 more missing.
- Sources: [University of Illinois (compilation of BBC, NPR, PBS, NYT coverage)](http://www.aems.illinois.edu/resources/currentevents/sichuanearthquake.html); [NASA Earth Observatory](https://science.nasa.gov/earth/earth-observatory/earthquake-near-chengdu-china-19947/)
- Date check: Wikipedia 2008-05-12 (matched: china, earthquake, sichuan)
- Measured short-window move: SPY +1.14%, TLT -1.29%, USO -0.40%, GLD -2.32%

**ev0174 · Trump wins 2024 US presidential election** (elections_politics)
- First report: 2024-11-06 01:46 ET → short window 2024-11-05 close to 2024-11-06 close
- Region: United States · severity major · anticipation partly_expected · sectors - · tickers -
- Fox News projected Trump the winner at 1:46 a.m. ET on Nov 6, 2024. The AP made its call at 5:34 a.m. ET after declaring Wisconsin for Trump.
- Sources: [FOX 5 DC](https://www.fox5dc.com/news/live-updates-2024-election-results-polls-close-kamala-harris-donald-trump.amp); [PBS NewsHour/AP](https://www.pbs.org/newshour/politics/why-ap-called-wisconsin-and-the-white-house-for-donald-trump-2)
- Date check: Wikipedia 2024-11-06 (matched: donald, election, harris, kamala, party, presidential)
- Measured short-window move: SPY +2.49%, TLT -2.74%, USO -0.25%, GLD -3.04%

**ev0005 · Bush re-elected over Kerry** (elections_politics)
- First report: 2004-11-03 → short window 2004-11-02 close to 2004-11-04 close
- Region: United States · severity major · anticipation partly_expected · sectors - · tickers -
- George W. Bush won a second term after Ohio became the deciding state, and John Kerry conceded on the morning of Nov 3. Date and time not verified by search.
- Sources: none: LLM knowledge only
- Date check: Wikipedia 2004-11-03 (matched: democratic, elected, kerry, party, republican)
- Measured short-window move: SPY +2.64%, TLT +0.15%, USO +0.00%, GLD +0.00%

**ev0166 · Hong Kong court orders liquidation of China Evergrande** (financial_stress)
- First report: 2024-01-29 → short window 2024-01-26 close to 2024-01-30 close
- Region: China / Hong Kong · severity moderate · anticipation widely_expected · sectors real_estate · tickers -
- A Hong Kong judge ordered the winding up of China Evergrande after repeated adjournments in a creditor petition. The developer had defaulted on its offshore debt in late 2021.
- Sources: none: LLM knowledge only
- Date check: Wikipedia 2024-01-29 (matched: court, evergrande, group, liquidation, orders)
- Measured short-window move: SPY +0.71%, TLT +2.07%, USO -0.40%, GLD +0.84%

**ev0010 · Refco discloses $430 million hidden debt, then files for bankruptcy** (financial_stress)
- First report: 2005-10-10 → short window 2005-10-07 close to 2005-10-11 close
- Region: United States · severity moderate · anticipation surprise · sectors financials · tickers RFX
- On Oct. 10, 2005 Refco disclosed that its CEO had concealed about $430 million of debt from the company's books and auditors. The company filed for Chapter 11 on Oct. 17, 2005.
- Sources: [Associated Press via News On 6](https://www.newson6.com/story/5e368c972f69d76f620a3ea5/refco-to-sell-core-business-for-768-million-files-for-bankruptcy); [Wikipedia](https://en.wikipedia.org/wiki/Refco)
- Date check: Wikipedia 2005-10-10 (matched: million, thomas)
- Measured short-window move: SPY -0.99%, TLT -0.27%, USO +0.00%, GLD +0.06%

**ev0135 · South Africa announces Omicron (B.1.1.529) variant** (health_pandemic)
- First report: 2021-11-25 → short window 2021-11-24 close to 2021-11-29 close
- Region: South Africa / global · severity major · anticipation surprise · sectors airlines, broad_market, health_care · tickers -
- South African scientists announced detection of a heavily mutated SARS-CoV-2 variant, B.1.1.529. WHO designated it a variant of concern, named Omicron, the next day. The date comes from memory and was not confirmed by search.
- Sources: none: LLM knowledge only
- Date check: Wikipedia 2021-11-25 (matched: africa, botswana, omicron, south, variant)
- Measured short-window move: SPY -1.03%, TLT +1.70%, USO -10.22%, GLD -0.30%; beyond-factor: JETS -3.31%, SPY +0.24%

**ev0016 · CDC announces XDR-TB traveller Andrew Speaker case** (health_pandemic)
- First report: 2007-05-29 → short window 2007-05-25 close to 2007-05-30 close
- Region: United States / Europe · severity minor · anticipation surprise · sectors airlines, health_care · tickers -
- CDC publicly disclosed an American man with extensively drug-resistant tuberculosis who had flown on transatlantic commercial flights. It was the first federal isolation order for a US citizen in decades. Date and details are from memory and were not confirmed by search.
- Sources: none: LLM knowledge only
- Date check: no Wikipedia match within 2 days
- Measured short-window move: SPY +1.18%, TLT +0.20%, USO -2.61%, GLD -0.34%; beyond-factor: JETS -1.29%

**ev0170 · Judge Mehta rules Google illegally maintained search monopoly** (regulatory_court)
- First report: 2024-08-05 → short window 2024-08-02 close to 2024-08-06 close
- Region: United States · severity major · anticipation partly_expected · sectors tech, communications · tickers GOOGL, GOOG
- A federal judge ruled that Google violated Section 2 of the Sherman Act by maintaining monopolies in general search services and general search text advertising through exclusive default distribution deals.
- Sources: [Al Jazeera](https://www.aljazeera.com/economy/2024/8/6/google-has-illegal-monopoly-over-internet-search-us-judge-rules); [Goodwin Procter](https://www.goodwinlaw.com/en/insights/publications/2024/08/alerts-technology-antc-google-is-an-illegal-monopoly-federal-court-rules)
- Date check: Wikipedia 2024-08-05 (matched: court, district, google, illegally, judge, mehta)
- Measured short-window move: SPY -2.02%, TLT -1.72%, USO -1.28%, GLD -2.06%; beyond-factor: GOOG -1.48%, GOOGL -2.15%

**ev0001 · EU Commission antitrust decision against Microsoft (media player / server interoperability)** (regulatory_court)
- First report: 2004-03-24 → short window 2004-03-23 close to 2004-03-25 close (macro release in window)
- Region: European Union · severity major · anticipation widely_expected · sectors tech, software · tickers MSFT
- The European Commission announced it found Microsoft abused its dominant position in PC operating systems, imposing a record fine of about 497 million euros and remedies on bundling Windows Media Player and on server interoperability.
- Sources: none: LLM knowledge only
- Date check: no Wikipedia match within 2 days
- Measured short-window move: SPY +1.35%, TLT -0.44%, USO +0.00%, GLD +0.00%; beyond-factor: MSFT +3.07%

**ev0192 · Executive Order 14404 creates new Cuba sanctions authority** (sanctions)
- First report: 2026-05-01 → short window 2026-04-30 close to 2026-05-04 close
- Region: Cuba · severity moderate · anticipation partly_expected · sectors broad_market, metals_mining, defense, banks · tickers -
- Executive Order 14404, dated May 1, 2026, authorizes blocking sanctions on persons operating in specified sectors of the Cuban economy and secondary sanctions on foreign financial institutions. It builds on the national emergency declared in EO 14380 of January 29, 2026. The date is the order date; the first press report may be earlier.
- Sources: [Holland & Knight](https://www.hklaw.com/en/insights/publications/2026/10/ofac-issues-new-cuba-sanctions-program-and-amends-the-cacr); [Mondaq](https://www.mondaq.com/export-controls-trade-investment-sanctions/1856302/ofac-expands-cuba-sanctions-and-updates-iran-sanctions)
- Date check: no Wikipedia match within 2 days
- Measured short-window move: SPY -0.09%, TLT -0.41%, USO +0.35%, GLD -2.11%; beyond-factor: ITA -1.11%, SPY +0.14%

**ev0002 · US lifts most sanctions on Libya** (sanctions)
- First report: 2004-04-23 → short window 2004-04-22 close to 2004-04-26 close
- Region: Libya · severity moderate · anticipation widely_expected · sectors energy, broad_market · tickers -
- President Bush announced the lifting of most US economic sanctions on Libya after it renounced weapons of mass destruction programs, allowing US trade and oil-sector activity to resume. Source URLs were not retrieved in this session, so the date is from background knowledge and is unconfirmed.
- Sources: none: LLM knowledge only
- Date check: no Wikipedia match within 2 days
- Measured short-window move: SPY -0.04%, TLT -0.40%, USO +0.00%, GLD +0.00%; beyond-factor: XLE +0.15%, SPY +0.13%

**ev0179 · DRC suspends cobalt exports** (supply_shock)
- First report: 2025-02-22 → short window 2025-02-21 close to 2025-02-25 close
- Region: Democratic Republic of Congo · severity major · anticipation surprise · sectors metals_mining, materials, autos · tickers -
- The DRC's mineral regulator ARECOMS announced a four-month suspension of cobalt exports. The DRC is the world's largest cobalt producer. Details are from memory and were not verified by search.
- Sources: none: LLM knowledge only
- Date check: Wikipedia 2025-02-22 (matched: congo, democratic, republic)
- Measured short-window move: SPY -0.95%, TLT +2.02%, USO -1.62%, GLD -0.78%; beyond-factor: CARZ -1.34%

**ev0014 · BP begins shutdown of Prudhoe Bay oil field over pipeline corrosion** (supply_shock)
- First report: 2006-08-07 → short window 2006-08-04 close to 2006-08-08 close (macro release in window)
- Region: United States (Alaska) · severity major · anticipation surprise · sectors energy · tickers BP
- BP announced it had begun a phased shutdown of the Prudhoe Bay field after finding severe corrosion and a small spill in an oil transit line. The company estimated the shutdown would cut Alaska North Slope output by about 400,000 barrels per day.
- Sources: [CNNMoney](https://money.cnn.com/2006/08/07/news/international/oil_alaska/index.htm) (link failed); [NPR](https://www.npr.org/2006/08/08/5626067/bp-pipeline-maintenance-program-draws-scrutiny)
- Date check: Wikipedia 2006-08-07 (matched: alaska, corrosion, pipeline, prudhoe)
- Measured short-window move: SPY -0.62%, TLT -0.16%, USO +2.13%, GLD -0.48%; beyond-factor: XLE +0.84%

**ev0175 · Trump threatens 25% tariffs on Canada and Mexico and 10% on China** (trade_tariffs)
- First report: 2024-11-25 → short window 2024-11-22 close to 2024-11-26 close
- Region: North America / China · severity major · anticipation partly_expected · sectors broad_market, autos, energy · tickers -
- The president-elect said on social media that he would impose a 25% tariff on all products from Canada and Mexico, and an additional 10% tariff on China, on his first day in office. He linked the move to migration and drug trafficking.
- Sources: [Wikipedia](https://en.wikipedia.org/wiki/Tariffs_in_the_second_Trump_administration)
- Date check: Wikipedia 2024-11-25 (matched: donald, trump)
- Measured short-window move: SPY +0.86%, TLT +2.19%, USO -3.56%, GLD -2.76%; beyond-factor: XLE -1.78%, CARZ -0.28%, SPY +0.03%

**ev0033 · Obama approves tariffs on Chinese tire imports** (trade_tariffs)
- First report: 2009-09-11 → short window 2009-09-10 close to 2009-09-14 close
- Region: United States / China · severity moderate · anticipation partly_expected · sectors autos, consumer_discretionary · tickers -
- President Obama announced a safeguard tariff of up to 35% on Chinese passenger and light-truck tire imports, following a US International Trade Commission recommendation. China took the matter to the WTO as DS399.
- Sources: [World Trade Organization](https://www.wto.org/english/tratop_e/dispu_e/cases_e/ds399_e.htm)
- Date check: no Wikipedia match within 2 days
- Measured short-window move: SPY +0.47%, TLT -0.45%, USO -4.44%, GLD +0.27%; beyond-factor: CARZ -0.99%

## 4. Events with no Wikipedia match within 2 days (check the date)

- ev0001 · 2004-03-24 · EU Commission antitrust decision against Microsoft (media player / server interoperability) (regulatory_court; llm_knowledge_only)
- ev0002 · 2004-04-23 · US lifts most sanctions on Libya (sanctions; llm_knowledge_only)
- ev0016 · 2007-05-29 · CDC announces XDR-TB traveller Andrew Speaker case (health_pandemic; llm_knowledge_only)
- ev0021 · 2007-10-25 · US designates IRGC, Bank Melli and Bank Mellat (sanctions; llm_knowledge_only)
- ev0032 · 2009-04-21 · CDC reports novel swine-origin H1N1 influenza (2009 pandemic) (health_pandemic; cited)
- ev0033 · 2009-09-11 · Obama approves tariffs on Chinese tire imports (trade_tariffs; cited)
- ev0038 · 2010-05-06 · US equity 'Flash Crash' (financial_stress; cited)
- ev0050 · 2011-10-12 · Thailand floods halt Western Digital hard drive production (supply_shock; cited)
- ev0070 · 2014-11-27 · OPEC decides in Vienna to leave output ceiling unchanged (supply_shock; cited)
- ev0073 · 2015-01-15 · Swiss National Bank abandons euro floor (financial_stress; llm_knowledge_only)
- ev0074 · 2015-02-04 · FCC chairman proposes Title II net neutrality rules (regulatory_court; llm_knowledge_only)
- ev0105 · 2018-12-22 · US partial government shutdown begins (2018–19) (elections_politics; cited)
- ev0118 · 2020-03-06 · OPEC+ talks collapse in Vienna over deeper output cuts (supply_shock; llm_knowledge_only)
- ev0129 · 2021-03-26 · Archegos Capital margin-call liquidations (financial_stress; llm_knowledge_only)
- ev0147 · 2022-10-07 · US imposes sweeping advanced chip export controls on China (trade_tariffs; cited)
- ev0149 · 2022-11-02 · CoinDesk reveals Alameda balance sheet, leading to FTX collapse (financial_stress; llm_knowledge_only)
- ev0156 · 2023-06-05 · SEC sues Binance and founder Changpeng Zhao (regulatory_court; cited)
- ev0158 · 2023-07-03 · China announces gallium and germanium export controls (trade_tariffs; cited)
- ev0167 · 2024-03-25 · H5N1 bird flu confirmed in US dairy cattle (health_pandemic; llm_knowledge_only)
- ev0189 · 2025-10-22 · OFAC blocks Rosneft and Lukoil (sanctions; cited)
- ev0192 · 2026-05-01 · Executive Order 14404 creates new Cuba sanctions authority (sanctions; cited)
