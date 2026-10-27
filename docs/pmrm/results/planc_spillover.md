# Sector layer, step 2: earnings spillover to sector peers, and sector-tagged news

Run 2026-10-24. Script: `scripts/planc_spillover.py`. Raw numbers: `planc_spillover.json`. No LLM calls, $0. DECISIONS #216.

## In plain words

**The question.** When a company reports earnings, how do its sector peers move? For example, NVDA reports: what happens to the other chip stocks?

**The design:**
- **Peers:** the other S&P 500 stocks assigned the same best sector in step 1 (#215), excluding the reporter. The sector ETF itself would be contaminated: NVDA is about a fifth of SMH.
- **Peer move:** each peer's move beyond the 13 macro factors on the reporter's reaction day, averaged, and scaled so 1 = an ordinary day for that peer group.
- **Data:** 18,446 reports, 2011–2026; "leaders" are the 3 largest names in each sector by S&P 500 weight (2,675 reports).

| | All reporters | Leaders |
|---|---|---|
| Peer move (RMS, × an ordinary day) | 1.21 | 1.29 |
| Peer moves > 2× normal | 9% (normal 4.6%) | 10% |
| Direction from the reporter's **EPS surprise** | 51% | 51% |
| Direction from the reporter's **own move** | **58%** (63% after big moves) | **57%** (63%) |
| Peers' move as a fraction of the reporter's (noise units) | 0.07 | 0.05 |

The same direction share holds in every period (2026: 58–61%).

**By sector (leaders):**

| Peers move this fraction of the leader's move | Sectors |
|---|---|
| Largest | regional banks 0.25 (same way 68%), transports 0.16, energy 0.13, semis 0.12 (63%), materials 0.10 |
| About none | software, biotech, insurers, homebuilders, retail |
| A slight reversal | medtech |

**What it means:**
- Spillover is real but partial: peers move a fraction of the reporter's move, in its direction about 3 times in 5.
- The headline EPS surprise says nothing about peers. The reporter's actual reaction does.

**Sector-tagged news.** Benzinga rarely tags the sector ETFs themselves (SMH about 230 items in 2024, mostly company news that mentions it), so there's no clean sector-headline set from that source. For user scenarios the direction comes from the user ("a big positive headline for semis"); what's needed is the size of a big sector day and how it reaches each holding, both measured from the sector factors (#217).
