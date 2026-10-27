# Attribution and licenses for `knowledge/`

`knowledge/options/corpus.jsonl` is a searchable reference corpus for Worsey's experts, built by
`scripts/build_options_corpus.py`. Every chunk records its source, URL and license, and replies that use a chunk cite it.

| Source | License | Notes |
|---|---|---|
| Wikipedia articles (listed in the build script) | **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/) | Text © Wikipedia contributors. Changes: plain-text extraction, "See also"/"References" removed, split into chunks. This file and any adaptation of these chunks are shared under CC BY-SA 4.0. |
| SEC Investor.gov (*Investor Bulletin: An Introduction to Options*; glossary entries) | US government work, public domain in the US | Courtesy of the U.S. Securities and Exchange Commission. |
| CFTC Glossary | US government work, public domain in the US | Courtesy of the U.S. Commodity Futures Trading Commission. |
| Quant Stack Exchange answers (`knowledge/options/quantse.jsonl`, built by `scripts/build_quantse_corpus.py`) | **CC BY-SA 4.0** | Each chunk names the answer's author and links to the answer (https://quant.stackexchange.com/a/<id>). Fetched via the public Stack Exchange API: top-voted questions tagged options, implied-volatility, volatility, greeks, option-pricing, delta-hedging, volatility-smile; answers scoring ≥10. Changes: HTML stripped, code blocks omitted, split into chunks. Used for cited lookup, not model training. Remove by deleting the file. |

**Excluded on purpose (Oct 18):**
- Reddit (r/options and similar): Reddit's Data API terms require a commercial agreement and bar using content for AI without permission; scraping is prohibited.
- OpenStax *Principles of Finance* 2e: CC BY-NC-SA, and its pages prohibit ingestion into LLMs or generative AI without permission.
- Copyrighted works (OCC *Characteristics and Risks of Standardized Options*, FINRA, Cboe, Natenberg, Sinclair, Hull, Moontower, Lebron's *Options 101*, Debellefroid's *Derivatives Academy*): only by licence or paraphrased into the trainer's own files.
