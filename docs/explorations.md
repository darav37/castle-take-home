# Other explorations

First passes, built to see whether the idea holds. None has the evidence behind it that the event pipeline has.

- **Options desk.** Turns a view ("I think oil is going up, but I'm not sure when") into priced option structures from a strategy library the trainer wrote, with a playbook of rules that run in code (implied-vs-needed move, IV z-scores, put-spread sizing to the book) and a paper book that tracks positions, drift and staleness. It asks a couple of questions before it prices anything. Weakest point: the library has about eight strategies and the golden set of labelled views isn't finished.
- **ETF desk.** Matches a view or a holding to ETFs by return similarity over a panel of 18,470 tickers and 879 days, then explains the match. Tested on real data in a first pass with two fixes; not scored against a baseline.
- **Natural-world analyst.** A commodities desk that reads weather and releases: WASDE days move corn about 4.6× a normal hour, EIA gas 1.65×, crude 1.34×; hurricane landfalls barely move markets because forecasts already price them. It informs the analyst; it doesn't produce forecasts of its own.
- **Train mode.** The trainer corrects Worsey in chat with `/fix`; the fix is distilled into a rule and written into the library only after approval. Nothing is trained: it edits files under `library/`, and code enforces that nothing else can be edited.
- **The v1 engine.** The first engine regressed each position on daily changes in single prediction markets. It was replaced by the factor engine (Part III) after a test showed those regressions carry almost no portfolio-level signal (held-out R² under 1%). It stays in the code for the record, and a few tests use its recorded conversations as fixtures.
