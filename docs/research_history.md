# Research history: what was tried, and what was dropped

Short. The decisions that mattered are in [decisions.md](decisions.md), the tests in [results/](results/RESULTS_SUMMARY.md).

1. **Single prediction markets as portfolio factors (v1).** Regress each position on daily changes in each market's probability. Held-out R² under 1%, mostly market exposure; only 1% of events had a significant book-level effect (#153, #19). Dropped as the engine; kept as the baseline.
2. **Real events instead of daily odds (Plan B).** Measure what past events did to a 13-factor map of the portfolio. Large events move factors; typical ones don't (#162, #171). This became the engine (#179).
3. **Regime models and latent factors.** Regime labels, recency weighting and latent factors in the release model added little over the factor map (#165, #169).
4. **An LLM adjusting the engine's numbers.** Made results worse; now the LLM never touches a number (#180, #182).
5. **A learned event-to-move forecaster on text embeddings (Plan C).** Lost to plain normal noise; the best version is a gate on real events plus the LLM's direction, run as a frozen shadow (#187–201; [plan_c.md](research/plan_c.md)).
6. **Targeted sectors and countries for macro events.** No edge on held-out years (#211). Per-holding sector and country layers did help (#215, #238).
7. **Hedging.** Always hedging costs about $130–150 per $100k per event and pays about 3 times in 10, so the default is "probably not worth hedging" (#199–200).
8. **Drawing the scenario spread by event category.** Worse on the live path (#239). Kept behind a switch.
9. **The first scenario methodology** is kept for the record: [methodology_v1.md](research/methodology_v1.md).
