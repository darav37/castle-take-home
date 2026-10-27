# Module map

The packages stay as they are; this says which belong to the event pipeline (question 1), which are exploratory, and which are shared.

## Event pipeline (the headline)
- `planb/`: the engine. `live.py` (scenarios, ranges, legs, the live scan), `probability.py` (tiers, longshot correction), `forecaster.py` (tier 3), `factors.py`, `impact.py`, `library.py` (the past-event library), `longshot.json`, `countries.json`, `sectors.json`.
- `discovery.py` (the event universe), `scan_messages.py` (the messages the chat sends), `lookthrough.py`, `sectors.py`, `portfolio.py`.
- `stocks.py`: single-stock events (earnings, news).
- `pmrm/`, `planc/`: research code behind the results (tests and harnesses), not run by the chat.

## Exploratory
- `options/`, `hedge.py`: the options desk and hedge comparator. `etfmatch.py`: the ETF desk. `natural.py`: the natural-world analyst. `agent/train*.py`: train mode. `scenario/`: the v1 engine.

## Shared
- `agent/`: the loop, the router, the grounding check, the budget caps, memory and tools.
- `data/`: Massive, Kalshi, Polymarket and cache clients (every call is disk-cached; `WORSEY_OFFLINE=1` forbids the network).
- `server.py`, `web/`: the group-chat UI. `cli.py`: the `worsey` command.
