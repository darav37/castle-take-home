"""Tunable defaults in one place (METHODOLOGY §16 and the trainer's rulings). Code reads these; tests may override."""

import os

# --- β estimation (METHODOLOGY §4)
SHRINK_K = 4.0                 # β·t²/(t²+k)
HAC_LAGS = 2                   # Newey-West lags for daily time-series regressions (ruling C2)
MIN_HISTORY_DAYS = 60          # tier 1a: trading days of Δp
JUMP_PP = 0.05                 # a "jump day" is |Δp| ≥ 5pp ...
MIN_JUMP_DAYS = 5              # ... and tier 1a needs at least 5 of them
PRIOR_BUCKETS = {"none": 0.0, "small": 0.01, "medium": 0.03, "large": 0.08}  # tier-3 |β| (ruling C1)
PRIOR_SD_FRACTION = 1.0        # prior std = 100% of the bucket value
LOADING_WINDOW_DAYS = 504      # ~2 years, rolling (C9)
RIDGE_ALPHA = 0.01             # ridge penalty on standardized factors (C9)

# --- horizon, discovery, probabilities (METHODOLOGY §5–§7)
HORIZON_DAYS = 91
# scenario engine behind scan_risks / analyze_event / compare_hedges (DECISIONS #179): "planb" (factor map + assisted
# loop, src/worsey/planb/live.py) or "v1" (prediction-market betas, METHODOLOGY_SCENARIOS.md; kept for replays and tests)
SCENARIO_ENGINE = os.environ.get("WORSEY_SCENARIO_ENGINE", "planb")
MAX_RESOLVE_DAYS = 365
SPREAD_USE_LAST = 0.10         # spread > 10pp → use last trade
VENUE_GAP_FLAG = 0.05          # cross-venue disagreement flag
SHORTLIST = 50

# --- dependence (METHODOLOGY §8)
LOGIT_CLIP = (0.01, 0.99)
COPULA_DRAWS = 50_000
TAIL = 0.05
THEME_CORR = 0.5
# Fallback for pairs without enough overlapping history: conservative FOR TAIL RISK.
# Same-sign exposures get +0.5, opposite-sign −0.5 (both raise the P&L variance), never 0. Labelled in output.
FALLBACK_CORR = 0.5
MIN_OVERLAP_DAYS = 30
LADDER_MONOTONE_TOL = 0.03        # ladder prices may break monotonicity by at most 3pp

# --- look-through (METHODOLOGY §3, ruling C11)
LOOKTHROUGH_COVERAGE = 0.90    # keep top holdings to ≥ 90% weight; the rest → "other holdings"

# --- options
RISK_FREE = 0.04
# Legs are priced at the IV implied by the quoted mid, so a leg's model value equals its cost
# at entry. Vendor IV is kept for display. "vendor" switches to vendor IV where present.
PRICE_LEGS_WITH = "mid_iv"

# --- agent (SPEC §11b)
import os as _os
# Router model, by the trainer's rule on the routing eval (Haiku only if ≥ 95% and not worse than Sonnet, pooled):
# Oct 16, 3 runs × 30 cases: Sonnet 90/90, Haiku 87/90 (all three misses on the §0 "NVDA rips on earnings" turn) → Sonnet.
ROUTER_MODEL = _os.environ.get("WORSEY_ROUTER_MODEL", "claude-sonnet-5-5")
EXPERT_MODEL = _os.environ.get("WORSEY_EXPERT_MODEL", "claude-sonnet-5-5")
ROUTER_MAX_TOKENS = 2048        # low, but 1024 cut a long Iran reply mid-sentence (thinking counts against it)
EXPERT_MAX_ROUNDS = 4     # quality over latency (Oct 17, DECISIONS #98): was 3
EXPERT_TIMEOUT_S = 45.0   # was 10s for the old 15s budget; the full-answer budget is now 60s
WARM_TTL_S = 1200         # an expert session goes cold after 20 idle minutes (DECISIONS #108)
WARM_MAX_MESSAGES = 60    # ... or when its transcript gets this long; it then writes a handoff note and restarts
WEB_SEARCH_MAX_USES = 4   # equity_analyst: web searches per expert call (billed per search; counts toward budget.py)

# --- routing coverage rules (SPEC §3.4h): code decides which experts an entry tool runs
WEATHER_SENSITIVE = {"UNG", "VST", "CEG", "NRG", "XLU", "CORN", "WEAT", "SOYB", "DBA", "USO", "BNO"}
WEATHER_MATERIALITY = 0.10        # share of book in weather-sensitive names (stand-in until weather loadings exist)
EQUITY_ANALYST_NAME_WEIGHT = 0.05  # look-through weight above which a name gets equity_analyst coverage
BUILT_EXPERTS = {"event_analyst", "options_desk", "equity_analyst", "etf_desk", "recent_markets_analyst", "natural_world_analyst"}
# natural_world_analyst (was meteorologist, #116): commodities desk with weather inside, built Oct 18
TRAINERS = ["darin"]   # who may /fix Worsey in chat (in-chat train mode)
RSS_FEEDS: list = []   # The trainer's curated newsletter/blog RSS list for recent_markets_analyst (#121); empty until he picks it

# --- worsey build (offline universe scan; SPEC §11b rule 1)
BUILD_EXCLUDED_CATEGORIES = {"Sports", "Entertainment"}   # excluded at SERIES level; others out via IN_SCOPE
BUILD_MIN_VOLUME = 10_000          # contracts traded, lifetime
BUILD_MAX_MARKETS = 250            # top N by volume get hourly history (≈3 calls each)
BUILD_HISTORY_DAYS = 400           # Δp history for β (≈ 275 trading days)
# IV history (options/vol_history.py;): backfilled once from Massive option bars, then a daily
# snapshot per build. Underlyings outside this list get "no history yet" from the z-score signals.
VOL_HISTORY_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "SMH", "XLK", "XLE", "XLF", "GLD", "TLT"]
VOL_HISTORY_DAYS = 365
BUILD_EXTRA_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "GLD", "USO", "UUP", "VIXY", "XLE", "XLK", "SMH", "XLF", "XLU"]
# Standing calendar (METHODOLOGY §5.2 source 3): series always checked regardless of the volume cap
# (still subject to horizon, history, liquidity and tier-1 eligibility): Fed, CPI, payrolls, recession, tariff policy.
BUILD_STANDING_SERIES = ["KXFEDDECISION", "KXRATECUT", "KXRATECUTCOUNT", "KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXU3",
                         "KXRECSSNBER", "KXGDP", "KXTRUMPTARIFFHEAR", "KXTARIFFSECTOR", "KXNEWTARIFFS", "KXCIT301",
                         "KXEFFTARIFF", "KXTARIFFRATEPRC", "KXTARIFFRATECAN", "KXUSAIRANAGREEMENT", "KXHORMUZNORM"]

# --- option chain windows (demo latency, Oct 16): fetch only what a tool can use
CHAIN_STRIKE_BAND = 0.30        # strikes within ±30% of spot
CHAIN_MAX_DAYS = 200            # options desk: expiries out to ~6.5 months
HEDGE_EXPIRY_PAD_DAYS = 60      # hedges: expiries from the event's resolution to +60 days
# Hedge strikes from the scenario's horizon vol: σ_h = ATM IV · √(time to resolution).
HEDGE_LONG_SIGMA = 0.35          # long leg ~0.25–0.5 σ_h out of the money (never in the money)
HEDGE_SHORT_SIGMA = 1.25         # short leg ~1–1.5 σ_h out of the money
HEDGE_MIN_HORIZON_DAYS = 7       # floor for σ_h on events that resolve within days
HEDGE_BETA_DAYS = 252            # window for each position's β to the hedge underlying

# --- ranking: pooled tier 1(b), structural prior, empirical Bayes, divider
POOLED_SERIES = ["KXFEDDECISION", "KXCPI", "KXPAYROLLS"]
POOLED_MIN_RELEASES = 8            # a pooled key needs ≥ 8 releases to count as tier 1(b)
POOLED_START = "2021-01-01"        # settled history used for pooled windows (CPI series starts mid-2021)
DIVIDER_EVIDENCE = 2.0             # above the divider: |B_post| / sd_post ≥ 2 ...
DIVIDER_MATERIAL_PCT = 0.0025      # ... or a tier-2 mechanism whose σ ≥ 0.25% of the book
SHOCK_MAP_BATCH = 20               # markets per Claude call when asking for tier-3 shock maps in `worsey build`

# --- tier-2 event-type templates from analog dates. Code computes the realized moves; surprise scores are the trainer's to add (train mode), so
# every analog counts as a full surprise for now and the result is labelled unreviewed.
STRONG_MATCH_SCORE = 0.20           # keyword score below which a market match is "weak"
EVENT_TYPE_KEYWORDS = {"trade_policy": ["tariff", "tariffs", "trade war", "trade deal", "section 301", "duties"]}
TEMPLATE_ANALOGS = {
    "trade_policy": [
        {"date": "2018-03-22", "what": "Section 301 China tariffs announced", "after": 1},
        {"date": "2018-06-15", "what": "$50bn China tariff list finalized", "after": 1},
        {"date": "2019-05-06", "what": "tariff hike on $200bn of China imports threatened", "after": 1},
        {"date": "2019-08-01", "what": "10% tariff on the remaining $300bn announced", "after": 2},
        {"date": "2025-04-03", "what": "'Liberation Day' reciprocal tariffs (announced Apr 2 after the close)", "after": 2},
    ],
}
