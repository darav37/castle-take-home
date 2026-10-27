"""SIC code → GICS-style sector (11) and, for a few demo-relevant codes, an industry.

ETF Global constituents carry no sector (CP-L F17), but Massive's ticker overview has a SIC code.
SIC predates GICS, so this is a small static approximation: 2-digit major groups map to a sector,
with exact-code overrides where GICS splits a group (pharma inside chemicals, computers inside
machinery, autos inside transport equipment, REITs inside finance…). Good enough for
concentration ("you're 31% semis"); not a classification service.
"""

from __future__ import annotations

SECTORS = ["Information Technology", "Health Care", "Financials", "Consumer Discretionary", "Communication Services",
           "Industrials", "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials"]

# exact 4-digit overrides (checked first), then 3-digit, then 2-digit major groups
EXACT = {
    3674: ("Information Technology", "Semiconductors"),
    3672: ("Information Technology", "Electronic Components"),
    3577: ("Information Technology", "Computer Hardware"),
    3571: ("Information Technology", "Computer Hardware"),
    3572: ("Information Technology", "Computer Hardware"),
    3576: ("Information Technology", "Networking"),
    3661: ("Information Technology", "Communications Equipment"),
    3663: ("Information Technology", "Communications Equipment"),
    7372: ("Information Technology", "Software"),
    7373: ("Information Technology", "IT Services"),
    7374: ("Information Technology", "IT Services"),
    7370: ("Communication Services", "Interactive Media"),  # GOOGL, META file here (approximation)
    7841: ("Communication Services", "Entertainment"),
    3711: ("Consumer Discretionary", "Automobiles"),
    3714: ("Consumer Discretionary", "Auto Parts"),
    3651: ("Consumer Discretionary", "Household Durables"),
    5961: ("Consumer Discretionary", "Internet Retail"),
    5331: ("Consumer Staples", "Consumer Staples Retail"),
    5912: ("Consumer Staples", "Drug Retail"),
    2844: ("Consumer Staples", "Personal Products"),
    6798: ("Real Estate", "REITs"),
    6324: ("Health Care", "Managed Health Care"),
    5122: ("Health Care", "Health Care Distributors"),
    3812: ("Industrials", "Aerospace & Defense"),
    3721: ("Industrials", "Aerospace & Defense"),
    3724: ("Industrials", "Aerospace & Defense"),
    3728: ("Industrials", "Aerospace & Defense"),
    4911: ("Utilities", "Electric Utilities"),
    2911: ("Energy", "Oil & Gas Refining"),
    1311: ("Energy", "Oil & Gas E&P"),
}
THREE = {
    283: ("Health Care", "Pharmaceuticals & Biotech"),
    384: ("Health Care", "Health Care Equipment"),
    357: ("Information Technology", "Computer Hardware"),
    363: ("Consumer Discretionary", "Household Appliances"),
    541: ("Consumer Staples", "Food Retail"),
    514: ("Consumer Staples", "Food Distributors"),
}
TWO = {
    **{k: "Consumer Staples" for k in range(1, 10)},
    10: "Materials", 12: "Energy", 13: "Energy", 14: "Materials",
    15: "Industrials", 16: "Industrials", 17: "Industrials",
    20: "Consumer Staples", 21: "Consumer Staples", 22: "Consumer Discretionary", 23: "Consumer Discretionary",
    24: "Materials", 25: "Consumer Discretionary", 26: "Materials", 27: "Communication Services", 28: "Materials",
    29: "Energy", 30: "Materials", 31: "Consumer Discretionary", 32: "Materials", 33: "Materials", 34: "Industrials",
    35: "Industrials", 36: "Information Technology", 37: "Industrials", 38: "Information Technology",
    39: "Consumer Discretionary",
    **{k: "Industrials" for k in range(40, 48)}, 48: "Communication Services", 49: "Utilities",
    50: "Industrials", 51: "Industrials",
    **{k: "Consumer Discretionary" for k in range(52, 60)}, 54: "Consumer Staples",
    **{k: "Financials" for k in range(60, 65)}, 65: "Real Estate", 67: "Financials",
    70: "Consumer Discretionary", 72: "Consumer Discretionary", 73: "Information Technology",
    75: "Consumer Discretionary", 78: "Communication Services", 79: "Communication Services",
    80: "Health Care", 82: "Consumer Discretionary", 87: "Industrials",
}


def sic_to_sector(sic) -> tuple[str, str | None]:
    """(sector, industry or None). Unknown or missing SIC → ("unknown", None)."""
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return "unknown", None
    if code in EXACT:
        return EXACT[code]
    if code // 10 in THREE:
        return THREE[code // 10]
    return TWO.get(code // 100, "unknown"), None
