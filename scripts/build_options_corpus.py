"""Build the options/derivatives reference corpus.

Only legally reusable sources, each chunk carrying its source, URL and license:
- Wikipedia (CC BY-SA 4.0: attribution + share-alike; see knowledge/ATTRIBUTION.md)
- SEC Investor.gov (US government work, public domain)
- CFTC glossary (US government work, public domain)
Excluded on purpose: OpenStax Principles of Finance 2e (CC BY-NC-SA, and its pages prohibit LLM ingestion without
permission); OCC, FINRA, Cboe, Natenberg, Sinclair, Hull, Moontower, Lebron, Debellefroid (copyrighted).

  python scripts/build_options_corpus.py      # → knowledge/options/corpus.jsonl
"""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge" / "options" / "corpus.jsonl"
UA = {"User-Agent": "Worsey/0.1 (https://github.com/darav37/worsey; options reference corpus) python-requests"}

WIKI = [
    "Option (finance)", "Call option", "Put option", "Option style", "Moneyness", "Strike price", "Expiration (options)",
    "Exercise (options)", "Option time value", "Intrinsic value (finance)", "Put–call parity", "Black–Scholes model",
    "Binomial options pricing model", "Greeks (finance)", "Implied volatility", "Volatility (finance)",
    "Volatility smile", "Volatility risk premium", "VIX", "Variance swap", "Volatility arbitrage", "Delta neutral",
    "Gamma scalping", "Options strategy", "Covered call", "Protective put", "Married put", "Collar (finance)",
    "Bull spread", "Bear spread", "Vertical spread", "Credit spread (options)", "Debit spread", "Calendar spread",
    "Diagonal spread", "Ratio spread", "Backspread", "Straddle", "Strangle (options)", "Butterfly (options)",
    "Iron butterfly", "Iron condor", "Condor (options)", "Box spread", "Risk reversal", "Synthetic position",
    "Naked put", "Naked call", "Cash-secured put", "Pin risk (options)", "Early exercise", "Open interest",
    "Option chain", "Options clearing", "Margin (finance)", "Portfolio margin", "LEAPS (finance)", "Weekly options",
    "Exotic option", "Barrier option", "Asian option", "Binary option", "Futures contract", "Option on futures",
    "Contango", "Normal backwardation", "Convenience yield", "Basis (futures)", "Roll yield", "Hedge (finance)",
    "Delta hedging", "Beta (finance)", "Earnings announcement", "Event study",
]
SEC = ["https://www.investor.gov/news-alerts/investor-bulletins/investor-bulletin-introduction-options",
       "https://www.investor.gov/introduction-investing/investing-basics/glossary/options-contract",
       "https://www.investor.gov/introduction-investing/investing-basics/glossary/stock-options"]
CFTC = "https://www.cftc.gov/LearnAndProtect/EducationCenter/CFTCGlossary/index.htm"


def chunks(text: str, size: int = 1100) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?==+ )", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        p = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1:", p, flags=re.M)
        if len(cur) + len(p) > size and cur:
            out.append(cur.strip())
            cur = ""
        cur += p + "\n"
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) > 80]


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|noscript)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>|</dt>|</dd>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    return "\n\n".join(l for l in lines if len(l) > 40)


def wiki(title: str) -> list[dict]:
    params = {"action": "query", "prop": "extracts|info", "explaintext": 1, "format": "json", "redirects": 1, "inprop": "url",
              "titles": title}
    for attempt in range(4):   # Wikipedia rate-limits bursts; back off and retry, then skip the title
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=30, params=params)
        try:
            data = r.json()
            break
        except ValueError:
            time.sleep(2 ** attempt * 2)
    else:
        print(f"  skipped {title}: HTTP {r.status_code}")
        return []
    pages = data.get("query", {}).get("pages", {})
    p = next(iter(pages.values()), {})
    if "missing" in p or not p.get("extract"):
        return []
    url = p.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    text = p["extract"].split("\n== See also ==")[0].split("\n== References ==")[0]
    return [{"source": "Wikipedia", "title": p["title"], "url": url, "license": "CC BY-SA 4.0", "text": c} for c in chunks(text)]


def page(url: str, source: str, title: str) -> list[dict]:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Worsey/0.1"}, timeout=30)
    if r.status_code != 200:
        return []
    return [{"source": source, "title": title, "url": url, "license": "US government work (public domain)", "text": c}
            for c in chunks(strip_html(r.text))]


def main():
    rows = []
    for t in WIKI:
        got = wiki(t)
        rows += got
        print(f"wiki  {len(got):3}  {t}")
        time.sleep(1.0)
    for u in SEC:
        got = page(u, "SEC Investor.gov", u.rstrip("/").rsplit("/", 1)[-1].replace("-", " "))
        rows += got
        print(f"sec   {len(got):3}  {u}")
    got = page(CFTC, "CFTC", "CFTC Glossary")
    rows += got
    print(f"cftc  {len(got):3}  glossary")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(dict(r, id=i)) + "\n" for i, r in enumerate(rows)), encoding="utf-8")
    print(f"{len(rows)} chunks → {OUT}")


if __name__ == "__main__":
    main()
