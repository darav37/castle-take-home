"""Quant Stack Exchange answers for the options reference corpus.

Top-voted questions on quant.stackexchange.com for options/vol tags, and their best answers (score >= MIN_SCORE).
Content is CC BY-SA 4.0: every chunk carries the author, a link to the answer, and the license; replies cite it. Used for
cited lookup (retrieval), not model training. Fetched through the public Stack Exchange API (no key; small quota use).
Kept in its own file so it can be removed by deleting knowledge/options/quantse.jsonl.

  python scripts/build_quantse_corpus.py
"""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "knowledge" / "options" / "quantse.jsonl"
API = "https://api.stackexchange.com/2.3"
TAGS = ["options", "implied-volatility", "volatility", "greeks", "option-pricing", "delta-hedging", "volatility-smile"]
MIN_SCORE = 10
PER_TAG = 100


def strip(body: str) -> str:
    body = re.sub(r"(?is)<pre[^>]*>.*?</pre>", " [code omitted] ", body)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>|</blockquote>", "\n", body)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    return "\n".join(re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip())


def get(path: str, **params) -> dict:
    params.update(site="quant", pagesize=100)
    for attempt in range(4):
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        d = r.json()
        if d.get("backoff"):
            time.sleep(int(d["backoff"]) + 1)
        if "items" in d:
            return d
        time.sleep(2 ** attempt)
    return {"items": []}


def chunks(text: str, size: int = 1100) -> list[str]:
    out, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) > size and cur:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) > 80]


def main():
    questions = {}
    for tag in TAGS:
        d = get("/questions", tagged=tag, sort="votes", order="desc", filter="withbody")
        for q in d["items"][:PER_TAG]:
            questions[q["question_id"]] = q
        print(f"{tag:20} {len(d['items'])} questions")
        time.sleep(0.5)
    ids = list(questions)
    rows = []
    for i in range(0, len(ids), 100):
        d = get(f"/questions/{';'.join(map(str, ids[i:i + 100]))}/answers", sort="votes", order="desc", filter="withbody")
        for a in d["items"]:
            if a.get("score", 0) < MIN_SCORE:
                continue
            q = questions[a["question_id"]]
            author = (a.get("owner") or {}).get("display_name", "unknown")
            link = f"https://quant.stackexchange.com/a/{a['answer_id']}"
            title = html.unescape(q["title"])
            for c in chunks(strip(a["body"])):
                rows.append({"source": "Quant Stack Exchange", "title": title, "url": link,
                             "license": f"CC BY-SA 4.0 (Stack Exchange; answer by {html.unescape(author)})",
                             "score": a["score"], "text": c})
        time.sleep(0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(dict(r, id=f"qse{n}")) + "\n" for n, r in enumerate(rows)), encoding="utf-8")
    print(f"{len(rows)} chunks from {len({r['url'] for r in rows})} answers → {OUT}")


if __name__ == "__main__":
    main()
