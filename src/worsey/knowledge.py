"""Reference corpora the experts can search.

knowledge/options/corpus.jsonl: Wikipedia (CC BY-SA 4.0), SEC Investor.gov and the CFTC glossary (US government,
public domain), built by scripts/build_options_corpus.py; knowledge/options/quantse.jsonl: top-voted Quant Stack Exchange
answers (CC BY-SA 4.0, author named), built by scripts/build_quantse_corpus.py and removable by deleting the file. Every chunk keeps its source, URL and license; a reply that
uses one cites it. Search is plain BM25 over words: no embeddings, no model call, deterministic.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPORA = {"options": [ROOT / "knowledge" / "options" / "corpus.jsonl", ROOT / "knowledge" / "options" / "quantse.jsonl"]}
STOP = set("the a an of and or to in on for is are be by with as at from that this it its their your you we".split())


def _tok(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 1]


@lru_cache(maxsize=4)
def _index(name: str):
    rows = [json.loads(l) for path in CORPORA[name] if path.exists()
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [Counter(_tok(r["title"] + " " + r["title"] + " " + r["text"])) for r in rows]   # title counted twice
    df = Counter(w for d in docs for w in d)
    avg = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
    return rows, docs, df, avg


def search(query: str, corpus: str = "options", k: int = 4) -> list[dict]:
    rows, docs, df, avg = _index(corpus)
    n = len(rows)
    if not n:
        return []
    q = _tok(query)
    scores = []
    for i, d in enumerate(docs):
        ln = sum(d.values())
        s = 0.0
        for w in q:
            if w in d:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * d[w] * 2.2 / (d[w] + 1.2 * (0.25 + 0.75 * ln / avg))
        if s > 0:
            scores.append((s, i))
    out = []
    for s, i in sorted(scores, reverse=True)[:k]:
        r = rows[i]
        out.append({"title": r["title"], "source": r["source"], "url": r["url"], "license": r["license"],
                    "text": r["text"][:1100], "score": round(s, 2)})
    return out
