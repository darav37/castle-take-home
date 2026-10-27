"""Systematic event candidates from Wikipedia's Current events portal (PLAN_B §5.1;).

Monthly pages Portal:Current_events/<Month>_<Year>, 2004 to now. Each day section holds nested lists of news items; an
item's text is prefixed with its parent topic, and its external links (news sources) are kept as citations. Section
headings (newer pages) drop sport and culture; a keyword filter keeps economically relevant items. Code only, free;
the pages are cached. Wikipedia text is CC BY-SA 4.0: only short item texts are used, with the page cited.
"""

from __future__ import annotations

import html as H
import re
from datetime import date
from html.parser import HTMLParser

from . import library as L

SKIP_SECTIONS = re.compile(r"sport|arts|culture|entertainment|religion|obituar", re.I)
RELEVANT = re.compile(
    r"tariff|sanction|embargo|export control|trade (?:war|deal|talks|agreement)|opec|\boil\b|crude|natural gas|pipeline|"
    r"refiner|\bport\b|shipping|strait|canal|strike|walkout|central bank|interest rate|federal reserve|\bfed\b|\becb\b|"
    r"bank of (?:england|japan|canada|china)|default|bankrupt|bailout|insolven|recession|inflation|gdp|currency|devalu|"
    r"peg\b|stock market|stock exchange|shares|\bbonds?\b|yield|election|referendum|impeach|coup|resign|shutdown|"
    r"debt ceiling|budget|stimulus|invade|invasion|missile|airstrike|\bwar\b|ceasefire|nuclear|terror|attack|earthquake|"
    r"hurricane|typhoon|cyclone|flood|wildfire|drought|eruption|pandemic|epidemic|outbreak|virus|lockdown|vaccine|"
    r"supreme court|court rules|antitrust|regulat|\bban\b|merger|acquisition|semiconductor|chip|cyberattack|hack", re.I)


class _Items(HTMLParser):
    """Collect (text, links) for every list item, with parent-topic prefixes, inside one day's section."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.out, self.heading, self.in_head, self.skip = [], [], "", False, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "li":
            self.stack.append({"text": "", "links": [], "has_child": False})
            if len(self.stack) > 1:
                self.stack[-2]["has_child"] = True
        elif tag == "a" and self.stack and "external" in (a.get("class") or "") and a.get("href", "").startswith("http"):
            self.stack[-1]["links"].append(a["href"])
        elif tag in ("b", "div") and ("current-events-content-heading" in (a.get("class") or "") or (tag == "b" and not self.stack)):
            self.in_head, self.heading = True, ""

    def handle_endtag(self, tag):
        if tag == "li" and self.stack:
            it = self.stack.pop()
            parents = " / ".join(p["text"].strip() for p in self.stack if p["text"].strip())
            txt = re.sub(r"\s+", " ", it["text"]).strip()
            if not it["has_child"] and len(txt) >= 30 and not self.skip:
                self.out.append({"text": (parents + ": " if parents else "") + txt, "links": it["links"][:3],
                                 "section": self.heading})
        elif tag in ("b", "div") and self.in_head:
            self.in_head = False
            self.skip = bool(SKIP_SECTIONS.search(self.heading))

    def handle_data(self, data):
        if self.in_head:
            self.heading += data.strip()
        elif self.stack:
            self.stack[-1]["text"] += data


def month_items(year: int, month: int) -> list[dict]:
    """Every relevant news item of one month: {day, text, links, section}."""
    mon = date(year, month, 1).strftime("%B")
    url = f"https://en.wikipedia.org/wiki/Portal:Current_events/{mon}_{year}"
    today = date.today()
    recent = (year, month) == (today.year, today.month) or (date(year, month, 28) <= today <= date(year, month, 28).replace(day=1) + __import__("datetime").timedelta(days=40))
    # the current month's page keeps growing: refetch once a day (a frozen copy stalled the daily log on Oct 22-23, #222)
    t = L.web_fresh(url, today.isoformat()) if recent else L.web(url)
    marks = [(m.start(), int(m.group(1))) for m in re.finditer(rf'id="{year}_{mon}_(\d{{1,2}})"', t)]
    out = []
    for k, (pos, day) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(t)
        sec = t[pos:end]
        sec = sec[: sec.find('class="current-events-more') if 'class="current-events-more' in sec else len(sec)]
        p = _Items()
        p.feed(sec)
        for it in p.out:
            if RELEVANT.search(it["text"]):
                out.append({"day": date(year, month, day).isoformat(), **it, "text": H.unescape(it["text"])[:400]})
    return out
