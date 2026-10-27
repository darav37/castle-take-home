"""Tables and the review sample for docs/pmrm/results/b2_library.md (reads cache/pmrm/b2/). No network, no LLM."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B2 = ROOT / "cache" / "pmrm" / "b2"


def main():
    U = json.loads((B2 / "unscheduled.json").read_text(encoding="utf-8"))
    L = json.loads((B2 / "library.json").read_text(encoding="utf-8"))
    byid = {r["id"]: r for r in L if r["kind"] == "unscheduled"}
    df = pd.DataFrame(U)
    df["year"] = df.date_first_report.str[:4].astype(int)
    df["macro"] = [bool(byid.get(i, {}).get("w1", {}).get("macro_release_in_window")) for i in df.id]
    out = []
    p = out.append
    p("| Category | Events | Cited sources | Working link | Wikipedia date match | Years | Macro release in short window |")
    p("|---|---|---|---|---|---|---|")
    for c, g in df.groupby("category"):
        p(f"| {c} | {len(g)} | {(g.sourcing == 'cited').sum()} | {(g.sources_ok > 0).sum()} | {g.wiki_match.notna().sum()} | "
          f"{g.year.min()}–{g.year.max()} | {int(g.macro.sum())} |")
    p(f"| **All** | **{len(df)}** | {(df.sourcing == 'cited').sum()} | {(df.sources_ok > 0).sum()} | {df.wiki_match.notna().sum()} | "
      f"{df.year.min()}–{df.year.max()} | {int(df.macro.sum())} |")
    p("")
    p("**By period:** " + ", ".join(f"{a}–{b}: {((df.year >= a) & (df.year <= b)).sum()}" for a, b in [(2004, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]))
    p("")
    p("**Severity:** " + ", ".join(f"{k}: {v}" for k, v in df.severity.fillna("unset").value_counts().items())
      + ". **Anticipation:** " + ", ".join(f"{k}: {v}" for k, v in df.anticipation.fillna("unset").value_counts().items())
      + f". **Targeted tickers given:** {(df.targeted_tickers.str.len() > 0).sum()}. **Prediction market named:** {df.prediction_market.notna().sum()}.")
    S = pd.DataFrame([r for r in L if r["kind"] == "scheduled"])
    p("")
    p("| Release | Records | From | sd of release-day 2y change (bp) | sd of SPY 1-day move | Data surprise available |")
    p("|---|---|---|---|---|---|")
    for rel, g in S.groupby("release"):
        d2 = g.surprise.map(lambda s: s["d2y"]).astype(float)
        spy = g.w1.map(lambda w: w["panel"].get("SPY")).astype(float)
        p(f"| {rel} | {len(g)} | {g.date.min()} | {d2.std() * 100:.1f} | {spy.std() * 100:.2f}% | {g.surprise.map(lambda s: s['data_surprise']).notna().sum()} |")
    p("")
    p("## Review sample (2 random events per category, seed 1729)")
    p("")
    for c, g in df.groupby("category"):
        for _, e in g.sample(2, random_state=1729).iterrows():
            r = byid.get(e.id, {})
            w = r.get("w1", {})
            pn = w.get("panel", {})
            src = "; ".join(f"[{s.get('publisher') or 'link'}]({s['url']}){'' if s.get('url_ok') else ' (link failed)'}" for s in e.sources) or "none: LLM knowledge only"
            wm = e.wiki_match
            t = f" {e.time_first_report_et} ET" if isinstance(e.time_first_report_et, str) and e.time_first_report_et else ""
            p(f"**{e.id} · {e.title}** ({c})")
            p(f"- First report: {e.date_first_report}{t} → short window {w.get('start')} close to {w.get('end')} close"
              + (" (macro release in window)" if w.get("macro_release_in_window") else ""))
            p(f"- Region: {e.region} · severity {e.severity} · anticipation {e.anticipation} · sectors {', '.join(e.targeted_sectors) or '-'}"
              f" · tickers {', '.join(e.targeted_tickers) or '-'}")
            p(f"- {e.description}")
            p(f"- Sources: {src}")
            p("- Date check: " + (f"Wikipedia {wm['day']} (matched: {', '.join(wm['matched'])})" if isinstance(wm, dict) else "no Wikipedia match within 2 days"))
            tg = r.get("targeted", {})
            ab = ", ".join(f"{t2} {v['abnormal_w1'] * 100:+.2f}%" for t2, v in tg.items() if v.get("abnormal_w1") is not None)
            p(f"- Measured short-window move: SPY {pn.get('SPY', 0) * 100:+.2f}%, TLT {pn.get('TLT', 0) * 100:+.2f}%, USO {pn.get('USO', 0) * 100:+.2f}%, "
              f"GLD {pn.get('GLD', 0) * 100:+.2f}%" + (f"; beyond-factor: {ab}" if ab else ""))
            p("")
    p("## Events with no Wikipedia match within 2 days (check the date)")
    p("")
    for _, e in df[df.wiki_match.isna()].iterrows():
        p(f"- {e.id} · {e.date_first_report} · {e.title} ({e.category}; {e.sourcing})")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
