// Worsey group-chat UI (SPEC §1d-0). Renders the server's event stream: message → status → card → reply → done.
// Every number shown comes from a tool payload; the page only formats it (and draws payoff geometry from the legs).
"use strict";

const qs = new URLSearchParams(location.search);
const USER = (qs.get("user") || "darin").toLowerCase();
const CHAT = qs.get("chat") || "demo";
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input");
let cfg = { members: {} }, ws = null, turn = null, quote = null, typing = false, pending = 0;
const eventNames = {};   // event_id → title, from scan rows and analyze_event meta

// ---------- formatting ----------
const MINUS = "−";
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function money(v, signed = true) {
  if (v === null || v === undefined || v === "unbounded") return v === "unbounded" ? "unlimited" : "—";
  const a = Math.abs(v), sign = v < 0 ? MINUS : (signed && v > 0 ? "+" : "");
  const body = a >= 1000 ? `$${(a / 1000).toFixed(a >= 100000 ? 0 : 1)}k` : `$${Math.round(a)}`;
  return sign + body;
}
function cls(v) { return v > 0.5 ? "pos" : v < -0.5 ? "neg" : "zero"; }
function m(v, signed = true) { return `<span class="num ${signed ? cls(v) : ""}">${money(v, signed)}</span>`; }
function pct(p) { if (p === null || p === undefined) return "—"; const x = p * 100; return x < 10 ? `${x.toFixed(1)}%` : `${Math.round(x)}%`; }
function clean(t) { return String(t || "").replace(/\*\*/g, "").replace(/\s+/g, " ").trim(); }
function short(t) {
  const s = clean(t).replace(/^(Will|Is|Are|Does|Do) (the )?/i, "").replace(/\?$/, "");
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function nameOf(r) { return r.short || short(r.event || r.event_id); }   // build-time short name
function niceDate(iso) { const d = new Date(iso + "T12:00:00"); return isNaN(d) ? iso : d.toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
function el(tag, className, html) { const e = document.createElement(tag); if (className) e.className = className; if (html !== undefined) e.innerHTML = html; return e; }

// ---------- tiny markdown (bold, italic, code, lists, tables, headings) ----------
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>").replace(/@worsey\b/gi, '<span class="mention">@worsey</span>');
}
function md(text) {
  const lines = String(text || "").replace(/\r/g, "").split("\n");
  let out = "", i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^\s*\|/.test(l)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r));
      out += '<div class="tbl"><table>' + body.map((r, k) => "<tr>" + cells(r).map((c) => k === 0 ? `<th>${inline(c)}</th>` : `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</table></div>";
      continue;
    }
    if (/^\s*([-*•]|\d+[.)])\s+/.test(l)) {
      const ordered = /^\s*\d/.test(l), items = [];
      while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*•]|\d+[.)])\s+/, ""));
      out += (ordered ? "<ol>" : "<ul>") + items.map((x) => `<li>${inline(x)}</li>`).join("") + (ordered ? "</ol>" : "</ul>");
      continue;
    }
    const h = l.match(/^#{1,4}\s+(.*)/);
    if (h) { out += `<h4>${inline(h[1])}</h4>`; i++; continue; }
    if (!l.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(\||[-*•]\s|\d+[.)]\s|#)/.test(lines[i])) para.push(lines[i++]);
    out += `<p>${para.map(inline).join("<br>")}</p>`;   // replies are written as short lines: keep the breaks
  }
  return out;
}

// ---------- chat chrome ----------
function color(u) { return (cfg.members || {})[u] || "#777"; }
function scrollDown() { window.scrollTo({ top: document.body.scrollHeight }); }
function setStatus(text) {
  $("status").hidden = !text;
  $("status-text").textContent = text ? `worsey is ${text}` : "";
}
function addMessage(ev) {
  const mine = ev.user === USER;
  const box = el("div", "msg" + (mine ? " mine" : ""));
  box.appendChild(el("div", "who", esc(ev.user))).style.color = color(ev.user);
  const b = el("div", "bubble");
  if (ev.quote) b.appendChild(el("div", "qline", esc(clean(ev.quote).slice(0, 90))));
  b.appendChild(el("div", "", inline(ev.text)));
  box.appendChild(b);
  log.appendChild(box);
  turn = null;
}
const turns = {}, active = {};   // request id → its turn element; rid → latest status (concurrent requests, SPEC §1e)
function ensureTurn(rid) {
  if (rid) {
    if (!turns[rid]) {
      const t = el("div", "turn");
      t.appendChild(el("div", "who", "worsey"));
      t.cards = [];
      log.appendChild(t);
      turns[rid] = t;
    }
    return turns[rid];
  }
  if (!turn) {
    turn = el("div", "turn");
    turn.appendChild(el("div", "who", "worsey"));
    turn.cards = [];
    log.appendChild(turn);
  }
  return turn;
}
function showStatus() {   // the latest status of any running request
  const vals = Object.values(active);
  setStatus(vals.length ? vals[vals.length - 1] : "");
}

// ---------- cards ----------
function series(id) { return String(id || "").split("-").slice(0, 2).join("-"); }
function dot(r) {
  const e = r.evidence ?? 0, k = e >= 2 ? "strong" : e >= 1 ? "med" : "";
  return `<span class="dot ${k}" title="evidence ${e.toFixed ? e.toFixed(1) : e} · β tier ${esc(r.beta_tier)}"></span>`;
}
function scanRow(r, extra = "") {
  const row = el("div", "srow" + extra, `${dot(r)}
    <div class="name"><span class="t" title="${esc(clean(r.event))}">${esc(nameOf(r))}</span>
      <small><span class="badge ${r.market?.match === "exact" ? "" : "dim"}">${esc(r.market?.match || "proxy")}</span>
      ${r.flags?.includes("longshot") ? " longshot" : ""}${r.resolves_in_horizon === false ? ` · resolves ${esc(niceDate(r.close_date))}` : ""}</small></div>
    <span class="num">${pct(r.p)}</span>${m(r.pnl_yes)}${m(r.pnl_no)}`);
  row.addEventListener("click", () => { input.value = `@worsey tell me more about ${nameOf(r)}`; input.focus(); });
  return row;
}
function scanRows(rows, into) {
  // UI-level theme grouping (METHODOLOGY §8.1): same series → one row, headed by the largest σ, expandable.
  // The backend's joint-σ theme ranking is not built yet; the head's own σ orders the group (flagged in the "how").
  const groups = [];
  for (const r of rows) {
    const g = groups.find((x) => x.key === series(r.event_id));
    g ? g.rows.push(r) : groups.push({ key: series(r.event_id), rows: [r] });
  }
  for (const g of groups) {
    const head = scanRow(g.rows[0]);
    into.appendChild(head);
    if (g.rows.length > 1) {
      const kids = g.rows.slice(1).map((r) => scanRow(r, " child"));
      const btn = el("button", "expander", `+${kids.length} related`);
      head.querySelector(".name small").append(" · ", btn);
      kids.forEach((k) => { k.hidden = true; into.appendChild(k); });
      btn.addEventListener("click", (e) => { e.stopPropagation(); const open = kids[0].hidden; kids.forEach((k) => (k.hidden = !open)); btn.textContent = open ? "hide" : `+${kids.length} related`; });
    }
  }
}
function headRow() { return el("div", "srow head", `<span></span><span>event</span><span class="num">odds</span><span class="num">if YES</span><span class="num">if NO</span>`); }
function cardScan(p) {
  [...(p.events || []), ...(p.drift || [])].forEach((r) => (eventNames[r.event_id] = nameOf(r)));
  const c = el("div", "card");
  c.appendChild(el("div", "card-h", `<b>What could hit your book</b><small>resolves in the next ${p.horizon_days} days · ranked by size of the swing</small>`));
  const b = el("div", "card-b");
  const strong = (p.events || []).filter((r) => r.above_divider), weak = (p.events || []).filter((r) => !r.above_divider);
  b.appendChild(headRow());
  if (strong.length) scanRows(strong, b);
  else b.appendChild(el("div", "srow head", `<span></span><span>no event clears the evidence bar</span>`));
  for (const [label, rows] of [["weak evidence", weak], ["odds could move: resolves after your horizon", p.drift || []]]) {
    if (!rows.length) continue;
    const d = el("details", "sect"); d.appendChild(el("summary", "", `${label} (${rows.length})`));
    d.appendChild(headRow()); scanRows(rows, d); b.appendChild(d);
  }
  c.appendChild(b);
  return c;
}

function hedgeRows(rows) {
  // if no option hedge pays on YES: show only the best one (smallest loss if YES), the rest behind one line
  const opts = rows.filter((r) => r.hedge !== "do_nothing" && r.hedge !== "prediction_market");
  if (opts.length < 2 || opts.some((r) => !(r.flags || []).includes("does_not_pay_if_yes"))) return { shown: rows, hidden: [] };
  const best = opts.reduce((a, b) => (b.residual_if_yes > a.residual_if_yes ? b : a));
  return { shown: rows.filter((r) => !opts.includes(r) || r === best), hidden: opts.filter((r) => r !== best) };
}
function hedgeTable(rows) {
  const box = el("div", "hbody");
  box.appendChild(el("div", "hrow head", `<span>hedge</span><span class="num">cost</span><span class="num">book if YES</span><span class="num">if NO</span><span class="num">cover</span>`));
  const { shown, hidden } = hedgeRows(rows);
  hedgeRowsInto(box, shown);
  if (hidden.length) {
    const d = el("details", "sect");
    const n = hidden.length;
    d.appendChild(el("summary", "", `${n} other option hedge${n > 1 ? "s" : ""} ${n > 1 ? "don’t" : "doesn’t"} pay either`));
    hedgeRowsInto(d, hidden);
    box.appendChild(d);
  }
  return box;
}
function hedgeRowsInto(box, rows) {
  for (const r of rows) {
    const flags = r.flags || [];
    const badges = (flags.includes("cost_exceeds_loss") ? ' <span class="badge warn">costs more than it protects</span>' : "")
      + (flags.includes("does_not_pay_if_yes") ? ' <span class="badge warn">doesn’t pay if YES</span>' : "");
    const cover = r.coverage >= 0.005 ? `${Math.round(r.coverage * 100)}%` : r.coverage > 0 ? "<1%" : "none";
    box.appendChild(el("div", "hrow", `<span class="nm">${esc(hedgeName(r))}${badges}</span><span class="num">${r.cost ? money(r.cost, false) : "$0"}</span>${m(r.residual_if_yes)}${m(r.book_if_no)}<span class="num">${cover}</span>`));
    box.appendChild(el("div", "basis", esc(r.basis_risk || "")));
  }
}
function cardHedge(p) {
  const c = el("div", "card");
  const name = p.scenario || (p.unpriced ? "a big tariff escalation" : (eventNames[p.event_id] || p.event_id));
  const sub = p.scenario ? `if it happens · middle case ${money(p.loss_if_yes)}` + (p.kalshi_odds ? ` · Kalshi odds ${pct(p.kalshi_odds.p)}` : "")
    : p.unpriced ? `no market prices this · typical analog loss ${money(p.loss_if_yes)}` : `odds ${pct(p.p)} · your loss if YES ${money(p.loss_if_yes)}`;
  const h = el("div", "card-h", `<b>Hedging: ${esc(name)}</b><small>${sub}</small>`);
  c.appendChild(h);
  const v = p.should_you_hedge;
  if (v) {   // code's verdict first (DECISIONS #200): most of the time, protection isn't worth the premium
    const hst = v.history || {};
    const paid = hst.share_that_paid_for_themselves != null ? ` · puts like this paid off ${Math.round(hst.share_that_paid_for_themselves * 10)} times in 10` : "";
    const cost = v.cheapest_hedge ? ` · cheapest hedge ${money(v.cheapest_hedge.cost_usd, false)}` : "";
    c.appendChild(el("div", `verdict ${v.verdict === "skip by default" ? "skip" : "consider"}`,
      `<b>${v.verdict === "skip by default" ? "Probably not worth hedging" : "Worth considering a hedge"}</b>` +
      `<small>${esc(v.why)}. Normal 2-day bad case ${money(v.your_normal_2_day_bad_case_usd)}${cost}${paid}</small>`));
  }
  const body = el("div", "card-b");
  body.appendChild(hedgeTable(p.rows || []));
  c.appendChild(body);
  if (p.stress) {
    const t = el("div", "toggle", `<button class="on">${p.scenario ? "middle" : "typical"}</button><button>${p.scenario ? "bad case" : "stress"}</button>`);
    h.appendChild(t);
    const [bt, bs] = t.querySelectorAll("button");
    const show = (stress) => {
      bt.classList.toggle("on", !stress); bs.classList.toggle("on", stress);
      body.replaceChildren(hedgeTable(stress ? p.stress.rows : p.rows));
      h.querySelector("small").textContent = stress ? `${p.stress.case} · loss ${money(p.stress.loss_if_it_happens)}` : sub;
    };
    bt.onclick = () => show(false); bs.onclick = () => show(true);
  }
  return c;
}

function cardEarnings(p) {
  // single-name risk (DECISIONS #204): who reports soon, what rides on it, what options price vs the stock's history
  const c = el("div", "card");
  const tot = p.total_dollars_at_implied_moves ? ` · ${money(p.total_dollars_at_implied_moves, false)} at the options' implied moves` : "";
  c.appendChild(el("div", "card-h", `<b>Earnings in the next ${p.window_days} days</b><small>${p.rows.length} of your names${tot}</small>`));
  const box = el("div", "card-b");
  box.appendChild(el("div", "hrow head", `<span>name</span><span class="num">reports</span><span class="num">your $</span><span class="num">options ±</span><span class="num">last 8 median</span>`));
  for (const r of p.rows) {
    const via = Object.keys(r.held_via || {}).filter((k) => k !== r.ticker);
    const nm = `${esc(r.ticker)}${via.length ? ` <small>via ${esc(via.join(", "))}</small>` : ""}`;
    const im = r.implied_move != null ? `${(r.implied_move * 100).toFixed(1)}%` : "–";
    const hm = r.past_median_move != null ? `${(r.past_median_move * 100).toFixed(1)}%` : "–";
    box.appendChild(el("div", "hrow", `<span class="nm">${nm}</span><span class="num">${esc(r.reports.slice(5))}</span><span class="num">${money(r.your_exposure_usd, false)}</span><span class="num">${im}</span><span class="num">${hm}</span>`));
  }
  c.appendChild(box);
  return c;
}

function cardStress(p) {
  // black-swan mode (DECISIONS #208): the analyst's shock (shown as an assumption), code's mapping to each holding
  const c = el("div", "card");
  const bk = p.book_usd || {};
  c.appendChild(el("div", "card-h", `<b>Stress test: ${esc(p.scenario)}</b><small>a scenario assumption, not a forecast · book ${money(bk.as_written)} (half the shock ${money(bk.if_the_shock_is_half)}, 1.5× ${money(bk.if_it_is_1_5x)})</small>`));
  const b = el("div", "card-b");
  b.appendChild(el("div", "rrow head", `<span>assumed move</span><span class="num">size</span><span class="num">set by</span>`));
  for (const [k, v] of Object.entries(p.shock || {})) b.appendChild(el("div", "rrow", `<span>${esc(k)}${v.why ? `<small>${esc(v.why)}</small>` : ""}</span><span class="num">${esc(v.move)}</span><span class="num"><small>${v.set_by === "analyst" ? "analyst" : "co-move"}</small></span>`));
  b.appendChild(el("div", "rrow head", `<span>holding</span><span class="num">as written</span><span class="num">1.5×</span>`));
  for (const x of p.by_position || []) b.appendChild(el("div", "rrow", `<span>${esc(x.ticker)}<small>±${money(x.plus_or_minus_its_own_noise_usd, false)} own noise</small></span>${m(x.pnl_usd)}${m(x.pnl_1_5x_shock_usd)}`));
  for (const f of p.beyond_history || []) b.appendChild(el("div", "basis", `⚠ ${esc(f)}`));
  c.appendChild(b);
  return c;
}

function cardUpcoming(p) {
  // discovery (DECISIONS #218): upcoming market-moving events, odds, and what's already priced
  const c = el("div", "card");
  c.appendChild(el("div", "card-h", `<b>Coming up: what could move your book</b><small>next ${p.horizon_days} days · odds from Kalshi and Polymarket · already-priced part removed</small>`));
  const b = el("div", "card-b");
  b.appendChild(el("div", "hrow head", `<span>event</span><span class="num">odds</span><span class="num">if yes</span><span class="num">if no</span><span class="num">at stake</span>`));
  for (const r of p.rows || []) {
    const pr = r.mostly_priced ? ` <small>mostly priced</small>` : "";
    b.appendChild(el("div", "hrow", `<span class="nm">${esc(r.title)}<small>closes ${esc(r.closes)}${r.why_it_moves ? " · " + esc(r.why_it_moves) : ""}</small>${pr}</span><span class="num">${pct(r.odds_yes)}</span>${m(r.if_yes_usd)}${m(r.if_no_usd)}<span class="num">±${money(r.risk_left_usd, false)}</span>`));
  }
  c.appendChild(b);
  return c;
}

function cardWatchlist(p) {
  // the event universe matched to this book (DECISIONS #222): headline, tail risks, noise for you
  const c = el("div", "card");
  c.appendChild(el("div", "card-h", `<b>Watchlist: what's coming, matched to your book</b><small>next ${p.horizon_days} days from ${esc(p.as_of)} · impactful = at least ±${money(p.impact_threshold_usd, false)} (half your normal 2-day swing)</small>`));
  const b = el("div", "card-b");
  // Part II (#233): every probability carries its tier
  const TIER = { 1: "mkt", "2a": "derived", "2b": "bounded", 3: "forecast" };
  const odds = (r) => {
    if (r.odds == null) return "–";
    if (r.tier == null) return `<span title="${esc(r.odds_note || "")}">${pct(r.odds)}*</span>`;
    const raw = r.odds_raw != null && Math.abs(r.odds_raw - r.odds) >= 0.005 ? ` (mkt ${pct(r.odds_raw)})` : "";
    const tip = r.odds_label || (r.tier === 1 ? "market price, corrected for the measured longshot bias" : "");
    const spread = r.forecast && r.forecast.spread ? ` · agents ${pct(r.forecast.spread[0])}–${pct(r.forecast.spread[1])}` : "";
    return `<span title="${esc(tip + raw + spread)}">${pct(r.odds)}<small> ${TIER[r.tier] || ""}</small></span>`;
  };
  const src = { market: "market", calendar: "calendar", news: "news", proposal: "for your book", user: "yours" };
  const hist = (r) => r.from_history ? `from history: ${r.from_history_words || r.from_history}${r.from_history_bad_case_usd != null ? ` (bad case ${money(r.from_history_bad_case_usd)})` : ""}` : "";
  const sub = (r) => [r.source === "market" ? (r.venue === "polymarket" ? "Polymarket" : "Kalshi") : src[r.source], r.when, r.linked_to, hist(r), r.why_this_book || r.why_it_moves || r.how,
    (r.by_position || []).slice(0, 3).map((x) => `${x.ticker} ${money(x.full_impact_usd)}`).join(", "),
    r.mostly_priced ? "mostly priced" : ""].filter(Boolean).map(esc).join(" · ");
  const group = (title, rows, note) => {
    if (!rows || !rows.length) return;
    b.appendChild(el("div", "basis", `<b>${title}</b> ${note}`));
    b.appendChild(el("div", "hrow head w4", `<span>event</span><span class="num">odds</span><span class="num">if it happens</span><span class="num">at stake</span>`));
    for (const r of rows) {
      const full = r.full_impact_usd ?? r.swing_usd;
      const ifyes = r.if_yes_usd != null ? r.if_yes_usd : full;
      b.appendChild(el("div", "hrow w4", `<span class="nm">${esc(r.title)}<small>${sub(r)}</small></span><span class="num">${odds(r)}</span>${m(ifyes)}<span class="num">±${money(r.at_stake_usd ?? Math.abs(full), false)}</span>`));
    }
  };
  group("Your events", p.yours, "· what you asked about, priced the same way");
  for (const u of p.yours || []) {
    const rel = (u.related || []).map((r) => `${esc(r.title)}${r.odds != null ? " " + pct(r.odds) : ""}`).join("; ");
    b.appendChild(el("div", "basis", `${esc(u.title)}: <b>${esc(u.verdict || "")}</b>${rel ? ` · same move, folded here: ${rel}` : ""}`));
  }
  group((p.yours || []).length ? "Beyond what you asked: headline" : "Headline", p.headline, "· coming up and big for this book");
  group("Tail risks", p.tail_risks, "· big for this book, no live catalyst");
  if ((p.noise_for_you || []).length) b.appendChild(el("div", "basis", `<b>Noise for you:</b> ${p.noise_for_you.map((r) => `${esc(r.title)} (±${money(Math.abs(r.full_impact_usd ?? r.swing_usd), false)})`).join(", ")}`));
  b.appendChild(el("div", "basis", "Odds: mkt = a market is the event (its price corrected for the longshot bias; hover for the raw price), derived = from a market's strike or deadline ladder, bounded = from logically related markets, forecast = Worsey's forecaster (no market lists it). Market odds price part of the move (if it happens shows what's left). * = base rate: how often events of that kind happen, not this event's odds. Calendar items show a typical swing, not a direction."));
  c.appendChild(b);
  return c;
}

function hedgeName(r, id) { return r.hedge === "prediction_market" ? "Buy the YES contract" : r.name; }
const STRAT_NAMES = { long_call: "Long call", long_put: "Long put", call_debit_spread: "Call spread", put_debit_spread: "Put spread",
  short_put: "Short put", covered_call: "Covered call", collar: "Collar", straddle: "Straddle", strangle: "Strangle", calendar: "Calendar" };
function legText(l, und) { return `${l.qty > 0 ? "buy" : "sell"} ${Math.abs(l.qty)} ${und} ${l.strike} ${l.kind} · ${l.days}d`; }
function payoff(c, spot, sd, view) {
  const w = Math.max(0.15, 2.5 * (sd || 0.06), Math.abs(view?.mean_move || 0) + 2.5 * (view?.sd || 0));
  const lo = spot * (1 - w), hi = spot * (1 + w), N = 80;
  const f = (S) => c.legs.reduce((a, l) => a + l.qty * 100 * (l.kind === "call" ? Math.max(S - l.strike, 0) : l.kind === "put" ? Math.max(l.strike - S, 0) : (S - spot)), 0) - c.cost;
  const xs = [...Array(N + 1)].map((_, i) => lo + (hi - lo) * i / N), ys = xs.map(f);
  const ymin = Math.min(...ys, 0), ymax = Math.max(...ys, 0), W = 360, H = 70, pad = 4;
  const X = (x) => (x - lo) / (hi - lo) * W, Y = (y) => pad + (ymax - y) / ((ymax - ymin) || 1) * (H - 2 * pad);
  const pts = xs.map((x, i) => `${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join(" ");
  // the user's view as a lognormal density on spot at the horizon (shape only, scaled to the chart height)
  let dens = "";
  if (view && view.sd > 0) {
    const mu = Math.log(1 + view.mean_move) - 0.5 * view.sd ** 2;
    const pdf = xs.map((x) => Math.exp(-((Math.log(x / spot) - mu) ** 2) / (2 * view.sd ** 2)) / x);
    const pm = Math.max(...pdf);
    dens = `<polygon points="0,${H} ${xs.map((x, i) => `${X(x).toFixed(1)},${(H - pdf[i] / pm * (H - 6)).toFixed(1)}`).join(" ")} ${W},${H}" fill="var(--accent-soft)" stroke="none"/>`;
  }
  return `<svg class="payoff" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="payoff at expiry with your view">
    ${dens}
    <line x1="0" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line)" stroke-dasharray="3 3"/>
    <line x1="${X(spot)}" x2="${X(spot)}" y1="0" y2="${H}" stroke="var(--muted)" stroke-width="1"/>
    <text x="${Math.min(X(spot) + 3, W - 30)}" y="10" font-size="9" fill="var(--muted)">spot</text>
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
}
function cardStrategyCompact(p) {
  // the summary while a structure is being chosen: one line each, no charts
  const v = p.view || {}, c = el("div", "card compact");
  const move = v.target_move_pct ? `${v.direction === "down" ? MINUS : "+"}${Math.abs(v.target_move_pct)}%` : (v.direction || "");
  c.appendChild(el("div", "card-h", `<b>${esc(v.underlying || "")} ${esc(move)}${v.horizon ? " by " + esc(niceDate(v.horizon)) : ""}: options priced</b><small>summary · ask for the full payoff chart once you pick one</small>`));
  for (const x of (p.candidates || []).slice(0, 4)) {
    const legs = (x.legs || []).map((l) => `${l.qty > 0 ? "buy" : "sell"} ${l.strike} ${l.kind}${l.delta !== undefined && l.kind !== "stock" ? ` (Δ${Math.abs(l.delta).toFixed(2)})` : ""}`).join(" / ");
    const sized = x.sized_to_budget && x.sized_to_budget.contracts ? ` · ${x.sized_to_budget.contracts}× = ${money(x.sized_to_budget.total_cost, false)}` : "";
    c.appendChild(el("div", "crow", `<div><b>${esc(STRAT_NAMES[x.id] || x.id.replace(/_/g, " "))}</b> <small>${esc(legs)}</small></div>
      <div class="cnum"><span>cost ${money(x.cost, false)}${sized}</span><span>max loss ${money(x.max_loss)}</span><span>E[P&amp;L] <b class="${cls(x.expected_pnl)}">${money(x.expected_pnl)}</b></span><span>${pct(x.prob_profit)} win</span></div>`));
  }
  return c;
}

function cardStrategy(p) {
  if (p.compact) return cardStrategyCompact(p);
  const v = p.view || {}, c = el("div", "card");
  const move = v.target_move_pct ? `${v.direction === "down" ? MINUS : "+"}${Math.abs(v.target_move_pct)}%` : v.direction;
  const changed = p.view_diff?.horizon ? ` · horizon ${niceDate(p.view_diff.horizon.from)} → ${niceDate(p.view_diff.horizon.to)}` : "";
  c.appendChild(el("div", "card-h", `<b>Your view: ${esc(v.underlying)} ${esc(move)} by ${esc(niceDate(v.horizon))}</b><small>spot $${p.market?.spot} · options priced off the live chain${changed}</small>`));
  const cands = (p.candidates || []).slice(0, 3);
  if (p.teaching || cands.some((x) => x.expected_pnl < 0)) {   // only when E[P&L | your view] < 0
    c.appendChild(el("div", "banner", "Even if you’re right on direction, buying these loses money on average under your own view."));
  }
  for (const x of cands) {
    const s = el("div", "strat");
    s.innerHTML = `<div class="sn">${esc(STRAT_NAMES[x.id] || x.id.replace(/_/g, " "))}</div>
      <div class="legs">${x.legs.map((l) => esc(legText(l, v.underlying))).join(" / ")}</div>
      <div class="evs">
        <div class="ev"><small>expected P&amp;L<br>under your view</small><b class="${cls(x.expected_pnl)}">${money(x.expected_pnl)}</b><small>${pct(x.prob_profit)} chance of profit</small></div>
        <div class="ev mkt"><small>expected P&amp;L<br>at market pricing</small><b class="${cls(x.expected_pnl_market)}">${money(x.expected_pnl_market)}</b><small>${pct(x.prob_profit_market)} chance of profit</small></div>
      </div>
      ${x.edge_direction !== undefined ? `<div class="edge">from your direction ${money(x.edge_direction)} · from vol ${money(x.edge_vol)}</div>` : ""}
      <div class="kv"><div><small>cost</small>${money(x.cost, false)}</div><div><small>max loss</small>${money(x.max_loss)}</div>
        <div><small>max gain</small>${money(x.max_gain)}</div><div><small>breakeven</small>${(x.breakevens || []).map((b) => "$" + b).join(", ") || "—"}</div></div>
      ${payoff(x, p.market?.spot, p.your_distribution?.sd, p.your_distribution)}
      <div class="legend"><span class="sw"></span>your view at the horizon · line: payoff at expiry</div>`;
    c.appendChild(s);
  }
  return c;
}

function cardTemplate(p) {
  const c = el("div", "card");
  const kind = (p.type || "").replace(/_/g, " ");
  c.appendChild(el("div", "card-h", `<b>If it happens: ${esc(kind)}</b><small>no market prices it · estimate from past episodes</small>`));
  const [lo, hi] = p.range || [p.pnl_if_it_happens, p.pnl_if_it_happens];
  const pos = (v) => `${(((v - lo) / ((hi - lo) || 1)) * 100).toFixed(1)}%`;
  const b = el("div", "card-b");
  b.appendChild(el("div", "tmpl-top", `<div><small>typical hit (median episode)</small><div class="big neg">${money(p.pnl_if_it_happens)}</div></div>`
    + (p.average !== undefined ? `<div><small>average</small><div class="mid neg">${money(p.average)}</div></div>` : "")));
  b.appendChild(el("div", "range", `<div class="bar"></div>${(p.analogs || []).map((a) => `<span class="tick" style="left:${pos(a.book_pnl)}" title="${esc(a.what)}"></span>`).join("")}<span class="typ" style="left:calc(${pos(p.pnl_if_it_happens)} - 2px)"></span>`));
  b.appendChild(el("div", "range-l", `<span>worst ${money(lo)}</span><span>mildest ${money(hi)}</span>`));
  for (const a of p.analogs || []) b.appendChild(el("div", "arow" + (a.typical ? " typ-row" : ""), `<small>${esc(a.date)}${a.typical ? "<br>typical" : ""}</small><span>${esc(a.what)}</span>${m(a.book_pnl)}`));
  b.appendChild(el("div", "basis", esc(p.status || "")));
  c.appendChild(b);
  return c;
}
function cardChart(p) {
  const c = el("div", "card"), src = p.path || p.file || p.chart;
  if (src) c.appendChild(el("img", "", "")).setAttribute("src", "/charts/" + String(src).split(/[\\/]/).pop());
  c.querySelector("img")?.setAttribute("style", "width:100%;display:block");
  return c;
}
function cardSources(p) {
  // the equity analyst's web citations (DECISIONS #101): every company fact in the reply traces to one of these
  const seen = new Set(), items = (p.items || []).filter((x) => x.url && !seen.has(x.url) && seen.add(x.url));
  const c = el("div", "card sources");
  c.appendChild(el("div", "card-h", `<b>Sources</b><small>${items.length} public source${items.length === 1 ? "" : "s"} read for this answer</small>`));
  const b = el("div", "card-b");
  for (const x of items.slice(0, 6)) {
    let host = ""; try { host = new URL(x.url).hostname.replace(/^www\./, ""); } catch (_) { /* keep empty */ }
    b.appendChild(el("a", "src", `<span>${esc(x.title || host)}</span><small>${esc(host)}</small>`)).setAttribute("href", x.url);
  }
  b.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
  c.appendChild(b);
  return c;
}
function cardEtfMatch(p) {
  // ETF matching for restricted traders (SPEC §5a): what to put on, how close out of sample, what's lost
  const c = el("div", "card etf"), tgt = Object.keys(p.target || {}).join(" + ");
  const pc = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
  const pp = (v) => (v === null || v === undefined ? "—" : `${v >= 0 ? "+" : MINUS}${Math.abs(v * 100).toFixed(1)}%`);
  c.appendChild(el("div", "card-h", `<b>Tracking ${esc(tgt)} with ETFs you can hold</b><small>${p.universe?.permitted ?? "?"} permitted funds · tested on ${esc(p.test?.[0] || "")} → ${esc(p.test?.[1] || "")} (unseen, buy-and-hold)</small>`));
  const b = el("div", "card-b");
  const w = p.replica?.weights || {}, order = Object.fromEntries((p.order || []).map((o) => [o.ticker, o]));
  b.appendChild(el("div", "erow head", `<span>fund</span><span>kind</span><span class="num">weight</span><span class="num">$</span>`));
  for (const [t, v] of Object.entries(w))
    b.appendChild(el("div", "erow", `<span><b>${esc(t)}</b><small>${esc((p.replica.names || {})[t] || "")}</small></span><span><span class="badge dim">${esc((p.replica.kinds || {})[t] || "")}</span></span><span class="num">${pc(v)}</span><span class="num">${order[t] ? money(order[t].usd, false) : "—"}</span>`));
  if ((p.replica?.cash || 0) > 1e-4)
    b.appendChild(el("div", "erow", `<span><b>cash</b><small>T-bills</small></span><span></span><span class="num">${pc(p.replica.cash)}</span><span class="num">${order.cash ? money(order.cash.usd, false) : "—"}</span>`));
  const m = p.test_metrics || {}, rows = [["This mix", m.replica_buy_and_hold], ["SPY alone", m.spy_alone],
    [`Best single (${m.best_single_etf?.ticker || "?"})`, m.best_single_etf], ["Holdings-based", m.holdings_based]].filter((r) => r[1]);
  b.appendChild(el("div", "erow head mt", `<span>out of sample</span><span class="num">tracking err</span><span class="num">moves caught</span><span class="num">return gap</span>`));
  for (const [label, x] of rows)
    b.appendChild(el("div", "erow" + (label === "This mix" ? " me" : ""), `<span>${esc(label)}</span><span class="num">${pc(x.tracking_error)}</span><span class="num">${pc(x.r2)}</span><span class="num">${pp(x.return_gap)}</span>`));
  const L = p.lost || {}, lt = L.lookthrough || {}, lines = [];
  if (L.oos_r2 !== undefined && L.oos_r2 !== null) lines.push(`Catches ${pc(L.oos_r2)} of ${esc(tgt)}'s daily moves; the rest is company-specific.`);
  if (lt.share_of_target_held_directly !== undefined && lt.share_of_target_held_directly !== null)
    lines.push(`Holds ${pc(lt.replica_weight_in_target)} ${esc(tgt)} directly; the rest is mostly ${(lt.top_other_names || []).slice(0, 3).map((x) => esc(x.name)).join(", ")}.`);
  const ec = L.earnings_capture;
  if (ec && ec.median_captured !== null && ec.median_captured !== undefined) lines.push(`On past earnings days it caught a median ${pc(ec.median_captured)} of the stock's move.`);
  if (p.stability && !p.stability.stable) lines.push("Unstable: the best mix changes with the window, treat it as rough.");
  const ex = Object.entries(p.universe?.excluded_by_reason || {}).slice(0, 3).map(([k, v]) => `${v} ${esc(k)}`).join(" · ");
  if (ex) lines.push(`Excluded by your rules: ${ex}.`);
  b.appendChild(el("div", "lost", lines.map((x) => `<div>${x}</div>`).join("")));
  c.appendChild(b);
  return c;
}

// ---- Plan B cards (DECISIONS #179): the risk scan and one scenario's ranges
function cardRisks(p) {
  (p.risks || []).forEach((r) => (eventNames[r.risk_id] = r.risk));
  const c = el("div", "card");
  c.appendChild(el("div", "card-h", `<b>What could hit your book</b><small>past big events replayed on your book today · bad case = worst 1 in 10</small>`));
  const b = el("div", "card-b");
  b.appendChild(el("div", "rrow head", `<span>risk</span><span class="num">typical</span><span class="num">bad case</span>`));
  for (const r of p.risks || []) {
    const k = r.kalshi && r.kalshi.next_on_kalshi ? ` · next ${esc(r.kalshi.next_on_kalshi)}` + (r.kalshi.most_likely ? ` (${esc(clean(r.kalshi.most_likely.outcome))} ${pct(r.kalshi.most_likely.p)})` : "") : "";
    b.appendChild(el("div", "rrow", `<span>${esc(r.risk)}<small>${esc(r.how_often)}${k}</small></span>${m(r.typical_usd)}${m(r.bad_case_1_in_10_usd)}`));
    const bp = r.by_position;
    if (bp && bp.positions && bp.positions.length) {   // per holding (DECISIONS #207)
      const d = el("details", "bypos");
      d.appendChild(el("summary", "", "by position"));
      d.appendChild(el("div", "rrow head", `<span>holding</span><span class="num">typical</span><span class="num">its bad case</span><span class="num">in book's bad case</span>`));
      for (const x of bp.positions) d.appendChild(el("div", "rrow", `<span>${esc(x.ticker)}</span>${m(x.typical_usd)}${m(x.own_bad_case_1_in_10_usd)}${m(x.in_the_books_bad_case_usd)}`));
      b.appendChild(d);
    }
  }
  for (const sr of p.sector_risks || [])   // sector layer (DECISIONS #217)
    b.appendChild(el("div", "rrow", `<span>a bad ${esc(sr.sector)} day<small>1 in 20 · ${Object.keys(sr.by_holding_usd || {}).map(esc).join(", ")}</small></span><span class="num"></span>${m(sr.a_bad_2_day_sector_move_1_in_20_usd)}`));
  const cov = p.coverage || {};
  if (cov.verdict) b.appendChild(el("div", "basis", `Factors explain ${pct(cov.share_explained)} of this book's daily moves: ${esc(cov.verdict)}.`));
  c.appendChild(b);
  return c;
}
function rangeRow(label, q, lo, hi) {
  const pos = (v) => `${(((v - lo) / ((hi - lo) || 1)) * 100).toFixed(1)}%`;
  const [a, z] = q.likely_range, [wa, wz] = q.wide_range;
  return `<div class="sc-row"><small>${label}</small><div class="mid ${cls(q.middle)}">${money(q.middle)}</div>`
    + `<div class="range"><div class="wide" style="left:${pos(wa)};width:calc(${pos(wz)} - ${pos(wa)})"></div>`
    + `<div class="likely" style="left:${pos(a)};width:calc(${pos(z)} - ${pos(a)})"></div><span class="typ" style="left:calc(${pos(q.middle)} - 2px)"></span></div>`
    + `<small class="num">likely ${money(a)} to ${money(z)} · 9 in 10 ${money(wa)} to ${money(wz)}</small></div>`;
}
function cardScenario(p) {
  const c = el("div", "card");
  const odds = p.probability && typeof p.probability === "object" ? `Kalshi odds ${pct(p.probability.p)}` : "if it happens · no market prices it";
  c.appendChild(el("div", "card-h", `<b>If it happens: ${esc(p.scenario)}</b><small>${esc(p.severity)} · ${odds}</small>`));
  const b = el("div", "card-b");
  const fh = p.if_it_happens.first_hours, ed = p.if_it_happens.end_of_day;
  const lo = Math.min(fh.wide_range[0], ed.wide_range[0], 0), hi = Math.max(fh.wide_range[1], ed.wide_range[1], 0);
  b.appendChild(el("div", "sc", rangeRow("first hours", fh, lo, hi) + rangeRow("end of day", ed, lo, hi)));
  for (const r of (p.by_position || []).slice(0, 5))
    b.appendChild(el("div", "arow", `<small>${esc(r.ticker)}</small><span>${(r.move_if_it_happens * 100).toFixed(1)}%</span>${m(r.pnl_if_it_happens)}`));
  for (const sx of p.sector_news || [])   // sector legs and peer spillover (DECISIONS #217)
    b.appendChild(el("div", "arow", `<small>${esc(sx.sector)} ${sx.from_company_news ? `(spillover from ${esc(sx.from_company_news)})` : esc(sx.direction + (sx.size === "big" ? ", big" : ""))}</small><span>${(sx.holdings_it_reaches || []).map(esc).join(", ") || "none of your holdings"}</span>${m(sx.your_book_usd || 0)}`));
  for (const lg of p.company_news || [])   // headline legs (DECISIONS #213)
    b.appendChild(el("div", "arow", `<small>${esc(lg.ticker)} headline<br>${esc(lg.direction)} ${esc(lg.news_type)}</small><span>${(lg.move_middle * 100).toFixed(1)}% on ${money(lg.your_exposure_usd, false)}</span>${m(lg.pnl_middle_usd)}`));
  if (p.verdict) b.appendChild(el("div", "basis", `<b>${esc(p.verdict.call === "noise" ? "Noise for your book" : p.verdict.call === "small" ? "Small for your book" : "Meaningful for your book")}</b>: ${esc(p.verdict.say)} (normal 2-day swing ±${money(p.verdict.normal_2_day_swing_usd, false)}).`));
  if (p.mode && p.history_average_for_events_like_this && p.factor_middle_moves) {
    const h = p.history_average_for_events_like_this;
    b.appendChild(el("div", "basis", `Assumes a real market mover (oil ${esc(p.factor_middle_moves.oil)}, US stocks ${esc(p.factor_middle_moves.market)}). History's average for events like this: ${money(h.middle_usd)} (9 in 10 ${money(h.range_9_in_10_usd[0])} to ${money(h.range_9_in_10_usd[1])}).`));
  }
  for (const o of p.options || [])   // options, fully repriced (DECISIONS #209)
    b.appendChild(el("div", "arow", `<small>${esc(o.position)}</small><span>bad case ${money(o.bad_case_1_in_20_usd)}${o.max_loss_usd != null ? ` · max loss ${money(o.max_loss_usd)}` : ""}</span>${m(o.pnl_middle_usd)}`));
  const tail = (p.tail_probabilities_end_of_day || [])[0], worst = (p.nearest_past_events_on_this_book || {}).worst;
  if (worst) b.appendChild(el("div", "arow", `<small>${esc(worst.date)}<br>worst like it</small><span>${esc(worst.what)}</span>${m(worst.your_book_usd)}`));
  b.appendChild(el("div", "basis", `Bad case (1 in 20, end of day): ${money(p.bad_case_1_in_20_end_of_day)}.`
    + (tail ? ` Chance of losing more than ${money(tail.loss_over_usd, false)}: ${pct(tail.probability)}.` : "")
    + ` ${esc((p.coverage || {}).verdict || "")}`));
  c.appendChild(b);
  return c;
}

const CARDS = { risks: cardRisks, scenario: cardScenario, scan: cardScan, hedge: cardHedge, earnings: cardEarnings, stress: cardStress, upcoming: cardUpcoming, watchlist: cardWatchlist, strategy: cardStrategy, strategy_full: cardStrategy, template: cardTemplate, chart: cardChart, sources: cardSources, etf_match: cardEtfMatch };

// ---------- the "how" panel ----------
function howPanel(ev, t) {
  const d = el("details", "how"); d.appendChild(el("summary", "", "how I got this"));
  const w = el("div", "hw"), tm = ev.timings || {}, src = ev.sources || {};
  for (const r of ev.route || []) {
    const steps = (r.plan || []).map((s) => `${esc(s.expert.replace(/_/g, " "))} ${s.run ? "<b>ran</b>" : "skipped"} (${esc(s.why)})`).join("; ");
    w.appendChild(el("div", "", `<b>route</b> ${esc(r.entry)}${r.query ? ` “${esc(r.query)}”` : ""} → ${steps || "code only"}`));
  }
  if (!(ev.route || []).length && (ev.tools || []).length) w.appendChild(el("div", "", `<b>route</b> ${esc(ev.tools.join(" → "))}`));
  if (tm.total_s !== undefined) w.appendChild(el("div", "", `<b>timings</b> first card ${tm.first_card_s ?? "—"}s · total ${tm.total_s}s · router ${tm.router_llm_s}s · experts ${tm.expert_llm_s}s · ${tm.llm_calls} LLM calls`));
  if (src.build) w.appendChild(el("div", "", `<b>data</b> ${esc(src.markets)} quotes ${esc(src.quotes)} · ${esc(src.prices_options_holdings)} prices &amp; chains as of ${esc(src.as_of)} · build ${esc(src.build)} · ${esc(src.mode)}`));
  const scan = (t?.cards || []).find((c) => c.card === "scan");
  if (scan) {
    const rows = [...(scan.payload.events || []), ...(scan.payload.drift || [])];
    w.appendChild(el("div", "", `<b>β per row</b> ${rows.map((r) => `${esc(nameOf(r))}: tier ${esc(r.beta_tier)}, prior ${esc(r.prior_tier)}`).join(" · ")}`));
    w.appendChild(el("div", "", `<b>ranking</b> ${esc(scan.payload.ranking?.method || "")}; related rows grouped by series in the UI (joint-σ themes not built yet)`));
  }
  if (ev.grounding) w.appendChild(el("div", "", `<b>grounding</b> every number traced to a tool result`));
  d.appendChild(w);
  return d;
}

// ---------- estimated spend on Castle's key ----------
function showSpend(u) {
  const el_ = $("spend");
  if (!u || !u.spend) { el_.hidden = true; return; }
  const d = (v) => `$${Number(v).toFixed(2)}`;
  el_.hidden = false;
  el_.textContent = `est. spend: ${d(u.spend.hour)} last hour · ${d(u.spend.day)} today · ${d(u.spend.all)} since ${u.spend.since || "today"} · ${u.calls_day}/${u.max_day} calls today`;
  el_.title = u.spend.estimate || "";
}

// ---------- event stream ----------
function handle(ev) {
  switch (ev.type) {
    case "message": addMessage(ev); break;
    case "status": ensureTurn(ev.rid); active[ev.rid || "_"] = ev.text; showStatus(); break;
    case "card": {
      if (ev.card === "meta") { if (ev.payload?.event_id) eventNames[ev.payload.event_id] = nameOf(ev.payload); break; }
      const t = ensureTurn(ev.rid), f = CARDS[ev.card];
      t.cards.push(ev);
      if (!f) break;
      const node = f(ev.payload); node.dataset.card = ev.card;
      const prev = t.querySelector(`.card[data-card="${ev.card}"]`);  // a re-run in the same turn updates its card
      prev ? prev.replaceWith(node) : t.appendChild(node);
      break;
    }
    case "reply": {
      const t = ensureTurn(ev.rid);
      if (ev.reply_to) {   // not answering the latest message: quote the request and @mention its sender (#109)
        const who = ev.reply_to.user === "worsey" ? "" : `<span class="mention">@${esc(ev.reply_to.user)}</span> `;
        t.appendChild(el("div", "qline", `${who}${esc(clean(ev.reply_to.text))}`));
      }
      const r = el("div", "reply", md(ev.text));
      t.appendChild(r);
      const q = el("button", "quote-btn", "↩ reply");
      q.onclick = () => { quote = "worsey: " + clean(ev.text).slice(0, 140); $("quote-text").textContent = clean(ev.text).slice(0, 80); $("quote").hidden = false; input.focus(); };
      t.appendChild(q);
      break;
    }
    case "done": { if (ev.usage) showSpend(ev.usage); delete active[ev.rid || "_"]; showStatus(); const t = ensureTurn(ev.rid); if (ev.route?.length || ev.timings?.total_s) t.appendChild(howPanel(ev, t)); t.classList.add("done"); turn = null; pending = Math.max(0, pending - 1); updateDemo(); break; }
    case "script_end": log.appendChild(el("div", "note", "end of script")); break;
  }
  scrollDown();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws?chat_id=${encodeURIComponent(CHAT)}&user=${encodeURIComponent(USER)}`);
  ws.onmessage = (m) => handle(JSON.parse(m.data));
  ws.onclose = () => setTimeout(connect, 1500);
}

// ---------- composer + demo mode ----------
$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || !ws) return;
  ws.send(JSON.stringify({ type: "message", text, quote }));
  input.value = ""; quote = null; $("quote").hidden = true;
});
$("quote-x").onclick = () => { quote = null; $("quote").hidden = true; };

function updateDemo() {
  if (!cfg.replay) return;
  const left = cfg.script.length - cfg.cursor;
  $("demo").hidden = false;
  $("demo-text").textContent = left > 0 ? `demo · ${cfg.replay} · press → for the next message (${cfg.cursor + 1}/${cfg.script.length})` : `demo · ${cfg.replay} · script done`;
  $("demo-next").disabled = left <= 0 || pending > 0;
}
async function nextScripted() {
  if (!cfg.replay || typing || pending > 0 || cfg.cursor >= cfg.script.length) return;
  typing = true;
  const line = "@worsey " + cfg.script[cfg.cursor];
  input.value = "";
  for (const ch of line) { input.value += ch; await new Promise((r) => setTimeout(r, qs.has("fast") ? 0 : 18)); }
  input.value = ""; cfg.cursor++; pending++; typing = false; updateDemo();
  ws.send(JSON.stringify({ type: "next" }));
}
$("demo-next").onclick = nextScripted;
document.addEventListener("keydown", (e) => {
  if (document.activeElement === input && input.value) return;
  if (e.key === "ArrowRight" || (e.key === " " && document.activeElement !== input)) { e.preventDefault(); nextScripted(); }
});

(async function init() {
  try { cfg = await (await fetch(`/config?chat_id=${encodeURIComponent(CHAT)}`)).json(); } catch (_) { /* offline page still renders */ }
  const isDM = cfg.kind === "dm";
  $("chat-name").innerHTML = (isDM ? `DM with Worsey · ${esc(USER)}` : `${esc(CHAT)} · you are ${esc(USER)}`) + (cfg.live ? ` · live, ${esc(cfg.live.data)}` : "")
    + ` · <a class="dmlink" href="?user=${encodeURIComponent(USER)}${isDM ? "" : "&chat=dm"}">${isDM ? "back to group" : "DM Worsey"}</a>`;
  showSpend(cfg.live);
  const mem = $("members");
  for (const [u, col] of Object.entries(cfg.members || {})) { const s = el("span", "member", esc(u[0].toUpperCase())); s.style.background = col; s.title = u; mem.appendChild(s); }
  const w = el("span", "member", "W"); w.style.background = "var(--accent)"; w.title = "worsey"; mem.appendChild(w);
  if (!cfg.mention_only) input.placeholder = "Message Worsey";
  updateDemo();
  connect();
})();
