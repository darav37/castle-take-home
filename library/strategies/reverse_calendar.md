---
id: reverse_calendar
name: Reverse calendar (sell back, buy weighted front)
direction: neutral
vol_view: [expand, none]
risk: undefined
gate: sophisticated
# ratio 1.4 ≈ flat vega when the back leg is ~2x the front tenor (vega ~ sqrt(T)); the builder's ratio is static
legs: [{type: call, side: long, strike: atm, expiry: horizon, ratio: 1.4}, {type: call, side: short, strike: atm, expiry: back}]
---
## When it fits
Honestly, I don't know how much use a retail trader gets from a calendar. Unless you're vigilant about hedging your delta, delta will usually be a driver of P&L variance at the very least. Worsey may be able to help you stay vigilant about that. So this is for traders who are already comfortable with regular calls, puts and straddles.

A calendar is a trade on the **term structure of vol**, not on direction. There are two ways to play it:
- **Long calendar (sell front, buy back):** You're convicted that the term structure is inverted to some degree, with front vols above back vols, and that it normalizes.
- **Reverse calendar (sell back, buy weighted front):** Longer-dated vols are elevated. You sell those and buy some weighted vega in the front against them to play for a short-term spike.

It's similar to rolldown and curve-carry strategies in VIX and commodity futures curves, just for equity vol, traded through options.

## What you're really betting on
The *shape* of the vol curve: how front-month IV moves relative to back-month IV. It's not where the stock goes. The idea only stays clean while spot stays near the strike (see What goes wrong).

## How volatility helps or hurts
It depends on how you weight the legs, and it depends on how you want to play it:
- **Scaled by √T (weighted-vega neutral):** If the vol curve moves in a √T fashion, the short end moving more than the back, it doesn't matter to your position. You're isolating the shape view.
- **Not scaled, 1:1 vega (flat vega):** If the curve moves in parallel, it doesn't matter to your position. But you're long weighted vega. In a selloff the short end will likely pick up more, so you'd make more money. That makes it inherently a more defensive way to run the reverse calendar.

The long calendar is short front gamma and long back vega. It likes a quiet stock and a curve that normalizes. The reverse calendar is long front gamma and wants the short-term spike.

*(Background: the idea that vol moves roughly in proportion to 1/√T along the curve, and the time-weighted vega that follows from it, is covered in Sinclair. Moontower writes about term-structure "carry" and how it compares to futures curves.)*

## What goes wrong
- **Spot runs away from the strike.** Once you're away from the strike, you have less of a calendar and more of a skew trade. Path dependency, meaning how much gamma you realize and how you hedge your delta, now probably matters more than the original calendar idea itself.
- **Delta-driven P&L variance.** Even near the strike, unhedged delta can swamp the term-structure view.
- **The curve inverts further.** You sold the front because it looked rich, and it gets richer: an event, a selloff, or stress hitting the short end first.
- **It needs rebalancing.** This is another reason it's less great for retail. It typically requires some rebalancing to stay the trade you put on.

## Common mistakes
- Putting on a calendar without understanding *why* the curve looks the way it does. Is there an event in the front? Has there been a recent selloff? A structural reason?
- Choosing the leg ratio without realizing it changes the trade: √T-weighted versus flat vega are different bets.
- Leaving it alone after spot has moved, and treating it as the same calendar when it's now a skew/gamma position.
- Ignoring delta.

## Sources / further reading
- Natenberg, *Option Volatility & Pricing*: covers calendar (time) spreads, and how their Greeks change as spot moves away from the strike.
- Sinclair, *Volatility Trading* / *Positional Option Trading*: covers term structure, the √T approximation for how vol moves across expiries, and time-weighted vega.
- Moontower (Kris Abdelmessih): writes on vol term structure as a curve, and the analogy to roll/carry in futures curves.
- Lebron: *TBD, the trainer to point to the relevant piece.*

## Playbook rules (machine-usable)
1. **Gate:** Only suggest calendars if the user has already shown interest and some competence in calls, puts and straddles (`user.sophistication ≥ intermediate` or explicit opt-in). Never surface them to beginners unprompted.
2. **Term-structure flag (heuristic, not a signal):** `ts_spread = IV_atm(front) − IV_atm(back)`. Compute `z = zscore(ts_spread, trailing ~1y)`.
   - `z` high (e.g. ≥ +1, front rich / inverted) → candidate **long calendar**.
   - `z` low (e.g. ≤ −1, back rich) → candidate **reverse calendar**.
3. **Explain the curve:** Whenever rule 2 fires, Worsey must give the *reasons* the curve looks this way: events in the front expiry (earnings, FOMC/CPI, investor day), recent realized vol / a selloff, or a structural reason. It must also say what could push it further in the same direction (more inversion). Don't present the z-score alone.
4. **Weighting choice — show both and let the user pick:**
   - √T-weighted: `n_front / n_back = (vega_back / vega_front) × sqrt(T_front / T_back)` → weighted-vega neutral.
   - Flat vega: `n_front × vega_front = n_back × vega_back` → parallel-shift neutral, long weighted vega. Label this "more defensive in a selloff."
5. **Delta warning:** Always show the position delta and note that unhedged delta will drive P&L variance. If the user is in delta-hedging mode → offer a hedge suggestion. Never execute without approval.
6. **Moneyness drift alert:** If `|spot − K| / spot > 0.5 × front implied move` (or the front-leg |delta| moves outside ~0.35–0.65) → alert: "this is now less a calendar and more a skew/gamma trade. Consider rebalancing (re-strike) or closing."
7. **Front-expiry alert:** As the front leg nears expiry, alert the user to close or roll it. Pin and assignment risk make leaving it unmanaged a bad default.
8. **Reverse calendar risk label:** The short back leg against a smaller long front is not defined risk if unwound badly or ratio'd. Label the risk clearly, and require the sophisticated gate (rule 1).
