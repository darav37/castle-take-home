---
id: long_risky
name: Long the risky (long put, short call — long skew)
direction: down
vol_view: [none]
risk: undefined
gate: delta_hedged_with_education_and_margin   # The trainer: only if properly delta-hedged, plus education and margin
legs: [{type: put, side: long, strike: 'delta:0.25'}, {type: call, side: short, strike: 'delta:0.25'}]
---
## When it fits
There are two very different trades with the same name. It's important that retail understands which one they're in:
- **Delta-hedged:** A risk reversal is a pure implied skew play. It can be a realized skew play as well, meaning you're betting on whether vol actually rises when the stock falls (spot–vol correlation).
- **Not delta-hedged:** It's basically a bunch of delta, with some effects from the vol surface on top.

The convention, in trader speak:
- **Long the risky:** You're long the put and short the call, which makes you long skew. That's quite defensive. You pay up for downside protection by selling upside.
- **Short the risky:** You're short the put and long the call, which makes you short skew. This is the bullish version, where selling the rich put funds the call. It's high yield, but it's not for the faint of heart. Any retail trader who wants to go short the risky needs to really know what they're doing.

The skew view is the "why now": when put skew is rich, being short the risky gets you paid more for it, and long the risky costs more.

## What you're really betting on
- **Hedged:** the shape of the smile. Either implied put vol versus call vol moves your way, or realized spot–vol correlation does: vol picking up (long skew) or not picking up (short skew) as spot falls.
- **Unhedged:** mainly direction. Skew changes are a second-order effect on your P&L.

## How volatility helps or hurts
- **Long the risky (long skew):** It helps you when the stock sells off and the downside vols pick up. Puts get bid harder than calls, so you make money on the surface on top of your delta.
- **Short the risky (short skew):** It hurts you in exactly that scenario. The stock drops, put vol explodes, and you lose on delta *and* on skew at the same time.

*(Background: the persistent equity put skew, and the spot–vol relationship behind it, is covered in Sinclair and Natenberg. Moontower writes on why short skew behaves like selling insurance.)*

## What goes wrong
- **Short the risky in a selloff.** This is the blowup. The short put goes in the money, vol spikes, and the loss gaps through whatever you expected. You pay high yield most of the time and take a large loss occasionally.
- **Thinking an unhedged risky is a skew trade.** It isn't. Delta dominates.
- **Margin.** Short the risky requires margin at or above what your brokerage would require. Undersized margin turns a bad day into forced liquidation.
- **Naked short call on long the risky.** Long skew is defensive in a selloff, but the short call has uncapped losses in a squeeze or a big gap up.

## Common mistakes
- Mixing up the conventions. Retail sources often call "buy call, sell put" a "bullish risk reversal." Always state the legs explicitly.
- Going short the risky because it's "free" or a credit, without understanding that you're short crash risk.
- Not knowing whether you're running it hedged (a skew trade) or unhedged (a delta trade).
- Ignoring where skew sits relative to its own history before choosing a side.

## Sources / further reading
- Natenberg, *Option Volatility & Pricing*: covers risk reversals and the volatility smile/skew.
- Sinclair, *Volatility Trading* / *Positional Option Trading*: covers equity skew, spot–vol correlation, and the difference between implied and realized skew.
- Moontower (Kris Abdelmessih): writes on reading skew as insurance pricing, and on being short skew as selling crash insurance.
- Lebron: *TBD, the trainer to point to the relevant piece.*

## Playbook rules (machine-usable)
1. **Always state the legs and the convention explicitly** in plain English ("you are buying the 25d put and selling the 25d call"). Never rely on "long/short risky" alone.
2. **Gating:**
   - **Short the risky:** suggest only to sophisticated users (`user.sophistication ≥ advanced` or explicit opt-in) AND only when the user's account margin ≥ the brokerage requirement for the short put (Worsey's check should meet or exceed the broker's). Otherwise don't suggest it.
   - **Long the risky:** the short call is naked. Only open if the position is properly delta-hedged, AND the user has gone through the education (hedged vs unhedged, naked-call risk), AND margin ≥ the brokerage requirement. Otherwise don't suggest it.
3. **Hedged vs unhedged disclosure:** Always show the position delta, and state which trade the user is in:
   - unhedged → "this is mostly a delta trade; skew is a secondary effect."
   - hedged (sophisticated mode, approval required to execute) → "this is a skew trade: implied (put vs call vol) and realized (spot–vol correlation)."
4. **Skew flag (heuristic, not a signal):** `rr_skew = IV(25d put) − IV(25d call)`. `z = zscore(rr_skew, trailing ~1y)`.
   - `z` high → put skew rich → short the risky pays more / long the risky costs more. Flag it, and note why (e.g. recent selloff, event, hedging demand).
   - `z` low → skew cheap → long the risky is relatively inexpensive protection.
5. **Realized skew context:** Show recent realized spot–vol correlation, e.g. `corr(daily returns, daily change in ATM IV)` over trailing 1–3m, and compare it to what implied skew is pricing.
6. **Stress scenario (always):** Show P&L for a −10% / −20% spot move with ATM IV +10 / +20 vol points and skew steepening. For short the risky, lead with this.
7. **Margin warning:** For any short-put-bearing variant, display the margin required and the margin required under the stress scenario.
