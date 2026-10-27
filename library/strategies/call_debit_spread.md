---
id: call_debit_spread
name: Long call spread
aka: [long call spread, call spread, bull call spread]
direction: up
vol_view: [none, contract]
risk: defined
legs: [{type: call, side: long, strike: atm_or_slightly_otm}, {type: call, side: short, strike: target}]
---
## When it fits
You think the stock goes higher and you want a cheaper way to say that than buying a call outright. Because you sell the top strike, the call spread costs less than the call. The price you pay is that your upside is capped.

It makes the most sense when upside call skew is expensive, meaning your upper-strike calls trade at higher vols than the lower strikes, especially to a severe degree. Selling that rich upper strike really cheapens the package. You'll usually see this in single names during chasey environments, when everyone wants tech or AI upside. In SPX or other indices you probably won't see this effect.

To judge whether skew is "expensive," compare the current level to its own history. 25d call vol minus ATM vol is a fine measure. This is a "this is relatively rich right now" read, not a tradeable signal on its own.

If you're very highly convicted and want maximum leverage, an outright call is more efficient.

## What you're really betting on
That the stock gets up toward your short strike by expiry, and not much beyond it, since you don't get paid past the top strike. Strike width and placement are about your goals:
- **Budget:** you likely have an amount of premium you want to spend, so tailor the strikes around that.
- **Leverage:** how many times your premium you get back in the best case, where you finish at or above the top strike.

How much leverage you want against how likely it is to work against how much you spend is your call as the trader. That depends on what share of your portfolio this is, how much leverage you want, and how convicted you are. Wanting to triple versus quadruple your premium is a real tradeoff, and you should understand every side of it before you pick.

## How volatility helps or hurts
Vol level drives the initial premium, so you need to understand where vol is when you buy. Near the money, you're still long vol. The stock can move higher and you can still lose money if vol collapses.

The spread carries much less vega than an outright call, and the sign flips as the stock rallies. Once spot is above your short strike, you're effectively short vol and a vol crush helps you. In practice, though, by that point you should probably have done something already: either taken the profit and run, or rolled higher to get your convexity back if you're still convicted.

*(Background: the way a vertical's delta, gamma and vega change sign depending on where spot sits relative to the strikes is covered in Natenberg's treatment of vertical spreads.)*

## What goes wrong
- **Expiry.** People need to understand expiry. If your call spread expires tomorrow and the stock is up 10%, understand what that actually means. The premium goes to zero if things don't move exactly how you need them to.
- **Vol collapse.** The stock goes up but vol comes in, and you still lose. This is mostly a near-the-money problem.
- **Theta.** You lose theta every day, and you need to be aware of that.
- **Capped upside.** If the stock rips well past your top strike, you get the same payout as if it had just touched the strike. If that's the scenario you're excited about, you wanted the outright call.

## Common mistakes
- Buying very short-dated spreads (under 5 days) when it isn't an explicit earnings play.
- Picking a tenor without looking at the calendar. Earnings, investor days, and macro events (depending on the underlier) all matter for which expiry you choose.
- Not cutting a position. Don't be afraid to cut a position if you've made some profit and feel less certain about where the stock goes from here.
- Holding a spread that's deep in profit with spot through the short strike. Your upside is basically gone at that point. If you're still convicted, roll it higher.
- Ignoring the vol level you paid. A spread bought when vol is elevated can lose money on a correct directional call.

## Sources / further reading
- Natenberg, *Option Volatility & Pricing*: covers vertical spreads, and how a spread's Greeks behave as spot moves through the strikes.
- Sinclair, *Volatility Trading* / *Positional Option Trading*: covers judging implied vol and skew against their own history rather than in absolute terms.
- Moontower (Kris Abdelmessih): writes on reading a vertical spread's price as a market-implied bet on where the stock lands, and on how skew changes the relative value of spreads versus outright options.
- Lebron: *TBD, the trainer to point to the relevant piece.*

## Playbook rules (machine-usable)
1. **Skew-rich flag (informational, not a trigger):** `skew = IV(25d call) − IV(ATM)` for the chosen expiry. If `skew ≥ 80th percentile of trailing 1y` → flag "upside skew relatively rich; call spread is cheapened by the short strike." Expect this to rarely fire on index underliers (SPX etc.).
2. **Leverage preference:** if the user states very high conviction AND wants maximum leverage → surface the outright call alongside (or instead of) the spread.
3. **Don't choose the payout ratio for the user.** For each candidate short strike, display:
   - `premium` and `premium / portfolio_value`
   - `max_payout_multiple = width / premium`
   - `P(S_T ≥ K_short)` (risk-neutral; |delta of short strike| is an acceptable proxy)
   - `breakeven = K_long + premium`
4. **Short-dated warning:** if `DTE < 5` AND there is no earnings date inside the window → warn strongly about expiry risk and theta.
5. **Event coverage check:** list earnings, investor days, and relevant macro events (FOMC/CPI etc. for index/macro-sensitive names) between today and expiry. Show whether the chosen expiry covers the user's catalyst.
6. **Always show theta** in $/day for the position.
7. **Vol-exposure alert:** if spot is near or below `K_long` → note the position is long vega (a vol drop hurts even if the stock rises). If `spot > K_short` → note the position is now effectively short vega and most of the upside is spent.
8. **Roll alert (alert, not instruction):** if `spread_value ≥ 0.5–0.7 × width` with meaningful DTE remaining → alert: "consider taking profit, or rolling strikes higher if still convicted."
9. **Cut reminder:** if the position is in profit and the user expresses reduced conviction → remind them that cutting is a legitimate choice.
