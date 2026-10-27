---
id: put_debit_spread
name: Long put spread
aka: [long put spread, put spread, bear put spread, put debit spread]
direction: down
vol_view: [none, contract]
risk: defined
legs: [{type: put, side: long, strike: atm_or_slightly_otm}, {type: put, side: short, strike: target}]
---

## When it fits
Puts are expensive, and retail traders should be very aware of that. The skew exists, especially for equities and equity indices. So puts should be used for one of two things:
- **Protection.** You really think about the sizing and the scenarios. Resigning yourself to burning the premium for the protection is totally fine for a retail trader.
- **A view.** You're super highly convicted on a reversion or a down move soon.

The outright put will obviously be more profitable if it's well executed and you have high conviction. The put spread is the cheaper version. Basically all of the call spread logic applies (see [[call_debit_spread]]): you're giving up the payout past the lower strike to pay less. On top of that, you're selling the more expensive vol at the lower strike and buying the cheaper vol at the upper strike, which cheapens the spread overall.

## What you're really betting on
That the stock gets down toward your short strike by expiry, but not far beyond it, since you don't get paid below the lower strike. As with the call spread, pick the strikes around your premium budget, or around the leverage you want (best-case payout ÷ premium).

When you hold it as protection, you're not really betting. You're giving up cash for protection. If the market doesn't crash, you just lose that money, and you should be up front with yourself about that.

## How volatility helps or hurts
Near the money it's a long-vol position, like the call spread, but the net vega is small because you're short the richer lower strike. In a selloff, vol and skew usually pick up together, so the lower strike you're short gets bid too, which caps what the spread can gain from the vol spike.

*(Background: the persistent put skew in equities, and why selling the lower strike cheapens a put spread, are covered in Natenberg's and Sinclair's treatments of skew and verticals. Moontower writes about put skew as the price of crash insurance.)*

## What goes wrong
- **The crash tail.** As a hedge, the put spread stops paying below the short strike, which is exactly where a crash hurts most. Anyone using it for protection needs to see what the portfolio looks like in a move well past the short strike.
- **Rolling down mid-crash is expensive.** You can roll the put spread down along the way. But at that point you'd expect vol to be spiking on the crash, so rolling could cost more than you'd expect.
- **No crash.** On a short-term put, if the market doesn't crash you just lose the premium.
- **Expiry and theta.** Same as the call spread: understand what short-dated really means, and you lose theta every day.

## Common mistakes
- Treating puts as cheap because the premium looks small, without realizing skew makes them structurally expensive.
- Using a put spread as "protection" without seeing the uncovered tail below the short strike.
- Sizing protection by feel instead of against the portfolio's actual exposure.
- Holding a longer-term protective put spread in the book without knowing what it costs per year to keep rolling it.
- Under 5 days to expiry, with no explicit event, it's the same warning as the call spread.

## Sources / further reading
- Natenberg, *Option Volatility & Pricing*: covers vertical spreads, and how skew changes their relative value.
- Sinclair, *Volatility Trading* / *Positional Option Trading*: covers equity put skew and the cost of carrying protection.
- Moontower (Kris Abdelmessih): writes on put skew as insurance pricing and on the cost of hedging.
- Lebron: *TBD, the trainer to point to the relevant piece.*

## Playbook rules (machine-usable)
1. **Inherit the call spread rules:** the payout/odds/premium table, under-5-days warning, event coverage, theta/day, vol exposure, and the roll alert at 50–70% of max all apply, mirrored for puts.
2. **Skew context (informational):** always note that put skew makes puts structurally expensive in equities and indices. Show `skew_rr` and its 1y z-score when history exists, and the vol difference between the long and short strikes (the "selling the richer vol" benefit).
3. **Outright comparison:** if the user is highly convicted on a near-term down move, or says they want protection and accept burning the premium, show the outright put alongside the spread.
4. **Uncovered tail (always, when used as a hedge):** show the hedged portfolio's P&L at a move past the short strike (−20% and −30%), next to the unhedged P&L, so the gap below the short strike is obvious.
5. **Roll-down cost note:** when the user asks about rolling down during a selloff (or spot is through the short strike), note that vol is likely spiking, so the roll will cost more. Show the reprice at the stressed vol.
6. **Sizing for protection:** size against the beta-adjusted portfolio notional, and show the hedge ratio (spread delta-notional ÷ beta-adjusted notional).
7. **Annualized cost:** for a protective spread meant to stay in the book, show `premium / portfolio_value × (365 / DTE)` as "% of the portfolio per year if rolled."
8. **Short-term protection disclosure:** for a short-dated protective put spread, state plainly: "if the market doesn't drop past X, you lose this premium. You're giving up cash for protection."
