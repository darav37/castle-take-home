---
id: long_strangle
name: Long strangle
aka: [long strangle, strangle, buying the strangle]
direction: neutral
vol_view: [expand]
risk: defined
legs: [{type: put, side: long, strike: 'delta:0.25'}, {type: call, side: long, strike: 'delta:0.25'}]
---

## When it fits
Mostly the same as the straddle (see [[long_straddle]]): you need conviction that the stock moves more than it's expected to. You widen out your straddle because it's cheaper. The catch is that it could, of course, expire worthless, and that's key to all of this.

People buy strangles for two different reasons, and why you're buying it changes how you run it:
- **A short-term kurtosis play.** Make it wingy and get out early.
- **A directional view with both legs out of the money.** You think the stock will actually get through one of the strikes, so you hold that leg longer.

## What you're really betting on
That the stock gets past one of your strikes by more than you paid for the whole package, before theta eats it. Your breakevens are further out than the straddle's, so the move has to be bigger, but you've spent less to get there.

## How volatility helps or hurts
You're long vega and long gamma, like the straddle, and you pay theta every day.

Depending on how wide it is, if you go to roughly 30-delta legs and lower, you start to pick up some kurtosis as well. If there's a weird, scary vol day, your premium may pick up and counteract the theta you're losing, which can be nice. But at that point you should probably sell it.

*(Background: why far-OTM options carry exposure to the tails of the distribution, the "kurtosis" or vol-of-vol in the wings, is covered in Sinclair. Natenberg compares straddles and strangles as volatility spreads.)*

## What goes wrong
- **It expires worthless.** Both legs finish out of the money, and the whole premium is gone. That's more likely than with a straddle, and it's the trade-off for paying less.
- **Theta on short-dated strangles grows quickly.** If you're up on a vol pop and don't take it, the time decay can take it back fast.
- **Holding a kurtosis play as if it were a directional view, or the reverse.** The two get managed differently.

## Common mistakes
- Picking a strangle only because it's cheaper, without seeing how much bigger the move now has to be.
- Sitting on a short-dated strangle after a vol spike instead of taking the profit.
- On a directional strangle that's working (say you're up 3% and getting closer to the call): not thinking about keeping it. If you're convicted on direction, you can even roll the put into a higher call.

## Sources / further reading
- Natenberg, *Option Volatility & Pricing*: covers straddles vs strangles and how their Greeks compare.
- Sinclair, *Volatility Trading*: covers wing options, kurtosis, and vol-of-vol exposure.
- Moontower (Kris Abdelmessih): writes on strangles as cheaper but more binary vol bets.
- Lebron: *TBD, the trainer to point to the relevant piece.*

## Playbook rules (machine-usable)
1. **Inherit the straddle rules:** the implied move and breakevens, the event comparison, the cheap-IV flag with its value-trap caution, the path-dependency note, theta $/day, and the event-crush and post-event flags all apply.
2. **Breakevens vs the straddle:** show the strangle's breakevens and cost next to the ATM straddle's, so the trade-off (cheaper, but a bigger move needed) is explicit.
3. **Expire-worthless odds:** always show `P(K_put < S_T < K_call)`, the chance both legs finish out of the money and the whole premium is lost.
4. **Kurtosis note:** if both legs are at or below about 30 delta → note the position picks up some kurtosis (wings can get bid on a scary vol day and offset theta).
5. **Ask why:** if the user hasn't said, ask whether this is a short-term kurtosis play or a directional view with both legs out of the money. It changes rules 6–8.
6. **Short-dated / kurtosis play, profit alerts:** alert at each +10% of the entry premium (+10%, +20%, …). The default framing is "consider taking it", because theta grows quickly on short-dated strangles.
7. **Longer-dated (weeks to months), vol alert:** alert when the wing IV's daily change is ≥ 1.5 z-score vs its own history. The note: the vol pop may be the time to get out.
8. **Directional strangle working:** if spot has moved toward one strike (it's now the nearer leg and the user said directional) → note that keeping it can make sense, and that if they're convicted, they can roll the far leg (e.g. the put) into a higher strike on the winning side.
