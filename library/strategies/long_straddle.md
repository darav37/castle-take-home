---
id: long_straddle
name: Long straddle
direction: neutral
vol_view: [expand]
risk: defined
legs: [{type: call, side: long, strike: atm}, {type: put, side: long, strike: atm}]
---
## When it fits
You need conviction that the market is going to move. More importantly, you need conviction that it moves *more than it's expected to*. You don't care which direction, but you're paying for a move, and the market has already priced one in.

- **Short-dated (0DTE / 1DTE, earnings, single events):** You need to know the implied move, which is the straddle price, i.e. your breakeven, and how it compares to what this stock has actually done on past similar events. If the stock doesn't move more than the breakeven, you lose money.
- **Longer-dated:** There's path dependency. Whether this works depends on realized vol versus implied vol over the life of the option, not just where the stock ends up.

When IV is sitting near the bottom of its own range, that can be a sign it might be a good time to buy. That's a rough heuristic and not a signal. Past results don't tell you anything about the future, and you should just know what you're getting into.

## What you're really betting on
That realized movement beats what the market is charging you for. On a single event, the bet is that the actual move exceeds the implied move. On a longer-dated straddle, the bet is that realized vol over the holding period beats the implied vol you paid.

*(Background: a straddle's price is roughly 0.8 × σ√T × spot, which is why it reads as the market's expected move. Natenberg and Moontower both walk through this.)*

## How volatility helps or hurts
You're long vol in every sense. You're long vega, so rising IV helps you and falling IV hurts you. You're long gamma, so big moves help you. And you pay theta every day you hold it.

On an event trade, IV is inflated going in because the event is priced in. Once the event passes, that event vol disappears. If the move wasn't big enough, you get hit twice: the stock didn't move enough, and the vol you paid for evaporated.

## What goes wrong
- **Buying earnings moves.** This is always a classic. The implied move is priced in, the stock moves (maybe even a lot), but not more than the breakeven, and the event vol crushes on the way out.
- **The value trap of buying vol just because it's low.** If realized is very low and there's good reason for it, that can continue. Having no catalyst in the window is an especially good reason to pause. Knowing whether low vol is a value trap or a real opportunity is exactly what alpha is. Worsey can't solve that for you, but it can show you both sides.
- **Theta.** You bleed every day the stock sits still.
- **Path dependency on longer tenors.** The stock can end up roughly where you'd want, but if it got there quietly, realized vol never paid for the theta.

## Common mistakes
- Not knowing the implied move (your breakeven) before buying a short-dated straddle.
- Not comparing the implied move to the stock's historical moves on the same kind of event.
- Treating low IV as automatically cheap.
- Holding through an event without understanding that event vol disappears after it.
- After the event: not thinking about whether to keep the position. Keeping it is a legitimate choice. The straddle is cheaper and has less theta now, but there's also obviously less reason for the stock to move.

## Sources / further reading
- Natenberg, *Option Volatility & Pricing*: covers straddles as the purest vega/gamma position, and the approximation linking straddle price to expected move.
- Sinclair, *Volatility Trading* / *Positional Option Trading*: covers the gap between implied and realized vol, and how event vol is typically priced relative to what actually happens.
- Moontower (Kris Abdelmessih): writes on reading a straddle as the market's implied move, and why "cheap" vol has to be judged against what's likely to be realized.
- Lebron: *TBD, the trainer to point to the relevant piece.*

## Playbook rules (machine-usable)
1. **Always show the implied move:** `implied_move_$ = straddle_price`, `implied_move_% = straddle_price / spot`. Show the breakevens as `K ± straddle_price`.
2. **Short-dated / event comparison:** If `DTE ≤ 1` OR an earnings/known event falls inside expiry → also show the stock's historical realized moves on comparable past events (e.g. last 8 earnings: abs % move close→close across the event, plus median and max). Explicitly state: "if the move is smaller than X%, you lose money."
3. **Cheap-IV flag (informational, not a signal):** If `zscore(IV_atm, trailing 1y) ≤ −1` → flag "IV is near the bottom of its 1y range." Always pair it with the value-trap caveat (rule 4).
4. **Value-trap caution:** If rule 3 fires AND (`RV_30d < IV_atm` OR there's no earnings/macro catalyst before expiry) → warn: "low vol can stay low; realized is below implied and/or no catalyst in the window." Show both sides. Don't pick a side.
5. **Longer-dated path dependency:** If `DTE > ~10` and there's no single dominant event → show `IV_atm` vs `RV_10d / RV_30d`, and note that P&L depends on realized vs implied over the life of the trade, not just the terminal price.
6. **Always show theta** in $/day, and as a % of premium.
7. **Event-crush estimate:** If an event is inside expiry → estimate the post-event IV (e.g. the next non-event expiry's IV, or the ex-event term structure). Show the approximate vega loss if the stock doesn't move.
8. **Post-event flag:** After the event date passes with the position still open → alert "event vol has come out." Show the pros of holding (cheaper, less theta) and the cons (less reason to move). Don't instruct.
9. **Delta-hedging (opt-in only):** If the position has meaningful delta AND the user has indicated a sophisticated/active mode → Worsey *may suggest* a delta hedge (gamma scalp). Never execute without explicit user approval. For default users, don't suggest it unprompted.
10. **Earnings straddle warning:** If the trade is a straddle bought before earnings → always surface the "classic retail loss" note: you need move > implied move, and vol crushes after the print.
