# The chase: 2x, 5x, 10x, 100x, 1000x (`nardis_neural.solana.chase`)

The addon always chases the same five multiples. `CHASE_TARGETS = (2, 5, 10, 100, 1000)` is a
constant, not a setting. Every token assessment, every ranking and every piece of trade
advice is scored against all five, and a configuration cannot remove them: `MoonshotSpec`
adds back any target left out of its levels. Tests lock both.

## 1. When is a chase worth it?

A position that misses its target is assumed to end at `L`, a cut loser (0.7x measured on
real pump.fun data; the learning layer uses the average loser of Nardis's own history once
it has 20 of them). Chasing `k` has positive expectancy when

```
p · k + (1 − p) · L > 1   ⇔   p > p* = (1 − L) / (k − L)
```

| target | 2x | 5x | 10x | 100x | 1000x |
|---|---|---|---|---|---|
| break-even P(reach), L = 0.7 | 23 % | 7.0 % | 3.2 % | 0.30 % | 0.030 % |

`edge_{k}x = p / p*`. Above 1 the chase pays on average; 3 means three times the odds needed.

## 2. What every answer carries

For each target: `p_{k}x`, `break_even_{k}x`, `edge_{k}x`, `lift_{k}x` (against the average
trade, in trade advice) and `proven_{k}x`. Then:

* **`chase_target`**: the most ambitious multiple that is +EV (edge > 1) *and proven*, or 0.
  This is the number Nardis should aim for on this trade.
* `chase_edge`: its edge ratio.
* `tail_ev`: expected multiple of a position that banks each rung it reaches.
* `crazy_shot`: the edge of the 100x / 1000x chase.

Probabilities are made monotone (reaching 10x implies reaching 5x, and so on).

**Proven means real hits.** A Bayesian prior on a few hundred trades still gives 1000x a
probability above its 0.03 % break-even. So in trade advice a target can only become the
`chase_target` once Nardis's history holds at least 3 real hits at that level; until then
its edge is reported but marked unproven. A token the manipulation guard vetoes is never a
chase, whatever its tail looks like.

## 3. Where it shows up

| where | fields |
|---|---|
| `SolanaAssessment.moonshot` (every assessment) | `chase_target`, `chase_edge`, `tail_ev`, `crazy_shot`, `edge_2x` … `edge_1000x` |
| `GET /ranking` | the same, per candidate |
| `TradeAdvice.chase` / `POST /advise_trade` | the full profile for Nardis's proposal, learned from its own trades |
| `TradeAdvice.size_multiplier` | tilted up (at most 1.5x, within the overall cap of 2) for a proven chase: `1 + 0.25·log2(edge)` |
| `TradeAdvice.reason` | e.g. `learned; chase 10x (edge 2.4x break-even)` |

The learning layer trains a classifier for every target (win, 2x, 5x, 10x, 100x, 1000x) from
the peak multiples Nardis reports. Each one needs at least 8 hits and 8 misses before it is
tried, and it is only deployed if it beats the base rate on Nardis's newest trades.

## 4. What it does not do

It chases returns without betting the account. The size caps, the drawdown governor and the
daily loss stop stay in force. With 10x hit rates in the low single digits, a strategy
that stakes everything on each shot is ruined long before the hits compound. Staying in the
game is what lets a proven 10x edge compound.
