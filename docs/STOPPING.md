# Optimal-stopping exits (`nardis_neural.solana.stopping`)

Entries get most of the attention, but for a fat-tailed launch market the exit decides
most of the result. A token that prints 50x and then rugs is worth 50x only to a holder who
sold in time. This module treats the exit as what it is mathematically: an **optimal
stopping problem**.

## 1. The problem

Holding a ticket is an American option on its own liquidation value. At each decision
time `t_k` we may sell, receiving `V_k`, or continue. `V_k` is the executable multiple of
the whole ticket if it were sold at `t_k`: the sale fills at `t_k + latency` against the
pool, with our own reserve shift, price impact and fees (`moonshot.labels.position_marks`,
the same maths as every other label).

With a utility `u`, the value of the best possible exit rule is the **Snell envelope**

```
U_K = u(V_K)
U_k = max( u(V_k),  C_k ),      C_k = E[ U_{k+1} | state_k ]
```

and the optimal rule stops at the first `k` with `u(V_k) ≥ C_k`. `C_k` is the
**continuation value**: what holding on is worth, given everything known now.

### Why log utility

With linear utility (maximise E[V]), fat tails dominate. A 1 % chance of 1000x is worth
10x, so the rule learns to hold almost everything "for the lottery". With `u(V) = log V`,
the rule maximises the expected growth rate of capital, which is the Kelly criterion
applied to the exit. It still holds a runner while its continuation value is high, but it
does not trade a likely 3x for a small chance at 100x. Both utilities are fitted and
reported; the installed model uses log.

## 2. The estimator

**Longstaff–Schwartz** (2001) estimates `C_k` by regressing realised future values on the
current state along observed paths. Here:

* **State**: all 73 causal market features at `t_k` (flow, wallets, criticality, curve
  state…), plus time held (log seconds), the current log multiple, the running peak log
  multiple and the drawdown from that peak. The peak and drawdown are path-dependent;
  they are what makes a trailing rule possible.
* **Regressor**: gradient-boosted trees (`max_depth` 4, learning rate 0.05, L2 1.0),
  exported to plain arrays (`continuation.npz`), so no scikit-learn object is pickled.
* **Fitted policy iteration**: paths have different lengths, so the regression is pooled
  across all decisions rather than run per date. Start from "hold to the end". Regress the
  realised utility of following the current policy from `k+1` on the state at `k`. Switch
  to "stop when `u(V_k) ≥ Ĉ(state_k)`", recompute the realised utilities and refit. Four
  rounds are used; the report lists how many decisions changed and the realised utility in
  each round.
* **Decision grid**: the token's own snapshots after entry, at most one per 30 s
  (`--spacing`), up to the moonshot horizon (6 h).

## 3. Honest evaluation

In-sample LSM is biased upward, because the same noise decides both the regression and
the stop. So:

* tokens are split by launch time, earliest 65 % for training;
* training paths are **truncated at the cutoff** (the first test signal): no training
  decision uses a sale that fills after the cutoff;
* the policy is scored **once** on the later tokens, over their full paths;
* **every** test token is entered at its first entry-window snapshot, so only the exit is
  compared. Entry selection is a separate engine.

Compared exits: the stopping policy (log and linear), hold to the horizon, sell after 5 or
30 minutes, the moonshot take-profit ladder (with its trailing and hard stops, which
react tick by tick rather than on the 30 s grid), and the **hindsight-best** exit (the
highest mark on the grid). The hindsight exit cannot be traded; it is the ceiling.

## 4. Results on the simulator

Two 150-launch, 12-hour markets (seed 7): `degen` and `adversarial` with herding. There are
52 test tokens in each, with 0.5 SOL tickets.

| exit | adversarial + herding: mean log x · PnL (SOL) · share > 1x | degen: mean log x · PnL (SOL) · share > 1x |
|---|---|---|
| **optimal stopping, log utility** | **+1.00** · +268 · **85 %** | **+1.07** · +171 · **88 %** |
| optimal stopping, linear utility | +0.49 · +138 · 60 % | +0.80 · +214 · 83 % |
| hold to horizon | +0.58 · +199 · 67 % | +0.66 · **+385** · 67 % |
| take-profit ladder | +0.52 · +37 · 73 % | +0.79 · +172 · 73 % |
| sell after 5 min | +0.50 · +61 · 67 % | +0.24 · +28 · 69 % |
| sell after 30 min | +0.48 · +56 · 67 % | +0.55 · +99 · 67 % |
| hindsight-best (ceiling) | +1.33 · +379 · 88 % | +1.41 · +449 · 98 % |

The log-utility policy holds for a median of about 6 minutes (355 s and 370 s).

### Robustness: six markets

The same research was repeated on seeds 19 and 23 of both markets. The table shows mean
log multiple per ticket (52 test tokens each).

| exit | adv 7 | adv 19 | adv 23 | degen 7 | degen 19 | degen 23 | mean ± sd |
|---|---|---|---|---|---|---|---|
| **optimal stopping, log** | **+1.00** | **+0.92** | **+0.52** | **+1.07** | **+0.86** | **+0.46** | **+0.81 ± 0.26** |
| optimal stopping, linear | +0.49 | +0.53 | +0.14 | +0.80 | +0.61 | +0.23 | +0.47 ± 0.25 |
| hold to horizon | +0.58 | +0.54 | +0.38 | +0.66 | +0.53 | +0.11 | +0.47 ± 0.20 |
| take-profit ladder | +0.52 | +0.48 | +0.50 | +0.79 | +0.60 | +0.23 | +0.52 ± 0.18 |
| hindsight-best (ceiling) | +1.33 | +1.28 | +1.16 | +1.41 | +1.24 | +0.78 | +1.20 ± 0.22 |

Total PnL in SOL (0.5 SOL tickets), same order: log policy +268, +274, +46, +171, +436,
+91 (mean +214); hold +199, +271, +302, +385, +357, +275 (mean +298); ladder +37, +48,
+44, +172, +180, +61 (mean +90).

How to read this:

* **Log growth per ticket, the quantity that compounds, is highest for the log-utility
  policy in all six markets.** On average it is +0.81 against +0.52 for the ladder and
  +0.47 for holding, and it captures about two thirds of the hindsight ceiling, out of
  sample. The weakest win is adversarial seed 23 (+0.52 against +0.50 for the ladder).
* **Its median ticket is the best of every tradeable rule in all six markets** (1.15x to
  2.62x). It has the highest share of winning tickets in 5 of 6 markets (60 % to 88 %; on adversarial seed 23, selling after 5 minutes wins 62 % against 60 %).
* **The trade-off is the fat right tail, and it is intended.** Holding everything to the
  horizon makes more SOL on average (+298 against +214). A few tokens that never stop
  running pay for all the rugs, and the log policy sells some of them early, because a
  Kelly bettor should. The log policy beat holding on total PnL in 2 of 6 markets, tied in
  1 and lost in 3. It beat the ladder on total PnL in 4 of 6 and tied in 2.
* **Choose by objective.** For compounding a bankroll (sizing with the capital engine,
  where a drawdown shrinks every later stake), log growth is the right target and the
  stopping policy is the best exit measured. For a small fixed lottery budget, where only
  total SOL matters, holding the runners pays more on these simulated markets.
* **Linear utility behaves as theory predicts**: it holds more and gets a lower median.
  It does not reliably beat simply holding, because the fat tail makes its regression
  target very noisy.
* The policy iteration converges: decisions that change per round fall from about 2 600 to
  about 500, and the realised training utility plateaus by round 3 to 4.
* These are simulations. Real launch markets are harsher. Paper trade the advice next to
  the ladder and compare with the forward ledger before relying on it.

## 5. Use

```bash
nardis-neural solana stopping-research --workspace ws        # research + install ws/stopping/
```

```python
brain.fit_stopping()                              # same as the command
advice = brain.hold_advice(mint, t_signal)        # a held ticket signalled at t_signal
advice["liquidation_multiple"]                    # executable multiple if sold now
advice["sell_now_utility"], advice["continuation_utility"]
advice["advantage"]                               # > 0: holding is worth more than selling now
```

`hold_advice` marks the position on the model's decision grid from the signal to now (for
the running peak and drawdown), builds the current features and compares the two values.
It is advice for the trading system, not an order. The workspace keeps
`stopping/continuation.npz`, `stopping/stopping.json` (with the research report) and
`stopping/REPORT.md`. Fitting takes seconds; the research on a 150-token market takes
about 10 minutes on a CPU, most of it building the feature snapshots.
