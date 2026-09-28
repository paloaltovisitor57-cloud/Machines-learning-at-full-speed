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

How to read this:

* **Log growth per ticket, the quantity that compounds, is highest for the stopping
  policy in both markets.** It is +1.00 and +1.07 against +0.52 to +0.79 for the ladder
  and +0.58 to +0.66 for holding. It captures about 75 % of the hindsight ceiling in both
  markets, out of sample.
* **Its median ticket is the best of every rule**: 1.66x and 2.62x, against 1.15x to 1.61x
  for the others. It also has the highest share of winning tickets, 85 % and 88 %.
* **The trade-off is the fat right tail, and it is intended.** In the degen market,
  holding everything to the horizon makes the most SOL (+385). A few tokens that never
  stop running pay for all the rugs. The log policy sells some of those runners early,
  because a Kelly bettor should. In the adversarial market, where runners are rarer and
  rugs are staged, the log policy also wins on total PnL (+268 against +199).
* **Linear utility behaves as theory predicts**: it holds more and gets a lower median.
  It does not reliably beat simply holding, because the fat tail makes its regression
  target very noisy.
* The policy iteration converges: decisions that change per round fall from 2 600 to about
  500, and the realised training utility plateaus by round 3 to 4.
* These are simulations with one seed per market. Real launch markets are harsher. Paper
  trade the advice next to the ladder and compare with the forward ledger before relying
  on it.

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
