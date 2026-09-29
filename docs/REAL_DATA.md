# Real-data results (mainnet pump.fun)

Everything before this page was measured on the simulator. This is the first run on real
mainnet history. It is one short window, so read it as a first measurement, not a verdict.

## 1. Data

| | |
|---|---|
| window | 2026-09-28 20:24 → 21:54 UTC (1.5 hours) |
| source | pump.fun bonding-curve program, replayed through a hosted archival RPC node |
| transactions fetched | 469 060 successful transactions (failed bot transactions are skipped) |
| decoded events | 500 192 |
| tokens created in the window, SOL-priced | **1 985** (1 286 train, 693 test, split by launch time) |
| excluded | 452 tokens on pump.fun's newer BuyV2 / SellV2 curves, which report no SOL amounts or reserves; tokens created before the window; pre-existing AMM pools touched by the same transactions |

Ingesting real history found and fixed five problems the simulator could not show. All
five are covered by tests:

* version-1 transactions (about 16 % of pump.fun traffic) were rejected by the RPC request;
* a fresh TLS connection per request capped the node at about 12 requests/s. Kept-alive
  connections reach about 58/s, so a replay runs at 0.63x real time;
* one-sided pool balance changes were read as swaps and divided by zero;
* BuyV2 / SellV2 trades were decoded as zero-SOL trades;
* the first sighting of an established pool (e.g. SOL/USDC) was recorded as a new launch.

## 2. Entry model: not measurable on this window

With a 1-hour horizon, a test token's outcome is only *resolved* once it has been idle for
30 minutes or has run its full hour. Test tokens launched in the last ~32 minutes of a
90-minute window, so almost none can resolve. The only ones that do are the few whose
outcome is already settled by a large move. Consequences:

* the resolved test rows are 100 % "≥ 2x", so calibration, AUC and the test NLL (0.000 against
  0.286 feature-free) are artefacts of censoring, not skill;
* 64 % of training labels are censored. The tail model learns "still running" as "large
  tail", predicts an expected payoff above the ticket for every token, and so the entry
  policy buys **every launch**. All thresholds pick the same tokens (PBO 100 %).

What *is* measurable is the cost of buying everything, which is the base rate the entry
model must beat:

| policy (0.5 SOL tickets, test tokens) | tickets | total PnL | mean multiple | median | winners |
|---|---|---|---|---|---|
| every launch (= the model on this window) | 659 | **−8.31 SOL** | 0.97x | 0.98x | 12 % |
| momentum, same budget | 646 | −14.31 SOL | 0.96x | 0.98x | – |
| random, same budget | – | −14.33 SOL (p95 −8.01) | – | – | – |

| bankroll, 100 SOL | tickets | return | max drawdown | bootstrap P(loss) |
|---|---|---|---|---|
| flat 0.5 SOL | 659 | −7.8 % | 12.5 % | 82 % |
| capital allocator | 145 | **+0.0 %** | **1.6 %** | 69 % |

Deflated Sharpe Ratio: 0.23 (per-ticket Sharpe −0.03). **There is no demonstrated entry edge
on real data.** The simulator's entry results do not transfer until labels resolve. That
needs windows of at least 8 to 12 hours, so every test token has its full horizon.

## 3. Exit model: the part that transfers

Every test token entered, only the exit differs (645 paths, decisions every 30 s):

| exit | PnL (SOL) | mean multiple | mean log multiple | winners |
|---|---|---|---|---|
| **optimal stopping, log utility** | **−4.60** | 0.99x | **−0.088** | **12 %** |
| take-profit ladder | −7.50 | 0.98x | −0.142 | 11 % |
| optimal stopping, linear utility | −11.54 | 0.96x | −0.238 | 9 % |
| sell after 5 min | −18.20 | 0.94x | −0.312 | 9 % |
| hold to horizon | −18.81 | 0.94x | −0.329 | 9 % |
| hindsight-best (ceiling) | +63.65 | 1.20x | +0.008 | 21 % |

* The log-utility stopping rule loses **39 % less than the ladder and 76 % less than
  holding**, out of sample, which matches its ranking on all six simulated markets.
* Its median holding time is **10 seconds**. On real launches the best exit is usually
  "leave almost at once": the typical token decays from the first minute. The
  exception is the tokens it keeps holding while their continuation value stays high.

## 4. Crash model (Tape Transformer): transfers well

Trained on 982 tokens, scored on 688 later tokens (snapshots every 60 s; 10 s snapshots
exceeded the 15 GB of memory on this machine):

| collapse (value halves) within | AUC | predicted | observed | Brier |
|---|---|---|---|---|
| 1 minute | **0.962** | 0.040 | 0.032 | 0.022 |
| 5 minutes | **0.965** | 0.077 | 0.066 | 0.033 |
| 15 minutes | **0.969** | 0.129 | 0.133 | 0.044 |

The collapse probabilities rank real rugs and dumps almost perfectly, and they are well
calibrated. The learned exit alarm (chosen on the tune tokens) improved the ladder from
−14.81 to −13.88 SOL on the test tokens. The tape's entry (tail) head has the same
censoring problem as section 2.

## 5. Simulator against reality

| | simulator (six markets) | real (one 1.5 h window) |
|---|---|---|
| entry edge | large, Deflated Sharpe 1.00 | not measurable; buying everything loses about 3 % per ticket |
| log-utility stopping vs ladder | better in 6 / 6 | better (−0.088 vs −0.142) |
| collapse AUC 1 / 5 / 15 min | 0.87–0.94 / 0.96–0.97 / 0.96–0.98 | 0.962 / 0.965 / 0.969 |
| allocator vs flat stake | smaller, safer returns | break-even vs −7.8 % |

## 6. What this means for paper trading

* **Do not act on the entry score with real money yet.** On real data it has not been shown
  to beat buying every launch, which loses.
* **The exit side is the usable part today**: the collapse probabilities
  (`assessment.tape["p_collapse_1m" / "_5m"]`) and `brain.hold_advice`. Log them next to
  Nardis's own exits during paper trading.
* **Keep the capital allocator's defaults.** On this window it was the only sizing that did
  not lose.
* **Next measurement:** a window of 8 to 12 hours so that entry labels resolve, then the same
  report. The ingest can now do that at 0.63x real time.

## 7. Following graduates through PumpSwap

Of the 1 985 tokens, 22 graduated to PumpSwap. All 22 were followed from creation through their
PumpSwap trading, about 9 hours after the window, with prices from PumpSwap's own
`BuyEvent` / `SellEvent` reserves. Multiples are measured from an entry 20 seconds after launch;
21 tokens had an entry price.

| | tokens |
|---|---|
| peak ≥ 2x | 10 |
| peak ≥ 10x | **2** (31.3x after 1 minute, 28.4x after 17 minutes) |
| peak ≥ 50x / ≥ 100x | **0** / **0** |
| worth under 0.1x at their last trade | 16 |

* **Every peak came within 17 minutes of launch, at or before graduation.** After graduating,
  almost every token lost nearly all of its value within hours, and most stopped trading
  within 2 hours.
* On this window, runner mode (holding for the tail) would have been harmful after
  graduation. Nardis should treat graduation as a *decision point*, not a reason to hold.
* The first decoding pass mixed other pools into these prices (reserves above the 1 billion
  supply). They are now decoded from PumpSwap's own events, and the remaining 313 implausible
  swaps from the fallback heuristic were dropped.
* One 90-minute window of launches is a small sample for a 1-in-thousands event. 100x runners
  exist but did not occur here; measuring their frequency needs days of launches.

## 8. Does the model find the 2x and 10x tokens? (2 h 20 min window, 30-minute horizon)

A second real window: 2 hours 20 minutes of pump.fun (740 050 events, **2 969 SOL-priced
launches**). Here the outcome is measured over **30 minutes after entry**, so most test
tokens' outcomes resolve, unlike the 1-hour horizon of section 2. The tail model was trained on
the earlier 65 % of tokens (34 % of labels censored at the cutoff) and scored once on the 351
later tokens whose 30-minute outcome is known. It has one entry per token, 20 s to 10 min
after launch.

| out of sample | tokens | reached 2x | reached 10x | ladder mean / median multiple |
|---|---|---|---|---|
| every launch | 351 | 11.4 % (95 % CI 8.5–15.1 %) | 1.1 % (4 tokens) | 0.97x / 0.97x |
| model's top 25 % by P(≥10x) | 88 | 18.2 % | 2.3 % (2) | 1.08x / 0.95x |
| model's top 10 % | 35 | 25.7 % (14–42 %) | 2.9 % (1) | 1.00x / 0.97x |
| **model's top 5 %** | **18** | **44.4 % (25–66 %)** | 5.6 % (1) | **1.17x / 1.09x** |
| break-even (L = 0.7) | | 23 % | 3.2 % | 1.00x |

Ranking skill, out of sample: **AUC 0.87 for reaching 2x** and **0.78 for reaching 10x**.

* **The ranking is real.** The model's top 5 % reached 2x **3.9 times** as often as an
  average launch. The 95 % interval (25–66 %) sits above the 23 % break-even for a 2x chase.
  On a window this short, that is the first real evidence of a +EV chase target (2x), and
  those tickets returned 1.17x on average against 0.97x for buying everything.
* **10x is not measurable yet.** Only 4 of 351 test tokens reached 10x within 30 minutes, so
  the top-5 % rate (1 of 18) could be anywhere from 1 % to 26 %. Judging a 3.2 % break-even
  needs hundreds more resolved launches, i.e. days of history, not hours.
* **The ladder captured 1 of the 4 10x tokens** at 10x or more. Selling tranches on the way
  up banks part of a run, not all of it. Holding the rest for 10x is a decision for the exit
  model and Nardis.
* Small samples: 18 tokens in the top 5 %. Treat this as a direction, confirmed or refuted
  by the forward ledger and the learning layer as Nardis trades.

## 9. Runner identification: which tokens go to 2x, 5x, 10x?

Same 2 h 20 min window, 30-minute horizon, snapshots every 20 s; trained on the earlier 1 930
tokens and scored on the later 1 039 (838 entries). Here a row counts as a known positive as
soon as the token has reached the target, even if the window then ends, and censored rows
below the target are left out. That keeps the rare runners, but it makes the absolute hit
rates below **optimistic** (the 2x base rate is 22 % here against 11 % on resolved rows only).
The comparison between models is fair, because they are scored on the same rows.

| target (known rows / hits) | tail model AUC | runner detector AUC | average AUC | tail: top 10 % hits |
|---|---|---|---|---|
| 2x (399 / 88) | **0.842** | 0.770 | 0.830 | 26 of 40 |
| 5x (363 / 20) | 0.824 | 0.809 | **0.832** | 10 of 36 |
| 10x (354 / 7) | **0.846** | 0.746 | 0.838 | **5 of 35** |
| 100x / 1000x | no hits in the window | | | |

* **The tail model is already a strong runner identifier on real data** (AUC 0.82–0.85 for
  2x, 5x and 10x). Its top 10 % by P(≥10x) contained 5 of the 7 tokens that reached 10x.
* **The boosted runner detector does not beat it yet.** With 7 real 10x examples there is
  too little to learn feature interactions from. The detector therefore only feeds the chase
  for targets where the average matched or beat the tail model out of sample (here: 5x). This
  gate is re-decided by every `runner-research` run as history grows.

**What identifies runners** (permutation importance on the later tokens: how much the AUC
drops when a feature is scrambled):

| rank | 2x runners | 10x runners |
|---|---|---|
| 1 | `rv_30s`: 30-second realised volatility | `rv_300s`: 5-minute realised volatility |
| 2 | `creator_prior_best_peak_log`: the creator's best previous launch | `avg_buy_size_60s_log`: average buy size |
| 3 | `bot_share_60s`: share of bot trading | `market_volume_300s_log`: market-wide volume |
| 4 | `rv_300s` | `market_launches_600s_log`: how busy the launch market is |
| 5 | `since_last_trade_log` | `dev_sold_fraction`: how much the dev has sold |
| 6 | `net_flow_300s`: net buying | `sniper_share` |
| 7 | `sell_branching_ratio` (criticality) | `buy_branching_ratio` (criticality: herding) |
| 8 | `market_launches_600s_log` | `bot_share_60s` |

Runners are identified mainly through **early volatility, buy size, the creator's track
record, how busy the whole launch market is, who is trading (bots, snipers, the dev) and the
criticality (herding) measures**. Permutation importance says which signals matter, not in
which direction. One window of 7 10x runners is a first look; the ranking will firm up as
more history is added.

## 10. A larger benchmark, and what today's detection changes are worth

A third, larger window: 3 h 40 min of pump.fun (18:14–21:54 UTC, 28 Sep; 541 781 events,
**5 006 SOL-priced launches**, 4 707 with names backfilled from their creation transactions).
It uses snapshots every 20 s and a 30-minute horizon. The earliest 65 % of tokens are
used for training and the latest 35 % for testing. Only test tokens whose full 30 minutes
lie inside the window are scored (760 tokens), so no outcome is cut off by the end of the
data.

The production tail model on this benchmark:

| target | hits / 760 | AUC | average precision | top 10 % hit rate (base rate) |
|---|---|---|---|---|
| 2x | 74 | **0.896** | 0.391 | 39 % (9.7 %) |
| 5x | 23 | **0.917** | 0.168 | 18 % (3.0 %) |
| 10x | 10 | **0.923** | 0.106 | 9.2 % (1.3 %): 7 of the 10 10x tokens |

**Before / after on identical tokens.** The same model was trained on features from the code
before today's detection changes (73 features, runner skill credited only after 90 minutes)
and after them (76 features: hits credited the moment they happen, plus narratives). The
paired token bootstrap of the AUC difference gives:

| change | 2x | 5x | 10x |
|---|---|---|---|
| all of today's changes vs old | +0.000 [−0.010, +0.009] | **+0.026 [+0.006, +0.047]** (P = 99.3 %) | +0.029 [−0.007, +0.068] (P = 94 %) |
| early runner-skill credit alone | −0.004 (n.s.) | −0.002 (n.s.) | −0.015 (n.s.) |
| **narratives** on top | +0.004 (n.s.) | **+0.028 [+0.007, +0.050]** (P = 99.6 %) | **+0.044 [+0.008, +0.081]** (P = 98.5 %) |

* **Narratives are a significant improvement for identifying big runners** (5x and 10x).
  Their intervals exclude zero, and 2x is unaffected.
* **Early runner-skill credit is neutral on this benchmark.** It is kept because it is the
  correct causal behaviour (live, a wallet's hit is known the moment it happens), but it is
  not claimed as a gain: 3.7 hours give wallet track records little time to build either way.

## 11. Trying to beat the production runner model

Six independent attempts were made to beat the production tail model on the section 10
benchmark. Each chose its configuration on an inner, time-ordered validation split of the
training tokens only, then scored the 760 test tokens once against the control, using the
same paired token bootstrap.

| attempt | what was tried | result on test |
|---|---|---|
| tuned tree detector | 131 HistGradientBoosting configurations per target, top-3 ensemble | worse at 2x (AUC 0.879 vs 0.896) and 5x (0.890 vs 0.917), level at 10x |
| stacking | tail model + direct classifiers, out-of-fold combiners | no combiner beat the best single model; no test gain |
| diverse ensemble | tail + gradient boosting + logistic + extra trees, rank-averaged | 5x AP 0.168 → 0.199, but 2x slightly worse; nothing significant |
| tail capacity | hidden 32–128, 2–4 components, dropout, 5–10 members | significantly worse at 5x and 10x; seed variance exceeds capacity effects |
| feature ablation | drop each of 10 feature families | dropping narratives looked best on validation, but lost 0.028 / 0.044 AUC at 5x / 10x on test, confirming section 10 |
| token weighting | 21 row / token / first-K weighting schemes | worse at 5x and 10x; equal weight per token, as production does, is right |

Two further attempts (label engineering and entry timing) were cut short by an
infrastructure restart and are not reported.

**Verdict: the production tail model stays.** On about 5 000 tokens, no change of model family,
size, weighting or feature set beat it. The next gains have to come from better inputs, not a
bigger model.

**Bug found and fixed: buyers were glued into one funding cluster.** When a buy pays SOL to
the bonding curve, the payment is a System transfer made inside pump.fun's instruction. It was
being recorded as *funding*, so union-find joined every buyer of a token through the curve
account. On the 22 fetched 10-minute segments (the 3 h 40 min of section 10), the median share
of a token's buyers (tokens with ≥ 20 buyers) in its largest "cluster" was **0.71** (p90 0.91). That made every
cluster feature (`holder_clusters_log`, `top_cluster_share`, `bundle_share`,
`creator_cluster_share` and the graph and tape cluster fields) measure "bought something"
rather than "shares an operator". Two fixes:

* the decoder keeps only **top-level** System transfers; payments that a program makes
  inside its own instruction (curve, fee and creator-vault payments) are not funding;
* `WalletIntel` stops merging through a **sink**: a destination paid by more than
  `hub_threshold` distinct wallets, mirroring the existing rule for exchange-like sources.

Replaying the same transfers with both fixes, the median largest-cluster share falls to
**0.04** (p90 0.11). The cluster features now describe operators. Their value for runner
detection has not been re-measured yet: the stored segments were decoded before the fix, so
a fresh fetch is needed.

**Next inputs, from the research sweep (ranked, not yet built):**

1. *Hidden supply*: in 47 % of fully observed launches a wallet sells more than it was seen
   buying (9 % of all sell SOL), so concentration features understate insider supply.
   Reading pre-trade token balances fixes this.
2. *Real funding graph*: only 5 % of buyers ever appear as a transfer destination, because
   funding happens in separate System-program transactions that are not fetched. Backfilling
   each new wallet's first funder would make fresh-wallet, bundle and creator-family features
   work.
3. *Deployer fingerprints*: creator track record is already the second-strongest 2x feature.
   Linking serial deployers that rotate addresses extends it.
4. *Co-signed multi-wallet transactions*: 3 301 in 3.7 h, 2 372 of them all-sell dumps
   across 585 tokens; a clean operator link and a coordinated-dump alarm.
5. *Wash-adjusted order flow*: 19.6 % of trades reverse the same wallet's previous trade
   in the same token within 10 s, which inflates volume and buyer counts.
