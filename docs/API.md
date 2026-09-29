# HTTP API reference

The addon runs next to Nardis as a local HTTP/JSON sidecar. Nardis calls it from any language.
Every answer is advice. The service holds no keys, signs nothing and sends no transactions. Its
only chain access is the read-only RPC used by the optional stream thread.

Code: [service.py](../src/nardis_neural/solana/service.py) (routing and endpoints),
[cli.py](../src/nardis_neural/solana/cli.py) (`solana serve`),
[brain.py](../src/nardis_neural/solana/brain.py) (everything the endpoints call).
Tests: [test_solana_service.py](../tests/test_solana_service.py).
Start here for the merge: [HANDOFF.md](HANDOFF.md).

Every example response below is a real response. A scratch script recorded them after starting
`make_server` on a tiny synthetic workspace: 8 simulated tokens (`simulate_launches`, seed 2), a
1-epoch, 1-member neural ensemble, and tail, stopping and Tape models fitted on that toy market
only, so that every field appears. **All numbers in the examples are synthetic and mean nothing
about real markets.** Long arrays are trimmed and marked with `…`.

Units everywhere: probabilities are in [0, 1]. Multiples are SOL returned per SOL staked, after
latency, price impact and fees, unless stated otherwise. Times are Unix seconds on the market
clock (the time of the latest ingested event). SOL amounts are in SOL, not lamports. NaN and
infinity are sent as `null`.

## Start the service

```bash
# live: stream pump.fun through a read-only RPC in the background
nardis-neural solana serve --workspace ws --port 8787 --rpc "$SOLANA_RPC_URL"

# no RPC: Nardis pushes the transactions it already receives through POST /ingest
nardis-neural solana serve --workspace ws --no-stream

# also keep learning from Nardis's trade archive
nardis-neural solana serve --workspace ws --no-stream --archive /data/nardis/parquet
```

| option | default | meaning |
|---|---|---|
| `--workspace`, `-w` | required | Solana workspace directory (from `solana bootstrap`) |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8787` | HTTP port |
| `--rpc` | env `SOLANA_RPC_URL` | read-only RPC endpoint; required unless `--no-stream` |
| `--stream / --no-stream` | `--stream` | run the background chain stream thread |
| `--poll-interval` | `2.0` | seconds between RPC polls |
| `--workers` | `6` | parallel `getTransaction` calls |
| `--pumpswap / --no-pumpswap` | `--no-pumpswap` | also poll PumpSwap (graduated tokens; much heavier) |
| `--archive`, `-a` | none | trade archive to keep training on (Parquet file or directory, or SQLite) |
| `--table` | none | SQLite table holding the trades |
| `--archive-every` | `600.0` | seconds between archive rescans |
| `--map` | none | `field=column`, repeatable |
| `--features` | none | comma-separated feature columns of the archive |
| `--return-percent` | off | the archive's return column is in percent |
| `--device` | auto | `cpu`, `cuda`, `cuda:0` or `mps` |

`serve` has **no** alert options. The push alerts described in
[INTEGRATION.md](INTEGRATION.md) (`--alert-url`, `--alert-target`, `--alert-min-edge`,
`--alert-every`) are not implemented in the CLI. Passing them fails with
`No such option: --alert-url`. See [Push alerts](#push-alerts).

The server prints `addon listening on http://HOST:PORT (stream on|off)` and serves until
Ctrl-C. On Ctrl-C it closes the socket, stops the background threads and saves the workspace.

## Overview

| method | path | inputs | returns | needs | side effects |
|---|---|---|---|---|---|
| GET | `/health` | none | clock, token count, installed models, learner and thread status | nothing | none |
| GET | `/tokens` | `active_seconds=120` | active tokens, youngest first | nothing | none |
| GET | `/moonshots` | `target=10`, `min_edge=1`, `limit=20` | entry-window tokens whose odds of `target` beat break-even by `min_edge`, best edge first | tail model | assessment bookkeeping |
| GET | `/ranking` | `limit=20` | entry-window tokens, best `chase_score` first | tail model | assessment bookkeeping |
| GET | `/assess` | `mint` | the full assessment of one token | tracked mint | assessment bookkeeping |
| POST | `/advise_trade` | `trade_id`, `mint`, `t?`, `features?`, `with_market?` | P(win), P(10x), P(100x), expected multiple, size multiplier, veto, chase | nothing | stores the proposal as pending; assessment bookkeeping when `with_market` |
| POST | `/settle_trade` | `trade_id`, `multiple`, `t_exit?`, `peak_multiple?` | accepted flag and settled count | nothing | learns the trade; refits when due |
| POST | `/hold_advice` | `mint`, `t_signal` | sell-now vs continuation utility, P(collapse) | stopping and/or Tape model | assessment bookkeeping when the mint is tracked |
| POST | `/allocate` | `equity_sol`, `open_stakes?`, `peak_equity_sol?` | recommended stakes | tail model | assessment bookkeeping |
| POST | `/ingest` | `transactions` | events accepted, rejected, undecodable | nothing | feeds the market state |
| POST | `/save` | none | `{"saved": true}` | nothing | writes the workspace to disk |

Nothing else is routed ([service.py](../src/nardis_neural/solana/service.py) lines 72 to 86).

"Assessment bookkeeping" means the call runs `SolanaBrain.assess_many`. That also queues the
assessment for continual learning (`brain.pending`), shadows the challenger model, feeds the
online moonshot buffer and feeds the forward-test ledger (paper tickets). Reads are therefore
not free of state. See [Operations](#operations).

## Conventions

* **Transport.** HTTP/1.0 (the `BaseHTTPRequestHandler` default), one connection per request,
  one thread per connection (`ThreadingHTTPServer`). Responses are
  `Content-Type: application/json` with a `Content-Length`.
* **Request bodies.** POST bodies are read by `Content-Length` and parsed as JSON. An empty body
  is `{}`. The request `Content-Type` is not checked. Chunked bodies are not supported: no
  `Content-Length` means an empty body.
* **Paths.** A trailing slash is ignored (`/health/` works). Query strings are parsed on every
  route; a repeated parameter keeps its last value. POST routes ignore the query string.
* **Numbers.** Times are Unix seconds of **chain time** (block time of the events ingested), not
  wall-clock time. SOL amounts are in SOL. Multiples are SOL returned per SOL staked.
  Probabilities are in [0, 1].
* **NaN and infinity.** Every response passes through `_clean`: non-finite floats become `null`.
* **Missing models.** Fields that come from a model that is not installed are empty objects
  (`{}`) or `null`, never invented. `GET /health` → `models` says which models are installed.
* **Concurrency.** Every request runs under one re-entrant lock (the brain is not thread-safe).
  Requests are served one at a time. Background threads take the same lock.

## Errors

| status | when | body |
|---|---|---|
| 404 | no route for this method and path (including a wrong method, e.g. `GET /advise_trade`) | `{"error": "unknown endpoint GET /advise_trade"}` |
| 400 | body is not valid JSON | `{"error": "invalid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}` |
| 400 | body is valid JSON but not an object | `{"error": "the body must be a JSON object"}` |
| 400 | the handler raised `KeyError`, `ValueError` or `TypeError` (missing field, bad number, unknown mint) | `{"error": "<ExceptionType>: <message>"}` |
| 409 | the handler raised `RuntimeError` (a required model is not installed) | `{"error": "<message>"}` (no type prefix) |
| no response | the handler raised any other exception | the connection is closed without a status line; the server keeps running |

Recorded examples:

```text
GET  /nope                          404 {"error": "unknown endpoint GET /nope"}
POST /health                        404 {"error": "unknown endpoint POST /health"}
GET  /assess                        400 {"error": "ValueError: pass ?mint=<address>"}
GET  /assess?mint=unknown           400 {"error": "KeyError: 'token unknown is not tracked'"}
GET  /tokens?active_seconds=abc     400 {"error": "ValueError: could not convert string to float: 'abc'"}
GET  /ranking?limit=x               400 {"error": "ValueError: invalid literal for int() with base 10: 'x'"}
GET  /moonshots?target=7            400 {"error": "ValueError: target must be one of 2, 5, 10, 100, 1000"}
POST /advise_trade {"mint": "x"}    400 {"error": "KeyError: 'trade_id'"}
POST /advise_trade features {"a": "high"}
                                    400 {"error": "ValueError: could not convert string to float: 'high'"}
POST /settle_trade {"trade_id": "n2"}
                                    400 {"error": "KeyError: 'multiple'"}
POST /hold_advice {"mint": …}       400 {"error": "KeyError: 't_signal'"}
POST /allocate {}                   400 {"error": "KeyError: 'equity_sol'"}
POST /ingest {"transactions": "nope"}
                                    400 {"error": "ValueError: transactions must be a list"}
GET  /ranking   (no tail model)     409 {"error": "no tail model installed: run fit_moonshot / solana moonshot-research"}
```

A `KeyError` message keeps Python's quotes (`'trade_id'`). Any `RuntimeError`, including one
raised deep inside a model, is reported as 409. Treat 409 as "not ready", 400 as "fix the
request", and a closed connection or a timeout as "no advice".

---

## GET /health

Liveness and status. Cheap: it reads counters only (1 to 3 ms on the toy workspace, including
the HTTP round trip).

**Query:** none.

| field | type | meaning |
|---|---|---|
| `ok` | bool | always `true` when the service answers |
| `market_time` | float | chain time of the newest ingested event (Unix s); the clock every other answer uses |
| `tokens` | int | tokens held in memory (tracked, not necessarily active) |
| `models.moonshot` | bool | tail model installed (`moonshot-research` or online refit); needed by `/ranking`, `/moonshots`, `/allocate` |
| `models.tape` | bool | Tape Transformer installed (`tape-research`); gives `p_collapse_*` |
| `models.stopping` | bool | exit model installed (`stopping-research`); needed for `/hold_advice` utilities |
| `models.edge` | bool | edge model installed (`edge-research`); fills `edge` in `/assess` |
| `learner.settled_trades` | int | Nardis trades learned so far (HTTP settles plus archive) |
| `learner.pending_trades` | int | advised trades not settled yet |
| `learner.deployed_levels` | list[float] | outcome levels with a deployed classifier (subset of `1, 2, 5, 10, 100, 1000`); empty means base rates |
| `learner.value_model` | bool | expected-multiple model deployed |
| `stream` | object | stream thread counters `polls`, `events`, `rejected`, `errors`; `{}` when not streaming |
| `archive` | object | archive thread `scans`, `errors`, `last` (`added`, `already_known`, `total`, `features`), `last_error` after a failure; `{}` without `--archive` |
| `alerts` | object | alert thread `scans`, `sent`, `errors`, `url`; `{}` when not running |

```bash
curl -s localhost:8787/health
```

```json
{"ok": true, "market_time": 1750005776.0000057, "tokens": 11,
 "models": {"moonshot": true, "tape": true, "stopping": true, "edge": false},
 "learner": {"settled_trades": 2, "pending_trades": 2, "deployed_levels": [], "value_model": false},
 "stream": {"polls": 6, "events": 0, "rejected": 0, "errors": 0},
 "archive": {},
 "alerts": {"scans": 3, "sent": 1, "errors": 0, "url": "http://127.0.0.1:36915/moonshot"}}
```

A freshly bootstrapped workspace (neural ensemble and risk model only) answers
`"models": {"moonshot": false, "tape": false, "stopping": false, "edge": false}`.

Notes:

* `stream.rejected` counts events the market refused: a second launch of a known mint, an
  event older than the token's last event, or an event of an unknown token.
* Not exposed: the stream reader's own `gaps` counter (polls whose backlog exceeded 50 000
  signatures and skipped older activity), the runner detector (`runner-research`) and the risk
  model.

## GET /tokens

Tokens that traded recently.

**Query:**

| parameter | type | default | meaning |
|---|---|---|---|
| `active_seconds` | float | `120` | a token is listed if its last event is at most this many seconds before `market_time` |

**Response:**

| field | type | meaning |
|---|---|---|
| `time` | float | `market_time` of the answer |
| `tokens[]` | list | sorted by `age_seconds` ascending (youngest first) |
| `tokens[].mint` | string | mint address |
| `tokens[].age_seconds` | float | `market_time` minus the launch time |
| `tokens[].venue` | string | venue **at launch**: `pump_fun`, `pumpswap`, `raydium`, `orca` or `meteora` |

```bash
curl -s 'localhost:8787/tokens?active_seconds=120'
```

```json
{"time": 1750005776.0000057,
 "tokens": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000038146973, "venue": "pump_fun"}]}
```

`venue` is the launch venue, so a graduated pump.fun token still shows `pump_fun` here while
`/assess` shows its current venue. Cost is a scan of the in-memory tokens (about 1 ms on the toy
workspace). No model is run.

## GET /ranking

Moonshot candidates: active tokens inside the tail model's entry window, not vetoed by the
manipulation guard, best `chase_score` first. It is a ranking, not an order.

**Query:**

| parameter | type | default | meaning |
|---|---|---|---|
| `limit` | int | `20` | maximum rows; negative values give an empty list |

Selection (not configurable over HTTP): last event within 120 s of `market_time`; age between the
tail model's `min_entry_age_seconds` and `max_entry_age_seconds` (defaults 20 s and 600 s in
`MoonshotSpec`); vetoed tokens removed.

**Response:** `{"time": float, "candidates": [candidate, …]}`. The candidate row is built by
`AddonService._candidate` and is shared by `/ranking`, `/moonshots` and the push alert:

| field | type | meaning |
|---|---|---|
| `mint` | string | mint address |
| `age_seconds` | float | seconds since launch |
| `chase_score` | float | guard `trust` × expected multiple (the tail/Tape blend when both models are installed); 0 when vetoed; the sort key |
| `expected_multiple` | float | `expected_multiple_blend` (mean of tail and Tape expected ladder multiple) when the Tape model is installed, else the tail model's: the expected payoff of the take-profit ladder per SOL. Not trust-adjusted |
| `chase_target` | float | highest target in 2, 5, 10, 100, 1000 whose edge exceeds 1; `0` when none. There is no proven-hits gate on this view and trust does not lower it; only a veto zeroes it |
| `chase_edge` | float | edge of `chase_target`; `0` when none |
| `tail_ev` | float | expected multiple of a ladder that banks each target reached, losers at 0.7 |
| `lottery_kelly` | float | log-optimal fraction of bankroll under the predicted payoff, × 0.25, capped at 0.05, then × `trust`; 0 when vetoed |
| `trust` | float | product of the manipulation guard's factors, in [0, 1]; more red flags never raise it |
| `p_collapse_1m`, `p_collapse_5m` | float or null | Tape model: P(the value of a ticket bought now halves) within 1 / 5 minutes; `null` without a Tape model |
| `flags` | list[string] | human-readable red flags of the token, including `moonshot veto: …` reasons |
| `p_ge_{k}x` | float | calibrated P(peak multiple ≥ k) of a ticket bought now, for k in 2, 5, 10, 100, 1000, from the tail model alone; non-increasing in k |
| `edge_{k}x` | float | P(reach k) divided by its break-even probability (above 1: +EV). For targets in the runner detector's `blend_targets`, P is the average of `p_ge_{k}x` and `runner_p_{k}x`, so `edge_{k}x` can differ from `p_ge_{k}x` ÷ break-even |
| `runner_p_{k}x` | float | runner detector's P(reach k); present only for targets a runner detector was trained on (`runner-research`) |

Break-even probabilities with a 0.7x loser ([chase.py](../src/nardis_neural/solana/chase.py)):
2x 0.2308, 5x 0.0698, 10x 0.0323, 100x 0.00302, 1000x 0.000300.

Trust does not lower `p_ge_*`, `edge_*` or `chase_target`. The guard docstring
([guard.py](../src/nardis_neural/solana/moonshot/guard.py) line 9) says trust multiplies the tail
probabilities; the code applies it only to `chase_score` and `lottery_kelly`. Check `trust`
separately.

```bash
curl -s 'localhost:8787/ranking?limit=3'
```

```json
{"time": 1750005776.0000057,
 "candidates": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000038146973,
   "chase_score": 2.1655549353104893, "expected_multiple": 2.7588456465188127,
   "chase_target": 1000.0, "chase_edge": 3.0019421633196792, "tail_ev": 4.824028049044891,
   "lottery_kelly": 0.0392474826934056, "trust": 0.7849496538681119,
   "p_collapse_1m": 0.08593140542507172, "p_collapse_5m": 0.13963219629806367, "flags": [],
   "p_ge_2x": 0.31711482972125027, "edge_2x": 1.3741642621254178,
   "p_ge_5x": 0.22767657596076118, "edge_5x": 3.2633642554375766,
   "p_ge_10x": 0.1521002368487417, "edge_10x": 4.715107342310993,
   "p_ge_100x": 0.016190618996454895, "edge_100x": 5.359094887826569,
   "p_ge_1000x": 0.0009012134984448153, "edge_1000x": 3.0019421633196792}]}
```

The toy tail model was fitted with a widened entry window (0 s to 10^6 s) so that a
1348-second-old token could appear. With the default window a token older than 600 s is never a
candidate.

**Errors:** 409 without a tail model; 400 for a non-integer `limit`.

**Cost:** one batched assessment of every candidate (neural ensemble, risk, tail, Tape). 30 to
45 ms on the toy workspace with one active token; it grows with the number of active tokens in
the entry window.

## GET /moonshots

The same candidates as `/ranking`, filtered to those whose odds of one target beat its
break-even by a factor, sorted by that factor.

**Query:**

| parameter | type | default | meaning |
|---|---|---|---|
| `target` | float | `10` | one of `2`, `5`, `10`, `100`, `1000`; anything else is a 400 |
| `min_edge` | float | `1.0` | keep candidates with `edge_{target}x ≥ min_edge` (1 = break-even, 2 = twice the odds needed) |
| `limit` | int | `20` | maximum rows, applied after filtering |

**Response:**

| field | type | meaning |
|---|---|---|
| `time` | float | `market_time` |
| `target` | float | echo of `target` |
| `min_edge` | float | echo of `min_edge` |
| `candidates[]` | list | candidate rows (see [/ranking](#get-ranking)) plus `target` and `edge` (= `edge_{target}x`), sorted by `edge` descending |

```bash
curl -s 'localhost:8787/moonshots?target=10&min_edge=1'
```

```json
{"time": 1750005776.0000057, "target": 10.0, "min_edge": 1.0,
 "candidates": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000038146973,
   "chase_score": 2.1655549353104893, "expected_multiple": 2.7588456465188127, "…": "…",
   "p_ge_10x": 0.1521002368487417, "edge_10x": 4.715107342310993, "…": "…",
   "target": 10.0, "edge": 4.715107342310993}]}
```

**Errors:** 409 without a tail model; 400 for a target outside the five chase targets or a
non-numeric parameter. A negative `limit` drops rows from the end of the list (`rows[:limit]`),
unlike `/ranking`, which clamps it to 0.

**Cost:** as `/ranking` (29 to 39 ms on the toy workspace).

## GET /assess

Everything the addon knows about one token now. Use it for inspection and logging; `/ranking`
and `/moonshots` already carry the fields most decisions need.

**Query:**

| parameter | type | default | meaning |
|---|---|---|---|
| `mint` | string | required | a mint the market is tracking |

**Response** (`SolanaAssessment` without its `features` field):

| field | type | meaning |
|---|---|---|
| `mint` | string | mint address |
| `timestamp` | float | `market_time` of the assessment |
| `venue` | string | current venue (changes on graduation) |
| `model_version` | string | champion neural model version |
| `prediction` | object | `NeuralPrediction`, see below |
| `risk` | object | calibrated P(`rug`), P(`graduation`), P(`dev_dump`) within the 300 s risk horizon; `{}` without a risk model. A rug is a ≥ 60 % price fall (unless the token graduated) or a pull of ≥ 80 % of pool SOL; a dev dump is the creator selling ≥ 50 % of its holdings |
| `risk_uncertainty` | object | ensemble spread of each risk probability |
| `round_trip_cost` | float | fractional cost of buying and selling `trade_size_sol` (default 1 SOL) now |
| `expected_net_return` | object | per horizon: E[simple return] minus `round_trip_cost` |
| `prob_net_positive` | object | per horizon: P(return beats `round_trip_cost`) |
| `flags` | list[string] | red flags (live authorities, concentration, dev selling, bundles, snipers, fresh wallets, bots, thin liquidity, OOD > 1, P(rug) > 0.5), plus `moonshot veto: <reason>` for each guard veto |
| `edge` | object | edge engine, see below; `{}` until `edge-research` |
| `moonshot` | object | tail view, see below; `{}` until a tail model is installed |
| `tape` | object | Tape Transformer view, see below; `{}` until `tape-research` |

`prediction` (neural ensemble). Horizons come from `solana.yaml`; the defaults are `15s`, `60s`
and `5m`, with upside thresholds 0.05 / 0.12 / 0.25 and downside thresholds 0.05 / 0.10 / 0.20.

| key | meaning |
|---|---|
| `expected_returns` | expected log return per horizon |
| `return_std`, `return_quantiles` (`q10`, `q50`, `q90`) | spread and quantiles of the return per horizon |
| `upside_probabilities` | calibrated P(log return > upside threshold) per horizon |
| `downside_probabilities` | calibrated P(max drawdown > downside threshold) per horizon |
| `maximum_upside`, `maximum_drawdown`, `predicted_volatility` | per horizon |
| `epistemic_uncertainty`, `aleatoric_uncertainty`, `total_uncertainty`, `confidence`, `confidence_components` | uncertainty decomposition |
| `ood_score`, `member_disagreement` | out-of-distribution score and ensemble disagreement |
| `market_embedding`, `expert_weights`, `regime_cluster` | latent vector (list of floats), mixture-of-experts gate weights, regime cluster (int or null) |

`edge` (edge engine): `p_win`, `expected_net`, `uncertainty`, `edge_score` (E[net] − λ·σ, a
lower confidence bound), `kelly_fraction`, `threshold` (chosen by the research),
`above_threshold` (1.0 / 0.0).

`moonshot` (tail model, guard, chase profile, runner detector):

| key | meaning |
|---|---|
| everything in the candidate table of [/ranking](#get-ranking) | same meaning (`p_ge_{k}x`, `edge_{k}x`, `expected_multiple`, `lottery_kelly`, `trust`, `chase_score`, `chase_target`, `chase_edge`, `tail_ev`, `runner_p_{k}x`) |
| `median_multiple` | median predicted peak multiple |
| `tail_index` | far-tail exponent; smaller is heavier |
| `epistemic` | largest ensemble spread of P(≥ k) |
| `out_of_range_share` | share of inputs outside the training range |
| `in_entry_window` | 1.0 / 0.0 |
| `vetoed` | 1.0 / 0.0 |
| `chase_rank` | 1 = best `chase_score` within the batch assessed |
| `crazy_shot` | max of `edge_100x` and `edge_1000x` |
| `expected_multiple_blend` | 0.5 × (tail + Tape expected multiple); only with a Tape model |
| `guard.<factor>` | one trust factor each, 0 to 1: `guard.rug`, `guard.mint_authority`, `guard.freeze_authority`, `guard.lp`, `guard.wash`, `guard.bundle`, `guard.creator_cluster`, `guard.rug_wallets`, `guard.hidden_cluster`, `guard.concentration`, `guard.dev_selling`, `guard.ood`, `guard.out_of_range`, `guard.epistemic` |

Guard vetoes (defaults in `GuardConfig`): P(rug) ≥ 0.6, live mint or freeze authority, bundled
supply ≥ 20 %, creator cluster ≥ 30 %, bots ≥ 80 % of the last 60 s's traders, one hidden wallet
cluster ≥ 25 % of supply, OOD score ≥ 4, or ≥ 25 % of inputs outside the training range.

`tape` (Tape Transformer): `p_ge_{k}x`, `expected_multiple`, `median_multiple`, `lottery_kelly`
(not trust-scaled), `tail_index`, `epistemic`, `p_collapse_1m`, `p_collapse_5m`,
`p_collapse_15m`, `p_collapse_1h`.

```bash
curl -s 'localhost:8787/assess?mint=2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu'
```

```json
{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "timestamp": 1750005776.0000057,
 "venue": "pump_fun", "model_version": "m-20260929T152816-f411e7",
 "prediction": {"observation_id": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu@1750005776.000",
   "horizons": ["15s", "60s", "5m"],
   "expected_returns": {"15s": -4.38987735833507e-05, "60s": 0.008976126089692116, "5m": -0.007524493150413036},
   "upside_probabilities": {"15s": 0.4229918801307934, "60s": 0.15567938814767576, "5m": 0.21436748849104295},
   "ood_score": 0.36355089675319663, "market_embedding": [0.15528155863285065, 2.4403328895568848, "…"],
   "regime_cluster": 0, "…": "…"},
 "risk": {"rug": 2.77974308830164e-06, "graduation": 0.19360276063283285, "dev_dump": 1.0335622087499622e-06},
 "risk_uncertainty": {"rug": 0.04492320628594413, "graduation": 0.04258397680960071, "dev_dump": 0.01620270425999051},
 "round_trip_cost": 0.019899999722838402,
 "expected_net_return": {"15s": -0.019055127607989004, "60s": -0.008143536418427633, "5m": -0.016923468885910262},
 "prob_net_positive": {"15s": 0.31971300392319807, "60s": 0.4420875147380754, "5m": 0.4254662016173831},
 "flags": [],
 "edge": {},
 "moonshot": {"p_ge_2x": 0.31711482972125027, "p_ge_10x": 0.1521002368487417, "p_ge_1000x": 0.0009012134984448153,
   "median_multiple": 1.3256734563689214, "expected_multiple": 2.865287458923346,
   "lottery_kelly": 0.0392474826934056, "in_entry_window": 1.0, "trust": 0.7849496538681119,
   "vetoed": 0.0, "chase_score": 2.1655549353104893, "guard.wash": 1.0,
   "guard.hidden_cluster": 0.8003358662128448, "chase_target": 1000.0, "chase_edge": 3.0019421633196792,
   "chase_rank": 1.0, "expected_multiple_blend": 2.7588456465188127, "…": "…"},
 "tape": {"p_ge_10x": 0.11964339846562753, "expected_multiple": 2.65240383411428,
   "p_collapse_1m": 0.08593140542507172, "p_collapse_5m": 0.13963219629806367,
   "p_collapse_15m": 0.17105884817507133, "p_collapse_1h": 0.25905224331192445, "…": "…"}}
```

A vetoed token looked like this in an earlier toy run (a different pushed token):
`"flags": ["creator's funding cluster holds 18.8%", "volume dominated by high-frequency bots
(possible wash trading)", "moonshot veto: wash trading: bots are 91% of volume"]`, with
`"vetoed": 1.0`, `"chase_score": 0.0`, `"lottery_kelly": 0.0`, `"chase_target": 0.0`.

**Errors:** 400 `ValueError: pass ?mint=<address>` without `mint`; 400
`KeyError: 'token … is not tracked'` for a mint not in memory (never seen, or evicted after two
idle hours in streaming mode).

**Cost:** one assessment, 22 to 77 ms on the toy workspace. The docs measure the full neural
ensemble alone at about 49 ms p50 per observation on a CPU
([ARCHITECTURE.md](ARCHITECTURE.md) §9); a real workspace will be slower than the toy one.

## POST /advise_trade

Call **before every trade**. The meta-learner scores the proposal from Nardis's own settled
trades and remembers it until `/settle_trade`.

**Body:**

| field | type | required | default | meaning |
|---|---|---|---|---|
| `trade_id` | string (any JSON value, converted with `str`) | yes | | Nardis's id of this trade; the same id must be used to settle |
| `mint` | string | yes | | token to trade; need not be tracked |
| `t` | float | no | `market_time` | proposal time (Unix s); orders trades for the time-ordered refit |
| `features` | object of name → number | no | `{}` | Nardis's own signal values at entry; each value goes through `float()` (numbers, numeric strings and booleans pass; `null` or text is a 400) |
| `with_market` | bool | no | `true` | join the addon's current view of the token as extra features |

With `with_market` and a tracked mint, the addon's view is added as features named
`addon_moonshot_*`, `addon_tape_*`, `addon_edge_*`, `addon_risk_*` and
`addon_feature_buy_branching_ratio`, `addon_feature_endogenous_buy_share`,
`addon_feature_top_cluster_share`, `addon_feature_liquidity_sol_log`. They are stored with the
proposal and are not echoed in the response.

**Response** (`TradeAdvice`):

| field | type | meaning |
|---|---|---|
| `trade_id` | string | echo |
| `p_win` | float | P(multiple > 1) |
| `p_10x` | float | P(best of peak and final multiple ≥ 10), capped by the lower targets and `p_win`; same as `chase.p_10x` |
| `p_100x` | float | same for 100 |
| `expected_multiple` | float | expected final multiple; the mean of Nardis's settled multiples before a value model is deployed (1.0 with no history), Duan-smeared model output after |
| `size_multiplier` | float | scale for Nardis's own stake, 0 to 2: (P(win) ÷ base P(win)) × (E[multiple] ÷ mean multiple), × the chase tilt, clipped to [0, 2]. Exactly 1.0 while `source` is `prior` |
| `veto` | bool | `true` only from a deployed model, when P(win) < 0.5 × base P(win) **and** expected multiple < 1. Never true from priors |
| `reason` | string | `no settled history`, `base rates only`, `learned`, or `similar trades have been losing: …`; `; chase Kx (edge E.Ex break-even)` is appended when a (proven) chase target exists and there is no veto |
| `evidence` | int | settled trades the answer rests on |
| `source` | string | `prior` (Jeffreys base rates) or `learned` (a deployed model) |
| `chase` | object | the chase profile of this proposal, next table |

The `chase` object, per target k in 2, 5, 10, 100, 1000:

| key | meaning |
|---|---|
| `p_{k}x` | P(reach k), made non-increasing in k |
| `break_even_{k}x` | (1 − L) ÷ (k − L), with L the mean losing multiple (multiple ≤ 1) of Nardis's trades, clipped to [0, 0.99], once it has 20 losers; else 0.7 |
| `edge_{k}x` | `p_{k}x` ÷ `break_even_{k}x` |
| `lift_{k}x` | `p_{k}x` ÷ Nardis's base rate of reaching k |
| `proven_{k}x` | 1.0 once at least 3 of Nardis's settled trades reached k (peak or realised), else 0.0 |
| `tail_ev` | expected multiple of a position that banks each rung reached, losers at L |
| `chase_target` | highest *proven* target with edge > 1, else 0 |
| `chase_edge` | its edge ratio |
| `crazy_shot` | max of `edge_100x` and `edge_1000x` |

```bash
curl -s localhost:8787/advise_trade -d '{"trade_id": "n2", "mint": "Mint0005RUG",
  "t": 1750004167.8084893, "features": {"nardis_score": 0.4}, "with_market": false}'
```

Response after one settled trade (a 2.5x winner with a 4x peak), so still base rates:

```json
{"trade_id": "n2", "p_win": 0.75, "p_10x": 0.25, "p_100x": 0.25, "expected_multiple": 2.5,
 "size_multiplier": 1.0, "veto": false, "reason": "base rates only", "evidence": 1, "source": "prior",
 "chase": {"p_2x": 0.75, "break_even_2x": 0.23076923076923078, "edge_2x": 3.25, "lift_2x": 1.0, "proven_2x": 0.0,
   "p_5x": 0.25, "break_even_5x": 0.06976744186046513, "edge_5x": 3.5833333333333326, "lift_5x": 1.0, "proven_5x": 0.0,
   "p_10x": 0.25, "break_even_10x": 0.03225806451612903, "edge_10x": 7.75, "lift_10x": 1.0, "proven_10x": 0.0,
   "p_100x": 0.25, "break_even_100x": 0.003021148036253777, "edge_100x": 82.74999999999999, "lift_100x": 1.0, "proven_100x": 0.0,
   "p_1000x": 0.25, "break_even_1000x": 0.00030021014710297214, "edge_1000x": 832.7499999999999, "lift_1000x": 1.0, "proven_1000x": 0.0,
   "tail_ev": 251.175, "chase_target": 0.0, "chase_edge": 0.0, "crazy_shot": 832.7499999999999}}
```

With no settled trades at all every probability is 0.5 (the Jeffreys prior (0 + 0.5) / (0 + 1)),
`reason` is `no settled history`, and the chase shows `edge_1000x` about 1665.5 and `tail_ev`
about 500.

Read the cold start carefully. With few trades, `edge_1000x` is in the hundreds or thousands
because the prior puts P(1000x) near 0.5. That is why `chase_target` stays 0 until a target is
proven by 3 real hits. While `source` is `prior`, act on `chase_target`, `proven_*`, `source` and
`evidence`, not on raw `edge_*`, `tail_ev` or `crazy_shot`.

Learning rules ([metalabel.py](../src/nardis_neural/solana/metalabel.py)): base rates until 50
trades have settled; then a refit every 25 new trades; a level's classifier needs at least 8
positives; a model is deployed only if it beats the base rate on the newest 20 % of trades.

**Errors:** 400 without `trade_id` or `mint`, or with a non-numeric feature value.

**Side effects:** the proposal (with its features, frozen now) is stored as pending under
`trade_id`, replacing any earlier pending proposal with the same id. It is persisted by the next
save. Pending proposals never expire. With `with_market` and a tracked mint, the token is
assessed (bookkeeping as above).

**Cost:** measured in the docs on a 4-core CPU: 1.3 ms median with `"with_market": false`, about
20 ms with the market view ([INTEGRATION.md](INTEGRATION.md)). The toy run measured 1.0 to 1.3 ms
and 17 to 39 ms, including the HTTP round trip.

## POST /settle_trade

Call **after every trade closes**. Learning happens here.

**Body:**

| field | type | required | default | meaning |
|---|---|---|---|---|
| `trade_id` | string | yes | | the id used in `/advise_trade` |
| `multiple` | float | yes | | SOL returned per SOL staked, fees included; negative values are stored as 0 |
| `peak_multiple` | float or null | no | `null` | best multiple the position reached; sharpens the 2x to 1000x labels (a target counts as reached when `max(peak_multiple, multiple) ≥ k`) |
| `t_exit` | float | no | `market_time` | exit time (Unix s); accepted but not used by the learner |

**Response:**

| field | type | meaning |
|---|---|---|
| `accepted` | bool | `true` if a pending proposal with this id was found and learned; `false` for an unknown or already settled id |
| `settled_trades` | int | settled trades after this call |

```bash
curl -s localhost:8787/settle_trade -d '{"trade_id": "n1", "multiple": 2.5, "peak_multiple": 4}'
```

```json
{"accepted": true, "settled_trades": 1}
```

Settling the same id again, or an id never advised, returns
`{"accepted": false, "settled_trades": 1}` with status 200. A trade that was never advised is
not learned, so call `/advise_trade` before every trade, even in shadow mode.

**Side effects and cost:** the trade is appended to the learner's history. Once at least 50
trades have settled and 25 have arrived since the last refit, the call refits every level (1x,
2x, 5x, 10x, 100x, 1000x) and the value model with gradient-boosted trees, champion /
challenger on the newest 20 % (see [INTEGRATION.md](INTEGRATION.md)). That refit runs inside the
request and holds the lock. The toy run measured about 1 ms for a settle without a refit; the
refit duration was not measured. The learner is written to disk by the next save.

**Idempotency:** a settled id is removed from pending, so a repeated settle is ignored. But
advising an already settled id again creates a new pending proposal, and settling it then adds
the trade a second time. Use a fresh `trade_id` per trade.

## POST /hold_advice

Call while a position is open: is holding worth more than selling now?

**Body:**

| field | type | required | meaning |
|---|---|---|---|
| `mint` | string | yes | token held |
| `t_signal` | float | yes | Unix time (chain time) at which the position was signalled / entered |

The addon rebuilds a `MoonshotSpec.size_sol` ticket (the installed tail model's spec, default
0.5 SOL) bought at `t_signal` plus the spec's latency (1 s by default) from its own pool data. It
does not see Nardis's real fill, size or slippage.

**Response:** an object whose keys depend on the installed models.

| field | type | when present | meaning |
|---|---|---|---|
| `liquidation_multiple` | float | stopping model installed and the position could be marked | executable multiple of that ticket if sold now, latency, impact and fees included |
| `sell_now_utility` | float | same | utility of selling now (log utility, or power utility in runner mode) |
| `continuation_utility` | float | same | estimated utility of holding on under the exit model |
| `advantage` | float | same | `continuation_utility − sell_now_utility`; above 0 means holding is worth more |
| `p_collapse_1m`, `p_collapse_5m`, `p_collapse_15m` | float | Tape model installed and the mint tracked | P(value halves within 1 / 5 / 15 minutes) |

```bash
curl -s localhost:8787/hold_advice -d '{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "t_signal": 1750004458.000002}'
```

```json
{"liquidation_multiple": 1.792699375376497, "sell_now_utility": 0.5837225148524764,
 "continuation_utility": 0.7843400877896565, "advantage": 0.20061757293718008,
 "p_collapse_1m": 0.08593140542507172, "p_collapse_5m": 0.13963219629806367,
 "p_collapse_15m": 0.17105884817507133}
```

Other recorded cases:

* no stopping and no Tape model: `{}` (status 200);
* `t_signal` in the future (nothing to mark yet): only the `p_collapse_*` keys;
* stopping model installed and an untracked mint: 400 `KeyError: 'unknown token NotATrackedMint'`.
  Without a stopping model the same untracked mint returns `{}`.

Treat an empty object as "no advice", never as "hold".

**Cost:** marks the position on the exit model's decision grid (30 s by default) from `t_signal`
to now, then runs one assessment for the collapse odds. 30 to 65 ms on the toy workspace; it
grows with the holding time.

## POST /allocate

Recommended stakes for the current moonshot candidates under the capital engine's limits
([CAPITAL.md](CAPITAL.md)). Advice only.

**Body:**

| field | type | required | default | meaning |
|---|---|---|---|---|
| `equity_sol` | float | yes | | current bankroll in SOL |
| `open_stakes` | object mint → SOL | no | `{}` | stakes Nardis already holds; they count toward exposure and position limits, and those mints get `already open` |
| `peak_equity_sol` | float or null | no | `equity_sol` | peak bankroll, for the drawdown governor |

Limits are the `AllocatorConfig` defaults and cannot be changed over HTTP: 4 % of equity per
position, 2 % of the pool's real SOL liquidity, 8 % per creator family, 30 % total exposure, 12
positions, 0.02 SOL minimum stake, stakes reaching zero at a 35 % drawdown, `kelly_scale` 1.

The fraction is `lottery_kelly × kelly_scale × exp(−epistemic ÷ 0.15) × track record × drawdown
governor`. `lottery_kelly` already includes the guard's trust. The track record is the forward
ledger's realised ÷ predicted payoff of paper tickets, 1.0 until 10 tickets have settled, clipped
to [0.25, 1.25]. Candidates with expected multiple < 1 or `lottery_kelly` ≤ 0 get `no edge`.

**Response:** `{"allocations": [allocation, …]}`, one row per candidate of `/ranking`, best
expected edge first.

| field | type | meaning |
|---|---|---|
| `mint` | string | candidate |
| `stake_sol` | float | recommended stake in SOL; 0 when cut |
| `fraction` | float | `stake_sol / equity_sol` |
| `reason` | string | binding limit: `sized`, `position cap`, `liquidity cap`, `family cap`, `exposure cap`, `position count cap`, any of these plus ` (below minimum)` when the stake fell under 0.02 SOL, `already open`, `no edge`, `drawdown governor`, `daily loss stop` |

```bash
curl -s localhost:8787/allocate -d '{"equity_sol": 10.0}'
```

```json
{"allocations": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu",
  "stake_sol": 0.392474826934056, "fraction": 0.0392474826934056, "reason": "sized"}]}
```

With `{"equity_sol": 10.0, "open_stakes": {"2niH9…sgJu": 0.3}, "peak_equity_sol": 14.0}` the
same candidate returns `"stake_sol": 0.0, "fraction": 0.0, "reason": "already open"`.

Notes:

* The HTTP call is stateless. Each call starts a fresh book, so the 15 % daily loss stop never
  triggers through this endpoint, and creator-family limits do not see the families of
  `open_stakes`. Nardis must enforce its own daily stop.
* Pass `peak_equity_sol` on every call. Without it the drawdown governor sees no drawdown.
* The CLI `solana allocate` prints the track-record scale; this endpoint does not.
* Errors: 409 without a tail model; 400 without `equity_sol`.
* Cost: as `/ranking` (27 to 41 ms on the toy workspace).

## POST /ingest

Push chain transactions that Nardis already receives, instead of (or in addition to) the
built-in RPC stream. See [/ingest transaction format](#ingest-transaction-format).

**Body:**

| field | type | required | meaning |
|---|---|---|---|
| `transactions` | list of objects | yes | `getTransaction` results in `jsonParsed` encoding; any order (sorted by `slot` before decoding) |

**Response:**

| field | type | meaning |
|---|---|---|
| `events` | int | decoded events the market accepted (launches, swaps, migrations, liquidity changes, SOL transfers) |
| `rejected` | int | decoded events the market refused (`KeyError` or `ValueError`: second launch of a mint, event older than the token's last event, event of an unknown token) |
| `undecodable_transactions` | int | transactions the decoder could not read (missing keys, bad indices, bad numbers) |

```bash
curl -s localhost:8787/ingest -d @batch.json      # {"transactions": [ … ]}
```

```json
{"events": 1956, "rejected": 0, "undecodable_transactions": 0}
```

That call carried 1956 synthetic transactions for 3 simulated tokens and took 147 to 442 ms in
two runs. Recorded edge cases:

| request | status and response |
|---|---|
| `{"transactions": [{"slot": 1, "meta": null}]}` | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 1}` |
| a failed transaction (`meta.err` not null) | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 0}` (ignored, not counted) |
| `{"transactions": [{"slot": "abc"}]}` | 400 `ValueError: invalid literal for int() with base 10: 'abc'` |
| `{"transactions": "nope"}` | 400 `ValueError: transactions must be a list` |
| `{}` | 400 `KeyError: 'transactions'` |
| `{"transactions": ["x"]}` | no response: the connection closes (unhandled `AttributeError`) |
| a transaction with `"meta": null` and a `transaction` object | no response: the connection closes (unhandled `AttributeError` in the decoder) |
| the first 5 transactions sent again | 200 `{"events": 5, …}`: accepted a second time |

Side effects: events enter the market state (and the event history unless the brain is in
streaming mode). The decoder is created on the first call and keeps per-token state (venue, pool
vaults, virtual reserves) for the life of the process; it is not saved.

There is **no de-duplication**. The built-in stream drops repeated signatures; `/ingest` does
not. A re-sent transaction is decoded again and gets a new timestamp just after the last one
(see below), so swaps and transfers are counted twice. Nardis must send each transaction once.

## POST /save

Checkpoint the workspace now: neural learner state, event history (or the compact market pickle
in streaming mode), online moonshot buffer, forward ledger, meta-learner (`meta/`), risk samples
and `solana_state.json`.

**Body:** none needed. A body, if sent, must still be a JSON object (else 400); its content is
ignored.

```bash
curl -s -X POST localhost:8787/save
```

```json
{"saved": true}
```

Cost: 112 to 121 ms on the toy workspace in non-streaming mode (it rewrites the event history
Parquet files). It holds the lock while writing.

---

## Push alerts

`AddonService.alerts(url, target=10.0, min_edge=2.0, every=5.0, limit=20, post=None,
remember_seconds=3600.0)` ([service.py](../src/nardis_neural/solana/service.py) line 179) starts
a daemon thread named `addon-alerts` that pushes each new moonshot candidate to `url`. **The
`serve` command does not start it and has no flag for it.** To use it today, start the service
from Python (below) or add the options to `serve`. Until then, Nardis can poll
`GET /moonshots?target=10&min_edge=2`, which returns the same rows.

```python
from nardis_neural.solana.brain import SolanaBrain
from nardis_neural.solana.service import AddonService, make_server

service = AddonService(SolanaBrain("ws"))
service.alerts("http://127.0.0.1:9000/moonshot", target=10.0, min_edge=2.0)
server = make_server(service, "127.0.0.1", 8787)
try:
    server.serve_forever()
finally:
    server.server_close()
    service.stop()
```

Behaviour:

* Every `every` seconds it runs the `/moonshots` logic with `target`, `limit` and `min_edge`
  under the lock.
* Each candidate whose mint was not sent in the last `remember_seconds` (wall clock) is POSTed as
  JSON, one request per candidate, with `Content-Type: application/json` and a 5 s timeout.
  With `post` set to a callable, the payload goes to that callable instead of HTTP.
* A delivery counts as sent when the call returns without raising. For HTTP that means the
  receiver answered without a 4xx or 5xx status; the response body is ignored. The mint is
  remembered only after a successful send.
* A failed delivery (connection error, timeout, 4xx or 5xx status) increments `errors`. The mint
  is not remembered, so it is retried at the next scan if it still qualifies. There is no
  backoff.
* After `remember_seconds` a mint is forgotten, so a token that still qualifies an hour later is
  sent again. The memory is in-process: a restart re-sends current candidates.
* Without a tail model every scan fails with a counted error. The thread never stops on errors.
* Deliveries are sequential and outside the lock. A slow receiver delays the next scan (up to
  5 s per candidate) but not HTTP requests.
* Counters appear in `GET /health` → `alerts`: `scans`, `sent`, `errors`, `url`.

**Payload** (recorded; one POST per candidate):

```json
{"type": "moonshot", "target": 2.0,
 "candidate": {"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000038146973,
   "chase_score": 2.1655549353104893, "expected_multiple": 2.7588456465188127,
   "chase_target": 1000.0, "chase_edge": 3.0019421633196792, "tail_ev": 4.824028049044891,
   "lottery_kelly": 0.0392474826934056, "trust": 0.7849496538681119,
   "p_collapse_1m": 0.08593140542507172, "p_collapse_5m": 0.13963219629806367, "flags": [],
   "p_ge_2x": 0.31711482972125027, "edge_2x": 1.3741642621254178, "…": "…",
   "p_ge_1000x": 0.0009012134984448153, "edge_1000x": 3.0019421633196792,
   "target": 2.0, "edge": 1.3741642621254178}}
```

`candidate` is exactly a `/moonshots` row, including `target` and `edge`. In the toy run three
scans at `every=0.2` produced one POST for the one qualifying mint (`scans: 3, sent: 1,
errors: 0`). The test suite checks that a failed first delivery is retried and then sent once.

## /ingest transaction format

Each element is the `result` of a Solana `getTransaction` call with `"encoding": "jsonParsed"`.
The built-in stream requests `"maxSupportedTransactionVersion": 1` and
`"commitment": "confirmed"`. The decoder
([decoder.py](../src/nardis_neural/solana/ingest/decoder.py)) reads:

| path | use |
|---|---|
| `slot` | ordering within a batch (`int`) |
| `blockTime` | event time (1 s resolution); `null` becomes 0 |
| `meta.err` | not null: the transaction is ignored |
| `meta.fee` | priority fee = `fee − 5000 × signatures` lamports, attached to the first swap |
| `meta.logMessages` | pump.fun and PumpSwap `Program data:` events (create, trade, complete) |
| `meta.preTokenBalances`, `meta.postTokenBalances` | AMM pool vault deltas (direction, size, reserves) |
| `meta.innerInstructions` | inner instructions (programs invoked, inner transfers) |
| `meta.loadedAddresses` | extra account keys of versioned transactions |
| `transaction.signatures` | count for the fee split |
| `transaction.message.accountKeys` | strings or `{"pubkey": …}` objects; key 0 is the payer |
| `transaction.message.instructions` | `programId`; `parsed` system `transfer` / `transferWithSeed` for SOL transfers and Jito tips |

What comes out: pump.fun `CreateEvent` → launch, `TradeEvent` → swap (virtual reserves),
`CompleteEvent` → migration; PumpSwap `BuyEvent` / `SellEvent` → swaps with exact pool reserves;
Raydium AMM v4 / CPMM, Meteora DLMM and Orca Whirlpool pools → swaps or liquidity changes from
vault deltas; outer SOL transfers of at least 0.05 SOL → funding-graph transfers; transfers to
Jito tip accounts → `jito_tip` on the first swap. A token first seen mid-stream gets an implicit
launch with creator `unknown`. Mint authorities of AMM-launched tokens are assumed revoked,
because `/ingest` has no RPC to look them up.

Timing rule: the decoder's clock is `max(blockTime, previous + 1 µs)`. Send transactions in slot
order across calls. A transaction older than the last one pushed is stamped just after it, not at
its own block time.

A pump.fun buy (synthetic, from `nardis_neural.solana.ingest.encode`):

```json
{"slot": 4375010610, "blockTime": 1750004244,
 "meta": {"err": null, "fee": 793906,
   "logMessages": ["Program data: vdt/007mYe5T/DbykrEm2Yxfu2rQKOWBpcKHRX168Jcegcm4axIWgoDrFjgAAAAAqaF+…"],
   "preTokenBalances": [], "postTokenBalances": [], "innerInstructions": []},
 "transaction": {"signatures": ["5eW5T91qLGi7P58cJkxwThrT44MUBfxwRUtgN97HU3diwAv3CxNhssEwJY9m3cApi9JMngqJZWuZNHwEUTDGGf3u"],
   "message": {"accountKeys": [{"pubkey": "F4iUDZeQKbGpYgGtQwQJF6KqMnoRuDdtpbTWpvqXRvyo", "signer": true, "writable": true}],
     "instructions": [
       {"programId": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P", "accounts": [], "data": ""},
       {"programId": "11111111111111111111111111111111", "program": "system",
        "parsed": {"type": "transfer", "info": {"source": "F4iUDZeQKbGpYgGtQwQJF6KqMnoRuDdtpbTWpvqXRvyo",
          "destination": "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT", "lamports": 787668}}}]}}}
```

It decodes to a `TokenLaunch` (implicit) and `Swap(is_buy=True, sol_amount=0.941026176,
sol_reserve=30.931615914, priority_fee=0.000788906, jito_tip=0.000787668, …)`.

Only synthetic encoder output has been pushed through `/ingest` so far, not real mainnet
`getTransaction` JSON.

---

## Operations

### Threads

| thread | started by | period | lock use | what it does |
|---|---|---|---|---|
| HTTP handlers | `make_server` (`ThreadingHTTPServer`) | per request | whole request | one thread per connection |
| `addon-stream` | `serve` unless `--no-stream` | poll every `--poll-interval` (2 s) | per 50-event chunk, then for upkeep | polls the RPC, ingests events; `resolve()` every 10 s; `maintenance()` and `evict()` every 600 s; `save()` every 300 s |
| `addon-archive` | `serve --archive` | every `--archive-every` (600 s) | whole scan | learns the archive trades it has not seen (by `trade_id`), refits once at the end when at least 50 trades are known, saves `meta/` when something was added |
| `addon-alerts` | Python only (`service.alerts`) | every 5 s | while scanning | pushes new moonshot candidates |

All three background threads are daemon threads. They stop when `service.stop()` sets the stop
event, and none of them dies on an exception: errors are counted in `/health`.

### The lock

One `threading.RLock` serialises every request and the background work. A request waits while
the stream thread runs `maintenance()` (neural adaptation or retraining, risk refits and the
online tail-model refit every 6 hours of chain time) and while the archive thread replays a large
archive. How long those pauses last on a real workspace was not measured; the tiny config's
adaptation report recorded 9.8 s. Set Nardis's client timeouts accordingly and treat a timeout as
"no advice".

### Streaming mode and memory

With the stream thread (the default) the brain switches to bounded memory: no event history is
kept, tokens idle for two hours are evicted at each maintenance, and `save()` writes
`stream/market.pkl`. Restarting on that workspace resumes from the checkpoint and from the RPC
cursor in `stream_cursor.json`.

### `--no-stream` with `/ingest`

Only the stream thread calls `resolve()`, `maintenance()`, `evict()` and the periodic `save()`,
and only it switches the brain to streaming mode. In `--no-stream` mode none of that runs:

* no automatic checkpoint (call `POST /save` yourself, for example every 5 minutes and at
  shutdown);
* the brain is not in streaming mode, so every pushed event is kept in the event history and no
  token is ever evicted: memory grows for as long as the process runs;
* assessments queued for continual learning are never labelled, the forward-test ledger never
  settles tickets (the allocator's track record stays 1.0), and no model refits from the live
  market. Only the meta-learner (`/settle_trade`, `--archive`) keeps learning.

[INTEGRATION.md](INTEGRATION.md) says the save is "also automatic every 5 minutes" and that the
sidecar runs in bounded-memory mode. Both are true only with the stream thread.

### No assessment timer

`serve` does not assess on a timer, unlike `solana stream`. Assessments, and with them continual
learning, the online tail refit and the forward ledger's paper tickets, happen only when Nardis
calls `/ranking`, `/moonshots`, `/assess`, `/allocate`, `/advise_trade` (with the market view) or
`/hold_advice`. Poll `/ranking` or `/moonshots` on a timer if those should keep learning.

### Shutdown

`serve` saves on Ctrl-C (`KeyboardInterrupt`). A plain `SIGTERM` (for example `systemctl stop` or
`docker stop`) is not handled, so the final save is skipped. What survives is the last periodic
save (at most 5 minutes old, stream mode only) or the last `POST /save`. Send `SIGINT`, or call
`POST /save` before stopping.

### Latency

Measured numbers from the docs: `advise_trade` 1.3 ms median without the market view and about
20 ms with it on a 4-core CPU ([INTEGRATION.md](INTEGRATION.md)); the neural ensemble alone about
49 ms p50 per observation, 39 ms with `mc_dropout_samples: 0`
([ARCHITECTURE.md](ARCHITECTURE.md) §9). Every figure in the endpoint sections above comes from
the toy workspace (tiny network, 1 member) and includes the HTTP round trip on 127.0.0.1. Use
them only for relative cost.

### Security

* No authentication, no TLS, no CORS headers, no request size limit (a body is read fully by its
  `Content-Length`).
* Any caller that can reach the port can poison the learner (`/settle_trade`), feed fake chain
  data (`/ingest`) and write to disk (`/save`).
* Keep the default `--host 127.0.0.1`. Binding to another address exposes all of the above. Do it
  only behind a firewall or an authenticating reverse proxy.
* The service cannot move funds: there is no wallet, key, signing or transaction-sending code on
  any path. `SOLANA_RPC_URL` is only used for read-only RPC calls.
* Access logging is switched off (`log_message` returns nothing).
* If the machine sets `HTTP_PROXY` / `HTTPS_PROXY`, make sure the client does not send
  `127.0.0.1` through the proxy (for example `NO_PROXY=127.0.0.1,localhost` or
  `curl --noproxy '*'`).

### Integration checklist for Nardis

1. At start-up call `GET /health`. Check `models` before relying on `/ranking`, `/moonshots`,
   `/allocate` and `/hold_advice`.
2. Before each trade call `POST /advise_trade` with a unique `trade_id` and stable feature names.
3. After each trade call `POST /settle_trade` with the same id, `multiple` net of fees and, when
   known, `peak_multiple`.
4. While holding, poll `POST /hold_advice`. An empty object is no advice.
5. Poll `GET /moonshots` or `GET /ranking` on a timer if the forward ledger should record.
6. With `--no-stream`, push transactions once each, in slot order, and call `POST /save`
   periodically and before shutdown.
7. Treat 400 as a client bug, 409 as "model not installed", and a closed connection or a timeout
   as "no advice". Never block trading on the sidecar being up.
