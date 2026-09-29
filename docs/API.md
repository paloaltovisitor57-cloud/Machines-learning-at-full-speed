# HTTP API reference

The addon runs next to Nardis as a local HTTP/JSON sidecar. Nardis calls it from any language.
Every answer is advice. The service holds no keys, signs nothing and sends no transactions. Its
only chain access is the read-only RPC used by the optional stream thread.

Code: [service.py](../src/nardis_neural/solana/service.py) (routing and endpoints),
[cli.py](../src/nardis_neural/solana/cli.py) (`solana serve`),
[brain.py](../src/nardis_neural/solana/brain.py) (everything the endpoints call).
Tests: [test_solana_service.py](../tests/test_solana_service.py).
Start here for the merge: [HANDOFF.md](HANDOFF.md).

Every example response below is a real response of the current code (commit `d8785d4` plus the
later `/ingest` and decoder changes), recorded on 29 September 2026. A scratch script started
`make_server` on a tiny synthetic workspace: 8 simulated tokens (`simulate_launches`, seed 2), a
1-epoch, 1-member neural ensemble, and tail, stopping and Tape models fitted on that toy market
only, so that every field appears. The service ran its background thread without a streamer, as
`serve --no-stream` does. The alert, SIGTERM and start-up behaviour was then checked with the real
`nardis-neural solana serve` command on the same workspace. **All numbers in the examples are
synthetic and mean nothing about real markets.** Long arrays are trimmed and marked with `…`.

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

# also push each new moonshot candidate to Nardis the moment it qualifies
nardis-neural solana serve --workspace ws --archive /data/nardis/parquet \
    --alert-url http://127.0.0.1:9000/moonshot --alert-target 10 --alert-min-edge 2
```

| option | default | meaning |
|---|---|---|
| `--workspace`, `-w` | required | Solana workspace directory (from `solana bootstrap`) |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8787` | HTTP port |
| `--rpc` | env `SOLANA_RPC_URL` | read-only RPC endpoint; required unless `--no-stream` |
| `--stream / --no-stream` | `--stream` | poll the chain in the background thread; with `--no-stream` the thread only does the upkeep |
| `--poll-interval` | `2.0` | seconds between RPC polls (between upkeep rounds with `--no-stream`) |
| `--workers` | `6` | parallel `getTransaction` calls |
| `--pumpswap / --no-pumpswap` | `--no-pumpswap` | also poll PumpSwap (graduated tokens; much heavier) |
| `--archive`, `-a` | none | trade archive to keep training on (Parquet file or directory, or SQLite) |
| `--table` | none | SQLite table holding the trades |
| `--archive-every` | `600.0` | seconds between archive rescans |
| `--map` | none | `field=column`, repeatable |
| `--features` | none | comma-separated feature columns of the archive |
| `--return-percent` | off | the archive's return column is in percent |
| `--alert-url` | none | POST each new moonshot candidate as JSON to this URL; without it no alert thread runs |
| `--alert-target` | `10.0` | chase target of the alerts; must be 2, 5, 10, 100 or 1000 (anything else exits with code 2) |
| `--alert-min-edge` | `2.0` | alert when `edge_{target}x` is at least this |
| `--alert-every` | `5.0` | seconds between alert scans |
| `--assess-every` | `10.0` | seconds between assessment rounds of the active tokens; `0` turns them off |
| `--device` | auto | `cpu`, `cuda`, `cuda:0` or `mps` |

The server prints `addon listening on http://HOST:PORT (stream on|off)` and serves until Ctrl-C
or SIGTERM. Either one closes the socket, stops the background threads and saves the workspace
(with the stream on, it then commits the RPC cursor the saved state covers). With the real
command on the toy workspace, SIGTERM exited with code 0 after 1.1 s and rewrote
`solana_state.json`, `pending.pkl`, `stream/market.pkl` and `meta/meta.json`.

`serve` always starts one background thread (`addon-stream`). With the stream on it polls the
RPC; with `--no-stream` it only does the upkeep. Either way it assesses the active tokens, resolves
matured outcomes, evicts idle tokens, runs maintenance and saves every 5 minutes. See
[Operations](#operations).

## Overview

| method | path | inputs | returns | needs | side effects |
|---|---|---|---|---|---|
| GET | `/health` | none | clock, token count, installed models, learner, thread and push counters | nothing | none |
| GET | `/tokens` | `active_seconds=120` | active tokens, youngest first | nothing | none |
| GET | `/moonshots` | `target=10`, `min_edge=1`, `limit=20` | entry-window tokens whose odds of `target` beat break-even by `min_edge`, best edge first | tail model | assessment bookkeeping |
| GET | `/ranking` | `limit=20` | entry-window tokens, best `chase_score` first | tail model | assessment bookkeeping |
| GET | `/assess` | `mint` | the full assessment of one token | tracked mint | assessment bookkeeping |
| POST | `/advise_trade` | `trade_id`, `mint`, `t?`, `features?`, `with_market?` | P(win), P(10x), P(100x), expected multiple, size multiplier, veto, chase | nothing | stores the proposal as pending; assessment bookkeeping when `with_market` |
| POST | `/settle_trade` | `trade_id`, `multiple`, `t_exit?`, `peak_multiple?` | accepted flag and settled count | nothing | learns the trade; refits when due |
| POST | `/hold_advice` | `mint`, `t_signal` | sell-now vs continuation utility, P(collapse) | stopping and/or Tape model | assessment bookkeeping when the mint is tracked |
| POST | `/allocate` | `equity_sol`, `open_stakes?`, `peak_equity_sol?`, `day_start_equity_sol?` | recommended stakes | tail model | assessment bookkeeping; remembers the day's book in `capital/book.json` |
| POST | `/ingest` | `transactions` | events accepted, rejected, undecodable, duplicates | nothing | feeds the market state |
| POST | `/save` | none | `{"saved": true}` | nothing | writes the workspace to disk and commits the stream cursor |

Nothing else is routed ([service.py](../src/nardis_neural/solana/service.py) lines 105 to 119).

"Assessment bookkeeping" means the call runs `SolanaBrain.assess_many`. That also queues the
assessment for continual learning (`brain.pending`), shadows the challenger model, feeds the
online moonshot buffer and feeds the forward-test ledger (paper tickets). Reads are therefore
not free of state. The background thread runs the same bookkeeping on its own every
`--assess-every` seconds. See [Operations](#operations).

## Conventions

* **Transport.** HTTP/1.0 (the `BaseHTTPRequestHandler` default), one connection per request,
  one thread per connection (`ThreadingHTTPServer`). Responses are
  `Content-Type: application/json` with a `Content-Length`.
* **Request bodies.** POST bodies are read by `Content-Length` and parsed as JSON. An empty body
  is `{}`. The request `Content-Type` is not checked. Chunked bodies are not supported: no
  `Content-Length` means an empty body. A negative or non-numeric `Content-Length` is a 400. The
  largest body accepted is 16 MiB (`MAX_BODY_BYTES`, 16 777 216 bytes); a larger one is a 413.
  Bodies up to 64 MiB (`MAX_DRAIN_BYTES`) are read and discarded first, so an ordinary client
  sees the 413. Above 64 MiB the body is not read and the connection is closed after the answer,
  which some clients report as a reset.
* **Paths.** A trailing slash is ignored (`/health/` works). Query strings are parsed on every
  route; a repeated parameter keeps its last value. POST routes ignore the query string.
* **Numbers.** Times are Unix seconds of **chain time** (block time of the events ingested), not
  wall-clock time. SOL amounts are in SOL. Multiples are SOL returned per SOL staked.
  Probabilities are in [0, 1].
* **NaN and infinity.** Every response passes through `_clean`: non-finite floats become `null`.
* **Missing models.** Fields that come from a model that is not installed are empty objects
  (`{}`) or `null`, never invented. `GET /health` → `models` says which models are installed.
* **Concurrency.** Every request runs under one re-entrant lock (the brain is not thread-safe).
  Requests are served one at a time. Background threads take the same lock for the work that
  touches the brain (see [The lock](#the-lock)).

## Errors

Every failed request gets an HTTP status and a JSON body `{"error": …}`, and the server keeps
running after any of them.

| status | when | body |
|---|---|---|
| 400 | `Content-Length` negative or not a number (the connection is then closed) | `{"error": "invalid Content-Length"}` |
| 400 | body is not valid JSON, including invalid UTF-8 and absurdly deep nesting | `{"error": "invalid JSON: <parser message>"}` |
| 400 | body is valid JSON but not an object | `{"error": "the body must be a JSON object"}` |
| 400 | the handler raised `KeyError`, `ValueError` or `TypeError` (missing field, bad number, unknown mint, a non-object element in `/ingest`) | `{"error": "<ExceptionType>: <message>"}` |
| 404 | no route for this method and path (including a wrong method, e.g. `GET /advise_trade`) | `{"error": "unknown endpoint GET /advise_trade"}` |
| 409 | `/ranking`, `/moonshots` or `/allocate` while no tail model is installed | `{"error": "no tail model installed: run fit_moonshot / solana moonshot-research"}` |
| 413 | `Content-Length` above 16 MiB (the connection is then closed) | `{"error": "body over 16777216 bytes"}` |
| 500 | any other exception inside a handler, including a `RuntimeError` while the tail model is installed, or an answer JSON cannot hold | `{"error": "internal error: <ExceptionType>: <message>"}` (cut to 500 characters) |

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
POST /advise_trade {not json        400 {"error": "invalid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}
POST /advise_trade <invalid UTF-8>  400 {"error": "invalid JSON: Expecting value: line 1 column 1 (char 0)"}
POST /advise_trade [[[…]]] 100 000 deep
                                    400 {"error": "invalid JSON: maximum recursion depth exceeded while decoding a JSON array from a unicode string"}
POST /advise_trade [1, 2]           400 {"error": "the body must be a JSON object"}
POST /settle_trade {"trade_id": "n2"}
                                    400 {"error": "KeyError: 'multiple'"}
POST /hold_advice {"mint": …}       400 {"error": "KeyError: 't_signal'"}
POST /allocate {}                   400 {"error": "KeyError: 'equity_sol'"}
POST /ingest {"transactions": "nope"}
                                    400 {"error": "ValueError: transactions must be a list"}
POST /ingest {"transactions": ["x"]}
                                    400 {"error": "ValueError: every transaction must be a JSON object"}
POST /ingest  Content-Length: -5    400 {"error": "invalid Content-Length"}
POST /ingest  Content-Length: abc   400 {"error": "invalid Content-Length"}
POST /ingest  17 825 823-byte body  413 {"error": "body over 16777216 bytes"}
POST /ingest  Content-Length 64 MiB + 1, no body sent
                                    413 {"error": "body over 16777216 bytes"}
GET  /ranking   (no tail model)     409 {"error": "no tail model installed: run fit_moonshot / solana moonshot-research"}
GET  /assess    (fault injected: ZeroDivisionError in assess)
                                    500 {"error": "internal error: ZeroDivisionError: division by zero"}
GET  /assess    (fault injected: RuntimeError, tail model installed)
                                    500 {"error": "internal error: RuntimeError: internal failure deep in a model"}
```

The 17 825 823-byte body was sent in full by an ordinary client, which read the 413. No request
without an injected fault produced a 500 in the recording; the two 500 rows replaced
`brain.assess` with a function that raises, as the test suite does. `GET /health` answered 200
after each of them.

A `KeyError` message keeps Python's quotes (`'trade_id'`). Treat 400 as "fix the request", 409 as
"not ready", 413 as "send less", 500 as "no advice, report it", and a timeout as "no advice".

---

## GET /health

Liveness and status. Cheap: it reads counters only (1 to 2 ms on the toy workspace, including
the HTTP round trip, when nothing else holds the lock).

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
| `models.runners` | bool | runner detector installed (`runner-research`); adds `runner_p_{k}x` |
| `models.risk` | bool | launch-risk model installed (`bootstrap`); fills `risk` in `/assess` |
| `learner.settled_trades` | int | Nardis trades learned so far (HTTP settles plus archive) |
| `learner.pending_trades` | int | advised trades not settled yet (at most 20 000, see [/advise_trade](#post-advise_trade)) |
| `learner.deployed_levels` | list[float] | outcome levels with a deployed classifier (subset of `1, 2, 5, 10, 100, 1000`); empty means base rates |
| `learner.value_model` | bool | expected-multiple model deployed |
| `stream` | object | counters of the background thread, next table; `{}` only when no thread was started (a bare `AddonService` in Python) |
| `ingest` | object | `POST /ingest` counters since start: `transactions` (received minus duplicates), `duplicates`, `undecodable` |
| `archive` | object | archive thread `scans`, `errors`, `last` (`added`, `already_known`, `total`, `features`, `unreadable_files`), `last_error` after a failure; `{}` without `--archive` |
| `alerts` | object | alert thread `scans`, `sent`, `errors`, `url`; `{}` without `--alert-url` |

`stream` counters. All counts are since the process started.

| key | present | meaning |
|---|---|---|
| `polls` | always | RPC polls fully ingested (0 with `--no-stream`) |
| `events` | always | events from the RPC stream the market accepted |
| `rejected` | always | events from the RPC stream the market refused |
| `errors` | always | failed polls plus failed upkeep rounds (each is retried at the next round) |
| `assessments` | always | tokens assessed by the assessment rounds |
| `resolved` | always | assessments labelled by `resolve()` |
| `maintenance` | always | maintenance runs |
| `evicted` | always | idle tokens forgotten |
| `saves` | always | checkpoints written by the thread (periodic and maintenance) |
| `gaps` | stream on | polls whose backlog exceeded 50 000 signatures, so older activity was skipped; counted once per skipped backlog |
| `decode_errors` | stream on | transactions the decoder could not read; each is skipped, the rest of the poll is kept |
| `fetch_errors` | stream on | transactions skipped because `getTransaction` kept failing after 3 failed polls in a row |

```bash
curl -s localhost:8787/health
```

Recorded with `--no-stream` behaviour, after pushing transactions and with an alert receiver
running:

```json
{"ok": true, "market_time": 1750005776.0000038, "tokens": 11,
 "models": {"moonshot": true, "tape": true, "stopping": true, "edge": false, "runners": false, "risk": true},
 "learner": {"settled_trades": 2, "pending_trades": 3, "deployed_levels": [], "value_model": false},
 "stream": {"polls": 0, "events": 0, "rejected": 0, "errors": 0, "assessments": 2, "resolved": 4,
   "maintenance": 0, "evicted": 0, "saves": 0},
 "ingest": {"transactions": 1966, "duplicates": 8, "undecodable": 6},
 "archive": {},
 "alerts": {"scans": 55, "sent": 1, "errors": 0, "url": "http://127.0.0.1:43141/moonshot"}}
```

The `stream` block of a service polling a streamer (a fake one in the recording) adds the three
reader counters:

```json
{"polls": 8, "events": 0, "rejected": 0, "errors": 0, "assessments": 0, "resolved": 0,
 "maintenance": 0, "evicted": 0, "saves": 0, "gaps": 1, "decode_errors": 2, "fetch_errors": 0}
```

A freshly bootstrapped workspace (neural ensemble and risk model only) answers
`"models": {"moonshot": false, "tape": false, "stopping": false, "edge": false, "runners": false,
"risk": true}`.

Notes:

* `stream.rejected` counts events the market refused: a second launch of a known mint, an
  event older than the token's last event, or an event of an unknown token.
* `stream.assessments` stops growing while the market clock stands still (no events pushed, RPC
  down): a round on an unchanged clock is skipped.

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
{"time": 1750005776.0000038,
 "tokens": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000019073486, "venue": "pump_fun"}]}
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

Trust does not lower `p_ge_*`, `edge_*`, `tail_ev` or `chase_target`; it scales only
`chase_score` and `lottery_kelly` ([guard.py](../src/nardis_neural/solana/moonshot/guard.py)
module docstring). A veto zeroes `chase_score`, `lottery_kelly`, `chase_target` and `chase_edge`.
Check `trust` separately.

```bash
curl -s 'localhost:8787/ranking?limit=3'
```

```json
{"time": 1750005776.0000038,
 "candidates": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000019073486,
   "chase_score": 2.136112535983779, "expected_multiple": 2.7213369979528363,
   "chase_target": 1000.0, "chase_edge": 3.0019421633196792, "tail_ev": 4.824028049044891,
   "lottery_kelly": 0.03924748271880144, "trust": 0.7849496543760288,
   "p_collapse_1m": 0.11517944931983948, "p_collapse_5m": 0.2044762307757133, "flags": [],
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
36 ms on the toy workspace with one active token; it grows with the number of active tokens in
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
{"time": 1750005776.0000038, "target": 10.0, "min_edge": 1.0,
 "candidates": [{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000019073486,
   "chase_score": 2.136112535983779, "expected_multiple": 2.7213369979528363, "…": "…",
   "p_ge_10x": 0.1521002368487417, "edge_10x": 4.715107342310993, "…": "…",
   "target": 10.0, "edge": 4.715107342310993}]}
```

**Errors:** 409 without a tail model; 400 for a target outside the five chase targets or a
non-numeric parameter. A negative `limit` returns no candidates, as on `/ranking` (recorded:
`?target=2&min_edge=0&limit=-1` gave `"candidates": []`).

**Cost:** as `/ranking` (29 to 37 ms on the toy workspace).

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
{"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "timestamp": 1750005776.0000038,
 "venue": "pump_fun", "model_version": "m-20260929T175330-f11330",
 "prediction": {"observation_id": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu@1750005776.000",
   "horizons": ["15s", "60s", "5m"],
   "expected_returns": {"15s": -4.402220656629652e-05, "60s": 0.008976123295724392, "5m": -0.007524444721639156},
   "upside_probabilities": {"15s": 0.4229918801307934, "60s": 0.15568001871239268, "5m": 0.21436587382605815},
   "ood_score": 0.3635950227183089, "market_embedding": [0.15531930327415466, 2.4403207302093506, "…"],
   "regime_cluster": 0, "…": "…"},
 "risk": {"rug": 2.7790960205362644e-06, "graduation": 0.1935934623082479, "dev_dump": 1.033290721575326e-06},
 "risk_uncertainty": {"rug": 0.04492832581101929, "graduation": 0.04258664793634356, "dev_dump": 0.016205861309102874},
 "round_trip_cost": 0.019899999722838402,
 "expected_net_return": {"15s": -0.019055252402550838, "60s": -0.008143539245242531, "5m": -0.016923422478445953},
 "prob_net_positive": {"15s": 0.3197118387322463, "60s": 0.44208749976255535, "5m": 0.42546632505203075},
 "flags": [],
 "edge": {},
 "moonshot": {"p_ge_2x": 0.31711482972125027, "p_ge_10x": 0.1521002368487417, "p_ge_1000x": 0.0009012134984448153,
   "median_multiple": 1.3256734563689214, "expected_multiple": 2.865287458923346,
   "lottery_kelly": 0.03924748271880144, "in_entry_window": 1.0, "trust": 0.7849496543760288,
   "vetoed": 0.0, "chase_score": 2.136112535983779, "guard.wash": 1.0,
   "guard.hidden_cluster": 0.8003358662128448, "chase_target": 1000.0, "chase_edge": 3.0019421633196792,
   "chase_rank": 1.0, "expected_multiple_blend": 2.7213369979528363, "…": "…"},
 "tape": {"p_ge_10x": 0.13207589168033124, "expected_multiple": 2.577386536982327,
   "p_collapse_1m": 0.11517944931983948, "p_collapse_5m": 0.2044762307757133,
   "p_collapse_15m": 0.2990116535734155, "p_collapse_1h": 0.38759379455974974, "…": "…"}}
```

A vetoed token looked like this in an earlier toy run (a different pushed token):
`"flags": ["creator's funding cluster holds 18.8%", "volume dominated by high-frequency bots
(possible wash trading)", "moonshot veto: wash trading: bots are 91% of volume"]`, with
`"vetoed": 1.0`, `"chase_score": 0.0`, `"lottery_kelly": 0.0`, `"chase_target": 0.0`.

**Errors:** 400 `ValueError: pass ?mint=<address>` without `mint`; 400
`KeyError: 'token … is not tracked'` for a mint not in memory (never seen, or evicted after two
idle hours in streaming mode).

**Cost:** one assessment, 29 to 74 ms on the toy workspace. The docs measure the full neural
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
| `p_10x` | float | P(best of peak and final multiple ≥ 10), capped by the lower targets and `p_win`, and at the 10x break-even probability until 10x is proven; same as `chase.p_10x` |
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
| `p_{k}x` | P(reach k), made non-increasing in k; capped at `break_even_{k}x` while k is not proven |
| `break_even_{k}x` | (1 − L) ÷ (k − L), with L the mean losing multiple (multiple ≤ 1) of Nardis's trades, clipped to [0, 0.99], once it has 20 losers; else 0.7 |
| `edge_{k}x` | `p_{k}x` ÷ `break_even_{k}x`; at most 1 while k is not proven |
| `lift_{k}x` | `p_{k}x` ÷ Nardis's base rate of reaching k (the base rate is capped the same way) |
| `proven_{k}x` | 1.0 once at least 3 of Nardis's settled trades reached k (peak or realised), else 0.0 |
| `tail_ev` | expected multiple of a position that banks each **proven** rung reached, losers at L; unproven rungs count as never reached, so it is a lower bound and equals L when nothing is proven |
| `chase_target` | highest *proven* target with edge > 1, else 0 |
| `chase_edge` | its edge ratio |
| `crazy_shot` | max of `edge_100x` and `edge_1000x` |

```bash
curl -s localhost:8787/advise_trade -d '{"trade_id": "n2", "mint": "Mint0005RUG",
  "t": 1750004167.8084893, "features": {"nardis_score": 0.4}, "with_market": false}'
```

Response after one settled trade (a 2.5x winner with a 4x peak), so still base rates:

```json
{"trade_id": "n2", "p_win": 0.75, "p_10x": 0.03225806451612903, "p_100x": 0.003021148036253777,
 "expected_multiple": 2.5, "size_multiplier": 1.0, "veto": false, "reason": "base rates only",
 "evidence": 1, "source": "prior",
 "chase": {"p_2x": 0.23076923076923078, "break_even_2x": 0.23076923076923078, "edge_2x": 1.0, "lift_2x": 1.0, "proven_2x": 0.0,
   "p_5x": 0.06976744186046513, "break_even_5x": 0.06976744186046513, "edge_5x": 1.0, "lift_5x": 1.0, "proven_5x": 0.0,
   "p_10x": 0.03225806451612903, "break_even_10x": 0.03225806451612903, "edge_10x": 1.0, "lift_10x": 1.0, "proven_10x": 0.0,
   "p_100x": 0.003021148036253777, "break_even_100x": 0.003021148036253777, "edge_100x": 1.0, "lift_100x": 1.0, "proven_100x": 0.0,
   "p_1000x": 0.00030021014710297214, "break_even_1000x": 0.00030021014710297214, "edge_1000x": 1.0, "lift_1000x": 1.0, "proven_1000x": 0.0,
   "tail_ev": 0.7, "chase_target": 0.0, "chase_edge": 0.0, "crazy_shot": 1.0}}
```

With no settled trades at all, `p_win` is 0.5 (the Jeffreys prior (0 + 0.5) / (0 + 1)), `reason`
is `no settled history`, `expected_multiple` is 1.0, and the chase is the same as above: every
`p_{k}x` sits at its break-even (`p_10x` 0.0323, `p_100x` 0.00302), every `edge_{k}x` and
`crazy_shot` is 1.0, and `tail_ev` is 0.7 (recorded). Before the hand-off fixes the same call
reported `edge_1000x` about 1665 and `tail_ev` about 500.

A target's odds leave break-even only once it is proven by 3 real hits among Nardis's settled
trades. `chase_target` stays 0 until then. While `source` is `prior`, act on `chase_target`,
`proven_*`, `source` and `evidence`; an `edge_{k}x` of exactly 1.0 means "not proven", not
"break-even measured".

Learning rules ([metalabel.py](../src/nardis_neural/solana/metalabel.py)): base rates until 50
trades have settled; then a refit every 25 new trades; a level's classifier needs at least 8
positives; a model is deployed only if it beats the base rate on the newest 20 % of trades.

**Errors:** 400 without `trade_id` or `mint`, or with a non-numeric feature value.

**Side effects:** the proposal (with its features, frozen now) is stored as pending under
`trade_id`, replacing any earlier pending proposal with the same id. It is persisted by the next
save. A `trade_id` that is already settled is scored but not stored again. At most 20 000
proposals wait (`max_pending`, saved in `meta/meta.json`); beyond that the oldest recorded one is
dropped, and settling it later returns `accepted: false`. There is no time-based expiry, so `t`
may be in any unit. With `with_market` and a tracked mint, the token is assessed (bookkeeping as
above).

**Cost:** measured in the docs on a 4-core CPU: 1.3 ms median with `"with_market": false`, about
20 ms with the market view ([INTEGRATION.md](INTEGRATION.md)). The toy run measured 1.0 to 1.1 ms
and 19 to 27 ms, including the HTTP round trip.

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
| `accepted` | bool | `true` if a pending proposal with this id was found and learned; `false` for an id never advised, already settled, or dropped from pending (beyond 20 000 waiting proposals) |
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

**Idempotency:** a trade id is learned once. A repeated settle returns `accepted: false`.
Advising an already settled id again returns advice but stores nothing, and settling it then
returns `accepted: false` without adding a history row (recorded: `n1` settled, re-advised,
settled with `multiple` 3.0, `{"accepted": false, "settled_trades": 1}`). Use a fresh `trade_id`
per trade.

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
 "p_collapse_1m": 0.11517944931983948, "p_collapse_5m": 0.2044762307757133,
 "p_collapse_15m": 0.2990116535734155}
```

Other recorded cases:

* no stopping and no Tape model: `{}` (status 200);
* `t_signal` in the future (nothing to mark yet): only the `p_collapse_*` keys;
* stopping model installed and an untracked mint: 400 `KeyError: 'unknown token NotATrackedMint'`.
  Without a stopping model the same untracked mint returns `{}`;
* `"t_signal": NaN`: 400 `ValueError: arange: cannot compute length`.

Treat an empty object as "no advice", never as "hold".

**Cost:** marks the position on the exit model's decision grid (30 s by default) from `t_signal`
to now, then runs one assessment for the collapse odds. 28 to 44 ms on the toy workspace; it
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
| `day_start_equity_sol` | float or null | no | the remembered opening equity | the market day's opening bankroll, for the 15 % daily loss stop; replaces the remembered value for the rest of that day |

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
  "stake_sol": 0.3924748271880144, "fraction": 0.03924748271880144, "reason": "sized"}]}
```

The recorded sequence on one market day, all for the same candidate:

| body | `stake_sol` | `reason` |
|---|---|---|
| `{"equity_sol": 10.0}` (first call of the day: 10 becomes the opening equity) | 0.392 | `sized` |
| `{"equity_sol": 10.0, "open_stakes": {"2niH9…sgJu": 0.3}, "peak_equity_sol": 14.0}` | 0.0 | `already open` |
| `{"equity_sol": 8.0}` (20 % below the remembered opening 10) | 0.0 | `daily loss stop` |
| `{"equity_sol": 8.0, "day_start_equity_sol": 9.0}` (11 % below 9) | 0.314 | `sized` |
| `{"equity_sol": 8.0}` (the opening is now 9) | 0.314 | `sized` |

After the sequence `capital/book.json` held `{"day": 20254, "day_start_equity": 9.0}`.

Notes:

* The book is remembered across calls and restarts in `capital/book.json` (written atomically).
  The first call of a market day records `equity_sol` as the day's opening equity; a call with
  `day_start_equity_sol` overrides it. The day is the UTC day of the **market clock**
  (`market_time // 86400`), not of the wall clock. Once equity is more than 15 % below the
  opening, every candidate gets `daily loss stop` for the rest of that day.
* The HTTP endpoint and `solana allocate` share the same file, so a CLI call changes the book the
  sidecar sees. `solana allocate --day-start-equity` sets the same value.
* `open_stakes` count toward their creator family's 8 % cap: each open mint is mapped to its
  creator family when the market knows it.
* Pass `peak_equity_sol` on every call. Without it the drawdown governor sees no drawdown.
* The CLI `solana allocate` prints the track-record scale; this endpoint does not.
* Errors: 409 without a tail model; 400 without `equity_sol`.
* Cost: as `/ranking` (27 to 46 ms on the toy workspace).

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
| `undecodable_transactions` | int | transactions without a `meta` object, or that the decoder could not read (missing keys, bad indices, bad numbers) |
| `duplicates` | int | transactions whose first signature was already ingested (in an earlier call or earlier in this batch); skipped |

```bash
curl -s localhost:8787/ingest -d @batch.json      # {"transactions": [ … ]}
```

```json
{"events": 1956, "rejected": 0, "undecodable_transactions": 0, "duplicates": 0}
```

That call carried 1956 synthetic transactions for 3 simulated tokens and took 128 ms. Recorded
edge cases, in order:

| request | status and response |
|---|---|
| the first 5 transactions sent again | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 0, "duplicates": 5}` |
| two transactions already sent, the first repeated in the same batch | 200 `{"events": 0, …, "duplicates": 3}` |
| 3 new transactions with `"meta": null` (and a `transaction` object with signatures) | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 3, "duplicates": 0}` |
| the same 3 resent complete | 200 `{"events": 3, "rejected": 0, "undecodable_transactions": 0, "duplicates": 0}`: ingested, not duplicates |
| `{"transactions": [{"slot": 1, "meta": null}]}` | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 1, "duplicates": 0}` |
| a `meta` object but no `transaction.message` | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 1, "duplicates": 0}` |
| a failed transaction (`meta.err` not null) | 200 `{"events": 0, "rejected": 0, "undecodable_transactions": 0, "duplicates": 0}` (ignored, not counted) |
| `{"transactions": [{"slot": "abc"}]}` | 400 `ValueError: invalid literal for int() with base 10: 'abc'` |
| `{"transactions": "nope"}` | 400 `ValueError: transactions must be a list` |
| `{"transactions": ["x"]}` | 400 `ValueError: every transaction must be a JSON object` (nothing in the batch is ingested) |
| `{}` | 400 `KeyError: 'transactions'` |

**De-duplication.** A transaction is identified by its first signature
(`transaction.signatures[0]`). The service remembers the signatures of the last 200 000
transactions it decoded (oldest dropped first) and skips a repeat, as the built-in stream does. A
transaction is remembered only after it decoded with a `meta` object, so one pushed first without
its `meta` (or one that failed to decode) is ingested when it is sent again complete. A
transaction with no signature is never treated as a duplicate. The memory is in-process: after a
restart a re-sent transaction is ingested again.

Side effects: events enter the market state (and the event history unless the brain is in
streaming mode, which `serve` always turns on). The decoder is created on the first call and
keeps per-token state (venue, pool vaults, virtual reserves); the state of evicted tokens is
dropped at each maintenance. It is not saved. `GET /health` → `ingest` counts transactions,
duplicates and undecodable transactions since start.

## POST /save

Checkpoint the workspace now: neural learner state, event history (or the compact market pickle
in streaming mode), pending assessments (`pending.pkl`), online moonshot buffer, forward ledger,
meta-learner (`meta/`), risk samples and `solana_state.json`. With the stream on, the RPC cursor
that the saved state covers is then written to `stream_cursor.json`. If the stream thread is in
the middle of ingesting a poll, the save waits until that poll is done.

**Body:** none needed. A body, if sent, must still be a JSON object (else 400); its content is
ignored.

```bash
curl -s -X POST localhost:8787/save
```

```json
{"saved": true}
```

Cost: 8 to 20 ms on the toy workspace in streaming mode (it writes `stream/market.pkl`, not the
event history). It holds the lock while writing. The background thread also saves every 5
minutes and after each maintenance, so `POST /save` is needed only before a backup.

---

## Push alerts

`serve --alert-url URL` starts a daemon thread named `addon-alerts` that pushes each new moonshot
candidate to `URL`:

```bash
nardis-neural solana serve --workspace ws --port 8787 \
    --alert-url http://127.0.0.1:9000/moonshot --alert-target 10 --alert-min-edge 2 --alert-every 5
```

| option | default | meaning |
|---|---|---|
| `--alert-url` | none | receiver URL; without it no alert thread runs |
| `--alert-target` | `10.0` | chase target; 2, 5, 10, 100 or 1000. Any other value exits with code 2 and `Invalid value: --alert-target must be one of 2, 5, 10, 100, 1000` |
| `--alert-min-edge` | `2.0` | send a candidate when `edge_{target}x` is at least this |
| `--alert-every` | `5.0` | seconds between scans |

The thread is `AddonService.alerts(url, target=10.0, min_edge=2.0, every=5.0, limit=20,
post=None, remember_seconds=3600.0)` ([service.py](../src/nardis_neural/solana/service.py)
line 228). A Python caller can start it the same way:

```python
from nardis_neural.solana.brain import SolanaBrain
from nardis_neural.solana.service import AddonService, make_server

service = AddonService(SolanaBrain("ws"))
service.stream(None)  # upkeep only, as serve --no-stream
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
* The POST goes through Python's `urllib`, which honours `HTTP_PROXY` and `NO_PROXY` for an
  `http://` URL. On a machine that sets `HTTP_PROXY`, keep the receiver's host in `NO_PROXY`
  (from reading the standard library; not tested with a proxy).

**Payload** (recorded; one POST per candidate):

```json
{"type": "moonshot", "target": 2.0,
 "candidate": {"mint": "2niH9kQ5LeWHbtBd8W9mVJ7G3b2LTp4YYSCcgTwZsgJu", "age_seconds": 1348.0000019073486,
   "chase_score": 2.136112535983779, "expected_multiple": 2.7213369979528363,
   "chase_target": 1000.0, "chase_edge": 3.0019421633196792, "tail_ev": 4.824028049044891,
   "lottery_kelly": 0.03924748271880144, "trust": 0.7849496543760288,
   "p_collapse_1m": 0.11517944931983948, "p_collapse_5m": 0.2044762307757133, "flags": [],
   "p_ge_2x": 0.31711482972125027, "edge_2x": 1.3741642621254178, "…": "…",
   "p_ge_1000x": 0.0009012134984448153, "edge_1000x": 3.0019421633196792,
   "target": 2.0, "edge": 1.3741642621254178}}
```

`candidate` is exactly a `/moonshots` row, including `target` and `edge`. Two recordings, each
with a local receiver on 127.0.0.1 that answered 204:

* in-process, `alerts(url, target=2.0, min_edge=0.0, every=0.2)`: three scans produced one POST
  for the one qualifying mint (`scans: 3, sent: 1, errors: 0`), and 55 scans later still one;
* the real command, `solana serve --no-stream --alert-url … --alert-target 2 --alert-min-edge 1
  --alert-every 1`: the receiver got one POST with `Content-Type: application/json` and the
  payload above; `/health` showed `"alerts": {"scans": 4, "sent": 1, "errors": 0, …}` after 4 s.

The test suite checks that a failed first delivery is retried and then sent once.

## /ingest transaction format

Each element is the `result` of a Solana `getTransaction` call with `"encoding": "jsonParsed"`.
The built-in stream requests `"maxSupportedTransactionVersion": 1` and
`"commitment": "confirmed"`. The decoder
([decoder.py](../src/nardis_neural/solana/ingest/decoder.py)) reads:

| path | use |
|---|---|
| `slot` | ordering within a batch (`int`) |
| `blockTime` | event time (1 s resolution); `null` becomes 0 |
| `meta` | must be an object; `null` or missing counts the transaction as undecodable (and its signature is not remembered) |
| `meta.err` | not null: the transaction is ignored |
| `meta.fee` | priority fee = `fee − 5000 × signatures` lamports, attached to the first swap |
| `meta.logMessages` | pump.fun and PumpSwap `Program data:` events (create, trade, complete) |
| `meta.preTokenBalances`, `meta.postTokenBalances` | AMM pool vault deltas (direction, size, reserves) |
| `meta.innerInstructions` | inner instructions (programs invoked, inner transfers) |
| `meta.loadedAddresses` | extra account keys of versioned transactions |
| `transaction.signatures` | count for the fee split; the first one is the de-duplication key |
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
| `addon-stream` | `serve`, always (with `--no-stream` it has no streamer) | a round every `--poll-interval` (2 s) | per 50-event chunk of a poll, then for each upkeep step | polls the RPC and ingests events (stream on only); then the upkeep below |
| `addon-archive` | `serve --archive` | every `--archive-every` (600 s) | only while learning new trades, refitting and saving `meta/`; reading and parsing the archive is outside the lock | learns the archive trades it has not seen (by `trade_id`), refits once at the end when at least 50 trades are known, saves `meta/` when something was added |
| `addon-alerts` | `serve --alert-url` (or `service.alerts`) | every `--alert-every` (5 s) | while scanning | pushes new moonshot candidates |

The upkeep, on its own wall-clock schedule inside `addon-stream`:

| step | every | what it does |
|---|---|---|
| assessment round | `--assess-every` (10 s; `0` = off) | `assess_active()` on every token active in the last 120 s; skipped while the market clock has not moved since the previous round. Feeds continual learning, the online tail buffer and the forward ledger |
| resolve | 10 s | labels matured assessments, settles forward tickets |
| evict and maintenance | 600 s | forgets tokens idle for 2 hours (and their decoder state), then `maintenance()`, which ends with a save; then commits the stream cursor |
| checkpoint | 300 s | `save()`, then commits the stream cursor |

All background threads are daemon threads. They stop when `service.stop()` sets the stop event,
and none of them dies on an exception: a failed poll or upkeep step is counted in
`stream.errors` and retried at the next round.

### The lock

One `threading.RLock` serialises every request and the background work that touches the brain.
A request waits while the upkeep runs an assessment round (one batched pass over the active
tokens) or `maintenance()` (neural adaptation or
retraining, risk refits and the online tail-model refit every 6 hours of chain time), and while
the archive thread learns a large batch of new trades. Maintenance still holds the lock for its
whole run, because it trains and swaps the models that requests read. How long those pauses last
on a real workspace was not measured; the tiny config's adaptation report recorded 9.8 s. Set
Nardis's client timeouts accordingly and treat a timeout as "no advice".

`maintenance()` no longer stops on an adaptation that cannot be built (`ValueError`, for example
too few experiences outside the embargo). It returns the message under `adapt_error` or
`full_retrain_error`, carries on with promotion, the refits and the save, and retries the
adaptation at the next run.

### Streaming mode and memory

`serve` always switches the brain to bounded memory, with or without the RPC stream: no event
history is kept, tokens idle for two hours are evicted at each maintenance, and `save()` writes
`stream/market.pkl` instead of `events/`. Restarting on that workspace resumes from the checkpoint
(and, with the stream on, from the RPC cursor in `stream_cursor.json`). A streamed workspace has
no event history for research commands: pass `--events <event directory>` to them.

### `--no-stream` with `/ingest`

With `--no-stream` the background thread runs the same upkeep as the stream: assessment rounds,
resolve, eviction, maintenance and a checkpoint every 5 minutes. Pushed events do not accumulate
in memory. Push each transaction once and in slot order; repeats are skipped by signature (see
[POST /ingest](#post-ingest)). `POST /save` is not needed for safety, only before a backup.

### Checkpoints and the RPC cursor

With the stream on, the cursor of the RPC reader (`stream_cursor.json`) advances in memory only
after a poll has been fetched and decoded in full, and it is written to disk only right after a
brain checkpoint, with the position that checkpoint covers. A checkpoint waits for a poll in
progress to finish. After a crash or a `kill -9`, the restarted service therefore replays the
chain from the last checkpoint instead of skipping what came after it.

A failed poll is retried in full at the next round. After 3 failed polls in a row
(`ChainStreamer.max_failed_polls`), a transaction whose `getTransaction` still fails after the
RPC client's own retries (6 for `serve`) is skipped and counted in `stream.fetch_errors`, so one
bad signature cannot stall the stream. A failure while listing signatures always retries the
whole poll. A transaction the decoder cannot read is skipped alone and counted in
`stream.decode_errors`.

### Shutdown

`serve` stops on Ctrl-C (SIGINT) or SIGTERM (`kill`, `systemctl stop`, `docker stop`). Both stop
the background threads, save the workspace and, with the stream on, commit the cursor. On the toy
workspace the real command exited with code 0 1.1 s after SIGTERM. What a `kill -9` or a crash
loses is at most the work since the last checkpoint (5 minutes, or the last maintenance), and with
the stream on that part of the chain is fetched again on restart.

### Latency

Measured numbers from the docs: `advise_trade` 1.3 ms median without the market view and about
20 ms with it on a 4-core CPU ([INTEGRATION.md](INTEGRATION.md)); the neural ensemble alone about
49 ms p50 per observation, 39 ms with `mc_dropout_samples: 0`
([ARCHITECTURE.md](ARCHITECTURE.md) §9). Every figure in the endpoint sections above comes from
the toy workspace (tiny network, 1 member) and includes the HTTP round trip on 127.0.0.1. Use
them only for relative cost.

### Security

* No authentication, no TLS, no CORS headers. Request bodies are limited to 16 MiB (413 above).
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
5. For entries, poll `GET /moonshots` or `GET /ranking`, or start `serve` with `--alert-url` and
   receive the candidates. The background thread assesses and feeds the forward ledger on its
   own.
6. With `--no-stream`, push transactions once each, in slot order. Check `ingest.undecodable` and
   `ingest.duplicates` in `/health`.
7. Treat 400 as a client bug, 409 as "model not installed", 413 as "send less", 500 or a timeout
   as "no advice". Never block trading on the sidecar being up.
8. Stop the sidecar with SIGTERM or Ctrl-C; both save.
