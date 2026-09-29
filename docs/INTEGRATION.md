# Integrating with the trading system (Nardis)

The trading system only needs four calls and three schemas. It never needs to know about
experts, ensembles, normalisation or checkpoints.

## Quickstart: the Solana addon as a sidecar (any language)

Nardis does not need to be written in Python. Run the addon next to it as a local HTTP/JSON
service and call it from anything:

```bash
export SOLANA_RPC_URL=https://…                       # read-only RPC; never a wallet key
nardis-neural solana serve --workspace ws --port 8787   # streams the chain in the background
```

| call | when Nardis makes it | returns |
|---|---|---|
| `GET /health` | at start-up, then periodically | market clock, tracked tokens, installed models, learner status |
| `GET /ranking?limit=20` | to look for entries | moonshot candidates with `chase_score`, expected multiple, P(≥10x / ≥100x), trust, crash risk, flags |
| `GET /assess?mint=…` | for one token | the full assessment (risk, tail, tape, edge, guard) |
| `POST /advise_trade` | **before every trade** | P(win / 10x / 100x) learned from Nardis's own trades, expected multiple, size multiplier 0–2, veto + reason |
| `POST /settle_trade` | **after every trade closes** | the learner updates (and refits when due) |
| `POST /hold_advice` | while a position is open | sell-now vs continuation value, P(collapse within 1 / 5 / 15 min) |
| `POST /allocate` | when sizing | recommended stakes under the capital engine's limits |
| `POST /save` | on shutdown (also automatic every 5 minutes) | checkpoints the workspace |

```bash
curl -s localhost:8787/advise_trade -d '{"trade_id": "t-123", "mint": "<mint>",
     "features": {"nardis_score": 0.82, "signal_strength": 3.1}}'
# example response: {"p_win": 0.41, "p_10x": 0.05, "p_100x": 0.004, "expected_multiple": 1.12,
#  "size_multiplier": 1.3, "veto": false, "reason": "learned", "evidence": 212, "source": "learned"}

curl -s localhost:8787/settle_trade -d '{"trade_id": "t-123", "multiple": 1.8, "peak_multiple": 3.1}'
curl -s localhost:8787/hold_advice -d '{"mint": "<mint>", "t_signal": 1790650000}'
```

* `features` is any set of named numbers Nardis has for the trade. Keep the names stable
  between trades; new names are picked up automatically.
* `multiple` is SOL returned per SOL staked, fees included. `peak_multiple` (optional) is the
  best the position reached, which sharpens the 10x / 100x learning.
* Every answer is advice. The service cannot sign or send transactions.
* Latency on a 4-core CPU: `advise_trade` 1.3 ms median (Nardis's features only,
  `"with_market": false`), about 20 ms with the addon's full market assessment of the token (the
  default). Requests are serialised with a lock, since the brain is not thread-safe.
* Bind to `127.0.0.1` (the default). The API has no authentication, so never expose it to a
  network without a firewall.

## 1. Install

```bash
pip install -e /path/to/this/repo      # Python ≥ 3.12, PyTorch ≥ 2.2
```

## 2. Build observations

Map your market state onto generic tensors. Feature *meaning* is up to you, but it must
be consistent between training data and live observations, and the dimensions must match
`features` in the config.

```python
import numpy as np
from nardis_neural import NeuralObservation, SequenceInput

obs = NeuralObservation(
    observation_id="SoLToken123:1718000000.25",  # any unique string
    timestamp=1718000000.25,  # prediction time, unix seconds
    current_features=np.asarray(current_vec, np.float32),  # (current_dim,)
    sequences={
        "fast": SequenceInput(values=bars_1s),  # (T, 6) oldest → newest
        "medium": SequenceInput(values=bars_5s),  # (T, 6)
        "slow": SequenceInput(
            values=bars_30s,  # (T, 7), optional
            time_deltas=ages_30s,
            mask=observed_30s,
        ),
    },
)
```

* Missing timescales can simply be omitted. Missing values may be NaN; all-NaN rows count
  as missing steps.
* Longer histories are truncated to the most recent `max_len` steps; shorter ones are
  padded internally.
* If you have a raw event stream instead of bars, use
  `nardis_neural.data.sequences.build_bars(event_times, event_values, now, timescale)`.
  It only uses events at or before `now`.
* An optional `GraphInput` (node features, `edge_index`, `edge_type`, `target_node`)
  carries wallet/token relations when `model.graph.enabled` is true.

## 3. Predict

```python
from nardis_neural import NeuralEngine

engine = NeuralEngine.load("workspaces/prod")  # registry root → loads current champion
pred = engine.predict(obs)  # or engine.predict_batch([...])

pred.expected_returns["2m"], pred.return_std["2m"]
pred.upside_probabilities["30s"], pred.downside_probabilities["5m"]  # calibrated
pred.maximum_upside, pred.maximum_drawdown, pred.predicted_volatility
pred.epistemic_uncertainty, pred.aleatoric_uncertainty, pred.total_uncertainty
pred.confidence, pred.ood_score, pred.member_disagreement
pred.market_embedding, pred.expert_weights, pred.regime_cluster
```

`NeuralPrediction` is a Pydantic model (`model_dump()` / `model_dump_json()`). It contains
**no trade decision**. How to act on probabilities, uncertainty and OOD scores is the
trading system's own responsibility.

## 4. Learn continuously

Use `ContinualLearner` instead of a bare engine when you want the loop that serves, shadows
and learns:

```python
from nardis_neural import ContinualLearner, NeuralOutcome

trainer = ContinualLearner("workspaces/prod")

pred = trainer.predict(obs)  # champion output; challenger shadows silently

# ... once the longest horizon has elapsed (partial horizons are fine):
trainer.add_experience(
    obs,
    NeuralOutcome(
        observation_id=obs.observation_id,
        returns={"30s": 0.012, "2m": -0.004, "5m": 0.031},  # log returns
        max_upside={"30s": 0.02, "2m": 0.02, "5m": 0.05},
        max_drawdown={"30s": 0.004, "2m": 0.015, "5m": 0.015},  # positive magnitudes
        volatility={"30s": 0.006, "2m": 0.011, "5m": 0.019},
    ),
)

# maintenance — on a timer or background worker, never on the hot path:
trainer.adapt_if_needed()  # clone champion → fine-tune → challenger
trainer.full_retrain_if_needed()  # fresh ensemble on weighted replay (+ drift trigger)
trainer.promote_if_ready()  # multi-gate decision on shadow outcomes
trainer.save()  # persist replay buffer, shadow records, counters
```

`examples/nardis_integration.py` is a complete runnable version of this, including a
`NeuralBrain` adapter class and bar construction from raw events. It is exercised by the
test suite.

## 5. Bootstrapping from historical data

Export historical observations with realised outcomes into any supported format (a
directory of `.npy`, `.npz`, `.pt` or `.parquet`; see `data/loaders.py` for keys and
columns), then:

```bash
nardis-neural init-config --out configs/nardis.yaml        # edit dims / horizons / thresholds
nardis-neural pretrain --data exports/unlabelled --config configs/nardis.yaml --out models/pre.pt   # optional
nardis-neural train --data exports/history.parquet --config configs/nardis.yaml \
    --workspace workspaces/prod --pretrained models/pre.pt
```

Parquet layout: `observation_id` (string), `timestamp` (float), `current` (list<float>),
`seq_<name>_values` (list<list<float>>, any length), optional `seq_<name>_time_deltas`,
`target_<task>_<horizon>` (float, null = unknown) for tasks `return`, `max_upside`,
`max_drawdown`, `volatility`.

## 6. Operational notes

* **Threading**: engines are read-only at inference. Maintenance (adapt, retrain) is CPU-
  or GPU-heavy, so run it in a separate process or worker that shares the workspace, and
  reload the champion (`NeuralEngine.load`) after a promotion.
* **Latency**: see `docs/ARCHITECTURE.md` §9 and `nardis-neural benchmark`.
  `ensemble.mc_dropout_samples: 0` and smaller sequence lengths reduce latency.
* **Monitoring**: `nardis-neural drift-report --reference … --current …` reports input,
  embedding, prediction and error drift. `nardis-neural status --workspace …` shows
  lifecycle state.
* **Safety**: the champion directory is immutable. Promotions and rollbacks only move a
  pointer in `registry.json`, and every transition is appended to its audit log.

## Forward test: score the ML signals before trusting them

`SolanaBrain` keeps a **forward-test ledger** (`nardis_neural.solana.forward`) that records
what the signals would have done, as they fire, with nothing chosen in hindsight:

* a paper ticket opens the first time a token is inside the moonshot entry window, is not
  vetoed by the manipulation guard, and its expected ladder payoff is at least the ticket;
* the Tape Transformer's collapse alarm (window and threshold chosen by `tape-research` on
  its tune period) arms an early exit;
* the ticket settles once its run is over, by simulating the ladder plus the alarm on the
  real pool path (latency, impact and fees included).

It runs inside every assessment round, so:

* during `solana stream` it is the **paper-trading scorecard** of the ML layer;
* during `solana stream-train` over weeks of history it is a **walk-forward backtest of the
  live system** at scale: every model refit, promotion and eviction happens exactly as it
  would live.

```bash
nardis-neural solana forward-report --workspace workspaces/sol
```

The report gives tickets, total paper PnL, mean and median multiple with a bootstrap CI,
hit rates, the number of alarm exits, predicted versus observed P(≥10x), and how the top
quintile by `chase_score` did compared with the rest. Compare it with your own algorithm's
paper results on the same days before letting the signals size real positions.


## Learning from Nardis's own trades (meta-labeling)

The rest of the addon learns from **the market**. `nardis_neural.solana.metalabel` learns from
**Nardis itself**: which of its own trades work, so that setups that keep failing get flagged
and shrunk and setups that keep working get more weight. Nardis stays in charge of every
decision; this layer only returns numbers.

```python
from nardis_neural.solana.metalabel import TradeOutcome, TradeProposal

# before each trade: send whatever named numbers Nardis has for it
advice = brain.advise_trade(
    TradeProposal("trade-123", mint, now, {"nardis_score": 0.82, "entry_reason": 3.0})
)
advice.p_win, advice.p_10x, advice.p_100x      # probabilities learned from Nardis's own history
advice.expected_multiple                       # Duan-smeared, so fat tails are not underestimated
advice.size_multiplier                         # scale Nardis's own stake: 0 … 2
advice.veto, advice.reason                     # a pattern that has been losing
advice.source                                  # "prior" (base rates) or "learned"

# after the trade closes (peak_multiple optional; it sharpens the 10x / 100x labels)
brain.settle_trade(TradeOutcome("trade-123", exit_time, multiple=1.8, peak_multiple=3.1))
brain.save()                                   # persists history and models in ws/meta/
```

`advise_trade` joins Nardis's features with the addon's own view of the token at that moment
(moonshot, tape, edge and risk outputs, criticality and cluster features, all prefixed
`addon_`), so the learner can also discover *which addon signals matter for Nardis's style*.

How it stays honest:

* **Cold start.** Until 50 trades have settled, answers are Jeffreys-prior base rates of the
  trades so far (size 1, no vetoes). No model is ever used before it has evidence.
* **Champion / challenger.** Every 25 settled trades it refits in time order: trained on the
  older 80 %, scored on the newest 20 %. A model is deployed only if it beats the base rate
  there (log loss for the win / 10x / 100x classifiers, squared error for the value model).
  Otherwise the base rate stays. Pure-noise features are therefore never deployed (tested).
* **Tails need evidence.** The 10x and 100x classifiers need at least 8 positive and 8 negative
  trades before they are even tried. Until then P(10x) and P(100x) are base rates, capped by
  P(win).
* **Veto only when both signals agree.** A veto needs P(win) below half the average *and* an
  expected multiple below 1.

On a synthetic stream where a hidden quality drives outcomes, after 800 trades it ranks
winners at AUC > 0.75 on 400 later proposals. It sizes the best sixth of proposals above 1.2x and the
worst sixth below 0.8x, and the trades it vetoes lose on average while the rest make money. The
real test is Nardis's paper trades: the more it trades, the better this layer gets.
