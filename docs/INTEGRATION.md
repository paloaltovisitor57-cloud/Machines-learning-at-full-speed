# Integrating with the trading system (Nardis)

The trading system only needs four calls and three schemas. It never needs to know about
experts, ensembles, normalisation or checkpoints.

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
    observation_id="SoLToken123:1718000000.25",   # any unique string
    timestamp=1718000000.25,                     # prediction time, unix seconds
    current_features=np.asarray(current_vec, np.float32),          # (current_dim,)
    sequences={
        "fast":   SequenceInput(values=bars_1s),    # (T, 6) oldest → newest
        "medium": SequenceInput(values=bars_5s),    # (T, 6)
        "slow":   SequenceInput(values=bars_30s,    # (T, 7), optional
                                time_deltas=ages_30s, mask=observed_30s),
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

engine = NeuralEngine.load("workspaces/prod")   # registry root → loads current champion
pred = engine.predict(obs)                      # or engine.predict_batch([...])

pred.expected_returns["2m"], pred.return_std["2m"]
pred.upside_probabilities["30s"], pred.downside_probabilities["5m"]   # calibrated
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

pred = trainer.predict(obs)                     # champion output; challenger shadows silently

# ... once the longest horizon has elapsed (partial horizons are fine):
trainer.add_experience(obs, NeuralOutcome(
    observation_id=obs.observation_id,
    returns={"30s": 0.012, "2m": -0.004, "5m": 0.031},          # log returns
    max_upside={"30s": 0.02, "2m": 0.02, "5m": 0.05},
    max_drawdown={"30s": 0.004, "2m": 0.015, "5m": 0.015},      # positive magnitudes
    volatility={"30s": 0.006, "2m": 0.011, "5m": 0.019},
))

# maintenance — on a timer or background worker, never on the hot path:
trainer.adapt_if_needed()          # clone champion → fine-tune → challenger
trainer.full_retrain_if_needed()   # fresh ensemble on weighted replay (+ drift trigger)
trainer.promote_if_ready()         # multi-gate decision on shadow outcomes
trainer.save()                     # persist replay buffer, shadow records, counters
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
