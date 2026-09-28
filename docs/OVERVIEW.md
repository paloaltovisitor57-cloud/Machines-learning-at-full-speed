# nardis-neural

**A continually learning, multi-expert neural "market-state brain" for a Solana trading
system.**

It takes structured market state (a current feature vector, multi-resolution temporal
sequences and optional wallet/token graphs) and returns **probabilistic forecasts**:
expected returns, upside and downside event probabilities, expected maximum upside and
maximum drawdown, volatility, calibrated confidence, epistemic and aleatoric
uncertainty, an out-of-distribution score, a latent market-state embedding and per-observation
expert weights. It keeps learning from newly labelled outcomes through a guarded
champion → challenger lifecycle.

> **Scope.** This repository is the ML brain *only*. It contains no Solana RPC, wallets,
> transaction execution, order routing or trading strategy, and it never emits BUY/SELL
> commands. The existing trading system decides what to do with the forecasts.

---

## Contents

- [At a glance](#at-a-glance)
- [Quick start](#quick-start)
- [Architecture](#architecture)
  - [Neural pathways](#neural-pathways)
  - [Multi-timescale processing](#multi-timescale-processing)
  - [Mixture of experts](#mixture-of-experts)
  - [Latent embedding & outputs](#latent-embedding--outputs)
- [Uncertainty, calibration, OOD](#uncertainty-calibration-ood)
- [Training](#training)
- [Continual learning](#continual-learning)
- [Champion / challenger lifecycle](#champion--challenger-lifecycle)
- [Regimes & drift](#regimes--drift)
- [Inference API](#inference-api)
- [CLI](#cli)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Testing & quality gates](#testing--quality-gates)
- [Performance](#performance)
- [Future Nardis integration](#future-nardis-integration)
- [Solana intelligence layer](#solana-intelligence-layer)
- [Edge engine](#edge-engine)
- [Moonshot engine](#moonshot-engine)
- [Tape Transformer](#tape-transformer)
- [Optional / not included](#optional--not-included)
- **[Part II — In depth](#part-ii--in-depth)**: architecture, continual learning, integration,
  Solana layer, edge engine, moonshot engine, Tape Transformer (the full contents of `docs/`)
- **[Part III — Generated reference](#part-iii--generated-reference)**: every CLI command and
  option, every configuration field and default, every feature, every output field, the
  public Python API and the test inventory

## At a glance

```mermaid
flowchart TB
    MD[Market data from the trading system] --> CUR[current-state features]
    MD --> FS[fast sequence · 1 s]
    MD --> MS[medium sequence · 5 s]
    MD --> SS[slow sequence · 30 s]
    MD -.-> GI[optional graph]

    subgraph BRAIN[nardis_neural]
        direction TB
        subgraph EXP[Specialised neural pathways]
            T[Transformer]
            R[GRU / LSTM]
            C[Temporal CNN]
            TAB[Residual MLP]
            G[Graph SAGE/GAT]
        end
        GATE[Dynamic MoE gating<br/>per-observation weights]
        LAT[Shared latent<br/>MarketStateEmbedding]
        HEADS[Multi-horizon heads<br/>return · upside · downside · drawdown · volatility]
        UNC[Deep ensemble + MC dropout<br/>calibration · OOD · confidence]
        EXP --> GATE --> LAT --> HEADS --> UNC
    end

    CUR --> TAB
    FS & MS & SS --> T & R & C
    GI -.-> G
    UNC --> PRED[NeuralPrediction<br/>no trade commands]
```

| Capability | Implementation |
|---|---|
| Temporal pathways | causal pre-LN Transformer, packed GRU/LSTM, dilated causal TCN, selective state-space (Mamba-style) SSM, residual MLP, optional relational GraphSAGE/GAT (pure PyTorch) |
| Hardware | one code path from a laptop CPU to a large GPU: `cpu-lite` / `cpu` / `gpu` / `gpu-frontier` profiles (≈0.17 M → 16.6 M parameters per member), auto-detected |
| Fusion | learned, per-observation softmax gate with load-balance, entropy and z-loss regularisers; expert dropout and noisy gating |
| Outputs | per horizon: return mean + variance + quantiles, max upside, max drawdown, volatility (heteroscedastic), upside/downside event probabilities |
| Uncertainty | deep ensemble (independent members), seeded MC dropout, epistemic / aleatoric / total decomposition, member disagreement |
| Calibration | temperature, Platt or isotonic, fitted on validation data only; Brier score, ECE, reliability bins |
| OOD & drift | Mahalanobis embedding distance, input z-score and disagreement OOD score; PSI / KS / Wasserstein / moment input drift, embedding, prediction and error drift |
| Continual learning | 3-pool replay (recent FIFO, historical reservoir, protected rare events), 5 sampling strategies, candidate cloning, distillation, EWC, full retraining with configurable weights |
| Lifecycle | immutable checkpoints, champion/candidate/challenger/retired/failed registry with audit log, shadow mode, 10-gate promotion, manual and optional automatic rollback |
| Representation | self-supervised pretraining (masked timestep, masked feature, contrastive), embedding export to Parquet/NumPy, KMeans / GMM / HDBSCAN regime discovery |
| Engineering | Pydantic v2 + YAML config, Typer CLI, CPU/CUDA/MPS, safe mixed precision, `mypy --strict`, `ruff`, {{TESTS}} test functions |

## Quick start

```bash
# Python ≥ 3.12
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# synthetic demo data (testing only — not a market simulator)
nardis-neural generate-synthetic --out data/history --n 4000 --config configs/small.yaml
nardis-neural generate-synthetic --out data/live    --n 1500 --seed 1 --start-time 1700100000 --config configs/small.yaml

# train a calibrated deep ensemble and register it as champion of a workspace
nardis-neural train --data data/history --config configs/small.yaml --workspace workspaces/demo

# predict
nardis-neural predict --model workspaces/demo --data data/live --limit 5
```

In Python:

```python
from nardis_neural import NeuralEngine

engine = NeuralEngine.load("workspaces/demo")
prediction = engine.predict(observation)  # observation: NeuralObservation
```

## Architecture

The full design rationale is in [docs/ARCHITECTURE.md](ARCHITECTURE.md).

### Neural pathways

| Expert | Highlights |
|---|---|
| **Transformer** (`models/transformer.py`) | pre-LayerNorm blocks, causal + key-padding attention mask via `scaled_dot_product_attention`, recency positional embedding + continuous time encoding, residuals, dropout, configurable depth/heads/width. Tests cover causality and padding invariance. |
| **Recurrent** (`models/recurrent.py`) | GRU by default (LSTM configurable). Observed steps are compacted and run as a packed sequence, so padding and interior gaps never enter the state. |
| **TCN** (`models/tcn.py`) | dilated causal convolutions, residual blocks, per-timestep LayerNorm (no future-leaking norms), unobserved steps held at zero, learned mixture over receptive-field scales. Tests cover causality and receptive field. |
| **SSM** (`models/ssm.py`) | selective state-space layers (Mamba-style): causal depthwise conv, input-dependent step size Δ and B/C matrices, diagonal stable A, gated output. Unobserved steps get Δ = 0 so the state passes through unchanged. Linear in sequence length, pure PyTorch (CPU / CUDA / MPS). Tests cover causality, gaps and padding invariance. |
| **Tabular** (`models/tabular.py`) | residual MLP with LayerNorm, GELU/SiLU and dropout for the immediate state vector. |
| **Graph** (`models/graph.py`, optional) | relational GraphSAGE or GAT over wallet→token / wallet→wallet / token→token edges, in pure PyTorch (no PyG dependency). Disabled by default; everything works without it. |

### Multi-timescale processing

Each sequence expert projects every timescale (feature dims may differ) together with
`log1p(age)` and the mask, adds a continuous time encoding and a **learned timescale
embedding**, and runs a **temporal core shared across timescales** (independent cores are
one config flag away). Masked last-step + mean pooling summarise each resolution, and an
attention-based **TimescaleFusion** combines them while masking missing timescales. At
inference all timescales are encoded in one stacked call. This is exact, because every
core is invariant to extra left padding.

### Mixture of experts

The gating network sees every expert's latent plus availability flags and outputs a
softmax over experts **for each observation**. Weights sum to 1, are returned with each
prediction, are exactly 0 for unavailable or runtime-disabled experts, start uniform
(zero-initialised gate) and are regularised against collapse by load-balance, entropy and
z-loss terms plus noisy gating and expert dropout. Experts are fused as `Σ w_e · A_e(h_e)`,
not by concatenation.

### Latent embedding & outputs

The mixture feeds a residual latent encoder that produces the **MarketStateEmbedding**,
used for regimes, nearest-neighbour search, OOD detection, drift monitoring and downstream
models. Separate per-horizon heads (default 30 s / 2 min / 5 min) predict:

- return: mean, heteroscedastic variance and non-crossing quantiles;
- max upside, max drawdown, volatility: softplus means with variances;
- `P(return > threshold)` and `P(drawdown > threshold)`: calibrated at inference.

## Uncertainty, calibration, OOD

- **Deep ensemble**: `ensemble.size` (default 3) independently initialised networks, each
  trained with its own optimiser, seed and data order (bootstrap optional).
- **MC dropout**: seeded extra passes through the decode stack, so inference is
  deterministic and cheap.
- **Decomposition**: aleatoric `E[σ²]`, epistemic `Var[μ]`, total; for events, expected
  entropy vs mutual information (BALD); plus member disagreement.
- **Calibration**: temperature, Platt or isotonic per (event, horizon), fitted on
  validation data only, with Brier / ECE / reliability reports.
- **OOD**: `ood_score` combines embedding Mahalanobis distance, input z-scores and
  disagreement, each normalised by its training 99th percentile.
- **Confidence** falls with epistemic uncertainty and with OOD.

## Training

```mermaid
flowchart LR
    D[(dataset<br/>npy · npz · pt · parquet)] --> S[chronological split<br/>+ embargo ≥ max horizon]
    S --> N[normaliser<br/>train rows only]
    N --> M1[member 1] & M2[member 2] & M3[member 3]
    M1 & M2 & M3 --> CAL[calibration<br/>validation only]
    CAL --> REF[reference stats · OOD · regimes]
    REF --> CK[(immutable checkpoint)]
```

`Trainer` supports CPU/CUDA/MPS, mixed precision (CUDA bf16/fp16, MPS opt-in), gradient
clipping and accumulation, cosine / one-cycle / plateau schedules, early stopping,
per-epoch checkpoints with exact resume, deterministic seeds, NaN/Inf detection, JSONL
metrics, configurable batch size and workers, class weighting, focal loss, balanced
sampling and rare-event oversampling. Multi-task losses (Gaussian NLL / Huber / MSE,
pinball, BCE / weighted BCE / focal, static or learned uncertainty task weights) are
logged per component. Datasets are memory-mapped, so they never have to fit in RAM.

**Self-supervised pretraining** (`nardis-neural pretrain`) trains the temporal encoders on
unlabelled sequences with masked-timestep reconstruction, masked-feature reconstruction and
NT-Xent contrastive learning. Load the result with `train --pretrained`.

## Continual learning

Details are in [docs/CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md).

```mermaid
flowchart LR
    O[new observation] --> CP[champion predicts]
    CP --> OUT[outcome known]
    OUT --> EXP[labelled Experience]
    EXP --> RB[(replay buffer<br/>recent · historical · rare)]
    RB -->|enough new data| CAND[candidate = clone champion]
    CAND --> FT[fine-tune: replay mixture<br/>+ distillation + EWC]
    FT --> VAL{offline validation}
    VAL -->|ok| SH[challenger in shadow mode]
    VAL -->|worse| FAIL[failed]
    SH --> PR{10 promotion gates}
    PR -->|pass| CH[new champion]
    PR -->|fail| KEEP[champion stays]
```

- **Replay**: a recent FIFO, a historical reservoir (old data is never simply dropped) and
  a protected rare-event store. Sampling can be uniform, recency-weighted, prioritized
  (with importance weights), rare-event or regime-balanced, and pools can be mixed.
- **Frequent adaptation**: clone the champion, fine-tune conservatively (low LR, few
  epochs, clipping) on a configurable mixture of recent, historical, rare and difficult
  samples, validate on the newest window with an embargo, and keep the input normaliser.
- **Anti-forgetting**: experience replay, **teacher/student distillation** from a frozen
  champion copy (tempered binary KL, regression and embedding terms, detached teacher) and
  optional **EWC** with a normalised diagonal Fisher.
- **Full retraining**: a fresh ensemble and normaliser trained on replay (plus optional
  external data), with configurable recent / historical / rare / difficult weights.
  Triggered by volume or input drift.

## Champion / challenger lifecycle

- **Registry** (`lifecycle/champion.py`): states `champion`, `candidate`, `challenger`,
  `retired`, `failed`. Model directories are immutable, `registry.json` is written
  atomically, and every transition goes to an audit log.
- **Shadow mode** (`lifecycle/shadow.py`): the challenger gets identical observations and
  its predictions are recorded but never returned. It is compared on regression, log loss,
  Brier, calibration, ranking, uncertainty quality and tails, per horizon, regime and time
  window.
- **Promotion** (`lifecycle/promotion.py`): minimum observations, return RMSE, log loss,
  tail MAE (required), plus Brier, ECE, rank correlation, NLL, time consistency and regime
  consistency. It produces JSON and Markdown reports.
- **Rollback** (`lifecycle/rollback.py`): manual, or automatic when live error degrades
  (**off by default**). It restores weights, normaliser, calibration, config and ensemble
  together.

## Regimes & drift

- `extract-embeddings` exports embeddings, gate weights, predictions and targets to
  Parquet or NPZ.
- `cluster-regimes` runs KMeans, GMM or HDBSCAN (optional automatic *k* via silhouette or
  BIC) and reports, per cluster: size, expert-weight profile, predicted and realised
  returns, errors, time span. Regimes are **discovered, never hand-defined**. Each engine
  also carries a fitted clusterer, so every prediction includes `regime_cluster`.
- `drift-report` covers four levels: raw input (PSI, KS, Wasserstein, moments), latent
  embedding (Mahalanobis distance ratio, KS, whitened mean shift), predictions, and model
  error.

## Inference API

```python
from nardis_neural import NeuralEngine, NeuralObservation, SequenceInput

engine = NeuralEngine.load("workspaces/prod")  # workspace → current champion
p = engine.predict(
    NeuralObservation(
        observation_id="tok:1718000000",
        timestamp=1718000000.0,
        current_features=current_vec,
        sequences={"fast": SequenceInput(values=bars_1s), "medium": SequenceInput(values=bars_5s)},
    )
)
p.expected_returns  # {"30s": …, "2m": …, "5m": …}
p.upside_probabilities, p.downside_probabilities, p.maximum_upside, p.maximum_drawdown
p.predicted_volatility, p.epistemic_uncertainty, p.aleatoric_uncertainty, p.total_uncertainty
p.confidence, p.ood_score, p.market_embedding, p.expert_weights, p.regime_cluster
```

The continual-learning API:

```python
from nardis_neural import ContinualLearner

trainer = ContinualLearner("workspaces/prod")
trainer.predict(observation)  # champion; challenger shadows
trainer.add_experience(observation, outcome)
trainer.adapt_if_needed()
trainer.full_retrain_if_needed()
trainer.promote_if_ready()
```

## CLI

| Command | Purpose |
|---|---|
| `init-config` | write the default YAML config |
| `generate-synthetic` | synthetic multi-regime dataset (npy / npz / pt / parquet, optional graphs, optional distribution shift) |
| `train` | train + calibrate an ensemble; `--workspace` registers it as champion (or as a challenger if a champion exists); `--pretrained`, `--resume` |
| `pretrain` | self-supervised encoder pretraining |
| `evaluate` | full metric report on a labelled dataset |
| `predict` | predictions from JSON observations or a dataset (JSONL out) |
| `extract-embeddings` | embeddings + gates + predictions → Parquet / NPZ |
| `cluster-regimes` | unsupervised regime discovery + per-cluster statistics |
| `ingest` | add labelled outcomes to the replay buffer (and to the shadow comparison) |
| `adapt` | continual adaptation (`--force` ignores the threshold) |
| `full-retrain` | weighted full retraining (`--data` for external data) |
| `shadow-evaluate` | challenger vs champion report (`--data` replays a dataset) |
| `promote` | evaluate promotion gates; `--version` forces a promotion |
| `rollback` | restore the previous (or `--to`) champion |
| `drift-report` | input / embedding / prediction / error drift |
| `inspect-model` | metadata, architecture and reference statistics |
| `status` | workspace lifecycle state |
| `benchmark` | latency, throughput and memory |
| `hardware` | detected device and recommended compute profile (`train --profile auto` applies it) |

## Configuration

Everything important is in YAML and validated by Pydantic (`src/nardis_neural/config.py`):
dimensions, timescales and sequence lengths, horizons and event thresholds, transformer
layers and heads, hidden sizes, TCN channels, recurrent depth and cell, dropout, experts
enabled, gating regularisers, ensemble size and MC samples, losses and task weights,
optimiser and LR, replay capacities and strategies, retraining thresholds and weights,
distillation and EWC weights, drift thresholds, promotion gates and rollback. Presets:

- `configs/default.yaml`: production-sized defaults (GPU or strong CPU);
- `configs/small.yaml`: compact model for CPU experiments.

## Repository layout

```
├── configs/                 default.yaml · small.yaml
├── README.md                generated: python -m nardis_neural.docgen (a test keeps it in sync)
├── docs/                    OVERVIEW.md · ARCHITECTURE.md · CONTINUAL_LEARNING.md · INTEGRATION.md ·
│                            SOLANA.md · EDGE.md · MOONSHOT.md · TAPE.md (the README's hand-written sources)
├── examples/                nardis_integration.py (runnable, tested)
├── src/nardis_neural/
│   ├── config.py            Pydantic config tree
│   ├── schemas.py           NeuralObservation / NeuralOutcome / NeuralPrediction
│   ├── synthetic.py         multi-regime synthetic market generator (tests/demos)
│   ├── benchmark.py         latency / throughput / memory
│   ├── hardware.py          device detection + cpu-lite / cpu / gpu / gpu-frontier profiles
│   ├── docgen.py            builds README.md from docs/ + the code (CLI, config, features, API)
│   ├── cli.py               Typer CLI
│   ├── data/                loaders · datasets · sequences · splits · normalization · replay
│   ├── models/              transformer · recurrent · tcn · ssm · tabular · graph · experts ·
│   │                        gating · fusion · heads · ensemble · main · common
│   ├── training/            trainer · losses · metrics · scheduling · pipeline ·
│   │                        continual · distillation · ewc · pretraining
│   ├── inference/           engine · uncertainty · calibration · ood
│   ├── regimes/             embeddings · clustering
│   ├── lifecycle/           checkpoints · champion · candidate · shadow · promotion · rollback
│   ├── monitoring/          drift
│   └── solana/              amm · events · market · wallets · features · labels · dataset ·
│                            risk · simulator · brain · config · cli · streaming ·
│                            ingest/ (decoder · rpc · stream · history · encode · pumpfun · base58)
│                            edge/ (barriers · model · trees · backtest · research)
│                            moonshot/ (labels · tail · guard · online · research)
│                            tape/ (features · dataset · model · research)
└── tests/                   {{TESTS}} test functions incl. synthetic end-to-end pipeline
```

## Testing & quality gates

```bash
ruff check .        # lint
ruff format --check .
mypy                # strict mode: src, tests and examples
pytest              # CUDA / MPS tests auto-skip when unavailable
```

The suite covers:

- Transformer shapes, causality and masking; recurrent and TCN causality; padding and gap
  invariance;
- gating sums, disabled experts and dynamic routing; gradients reaching every expert;
- probabilistic and multi-task losses; trainer resume and determinism;
- ensemble disagreement, calibration and OOD;
- checkpoint and normaliser persistence; replay strategies and persistence;
- candidate/champion isolation, adaptation, distillation, EWC and pretraining;
- regimes and drift; shadow mode, promotion gates and rollback;
- leakage-free chronological and walk-forward splits (property-based), deterministic
  inference, devices, every CLI command, the integration example;
- a **synthetic end-to-end test**: data → train ensemble → predict → uncertainty →
  experiences → replay → adaptation → shadow → promotion → checkpoint → reload → identical
  inference → rollback.

## Performance

Default configuration (3 members × 729 k parameters, 5 experts incl. SSM, 2 MC samples),
CPU only, 4-core Xeon @ 2.1 GHz:

| | p50 latency | throughput |
|---|---|---|
| single observation (full API) | ~49 ms | – |
| batch 32 | ~190 ms | ~170 obs/s |
| batch 256 | ~1.2 s | ~215 obs/s |
| single observation, `mc_dropout_samples: 0` | ~39 ms | – |
| single observation, `--profile cpu-lite` (2 × 168 k) | ~21 ms | ~640 obs/s at batch 256 |

Peak RSS is about 1 GB (0.8 GB for `cpu-lite`). GPUs are much faster. Run `nardis-neural benchmark` on your
hardware.

## Future Nardis integration

See [docs/INTEGRATION.md](INTEGRATION.md) and the runnable
[`examples/nardis_integration.py`](../examples/nardis_integration.py). In short: build a
`NeuralObservation` from generic tensors (`build_bars` turns raw event streams into
leakage-free bars), call `predict`, report `NeuralOutcome`s as horizons elapse, and run
`adapt_if_needed` / `full_retrain_if_needed` / `promote_if_ready` on a background timer.
The trading system never touches model internals.

## Solana intelligence layer

`nardis_neural.solana` specialises the brain for Solana memecoin markets. See
[docs/SOLANA.md](SOLANA.md).

- exact **pump.fun bonding-curve and AMM maths**: bonding progress, graduation,
  round-trip cost (fees + two-way impact) for a reference size;
- a **causal market state** fed by decoded events (launches, swaps, migrations, LP changes,
  SOL transfers) that rejects time travel;
- **wallet intelligence**: union-find funding clusters with exchange-hub detection
  (sybil / bundle discovery) and Beta-posterior reputations learned online *only* from
  outcomes that have already resolved, plus rug attribution to creator clusters. A second,
  **runner-specific skill** credits wallets that buy early into tokens that later run 10x;
- **67 named on-chain features**: holder concentration, dev / sniper / bundle /
  creator-cluster exposure, fresh wallets, smart-money flow, runner-skilled buyers,
  sybil-resistant counts per funding cluster, the **creator family's track record** (prior
  launches, best peak, rug and graduation rates), market-wide heat, bots, priority fees and
  Jito tips, authorities, liquidity. Also 1 s / 5 s / 30 s trade bars with forward-filled
  prices, and a live wallet→token / funding / cluster **graph** for the graph expert;
- **hindsight labels** (returns, cost-aware net returns, rug / graduation / dev-dump events)
  joined with **causally replayed features**; a test proves snapshots equal what a
  past-only market would compute;
- a **launch-risk ensemble** (P rug, P graduation, P dev dump) on
  `[embedding ‖ on-chain features]`, with token-disjoint validation and calibration;
- **`SolanaBrain`**: `ingest → assess_many → resolve → maintenance`, returning forecasts,
  risk, expected net return after costs, P(beat costs) and human-readable red flags. It
  plugs into the continual-learning, shadow, promotion and rollback machinery;
- **real-chain ingestion**: a decoder for `getTransaction` JSON (pump.fun Anchor events,
  venue-agnostic AMM vault-delta swaps and LP changes, SOL funding transfers, Jito tips,
  priority fees), a **read-only** RPC client, and a cursor-based live streamer
  (`solana backfill`, `solana stream`, `solana decode`);
- an **agent-based launch simulator** (retail, smart money, snipers, bots, loud and stealth
  rug crews, decoys, graduations, LP pulls) and the `nardis-neural solana …` CLI
  (`simulate`, `build-dataset`, `bootstrap`, `replay`, `assess`, `decode`, `backfill`, `stream`,
  `stream-train`, `init-config`);
- **training by streaming history** (`solana stream-train`): walks an archival RPC such as
  Old Faithful oldest → newest, fetching only pump.fun / PumpSwap transactions, and learns
  online with bounded memory (token eviction, compact wallet state, rolling buffers),
  gated online refits of the tail model and resumable checkpoints. Nothing is stored.

```python
from nardis_neural.solana import SolanaBrain, EventStore

brain = SolanaBrain.bootstrap("workspaces/sol", EventStore.load("history/"))
brain.ingest(event)  # decoded on-chain events, in time order
for report in brain.assess_active():  # one batched forward pass per round
    report.risk["rug"], report.expected_net_return["60s"], report.flags
brain.resolve()
brain.maintenance()
```

## Edge engine

`nardis_neural.solana.edge` hunts for **edge after costs** and checks whether it is real.
See [docs/EDGE.md](EDGE.md).

- **executable triple-barrier labels**: latency-delayed entry and exit, exact bonding-curve
  and AMM impact plus fees on both legs, take-profit / stop-loss / time exits;
- **walk-forward out-of-fold** retraining of the neural ensemble and risk model, so the
  second stage only ever learns from forecasts a live system would have seen;
- a **meta-labeling edge model**: a bootstrap ensemble predicting calibrated P(win) and
  expected net return; a lower-confidence-bound edge score; a capped fractional-Kelly hint;
- an **honest backtest**: fit / tune / test in time order, threshold chosen on tune, one
  shot on test, position constraints, bootstrap confidence intervals, compared with random,
  momentum and take-everything baselines at the same trade budget;
- `nardis-neural solana edge-research` installs the model; every `SolanaAssessment` then
  carries `edge` (p_win, expected_net, edge_score, kelly_fraction, above_threshold).

## Moonshot engine

`nardis_neural.solana.moonshot` is for the other end of the distribution: the rare launch
that runs 100x to 1000x. See [docs/MOONSHOT.md](MOONSHOT.md).

- **executable peak-multiple labels**: the best liquidation multiple a ticket bought now
  could have realised (latency on both legs, own impact, fees). Labels are
  **right-censored** while a token is still running, and dead tokens resolve causally;
- a **censored power-law tail model**: a deep ensemble of mixture-of-log-logistic
  networks gives calibrated P(peak ≥ 2x, 5x, 10x, 100x, 1000x) per token, with a learned
  tail index and epistemic spread;
- **decision helpers**: expected payoff of a take-profit ladder with a trailing moon bag,
  and a **lottery-Kelly** fraction that maximises expected log-wealth;
- **hard to fool**: per-level tail calibration on held-out tokens, inputs clamped to the
  training range, and a manipulation guard (rug risk, authorities, wash trading, bundles,
  creator clusters, OOD, ensemble disagreement) whose `trust` score and hard vetoes only
  ever lower the output. `SolanaBrain.moonshot_ranking()` returns active tokens by
  `chase_score`, best first;
- **honest research**: train on earlier tokens with labels truncated at the cutoff, score
  once on later tokens against every-launch, random and momentum tickets;
- `runner` launches in the simulator (`solana simulate --market degen`) so there are real
  100–1000x+ tails to find; `nardis-neural solana moonshot-research` installs the model and
  every `SolanaAssessment` then carries `moonshot` (`p_ge_10x`, `p_ge_1000x`,
  `expected_multiple`, `lottery_kelly`, `tail_index`, `in_entry_window`, …).

## Tape Transformer

`nardis_neural.solana.tape` reads the **raw trade tape** instead of aggregates. See
[docs/TAPE.md](TAPE.md).

- the last 96 trades, each with 18 features (side, size, timing, price move, fees, and the
  trader's skill, runner skill, cluster, creator link and freshness as known now), plus a
  **learned wallet embedding** keyed by a stable hash of the address;
- a small pre-LayerNorm **Transformer** with a summary token, fused with the 67 current
  features. Frequency gating and wallet dropout stop the wallet table from memorising noise;
- two heads: the censored power-law **tail** (P ≥ 2x … 1000x) and a discrete-time **collapse
  hazard** (P value halves within 1 min / 5 min / 15 min / 1 h), which is the exit signal;
- causal tape replay, research scored head-to-head against the tail model on identical rows,
  `nardis-neural solana tape-research`, and `SolanaAssessment.tape` on every assessment.
  On the simulator the collapse head reaches AUC 0.93–0.97; the tail model still ranks
  the far tail better.

## Optional / not included

- **Graph pathway**: implemented and tested, but disabled by default. It needs graph
  inputs from the trading system and no extra dependency.
- **Parquet datasets** carry features and targets but not graphs; use `.npy` directories,
  `.npz` or `.pt` for graph data.
- **Regime cluster ids are per model version**: each adaptation refits clusters on its own
  embedding space.
- **Offline candidate validation** uses a held-out half of the newest window, which is
  small by design. Shadow mode on genuinely new data is the decisive test.
- **Distributed or multi-node training** is intentionally out of scope (no Ray, Spark,
  Kafka and so on). Single-GPU or CPU training is supported.
- **Real-market data ingestion** is out of scope: the trading system produces the datasets.
  The synthetic generator exists to test the ML system, not to model profitability.
