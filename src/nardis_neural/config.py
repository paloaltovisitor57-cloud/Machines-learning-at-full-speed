"""Typed, validated configuration for every subsystem.

All configuration lives in one :class:`NeuralConfig` tree that is loaded from YAML and
validated with Pydantic v2.  Every checkpoint stores the full config so inference can be
reproduced exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REGRESSION_TASKS: tuple[str, ...] = ("return", "max_upside", "max_drawdown", "volatility")
"""Regression targets predicted for every horizon (as a Gaussian mean + log-variance)."""

POSITIVE_TASKS: frozenset[str] = frozenset({"max_upside", "max_drawdown", "volatility"})
"""Regression targets that are non-negative magnitudes (softplus-activated)."""

CLASSIFICATION_TASKS: tuple[str, ...] = ("upside", "downside")
"""Binary event targets: return > upside threshold, drawdown > downside threshold."""

ALL_TASKS: tuple[str, ...] = REGRESSION_TASKS + CLASSIFICATION_TASKS

EXPERT_NAMES: tuple[str, ...] = ("transformer", "recurrent", "tcn", "ssm", "tabular", "graph")


class _Base(BaseModel):
    """Shared settings: unknown keys are rejected and assignments are validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_attribute_docstrings=True)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
class TimescaleConfig(_Base):
    """One temporal stream (e.g. 1-second fast bars)."""

    name: str
    """Unique stream name; used in array keys (`seq_<name>_values`, `_mask`, `_time_deltas`)."""
    feature_dim: int = Field(gt=0)
    """Number of features per step in this stream."""
    max_len: int = Field(gt=0)
    """Steps kept per sample; shorter sequences are left-padded, longer keep the latest."""
    resolution_seconds: float = Field(gt=0)
    """Bar duration of this stream in seconds."""


class GraphInputConfig(_Base):
    """Shape of the optional per-observation wallet/token graph input."""

    node_feature_dim: int = Field(default=8, gt=0)
    """Number of features per graph node."""
    num_edge_types: int = Field(default=3, gt=0)
    """Default relation vocabulary: wallet→token, wallet→wallet, token→token."""


class FeatureConfig(_Base):
    """Input contract: current-state vector size, timescale sequences and optional graph."""

    current_dim: int = Field(default=16, gt=0)
    """Width of the current-state feature vector fed to the tabular expert."""
    timescales: list[TimescaleConfig] = Field(
        default_factory=lambda: [
            TimescaleConfig(name="fast", feature_dim=6, max_len=64, resolution_seconds=1.0),
            TimescaleConfig(name="medium", feature_dim=6, max_len=48, resolution_seconds=5.0),
            TimescaleConfig(name="slow", feature_dim=7, max_len=32, resolution_seconds=30.0),
        ]
    )
    """Temporal input streams; names must be unique."""
    graph: GraphInputConfig = Field(default_factory=GraphInputConfig)
    """Shape of the optional relational graph input."""

    @field_validator("timescales")
    @classmethod
    def _unique_names(cls, v: list[TimescaleConfig]) -> list[TimescaleConfig]:
        names = [t.name for t in v]
        if len(set(names)) != len(names):
            raise ValueError(f"timescale names must be unique, got {names}")
        return v

    def timescale(self, name: str) -> TimescaleConfig:
        """Return the timescale config called ``name`` (``KeyError`` if absent)."""
        for ts in self.timescales:
            if ts.name == name:
                return ts
        raise KeyError(name)

    @property
    def timescale_names(self) -> list[str]:
        """Names of the configured timescales, in config order."""
        return [t.name for t in self.timescales]


class HorizonConfig(_Base):
    """One forecast horizon and the return thresholds that define its upside and downside events."""

    name: str
    """Horizon label used in output keys and reports (e.g. `2m`)."""
    seconds: float = Field(gt=0)
    """Prediction horizon length in seconds."""
    upside_threshold: float = Field(default=0.02, gt=0)
    """Label `upside` = realised log-return > upside_threshold."""
    downside_threshold: float = Field(default=0.02, gt=0)
    """Label `downside` = realised max drawdown (positive magnitude) > downside_threshold."""


class TargetConfig(_Base):
    """Forecast targets: horizons and predictive quantiles."""

    horizons: list[HorizonConfig] = Field(
        default_factory=lambda: [
            HorizonConfig(name="30s", seconds=30, upside_threshold=0.01, downside_threshold=0.01),
            HorizonConfig(name="2m", seconds=120, upside_threshold=0.02, downside_threshold=0.02),
            HorizonConfig(name="5m", seconds=300, upside_threshold=0.03, downside_threshold=0.03),
        ]
    )
    """Prediction horizons; at least one, names must be unique."""
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    """Return quantile levels, sorted and inside (0, 1); empty disables the quantile head."""

    @field_validator("horizons")
    @classmethod
    def _unique(cls, v: list[HorizonConfig]) -> list[HorizonConfig]:
        if not v:
            raise ValueError("at least one horizon is required")
        names = [h.name for h in v]
        if len(set(names)) != len(names):
            raise ValueError("horizon names must be unique")
        return v

    @field_validator("quantiles")
    @classmethod
    def _q(cls, v: list[float]) -> list[float]:
        if any(not 0.0 < q < 1.0 for q in v) or sorted(v) != v:
            raise ValueError("quantiles must be sorted and inside (0, 1)")
        return v

    @property
    def horizon_names(self) -> list[str]:
        """Names of the configured horizons, in config order."""
        return [h.name for h in self.horizons]

    @property
    def max_horizon_seconds(self) -> float:
        """Length of the longest horizon in seconds."""
        return max(h.seconds for h in self.horizons)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class TransformerConfig(_Base):
    """Causal pre-LayerNorm Transformer expert."""

    layers: int = Field(default=2, ge=1)
    """Number of pre-LN Transformer blocks."""
    heads: int = Field(default=4, ge=1)
    """Attention heads; `model.d_model` must be divisible by this."""
    ff_multiplier: int = Field(default=2, ge=1)
    """Feed-forward hidden width as a multiple of `d_model`."""
    causal: bool = True
    """If true, each step attends only to itself and earlier steps."""
    max_positions: int = Field(default=512, gt=0)
    """Size of the learned recency embedding; older steps share the last position."""


class RecurrentConfig(_Base):
    """GRU / LSTM expert run on packed sequences."""

    cell: Literal["gru", "lstm"] = "gru"
    """Recurrent cell type: `gru` or `lstm`."""
    hidden_dim: int = Field(default=64, gt=0)
    """Hidden state size of the recurrent layers."""
    layers: int = Field(default=1, ge=1)
    """Number of stacked recurrent layers."""


class TCNConfig(_Base):
    """Dilated causal temporal-convolution expert."""

    channels: list[int] = Field(default_factory=lambda: [64, 64, 64])
    """Output channels of each residual block; block i uses dilation 2**i."""
    kernel_size: int = Field(default=3, ge=2)
    """Kernel size of the dilated causal convolutions."""


class SSMConfig(_Base):
    """Selective state-space (Mamba-style) expert."""

    state_dim: int = Field(default=8, ge=1)
    """State size per inner channel of the selective scan."""
    expand: int = Field(default=2, ge=1)
    """Inner width as a multiple of `d_model`."""
    conv_kernel: int = Field(default=4, ge=1)
    """Kernel size of the causal depthwise convolution."""
    layers: int = Field(default=1, ge=1)
    """Number of stacked selective SSM layers."""


class TabularConfig(_Base):
    """Residual MLP expert for the current-state vector."""

    width: int = Field(default=128, gt=0)
    """Hidden width of the residual MLP."""
    depth: int = Field(default=3, ge=1)
    """Number of residual blocks."""
    activation: Literal["gelu", "silu"] = "gelu"
    """Activation in the residual blocks: `gelu` or `silu`."""


class GraphModelConfig(_Base):
    """Optional relational graph expert (GraphSAGE or GAT)."""

    enabled: bool = False
    """Enable the graph expert; also requires `graph` in `model.experts`."""
    kind: Literal["sage", "gat"] = "sage"
    """Layer type: `sage` (relational GraphSAGE) or `gat` (graph attention)."""
    hidden_dim: int = Field(default=64, gt=0)
    """Hidden width of the graph layers."""
    layers: int = Field(default=2, ge=1)
    """Number of message-passing layers."""
    heads: int = Field(default=2, ge=1)
    """Attention heads for `gat` (hidden_dim must be divisible); unused by `sage`."""


class GatingConfig(_Base):
    """Mixture-of-experts gate and its anti-collapse regularisers."""

    hidden_dim: int = Field(default=64, gt=0)
    """Hidden width of the gating MLP."""
    temperature: float = Field(default=1.0, gt=0)
    """Gate logits are divided by this before the softmax (higher = softer routing)."""
    noise_std: float = Field(default=0.1, ge=0)
    """Gaussian noise on gate logits during training (noisy gating)."""
    expert_dropout: float = Field(default=0.05, ge=0, lt=1)
    """Probability of dropping an expert for a sample during training."""
    load_balance_weight: float = Field(default=0.01, ge=0)
    """Weight of the load-balance loss that discourages expert collapse."""
    entropy_weight: float = Field(default=0.001, ge=0)
    """Weight of the per-sample gate entropy bonus."""
    z_loss_weight: float = Field(default=0.0001, ge=0)
    """Weight of the z-loss penalising large gate logits."""


class ModelConfig(_Base):
    """Network architecture: widths, enabled experts and their settings."""

    d_model: int = Field(default=64, gt=0)
    """Width of expert representations; must be divisible by `transformer.heads`."""
    latent_dim: int = Field(default=64, gt=0)
    """Size of the MarketStateEmbedding produced before the output heads."""
    head_hidden_dim: int = Field(default=64, gt=0)
    """Hidden width of each per-horizon output head MLP."""
    dropout: float = Field(default=0.1, ge=0, lt=1)
    """Dropout rate used throughout the network (also drives MC dropout)."""
    experts: list[str] = Field(default_factory=lambda: ["transformer", "recurrent", "tcn", "ssm", "tabular"])
    """Enabled expert pathways, any of transformer/recurrent/tcn/ssm/tabular/graph."""
    share_encoder_across_timescales: bool = True
    """Use one temporal core for all timescales instead of one each."""
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    """Transformer expert settings."""
    recurrent: RecurrentConfig = Field(default_factory=RecurrentConfig)
    """Recurrent (GRU/LSTM) expert settings."""
    tcn: TCNConfig = Field(default_factory=TCNConfig)
    """Temporal convolution expert settings."""
    ssm: SSMConfig = Field(default_factory=SSMConfig)
    """Selective state-space expert settings."""
    tabular: TabularConfig = Field(default_factory=TabularConfig)
    """Residual MLP expert settings for the current-state features."""
    graph: GraphModelConfig = Field(default_factory=GraphModelConfig)
    """Graph expert settings."""
    gating: GatingConfig = Field(default_factory=GatingConfig)
    """Mixture-of-experts gating network settings."""

    @field_validator("experts")
    @classmethod
    def _experts(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(EXPERT_NAMES)
        if unknown:
            raise ValueError(f"unknown experts {sorted(unknown)}; valid: {EXPERT_NAMES}")
        if not v:
            raise ValueError("at least one expert must be enabled")
        if len(set(v)) != len(v):
            raise ValueError("duplicate expert names")
        return v

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.d_model % self.transformer.heads != 0:
            raise ValueError("d_model must be divisible by transformer.heads")
        return self

    @property
    def enabled_experts(self) -> list[str]:
        """Experts in canonical order; graph only when explicitly enabled."""
        out = [e for e in EXPERT_NAMES if e in self.experts]
        if not self.graph.enabled and "graph" in out:
            out.remove("graph")
        return out


# --------------------------------------------------------------------------------------
# Losses / training
# --------------------------------------------------------------------------------------
class LossConfig(_Base):
    """Multi-task loss: regression, quantile and event-classification terms and their weights."""

    regression_loss: Literal["gaussian", "huber", "mse"] = "gaussian"
    """Regression loss: `gaussian` NLL, `huber` or `mse`."""
    huber_delta: float = Field(default=1.0, gt=0)
    """Delta of the Huber loss (normalised target units)."""
    variance_loss_weight: float = Field(default=0.5, ge=0)
    """When the point loss is huber/mse, weight of a detached-mean Gaussian NLL that still
    trains the aleatoric log-variance heads."""
    quantile_loss_weight: float = Field(default=0.25, ge=0)
    """Weight of the pinball loss on the return quantile heads (0 = off)."""
    classification_loss: Literal["bce", "weighted_bce", "focal"] = "bce"
    """Event loss: `bce`, `weighted_bce` (uses `pos_weight`) or `focal`."""
    focal_gamma: float = Field(default=2.0, ge=0)
    """Focusing exponent of the focal loss."""
    focal_alpha: float | None = Field(default=None)
    """Positive-class weight alpha of the focal loss; None = no alpha weighting."""
    pos_weight: float | Literal["auto"] = "auto"
    """Positive class weight for weighted BCE; "auto" = neg/pos ratio on training labels."""
    task_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "return": 1.0,
            "max_upside": 0.5,
            "max_drawdown": 0.5,
            "volatility": 0.5,
            "upside": 1.0,
            "downside": 1.0,
        }
    )
    """Static per-task loss weights (missing tasks = 1); unused with learned weighting."""
    learned_uncertainty_weighting: bool = False
    """Learn task weights via homoscedastic uncertainty (Kendall et al.)."""

    @field_validator("task_weights")
    @classmethod
    def _tw(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(ALL_TASKS)
        if unknown:
            raise ValueError(f"unknown tasks in task_weights: {sorted(unknown)}")
        return v


class TrainingConfig(_Base):
    """Optimisation, early stopping, device, precision and sampling."""

    epochs: int = Field(default=20, ge=1)
    """Maximum training epochs per ensemble member."""
    batch_size: int = Field(default=128, ge=1)
    """Training batch size."""
    eval_batch_size: int = Field(default=512, ge=1)
    """Batch size for validation and inference."""
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    """Optimizer: `adamw`, `adam` or `sgd` (Nesterov, momentum 0.9)."""
    learning_rate: float = Field(default=1e-3, gt=0)
    """Base (peak) learning rate."""
    weight_decay: float = Field(default=1e-4, ge=0)
    """Weight decay passed to the optimizer."""
    scheduler: Literal["cosine", "onecycle", "plateau", "constant"] = "cosine"
    """LR schedule: `cosine` (warmup), `onecycle`, `plateau` (halve on stall) or `constant`."""
    warmup_fraction: float = Field(default=0.05, ge=0, lt=1)
    """Fraction of optimizer steps used for warmup (cosine / onecycle)."""
    grad_clip_norm: float | None = Field(default=1.0)
    """Max global gradient norm; None disables clipping."""
    grad_accumulation_steps: int = Field(default=1, ge=1)
    """Batches accumulated per optimizer step."""
    early_stopping_patience: int = Field(default=5, ge=1)
    """Epochs without validation improvement before stopping."""
    early_stopping_min_delta: float = Field(default=1e-4, ge=0)
    """Minimum validation loss decrease that counts as an improvement."""
    mixed_precision: Literal["auto", "off", "fp16", "bf16"] = "auto"
    """AMP mode: `auto` (bf16/fp16 on CUDA only), `off`, `fp16` or `bf16`."""
    device: str = "auto"
    """Torch device string; `auto` picks cuda, then mps, then cpu."""
    seed: int = 7
    """Base random seed; ensemble member i uses seed + 1000 * (i + 1)."""
    deterministic: bool = False
    """Request deterministic torch algorithms (warn-only) for reproducibility."""
    num_workers: int = Field(default=0, ge=0)
    """DataLoader worker processes (0 = load in the main process)."""
    max_nonfinite_steps: int = Field(default=10, ge=1)
    """Abort training after this many consecutive non-finite losses/gradients."""
    balanced_sampling: bool = False
    """Oversample so samples with and without any event get equal total weight."""
    rare_event_oversampling: float = Field(default=1.0, ge=1.0)
    """Sampling weight multiplier for rare events (1 = off)."""
    rare_event_return_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)
    """Quantile of max(|return|, drawdown) above which a sample is rare."""
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    """Most recent fraction of samples held out for validation (chronological)."""
    embargo_seconds: float | None = None
    """Gap between train and validation; defaults to the longest horizon (label overlap)."""
    bootstrap_members: bool = False
    """Train each ensemble member on a bootstrap resample of the training set."""
    max_train_batches_per_epoch: int | None = None
    """Cap on training batches per epoch; None = full pass."""


class NormalizationConfig(_Base):
    """Feature normalisation fitted on training data only."""

    method: Literal["robust", "standard"] = "robust"
    """robust = median / IQR scaling; standard = mean / std."""
    clip: float = Field(default=8.0, gt=0)
    """Normalised values are clipped to [-clip, clip]; NaN becomes 0."""
    max_fit_rows: int = Field(default=200_000, ge=100)
    """Maximum rows (randomly subsampled) used to fit normalisation statistics."""


class EnsembleConfig(_Base):
    """Deep-ensemble size and MC-dropout inference."""

    size: int = Field(default=3, ge=1)
    """Number of independently trained ensemble members."""
    mc_dropout_samples: int = Field(default=2, ge=0)
    """Monte-Carlo dropout passes per member at inference (0 = deterministic single pass)."""
    mc_seed: int = 1234
    """Base seed for MC dropout sampling, so inference is reproducible."""
    embedding_member: int = Field(default=0, ge=0)
    """Index of the member whose embedding is reported and used for OOD/regimes."""


class CalibrationConfig(_Base):
    """Post-hoc probability calibration of the event heads."""

    method: Literal["temperature", "platt", "isotonic", "none"] = "temperature"
    """Per-task, per-horizon event calibrator fitted on validation: temperature/platt/isotonic/none."""
    n_bins: int = Field(default=10, ge=2)
    """Bins used for ECE and reliability diagrams."""
    min_samples: int = Field(default=50, ge=1)
    """Minimum validation samples to fit a calibrator; otherwise identity."""


class OODConfig(_Base):
    """Out-of-distribution score: embedding distance, input z-scores and disagreement."""

    shrinkage: float = Field(default=0.1, ge=0, le=1)
    """Covariance shrinkage toward a scaled identity for the embedding Mahalanobis distance."""
    reference_quantile: float = Field(default=0.99, gt=0.5, lt=1)
    """Training quantile each OOD signal is divided by (score 1 = edge of training)."""
    embedding_weight: float = Field(default=0.5, ge=0)
    """Relative weight of the embedding-distance signal in `ood_score`."""
    input_weight: float = Field(default=0.25, ge=0)
    """Relative weight of the input z-score (RMS) signal in `ood_score`."""
    disagreement_weight: float = Field(default=0.25, ge=0)
    """Relative weight of the ensemble-disagreement signal in `ood_score`."""
    confidence_penalty: float = Field(default=1.0, ge=0)
    """How quickly confidence decays once ood_score exceeds 1."""


class ReplayConfig(_Base):
    """Experience replay buffer: recent, historical and rare pools."""

    recent_capacity: int = Field(default=5000, ge=1)
    """Size of the FIFO pool of the newest experiences."""
    historical_capacity: int = Field(default=20000, ge=1)
    """Size of the reservoir sample of experiences evicted from the recent pool."""
    rare_capacity: int = Field(default=5000, ge=1)
    """Size of the rare-event pool; the lowest-priority entry is evicted when full."""
    recency_half_life: float = Field(default=3600.0, gt=0)
    """Seconds for recency weight to halve."""
    priority_alpha: float = Field(default=0.6, ge=0)
    """Exponent applied to priorities in prioritised sampling (0 = uniform)."""
    priority_beta: float = Field(default=0.4, ge=0)
    """Importance-sampling correction exponent for prioritised sampling."""
    priority_epsilon: float = Field(default=1e-3, gt=0)
    """Constant added to priorities so none has zero probability."""
    rare_return_threshold: float = Field(default=0.05, gt=0)
    """|return| or drawdown above this at any horizon marks an experience as rare."""
    seed: int = 11
    """Random seed for replay sampling and reservoir replacement."""


class ContinualConfig(_Base):
    """Continual adaptation, distillation, EWC and full-retraining triggers."""

    min_new_samples: int = Field(default=500, ge=1)
    """New experiences since the last adaptation needed to trigger one."""
    adapt_samples: int = Field(default=4000, ge=1)
    """Number of replay experiences sampled to fine-tune an adaptation candidate."""
    adapt_epochs: int = Field(default=3, ge=1)
    """Fine-tuning epochs per adaptation."""
    adapt_learning_rate: float = Field(default=2e-4, gt=0)
    """Learning rate for adaptation fine-tuning."""
    adapt_validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    """Fraction of the newest experiences held out for validation."""
    recent_fraction: float = Field(default=0.5, ge=0)
    """Relative share of the adaptation sample drawn recency-weighted from recent data."""
    historical_fraction: float = Field(default=0.3, ge=0)
    """Relative share drawn uniformly from the historical pool."""
    rare_fraction: float = Field(default=0.1, ge=0)
    """Relative share drawn from the rare-event pool."""
    difficult_fraction: float = Field(default=0.1, ge=0)
    """Relative share drawn by priority (high past error) from all experiences."""
    distillation_weight: float = Field(default=0.5, ge=0)
    """Weight of distillation toward the champion during adaptation (0 = off)."""
    distillation_temperature: float = Field(default=2.0, gt=0)
    """Temperature for distilling the champion's event probabilities."""
    ewc_enabled: bool = True
    """Apply Elastic Weight Consolidation during adaptation."""
    ewc_weight: float = Field(default=10.0, ge=0)
    """EWC penalty strength (lambda); the Fisher is normalised to mean 1."""
    ewc_fisher_batches: int = Field(default=20, ge=1)
    """Batches used to estimate the diagonal Fisher information for EWC."""
    offline_max_degradation: float = Field(default=0.25, ge=0)
    """Candidate is rejected before shadow if its validation loss exceeds the
    champion's by more than this relative margin."""
    full_retrain_min_new_samples: int = Field(default=5000, ge=1)
    """New experiences since the last full retrain needed to trigger one."""
    full_retrain_recent_weight: float = Field(default=3.0, ge=0)
    """Sample weight of recent experiences in a full retrain."""
    full_retrain_historical_weight: float = Field(default=1.0, ge=0)
    """Sample weight of older experiences and external data in a retrain."""
    full_retrain_rare_weight: float = Field(default=2.0, ge=0)
    """Extra sample weight multiplier for rare experiences in a full retrain."""
    full_retrain_difficult_weight: float = Field(default=1.5, ge=0)
    """Extra weight multiplier for the top-20% priority experiences."""
    full_retrain_recent_window: int = Field(default=5000, ge=1)
    """Number of newest experiences treated as recent in a full retrain."""
    full_retrain_epochs: int | None = None
    """Epochs for a full retrain; None = `training.epochs`."""
    full_retrain_on_drift: bool = True
    """Also trigger a full retrain when input drift is detected in replay data."""


PretrainTask = Literal["masked_timestep", "masked_feature", "contrastive"]


def _default_pretrain_tasks() -> list[PretrainTask]:
    return ["masked_timestep", "masked_feature", "contrastive"]


class PretrainConfig(_Base):
    """Self-supervised encoder pretraining."""

    tasks: list[PretrainTask] = Field(default_factory=_default_pretrain_tasks)
    """Self-supervised objectives: `masked_timestep`, `masked_feature`, `contrastive`."""
    mask_ratio: float = Field(default=0.25, gt=0, lt=1)
    """Fraction of observed steps (or feature entries) masked for reconstruction."""
    epochs: int = Field(default=5, ge=1)
    """Pretraining epochs."""
    batch_size: int = Field(default=128, ge=1)
    """Pretraining batch size."""
    learning_rate: float = Field(default=1e-3, gt=0)
    """AdamW learning rate for pretraining."""
    contrastive_temperature: float = Field(default=0.2, gt=0)
    """Temperature of the NT-Xent contrastive loss."""
    augmentation_noise: float = Field(default=0.1, ge=0)
    """Std of Gaussian noise added to normalised sequences for contrastive views."""
    contrastive_weight: float = Field(default=0.5, ge=0)
    """Weight of the contrastive loss relative to the reconstruction losses."""


class RegimeConfig(_Base):
    """Unsupervised regime discovery on the latent embedding."""

    method: Literal["kmeans", "gmm", "hdbscan"] = "kmeans"
    """Clustering algorithm for regimes: `kmeans`, `gmm` or `hdbscan`."""
    n_clusters: int = Field(default=6, ge=2)
    """Number of regimes for kmeans/gmm when `auto_select` is off."""
    auto_select: bool = False
    """Choose k by silhouette (kmeans) / BIC (gmm) over ``k_min..k_max``."""
    k_min: int = Field(default=2, ge=2)
    """Smallest k tried when `auto_select` is on."""
    k_max: int = Field(default=10, ge=2)
    """Largest k tried when `auto_select` is on."""
    hdbscan_min_cluster_size: int = Field(default=25, ge=2)
    """Minimum cluster size for HDBSCAN."""
    max_fit_samples: int = Field(default=50_000, ge=10)
    """Maximum embeddings (randomly subsampled) used to fit the clusterer."""
    seed: int = 0
    """Random seed for subsampling and clustering."""


class DriftConfig(_Base):
    """Input, embedding, prediction and error drift thresholds."""

    psi_bins: int = Field(default=10, ge=2)
    """Bins for the population stability index."""
    psi_threshold: float = Field(default=0.2, gt=0)
    """A feature drifts when its PSI exceeds this."""
    ks_pvalue_threshold: float = Field(default=0.01, gt=0, lt=1)
    """A feature also drifts when its KS p-value is below this and Wasserstein exceeds its threshold."""
    wasserstein_threshold: float = Field(default=0.5, gt=0)
    """In reference standard deviations."""
    embedding_distance_threshold: float = Field(default=1.5, gt=0)
    """Current mean Mahalanobis distance / reference mean distance."""
    error_ratio_threshold: float = Field(default=1.3, gt=0)
    """Error drift when current / reference mean absolute return error exceeds this."""
    feature_fraction_threshold: float = Field(default=0.25, gt=0, le=1)
    """Fraction of drifting features needed to flag input drift overall."""


class PromotionConfig(_Base):
    """Gates a challenger must pass before it replaces the champion."""

    min_observations: int = Field(default=200, ge=1)
    """Minimum resolved shadow observations before promotion."""
    max_rmse_ratio: float = Field(default=1.0, gt=0)
    """Challenger return RMSE must be at most this multiple of the champion's."""
    max_log_loss_ratio: float = Field(default=1.0, gt=0)
    """Challenger mean event log loss must be at most this multiple of the champion's."""
    max_brier_ratio: float = Field(default=1.0, gt=0)
    """Challenger mean event Brier score must be at most this multiple of the champion's."""
    max_ece_increase: float = Field(default=0.02, ge=0)
    """Allowed absolute increase of mean event ECE over the champion."""
    min_rank_corr_delta: float = Field(default=-0.02)
    """Challenger return rank correlation must be at least champion + this."""
    max_tail_mae_ratio: float = Field(default=1.05, gt=0)
    """Challenger tail return/drawdown MAE must be at most this multiple."""
    max_nll_increase: float = Field(default=0.05, ge=0)
    """Allowed increase of the return Gaussian NLL (uncertainty quality)."""
    min_window_win_fraction: float = Field(default=0.5, ge=0, le=1)
    """Fraction of time windows the challenger must win or tie."""
    time_windows: int = Field(default=4, ge=1)
    """Number of consecutive time windows shadow records are split into."""
    max_regime_degradation: float = Field(default=0.15, ge=0)
    """Maximum relative degradation allowed in any single regime."""
    min_regime_observations: int = Field(default=30, ge=1)
    """Minimum observations for a regime to be included in the regime gate."""
    required_gates: list[str] = Field(
        default_factory=lambda: ["min_observations", "return_rmse", "log_loss", "tail_mae"]
    )
    """Gate names that must all pass; a required gate that was not evaluated counts as failed."""
    min_passed_fraction: float = Field(default=0.7, ge=0, le=1)
    """Fraction of all gates (required ones included) that must pass; required gates must pass too."""


class LifecycleConfig(_Base):
    """Model registry retention and automatic rollback."""

    keep_champions: int = Field(default=5, ge=1)
    """Former champions whose weights are kept (rollback targets); older retired and failed
    versions have their weights deleted after each promotion (registry entries stay)."""
    auto_rollback: bool = False
    """Automatically roll back to the previous champion on live degradation."""
    auto_rollback_min_observations: int = Field(default=200, ge=1)
    """Resolved live predictions averaged for the rollback check."""
    auto_rollback_degradation: float = Field(default=0.3, gt=0)
    """Roll back if live return MAE exceeds the promotion-time baseline by this fraction."""


class NeuralConfig(_Base):
    """Root configuration object."""

    features: FeatureConfig = Field(default_factory=FeatureConfig)
    """Input feature shapes."""
    targets: TargetConfig = Field(default_factory=TargetConfig)
    """Prediction horizons, event thresholds and return quantiles."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    """Network architecture."""
    loss: LossConfig = Field(default_factory=LossConfig)
    """Training objective."""
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    """Supervised training loop."""
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    """Feature normalisation."""
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    """Deep ensemble and MC dropout."""
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    """Event probability calibration."""
    ood: OODConfig = Field(default_factory=OODConfig)
    """Out-of-distribution scoring and confidence."""
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    """Experience replay buffer."""
    continual: ContinualConfig = Field(default_factory=ContinualConfig)
    """Continual adaptation and full retraining."""
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    """Self-supervised encoder pretraining."""
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    """Unsupervised regime discovery on embeddings."""
    drift: DriftConfig = Field(default_factory=DriftConfig)
    """Drift detection thresholds."""
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    """Champion/challenger promotion gates."""
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    """Model registry retention and automatic rollback."""

    @model_validator(mode="after")
    def _cross(self) -> NeuralConfig:
        if self.model.graph.enabled and "graph" not in self.model.experts:
            raise ValueError("model.graph.enabled requires 'graph' in model.experts")
        if self.ensemble.embedding_member >= self.ensemble.size:
            raise ValueError("ensemble.embedding_member must be < ensemble.size")
        return self

    @property
    def horizon_names(self) -> list[str]:
        """Horizon names (shortcut for ``targets.horizon_names``)."""
        return self.targets.horizon_names

    @property
    def embargo_seconds(self) -> float:
        """Split embargo: ``training.embargo_seconds``, or the longest horizon when unset."""
        if self.training.embargo_seconds is not None:
            return self.training.embargo_seconds
        return self.targets.max_horizon_seconds

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of the full configuration."""
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """Serialise the configuration to a YAML string (field order preserved)."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> None:
        """Write the configuration as YAML to ``path``."""
        Path(path).write_text(self.to_yaml())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuralConfig:
        """Validate a plain dict into a config (unknown keys are rejected)."""
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> NeuralConfig:
        """Load and validate a YAML config file (an empty file yields the defaults)."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        return cls.model_validate(raw)


def load_config(path: str | Path | None = None) -> NeuralConfig:
    """Load a YAML config, or return defaults when ``path`` is None."""
    if path is None:
        return NeuralConfig()
    return NeuralConfig.load(path)
