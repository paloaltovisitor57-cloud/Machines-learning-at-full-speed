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
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
class TimescaleConfig(_Base):
    """One temporal stream (e.g. 1-second fast bars)."""

    name: str
    feature_dim: int = Field(gt=0)
    max_len: int = Field(gt=0)
    resolution_seconds: float = Field(gt=0)


class GraphInputConfig(_Base):
    node_feature_dim: int = Field(default=8, gt=0)
    num_edge_types: int = Field(default=3, gt=0)
    """Default relation vocabulary: wallet→token, wallet→wallet, token→token."""


class FeatureConfig(_Base):
    current_dim: int = Field(default=16, gt=0)
    timescales: list[TimescaleConfig] = Field(
        default_factory=lambda: [
            TimescaleConfig(name="fast", feature_dim=6, max_len=64, resolution_seconds=1.0),
            TimescaleConfig(name="medium", feature_dim=6, max_len=48, resolution_seconds=5.0),
            TimescaleConfig(name="slow", feature_dim=7, max_len=32, resolution_seconds=30.0),
        ]
    )
    graph: GraphInputConfig = Field(default_factory=GraphInputConfig)

    @field_validator("timescales")
    @classmethod
    def _unique_names(cls, v: list[TimescaleConfig]) -> list[TimescaleConfig]:
        names = [t.name for t in v]
        if len(set(names)) != len(names):
            raise ValueError(f"timescale names must be unique, got {names}")
        return v

    def timescale(self, name: str) -> TimescaleConfig:
        for ts in self.timescales:
            if ts.name == name:
                return ts
        raise KeyError(name)

    @property
    def timescale_names(self) -> list[str]:
        return [t.name for t in self.timescales]


class HorizonConfig(_Base):
    name: str
    seconds: float = Field(gt=0)
    upside_threshold: float = Field(default=0.02, gt=0)
    """Label `upside` = realised log-return > upside_threshold."""
    downside_threshold: float = Field(default=0.02, gt=0)
    """Label `downside` = realised max drawdown (positive magnitude) > downside_threshold."""


class TargetConfig(_Base):
    horizons: list[HorizonConfig] = Field(
        default_factory=lambda: [
            HorizonConfig(name="30s", seconds=30, upside_threshold=0.01, downside_threshold=0.01),
            HorizonConfig(name="2m", seconds=120, upside_threshold=0.02, downside_threshold=0.02),
            HorizonConfig(name="5m", seconds=300, upside_threshold=0.03, downside_threshold=0.03),
        ]
    )
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

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
        return [h.name for h in self.horizons]

    @property
    def max_horizon_seconds(self) -> float:
        return max(h.seconds for h in self.horizons)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class TransformerConfig(_Base):
    layers: int = Field(default=2, ge=1)
    heads: int = Field(default=4, ge=1)
    ff_multiplier: int = Field(default=2, ge=1)
    causal: bool = True
    max_positions: int = Field(default=512, gt=0)


class RecurrentConfig(_Base):
    cell: Literal["gru", "lstm"] = "gru"
    hidden_dim: int = Field(default=64, gt=0)
    layers: int = Field(default=1, ge=1)


class TCNConfig(_Base):
    channels: list[int] = Field(default_factory=lambda: [64, 64, 64])
    kernel_size: int = Field(default=3, ge=2)


class SSMConfig(_Base):
    """Selective state-space (Mamba-style) expert."""

    state_dim: int = Field(default=8, ge=1)
    expand: int = Field(default=2, ge=1)
    conv_kernel: int = Field(default=4, ge=1)
    layers: int = Field(default=1, ge=1)


class TabularConfig(_Base):
    width: int = Field(default=128, gt=0)
    depth: int = Field(default=3, ge=1)
    activation: Literal["gelu", "silu"] = "gelu"


class GraphModelConfig(_Base):
    enabled: bool = False
    kind: Literal["sage", "gat"] = "sage"
    hidden_dim: int = Field(default=64, gt=0)
    layers: int = Field(default=2, ge=1)
    heads: int = Field(default=2, ge=1)


class GatingConfig(_Base):
    hidden_dim: int = Field(default=64, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    noise_std: float = Field(default=0.1, ge=0)
    """Gaussian noise on gate logits during training (noisy gating)."""
    expert_dropout: float = Field(default=0.05, ge=0, lt=1)
    """Probability of dropping an expert for a sample during training."""
    load_balance_weight: float = Field(default=0.01, ge=0)
    entropy_weight: float = Field(default=0.001, ge=0)
    z_loss_weight: float = Field(default=0.0001, ge=0)


class ModelConfig(_Base):
    d_model: int = Field(default=64, gt=0)
    latent_dim: int = Field(default=64, gt=0)
    head_hidden_dim: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    experts: list[str] = Field(default_factory=lambda: ["transformer", "recurrent", "tcn", "ssm", "tabular"])
    share_encoder_across_timescales: bool = True
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    recurrent: RecurrentConfig = Field(default_factory=RecurrentConfig)
    tcn: TCNConfig = Field(default_factory=TCNConfig)
    ssm: SSMConfig = Field(default_factory=SSMConfig)
    tabular: TabularConfig = Field(default_factory=TabularConfig)
    graph: GraphModelConfig = Field(default_factory=GraphModelConfig)
    gating: GatingConfig = Field(default_factory=GatingConfig)

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
    regression_loss: Literal["gaussian", "huber", "mse"] = "gaussian"
    huber_delta: float = Field(default=1.0, gt=0)
    variance_loss_weight: float = Field(default=0.5, ge=0)
    """When the point loss is huber/mse, weight of a detached-mean Gaussian NLL that still
    trains the aleatoric log-variance heads."""
    quantile_loss_weight: float = Field(default=0.25, ge=0)
    classification_loss: Literal["bce", "weighted_bce", "focal"] = "bce"
    focal_gamma: float = Field(default=2.0, ge=0)
    focal_alpha: float | None = Field(default=None)
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
    learned_uncertainty_weighting: bool = False

    @field_validator("task_weights")
    @classmethod
    def _tw(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(ALL_TASKS)
        if unknown:
            raise ValueError(f"unknown tasks in task_weights: {sorted(unknown)}")
        return v


class TrainingConfig(_Base):
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=128, ge=1)
    eval_batch_size: int = Field(default=512, ge=1)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    scheduler: Literal["cosine", "onecycle", "plateau", "constant"] = "cosine"
    warmup_fraction: float = Field(default=0.05, ge=0, lt=1)
    grad_clip_norm: float | None = Field(default=1.0)
    grad_accumulation_steps: int = Field(default=1, ge=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0)
    mixed_precision: Literal["auto", "off", "fp16", "bf16"] = "auto"
    device: str = "auto"
    seed: int = 7
    deterministic: bool = False
    num_workers: int = Field(default=0, ge=0)
    max_nonfinite_steps: int = Field(default=10, ge=1)
    """Abort training after this many consecutive non-finite losses/gradients."""
    balanced_sampling: bool = False
    rare_event_oversampling: float = Field(default=1.0, ge=1.0)
    """Sampling weight multiplier for rare events (1 = off)."""
    rare_event_return_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    embargo_seconds: float | None = None
    """Gap between train and validation; defaults to the longest horizon (label overlap)."""
    bootstrap_members: bool = False
    max_train_batches_per_epoch: int | None = None


class NormalizationConfig(_Base):
    method: Literal["robust", "standard"] = "robust"
    """robust = median / IQR scaling; standard = mean / std."""
    clip: float = Field(default=8.0, gt=0)
    max_fit_rows: int = Field(default=200_000, ge=100)


class EnsembleConfig(_Base):
    size: int = Field(default=3, ge=1)
    mc_dropout_samples: int = Field(default=2, ge=0)
    """Monte-Carlo dropout passes per member at inference (0 = deterministic single pass)."""
    mc_seed: int = 1234
    embedding_member: int = Field(default=0, ge=0)


class CalibrationConfig(_Base):
    method: Literal["temperature", "platt", "isotonic", "none"] = "temperature"
    n_bins: int = Field(default=10, ge=2)
    min_samples: int = Field(default=50, ge=1)


class OODConfig(_Base):
    shrinkage: float = Field(default=0.1, ge=0, le=1)
    reference_quantile: float = Field(default=0.99, gt=0.5, lt=1)
    embedding_weight: float = Field(default=0.5, ge=0)
    input_weight: float = Field(default=0.25, ge=0)
    disagreement_weight: float = Field(default=0.25, ge=0)
    confidence_penalty: float = Field(default=1.0, ge=0)
    """How quickly confidence decays once ood_score exceeds 1."""


class ReplayConfig(_Base):
    recent_capacity: int = Field(default=5000, ge=1)
    historical_capacity: int = Field(default=20000, ge=1)
    rare_capacity: int = Field(default=5000, ge=1)
    recency_half_life: float = Field(default=3600.0, gt=0)
    """Seconds for recency weight to halve."""
    priority_alpha: float = Field(default=0.6, ge=0)
    priority_beta: float = Field(default=0.4, ge=0)
    priority_epsilon: float = Field(default=1e-3, gt=0)
    rare_return_threshold: float = Field(default=0.05, gt=0)
    """|return| or drawdown above this at any horizon marks an experience as rare."""
    seed: int = 11


class ContinualConfig(_Base):
    min_new_samples: int = Field(default=500, ge=1)
    adapt_samples: int = Field(default=4000, ge=1)
    adapt_epochs: int = Field(default=3, ge=1)
    adapt_learning_rate: float = Field(default=2e-4, gt=0)
    adapt_validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    recent_fraction: float = Field(default=0.5, ge=0)
    historical_fraction: float = Field(default=0.3, ge=0)
    rare_fraction: float = Field(default=0.1, ge=0)
    difficult_fraction: float = Field(default=0.1, ge=0)
    distillation_weight: float = Field(default=0.5, ge=0)
    distillation_temperature: float = Field(default=2.0, gt=0)
    ewc_enabled: bool = True
    ewc_weight: float = Field(default=10.0, ge=0)
    ewc_fisher_batches: int = Field(default=20, ge=1)
    offline_max_degradation: float = Field(default=0.25, ge=0)
    """Candidate is rejected before shadow if its validation loss exceeds the
    champion's by more than this relative margin."""
    full_retrain_min_new_samples: int = Field(default=5000, ge=1)
    full_retrain_recent_weight: float = Field(default=3.0, ge=0)
    full_retrain_historical_weight: float = Field(default=1.0, ge=0)
    full_retrain_rare_weight: float = Field(default=2.0, ge=0)
    full_retrain_difficult_weight: float = Field(default=1.5, ge=0)
    full_retrain_recent_window: int = Field(default=5000, ge=1)
    full_retrain_epochs: int | None = None
    full_retrain_on_drift: bool = True


PretrainTask = Literal["masked_timestep", "masked_feature", "contrastive"]


def _default_pretrain_tasks() -> list[PretrainTask]:
    return ["masked_timestep", "masked_feature", "contrastive"]


class PretrainConfig(_Base):
    tasks: list[PretrainTask] = Field(default_factory=_default_pretrain_tasks)
    mask_ratio: float = Field(default=0.25, gt=0, lt=1)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    contrastive_temperature: float = Field(default=0.2, gt=0)
    augmentation_noise: float = Field(default=0.1, ge=0)
    contrastive_weight: float = Field(default=0.5, ge=0)


class RegimeConfig(_Base):
    method: Literal["kmeans", "gmm", "hdbscan"] = "kmeans"
    n_clusters: int = Field(default=6, ge=2)
    auto_select: bool = False
    """Choose k by silhouette (kmeans) / BIC (gmm) over ``k_min..k_max``."""
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=10, ge=2)
    hdbscan_min_cluster_size: int = Field(default=25, ge=2)
    max_fit_samples: int = Field(default=50_000, ge=10)
    seed: int = 0


class DriftConfig(_Base):
    psi_bins: int = Field(default=10, ge=2)
    psi_threshold: float = Field(default=0.2, gt=0)
    ks_pvalue_threshold: float = Field(default=0.01, gt=0, lt=1)
    wasserstein_threshold: float = Field(default=0.5, gt=0)
    """In reference standard deviations."""
    embedding_distance_threshold: float = Field(default=1.5, gt=0)
    """Current mean Mahalanobis distance / reference mean distance."""
    error_ratio_threshold: float = Field(default=1.3, gt=0)
    feature_fraction_threshold: float = Field(default=0.25, gt=0, le=1)
    """Fraction of drifting features needed to flag input drift overall."""


class PromotionConfig(_Base):
    min_observations: int = Field(default=200, ge=1)
    max_rmse_ratio: float = Field(default=1.0, gt=0)
    max_log_loss_ratio: float = Field(default=1.0, gt=0)
    max_brier_ratio: float = Field(default=1.0, gt=0)
    max_ece_increase: float = Field(default=0.02, ge=0)
    min_rank_corr_delta: float = Field(default=-0.02)
    max_tail_mae_ratio: float = Field(default=1.05, gt=0)
    max_nll_increase: float = Field(default=0.05, ge=0)
    """Allowed increase of the return Gaussian NLL (uncertainty quality)."""
    min_window_win_fraction: float = Field(default=0.5, ge=0, le=1)
    time_windows: int = Field(default=4, ge=1)
    max_regime_degradation: float = Field(default=0.15, ge=0)
    min_regime_observations: int = Field(default=30, ge=1)
    required_gates: list[str] = Field(
        default_factory=lambda: ["min_observations", "return_rmse", "log_loss", "tail_mae"]
    )
    min_passed_fraction: float = Field(default=0.7, ge=0, le=1)
    """Fraction of all (non-required) gates that must pass in addition to required ones."""


class LifecycleConfig(_Base):
    keep_champions: int = Field(default=5, ge=1)
    auto_rollback: bool = False
    auto_rollback_min_observations: int = Field(default=200, ge=1)
    auto_rollback_degradation: float = Field(default=0.3, gt=0)
    """Roll back if live return MAE exceeds the promotion-time baseline by this fraction."""


class NeuralConfig(_Base):
    """Root configuration object."""

    features: FeatureConfig = Field(default_factory=FeatureConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    ood: OODConfig = Field(default_factory=OODConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    continual: ContinualConfig = Field(default_factory=ContinualConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)

    @model_validator(mode="after")
    def _cross(self) -> NeuralConfig:
        if self.model.graph.enabled and "graph" not in self.model.experts:
            raise ValueError("model.graph.enabled requires 'graph' in model.experts")
        if self.ensemble.embedding_member >= self.ensemble.size:
            raise ValueError("ensemble.embedding_member must be < ensemble.size")
        return self

    @property
    def horizon_names(self) -> list[str]:
        return self.targets.horizon_names

    @property
    def embargo_seconds(self) -> float:
        if self.training.embargo_seconds is not None:
            return self.training.embargo_seconds
        return self.targets.max_horizon_seconds

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuralConfig:
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> NeuralConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        return cls.model_validate(raw)


def load_config(path: str | Path | None = None) -> NeuralConfig:
    """Load a YAML config, or return defaults when ``path`` is None."""
    if path is None:
        return NeuralConfig()
    return NeuralConfig.load(path)
