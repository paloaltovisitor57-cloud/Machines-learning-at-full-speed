"""Solana-specific configuration and its mapping onto the generic neural configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from nardis_neural.config import HorizonConfig, NeuralConfig, TimescaleConfig

CURRENT_FEATURES: tuple[str, ...] = (
    # lifecycle / pool state
    "age_log",
    "n_swaps_log",
    "since_last_trade_log",
    "bonding_progress",
    "migrated",
    "liquidity_sol_log",
    "liquidity_change_300s",
    "market_cap_sol_log",
    "round_trip_cost",
    "drawdown_from_ath",
    "runup_from_low",
    # price action
    "ret_5s",
    "ret_30s",
    "ret_60s",
    "ret_300s",
    "rv_30s",
    "rv_300s",
    # order flow
    "buys_60s_log",
    "sells_60s_log",
    "buy_ratio_60s",
    "buy_sol_share_60s",
    "net_flow_60s",
    "net_flow_300s",
    "volume_60s_log",
    "unique_buyers_60s_log",
    "unique_sellers_60s_log",
    "new_trader_share_60s",
    "avg_buy_size_60s_log",
    # holders & insiders
    "holders_log",
    "top10_share",
    "holder_hhi",
    "holder_gini",
    "dev_share",
    "dev_sold_fraction",
    "sniper_share",
    "bundle_share",
    "fresh_wallet_share_60s",
    "creator_cluster_share",
    # smart money & bots
    "smart_flow_300s",
    "smart_buyers_300s_log",
    "bot_share_60s",
    "rug_associated_share",
    # execution competition
    "priority_fee_60s_log",
    "jito_share_60s",
    "slot_density_60s",
    # token safety
    "mint_authority_revoked",
    "freeze_authority_revoked",
    "lp_burned_fraction",
)

BAR_FEATURES: tuple[str, ...] = (
    "log_return",
    "realized_vol",
    "volume_log",
    "buy_volume_share",
    "trades_log",
    "unique_wallets_log",
    "net_flow",
    "liquidity_log",
    "priority_fee_log",
    "range",
)

NODE_FEATURES: tuple[str, ...] = (
    "is_token",
    "volume_log",
    "position_share",
    "reputation",
    "evidence_log",
    "is_creator",
    "is_sniper",
    "cluster_size_log",
)

EDGE_TYPES: tuple[str, ...] = ("wallet_trades_token", "wallet_funded_wallet", "wallet_same_cluster")

RISK_LABELS: tuple[str, ...] = ("rug", "graduation", "dev_dump")


class BarSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    resolution_seconds: float = Field(gt=0)
    length: int = Field(gt=0)


def _default_horizons() -> list[HorizonConfig]:
    return [
        HorizonConfig(name="15s", seconds=15, upside_threshold=0.05, downside_threshold=0.05),
        HorizonConfig(name="60s", seconds=60, upside_threshold=0.12, downside_threshold=0.10),
        HorizonConfig(name="5m", seconds=300, upside_threshold=0.25, downside_threshold=0.20),
    ]


def _default_bars() -> list[BarSpec]:
    return [
        BarSpec(name="fast", resolution_seconds=1, length=60),
        BarSpec(name="medium", resolution_seconds=5, length=48),
        BarSpec(name="slow", resolution_seconds=30, length=40),
    ]


class SolanaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizons: list[HorizonConfig] = Field(default_factory=_default_horizons)
    bars: list[BarSpec] = Field(default_factory=_default_bars)
    trade_size_sol: float = Field(default=1.0, gt=0)
    """Reference position size for cost-aware labels and net-edge estimates."""
    graph_enabled: bool = True
    graph_top_k: int = Field(default=16, ge=1)
    graph_window_seconds: float = Field(default=600.0, gt=0)
    fresh_wallet_seconds: float = Field(default=3600.0, gt=0)
    sniper_slots: int = Field(default=2, ge=0)
    bundle_slots: int = Field(default=5, ge=0)
    bot_trade_threshold: int = Field(default=40, ge=1)
    funding_min_sol: float = Field(default=0.05, ge=0)
    hub_threshold: int = Field(default=50, ge=1)
    reputation_prior: tuple[float, float] = (1.0, 1.0)
    reputation_horizon: str = "60s"
    reputation_success_return: float = Field(default=0.10)
    reputation_entry_window: float = Field(default=60.0, gt=0)
    risk_horizon_seconds: float = Field(default=300.0, gt=0)
    rug_price_drop: float = Field(default=0.6, gt=0, lt=1)
    rug_liquidity_drop: float = Field(default=0.8, gt=0, lt=1)
    dev_dump_fraction: float = Field(default=0.5, gt=0, le=1)
    sample_interval_seconds: float = Field(default=10.0, gt=0)
    """Spacing of training snapshots per token when building datasets from event logs."""
    min_token_age_seconds: float = Field(default=3.0, ge=0)

    def neural_config(self, base: NeuralConfig | None = None) -> NeuralConfig:
        """A neural configuration whose input contract matches the Solana feature builder."""
        cfg = (base or NeuralConfig()).model_copy(deep=True)
        cfg.features.current_dim = len(CURRENT_FEATURES)
        cfg.features.timescales = [
            TimescaleConfig(
                name=b.name,
                feature_dim=len(BAR_FEATURES),
                max_len=b.length,
                resolution_seconds=b.resolution_seconds,
            )
            for b in self.bars
        ]
        cfg.features.graph.node_feature_dim = len(NODE_FEATURES)
        cfg.features.graph.num_edge_types = len(EDGE_TYPES)
        cfg.targets.horizons = [h.model_copy() for h in self.horizons]
        if self.graph_enabled:
            if "graph" not in cfg.model.experts:
                cfg.model.experts = [*cfg.model.experts, "graph"]
            cfg.model.graph.enabled = True
        cfg.replay.rare_return_threshold = max(cfg.replay.rare_return_threshold, 0.3)
        return NeuralConfig.model_validate(cfg.model_dump())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> SolanaConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})
