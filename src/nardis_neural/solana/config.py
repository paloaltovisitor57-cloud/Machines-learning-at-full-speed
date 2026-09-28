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
    # edge-seeking dynamics
    "bonding_velocity_60s",
    "holders_growth_60s",
    "smart_buyer_share_60s",
    "top_holder_sell_share_60s",
    "buy_acceleration_30s",
    # token safety
    "mint_authority_revoked",
    "freeze_authority_revoked",
    "lp_burned_fraction",
    # runner-specific wallet skill (early buyers of past 10x runs)
    "tail_smart_buy_share_60s",
    "tail_smart_buyers_300s_log",
    "early_buyer_tail_skill",
    # sybil-resistant counts (funding clusters instead of wallets)
    "holder_clusters_log",
    "holder_cluster_ratio",
    "buyer_clusters_60s_log",
    "top_cluster_share",
    # creator family track record (creator's funder, or the creator itself)
    "creator_prior_launches_log",
    "creator_prior_best_peak_log",
    "creator_prior_rug_rate",
    "creator_prior_graduation_rate",
    "since_creator_last_launch_log",
    # criticality of trade flow (self-exciting Hawkes fit, last 10 minutes)
    "buy_branching_ratio",
    "sell_branching_ratio",
    "buy_branching_trend_120s",
    "herding_timescale_log",
    "endogenous_buy_share",
    # market heat (all tokens)
    "market_launches_600s_log",
    "market_graduations_3600s_log",
    "market_volume_300s_log",
)

FEATURE_DOCS: dict[str, str] = {
    "age_log": "log1p of seconds since launch",
    "n_swaps_log": "log1p of swaps so far",
    "since_last_trade_log": "log1p of seconds since the last trade",
    "bonding_progress": "share of the pump.fun bonding curve completed (1 on an AMM)",
    "migrated": "1 once the token has graduated to an AMM pool",
    "liquidity_sol_log": "log1p of real SOL in the pool (virtual reserves excluded)",
    "liquidity_change_300s": "relative change of real liquidity over 5 min, clipped to [-1, 5]",
    "market_cap_sol_log": "log1p of market cap in SOL (spot price x supply)",
    "round_trip_cost": "fractional cost of buying and selling trade_size_sol now (fees + two-way impact)",
    "drawdown_from_ath": "log price now minus log of the highest traded price (<= 0)",
    "runup_from_low": "log price now minus log of the lowest traded price (>= 0)",
    "ret_5s": "log return over the last 5 s (forward-filled price)",
    "ret_30s": "log return over the last 30 s",
    "ret_60s": "log return over the last 60 s",
    "ret_300s": "log return over the last 5 min",
    "rv_30s": "realised volatility: root of summed squared per-trade log-price changes, 30 s",
    "rv_300s": "realised volatility over 5 min",
    "buys_60s_log": "log1p of buys in the last minute",
    "sells_60s_log": "log1p of sells in the last minute",
    "buy_ratio_60s": "buys / (buys + sells) in the last minute (0.5 when quiet)",
    "buy_sol_share_60s": "buy SOL / total SOL traded in the last minute (0.5 when quiet)",
    "net_flow_60s": "signed log1p of buy SOL minus sell SOL, last minute",
    "net_flow_300s": "signed log1p of buy SOL minus sell SOL, last 5 min",
    "volume_60s_log": "log1p of SOL volume in the last minute",
    "unique_buyers_60s_log": "log1p of distinct buying wallets in the last minute",
    "unique_sellers_60s_log": "log1p of distinct selling wallets in the last minute",
    "new_trader_share_60s": "share of last-minute trades by wallets new to this token in that minute",
    "avg_buy_size_60s_log": "log1p of the mean buy size (SOL) in the last minute",
    "holders_log": "log1p of wallets holding more than one token",
    "top10_share": "supply share of the 10 largest holders",
    "holder_hhi": "Herfindahl index of holder balances",
    "holder_gini": "Gini coefficient of holder balances",
    "dev_share": "creator's share of supply",
    "dev_sold_fraction": "creator's tokens sold / bought (capped at 1)",
    "sniper_share": "supply held by wallets whose first buy landed within sniper_slots of the launch",
    "bundle_share": "supply held by launch-window buyers co-funded with the creator or with each other",
    "fresh_wallet_share_60s": "share of last-minute buyers funded within fresh_wallet_seconds",
    "creator_cluster_share": "supply held by the creator's funding cluster",
    "smart_flow_300s": "signed log1p of 5-min flow weighted by each wallet's 60-s reputation skill",
    "smart_buyers_300s_log": "log1p of distinct 5-min buyers with reputation skill > 0.2",
    "bot_share_60s": "share of last-minute trades from wallets with >= bot_trade_threshold lifetime trades",
    "rug_associated_share": "share of 5-min volume from wallets linked to earlier rugs",
    "priority_fee_60s_log": "log1p of the mean priority fee in the last minute (micro-SOL)",
    "jito_share_60s": "share of last-minute trades that paid a Jito tip",
    "slot_density_60s": "trades in the last minute / 150 (about one per slot)",
    "bonding_velocity_60s": "real curve SOL added in the last minute / 85 (0 once off the curve)",
    "holders_growth_60s": "log1p holders now minus log1p holders a minute ago",
    "smart_buyer_share_60s": "share of last-minute distinct buyers with reputation skill > 0.2",
    "top_holder_sell_share_60s": "share of last-minute sell SOL coming from the current top-10 holders",
    "buy_acceleration_30s": "log1p buys in the last 30 s minus log1p buys in the 30 s before",
    "mint_authority_revoked": "1 if the mint authority is revoked (supply cannot be inflated)",
    "freeze_authority_revoked": "1 if the freeze authority is revoked (holders cannot be frozen)",
    "lp_burned_fraction": "share of LP tokens burned at launch (1 for bonding-curve launches)",
    "tail_smart_buy_share_60s": "share of last-minute buy SOL from wallets with runner skill > 0.5",
    "tail_smart_buyers_300s_log": "log1p of distinct 5-min buyers with runner skill > 0.5",
    "early_buyer_tail_skill": "mean runner skill of the token's first 20 buyers",
    "holder_clusters_log": "log1p of distinct funding clusters among holders",
    "holder_cluster_ratio": "holder clusters / holders (1 = independent wallets, low = sybil crowd)",
    "buyer_clusters_60s_log": "log1p of distinct funding clusters among last-minute buyers",
    "top_cluster_share": "supply held by the largest multi-wallet funding cluster other than the creator's",
    "creator_prior_launches_log": "log1p of the creator family's other launches",
    "creator_prior_best_peak_log": "log of the best peak multiple among the family's other launches",
    "creator_prior_rug_rate": "share of the family's other launches that rugged",
    "creator_prior_graduation_rate": "share of the family's other launches that graduated",
    "since_creator_last_launch_log": "log1p of seconds since the family's previous launch (1e7 if none)",
    "buy_branching_ratio": "Hawkes branching ratio of buys: follow-on buys each buy triggers (→1 = critical)",
    "sell_branching_ratio": "Hawkes branching ratio of sells (panic cascades)",
    "buy_branching_trend_120s": "buy branching ratio now minus two minutes ago (approaching criticality)",
    "herding_timescale_log": "log1p of the fitted excitation timescale 1/β of buys (seconds)",
    "endogenous_buy_share": "share of recent buys attributed to excitation by other buys (herding)",
    "market_launches_600s_log": "log1p of launches across the market in the last 10 min",
    "market_graduations_3600s_log": "log1p of graduations across the market in the last hour",
    "market_volume_300s_log": "log1p of SOL swapped across all tokens in the last 5 min",
}
"""One line per entry of :data:`CURRENT_FEATURES` (a test keeps the two in sync)."""

BAR_FEATURE_DOCS: dict[str, str] = {
    "log_return": "log return of the bar (forward-filled close)",
    "realized_vol": "realised volatility of per-trade log-price changes in the bar",
    "volume_log": "log1p of SOL volume",
    "buy_volume_share": "buy SOL / total SOL (0.5 when empty)",
    "trades_log": "log1p of trades",
    "unique_wallets_log": "log1p of distinct wallets",
    "net_flow": "signed log1p of buy SOL minus sell SOL",
    "liquidity_log": "log1p of real SOL liquidity at the bar's end",
    "priority_fee_log": "log1p of the mean priority fee (micro-SOL)",
    "range": "high minus low log price within the bar",
}

NODE_FEATURE_DOCS: dict[str, str] = {
    "is_token": "1 for the token node, 0 for wallets",
    "volume_log": "log1p of the wallet's SOL volume in this token within graph_window_seconds",
    "position_share": "wallet's balance / supply",
    "reputation": "wallet's 60-s reputation score (posterior mean)",
    "evidence_log": "log1p of the reputation evidence (resolved trades)",
    "is_creator": "1 for the token's creator",
    "is_sniper": "1 if the wallet bought within sniper_slots of launch",
    "cluster_size_log": "log1p of the wallet's funding-cluster size",
}


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
    """One trade-bar stream: its name, bar width in seconds and number of bars."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    name: str
    """Timescale name for this bar series (e.g. fast, medium, slow)."""
    resolution_seconds: float = Field(gt=0)
    """Width of each OHLCV bar in seconds."""
    length: int = Field(gt=0)
    """Number of most recent bars fed to the model for this timescale."""


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
    """Configuration of the Solana layer (``solana.yaml``): horizons, bars, graph, costs, labels and
    wallet intelligence.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    horizons: list[HorizonConfig] = Field(default_factory=_default_horizons)
    """Prediction horizons: name, length in seconds and upside/downside label thresholds."""
    bars: list[BarSpec] = Field(default_factory=_default_bars)
    """Multi-timescale bar series built for each token (one sequence input per entry)."""
    trade_size_sol: float = Field(default=1.0, gt=0)
    """Reference position size for cost-aware labels and net-edge estimates."""
    graph_enabled: bool = True
    """Build the wallet graph input and add the graph expert to the model."""
    graph_top_k: int = Field(default=16, ge=1)
    """Maximum number of wallets (by recent SOL volume) included as graph nodes."""
    graph_window_seconds: float = Field(default=600.0, gt=0)
    """Look-back window in seconds for choosing and weighting graph wallets."""
    fresh_wallet_seconds: float = Field(default=3600.0, gt=0)
    """A wallet first funded less than this many seconds ago counts as fresh."""
    sniper_slots: int = Field(default=2, ge=0)
    """Wallets whose first buy lands within this many slots of launch count as snipers."""
    bundle_slots: int = Field(default=5, ge=0)
    """Early buys within this many slots of launch are checked for bundling (shared cluster)."""
    bot_trade_threshold: int = Field(default=40, ge=1)
    """Wallets with at least this many recorded trades are treated as bots."""
    funding_min_sol: float = Field(default=0.05, ge=0)
    """SOL transfers below this size are ignored when linking funder and funded wallets."""
    hub_threshold: int = Field(default=50, ge=1)
    """A funder of more than this many wallets is a hub (exchange-like) and does not cluster them."""
    reputation_prior: tuple[float, float] = (1.0, 1.0)
    """Beta prior (alpha, beta) of each wallet's buy success rate."""
    reputation_horizon: str = "60s"
    """Name of the horizon after which a wallet's buy is scored for reputation."""
    reputation_success_return: float = Field(default=0.10)
    """Log-price return over the reputation horizon above which a buy counts as a success."""
    reputation_entry_window: float = Field(default=60.0, gt=0)
    """Reserved: not currently read by the Solana pipeline."""
    tail_entry_window: float = Field(default=300.0, gt=0)
    """Buys in a token's first seconds that count towards runner-specific (tail) skill."""
    tail_horizon_seconds: float = Field(default=5400.0, gt=0)
    """Seconds an early buy has to reach ``tail_multiple`` to count as a tail success."""
    tail_multiple: float = Field(default=10.0, gt=1)
    """An early buy is a tail success if price reaches this multiple of its entry in time."""
    tail_prior: tuple[float, float] = (0.1, 1.9)
    """Beta prior of the tail hit rate (mean 5 %: runners are rare)."""
    risk_horizon_seconds: float = Field(default=300.0, gt=0)
    """Window in seconds after a snapshot over which rug, graduation and dev-dump labels are set."""
    rug_price_drop: float = Field(default=0.6, gt=0, lt=1)
    """Fractional price fall within the risk horizon that labels a rug (unless the token graduated)."""
    rug_liquidity_drop: float = Field(default=0.8, gt=0, lt=1)
    """Fraction of pool SOL removed in a liquidity pull that labels a rug."""
    dev_dump_fraction: float = Field(default=0.5, gt=0, le=1)
    """Share of the creator's holdings sold that counts as a dev dump."""
    sample_interval_seconds: float = Field(default=10.0, gt=0)
    """Spacing of training snapshots per token when building datasets from event logs."""
    min_token_age_seconds: float = Field(default=3.0, ge=0)
    """Minimum token age in seconds before it is first sampled or assessed."""

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
        """Write the configuration to ``path`` as YAML."""
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> SolanaConfig:
        """Read a configuration from a YAML file (an empty file gives the defaults)."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})
