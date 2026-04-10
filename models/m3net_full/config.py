"""Configuration for the GPU-first full M3Net architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class M3NetFullConfig:
    """High-capacity M3Net configuration."""

    label_horizon: int = 5
    min_history: int = 120
    daily_lookback: int = 60
    minute_lookback_days: int = 20
    top_n: int = 20

    daily_input_dim: int = 8
    intraday_input_dim: int = 16
    memory_input_dim: int = 8
    factor_input_dim: int = 24

    hidden_dim: int = 128
    daily_layers: int = 3
    intraday_layers: int = 2
    fusion_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1

    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 20
    grad_clip_norm: float = 1.0
    train_ratio: float = 0.8
    random_state: int = 42

    device: str = "cuda"
    precision: Literal["fp32", "amp"] = "amp"

    return_loss_weight: float = 1.0
    rank_loss_weight: float = 0.45
    listwise_loss_weight: float = 0.25
    risk_loss_weight: float = 0.2
    confidence_loss_weight: float = 0.1
    top_pick_loss_weight: float = 0.08

    top_pick_quantile: float = 0.9
    weighted_return_alpha: float = 1.0
    rank_gap_power: float = 1.25
    listwise_temperature: float = 0.35
    listwise_topk_focus: int = 5
    listwise_tail_weight: float = 0.2

    graph_neighbor_k: int = 8
    graph_residual_weight: float = 0.4
    graph_temperature: float = 0.7
    graph_contrastive_loss_weight: float = 0.08
    graph_contrastive_temperature: float = 0.2
    graph_contrastive_neighbors: int = 6

    score_return_weight: float = 1.0
    score_top_pick_weight: float = 0.15
    score_confidence_weight: float = 0.1
    score_risk_weight: float = 0.1

    factor_columns: list[str] = field(default_factory=list)
    minute_summary_columns: list[str] = field(default_factory=list)
