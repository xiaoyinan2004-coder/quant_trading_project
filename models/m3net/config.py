"""Configuration objects for the first runnable M3-Net stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class M3NetStage1Config:
    """Training configuration for the lightweight M3-Net implementation."""

    factor_backend: str = "lightgbm"
    factor_model_params: Dict[str, float] = field(
        default_factory=lambda: {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
    )
    sequence_model_params: Dict[str, float] = field(
        default_factory=lambda: {
            "max_depth": 4,
            "learning_rate": 0.05,
            "max_iter": 120,
        }
    )
    label_horizon: int = 5
    train_ratio: float = 0.8
    min_history: int = 80
    sequence_lookback: int = 20
    top_n: int = 20
    random_state: int = 42

