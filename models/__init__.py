"""Machine learning models for quantitative stock selection."""

from .gradient_boosting_factor import (
    GradientBoostingFactorModel,
    MachineLearningStockSelector,
    PanelFactorDatasetBuilder,
)
from .m3net import M3NetStage1Config, M3NetStage1Model

try:
    from .m3net_full import M3NetFullConfig, M3NetFullDatasetBuilder, M3NetFullModel
except ImportError:  # pragma: no cover - optional GPU stack
    M3NetFullConfig = None
    M3NetFullDatasetBuilder = None
    M3NetFullModel = None

__all__ = [
    "GradientBoostingFactorModel",
    "MachineLearningStockSelector",
    "M3NetStage1Config",
    "M3NetStage1Model",
    "M3NetFullConfig",
    "M3NetFullDatasetBuilder",
    "M3NetFullModel",
    "PanelFactorDatasetBuilder",
]
