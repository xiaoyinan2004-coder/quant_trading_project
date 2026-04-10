"""Adaptive fusion logic for the first-stage M3-Net."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class AdaptiveExpertRouter:
    """Blend tabular and sequence experts with lightweight regime-aware weights."""

    FEATURE_COLUMNS = [
        "has_minute_features",
        "market_return_mean",
        "market_return_std",
        "positive_ratio",
        "score_dispersion",
        "tabular_rank_ic_20d",
        "sequence_rank_ic_20d",
        "tabular_rank_ic_60d",
        "sequence_rank_ic_60d",
        "expert_ic_gap_20d",
        "intraday_volatility",
        "realized_volatility",
        "intraday_volume_ratio",
        "volume_burst_ratio",
        "abs_score_gap",
    ]

    def __init__(
        self,
        use_learned_weights: bool = True,
        model_params: Optional[Dict[str, float]] = None,
        random_state: int = 42,
    ) -> None:
        self.use_learned_weights = use_learned_weights
        self.model_params = model_params or {}
        self.random_state = random_state
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.feature_medians: Dict[str, float] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        label_col: str = "label",
        tabular_col: str = "tabular_score",
        sequence_col: str = "sequence_score",
    ) -> "AdaptiveExpertRouter":
        if not self.use_learned_weights:
            return self

        train = frame.copy()
        required = [label_col, tabular_col, sequence_col]
        train = train.dropna(subset=required)
        if train.empty:
            return self

        features = self._prepare_feature_frame(train, fit=True)
        tabular_error = np.square(train[label_col].astype(float) - train[tabular_col].astype(float))
        sequence_error = np.square(train[label_col].astype(float) - train[sequence_col].astype(float))
        total_error = tabular_error + sequence_error + 1e-8
        target = np.clip(tabular_error / total_error, 0.05, 0.95)

        params = {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 120,
            "min_samples_leaf": 40,
            "random_state": self.random_state,
        }
        params.update(self.model_params)
        self.model = HistGradientBoostingRegressor(**params)
        self.model.fit(features, target)
        return self

    def combine(
        self,
        frame: pd.DataFrame,
        tabular_col: str = "tabular_score",
        sequence_col: str = "sequence_score",
    ) -> pd.DataFrame:
        fused = frame.copy()
        if self.model is not None and self.use_learned_weights:
            features = self._prepare_feature_frame(fused, fit=False)
            sequence_weight = np.clip(self.model.predict(features), 0.05, 0.95)
        else:
            sequence_weight = self._heuristic_weight(fused)

        fused["sequence_weight"] = sequence_weight
        fused["tabular_weight"] = 1.0 - fused["sequence_weight"]
        fused["router_mode"] = "learned" if self.model is not None and self.use_learned_weights else "heuristic"
        fused["score"] = (
            fused["tabular_weight"] * fused[tabular_col].fillna(0.0)
            + fused["sequence_weight"] * fused[sequence_col].fillna(0.0)
        )
        return fused

    def _heuristic_weight(self, frame: pd.DataFrame) -> np.ndarray:
        has_minute = frame.get("has_minute_features", pd.Series(False, index=frame.index)).astype(float)
        market_vol = frame.get("market_return_std", pd.Series(0.0, index=frame.index)).fillna(0.0)
        breadth = frame.get("positive_ratio", pd.Series(0.5, index=frame.index)).fillna(0.5)
        intraday_vol = frame.get("intraday_volatility", pd.Series(0.0, index=frame.index)).fillna(0.0)
        realized_vol = frame.get("realized_volatility", pd.Series(0.0, index=frame.index)).fillna(0.0)
        alpha_gap = frame.get("expert_ic_gap_20d", pd.Series(0.0, index=frame.index)).fillna(0.0)

        sequence_weight = 0.20 + 0.18 * has_minute + 0.18 * np.tanh(intraday_vol * 12.0)
        sequence_weight += 0.08 * np.tanh(realized_vol * 8.0)
        sequence_weight += 0.10 * np.tanh(alpha_gap * 3.0)
        sequence_weight += 0.08 * np.tanh(market_vol * 8.0)
        sequence_weight -= 0.08 * np.abs(breadth - 0.5) * 2.0
        return np.clip(sequence_weight, 0.10, 0.60)

    def _prepare_feature_frame(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        features = frame.copy()
        features["abs_score_gap"] = (
            pd.to_numeric(features.get("tabular_score", 0.0), errors="coerce").fillna(0.0)
            - pd.to_numeric(features.get("sequence_score", 0.0), errors="coerce").fillna(0.0)
        ).abs()
        prepared = pd.DataFrame(index=features.index)
        for column in self.FEATURE_COLUMNS:
            prepared[column] = pd.to_numeric(features.get(column, 0.0), errors="coerce")

        if fit:
            self.feature_medians = prepared.median().fillna(0.0).to_dict()

        return prepared.fillna(self.feature_medians)
