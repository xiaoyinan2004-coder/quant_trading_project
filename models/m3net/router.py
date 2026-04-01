"""Adaptive fusion logic for the first-stage M3-Net."""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdaptiveExpertRouter:
    """Blend tabular and sequence experts with lightweight regime-aware weights."""

    def combine(
        self,
        frame: pd.DataFrame,
        tabular_col: str = "tabular_score",
        sequence_col: str = "sequence_score",
    ) -> pd.DataFrame:
        fused = frame.copy()
        has_minute = fused.get("has_minute_features", pd.Series(False, index=fused.index)).astype(float)
        market_vol = fused.get("market_return_std", pd.Series(0.0, index=fused.index)).fillna(0.0)
        breadth = fused.get("positive_ratio", pd.Series(0.5, index=fused.index)).fillna(0.5)
        intraday_vol = fused.get("intraday_volatility", pd.Series(0.0, index=fused.index)).fillna(0.0)

        sequence_weight = 0.20 + 0.20 * has_minute + 0.25 * np.tanh(intraday_vol * 12.0)
        sequence_weight += 0.10 * np.tanh(market_vol * 8.0)
        sequence_weight -= 0.08 * np.abs(breadth - 0.5) * 2.0
        sequence_weight = np.clip(sequence_weight, 0.10, 0.55)

        fused["sequence_weight"] = sequence_weight
        fused["tabular_weight"] = 1.0 - fused["sequence_weight"]
        fused["score"] = (
            fused["tabular_weight"] * fused[tabular_col].fillna(0.0)
            + fused["sequence_weight"] * fused[sequence_col].fillna(0.0)
        )
        return fused

