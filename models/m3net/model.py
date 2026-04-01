"""Runnable first-stage M3-Net model for the current project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from models.gradient_boosting_factor import GradientBoostingFactorModel, PanelFactorDatasetBuilder

from .config import M3NetStage1Config
from .memory import MarketMemoryBank
from .router import AdaptiveExpertRouter
from .sequence import SequenceExpertModel, SequenceFeatureBuilder


@dataclass
class M3NetStage1Report:
    """High-level report for Stage 1 training."""

    factor_backend: str
    train_rows: int
    valid_rows: int
    fused_valid_rmse: float
    tabular_valid_rmse: float
    sequence_valid_rmse: float


class M3NetStage1Model:
    """First executable M3-Net stage.

    This version keeps the project grounded on CPU-friendly models:
    - tabular expert: LightGBM/XGBoost factor model
    - sequence expert: sklearn histogram boosting on daily/minute sequence features
    - memory: daily market regime summaries
    - router: regime-aware deterministic fusion
    """

    def __init__(self, config: Optional[M3NetStage1Config] = None) -> None:
        self.config = config or M3NetStage1Config()
        self.dataset_builder = PanelFactorDatasetBuilder()
        self.factor_model = GradientBoostingFactorModel(
            backend=self.config.factor_backend,
            model_params=self.config.factor_model_params,
            random_state=self.config.random_state,
        )
        self.sequence_builder = SequenceFeatureBuilder(lookback=self.config.sequence_lookback)
        self.sequence_model = SequenceExpertModel(
            model_params=self.config.sequence_model_params,
            random_state=self.config.random_state,
        )
        self.memory_bank = MarketMemoryBank()
        self.router = AdaptiveExpertRouter()
        self.report: Optional[M3NetStage1Report] = None

    def fit(
        self,
        stock_data: Dict[str, pd.DataFrame],
        minute_data: Optional[Dict[str, pd.DataFrame]] = None,
        label_col: str = "label",
    ) -> "M3NetStage1Model":
        factor_panel = self.dataset_builder.build_dataset(
            stock_data,
            label_horizon=self.config.label_horizon,
            min_history=self.config.min_history,
            label_col=label_col,
        )
        sequence_panel = self.sequence_builder.build_panel(stock_data, minute_data=minute_data)
        merged = factor_panel.merge(sequence_panel, on=["date", "symbol"], how="left")

        for column in self.sequence_builder.feature_columns:
            if column not in merged.columns:
                merged[column] = 0.0 if column == "has_minute_features" else pd.NA

        self.factor_model.fit(
            factor_panel,
            label_col=label_col,
            train_ratio=self.config.train_ratio,
        )
        self.sequence_model.fit(
            merged[["date", "symbol", label_col, *self.sequence_builder.feature_columns]].copy(),
            label_col=label_col,
            feature_columns=self.sequence_builder.feature_columns,
            train_ratio=self.config.train_ratio,
        )

        tabular_scored = self.factor_model.predict(factor_panel).rename(columns={"score": "tabular_score"})
        sequence_scored = self.sequence_model.predict(
            merged[["date", "symbol", label_col, *self.sequence_builder.feature_columns]].copy()
        ).rename(columns={"score": "sequence_score"})

        fused = tabular_scored.merge(
            sequence_scored[["date", "symbol", "sequence_score", *self.sequence_builder.feature_columns]],
            on=["date", "symbol"],
            how="left",
        )
        self.memory_bank.fit(fused.assign(label=fused[label_col]), label_col=label_col)
        fused = self._attach_memory(fused)
        fused = self.router.combine(fused)

        train_frame, valid_frame = self._time_split(fused)
        self.report = M3NetStage1Report(
            factor_backend=self.config.factor_backend,
            train_rows=len(train_frame),
            valid_rows=len(valid_frame),
            fused_valid_rmse=_rmse(valid_frame[label_col], valid_frame["score"]) if not valid_frame.empty else float("nan"),
            tabular_valid_rmse=_rmse(valid_frame[label_col], valid_frame["tabular_score"]) if not valid_frame.empty else float("nan"),
            sequence_valid_rmse=_rmse(valid_frame[label_col], valid_frame["sequence_score"]) if not valid_frame.empty else float("nan"),
        )
        return self

    def predict(
        self,
        stock_data: Dict[str, pd.DataFrame],
        minute_data: Optional[Dict[str, pd.DataFrame]] = None,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        factor_panel = self.dataset_builder.build_dataset(
            stock_data,
            label_horizon=self.config.label_horizon,
            min_history=self.config.min_history,
            drop_na_label=False,
        )
        sequence_panel = self.sequence_builder.build_panel(stock_data, minute_data=minute_data)
        merged = factor_panel.merge(sequence_panel, on=["date", "symbol"], how="left")

        for column in self.sequence_builder.feature_columns:
            if column not in merged.columns:
                merged[column] = 0.0 if column == "has_minute_features" else pd.NA

        tabular_scored = self.factor_model.predict(factor_panel).rename(columns={"score": "tabular_score"})
        sequence_scored = self.sequence_model.predict(
            merged[["date", "symbol", *self.sequence_builder.feature_columns]].copy()
        ).rename(columns={"score": "sequence_score"})

        fused = tabular_scored.merge(
            sequence_scored[["date", "symbol", "sequence_score", *self.sequence_builder.feature_columns]],
            on=["date", "symbol"],
            how="left",
        )
        fused = self._attach_memory(fused)
        fused = self.router.combine(fused)
        if as_of_date is not None:
            target_date = pd.Timestamp(as_of_date)
            fused = fused.loc[fused["date"] == target_date].reset_index(drop=True)
        return fused

    def select_top_stocks(
        self,
        stock_data: Dict[str, pd.DataFrame],
        minute_data: Optional[Dict[str, pd.DataFrame]] = None,
        top_n: Optional[int] = None,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        scored = self.predict(stock_data, minute_data=minute_data, as_of_date=as_of_date)
        if scored.empty:
            return scored
        if as_of_date is None:
            as_of_date = str(pd.to_datetime(scored["date"]).max().date())
        target_date = pd.Timestamp(as_of_date)
        return (
            scored.loc[scored["date"] == target_date]
            .sort_values("score", ascending=False)
            .head(top_n or self.config.top_n)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "factor_model": self.factor_model,
            "sequence_model": self.sequence_model,
            "memory_bank": self.memory_bank,
            "report": self.report,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "M3NetStage1Model":
        payload = joblib.load(path)
        instance = cls(config=payload["config"])
        instance.factor_model = payload["factor_model"]
        instance.sequence_model = payload["sequence_model"]
        instance.memory_bank = payload["memory_bank"]
        instance.report = payload["report"]
        return instance

    def _attach_memory(self, frame: pd.DataFrame) -> pd.DataFrame:
        memory = self.memory_bank.get_features(frame["date"])
        merged = frame.merge(memory, on="date", how="left")
        return merged

    def _time_split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = sorted(pd.to_datetime(frame["date"]).dropna().unique())
        if len(unique_dates) < 2:
            return frame.copy(), frame.iloc[0:0].copy()
        split_index = min(max(int(len(unique_dates) * self.config.train_ratio), 1), len(unique_dates) - 1)
        cutoff = pd.Timestamp(unique_dates[split_index - 1])
        train_frame = frame.loc[pd.to_datetime(frame["date"]) <= cutoff].copy()
        valid_frame = frame.loc[pd.to_datetime(frame["date"]) > cutoff].copy()
        return train_frame, valid_frame


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return float("nan")
    diff = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float((diff**2).mean() ** 0.5)

