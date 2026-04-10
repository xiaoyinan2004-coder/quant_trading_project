"""Runnable first-stage M3-Net model for the current project."""

from __future__ import annotations

import gc
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
    fused_valid_rank_ic: float
    tabular_valid_rank_ic: float
    sequence_valid_rank_ic: float
    router_mode: str


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
        self.memory_bank = MarketMemoryBank(
            short_lookback=self.config.alpha_memory_lookback_short,
            long_lookback=self.config.alpha_memory_lookback_long,
        )
        self.router = AdaptiveExpertRouter(
            use_learned_weights=self.config.use_learned_router,
            model_params=self.config.router_model_params,
            random_state=self.config.random_state,
        )
        self.report: Optional[M3NetStage1Report] = None
        self._cached_scored_panel: Optional[pd.DataFrame] = None

    def fit(
        self,
        stock_data: Dict[str, pd.DataFrame],
        minute_data: Optional[Dict[str, pd.DataFrame]] = None,
        label_col: str = "label",
    ) -> "M3NetStage1Model":
        self._cached_scored_panel = None
        factor_panel = self.dataset_builder.build_dataset(
            stock_data,
            label_horizon=self.config.label_horizon,
            min_history=self.config.min_history,
            label_col=label_col,
        )
        factor_panel = _downcast_numeric_columns(factor_panel)
        sequence_panel = self.sequence_builder.build_panel(stock_data, minute_data=minute_data)
        sequence_panel = _downcast_numeric_columns(sequence_panel)
        base_panel = factor_panel[["date", "symbol", label_col]].copy()
        merged = base_panel.merge(sequence_panel, on=["date", "symbol"], how="left")
        merged = _downcast_numeric_columns(merged)

        for column in self.sequence_builder.feature_columns:
            if column not in merged.columns:
                merged[column] = 0.0 if column == "has_minute_features" else pd.NA

        self.factor_model.fit(
            factor_panel,
            label_col=label_col,
            train_ratio=self.config.train_ratio,
        )
        sequence_input = merged[["date", "symbol", label_col, *self.sequence_builder.feature_columns]].copy()
        self.sequence_model.fit(
            sequence_input,
            label_col=label_col,
            feature_columns=self.sequence_builder.feature_columns,
            train_ratio=self.config.train_ratio,
        )
        gc.collect()

        tabular_scored = factor_panel[["date", "symbol", label_col]].copy()
        tabular_scored["tabular_score"] = self.factor_model.predict_scores(factor_panel)
        sequence_scored = sequence_input[["date", "symbol", *self.sequence_builder.feature_columns]].copy()
        sequence_scored["sequence_score"] = self.sequence_model.predict_scores(sequence_input)
        sequence_scored = _downcast_numeric_columns(sequence_scored)
        del base_panel
        del sequence_input
        del sequence_panel
        del merged
        gc.collect()

        fused = tabular_scored.merge(
            sequence_scored[["date", "symbol", "sequence_score", *self.sequence_builder.feature_columns]],
            on=["date", "symbol"],
            how="left",
        )
        fused = _downcast_numeric_columns(fused)
        del factor_panel
        del tabular_scored
        del sequence_scored
        gc.collect()
        self.memory_bank.fit(fused, label_col=label_col)
        fused = self._attach_memory(fused)
        self.router.fit(fused, label_col=label_col)
        fused = self.router.combine(fused)
        fused = _downcast_numeric_columns(fused)
        latest_date = pd.to_datetime(fused["date"]).max() if not fused.empty else None
        if latest_date is not None:
            self._cached_scored_panel = fused.loc[pd.to_datetime(fused["date"]) == latest_date].reset_index(drop=True).copy()

        train_frame, valid_frame = self._time_split(fused)
        fused_ic = _rank_ic_by_date(valid_frame, label_col=label_col, score_col="score")
        tabular_ic = _rank_ic_by_date(valid_frame, label_col=label_col, score_col="tabular_score")
        sequence_ic = _rank_ic_by_date(valid_frame, label_col=label_col, score_col="sequence_score")
        self.report = M3NetStage1Report(
            factor_backend=self.config.factor_backend,
            train_rows=len(train_frame),
            valid_rows=len(valid_frame),
            fused_valid_rmse=_rmse(valid_frame[label_col], valid_frame["score"]) if not valid_frame.empty else float("nan"),
            tabular_valid_rmse=_rmse(valid_frame[label_col], valid_frame["tabular_score"]) if not valid_frame.empty else float("nan"),
            sequence_valid_rmse=_rmse(valid_frame[label_col], valid_frame["sequence_score"]) if not valid_frame.empty else float("nan"),
            fused_valid_rank_ic=float(fused_ic.mean(skipna=True)) if not valid_frame.empty else float("nan"),
            tabular_valid_rank_ic=float(tabular_ic.mean(skipna=True)) if not valid_frame.empty else float("nan"),
            sequence_valid_rank_ic=float(sequence_ic.mean(skipna=True)) if not valid_frame.empty else float("nan"),
            router_mode=str(fused["router_mode"].iloc[0]) if not fused.empty and "router_mode" in fused.columns else "heuristic",
        )
        return self

    def predict(
        self,
        stock_data: Dict[str, pd.DataFrame],
        minute_data: Optional[Dict[str, pd.DataFrame]] = None,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        cached = self._predict_from_cache(as_of_date=as_of_date)
        if cached is not None:
            return cached

        factor_panel = self.dataset_builder.build_dataset(
            stock_data,
            label_horizon=self.config.label_horizon,
            min_history=self.config.min_history,
            drop_na_label=False,
        )
        factor_panel = _downcast_numeric_columns(factor_panel)
        sequence_panel = self.sequence_builder.build_panel(stock_data, minute_data=minute_data)
        sequence_panel = _downcast_numeric_columns(sequence_panel)
        base_panel = factor_panel[["date", "symbol"]].copy()
        merged = base_panel.merge(sequence_panel, on=["date", "symbol"], how="left")
        merged = _downcast_numeric_columns(merged)

        for column in self.sequence_builder.feature_columns:
            if column not in merged.columns:
                merged[column] = 0.0 if column == "has_minute_features" else pd.NA

        sequence_input = merged[["date", "symbol", *self.sequence_builder.feature_columns]].copy()
        tabular_scored = factor_panel[["date", "symbol"]].copy()
        tabular_scored["tabular_score"] = self.factor_model.predict_scores(factor_panel)
        sequence_scored = sequence_input[["date", "symbol", *self.sequence_builder.feature_columns]].copy()
        sequence_scored["sequence_score"] = self.sequence_model.predict_scores(sequence_input)
        sequence_scored = _downcast_numeric_columns(sequence_scored)
        del base_panel
        del sequence_input
        del sequence_panel
        del merged
        gc.collect()

        fused = tabular_scored.merge(
            sequence_scored[["date", "symbol", "sequence_score", *self.sequence_builder.feature_columns]],
            on=["date", "symbol"],
            how="left",
        )
        fused = _downcast_numeric_columns(fused)
        del factor_panel
        del tabular_scored
        del sequence_scored
        gc.collect()
        fused = self._attach_memory(fused)
        fused = self.router.combine(fused)
        fused = _downcast_numeric_columns(fused)
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
        instance._cached_scored_panel = None
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
        train_frame = frame.loc[pd.to_datetime(frame["date"]) <= cutoff]
        valid_frame = frame.loc[pd.to_datetime(frame["date"]) > cutoff]
        return train_frame, valid_frame

    def _predict_from_cache(self, as_of_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        if self._cached_scored_panel is None or self._cached_scored_panel.empty:
            return None

        cached = self._cached_scored_panel
        if as_of_date is None:
            return cached.copy()

        target_date = pd.Timestamp(as_of_date)
        if target_date not in set(pd.to_datetime(cached["date"]).unique()):
            return None
        return cached.loc[pd.to_datetime(cached["date"]) == target_date].reset_index(drop=True)


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return float("nan")
    diff = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float((diff**2).mean() ** 0.5)


def _rank_ic_by_date(frame: pd.DataFrame, label_col: str, score_col: str) -> pd.Series:
    if frame.empty or score_col not in frame.columns:
        return pd.Series(dtype=float)

    def _date_ic(group: pd.DataFrame) -> float:
        if group[label_col].nunique(dropna=True) <= 1:
            return float("nan")
        if group[score_col].nunique(dropna=True) <= 1:
            return float("nan")
        return float(group[score_col].corr(group[label_col], method="spearman"))

    return frame.groupby("date", group_keys=False).apply(_date_ic)


def _downcast_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()
    numeric_columns = result.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if pd.api.types.is_float_dtype(result[column]) or pd.api.types.is_integer_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], downcast="float")
    return result
