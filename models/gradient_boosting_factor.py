"""Gradient boosting factor models for cross-sectional stock selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from factors.a_share_factors import AShareFactorCalculator

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - handled at runtime
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - handled at runtime
    xgb = None


def _safe_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    diff = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else float("nan")


def _mean_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return float("nan")
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def _rank_ic_by_date(frame: pd.DataFrame, label_col: str, score_col: str) -> pd.Series:
    def _date_ic(group: pd.DataFrame) -> float:
        if group[label_col].nunique(dropna=True) <= 1:
            return np.nan
        if group[score_col].nunique(dropna=True) <= 1:
            return np.nan
        return group[score_col].corr(group[label_col], method="spearman")

    return frame.groupby("date", group_keys=False).apply(_date_ic)


@dataclass
class FactorModelReport:
    """Summary statistics for a model training run."""

    train_rows: int
    valid_rows: int
    feature_count: int
    train_rmse: float
    valid_rmse: float
    train_rank_ic: float
    valid_rank_ic: float
    valid_directional_accuracy: float


class PanelFactorDatasetBuilder:
    """Build a panel factor dataset from per-symbol OHLCV DataFrames."""

    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
    META_COLUMNS = ("date", "symbol")

    def __init__(
        self,
        factor_calculator: Optional[AShareFactorCalculator] = None,
        winsor_limits: Tuple[float, float] = (0.01, 0.99),
        normalize_cross_section: bool = True,
    ) -> None:
        self.factor_calculator = factor_calculator or AShareFactorCalculator()
        self.winsor_limits = winsor_limits
        self.normalize_cross_section = normalize_cross_section

    def build_dataset(
        self,
        stock_data: Dict[str, pd.DataFrame],
        label_horizon: int = 5,
        label_col: str = "label",
        min_history: int = 80,
        drop_na_label: bool = True,
    ) -> pd.DataFrame:
        """Create a cross-sectional factor panel suitable for model training."""
        frames: List[pd.DataFrame] = []

        for symbol, raw_df in stock_data.items():
            if raw_df is None or raw_df.empty:
                continue

            df = raw_df.copy()
            self._validate_price_frame(df, symbol)
            df = df.sort_index()
            if len(df) < max(min_history, label_horizon + 20):
                continue

            factors = self.factor_calculator.calculate_all_factors(df).copy()
            factors["symbol"] = symbol
            factors["date"] = pd.to_datetime(factors.index)
            factors[label_col] = df["close"].shift(-label_horizon) / df["close"] - 1.0
            factors = factors.reset_index(drop=True)
            frames.append(factors)

        if not frames:
            raise ValueError("No stock data satisfied the minimum history requirement.")

        panel = pd.concat(frames, ignore_index=True)
        panel = panel.replace([np.inf, -np.inf], np.nan)
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

        feature_columns = self.infer_feature_columns(panel, label_col=label_col)
        if self.normalize_cross_section and feature_columns:
            panel = panel.astype({col: "float64" for col in feature_columns})
            normalized = (
                panel.groupby("date", group_keys=False)[feature_columns]
                .apply(self._cross_sectional_normalize)
                .reset_index(drop=True)
            )
            panel.loc[:, feature_columns] = normalized

        valid_feature_mask = panel[feature_columns].notna().sum(axis=1) >= max(3, len(feature_columns) // 5)
        panel = panel.loc[valid_feature_mask].copy()
        if drop_na_label:
            panel = panel.dropna(subset=[label_col])

        return panel.reset_index(drop=True)

    def infer_feature_columns(self, panel: pd.DataFrame, label_col: str = "label") -> List[str]:
        """Infer usable feature columns from a factor panel."""
        excluded = set(self.META_COLUMNS) | {label_col}
        return [col for col in panel.columns if col not in excluded]

    def _cross_sectional_normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        lower_q, upper_q = self.winsor_limits
        lower = frame.quantile(lower_q)
        upper = frame.quantile(upper_q)
        clipped = frame.clip(lower=lower, upper=upper, axis=1)
        demeaned = clipped - clipped.mean()
        std = clipped.std(ddof=0).replace(0, np.nan)
        return demeaned.div(std)

    def _validate_price_frame(self, df: pd.DataFrame, symbol: str) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{symbol} is missing required columns: {missing}")


class GradientBoostingFactorModel:
    """Train a LightGBM or XGBoost regressor on factor data."""

    SUPPORTED_BACKENDS = ("lightgbm", "xgboost")

    def __init__(
        self,
        backend: str = "lightgbm",
        model_params: Optional[Dict[str, float]] = None,
        random_state: int = 42,
    ) -> None:
        backend = backend.lower()
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend = backend
        self.model_params = model_params or {}
        self.random_state = random_state

        self.model = None
        self.feature_columns: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.clip_lower: pd.Series = pd.Series(dtype=float)
        self.clip_upper: pd.Series = pd.Series(dtype=float)
        self.report: Optional[FactorModelReport] = None

    def fit(
        self,
        panel: pd.DataFrame,
        label_col: str = "label",
        feature_columns: Optional[Sequence[str]] = None,
        train_end: Optional[str] = None,
        valid_start: Optional[str] = None,
        valid_end: Optional[str] = None,
        train_ratio: float = 0.8,
    ) -> "GradientBoostingFactorModel":
        """Fit the model with a time-ordered train/validation split."""
        fit_panel = panel.copy()
        fit_panel["date"] = pd.to_datetime(fit_panel["date"])
        fit_panel = fit_panel.dropna(subset=[label_col]).sort_values(["date", "symbol"]).reset_index(drop=True)
        if fit_panel.empty:
            raise ValueError("No labeled rows are available for training.")

        self.feature_columns = list(feature_columns or self._infer_feature_columns(fit_panel, label_col))
        train_frame, valid_frame = self._time_split(
            fit_panel,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            train_ratio=train_ratio,
        )

        if train_frame.empty:
            raise ValueError("Training split is empty. Adjust the split dates or train_ratio.")

        self._fit_preprocessor(train_frame[self.feature_columns])
        x_train = self._prepare_features(train_frame)
        y_train = train_frame[label_col].astype(float)

        self.model = self._build_model()
        fit_kwargs = {}
        if not valid_frame.empty:
            x_valid = self._prepare_features(valid_frame)
            y_valid = valid_frame[label_col].astype(float)
            if self.backend == "lightgbm" and lgb is not None:
                fit_kwargs["eval_set"] = [(x_valid, y_valid)]
                fit_kwargs["eval_metric"] = "l2"
                fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]
            elif self.backend == "xgboost":
                fit_kwargs["eval_set"] = [(x_valid, y_valid)]
                fit_kwargs["verbose"] = False

        try:
            self.model.fit(x_train, y_train, **fit_kwargs)
        except TypeError:
            # Keeps compatibility with slightly older wrapper APIs.
            self.model.fit(x_train, y_train)

        train_scored = self.predict(train_frame)
        valid_scored = self.predict(valid_frame) if not valid_frame.empty else valid_frame.copy()

        train_ic = _rank_ic_by_date(train_scored, label_col=label_col, score_col="score")
        valid_ic = _rank_ic_by_date(valid_scored, label_col=label_col, score_col="score") if not valid_frame.empty else pd.Series(dtype=float)

        self.report = FactorModelReport(
            train_rows=len(train_frame),
            valid_rows=len(valid_frame),
            feature_count=len(self.feature_columns),
            train_rmse=_safe_rmse(train_frame[label_col], train_scored["score"]),
            valid_rmse=_safe_rmse(valid_frame[label_col], valid_scored["score"]) if not valid_frame.empty else float("nan"),
            train_rank_ic=float(train_ic.mean(skipna=True)),
            valid_rank_ic=float(valid_ic.mean(skipna=True)) if not valid_frame.empty else float("nan"),
            valid_directional_accuracy=(
                _mean_directional_accuracy(valid_frame[label_col], valid_scored["score"])
                if not valid_frame.empty
                else float("nan")
            ),
        )

        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Score a factor panel and return a copy with a `score` column."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        scored = panel.copy()
        if scored.empty:
            scored["score"] = []
            return scored

        features = self._prepare_features(scored)
        scored["score"] = self.model.predict(features)
        return scored

    def fit_predict(self, panel: pd.DataFrame, **fit_kwargs) -> pd.DataFrame:
        """Fit the model and score the full panel in one call."""
        self.fit(panel, **fit_kwargs)
        return self.predict(panel)

    def feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Return sorted feature importances."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            raise ValueError("Current model backend does not expose feature_importances_.")

        frame = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)
        if top_n is not None:
            return frame.head(top_n).reset_index(drop=True)
        return frame

    def select_top_stocks(
        self,
        scored_panel: pd.DataFrame,
        as_of_date: Optional[str] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Select the top-N stocks by predicted score on a given date."""
        if scored_panel.empty:
            return scored_panel.copy()

        frame = scored_panel.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        target_date = pd.Timestamp(as_of_date) if as_of_date else frame["date"].max()
        latest = frame.loc[frame["date"] == target_date].sort_values("score", ascending=False).head(top_n)
        return latest.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Persist model weights and preprocessing state."""
        payload = {
            "backend": self.backend,
            "model_params": self.model_params,
            "random_state": self.random_state,
            "feature_columns": self.feature_columns,
            "feature_medians": self.feature_medians,
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "report": self.report,
            "model": self.model,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "GradientBoostingFactorModel":
        """Restore a saved factor model from disk."""
        payload = joblib.load(path)
        instance = cls(
            backend=payload["backend"],
            model_params=payload["model_params"],
            random_state=payload["random_state"],
        )
        instance.feature_columns = payload["feature_columns"]
        instance.feature_medians = payload["feature_medians"]
        instance.clip_lower = payload["clip_lower"]
        instance.clip_upper = payload["clip_upper"]
        instance.report = payload["report"]
        instance.model = payload["model"]
        return instance

    def _infer_feature_columns(self, panel: pd.DataFrame, label_col: str) -> List[str]:
        excluded = {"date", "symbol", label_col}
        return [col for col in panel.columns if col not in excluded]

    def _fit_preprocessor(self, frame: pd.DataFrame) -> None:
        self.feature_medians = frame.median().to_dict()
        self.clip_lower = frame.quantile(0.01)
        self.clip_upper = frame.quantile(0.99)

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.loc[:, self.feature_columns].copy()
        x = x.fillna(self.feature_medians)
        if not self.clip_lower.empty and not self.clip_upper.empty:
            x = x.clip(lower=self.clip_lower, upper=self.clip_upper, axis=1)
        return x

    def _build_model(self):
        if self.backend == "lightgbm":
            if lgb is None:
                raise ImportError("LightGBM is not installed. Run `pip install lightgbm`.")
            params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_samples": 20,
                "random_state": self.random_state,
                "objective": "regression",
                "verbosity": -1,
            }
            params.update(self.model_params)
            return lgb.LGBMRegressor(**params)

        if xgb is None:
            raise ImportError("XGBoost is not installed. Run `pip install xgboost`.")
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": self.random_state,
            "objective": "reg:squarederror",
            "tree_method": "hist",
        }
        params.update(self.model_params)
        return xgb.XGBRegressor(**params)

    def _time_split(
        self,
        panel: pd.DataFrame,
        train_end: Optional[str],
        valid_start: Optional[str],
        valid_end: Optional[str],
        train_ratio: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        frame = panel.copy()
        unique_dates = np.sort(frame["date"].dropna().unique())
        if len(unique_dates) < 2:
            return frame, frame.iloc[0:0].copy()

        if train_end is not None:
            train_cutoff = pd.Timestamp(train_end)
        else:
            split_index = min(max(int(len(unique_dates) * train_ratio), 1), len(unique_dates) - 1)
            train_cutoff = pd.Timestamp(unique_dates[split_index - 1])

        train_frame = frame.loc[frame["date"] <= train_cutoff].copy()
        valid_frame = frame.loc[frame["date"] > train_cutoff].copy()

        if valid_start is not None:
            valid_frame = valid_frame.loc[valid_frame["date"] >= pd.Timestamp(valid_start)]
        if valid_end is not None:
            valid_frame = valid_frame.loc[valid_frame["date"] <= pd.Timestamp(valid_end)]

        return train_frame.reset_index(drop=True), valid_frame.reset_index(drop=True)


class MachineLearningStockSelector:
    """Convenience wrapper for scoring raw stock data with a trained model."""

    def __init__(
        self,
        model: GradientBoostingFactorModel,
        dataset_builder: Optional[PanelFactorDatasetBuilder] = None,
    ) -> None:
        self.model = model
        self.dataset_builder = dataset_builder or PanelFactorDatasetBuilder()

    def score_stock_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        label_horizon: int = 5,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert raw OHLCV data to factors and score the resulting panel."""
        panel = self.dataset_builder.build_dataset(
            stock_data,
            label_horizon=label_horizon,
            drop_na_label=False,
        )
        scored = self.model.predict(panel)
        if as_of_date is None:
            return scored
        target_date = pd.Timestamp(as_of_date)
        return scored.loc[scored["date"] == target_date].reset_index(drop=True)

    def select_stocks(
        self,
        stock_data: Dict[str, pd.DataFrame],
        top_n: int = 20,
        as_of_date: Optional[str] = None,
        label_horizon: int = 5,
    ) -> pd.DataFrame:
        """Return the highest-scoring stocks on the target date."""
        scored = self.score_stock_data(stock_data, label_horizon=label_horizon, as_of_date=as_of_date)
        return self.model.select_top_stocks(scored, as_of_date=as_of_date, top_n=top_n)
