"""Sequence features and lightweight sequence expert for M3-Net Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    changed = series.pct_change(periods)
    return changed.replace([np.inf, -np.inf], np.nan)


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or not np.isfinite(denominator) or abs(float(denominator)) < 1e-8:
        return float(default)
    return float(numerator) / float(denominator)


class SequenceFeatureBuilder:
    """Build lightweight time-series features from daily and minute bars."""

    DAILY_FEATURES = [
        "daily_return_1d",
        "daily_return_5d",
        "daily_return_10d",
        "daily_return_20d",
        "daily_volatility_5d",
        "daily_volatility_20d",
        "daily_volume_ratio_20d",
        "daily_trend_strength_10d",
        "daily_close_to_20d_high",
        "daily_close_to_20d_low",
    ]
    MINUTE_FEATURES = [
        "intraday_return",
        "intraday_range",
        "intraday_volatility",
        "realized_volatility",
        "intraday_volume_ratio",
        "volume_burst_ratio",
        "close_to_vwap",
        "morning_return",
        "afternoon_return",
        "open_30m_return",
        "close_30m_return",
        "morning_range",
        "afternoon_range",
        "intraday_trend_strength",
        "midday_reversal",
        "close_position_in_range",
    ]

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    @property
    def feature_columns(self) -> List[str]:
        return self.DAILY_FEATURES + self.MINUTE_FEATURES + ["has_minute_features"]

    def build_panel(
        self,
        stock_data: Dict[str, pd.DataFrame],
        minute_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        minute_data = minute_data or {}

        for symbol, daily_df in stock_data.items():
            if daily_df is None or daily_df.empty:
                continue

            daily_panel = self._build_daily_features(symbol, daily_df)
            minute_panel = self._build_minute_features(symbol, minute_data.get(symbol))
            if not minute_panel.empty:
                daily_panel = daily_panel.merge(minute_panel, on=["date", "symbol"], how="left")
            else:
                for column in self.MINUTE_FEATURES:
                    daily_panel[column] = np.nan
                daily_panel["has_minute_features"] = 0.0

            daily_panel["has_minute_features"] = daily_panel["has_minute_features"].fillna(0.0)
            frames.append(daily_panel)

        if not frames:
            return pd.DataFrame(columns=["date", "symbol", *self.feature_columns])

        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
        return panel

    def _build_daily_features(self, symbol: str, daily_df: pd.DataFrame) -> pd.DataFrame:
        frame = daily_df.copy().sort_index()
        close = frame["close"].astype(float)
        volume = frame["volume"].astype(float)

        result = pd.DataFrame(index=frame.index)
        result["daily_return_1d"] = _safe_pct_change(close, 1)
        result["daily_return_5d"] = _safe_pct_change(close, 5)
        result["daily_return_10d"] = _safe_pct_change(close, 10)
        result["daily_return_20d"] = _safe_pct_change(close, 20)
        result["daily_volatility_5d"] = close.pct_change().rolling(5).std()
        result["daily_volatility_20d"] = close.pct_change().rolling(20).std()
        result["daily_volume_ratio_20d"] = volume / volume.rolling(20).mean()
        ma10 = close.rolling(10).mean()
        rolling_high_20d = frame["high"].astype(float).rolling(20).max()
        rolling_low_20d = frame["low"].astype(float).rolling(20).min()
        result["daily_trend_strength_10d"] = close / ma10 - 1.0
        result["daily_close_to_20d_high"] = close / rolling_high_20d - 1.0
        result["daily_close_to_20d_low"] = close / rolling_low_20d - 1.0
        result["date"] = pd.to_datetime(result.index)
        result["symbol"] = symbol
        return result.reset_index(drop=True)

    def _build_minute_features(self, symbol: str, minute_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if minute_df is None or minute_df.empty:
            return pd.DataFrame(columns=["date", "symbol", *self.MINUTE_FEATURES, "has_minute_features"])

        frame = minute_df.copy().sort_index()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        grouped_rows = []
        daily_volume_totals = frame.groupby(frame.index.normalize())["volume"].sum().astype(float)
        volume_baseline = daily_volume_totals.rolling(self.lookback, min_periods=1).median().shift(1)

        for trade_date, group in frame.groupby(frame.index.normalize()):
            group = group.sort_index()
            open_price = float(group["open"].iloc[0])
            close_price = float(group["close"].iloc[-1])
            high_price = float(group["high"].max())
            low_price = float(group["low"].min())
            returns = group["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            typical_price = (group["high"] + group["low"] + group["close"]) / 3.0
            total_volume = float(group["volume"].sum())
            vwap = float((typical_price * group["volume"]).sum() / max(total_volume, 1.0))
            morning = group.between_time("09:30", "11:30")
            afternoon = group.between_time("13:00", "15:00")
            first_30m = group.between_time("09:30", "10:00")
            last_30m = group.between_time("14:30", "15:00")
            baseline = volume_baseline.get(trade_date, np.nan)
            if pd.isna(baseline) or baseline <= 0:
                intraday_volume_ratio = 1.0
            else:
                intraday_volume_ratio = total_volume / float(baseline)

            morning_return = (
                _safe_ratio(float(morning["close"].iloc[-1]), float(morning["open"].iloc[0]), default=1.0) - 1.0
                if not morning.empty
                else 0.0
            )
            afternoon_return = (
                _safe_ratio(float(afternoon["close"].iloc[-1]), float(afternoon["open"].iloc[0]), default=1.0) - 1.0
                if not afternoon.empty
                else 0.0
            )
            open_30m_return = (
                _safe_ratio(float(first_30m["close"].iloc[-1]), float(first_30m["open"].iloc[0]), default=1.0) - 1.0
                if not first_30m.empty
                else 0.0
            )
            close_30m_return = (
                _safe_ratio(float(last_30m["close"].iloc[-1]), float(last_30m["open"].iloc[0]), default=1.0) - 1.0
                if not last_30m.empty
                else 0.0
            )
            morning_range = (
                _safe_ratio(float(morning["high"].max()), float(morning["low"].min()), default=1.0) - 1.0
                if not morning.empty
                else 0.0
            )
            afternoon_range = (
                _safe_ratio(float(afternoon["high"].max()), float(afternoon["low"].min()), default=1.0) - 1.0
                if not afternoon.empty
                else 0.0
            )
            volume_burst_ratio = (
                float(group["volume"].max() / max(group["volume"].median(), 1.0))
                if not group["volume"].empty
                else 1.0
            )
            intraday_trend_strength = (
                float((close_price - open_price) / max(high_price - low_price, 1e-8))
                if high_price > low_price
                else 0.0
            )
            midday_reversal = afternoon_return - morning_return
            close_position_in_range = (
                float((close_price - low_price) / max(high_price - low_price, 1e-8))
                if high_price > low_price
                else 0.5
            )
            realized_volatility = float(np.sqrt(np.square(returns).sum())) if not returns.empty else 0.0

            grouped_rows.append(
                {
                    "date": pd.Timestamp(trade_date),
                    "symbol": symbol,
                    "intraday_return": _safe_ratio(close_price, open_price, default=1.0) - 1.0,
                    "intraday_range": _safe_ratio(high_price, low_price, default=1.0) - 1.0,
                    "intraday_volatility": float(returns.std()) if not returns.empty else 0.0,
                    "realized_volatility": realized_volatility,
                    "intraday_volume_ratio": intraday_volume_ratio,
                    "volume_burst_ratio": volume_burst_ratio,
                    "close_to_vwap": _safe_ratio(close_price, vwap, default=1.0) - 1.0,
                    "morning_return": morning_return,
                    "afternoon_return": afternoon_return,
                    "open_30m_return": open_30m_return,
                    "close_30m_return": close_30m_return,
                    "morning_range": morning_range,
                    "afternoon_range": afternoon_range,
                    "intraday_trend_strength": intraday_trend_strength,
                    "midday_reversal": midday_reversal,
                    "close_position_in_range": close_position_in_range,
                    "has_minute_features": 1.0,
                }
            )

        return pd.DataFrame(grouped_rows)


@dataclass
class SequenceExpertReport:
    """Summary metrics for the sequence expert."""

    train_rows: int
    valid_rows: int
    feature_count: int
    train_rmse: float
    valid_rmse: float


class SequenceExpertModel:
    """Lightweight sequence expert that runs on CPU-only commodity hardware."""

    def __init__(self, model_params: Optional[Dict[str, float]] = None, random_state: int = 42) -> None:
        self.model_params = model_params or {}
        self.random_state = random_state
        self.feature_columns: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.report: Optional[SequenceExpertReport] = None

    def fit(
        self,
        panel: pd.DataFrame,
        label_col: str = "label",
        feature_columns: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ) -> "SequenceExpertModel":
        frame = panel.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.dropna(subset=[label_col]).sort_values(["date", "symbol"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError("No labeled rows available for sequence training.")

        self.feature_columns = feature_columns or [
            col for col in frame.columns if col not in {"date", "symbol", label_col}
        ]
        self.feature_medians = frame[self.feature_columns].median().to_dict()

        unique_dates = np.sort(frame["date"].unique())
        split_index = min(max(int(len(unique_dates) * train_ratio), 1), max(len(unique_dates) - 1, 1))
        cutoff = pd.Timestamp(unique_dates[split_index - 1])
        train_frame = frame.loc[frame["date"] <= cutoff].copy()
        valid_frame = frame.loc[frame["date"] > cutoff].copy()

        self.model = HistGradientBoostingRegressor(
            random_state=self.random_state,
            **self.model_params,
        )
        self.model.fit(self._prepare_features(train_frame), train_frame[label_col].astype(float))

        train_pred = self.predict(train_frame)
        valid_pred = self.predict(valid_frame) if not valid_frame.empty else valid_frame.assign(score=[])
        self.report = SequenceExpertReport(
            train_rows=len(train_frame),
            valid_rows=len(valid_frame),
            feature_count=len(self.feature_columns),
            train_rmse=_rmse(train_frame[label_col], train_pred["score"]),
            valid_rmse=_rmse(valid_frame[label_col], valid_pred["score"]) if not valid_frame.empty else float("nan"),
        )
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Sequence expert has not been fitted.")

        scored = panel.copy()
        if scored.empty:
            scored["score"] = []
            return scored

        scored["score"] = self.predict_scores(scored)
        return scored

    def predict_scores(self, panel: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Sequence expert has not been fitted.")
        if panel.empty:
            return np.array([], dtype=float)
        return np.asarray(self.model.predict(self._prepare_features(panel)), dtype=float)

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.loc[:, self.feature_columns].copy()
        return features.fillna(self.feature_medians)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model_params": self.model_params,
                "random_state": self.random_state,
                "feature_columns": self.feature_columns,
                "feature_medians": self.feature_medians,
                "report": self.report,
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SequenceExpertModel":
        payload = joblib.load(path)
        instance = cls(payload["model_params"], random_state=payload["random_state"])
        instance.feature_columns = payload["feature_columns"]
        instance.feature_medians = payload["feature_medians"]
        instance.report = payload["report"]
        instance.model = payload["model"]
        return instance


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    diff = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else float("nan")
