"""Dataset builder for the full M3Net architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from models.gradient_boosting_factor import PanelFactorDatasetBuilder
from models.m3net.sequence import SequenceFeatureBuilder

from .config import M3NetFullConfig

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


DAILY_SEQUENCE_COLUMNS = ["open", "high", "low", "close", "volume", "return_1d", "range", "close_to_ma20"]
INTRADAY_SEQUENCE_COLUMNS = SequenceFeatureBuilder.MINUTE_FEATURES + ["has_minute_features"]
MEMORY_COLUMNS = [
    "market_return_mean_20d",
    "market_return_std_20d",
    "market_volume_ratio_20d",
    "market_breadth_20d",
    "market_return_mean_60d",
    "market_return_std_60d",
    "market_volume_ratio_60d",
    "market_breadth_60d",
]


@dataclass
class M3NetFullSample:
    date: pd.Timestamp
    symbols: list[str]
    daily_sequence: "torch.Tensor"
    intraday_sequence: "torch.Tensor"
    factor_features: "torch.Tensor"
    memory_features: "torch.Tensor"
    future_return: "torch.Tensor"
    future_risk: "torch.Tensor"


class M3NetFullDataset(Dataset):
    """Cross-sectional dataset where each item is one rebalance date."""

    def __init__(self, samples: list[M3NetFullSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> M3NetFullSample:
        return self.samples[index]


class M3NetFullDatasetBuilder:
    """Build cross-sectional training samples for the full M3Net model."""

    def __init__(self, config: Optional[M3NetFullConfig] = None) -> None:
        self.config = config or M3NetFullConfig()
        self.factor_builder = PanelFactorDatasetBuilder(normalize_cross_section=True)
        self.sequence_builder = SequenceFeatureBuilder(lookback=self.config.minute_lookback_days)

    def build_samples(
        self,
        stock_data: dict[str, pd.DataFrame],
        minute_data: Optional[dict[str, pd.DataFrame]] = None,
        rebalance_dates: Optional[Iterable[pd.Timestamp]] = None,
    ) -> list[M3NetFullSample]:
        if torch is None:
            raise ImportError("PyTorch is required to build M3Net full datasets.")

        factor_panel = self.factor_builder.build_dataset(
            stock_data,
            label_horizon=self.config.label_horizon,
            min_history=self.config.min_history,
            label_col="future_return",
        )
        sequence_panel = self.sequence_builder.build_panel(stock_data, minute_data=minute_data)
        merged = factor_panel.merge(sequence_panel, on=["date", "symbol"], how="left")
        merged["date"] = pd.to_datetime(merged["date"])

        factor_columns = self.factor_builder.infer_feature_columns(factor_panel, label_col="future_return")
        factor_columns = factor_columns[: self.config.factor_input_dim]
        minute_columns = [column for column in INTRADAY_SEQUENCE_COLUMNS if column in merged.columns]
        symbol_panels = {
            symbol: panel.sort_values("date").reset_index(drop=True)
            for symbol, panel in merged.groupby("symbol", sort=False)
        }
        available_dates = sorted(pd.to_datetime(merged["date"]).unique())
        target_dates = list(pd.to_datetime(list(rebalance_dates))) if rebalance_dates is not None else available_dates

        samples: list[M3NetFullSample] = []
        for trade_date in target_dates:
            sample = self._build_single_sample(
                trade_date=pd.Timestamp(trade_date),
                merged_panel=merged,
                symbol_panels=symbol_panels,
                stock_data=stock_data,
                factor_columns=factor_columns,
                minute_columns=minute_columns,
            )
            if sample is not None:
                samples.append(sample)

        self.config.factor_columns = factor_columns
        self.config.minute_summary_columns = minute_columns
        self.config.factor_input_dim = len(factor_columns)
        self.config.intraday_input_dim = len(minute_columns)
        return samples

    def _build_single_sample(
        self,
        trade_date: pd.Timestamp,
        merged_panel: pd.DataFrame,
        symbol_panels: dict[str, pd.DataFrame],
        stock_data: dict[str, pd.DataFrame],
        factor_columns: list[str],
        minute_columns: list[str],
    ) -> Optional[M3NetFullSample]:
        rows = merged_panel.loc[merged_panel["date"] == trade_date].copy()
        if rows.empty:
            return None

        daily_tensors: list["torch.Tensor"] = []
        intraday_tensors: list["torch.Tensor"] = []
        factor_tensors: list["torch.Tensor"] = []
        labels: list[float] = []
        risks: list[float] = []
        symbols: list[str] = []

        for _, row in rows.iterrows():
            symbol = str(row["symbol"])
            history = stock_data.get(symbol)
            if history is None or trade_date not in history.index:
                continue
            history = history.loc[history.index <= trade_date]
            if len(history) < self.config.daily_lookback:
                continue

            daily_tensor = self._build_daily_sequence(history.tail(self.config.daily_lookback))
            factor_tensor = self._to_tensor(pd.to_numeric(row[factor_columns], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
            intraday_tensor = self._build_intraday_tensor(
                symbol_panel=symbol_panels.get(symbol, pd.DataFrame()),
                trade_date=trade_date,
                minute_columns=minute_columns,
            )
            future_return = float(row["future_return"])
            future_risk = float(pd.to_numeric(history["close"], errors="coerce").pct_change().tail(20).std())

            if not np.isfinite(future_return):
                continue
            if not np.isfinite(future_risk):
                future_risk = 0.0

            daily_tensors.append(daily_tensor)
            intraday_tensors.append(intraday_tensor)
            factor_tensors.append(factor_tensor)
            labels.append(future_return)
            risks.append(future_risk)
            symbols.append(symbol)

        if len(symbols) < max(5, self.config.top_n):
            return None

        memory_features = self._build_memory_features(stock_data, trade_date)
        return M3NetFullSample(
            date=trade_date,
            symbols=symbols,
            daily_sequence=torch.stack(daily_tensors, dim=0),
            intraday_sequence=torch.stack(intraday_tensors, dim=0),
            factor_features=torch.stack(factor_tensors, dim=0),
            memory_features=memory_features,
            future_return=self._to_tensor(np.asarray(labels, dtype=np.float32)),
            future_risk=self._to_tensor(np.asarray(risks, dtype=np.float32)),
        )

    def _build_daily_sequence(self, frame: pd.DataFrame) -> "torch.Tensor":
        seq = frame.copy().astype(float)
        close = seq["close"].replace(0.0, np.nan)
        seq["return_1d"] = close.pct_change().fillna(0.0)
        seq["range"] = (seq["high"] / seq["low"].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        seq["close_to_ma20"] = (close / close.rolling(20).mean() - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        seq = seq[DAILY_SEQUENCE_COLUMNS].fillna(0.0)
        values = np.nan_to_num(seq.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.maximum(np.nanstd(values, axis=0, keepdims=True), 1e-6)
        centered = values - np.nanmean(values, axis=0, keepdims=True)
        return self._to_tensor(np.nan_to_num(centered / scale, nan=0.0, posinf=0.0, neginf=0.0))

    def _build_intraday_tensor(
        self,
        symbol_panel: pd.DataFrame,
        trade_date: pd.Timestamp,
        minute_columns: list[str],
    ) -> "torch.Tensor":
        if not minute_columns:
            return self._to_tensor(np.zeros((self.config.minute_lookback_days, 1), dtype=np.float32))

        if symbol_panel.empty:
            return self._to_tensor(np.zeros((self.config.minute_lookback_days, len(minute_columns)), dtype=np.float32))

        history = symbol_panel.loc[pd.to_datetime(symbol_panel["date"]) <= trade_date].copy()
        history = history.sort_values("date").tail(self.config.minute_lookback_days)
        frame = history.reindex(columns=minute_columns)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        values = frame.to_numpy(dtype=np.float32, copy=True)
        if len(values) < self.config.minute_lookback_days:
            pad_rows = self.config.minute_lookback_days - len(values)
            padding = np.zeros((pad_rows, len(minute_columns)), dtype=np.float32)
            values = np.vstack([padding, values])

        if "has_minute_features" in minute_columns and values.size:
            has_idx = minute_columns.index("has_minute_features")
            cont_idx = [idx for idx in range(len(minute_columns)) if idx != has_idx]
            if cont_idx:
                cont_values = values[:, cont_idx]
                cont_scale = np.maximum(np.nanstd(cont_values, axis=0, keepdims=True), 1e-6)
                cont_centered = cont_values - np.nanmean(cont_values, axis=0, keepdims=True)
                values[:, cont_idx] = np.nan_to_num(cont_centered / cont_scale, nan=0.0, posinf=0.0, neginf=0.0)
        return self._to_tensor(values)

    def _build_memory_features(
        self,
        stock_data: dict[str, pd.DataFrame],
        trade_date: pd.Timestamp,
    ) -> "torch.Tensor":
        market_returns = []
        market_volumes = []
        for frame in stock_data.values():
            history = frame.loc[frame.index <= trade_date]
            if len(history) < 60:
                continue
            returns = history["close"].pct_change().astype(float)
            volume_ratio = history["volume"].astype(float) / history["volume"].astype(float).rolling(20).mean()
            market_returns.append(returns)
            market_volumes.append(volume_ratio)

        if not market_returns:
            return self._to_tensor(np.zeros(len(MEMORY_COLUMNS), dtype=np.float32))

        returns_frame = pd.concat(market_returns, axis=1)
        volume_frame = pd.concat(market_volumes, axis=1)
        recent_20 = returns_frame.tail(20)
        recent_60 = returns_frame.tail(60)
        features = np.asarray(
            [
                recent_20.mean().mean(),
                recent_20.std().mean(),
                volume_frame.tail(20).mean().mean(),
                (recent_20 > 0).mean().mean(),
                recent_60.mean().mean(),
                recent_60.std().mean(),
                volume_frame.tail(60).mean().mean(),
                (recent_60 > 0).mean().mean(),
            ],
            dtype=np.float32,
        )
        return self._to_tensor(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))

    def _to_tensor(self, array: np.ndarray) -> "torch.Tensor":
        cleaned = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(cleaned)
