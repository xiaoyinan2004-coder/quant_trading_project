#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A-share minute-level data fetching, caching, and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover - runtime dependency
    ak = None


@dataclass(frozen=True)
class MinuteDataRequest:
    """Normalized request descriptor for cache key generation."""

    symbol: str
    start: str
    end: str
    period: str
    adjust: str = ""
    include_pre_market: bool = False


class AShareMinuteDataFetcher:
    """Fetch, cache, and normalize A-share minute bars from AKShare."""

    SUPPORTED_PERIODS = {"1", "5", "15", "30", "60"}
    HIST_RENAME_MAP = {
        "时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "均价": "avg_price",
        "最新价": "latest_price",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "振幅": "amplitude",
        "换手率": "turnover_rate",
    }

    def __init__(self, cache_dir: Optional[str] = None, provider=None) -> None:
        if provider is None and ak is None:
            raise ImportError("请先安装 akshare: pip install akshare")

        self.provider = provider or ak
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parents[1] / "data" / "cache" / "a_share_minute")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_historical(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str = "1",
        adjust: str = "",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical minute bars for an A-share symbol."""
        self._validate_period(period)
        request = MinuteDataRequest(
            symbol=symbol,
            start=start_datetime,
            end=end_datetime,
            period=period,
            adjust=adjust,
            include_pre_market=False,
        )

        if use_cache and not force_refresh:
            cached = self._load_cache(request)
            if cached is not None:
                return cached

        raw = self.provider.stock_zh_a_hist_min_em(
            symbol=symbol,
            start_date=start_datetime,
            end_date=end_datetime,
            period=period,
            adjust=adjust,
        )
        frame = self._normalize_minute_frame(
            raw,
            symbol=symbol,
            period=period,
            include_pre_market=False,
        )
        frame = self._slice_datetime(frame, start_datetime, end_datetime)

        if use_cache:
            self._save_cache(frame, request)
        return frame

    def fetch_intraday(
        self,
        symbol: str,
        start_time: str = "09:15:00",
        end_time: str = "15:00:00",
        trade_date: Optional[str] = None,
        use_cache: bool = False,
        force_refresh: bool = False,
        include_pre_market: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch current-day intraday minute bars with optional pre-market auction data.

        This uses `stock_zh_a_hist_pre_min_em`, which is best suited for the current
        trading day and includes the opening auction window.
        """
        request = MinuteDataRequest(
            symbol=symbol,
            start=start_time,
            end=end_time,
            period="1",
            adjust="",
            include_pre_market=include_pre_market,
        )

        if use_cache and not force_refresh:
            cached = self._load_cache(request)
            if cached is not None:
                cached = self._filter_trade_date(cached, trade_date)
                return self._filter_sessions(cached, include_pre_market=include_pre_market)

        raw = self.provider.stock_zh_a_hist_pre_min_em(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
        )
        frame = self._normalize_minute_frame(
            raw,
            symbol=symbol,
            period="1",
            include_pre_market=True,
        )
        frame = self._filter_trade_date(frame, trade_date)
        frame = self._filter_sessions(frame, include_pre_market=include_pre_market)

        if use_cache:
            self._save_cache(frame, request)
        return frame

    def fetch_batch_historical(
        self,
        symbols: Iterable[str],
        start_datetime: str,
        end_datetime: str,
        period: str = "1",
        adjust: str = "",
        use_cache: bool = True,
        force_refresh: bool = False,
        skip_failed: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch minute bars for multiple A-share symbols."""
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch_historical(
                    symbol=symbol,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    period=period,
                    adjust=adjust,
                    use_cache=use_cache,
                    force_refresh=force_refresh,
                )
            except Exception:
                if not skip_failed:
                    raise
        return result

    def resample(self, frame: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
        """Resample minute data to a higher frequency such as 5min or 15min."""
        if frame.empty:
            return frame.copy()

        base = frame.sort_index()
        aggregated = (
            base.resample(rule, label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "amount": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        aggregated["symbol"] = base["symbol"].iloc[0]
        aggregated["date"] = aggregated.index.date.astype(str)
        aggregated["time"] = aggregated.index.strftime("%H:%M:%S")
        aggregated["period"] = rule
        return aggregated

    def split_by_trade_date(self, frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split a minute DataFrame into per-trading-day DataFrames."""
        if frame.empty:
            return {}

        grouped: Dict[str, pd.DataFrame] = {}
        for trade_date, group in frame.groupby(frame.index.date):
            grouped[str(trade_date)] = group.copy()
        return grouped

    def _normalize_minute_frame(
        self,
        raw: pd.DataFrame,
        symbol: str,
        period: str,
        include_pre_market: bool,
    ) -> pd.DataFrame:
        if raw is None or raw.empty:
            raise ValueError(
                f"未获取到分钟级数据: {symbol}. "
                "如果是 1 分钟历史数据，请检查日期范围或数据源限制。"
            )

        frame = raw.rename(columns=self.HIST_RENAME_MAP).copy()
        if "datetime" not in frame.columns:
            raise ValueError(f"{symbol} 分钟数据缺少时间列: {list(raw.columns)}")

        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index("datetime").sort_index()

        numeric_columns = [
            column
            for column in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "avg_price",
                "latest_price",
                "pct_change",
                "change",
                "amplitude",
                "turnover_rate",
            ]
            if column in frame.columns
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        for required_column in ("open", "high", "low", "close"):
            if required_column not in frame.columns:
                raise ValueError(f"{symbol} 分钟数据缺少字段: {required_column}")

        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        if "amount" not in frame.columns:
            frame["amount"] = 0.0

        frame["symbol"] = str(symbol)
        frame["date"] = frame.index.strftime("%Y-%m-%d")
        frame["time"] = frame.index.strftime("%H:%M:%S")
        frame["period"] = str(period)
        frame = self._filter_sessions(frame, include_pre_market=include_pre_market)

        ordered_columns = [
            "symbol",
            "date",
            "time",
            "period",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "avg_price",
            "latest_price",
            "pct_change",
            "change",
            "amplitude",
            "turnover_rate",
        ]
        existing_columns = [column for column in ordered_columns if column in frame.columns]
        return frame.loc[:, existing_columns]

    def _filter_trade_date(self, frame: pd.DataFrame, trade_date: Optional[str]) -> pd.DataFrame:
        if not trade_date or frame.empty:
            return frame
        target = pd.Timestamp(trade_date).date()
        return frame.loc[frame.index.date == target].copy()

    def _filter_sessions(self, frame: pd.DataFrame, include_pre_market: bool) -> pd.DataFrame:
        if frame.empty:
            return frame

        time_values = frame.index.time
        morning_session = [(hour > 9 or (hour == 9 and minute >= 30)) and (hour < 11 or (hour == 11 and minute <= 30))
                           for hour, minute, _ in [(t.hour, t.minute, t.second) for t in time_values]]
        afternoon_session = [(hour > 13 or (hour == 13 and minute >= 0)) and (hour < 15 or (hour == 15 and minute <= 0))
                             for hour, minute, _ in [(t.hour, t.minute, t.second) for t in time_values]]
        regular_mask = pd.Series(morning_session, index=frame.index) | pd.Series(afternoon_session, index=frame.index)

        if include_pre_market:
            pre_market_mask = pd.Series(
                [
                    (t.hour > 9 or (t.hour == 9 and t.minute >= 15))
                    and (t.hour < 9 or (t.hour == 9 and t.minute <= 25))
                    for t in time_values
                ],
                index=frame.index,
            )
            return frame.loc[regular_mask | pre_market_mask].copy()
        return frame.loc[regular_mask].copy()

    def _slice_datetime(self, frame: pd.DataFrame, start_datetime: str, end_datetime: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_datetime)
        end_ts = pd.Timestamp(end_datetime)
        return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()

    def _cache_file(self, request: MinuteDataRequest) -> Path:
        scope = "intraday" if request.include_pre_market else "historical"
        filename = "__".join(
            [
                request.symbol,
                scope,
                request.period,
                request.adjust or "none",
                request.start.replace(":", "").replace(" ", "_").replace("-", ""),
                request.end.replace(":", "").replace(" ", "_").replace("-", ""),
            ]
        )
        return self.cache_dir / f"{filename}.csv"

    def _load_cache(self, request: MinuteDataRequest) -> Optional[pd.DataFrame]:
        path = self._cache_file(request)
        if not path.exists():
            return None
        frame = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
            dtype={"symbol": str, "date": str, "time": str, "period": str},
        )
        frame.index.name = "datetime"
        return frame

    def _save_cache(self, frame: pd.DataFrame, request: MinuteDataRequest) -> None:
        path = self._cache_file(request)
        frame.to_csv(path)

    def _validate_period(self, period: str) -> None:
        if period not in self.SUPPORTED_PERIODS:
            raise ValueError(f"不支持的分钟周期: {period}, 可选值: {sorted(self.SUPPORTED_PERIODS)}")


def fetch_a_share_minute_data(
    symbol: str,
    start_datetime: str,
    end_datetime: str,
    period: str = "1",
    adjust: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convenience function for fetching historical minute bars."""
    fetcher = AShareMinuteDataFetcher()
    return fetcher.fetch_historical(
        symbol=symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        period=period,
        adjust=adjust,
        use_cache=use_cache,
    )
