from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.a_share_minute_data import AShareMinuteDataFetcher


class FakeMinuteProvider:
    def __init__(self) -> None:
        self.hist_calls = 0
        self.pre_calls = 0

    def stock_zh_a_hist_min_em(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str,
        adjust: str,
    ) -> pd.DataFrame:
        self.hist_calls += 1
        return pd.DataFrame(
            {
                "时间": [
                    "2026-03-31 09:25:00",
                    "2026-03-31 09:30:00",
                    "2026-03-31 09:31:00",
                    "2026-03-31 11:31:00",
                    "2026-03-31 13:00:00",
                    "2026-03-31 15:01:00",
                ],
                "开盘": [10, 10.1, 10.2, 10.3, 10.25, 10.4],
                "收盘": [10, 10.15, 10.25, 10.28, 10.35, 10.4],
                "最高": [10, 10.2, 10.3, 10.35, 10.4, 10.5],
                "最低": [9.9, 10.0, 10.1, 10.2, 10.2, 10.3],
                "成交量": [100, 120, 110, 130, 140, 150],
                "成交额": [1000, 1200, 1100, 1300, 1400, 1500],
                "均价": [10.0, 10.1, 10.2, 10.25, 10.3, 10.4],
            }
        )

    def stock_zh_a_hist_pre_min_em(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        self.pre_calls += 1
        return pd.DataFrame(
            {
                "时间": [
                    "2026-04-01 09:14:00",
                    "2026-04-01 09:15:00",
                    "2026-04-01 09:20:00",
                    "2026-04-01 09:26:00",
                    "2026-04-01 09:30:00",
                    "2026-04-01 15:01:00",
                ],
                "开盘": [11.0, 11.0, 11.02, 11.05, 11.1, 11.2],
                "收盘": [11.0, 11.0, 11.02, 11.05, 11.1, 11.2],
                "最高": [11.0, 11.0, 11.03, 11.06, 11.12, 11.2],
                "最低": [11.0, 11.0, 11.01, 11.04, 11.08, 11.2],
                "成交量": [0, 0, 500, 1000, 3000, 100],
                "成交额": [0, 0, 5100, 11000, 33000, 1100],
                "最新价": [11.0, 11.0, 11.02, 11.05, 11.1, 11.2],
            }
        )


def test_fetch_historical_minute_normalizes_and_filters_sessions(tmp_path: Path) -> None:
    provider = FakeMinuteProvider()
    fetcher = AShareMinuteDataFetcher(cache_dir=str(tmp_path), provider=provider)

    frame = fetcher.fetch_historical(
        symbol="000001",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="1",
        use_cache=False,
    )

    assert provider.hist_calls == 1
    assert list(frame.index.strftime("%H:%M:%S")) == ["09:30:00", "09:31:00", "13:00:00"]
    assert {"symbol", "open", "high", "low", "close", "volume", "amount", "period"}.issubset(frame.columns)
    assert frame["symbol"].eq("000001").all()


def test_fetch_intraday_can_include_or_exclude_pre_market(tmp_path: Path) -> None:
    provider = FakeMinuteProvider()
    fetcher = AShareMinuteDataFetcher(cache_dir=str(tmp_path), provider=provider)

    full_frame = fetcher.fetch_intraday(
        symbol="000001",
        trade_date="2026-04-01",
        include_pre_market=True,
    )
    regular_frame = fetcher.fetch_intraday(
        symbol="000001",
        trade_date="2026-04-01",
        include_pre_market=False,
    )

    assert provider.pre_calls == 2
    assert list(full_frame.index.strftime("%H:%M:%S")) == ["09:15:00", "09:20:00", "09:30:00"]
    assert list(regular_frame.index.strftime("%H:%M:%S")) == ["09:30:00"]


def test_fetch_historical_uses_cache_when_enabled(tmp_path: Path) -> None:
    provider = FakeMinuteProvider()
    fetcher = AShareMinuteDataFetcher(cache_dir=str(tmp_path), provider=provider)

    first = fetcher.fetch_historical(
        symbol="000001",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="1",
        use_cache=True,
    )
    second = fetcher.fetch_historical(
        symbol="000001",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="1",
        use_cache=True,
    )

    assert provider.hist_calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_resample_and_split_by_trade_date(tmp_path: Path) -> None:
    provider = FakeMinuteProvider()
    fetcher = AShareMinuteDataFetcher(cache_dir=str(tmp_path), provider=provider)
    frame = fetcher.fetch_historical(
        symbol="000001",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="1",
        use_cache=False,
    )

    resampled = fetcher.resample(frame, rule="2min")
    split = fetcher.split_by_trade_date(frame)

    assert not resampled.empty
    assert "000001" == resampled["symbol"].iloc[0]
    assert list(split.keys()) == ["2026-03-31"]
