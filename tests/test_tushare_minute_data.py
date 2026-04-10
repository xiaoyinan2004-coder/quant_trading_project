from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.tushare_minute_data import TushareMinuteDataFetcher


class FakeTushareMinuteProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stk_mins(self, ts_code: str, start_date: str, end_date: str, freq: str) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame(
            {
                "ts_code": [ts_code] * 5,
                "trade_time": [
                    "2026-03-31 09:25:00",
                    "2026-03-31 09:30:00",
                    "2026-03-31 09:45:00",
                    "2026-03-31 13:00:00",
                    "2026-03-31 15:01:00",
                ],
                "open": [10.0, 10.1, 10.2, 10.3, 10.4],
                "high": [10.1, 10.2, 10.3, 10.4, 10.5],
                "low": [9.9, 10.0, 10.1, 10.2, 10.3],
                "close": [10.05, 10.15, 10.25, 10.35, 10.45],
                "vol": [1000, 1200, 1300, 1400, 1500],
                "amount": [10000, 12000, 13000, 14000, 15000],
            }
        )


def test_fetch_historical_normalizes_tushare_minute_frame(tmp_path: Path) -> None:
    provider = FakeTushareMinuteProvider()
    fetcher = TushareMinuteDataFetcher(
        token="fake-token",
        cache_dir=str(tmp_path),
        provider=provider,
    )

    frame = fetcher.fetch_historical(
        symbol="000001",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="15",
        use_cache=False,
    )

    assert provider.calls == 1
    assert list(frame.index.strftime("%H:%M:%S")) == ["09:30:00", "09:45:00", "13:00:00"]
    assert frame["symbol"].eq("000001").all()
    assert frame["period"].eq("15").all()


def test_fetch_historical_uses_cache_when_enabled(tmp_path: Path) -> None:
    provider = FakeTushareMinuteProvider()
    fetcher = TushareMinuteDataFetcher(
        token="fake-token",
        cache_dir=str(tmp_path),
        provider=provider,
    )

    first = fetcher.fetch_historical(
        symbol="600000",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="15",
        use_cache=True,
    )
    second = fetcher.fetch_historical(
        symbol="600000",
        start_datetime="2026-03-31 09:20:00",
        end_datetime="2026-03-31 15:00:00",
        period="15",
        use_cache=True,
    )

    assert provider.calls == 1
    pd.testing.assert_frame_equal(first, second)
