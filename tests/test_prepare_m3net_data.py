from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.prepare_m3net_data import (
    _ensure_columns,
    _fetch_daily_range_baostock,
    _fetch_minute_range_baostock_logged_in,
    _fetch_minute_range,
    _is_a_share_baostock_code,
    _load_symbol_list,
    _normalize_code,
    _write_daily_csv,
    _write_minute_csv,
)


class FakeFetcher:
    def get_stock_list(self) -> pd.DataFrame:
        return pd.DataFrame({"code": ["1", "000002", "600000"]})


class FakeMinuteFetcher:
    def __init__(self) -> None:
        self.called = None

    def fetch_akshare_a_stock_minute(self, **kwargs) -> pd.DataFrame:
        self.called = ("akshare", kwargs)
        return pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}, index=pd.to_datetime(["2026-01-01 09:30:00"]))

    def fetch_tushare_a_stock_minute(self, **kwargs) -> pd.DataFrame:
        self.called = ("tushare", kwargs)
        return pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}, index=pd.to_datetime(["2026-01-01 09:30:00"]))


def test_normalize_code_zero_pads_symbols() -> None:
    assert _normalize_code("1") == "000001"
    assert _normalize_code("600000") == "600000"


def test_load_symbol_list_uses_explicit_symbols_and_deduplicates(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("000001\n600000\n", encoding="utf-8")

    symbols = _load_symbol_list(["1", "000001", "2"], str(symbols_file), None, None, FakeFetcher(), "akshare", None)

    assert symbols == ["000001", "000002", "600000"]


def test_load_symbol_list_falls_back_to_stock_list_when_missing_inputs() -> None:
    symbols = _load_symbol_list(None, None, None, 2, FakeFetcher(), "akshare", None)
    assert symbols == ["000001", "000002"]


def test_load_symbol_list_can_reuse_existing_daily_directory(tmp_path: Path) -> None:
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    (daily_dir / "600000.csv").write_text("date,open\n2026-01-01,10\n", encoding="utf-8")
    (daily_dir / "000001.csv").write_text("date,open\n2026-01-01,11\n", encoding="utf-8")

    symbols = _load_symbol_list(None, None, str(daily_dir), None, FakeFetcher(), "akshare", None)

    assert symbols == ["000001", "600000"]


def test_ensure_columns_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame({"open": [1], "close": [1]})
    try:
        _ensure_columns(frame, ["open", "high"], "000001", "daily")
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("Expected missing column validation to fail.")


def test_write_daily_and_minute_csv_outputs_expected_format(tmp_path: Path) -> None:
    index = pd.to_datetime(["2026-04-01 09:30:00", "2026-04-01 09:45:00"])
    daily = pd.DataFrame(
        {
            "open": [10.0, 10.2],
            "high": [10.3, 10.4],
            "low": [9.9, 10.1],
            "close": [10.1, 10.35],
            "volume": [1000, 1200],
        },
        index=pd.to_datetime(["2026-04-01", "2026-04-02"]),
    )
    minute = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": ["2026-04-01", "2026-04-01"],
            "time": ["09:30:00", "09:45:00"],
            "period": ["15", "15"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.4],
            "low": [9.9, 10.1],
            "close": [10.1, 10.35],
            "volume": [1000, 1200],
            "amount": [10100, 12420],
        },
        index=index,
    )

    daily_dir = tmp_path / "daily"
    minute_dir = tmp_path / "minute"
    daily_dir.mkdir()
    minute_dir.mkdir()

    daily_path = _write_daily_csv(daily, "000001", daily_dir)
    minute_path = _write_minute_csv(minute, "000001", minute_dir)

    saved_daily = pd.read_csv(daily_path)
    saved_minute = pd.read_csv(minute_path)

    assert list(saved_daily.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert list(saved_minute.columns[:6]) == ["datetime", "symbol", "date", "time", "period", "open"]


def test_fetch_daily_range_baostock_raises_when_dependency_missing(monkeypatch) -> None:
    import scripts.prepare_m3net_data as module

    monkeypatch.setattr(module, "bs", None)

    try:
        _fetch_daily_range_baostock("000001", "2024-01-01", "2024-12-31")
    except ImportError as exc:
        assert "baostock" in str(exc)
    else:
        raise AssertionError("Expected baostock dependency check to fail.")


def test_is_a_share_baostock_code_filters_indices_and_keeps_common_a_share_codes() -> None:
    assert _is_a_share_baostock_code("sh.600000")
    assert _is_a_share_baostock_code("sz.000001")
    assert _is_a_share_baostock_code("sh.688001")
    assert not _is_a_share_baostock_code("sh.000001")
    assert not _is_a_share_baostock_code("sz.200001")


def test_fetch_minute_range_dispatches_by_source() -> None:
    fetcher = FakeMinuteFetcher()

    _fetch_minute_range(fetcher, "000001", "2026-01-01 09:30:00", "2026-01-01 15:00:00", "15", "", True, "tushare")
    assert fetcher.called is not None
    assert fetcher.called[0] == "tushare"

    _fetch_minute_range(fetcher, "000001", "2026-01-01 09:30:00", "2026-01-01 15:00:00", "15", "", True, "akshare")
    assert fetcher.called is not None
    assert fetcher.called[0] == "akshare"


def test_fetch_minute_range_baostock_logged_in_normalizes_frame(monkeypatch) -> None:
    import scripts.prepare_m3net_data as module

    class FakeResult:
        error_code = "0"
        error_msg = "success"
        fields = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"]

        def __init__(self) -> None:
            self.rows = [
                ["2026-03-31", "20260331093000000", "sz.000001", "10", "10.2", "9.9", "10.1", "1000", "10100", "3"],
                ["2026-03-31", "20260331094500000", "sz.000001", "10.1", "10.3", "10.0", "10.2", "1200", "12240", "3"],
            ]
            self.idx = -1

        def next(self) -> bool:
            self.idx += 1
            return self.idx < len(self.rows)

        def get_row_data(self):
            return self.rows[self.idx]

    class FakeBS:
        def query_history_k_data_plus(self, *args, **kwargs):
            return FakeResult()

    monkeypatch.setattr(module, "bs", FakeBS())
    frame = _fetch_minute_range_baostock_logged_in(
        "000001",
        "2026-03-31 09:30:00",
        "2026-03-31 15:00:00",
        "15",
        "",
    )
    assert list(frame.index.strftime("%H:%M:%S")) == ["09:30:00", "09:45:00"]
    assert frame["symbol"].eq("000001").all()
    assert frame["period"].eq("15").all()


def test_fetch_minute_range_rejects_adjust_for_tushare() -> None:
    fetcher = FakeMinuteFetcher()
    try:
        _fetch_minute_range(fetcher, "000001", "2026-01-01 09:30:00", "2026-01-01 15:00:00", "15", "qfq", True, "tushare")
    except ValueError as exc:
        assert "does not support --minute-adjust" in str(exc)
    else:
        raise AssertionError("Expected tushare minute adjust validation to fail.")
