"""Prepare daily and minute A-share CSV folders for M3-Net experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import sys
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.data_fetcher import DataFetcher

try:
    import baostock as bs
except ImportError:  # pragma: no cover - optional runtime dependency
    bs = None


REQUIRED_DAILY_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUIRED_MINUTE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_code(symbol: str) -> str:
    return str(symbol).strip().zfill(6)


def _load_symbol_list(
    symbols: list[str] | None,
    symbols_file: str | None,
    limit: int | None,
    fetcher: DataFetcher,
    daily_source: str,
    universe_date: str | None,
) -> list[str]:
    merged: list[str] = []

    if symbols:
        merged.extend(symbols)

    if symbols_file:
        file_path = Path(symbols_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Symbol list file not found: {file_path}")
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                merged.append(line)

    if not merged:
        if daily_source == "baostock":
            merged.extend(_fetch_baostock_universe(universe_date))
        else:
            stock_list = fetcher.get_stock_list()
            merged.extend(stock_list["code"].astype(str).tolist())

    normalized = []
    seen = set()
    for symbol in merged:
        code = _normalize_code(symbol)
        if code not in seen:
            seen.add(code)
            normalized.append(code)

    if limit is not None:
        return normalized[:limit]
    return normalized


def _ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str], symbol: str, data_type: str) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{data_type} data for {symbol} is missing columns: {missing}")
    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    return result


def _write_daily_csv(frame: pd.DataFrame, symbol: str, output_dir: Path) -> Path:
    normalized = _ensure_columns(frame, REQUIRED_DAILY_COLUMNS, symbol, "daily")
    payload = normalized.loc[:, REQUIRED_DAILY_COLUMNS].copy()
    payload.index.name = "date"
    path = output_dir / f"{symbol}.csv"
    payload.to_csv(path)
    return path


def _write_minute_csv(frame: pd.DataFrame, symbol: str, output_dir: Path) -> Path:
    normalized = _ensure_columns(frame, REQUIRED_MINUTE_COLUMNS, symbol, "minute")
    preferred = [
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
    columns = [column for column in preferred if column in normalized.columns]
    payload = normalized.loc[:, columns].copy()
    payload.index.name = "datetime"
    path = output_dir / f"{symbol}.csv"
    payload.to_csv(path)
    return path


def _fetch_daily_range(fetcher: DataFetcher, symbol: str, start: str, end: str) -> pd.DataFrame:
    return fetcher.fetch_akshare_a_stock(symbol, start, end)


def _fetch_daily_range_baostock(symbol: str, start: str, end: str) -> pd.DataFrame:
    if bs is None:
        raise ImportError("baostock is not installed. Run `pip install baostock`.")

    with _baostock_session():
        return _fetch_daily_range_baostock_logged_in(symbol, start, end)


def _fetch_daily_range_baostock_logged_in(symbol: str, start: str, end: str) -> pd.DataFrame:
    exchange = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    code = f"{exchange}.{symbol}"
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="2",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock query failed for {symbol}: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise ValueError(f"No daily data returned from baostock for {symbol}.")

    frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    for column in REQUIRED_DAILY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    return frame


@contextmanager
def _baostock_session():
    if bs is None:
        raise ImportError("baostock is not installed. Run `pip install baostock`.")
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login_result.error_msg}")
    try:
        yield
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def _is_a_share_baostock_code(code: str) -> bool:
    code = str(code)
    if code.startswith(("sh.600", "sh.601", "sh.603", "sh.605", "sh.688")):
        return True
    if code.startswith(("sz.000", "sz.001", "sz.002", "sz.003", "sz.300")):
        return True
    return False


def _fetch_baostock_universe(day: str | None) -> list[str]:
    query_day = day or pd.Timestamp.today().strftime("%Y-%m-%d")
    with _baostock_session():
        rs = bs.query_all_stock(day=query_day)
        if rs.error_code != "0":
            raise RuntimeError(f"baostock universe query failed: {rs.error_msg}")

        symbols: list[str] = []
        while rs.next():
            code, trade_status, _name = rs.get_row_data()
            if trade_status != "1":
                continue
            if not _is_a_share_baostock_code(code):
                continue
            symbols.append(code.split(".")[1])
    return symbols


def _fetch_minute_range(
    fetcher: DataFetcher,
    symbol: str,
    start_datetime: str,
    end_datetime: str,
    period: str,
    adjust: str,
    use_cache: bool,
) -> pd.DataFrame:
    return fetcher.fetch_akshare_a_stock_minute(
        symbol=symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        period=period,
        adjust=adjust,
        use_cache=use_cache,
    )


def _existing_csv_meets_row_threshold(path: Path, min_rows: int) -> bool:
    if not path.exists():
        return False
    if min_rows <= 0:
        return True
    try:
        row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
    except OSError:
        return False
    return row_count >= min_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local A-share daily/minute CSV folders for M3-Net.")
    parser.add_argument("--symbols", nargs="*", help="Explicit stock codes such as 000001 600000.")
    parser.add_argument("--symbols-file", help="UTF-8 text file with one stock code per line.")
    parser.add_argument("--limit", type=int, help="Only fetch the first N symbols after deduplication.")
    parser.add_argument("--daily-start", help="Daily start date in YYYY-MM-DD format.")
    parser.add_argument("--daily-end", help="Daily end date in YYYY-MM-DD format.")
    parser.add_argument("--minute-start", help="Minute start datetime in YYYY-MM-DD HH:MM:SS format.")
    parser.add_argument("--minute-end", help="Minute end datetime in YYYY-MM-DD HH:MM:SS format.")
    parser.add_argument("--minute-period", default="15", choices=["1", "5", "15", "30", "60"])
    parser.add_argument("--minute-adjust", default="")
    parser.add_argument("--daily-source", default="akshare", choices=["akshare", "baostock"])
    parser.add_argument("--universe-date", help="Universe snapshot date for auto-discovery, YYYY-MM-DD.")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-minute", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-existing-rows", type=int, default=0)
    args = parser.parse_args()

    if args.skip_daily and args.skip_minute:
        raise ValueError("Both daily and minute downloads are skipped. Nothing to do.")

    if not args.skip_daily and (not args.daily_start or not args.daily_end):
        raise ValueError("Daily download requires --daily-start and --daily-end.")

    if not args.skip_minute and (not args.minute_start or not args.minute_end):
        raise ValueError("Minute download requires --minute-start and --minute-end.")

    fetcher = DataFetcher()
    symbols = _load_symbol_list(
        args.symbols,
        args.symbols_file,
        args.limit,
        fetcher,
        daily_source=args.daily_source,
        universe_date=args.universe_date or args.daily_end,
    )

    output_root = Path(args.output_root)
    daily_dir = output_root / "daily"
    minute_dir = output_root / "minute"
    daily_dir.mkdir(parents=True, exist_ok=True)
    minute_dir.mkdir(parents=True, exist_ok=True)

    daily_success = 0
    minute_success = 0
    daily_failures: list[str] = []
    minute_failures: list[str] = []

    baostock_context = _baostock_session() if args.daily_source == "baostock" and not args.skip_daily else None
    if baostock_context is not None:
        baostock_context.__enter__()
    try:
        for index, symbol in enumerate(symbols, start=1):
            print(f"[{index}/{len(symbols)}] Processing {symbol}")

            if not args.skip_daily:
                try:
                    daily_target = daily_dir / f"{symbol}.csv"
                    if args.skip_existing and _existing_csv_meets_row_threshold(daily_target, args.min_existing_rows):
                        print(f"  daily skipped -> {daily_target}")
                    else:
                        if args.daily_source == "baostock":
                            daily = _fetch_daily_range_baostock_logged_in(symbol, args.daily_start, args.daily_end)
                        else:
                            daily = _fetch_daily_range(fetcher, symbol, args.daily_start, args.daily_end)
                        daily_path = _write_daily_csv(daily, symbol, daily_dir)
                        daily_success += 1
                        print(f"  daily ok -> {daily_path}")
                except Exception as exc:
                    daily_failures.append(symbol)
                    print(f"  daily failed -> {symbol}: {exc}")

            if not args.skip_minute:
                try:
                    minute_target = minute_dir / f"{symbol}.csv"
                    if args.skip_existing and _existing_csv_meets_row_threshold(minute_target, args.min_existing_rows):
                        print(f"  minute skipped -> {minute_target}")
                    else:
                        minute = _fetch_minute_range(
                            fetcher,
                            symbol,
                            args.minute_start,
                            args.minute_end,
                            args.minute_period,
                            args.minute_adjust,
                            args.use_cache,
                        )
                        minute_path = _write_minute_csv(minute, symbol, minute_dir)
                        minute_success += 1
                        print(f"  minute ok -> {minute_path}")
                except Exception as exc:
                    minute_failures.append(symbol)
                    print(f"  minute failed -> {symbol}: {exc}")
    finally:
        if baostock_context is not None:
            baostock_context.__exit__(None, None, None)

    print("Preparation completed.")
    if not args.skip_daily:
        print(f"Daily files written: {daily_success}")
        if daily_failures:
            print(f"Daily failures: {len(daily_failures)}")
    if not args.skip_minute:
        print(f"Minute files written: {minute_success}")
        if minute_failures:
            print(f"Minute failures: {len(minute_failures)}")


if __name__ == "__main__":
    main()
