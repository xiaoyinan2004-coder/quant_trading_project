"""Rolling evaluation utility for the lightweight M3-Net research workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.m3net import M3NetStage1Config, M3NetStage1Model
from scripts.train_m3net_stage1 import _load_price_folder


def _filter_stock_data_by_min_rows(
    stock_data: dict[str, pd.DataFrame],
    min_rows: int,
) -> dict[str, pd.DataFrame]:
    if min_rows <= 0:
        return stock_data
    return {symbol: frame for symbol, frame in stock_data.items() if len(frame) >= min_rows}


def _slice_daily_history(
    stock_data: dict[str, pd.DataFrame],
    end_date: pd.Timestamp,
    min_history: int,
    max_history_days: int | None = None,
) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in stock_data.items():
        subset = frame.loc[frame.index <= end_date]
        if max_history_days is not None and max_history_days > 0:
            subset = subset.tail(max_history_days)
        subset = subset.copy()
        if len(subset) >= min_history:
            sliced[symbol] = subset
    return sliced


def _slice_minute_history(
    minute_data: dict[str, pd.DataFrame] | None,
    end_date: pd.Timestamp,
    max_history_days: int | None = None,
) -> dict[str, pd.DataFrame] | None:
    if minute_data is None:
        return None

    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in minute_data.items():
        subset = frame.loc[frame.index.normalize() <= end_date.normalize()]
        if max_history_days is not None and max_history_days > 0 and not subset.empty:
            cutoff = end_date.normalize() - pd.Timedelta(days=max_history_days * 2)
            subset = subset.loc[subset.index.normalize() >= cutoff]
        subset = subset.copy()
        if not subset.empty:
            sliced[symbol] = subset
    return sliced


def _realized_forward_return(frame: pd.DataFrame, trade_date: pd.Timestamp, horizon: int) -> float | None:
    if trade_date not in frame.index:
        return None
    loc = frame.index.get_loc(trade_date)
    if isinstance(loc, slice):
        loc = loc.stop - 1
    if loc + horizon >= len(frame):
        return None
    start_price = float(frame["close"].iloc[loc])
    end_price = float(frame["close"].iloc[loc + horizon])
    if start_price == 0:
        return None
    return end_price / start_price - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a rolling M3-Net evaluation on local CSV data.")
    parser.add_argument("--daily-dir", required=True, help="Folder of per-symbol daily OHLCV CSV files.")
    parser.add_argument("--minute-dir", help="Folder of per-symbol minute CSV files.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--rebalance-freq", default="M", help="Pandas period alias, for example M or W.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--train-lookback-days", type=int, default=504)
    parser.add_argument("--output", default="artifacts/m3net_stage1_rolling_eval.csv")
    args = parser.parse_args()

    daily_data = _load_price_folder(Path(args.daily_dir))
    minute_data = _load_price_folder(Path(args.minute_dir)) if args.minute_dir else None
    daily_data = _filter_stock_data_by_min_rows(daily_data, args.min_stock_rows)
    if minute_data is not None:
        minute_data = {symbol: frame for symbol, frame in minute_data.items() if symbol in daily_data}

    config = M3NetStage1Config(factor_backend=args.backend, top_n=args.top_n)
    all_dates = sorted(set().union(*[set(pd.to_datetime(df.index)) for df in daily_data.values()]))
    rebalance_dates = (
        pd.Series(pd.to_datetime(all_dates))
        .groupby(pd.Series(pd.to_datetime(all_dates)).dt.to_period(args.rebalance_freq))
        .max()
        .tolist()
    )

    rows: list[dict[str, object]] = []
    min_required_days = max(config.min_history, config.label_horizon + config.alpha_memory_lookback_long)

    for trade_date in rebalance_dates:
        sliced_daily = _slice_daily_history(
            daily_data,
            pd.Timestamp(trade_date),
            min_required_days,
            max_history_days=args.train_lookback_days,
        )
        if len(sliced_daily) < max(config.top_n, 5):
            continue

        sliced_minute = _slice_minute_history(
            minute_data,
            pd.Timestamp(trade_date),
            max_history_days=args.train_lookback_days,
        )
        model = M3NetStage1Model(config=config)
        model.fit(sliced_daily, minute_data=sliced_minute)
        scored = model.select_top_stocks(
            sliced_daily,
            minute_data=sliced_minute,
            top_n=config.top_n,
            as_of_date=str(pd.Timestamp(trade_date).date()),
        )
        if scored.empty:
            continue

        realized_returns = []
        for _, row in scored.iterrows():
            realized = _realized_forward_return(
                daily_data[row["symbol"]],
                pd.Timestamp(row["date"]),
                config.label_horizon,
            )
            if realized is not None:
                realized_returns.append(realized)

        rows.append(
            {
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "selected_count": int(len(scored)),
                "avg_predicted_score": float(scored["score"].mean()),
                "avg_realized_return": float(pd.Series(realized_returns).mean()) if realized_returns else float("nan"),
                "router_mode": str(scored["router_mode"].iloc[0]) if "router_mode" in scored.columns else "unknown",
                "report": model.report,
            }
        )

    result = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved rolling evaluation to: {output_path}")
    if not result.empty:
        print(result.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
