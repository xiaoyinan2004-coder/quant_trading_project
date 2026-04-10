"""Unified rolling research runner for baseline, M3-Net, and ablation experiments."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.gradient_boosting_factor import GradientBoostingFactorModel, PanelFactorDatasetBuilder
from models.m3net import M3NetStage1Config, M3NetStage1Model
from scripts.evaluate_m3net_stage1 import (
    _filter_stock_data_by_min_rows,
    _realized_forward_return,
    _slice_daily_history,
    _slice_minute_history,
)
from scripts.train_m3net_stage1 import _load_price_folder


EXPERIMENT_NAMES = (
    "baseline_factor",
    "m3net_v2_lite",
    "m3net_no_learned_router",
    "m3net_no_minute",
)


def _build_rebalance_dates(stock_data: dict[str, pd.DataFrame], rebalance_freq: str) -> list[pd.Timestamp]:
    all_dates = sorted(set().union(*[set(pd.to_datetime(df.index)) for df in stock_data.values()]))
    return (
        pd.Series(pd.to_datetime(all_dates))
        .groupby(pd.Series(pd.to_datetime(all_dates)).dt.to_period(rebalance_freq))
        .max()
        .tolist()
    )


def _resolve_rebalance_dates(
    stock_data: dict[str, pd.DataFrame],
    rebalance_freq: str,
    single_trade_date: str | None = None,
) -> list[pd.Timestamp]:
    if single_trade_date:
        return [pd.Timestamp(single_trade_date)]
    return _build_rebalance_dates(stock_data, rebalance_freq)


def _portfolio_summary(returns: list[float]) -> dict[str, float]:
    if not returns:
        return {
            "periods": 0,
            "avg_return": float("nan"),
            "cumulative_return": float("nan"),
            "win_rate": float("nan"),
        }

    series = pd.Series(returns, dtype=float)
    cumulative = float((1.0 + series).prod() - 1.0)
    return {
        "periods": int(len(series)),
        "avg_return": float(series.mean()),
        "cumulative_return": cumulative,
        "win_rate": float((series > 0).mean()),
    }


def _fit_baseline_selector(
    stock_data: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    top_n: int,
    backend: str,
    label_horizon: int,
) -> pd.DataFrame:
    builder = PanelFactorDatasetBuilder()
    panel = builder.build_dataset(
        stock_data,
        label_horizon=label_horizon,
        min_history=80,
    )
    model = GradientBoostingFactorModel(backend=backend)
    model.fit(panel, train_ratio=0.8)
    scored = model.predict(panel)
    selected = model.select_top_stocks(scored, as_of_date=str(as_of_date.date()), top_n=top_n)
    selected = selected.copy()
    selected["model_name"] = "baseline_factor"
    if model.report is not None:
        selected["valid_rank_ic"] = model.report.valid_rank_ic
        selected["valid_rmse"] = model.report.valid_rmse
    return selected


def _fit_m3net_selector(
    stock_data: dict[str, pd.DataFrame],
    minute_data: dict[str, pd.DataFrame] | None,
    as_of_date: pd.Timestamp,
    config: M3NetStage1Config,
    model_name: str,
) -> tuple[pd.DataFrame, object]:
    model = M3NetStage1Model(config=config)
    model.fit(stock_data, minute_data=minute_data)
    selected = model.select_top_stocks(
        stock_data,
        minute_data=minute_data,
        top_n=config.top_n,
        as_of_date=str(as_of_date.date()),
    ).copy()
    selected["model_name"] = model_name
    return selected, model.report


def _evaluate_selection(
    selected: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    label_horizon: int,
) -> tuple[list[float], dict[str, float]]:
    realized_returns: list[float] = []
    for _, row in selected.iterrows():
        realized = _realized_forward_return(
            daily_data[row["symbol"]],
            pd.Timestamp(row["date"]),
            label_horizon,
        )
        if realized is not None:
            realized_returns.append(realized)
    return realized_returns, _portfolio_summary(realized_returns)


def _run_experiment(
    name: str,
    runner: Callable[[dict[str, pd.DataFrame], dict[str, pd.DataFrame] | None, pd.Timestamp], tuple[pd.DataFrame, object | None]],
    daily_data: dict[str, pd.DataFrame],
    minute_data: dict[str, pd.DataFrame] | None,
    rebalance_dates: list[pd.Timestamp],
    min_required_days: int,
    label_horizon: int,
    train_lookback_days: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_rows: list[dict[str, object]] = []
    picks_rows: list[pd.DataFrame] = []
    total_periods = len(rebalance_dates)

    for period_index, trade_date in enumerate(rebalance_dates, start=1):
        print(f"[{name}] period {period_index}/{total_periods} -> {pd.Timestamp(trade_date).date()}")
        sliced_daily = _slice_daily_history(
            daily_data,
            pd.Timestamp(trade_date),
            min_required_days,
            max_history_days=train_lookback_days,
        )
        if len(sliced_daily) < 5:
            print(f"[{name}] skipped {pd.Timestamp(trade_date).date()} because only {len(sliced_daily)} symbols met history requirements")
            continue
        sliced_minute = _slice_minute_history(
            minute_data,
            pd.Timestamp(trade_date),
            max_history_days=train_lookback_days,
        )

        selected, report = runner(sliced_daily, sliced_minute, pd.Timestamp(trade_date))
        del sliced_daily
        del sliced_minute
        gc.collect()
        if selected.empty:
            print(f"[{name}] no selections produced for {pd.Timestamp(trade_date).date()}")
            continue

        realized_returns, summary = _evaluate_selection(selected, daily_data, label_horizon=label_horizon)
        report_payload = asdict(report) if report is not None and hasattr(report, "__dataclass_fields__") else {}
        print(
            f"[{name}] completed {pd.Timestamp(trade_date).date()} | "
            f"selected={len(selected)} avg_realized={summary['avg_return']:.6f} "
            f"win_rate={summary['win_rate']:.2%}"
        )
        period_rows.append(
            {
                "experiment": name,
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "selected_count": int(len(selected)),
                "avg_predicted_score": float(selected["score"].mean()) if "score" in selected.columns else float("nan"),
                "avg_realized_return": summary["avg_return"],
                "cumulative_return_to_period": float((1.0 + pd.Series(realized_returns)).prod() - 1.0) if realized_returns else float("nan"),
                "win_rate_period": summary["win_rate"],
                **report_payload,
            }
        )
        selected = selected.copy()
        selected["trade_date"] = str(pd.Timestamp(trade_date).date())
        if realized_returns:
            selected["period_avg_realized_return"] = summary["avg_return"]
        picks_rows.append(selected)

    period_df = pd.DataFrame(period_rows)
    picks_df = pd.concat(picks_rows, ignore_index=True) if picks_rows else pd.DataFrame()
    return period_df, picks_df


def _build_experiments(
    backend: str,
    top_n: int,
) -> dict[str, Callable[[dict[str, pd.DataFrame], dict[str, pd.DataFrame] | None, pd.Timestamp], tuple[pd.DataFrame, object | None]]]:
    base_config = M3NetStage1Config(factor_backend=backend, top_n=top_n)

    def baseline_runner(stock_data: dict[str, pd.DataFrame], _: dict[str, pd.DataFrame] | None, as_of_date: pd.Timestamp):
        selected = _fit_baseline_selector(
            stock_data,
            as_of_date=as_of_date,
            top_n=top_n,
            backend=backend,
            label_horizon=base_config.label_horizon,
        )
        return selected, None

    def m3net_runner(stock_data: dict[str, pd.DataFrame], minute_subset: dict[str, pd.DataFrame] | None, as_of_date: pd.Timestamp):
        return _fit_m3net_selector(stock_data, minute_subset, as_of_date, base_config, "m3net_v2_lite")

    no_router_config = M3NetStage1Config(
        factor_backend=backend,
        top_n=top_n,
        use_learned_router=False,
    )

    def no_router_runner(stock_data: dict[str, pd.DataFrame], minute_subset: dict[str, pd.DataFrame] | None, as_of_date: pd.Timestamp):
        return _fit_m3net_selector(stock_data, minute_subset, as_of_date, no_router_config, "m3net_no_learned_router")

    no_minute_config = M3NetStage1Config(
        factor_backend=backend,
        top_n=top_n,
    )

    def no_minute_runner(stock_data: dict[str, pd.DataFrame], _: dict[str, pd.DataFrame] | None, as_of_date: pd.Timestamp):
        return _fit_m3net_selector(stock_data, None, as_of_date, no_minute_config, "m3net_no_minute")

    return {
        "baseline_factor": baseline_runner,
        "m3net_v2_lite": m3net_runner,
        "m3net_no_learned_router": no_router_runner,
        "m3net_no_minute": no_minute_runner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified rolling M3-Net research experiments.")
    parser.add_argument("--daily-dir", required=True, help="Folder of per-symbol daily OHLCV CSV files.")
    parser.add_argument("--minute-dir", help="Folder of per-symbol minute CSV files.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--train-lookback-days", type=int, default=504)
    parser.add_argument("--single-trade-date", help="Run only a single rebalance date, for example 2025-12-31.")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENT_NAMES), help="Run only the selected experiments.")
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    daily_data = _load_price_folder(Path(args.daily_dir))
    minute_data = _load_price_folder(Path(args.minute_dir)) if args.minute_dir else None
    daily_data = _filter_stock_data_by_min_rows(daily_data, args.min_stock_rows)
    if minute_data is not None:
        minute_data = {symbol: frame for symbol, frame in minute_data.items() if symbol in daily_data}
    rebalance_dates = _resolve_rebalance_dates(daily_data, args.rebalance_freq, args.single_trade_date)

    base_config = M3NetStage1Config(factor_backend=args.backend, top_n=args.top_n)
    min_required_days = max(base_config.min_history, base_config.label_horizon + base_config.alpha_memory_lookback_long)
    experiment_map = _build_experiments(args.backend, args.top_n)
    selected_experiments = args.experiments or list(EXPERIMENT_NAMES)

    all_periods: list[pd.DataFrame] = []
    all_picks: list[pd.DataFrame] = []

    for experiment_name in selected_experiments:
        runner = experiment_map[experiment_name]
        print(f"Running experiment: {experiment_name} ({len(rebalance_dates)} rebalance periods)")
        period_df, picks_df = _run_experiment(
            experiment_name,
            runner,
            daily_data,
            minute_data,
            rebalance_dates,
            min_required_days=min_required_days,
            label_horizon=base_config.label_horizon,
            train_lookback_days=args.train_lookback_days,
        )
        all_periods.append(period_df)
        all_picks.append(picks_df)
        gc.collect()

    periods = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    picks = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()

    summary_rows = []
    if not periods.empty:
        for experiment, group in periods.groupby("experiment"):
            summary_rows.append(
                {
                    "experiment": experiment,
                    "periods": int(len(group)),
                    "avg_realized_return": float(group["avg_realized_return"].mean()),
                    "win_rate": float((group["avg_realized_return"] > 0).mean()),
                    "cumulative_return_proxy": float((1.0 + group["avg_realized_return"].fillna(0.0)).prod() - 1.0),
                    "avg_fused_rank_ic": float(group["fused_valid_rank_ic"].mean()) if "fused_valid_rank_ic" in group.columns else float("nan"),
                    "avg_tabular_rank_ic": float(group["tabular_valid_rank_ic"].mean()) if "tabular_valid_rank_ic" in group.columns else float("nan"),
                    "avg_sequence_rank_ic": float(group["sequence_valid_rank_ic"].mean()) if "sequence_valid_rank_ic" in group.columns else float("nan"),
                }
            )
    summary = pd.DataFrame(summary_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    periods_path = output_dir / "rolling_periods.csv"
    picks_path = output_dir / "rolling_picks.csv"
    summary_path = output_dir / "summary.csv"
    periods.to_csv(periods_path, index=False)
    picks.to_csv(picks_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved periods to: {periods_path}")
    print(f"Saved picks to: {picks_path}")
    print(f"Saved summary to: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
