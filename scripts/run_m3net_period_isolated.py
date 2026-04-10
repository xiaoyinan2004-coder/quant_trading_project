"""Run research experiments with one subprocess per rebalance period."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.evaluate_m3net_stage1 import _filter_stock_data_by_min_rows
from scripts.run_m3net_research import EXPERIMENT_NAMES, _resolve_rebalance_dates
from scripts.train_m3net_stage1 import _load_price_folder


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research experiments in isolated subprocesses per rebalance date.")
    parser.add_argument("--daily-dir", required=True, help="Folder of per-symbol daily OHLCV CSV files.")
    parser.add_argument("--minute-dir", help="Folder of per-symbol minute CSV files.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--train-lookback-days", type=int, default=504)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENT_NAMES), required=True)
    parser.add_argument("--output-dir", default="artifacts/research_isolated")
    args = parser.parse_args()

    daily_data = _filter_stock_data_by_min_rows(_load_price_folder(Path(args.daily_dir)), args.min_stock_rows)
    rebalance_dates = _resolve_rebalance_dates(daily_data, args.rebalance_freq)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        experiment_dir = output_root / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        periods_frames: list[pd.DataFrame] = []
        picks_frames: list[pd.DataFrame] = []

        for index, trade_date in enumerate(rebalance_dates, start=1):
            trade_date_str = str(pd.Timestamp(trade_date).date())
            period_dir = experiment_dir / "periods" / trade_date_str
            period_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_m3net_research.py"),
                "--daily-dir",
                args.daily_dir,
                "--backend",
                args.backend,
                "--rebalance-freq",
                args.rebalance_freq,
                "--top-n",
                str(args.top_n),
                "--min-stock-rows",
                str(args.min_stock_rows),
                "--train-lookback-days",
                str(args.train_lookback_days),
                "--single-trade-date",
                trade_date_str,
                "--output-dir",
                str(period_dir),
                "--experiments",
                experiment,
            ]
            if args.minute_dir:
                command.extend(["--minute-dir", args.minute_dir])

            print(f"[isolated] {experiment} period {index}/{len(rebalance_dates)} -> {trade_date_str}")
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"[isolated] experiment {experiment} failed on {trade_date_str} with code {completed.returncode}"
                )

            period_frame = _read_csv_if_exists(period_dir / "rolling_periods.csv")
            picks_frame = _read_csv_if_exists(period_dir / "rolling_picks.csv")
            if not period_frame.empty:
                periods_frames.append(period_frame)
            if not picks_frame.empty:
                picks_frames.append(picks_frame)

        periods = pd.concat(periods_frames, ignore_index=True) if periods_frames else pd.DataFrame()
        picks = pd.concat(picks_frames, ignore_index=True) if picks_frames else pd.DataFrame()
        summary_rows: list[dict[str, float | int | str]] = []
        if not periods.empty:
            for experiment_name, group in periods.groupby("experiment"):
                summary_rows.append(
                    {
                        "experiment": experiment_name,
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

        periods.to_csv(experiment_dir / "rolling_periods.csv", index=False)
        picks.to_csv(experiment_dir / "rolling_picks.csv", index=False)
        summary.to_csv(experiment_dir / "summary.csv", index=False)
        print(f"[isolated] completed {experiment} -> {experiment_dir}")


if __name__ == "__main__":
    main()
