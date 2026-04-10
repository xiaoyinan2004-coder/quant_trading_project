"""Launch each research experiment in an isolated subprocess to reduce memory pressure."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m3net_research import EXPERIMENT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3-Net research experiments sequentially in isolated processes.")
    parser.add_argument("--daily-dir", required=True, help="Folder of per-symbol daily OHLCV CSV files.")
    parser.add_argument("--minute-dir", help="Folder of per-symbol minute CSV files.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--train-lookback-days", type=int, default=504)
    parser.add_argument("--output-dir", default="artifacts/research_seq")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENT_NAMES), help="Run only the selected experiments.")
    args = parser.parse_args()

    experiments = args.experiments or list(EXPERIMENT_NAMES)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for index, experiment in enumerate(experiments, start=1):
        experiment_dir = output_root / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
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
            "--output-dir",
            str(experiment_dir),
            "--experiments",
            experiment,
        ]
        if args.minute_dir:
            command.extend(["--minute-dir", args.minute_dir])

        print(f"[launcher] ({index}/{len(experiments)}) starting {experiment}")
        print(f"[launcher] command: {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"[launcher] experiment {experiment} failed with code {completed.returncode}")
        print(f"[launcher] completed {experiment}")

    print(f"[launcher] all experiments completed -> {output_root}")


if __name__ == "__main__":
    main()
