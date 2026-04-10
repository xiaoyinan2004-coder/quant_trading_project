"""Train the first-stage M3-Net on local A-share data files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.m3net import M3NetStage1Config, M3NetStage1Model


def _downcast_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], downcast="float")
    return result


def _load_price_folder(folder: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(folder.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date")
        elif "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            frame = frame.set_index("datetime")
        else:
            raise ValueError(f"{csv_path} is missing a date or datetime column.")
        data[csv_path.stem] = _downcast_numeric_columns(frame.sort_index())
    if not data:
        raise ValueError(f"No CSV files found in {folder}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the M3-Net Stage 1 model.")
    parser.add_argument("--daily-dir", required=True, help="Folder of per-symbol daily OHLCV CSV files.")
    parser.add_argument("--minute-dir", help="Folder of per-symbol minute CSV files.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--output", default="artifacts/m3net_stage1.joblib")
    args = parser.parse_args()

    daily_data = _load_price_folder(Path(args.daily_dir))
    minute_data = _load_price_folder(Path(args.minute_dir)) if args.minute_dir else None

    config = M3NetStage1Config(factor_backend=args.backend)
    model = M3NetStage1Model(config=config)
    model.fit(daily_data, minute_data=minute_data)
    model.save(args.output)

    top = model.select_top_stocks(daily_data, minute_data=minute_data, top_n=config.top_n)
    print("Training completed.")
    print(f"Model saved to: {args.output}")
    print(model.report)
    print(top[["date", "symbol", "score", "tabular_score", "sequence_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
