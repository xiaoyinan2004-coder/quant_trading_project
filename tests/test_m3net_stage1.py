from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.m3net import M3NetStage1Config, M3NetStage1Model


def make_synthetic_stock_data(
    symbols: int = 6,
    periods: int = 180,
    seed: int = 17,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    stock_data: dict[str, pd.DataFrame] = {}

    for idx in range(symbols):
        symbol = f"S{idx:03d}"
        quality = rng.normal(0.0, 0.8)
        latent_signal = 0.0
        close = 15.0 + idx
        rows = []

        for _ in dates:
            latent_signal = 0.86 * latent_signal + 0.22 * quality + rng.normal(0.0, 0.16)
            daily_return = 0.004 * latent_signal + rng.normal(0.0001, 0.009)
            open_price = close * (1.0 + rng.normal(0.0, 0.0025))
            close = max(close * (1.0 + daily_return), 1.0)
            high = max(open_price, close) * (1.0 + abs(rng.normal(0.005, 0.002)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0.005, 0.002)))
            volume = int(900_000 * (1.0 + 0.4 * abs(latent_signal) + rng.uniform(0.1, 0.5)))
            rows.append(
                {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        stock_data[symbol] = pd.DataFrame(rows, index=dates)

    return stock_data


def make_synthetic_minute_data(
    stock_data: dict[str, pd.DataFrame],
    bars_per_day: int = 6,
    seed: int = 123,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    minute_data: dict[str, pd.DataFrame] = {}
    offsets = [0, 15, 30, 45, 60, 75][:bars_per_day]

    for symbol, daily_df in stock_data.items():
        rows = []
        for trade_date, row in daily_df.tail(60).iterrows():
            base_open = float(row["open"])
            base_close = float(row["close"])
            session_prices = np.linspace(base_open, base_close, num=bars_per_day)
            for idx, offset in enumerate(offsets):
                ts = pd.Timestamp(trade_date).replace(hour=9, minute=30) + pd.Timedelta(minutes=offset)
                close_price = float(session_prices[idx] * (1.0 + rng.normal(0.0, 0.0015)))
                open_price = float(close_price * (1.0 + rng.normal(0.0, 0.0008)))
                high = max(open_price, close_price) * (1.0 + abs(rng.normal(0.0015, 0.0005)))
                low = min(open_price, close_price) * (1.0 - abs(rng.normal(0.0015, 0.0005)))
                volume = int(max(row["volume"] / bars_per_day * (1.0 + rng.normal(0.0, 0.08)), 100))
                rows.append(
                    {
                        "datetime": ts,
                        "symbol": symbol,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close_price,
                        "volume": volume,
                        "amount": volume * close_price,
                    }
                )

        frame = pd.DataFrame(rows).set_index("datetime").sort_index()
        frame["date"] = frame.index.strftime("%Y-%m-%d")
        frame["time"] = frame.index.strftime("%H:%M:%S")
        frame["period"] = "15"
        minute_data[symbol] = frame

    return minute_data


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_m3net_stage1_can_fit_and_rank_stocks(backend: str, tmp_path: Path) -> None:
    pytest.importorskip(backend)

    stock_data = make_synthetic_stock_data()
    minute_data = make_synthetic_minute_data(stock_data)
    config = M3NetStage1Config(
        factor_backend=backend,
        factor_model_params={
            "n_estimators": 80,
            "learning_rate": 0.08,
            "num_leaves": 15,
        }
        if backend == "lightgbm"
        else {
            "n_estimators": 60,
            "learning_rate": 0.08,
            "max_depth": 4,
        },
        sequence_model_params={"max_depth": 3, "learning_rate": 0.05, "max_iter": 80},
        label_horizon=5,
        train_ratio=0.75,
    )
    model = M3NetStage1Model(config=config)
    model.fit(stock_data, minute_data=minute_data)

    latest_date = str(max(df.index.max() for df in stock_data.values()).date())
    scored = model.predict(stock_data, minute_data=minute_data, as_of_date=latest_date)
    selected = model.select_top_stocks(stock_data, minute_data=minute_data, top_n=3, as_of_date=latest_date)

    assert not scored.empty
    assert {
        "tabular_score",
        "sequence_score",
        "score",
        "tabular_weight",
        "sequence_weight",
        "router_mode",
        "tabular_rank_ic_20d",
        "sequence_rank_ic_20d",
        "expert_ic_gap_20d",
    }.issubset(scored.columns)
    assert np.allclose(scored["tabular_weight"] + scored["sequence_weight"], 1.0)
    assert len(selected) == 3
    assert selected["score"].is_monotonic_decreasing
    assert model.report is not None
    assert model.report.valid_rows > 0
    assert model.report.router_mode in {"learned", "heuristic"}

    save_path = tmp_path / f"m3net_stage1_{backend}.joblib"
    model.save(save_path)
    loaded = M3NetStage1Model.load(save_path)
    reloaded = loaded.select_top_stocks(stock_data, minute_data=minute_data, top_n=2, as_of_date=latest_date)
    assert len(reloaded) == 2


def test_m3net_stage1_works_without_minute_data() -> None:
    pytest.importorskip("lightgbm")

    stock_data = make_synthetic_stock_data()
    model = M3NetStage1Model(
        M3NetStage1Config(
            factor_backend="lightgbm",
            factor_model_params={"n_estimators": 60, "learning_rate": 0.08, "num_leaves": 15},
            sequence_model_params={"max_depth": 3, "learning_rate": 0.05, "max_iter": 60},
            train_ratio=0.8,
        )
    )
    model.fit(stock_data, minute_data=None)
    selected = model.select_top_stocks(stock_data, minute_data=None, top_n=4)

    assert len(selected) == 4
    assert selected["has_minute_features"].eq(0.0).all()
    assert selected["router_mode"].isin(["learned", "heuristic"]).all()
