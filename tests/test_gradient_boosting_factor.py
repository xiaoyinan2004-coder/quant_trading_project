from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factors.a_share_factors import IndexEnhancementSelector
from models.gradient_boosting_factor import (
    GradientBoostingFactorModel,
    MachineLearningStockSelector,
    PanelFactorDatasetBuilder,
)


def make_synthetic_stock_data(
    symbols: int = 8,
    periods: int = 220,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=periods, freq="B")
    stock_data: dict[str, pd.DataFrame] = {}

    for idx in range(symbols):
        symbol = f"S{idx:03d}"
        quality = rng.normal(0.0, 0.5)
        signal = 0.0
        close = 20.0 + idx
        rows = []

        for _ in dates:
            signal = 0.88 * signal + 0.18 * quality + rng.normal(0.0, 0.18)
            daily_return = 0.0035 * signal + rng.normal(0.0002, 0.0075)
            open_price = close * (1.0 + rng.normal(0.0, 0.003))
            close = max(close * (1.0 + daily_return), 1.0)
            high = max(open_price, close) * (1.0 + abs(rng.normal(0.004, 0.002)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0.004, 0.002)))
            volume = int(1_000_000 * (1.0 + 0.6 * abs(signal) + rng.uniform(0.05, 0.5)))
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


def test_panel_factor_dataset_builder_creates_panel() -> None:
    builder = PanelFactorDatasetBuilder()
    panel = builder.build_dataset(make_synthetic_stock_data(), label_horizon=5)

    assert not panel.empty
    assert {"date", "symbol", "label"}.issubset(panel.columns)
    assert panel["symbol"].nunique() == 8
    assert panel["label"].notna().all()


@pytest.mark.parametrize(
    ("backend", "library_name", "params"),
    [
        (
            "lightgbm",
            "lightgbm",
            {"n_estimators": 80, "learning_rate": 0.08, "num_leaves": 15},
        ),
        (
            "xgboost",
            "xgboost",
            {"n_estimators": 60, "learning_rate": 0.08, "max_depth": 4},
        ),
    ],
)
def test_gradient_boosting_factor_model_pipeline(
    backend: str,
    library_name: str,
    params: dict[str, float],
    tmp_path: Path,
) -> None:
    pytest.importorskip(library_name)

    stock_data = make_synthetic_stock_data()
    builder = PanelFactorDatasetBuilder()
    panel = builder.build_dataset(stock_data, label_horizon=5)

    model = GradientBoostingFactorModel(backend=backend, model_params=params)
    scored = model.fit_predict(panel, train_ratio=0.75)
    latest = model.select_top_stocks(scored, top_n=3)
    importance = model.feature_importance(top_n=5)

    assert "score" in scored.columns
    assert not latest.empty
    assert len(latest) == 3
    assert not importance.empty
    assert model.report is not None
    assert model.report.feature_count == len(model.feature_columns)
    assert model.report.valid_rank_ic > -0.2

    save_path = tmp_path / f"{backend}_factor_model.joblib"
    model.save(save_path)
    loaded = GradientBoostingFactorModel.load(save_path)
    reloaded_scored = loaded.predict(panel.tail(20))

    assert "score" in reloaded_scored.columns
    assert reloaded_scored["score"].notna().all()


def test_machine_learning_stock_selector_returns_ranked_stocks() -> None:
    pytest.importorskip("lightgbm")

    stock_data = make_synthetic_stock_data()
    builder = PanelFactorDatasetBuilder()
    panel = builder.build_dataset(stock_data, label_horizon=5)

    model = GradientBoostingFactorModel(
        backend="lightgbm",
        model_params={"n_estimators": 60, "learning_rate": 0.08, "num_leaves": 15},
    )
    model.fit(panel, train_ratio=0.8)

    selector = MachineLearningStockSelector(model, dataset_builder=builder)
    selected = selector.select_stocks(stock_data, top_n=5)

    assert len(selected) == 5
    assert selected["score"].is_monotonic_decreasing


def test_index_enhancement_selector_can_use_trained_model() -> None:
    pytest.importorskip("lightgbm")

    stock_data = make_synthetic_stock_data()
    builder = PanelFactorDatasetBuilder()
    panel = builder.build_dataset(stock_data, label_horizon=5)

    model = GradientBoostingFactorModel(
        backend="lightgbm",
        model_params={"n_estimators": 60, "learning_rate": 0.08, "num_leaves": 15},
    )
    model.fit(panel, train_ratio=0.8)

    selector = IndexEnhancementSelector(model=model)
    selected_codes = selector.select_stocks(
        stock_data=stock_data,
        index_components=list(stock_data.keys()),
        top_n=5,
    )

    assert len(selected_codes) == 5
    assert all(code in stock_data for code in selected_codes)
