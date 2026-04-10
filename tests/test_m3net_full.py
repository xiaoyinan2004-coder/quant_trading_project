from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from models.m3net_full import M3NetFullConfig, M3NetFullDatasetBuilder, M3NetFullModel
from test_m3net_stage1 import make_synthetic_minute_data, make_synthetic_stock_data


def test_m3net_full_dataset_builder_and_forward_pass() -> None:
    stock_data = make_synthetic_stock_data(symbols=8, periods=220)
    minute_data = make_synthetic_minute_data(stock_data)
    config = M3NetFullConfig(
        daily_lookback=40,
        minute_lookback_days=10,
        top_n=4,
        hidden_dim=64,
        daily_layers=2,
        intraday_layers=1,
        fusion_layers=1,
        factor_input_dim=16,
    )
    builder = M3NetFullDatasetBuilder(config=config)
    rebalance_dates = pd.date_range("2024-08-30", periods=3, freq="M")

    samples = builder.build_samples(stock_data, minute_data=minute_data, rebalance_dates=rebalance_dates)

    assert samples
    sample = samples[-1]
    assert sample.daily_sequence.ndim == 3
    assert sample.intraday_sequence.ndim == 3
    assert sample.intraday_sequence.shape[1] == config.minute_lookback_days
    assert sample.factor_features.ndim == 2
    assert sample.memory_features.shape[-1] == config.memory_input_dim
    intraday_variation = (sample.intraday_sequence[:, 0, :] - sample.intraday_sequence[:, -1, :]).abs().sum().item()
    assert intraday_variation > 0

    model = M3NetFullModel(config=config)
    output = model(
        sample.daily_sequence,
        sample.intraday_sequence,
        sample.factor_features,
        sample.memory_features,
    )

    assert output.return_pred.shape[0] == len(sample.symbols)
    assert output.risk_pred.shape == output.return_pred.shape
    assert output.confidence.shape == output.return_pred.shape
    assert output.top_pick_prob.shape == output.return_pred.shape
    assert output.graph_projection.ndim == 2
    assert output.graph_positive_mask.ndim == 2
    assert output.graph_positive_mask.shape[0] == len(sample.symbols)
    loss = model.loss(output, sample.future_return, sample.future_risk)
    assert torch.isfinite(loss)
