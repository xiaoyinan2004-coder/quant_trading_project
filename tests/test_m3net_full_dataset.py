from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.m3net_full.config import M3NetFullConfig
from models.m3net_full.dataset import M3NetFullDatasetBuilder, torch


@pytest.mark.skipif(torch is None, reason="PyTorch is required for tensor dataset tests.")
def test_build_intraday_tensor_uses_writable_array_for_normalization() -> None:
    config = M3NetFullConfig(minute_lookback_days=3)
    builder = M3NetFullDatasetBuilder(config=config)
    minute_columns = ["intraday_return", "intraday_range", "has_minute_features"]
    symbol_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-27", "2026-03-30", "2026-03-31"]),
            "intraday_return": [0.01, 0.02, -0.01],
            "intraday_range": [0.03, 0.02, 0.01],
            "has_minute_features": [1.0, 1.0, 1.0],
        }
    )

    tensor = builder._build_intraday_tensor(
        symbol_panel=symbol_panel,
        trade_date=pd.Timestamp("2026-03-31"),
        minute_columns=minute_columns,
    )

    values = tensor.detach().cpu().numpy()
    assert values.shape == (3, 3)
    assert np.isfinite(values).all()
