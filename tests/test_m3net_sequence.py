from __future__ import annotations

import numpy as np
import pandas as pd

from models.m3net.sequence import SequenceFeatureBuilder


def test_build_minute_features_handles_zero_open_without_division_error() -> None:
    builder = SequenceFeatureBuilder(lookback=5)
    minute_index = pd.to_datetime(
        [
            "2026-03-31 09:30:00",
            "2026-03-31 09:45:00",
            "2026-03-31 10:00:00",
            "2026-03-31 13:00:00",
            "2026-03-31 15:00:00",
        ]
    )
    minute_df = pd.DataFrame(
        {
            "open": [0.0, 10.0, 10.2, 10.3, 10.4],
            "high": [10.1, 10.2, 10.3, 10.5, 10.6],
            "low": [0.0, 9.9, 10.1, 10.2, 10.3],
            "close": [10.0, 10.1, 10.25, 10.35, 10.45],
            "volume": [1000, 1200, 1300, 1400, 1500],
            "amount": [10000, 12000, 13000, 14000, 15000],
        },
        index=minute_index,
    )

    features = builder._build_minute_features("000001", minute_df)

    assert len(features) == 1
    feature_row = features.iloc[0]
    numeric_values = pd.to_numeric(feature_row[SequenceFeatureBuilder.MINUTE_FEATURES], errors="raise").to_numpy(dtype=float)
    assert np.isfinite(numeric_values).all()
    assert feature_row["intraday_return"] == 0.0
    assert feature_row["intraday_range"] == 0.0
