"""Market and alpha memory used by the lightweight M3-Net router."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MemorySnapshot:
    """Daily regime summary used by the routing layer."""

    date: pd.Timestamp
    market_return_mean: float
    market_return_std: float
    positive_ratio: float
    score_dispersion: float


class MarketMemoryBank:
    """Store rolling regime information derived from model outputs."""

    def __init__(self) -> None:
        self._memory = pd.DataFrame()

    def fit(self, scored_panel: pd.DataFrame, label_col: str = "label") -> "MarketMemoryBank":
        frame = scored_panel.copy()
        if frame.empty:
            self._memory = pd.DataFrame(
                columns=[
                    "date",
                    "market_return_mean",
                    "market_return_std",
                    "positive_ratio",
                    "score_dispersion",
                ]
            )
            return self

        frame["date"] = pd.to_datetime(frame["date"])
        grouped = frame.groupby("date", as_index=False).agg(
            market_return_mean=(label_col, "mean"),
            market_return_std=(label_col, "std"),
            positive_ratio=(label_col, lambda s: float((s > 0).mean())),
            score_dispersion=("tabular_score", "std"),
        )
        grouped["market_return_std"] = grouped["market_return_std"].fillna(0.0)
        grouped["score_dispersion"] = grouped["score_dispersion"].fillna(0.0)
        self._memory = grouped.sort_values("date").reset_index(drop=True)
        return self

    def get_features(self, dates: pd.Series) -> pd.DataFrame:
        """Return routing features aligned to a vector of dates."""
        if self._memory.empty:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(dates),
                    "market_return_mean": 0.0,
                    "market_return_std": 0.0,
                    "positive_ratio": 0.5,
                    "score_dispersion": 0.0,
                }
            )

        lookup = self._memory.set_index("date")
        rows = []
        for date in pd.to_datetime(dates):
            if date in lookup.index:
                row = lookup.loc[date]
            else:
                history = lookup.loc[lookup.index <= date]
                if history.empty:
                    row = pd.Series(
                        {
                            "market_return_mean": 0.0,
                            "market_return_std": 0.0,
                            "positive_ratio": 0.5,
                            "score_dispersion": 0.0,
                        }
                    )
                else:
                    row = history.iloc[-1]
            rows.append(
                {
                    "date": date,
                    "market_return_mean": float(row["market_return_mean"]),
                    "market_return_std": float(row["market_return_std"]),
                    "positive_ratio": float(row["positive_ratio"]),
                    "score_dispersion": float(row["score_dispersion"]),
                }
            )
        return pd.DataFrame(rows)

