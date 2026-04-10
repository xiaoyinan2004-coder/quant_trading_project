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

    def __init__(self, short_lookback: int = 20, long_lookback: int = 60) -> None:
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
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
                    "tabular_rank_ic_20d",
                    "sequence_rank_ic_20d",
                    "tabular_rank_ic_60d",
                    "sequence_rank_ic_60d",
                    "expert_ic_gap_20d",
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
        daily_ic = frame.groupby("date", as_index=False).apply(
            lambda group: pd.Series(
                {
                    "tabular_rank_ic": _safe_rank_ic(group[label_col], group["tabular_score"]),
                    "sequence_rank_ic": _safe_rank_ic(group[label_col], group["sequence_score"]),
                }
            )
        )
        grouped = grouped.merge(daily_ic, on="date", how="left")
        grouped["market_return_std"] = grouped["market_return_std"].fillna(0.0)
        grouped["score_dispersion"] = grouped["score_dispersion"].fillna(0.0)
        grouped["tabular_rank_ic"] = grouped["tabular_rank_ic"].fillna(0.0)
        grouped["sequence_rank_ic"] = grouped["sequence_rank_ic"].fillna(0.0)
        grouped = grouped.sort_values("date").reset_index(drop=True)
        grouped["tabular_rank_ic_20d"] = (
            grouped["tabular_rank_ic"].rolling(self.short_lookback, min_periods=3).mean().shift(1).fillna(0.0)
        )
        grouped["sequence_rank_ic_20d"] = (
            grouped["sequence_rank_ic"].rolling(self.short_lookback, min_periods=3).mean().shift(1).fillna(0.0)
        )
        grouped["tabular_rank_ic_60d"] = (
            grouped["tabular_rank_ic"].rolling(self.long_lookback, min_periods=5).mean().shift(1).fillna(0.0)
        )
        grouped["sequence_rank_ic_60d"] = (
            grouped["sequence_rank_ic"].rolling(self.long_lookback, min_periods=5).mean().shift(1).fillna(0.0)
        )
        grouped["expert_ic_gap_20d"] = grouped["sequence_rank_ic_20d"] - grouped["tabular_rank_ic_20d"]
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
                    "tabular_rank_ic_20d": 0.0,
                    "sequence_rank_ic_20d": 0.0,
                    "tabular_rank_ic_60d": 0.0,
                    "sequence_rank_ic_60d": 0.0,
                    "expert_ic_gap_20d": 0.0,
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
                            "tabular_rank_ic_20d": 0.0,
                            "sequence_rank_ic_20d": 0.0,
                            "tabular_rank_ic_60d": 0.0,
                            "sequence_rank_ic_60d": 0.0,
                            "expert_ic_gap_20d": 0.0,
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
                    "tabular_rank_ic_20d": float(row["tabular_rank_ic_20d"]),
                    "sequence_rank_ic_20d": float(row["sequence_rank_ic_20d"]),
                    "tabular_rank_ic_60d": float(row["tabular_rank_ic_60d"]),
                    "sequence_rank_ic_60d": float(row["sequence_rank_ic_60d"]),
                    "expert_ic_gap_20d": float(row["expert_ic_gap_20d"]),
                }
            )
        return pd.DataFrame(rows)


def _safe_rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat(
        [pd.to_numeric(y_true, errors="coerce"), pd.to_numeric(y_pred, errors="coerce")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return 0.0
    if aligned.iloc[:, 0].nunique() <= 1 or aligned.iloc[:, 1].nunique() <= 1:
        return 0.0
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    return float(0.0 if pd.isna(value) else value)
