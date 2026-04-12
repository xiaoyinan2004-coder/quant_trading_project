from __future__ import annotations

import pandas as pd

from scripts.update_m3net_paper_trade_ledger import (
    _build_ledger_summary,
    _build_period_ledger,
    _build_profile_mapping,
    _build_trade_history,
)


def test_build_profile_mapping_reads_selected_profile() -> None:
    profile_summary = pd.DataFrame(
        [
            {"profile": "default", "top_k": 5, "model": "m3net_risk_reranker"},
            {"profile": "default", "top_k": 20, "model": "m3net_full"},
            {"profile": "aggressive", "top_k": 5, "model": "m3net_reranker"},
        ]
    )

    mapping = _build_profile_mapping(profile_summary, "default")

    assert mapping == {5: "m3net_risk_reranker", 20: "m3net_full"}


def test_build_trade_history_and_summary_compute_weighted_returns() -> None:
    rolling_picks = pd.DataFrame(
        [
            {"model": "m3net_risk_reranker", "top_k": 2, "trade_date": "2026-01-31", "symbol": "AAA", "score": 0.9},
            {"model": "m3net_risk_reranker", "top_k": 2, "trade_date": "2026-01-31", "symbol": "BBB", "score": 0.8},
            {"model": "m3net_full", "top_k": 3, "trade_date": "2026-01-31", "symbol": "AAA", "score": 0.7},
            {"model": "m3net_full", "top_k": 3, "trade_date": "2026-01-31", "symbol": "BBB", "score": 0.6},
            {"model": "m3net_full", "top_k": 3, "trade_date": "2026-01-31", "symbol": "CCC", "score": 0.5},
        ]
    )
    daily_data = {
        "AAA": pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=pd.to_datetime(["2026-01-31", "2026-02-03", "2026-02-04"])),
        "BBB": pd.DataFrame({"close": [20.0, 18.0, 18.0]}, index=pd.to_datetime(["2026-01-31", "2026-02-03", "2026-02-04"])),
        "CCC": pd.DataFrame({"close": [5.0, 5.5, 6.0]}, index=pd.to_datetime(["2026-01-31", "2026-02-03", "2026-02-04"])),
    }

    trade_history = _build_trade_history(
        rolling_picks=rolling_picks,
        profile_name="default",
        profile_mapping={2: "m3net_risk_reranker", 3: "m3net_full"},
        daily_data=daily_data,
        label_horizon=1,
    )
    period_ledger = _build_period_ledger(trade_history)
    summary = _build_ledger_summary(period_ledger)

    two_stock = period_ledger.loc[period_ledger["top_k"] == 2].iloc[0]
    three_stock = period_ledger.loc[period_ledger["top_k"] == 3].iloc[0]

    assert round(float(two_stock["portfolio_realized_return"]), 6) == 0.0
    assert round(float(three_stock["portfolio_realized_return"]), 6) == 0.033333
    assert set(summary["top_k"]) == {2, 3}
