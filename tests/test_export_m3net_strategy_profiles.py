from __future__ import annotations

import pandas as pd

from scripts.export_m3net_strategy_profiles import (
    _build_auto_best_profile,
    _build_profile_latest_picks,
    _build_profile_periods,
    _build_profile_risk_summary,
    _build_profile_summary,
    _compute_max_drawdown,
)


def test_build_profile_summary_uses_selected_models() -> None:
    compare_summary = pd.DataFrame(
        [
            {"model": "m3net_reranker", "top_k": 5, "periods": 6, "avg_realized_return": 0.02, "win_rate": 0.8, "cumulative_return_proxy": 0.12},
            {"model": "m3net_risk_reranker", "top_k": 5, "periods": 6, "avg_realized_return": 0.015, "win_rate": 0.9, "cumulative_return_proxy": 0.10},
            {"model": "m3net_full", "top_k": 20, "periods": 6, "avg_realized_return": 0.018, "win_rate": 0.7, "cumulative_return_proxy": 0.11},
        ]
    )
    profiles = {
        "aggressive": {5: "m3net_reranker", 20: "m3net_full"},
        "risk_aware": {5: "m3net_risk_reranker", 20: "m3net_full"},
    }

    summary = _build_profile_summary(compare_summary, profiles)

    assert set(summary["profile"]) == {"aggressive", "risk_aware"}
    assert set(summary["top_k"]) == {5, 20}
    assert "avg_realized_return" in summary.columns


def test_build_profile_periods_filters_to_profile_models() -> None:
    aligned_periods = pd.DataFrame(
        [
            {"model": "m3net_reranker", "top_k": 5, "trade_date": "2026-01-31", "avg_realized_return": 0.01},
            {"model": "m3net_risk_reranker", "top_k": 5, "trade_date": "2026-01-31", "avg_realized_return": 0.02},
            {"model": "m3net_full", "top_k": 20, "trade_date": "2026-01-31", "avg_realized_return": 0.03},
        ]
    )
    profiles = {
        "aggressive": {5: "m3net_reranker", 20: "m3net_full"},
    }

    periods = _build_profile_periods(aligned_periods, profiles)

    assert set(periods["profile"]) == {"aggressive"}
    assert set(periods["model"]) == {"m3net_reranker", "m3net_full"}


def test_build_profile_latest_picks_uses_latest_trade_date() -> None:
    rolling_picks = pd.DataFrame(
        [
            {"model": "m3net_reranker", "top_k": 5, "trade_date": "2026-01-31", "symbol": "A", "score": 0.5},
            {"model": "m3net_reranker", "top_k": 5, "trade_date": "2026-02-28", "symbol": "B", "score": 0.7},
            {"model": "m3net_full", "top_k": 20, "trade_date": "2026-02-28", "symbol": "C", "score": 0.3},
        ]
    )
    profiles = {
        "aggressive": {5: "m3net_reranker", 20: "m3net_full"},
    }

    latest = _build_profile_latest_picks(rolling_picks, profiles)

    assert set(latest["trade_date"]) == {"2026-02-28"}
    assert set(latest["profile"]) == {"aggressive"}


def test_compute_max_drawdown_matches_expected_path() -> None:
    returns = pd.Series([0.10, -0.05, -0.10, 0.03])

    max_drawdown = _compute_max_drawdown(returns)

    assert round(max_drawdown, 6) == -0.145000


def test_build_profile_risk_summary_adds_drawdown_metrics() -> None:
    profile_periods = pd.DataFrame(
        [
            {"profile": "aggressive", "model": "m3net_reranker", "top_k": 5, "trade_date": "2026-01-31", "avg_realized_return": 0.10},
            {"profile": "aggressive", "model": "m3net_reranker", "top_k": 5, "trade_date": "2026-02-28", "avg_realized_return": -0.05},
            {"profile": "aggressive", "model": "m3net_reranker", "top_k": 5, "trade_date": "2026-03-31", "avg_realized_return": -0.10},
        ]
    )

    risk_summary = _build_profile_risk_summary(profile_periods)

    assert set(risk_summary.columns) >= {"max_drawdown", "worst_period_return"}
    row = risk_summary.iloc[0]
    assert row["profile"] == "aggressive"
    assert row["top_k"] == 5
    assert round(float(row["max_drawdown"]), 6) == -0.145000
    assert float(row["worst_period_return"]) == -0.10


def test_build_auto_best_profile_selects_best_model_per_top_k() -> None:
    compare_summary = pd.DataFrame(
        [
            {"model": "m3net_full", "top_k": 5, "avg_realized_return": 0.02, "win_rate": 0.6, "cumulative_return_proxy": 0.12},
            {"model": "m3net_reranker", "top_k": 5, "avg_realized_return": 0.01, "win_rate": 0.9, "cumulative_return_proxy": 0.11},
            {"model": "m3net_topk_reranker", "top_k": 10, "avg_realized_return": 0.03, "win_rate": 0.7, "cumulative_return_proxy": 0.16},
            {"model": "m3net_full", "top_k": 10, "avg_realized_return": 0.02, "win_rate": 0.8, "cumulative_return_proxy": 0.15},
        ]
    )

    auto_best = _build_auto_best_profile(compare_summary)

    assert auto_best == {5: "m3net_full", 10: "m3net_topk_reranker"}
