from __future__ import annotations

import pandas as pd

from scripts.export_m3net_paper_trade_plan import _build_trade_plan


def test_build_trade_plan_filters_profile_and_assigns_equal_weights() -> None:
    profile_summary = pd.DataFrame(
        [
            {
                "profile": "default",
                "top_k": 5,
                "model": "m3net_risk_reranker",
                "periods": 18,
                "avg_realized_return": 0.02,
                "win_rate": 0.7,
                "cumulative_return_proxy": 0.4,
            },
            {
                "profile": "default",
                "top_k": 20,
                "model": "m3net_full",
                "periods": 18,
                "avg_realized_return": 0.015,
                "win_rate": 0.6,
                "cumulative_return_proxy": 0.3,
            },
        ]
    )
    profile_risk_summary = pd.DataFrame(
        [
            {
                "profile": "default",
                "top_k": 5,
                "model": "m3net_risk_reranker",
                "periods": 18,
                "max_drawdown": -0.03,
                "worst_period_return": -0.02,
            },
            {
                "profile": "default",
                "top_k": 20,
                "model": "m3net_full",
                "periods": 18,
                "max_drawdown": -0.05,
                "worst_period_return": -0.04,
            },
        ]
    )
    profile_latest_picks = pd.DataFrame(
        [
            {"profile": "default", "model": "m3net_risk_reranker", "top_k": 5, "trade_date": "2026-04-10", "symbol": "A", "score": 0.9},
            {"profile": "default", "model": "m3net_risk_reranker", "top_k": 5, "trade_date": "2026-04-10", "symbol": "B", "score": 0.8},
            {"profile": "default", "model": "m3net_full", "top_k": 20, "trade_date": "2026-04-10", "symbol": "C", "score": 0.7},
            {"profile": "default", "model": "m3net_full", "top_k": 20, "trade_date": "2026-04-10", "symbol": "D", "score": 0.6},
            {"profile": "aggressive", "model": "m3net_reranker", "top_k": 5, "trade_date": "2026-04-10", "symbol": "X", "score": 1.0},
        ]
    )

    snapshot, plan = _build_trade_plan(
        profile_summary=profile_summary,
        profile_risk_summary=profile_risk_summary,
        profile_latest_picks=profile_latest_picks,
        profile_name="default",
    )

    assert list(snapshot["top_k"]) == [5, 20]
    assert set(plan["profile"]) == {"default"}
    assert set(plan["top_k"]) == {5, 20}
    assert float(plan.loc[(plan["top_k"] == 5) & (plan["symbol"] == "A"), "target_weight"].iloc[0]) == 0.5
    assert float(plan.loc[(plan["top_k"] == 20) & (plan["symbol"] == "C"), "target_weight"].iloc[0]) == 0.5


def test_build_trade_plan_raises_for_missing_profile() -> None:
    profile_summary = pd.DataFrame(
        [
            {"profile": "default", "top_k": 5, "model": "m3net_risk_reranker"},
        ]
    )
    empty = pd.DataFrame()

    try:
        _build_trade_plan(
            profile_summary=profile_summary,
            profile_risk_summary=empty,
            profile_latest_picks=empty,
            profile_name="missing",
        )
    except ValueError as exc:
        assert "Profile 'missing' not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing profile")
