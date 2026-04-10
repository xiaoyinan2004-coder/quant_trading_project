from __future__ import annotations
import pandas as pd
import pytest

from scripts.evaluate_m3net_full import (
    _align_common_valid_periods,
    _build_baseline_ranked_frame,
    _build_ensemble_ranked_frame,
    _build_m3net_ranked_frame,
    _build_reranker_candidate_frame,
    _build_reranker_ranked_frame,
    _build_reranker_training_rows,
    _build_risk_aware_reranker_ranked_frame,
    _build_standardized_ensemble_ranked_frame,
    _build_topk_reranker_ranked_frame,
    _compute_dynamic_baseline_weight,
    _parse_top_k_list,
    _standardize_series,
    _summarize_periods,
)
from scripts.evaluate_m3net_stage1 import _filter_stock_data_by_min_rows, _slice_daily_history
from scripts.run_m3net_research import EXPERIMENT_NAMES, _build_rebalance_dates, _portfolio_summary, _resolve_rebalance_dates
from test_gradient_boosting_factor import make_synthetic_stock_data


def test_build_rebalance_dates_returns_period_end_dates() -> None:
    stock_data = {
        "000001": pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.to_datetime(["2026-01-05", "2026-01-30", "2026-02-27"]),
        ),
        "000002": pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.to_datetime(["2026-01-06", "2026-01-29", "2026-02-26"]),
        ),
    }

    dates = _build_rebalance_dates(stock_data, "M")

    assert dates == [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-27")]


def test_portfolio_summary_handles_empty_and_non_empty_returns() -> None:
    empty = _portfolio_summary([])
    assert empty["periods"] == 0

    summary = _portfolio_summary([0.1, -0.05, 0.02])
    assert summary["periods"] == 3
    assert abs(summary["avg_return"] - 0.0233333333) < 1e-6
    assert 0.0 <= summary["win_rate"] <= 1.0


def test_filter_stock_data_by_min_rows_removes_short_histories() -> None:
    stock_data = {
        "000001": pd.DataFrame({"close": [1, 2, 3]}),
        "000002": pd.DataFrame({"close": list(range(10))}),
    }

    filtered = _filter_stock_data_by_min_rows(stock_data, 5)

    assert list(filtered.keys()) == ["000002"]


def test_experiment_names_cover_expected_research_variants() -> None:
    assert set(EXPERIMENT_NAMES) == {
        "baseline_factor",
        "m3net_v2_lite",
        "m3net_no_learned_router",
        "m3net_no_minute",
    }


def test_slice_daily_history_respects_max_history_days() -> None:
    dates = pd.date_range("2025-01-01", periods=12, freq="B")
    stock_data = {
        "000001": pd.DataFrame({"close": range(12)}, index=dates),
    }

    sliced = _slice_daily_history(
        stock_data,
        end_date=dates[-1],
        min_history=5,
        max_history_days=5,
    )

    assert len(sliced["000001"]) == 5
    assert sliced["000001"].index.min() == dates[-5]


def test_resolve_rebalance_dates_supports_single_trade_date() -> None:
    stock_data = {
        "000001": pd.DataFrame({"close": [1, 2, 3]}, index=pd.to_datetime(["2026-01-05", "2026-01-30", "2026-02-27"])),
    }

    dates = _resolve_rebalance_dates(stock_data, "M", single_trade_date="2026-03-31")

    assert dates == [pd.Timestamp("2026-03-31")]


def test_parse_top_k_list_includes_top_n_and_sorts() -> None:
    assert _parse_top_k_list(20, [10, 5, 20, 10]) == [5, 10, 20]
    assert _parse_top_k_list(20, None) == [20]


def test_summarize_periods_groups_by_model_and_top_k() -> None:
    periods = pd.DataFrame(
        [
            {"model": "m3net_full", "top_k": 5, "avg_realized_return": 0.01},
            {"model": "m3net_full", "top_k": 5, "avg_realized_return": -0.02},
            {"model": "baseline_factor", "top_k": 20, "avg_realized_return": 0.03},
        ]
    )

    summary = _summarize_periods(periods)

    assert set(summary["model"]) == {"m3net_full", "baseline_factor"}
    assert set(summary["top_k"]) == {5, 20}
    assert "cumulative_return_proxy" in summary.columns


def test_build_baseline_ranked_frame_scores_latest_trade_date() -> None:
    stock_data = make_synthetic_stock_data(symbols=6, periods=220)
    trade_date = max(frame.index.max() for frame in stock_data.values())

    ranked = _build_baseline_ranked_frame(
        stock_data=stock_data,
        as_of_date=trade_date,
        backend="lightgbm",
        label_horizon=5,
        top_n=3,
    )

    assert len(ranked) == 3
    assert ranked["score"].notna().all()


def test_align_common_valid_periods_drops_tail_nan_and_keeps_common_dates() -> None:
    periods = pd.DataFrame(
        [
            {"model": "baseline_factor", "top_k": 20, "trade_date": "2026-01-31", "avg_realized_return": 0.01},
            {"model": "m3net_full", "top_k": 20, "trade_date": "2026-01-31", "avg_realized_return": 0.02},
            {"model": "baseline_factor", "top_k": 20, "trade_date": "2026-02-28", "avg_realized_return": 0.03},
            {"model": "m3net_full", "top_k": 20, "trade_date": "2026-02-28", "avg_realized_return": float("nan")},
        ]
    )

    aligned = _align_common_valid_periods(periods)

    assert len(aligned) == 2
    assert set(aligned["trade_date"]) == {"2026-01-31"}


def test_build_ensemble_ranked_frame_returns_ranked_union() -> None:
    baseline_ranked = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.1],
            "pred_return": [0.9, 0.8, 0.1],
            "pred_risk": [float("nan")] * 3,
            "confidence": [float("nan")] * 3,
        }
    )
    m3net_ranked = pd.DataFrame(
        {
            "symbol": ["B", "C", "D"],
            "score": [0.2, 0.7, 0.95],
            "pred_return": [0.02, 0.07, 0.095],
            "pred_risk": [0.1, 0.2, 0.15],
            "confidence": [0.6, 0.7, 0.9],
            "top_pick_prob": [0.5, 0.6, 0.95],
        }
    )

    ranked = _build_ensemble_ranked_frame(
        baseline_ranked=baseline_ranked,
        m3net_ranked=m3net_ranked,
        baseline_weight=0.5,
        top_n=3,
    )

    assert len(ranked) == 3
    assert set(ranked["symbol"]).issubset({"A", "B", "C", "D"})
    assert ranked["score"].is_monotonic_decreasing


def test_standardize_series_returns_finite_values() -> None:
    series = pd.Series([1.0, 2.0, float("nan"), 4.0, 4.0])

    zscore = _standardize_series(series, method="zscore")
    robust = _standardize_series(series, method="robust_zscore")
    percentile = _standardize_series(series, method="percentile")

    assert zscore.notna().all()
    assert robust.notna().all()
    assert percentile.notna().all()


def test_build_standardized_ensemble_ranked_frame_returns_ranked_union() -> None:
    baseline_ranked = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.1],
            "pred_return": [0.9, 0.8, 0.1],
            "pred_risk": [float("nan")] * 3,
            "confidence": [float("nan")] * 3,
        }
    )
    m3net_ranked = pd.DataFrame(
        {
            "symbol": ["B", "C", "D"],
            "score": [0.2, 0.7, 0.95],
            "pred_return": [0.02, 0.07, 0.095],
            "pred_risk": [0.1, 0.2, 0.15],
            "confidence": [0.6, 0.7, 0.9],
            "top_pick_prob": [0.5, 0.6, 0.95],
        }
    )

    ranked = _build_standardized_ensemble_ranked_frame(
        baseline_ranked=baseline_ranked,
        m3net_ranked=m3net_ranked,
        baseline_weight=0.2,
        top_n=3,
        method="zscore",
    )

    assert len(ranked) == 3
    assert set(ranked["symbol"]).issubset({"A", "B", "C", "D"})
    assert ranked["score"].is_monotonic_decreasing


def test_build_m3net_ranked_frame_uses_top_pick_and_risk_components() -> None:
    class Config:
        score_return_weight = 1.0
        score_top_pick_weight = 0.6
        score_confidence_weight = 0.2
        score_risk_weight = 0.15

    ranked = _build_m3net_ranked_frame(
        symbols=["A", "B", "C"],
        pred_return=[0.05, 0.04, 0.03],
        pred_risk=[0.30, 0.05, 0.02],
        confidence=[0.7, 0.6, 0.5],
        top_pick_prob=[0.95, 0.2, 0.1],
        top_n=3,
        config=Config(),
    )

    assert list(ranked["symbol"])[:1] == ["A"]
    assert "top_pick_prob" in ranked.columns


def test_build_reranker_candidate_and_ranked_frames() -> None:
    m3net_ranked = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score": [0.8, 0.7, 0.5, 0.3],
            "pred_return": [0.06, 0.03, 0.01, -0.01],
            "pred_risk": [0.10, 0.05, 0.03, 0.02],
            "confidence": [0.9, 0.8, 0.5, 0.2],
            "top_pick_prob": [0.95, 0.4, 0.3, 0.1],
        }
    )
    baseline_ranked = pd.DataFrame(
        {
            "symbol": ["B", "A", "D", "C"],
            "score": [0.09, 0.08, 0.07, 0.01],
            "pred_return": [0.09, 0.08, 0.07, 0.01],
            "pred_risk": [float("nan")] * 4,
            "confidence": [float("nan")] * 4,
            "top_pick_prob": [float("nan")] * 4,
        }
    )

    candidate = _build_reranker_candidate_frame(m3net_ranked, baseline_ranked, top_n=4)
    assert len(candidate) == 4
    assert "baseline_score" in candidate.columns
    assert "score_gap_z" in candidate.columns

    history = [
        {
            "trade_date": "2026-01-31",
            "symbol": row["symbol"],
            "realized_return": float(value),
            **{col: float(row[col]) for col in candidate.columns if col in _build_reranker_candidate_frame(m3net_ranked, baseline_ranked, top_n=4).columns and col in [
                "m3net_score","baseline_score","pred_return","pred_risk","confidence","top_pick_prob",
                "m3net_rank_pct","baseline_rank_pct","pred_return_z","pred_risk_z","confidence_z","top_pick_z",
                "baseline_score_z","m3net_score_z","score_gap_z","return_confidence","top_pick_confidence"
            ]}
        }
        for row, value in zip(candidate.to_dict("records"), [0.05, 0.02, -0.01, -0.03], strict=False)
    ]
    history += [
        {
            "trade_date": "2026-02-28",
            "symbol": row["symbol"],
            "realized_return": float(value),
            **{col: float(row[col]) for col in candidate.columns if col in [
                "m3net_score","baseline_score","pred_return","pred_risk","confidence","top_pick_prob",
                "m3net_rank_pct","baseline_rank_pct","pred_return_z","pred_risk_z","confidence_z","top_pick_z",
                "baseline_score_z","m3net_score_z","score_gap_z","return_confidence","top_pick_confidence"
            ]}
        }
        for row, value in zip(candidate.to_dict("records"), [0.04, 0.01, 0.00, -0.02], strict=False)
    ]

    reranked, periods = _build_reranker_ranked_frame(candidate, history, train_min_periods=2, top_n=3)
    assert reranked is not None
    assert periods == 2
    assert len(reranked) == 3
    assert reranked["score"].is_monotonic_decreasing


def test_build_reranker_training_rows_extracts_realized_returns() -> None:
    stock_data = make_synthetic_stock_data(symbols=6, periods=220)
    trade_date = max(frame.index[-6] for frame in stock_data.values())
    symbols = list(stock_data.keys())[:3]
    candidate = pd.DataFrame(
        {
            "symbol": symbols,
            "score": [0.8, 0.7, 0.6],
            "baseline_score": [0.1, 0.2, 0.3],
            "pred_return": [0.05, 0.04, 0.03],
            "pred_risk": [0.1, 0.1, 0.1],
            "confidence": [0.8, 0.7, 0.6],
            "top_pick_prob": [0.9, 0.3, 0.2],
            "m3net_rank_pct": [1/3, 2/3, 1.0],
            "baseline_rank_pct": [1/3, 2/3, 1.0],
            "pred_return_z": [1.0, 0.0, -1.0],
            "pred_risk_z": [0.0, 0.0, 0.0],
            "confidence_z": [1.0, 0.0, -1.0],
            "top_pick_z": [1.0, 0.0, -1.0],
            "baseline_score_z": [-1.0, 0.0, 1.0],
            "m3net_score_z": [1.0, 0.0, -1.0],
            "score_gap_z": [2.0, 0.0, -2.0],
            "return_confidence": [0.04, 0.028, 0.018],
            "top_pick_confidence": [0.72, 0.21, 0.12],
        }
    )

    rows = _build_reranker_training_rows(candidate, stock_data, trade_date, horizon=5)
    assert rows
    assert all(pd.notna(row["realized_return"]) for row in rows)


def test_build_risk_aware_reranker_ranked_frame_returns_scores() -> None:
    candidate = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score": [0.8, 0.7, 0.6, 0.5],
            "m3net_score": [0.8, 0.7, 0.6, 0.5],
            "baseline_score": [0.2, 0.3, 0.1, 0.0],
            "pred_return": [0.06, 0.04, 0.02, -0.01],
            "pred_risk": [0.10, 0.08, 0.04, 0.03],
            "confidence": [0.8, 0.7, 0.6, 0.4],
            "top_pick_prob": [0.9, 0.5, 0.3, 0.1],
            "m3net_rank_pct": [0.25, 0.5, 0.75, 1.0],
            "baseline_rank_pct": [0.5, 0.25, 0.75, 1.0],
            "pred_return_z": [1.2, 0.4, -0.2, -1.4],
            "pred_risk_z": [1.0, 0.4, -0.4, -1.0],
            "confidence_z": [1.0, 0.3, -0.3, -1.0],
            "top_pick_z": [1.1, 0.2, -0.4, -0.9],
            "baseline_score_z": [0.2, 0.6, -0.2, -0.6],
            "m3net_score_z": [1.0, 0.3, -0.3, -1.0],
            "score_gap_z": [0.8, -0.3, -0.1, -0.4],
            "return_confidence": [0.048, 0.028, 0.012, -0.004],
            "top_pick_confidence": [0.72, 0.35, 0.18, 0.04],
        }
    )
    training_rows = []
    for trade_date, realized in [
        ("2026-01-31", [0.05, -0.03, -0.01, -0.02]),
        ("2026-02-28", [0.04, 0.01, -0.05, -0.03]),
        ("2026-03-31", [0.06, 0.02, -0.02, -0.04]),
    ]:
        for row, ret in zip(candidate.to_dict("records"), realized, strict=False):
            training_rows.append({"trade_date": trade_date, "symbol": row["symbol"], "realized_return": ret, **{k: row[k] for k in candidate.columns if k != "symbol"}})

    ranked, periods = _build_risk_aware_reranker_ranked_frame(
        candidate_frame=candidate,
        training_rows=training_rows,
        train_min_periods=3,
        top_n=3,
        downside_penalty=1.5,
        downside_power=1.5,
        inference_risk_weight=0.35,
    )

    assert ranked is not None
    assert periods == 3
    assert len(ranked) == 3
    assert ranked["score"].is_monotonic_decreasing


def test_build_topk_reranker_ranked_frame_returns_scores_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    candidate = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score": [0.8, 0.7, 0.6, 0.5],
            "m3net_score": [0.8, 0.7, 0.6, 0.5],
            "baseline_score": [0.2, 0.3, 0.1, 0.0],
            "pred_return": [0.06, 0.04, 0.02, -0.01],
            "pred_risk": [0.10, 0.08, 0.04, 0.03],
            "confidence": [0.8, 0.7, 0.6, 0.4],
            "top_pick_prob": [0.9, 0.5, 0.3, 0.1],
            "m3net_rank_pct": [0.25, 0.5, 0.75, 1.0],
            "baseline_rank_pct": [0.5, 0.25, 0.75, 1.0],
            "pred_return_z": [1.2, 0.4, -0.2, -1.4],
            "pred_risk_z": [1.0, 0.4, -0.4, -1.0],
            "confidence_z": [1.0, 0.3, -0.3, -1.0],
            "top_pick_z": [1.1, 0.2, -0.4, -0.9],
            "baseline_score_z": [0.2, 0.6, -0.2, -0.6],
            "m3net_score_z": [1.0, 0.3, -0.3, -1.0],
            "score_gap_z": [0.8, -0.3, -0.1, -0.4],
            "return_confidence": [0.048, 0.028, 0.012, -0.004],
            "top_pick_confidence": [0.72, 0.35, 0.18, 0.04],
        }
    )
    training_rows = []
    for trade_date, realized in [
        ("2026-01-31", [0.05, 0.03, -0.01, -0.02]),
        ("2026-02-28", [0.04, 0.01, 0.00, -0.03]),
        ("2026-03-31", [0.06, 0.02, -0.02, -0.04]),
    ]:
        for row, ret in zip(candidate.to_dict("records"), realized, strict=False):
            training_rows.append({"trade_date": trade_date, "symbol": row["symbol"], "realized_return": ret, **{k: row[k] for k in candidate.columns if k != "symbol"}})

    ranked, periods = _build_topk_reranker_ranked_frame(
        candidate_frame=candidate,
        training_rows=training_rows,
        train_min_periods=3,
        top_n=3,
        top_k_target=2,
        device=torch.device("cpu"),
        epochs=10,
        learning_rate=1e-3,
    )

    assert ranked is not None
    assert periods == 3
    assert len(ranked) == 3
    assert ranked["score"].is_monotonic_decreasing


def test_compute_dynamic_baseline_weight_prefers_recent_stronger_model() -> None:
    periods = pd.DataFrame(
        [
            {"model": "baseline_factor", "top_k": 20, "trade_date": "2026-01-31", "avg_realized_return": 0.03},
            {"model": "m3net_full", "top_k": 20, "trade_date": "2026-01-31", "avg_realized_return": 0.01},
            {"model": "baseline_factor", "top_k": 20, "trade_date": "2026-02-28", "avg_realized_return": 0.02},
            {"model": "m3net_full", "top_k": 20, "trade_date": "2026-02-28", "avg_realized_return": 0.00},
        ]
    )

    weight = _compute_dynamic_baseline_weight(
        periods=periods,
        trade_date=pd.Timestamp("2026-03-31"),
        reference_top_k=20,
        lookback_periods=2,
    )

    assert 0.5 < weight <= 0.9
