from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.run_m3net_backbone_stability_check import (
    _build_stability_summary,
    _build_train_command,
    _collect_seed_run,
    _compute_max_drawdown,
    _resolve_minute_dir,
)


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        daily_dir="data/daily",
        minute_dir=None,
        device="cuda",
        min_stock_rows=1000,
        rebalance_freq="M",
        train_lookback_periods=36,
        eval_lookback_periods=12,
        epochs=20,
        patience=4,
        top_n=20,
        top_k_list=[5, 10, 20],
        rank_loss_weight=0.45,
        listwise_loss_weight=0.25,
        top_pick_loss_weight=0.08,
        weighted_return_alpha=1.0,
        top_pick_quantile=0.9,
        listwise_topk_focus=5,
        graph_neighbor_k=8,
        graph_residual_weight=0.4,
        graph_contrastive_loss_weight=0.08,
        graph_contrastive_neighbors=6,
        reranker_train_min_periods=4,
        reranker_candidate_pool=20,
        topk_reranker_top_k_target=5,
        topk_reranker_epochs=120,
        topk_reranker_learning_rate=1e-3,
        risk_reranker_downside_penalty=1.5,
        risk_reranker_downside_power=1.5,
        risk_reranker_inference_risk_weight=0.35,
        baseline_backend="lightgbm",
        baseline_train_lookback_days=504,
    )


def test_build_train_command_includes_random_state() -> None:
    args = _make_args()

    command = _build_train_command(args, seed=21, checkpoint_path=Path("artifacts/checkpoints/model.pt"))

    seed_index = command.index("--random-state") + 1
    assert command[seed_index] == "21"


def test_compute_max_drawdown_for_stability_helper() -> None:
    series = pd.Series([0.1, -0.05, -0.1, 0.02])

    result = _compute_max_drawdown(series)

    assert round(result, 6) == -0.145000


def test_collect_seed_run_extracts_m3net_full_rows(tmp_path: Path) -> None:
    compare_summary = pd.DataFrame(
        [
            {"model": "m3net_full", "top_k": 5, "periods": 6, "avg_realized_return": 0.02, "win_rate": 0.7, "cumulative_return_proxy": 0.12},
            {"model": "m3net_reranker", "top_k": 5, "periods": 6, "avg_realized_return": 0.01, "win_rate": 0.4, "cumulative_return_proxy": 0.05},
        ]
    )
    rolling_periods = pd.DataFrame(
        [
            {"model": "m3net_full", "top_k": 5, "trade_date": "2026-01-31", "avg_realized_return": 0.10},
            {"model": "m3net_full", "top_k": 5, "trade_date": "2026-02-28", "avg_realized_return": -0.05},
            {"model": "m3net_full", "top_k": 5, "trade_date": "2026-03-31", "avg_realized_return": -0.10},
        ]
    )
    compare_path = tmp_path / "compare_summary.csv"
    rolling_path = tmp_path / "rolling_periods.csv"
    compare_summary.to_csv(compare_path, index=False)
    rolling_periods.to_csv(rolling_path, index=False)

    seed_run = _collect_seed_run(compare_path, rolling_path, seed=42, models=["m3net_full"])

    assert seed_run.iloc[0]["seed"] == 42
    assert seed_run.iloc[0]["model"] == "m3net_full"
    assert seed_run.iloc[0]["top_k"] == 5
    assert round(float(seed_run.iloc[0]["max_drawdown"]), 6) == -0.145000


def test_build_stability_summary_aggregates_seed_runs() -> None:
    seed_runs = pd.DataFrame(
        [
            {"seed": 7, "model": "m3net_full", "top_k": 5, "avg_realized_return": 0.01, "win_rate": 0.5, "cumulative_return_proxy": 0.05, "max_drawdown": -0.03},
            {"seed": 21, "model": "m3net_full", "top_k": 5, "avg_realized_return": 0.03, "win_rate": 0.8, "cumulative_return_proxy": 0.12, "max_drawdown": -0.02},
        ]
    )

    summary = _build_stability_summary(seed_runs)

    assert summary.iloc[0]["model"] == "m3net_full"
    assert summary.iloc[0]["num_seeds"] == 2
    assert round(float(summary.iloc[0]["mean_avg_realized_return"]), 6) == 0.020000
    assert round(float(summary.iloc[0]["worst_max_drawdown"]), 6) == -0.030000


def test_resolve_minute_dir_discovers_sibling_folder(tmp_path: Path) -> None:
    daily_dir = tmp_path / "daily"
    minute_dir = tmp_path / "minute"
    daily_dir.mkdir()
    minute_dir.mkdir()
    (minute_dir / "600000.csv").write_text("datetime,open\n2026-01-01 09:30:00,10\n", encoding="utf-8")

    resolved = _resolve_minute_dir(str(daily_dir), None)

    assert Path(resolved) == minute_dir
