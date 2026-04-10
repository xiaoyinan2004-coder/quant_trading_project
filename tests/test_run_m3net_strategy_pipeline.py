from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_m3net_strategy_pipeline import (
    _build_eval_command,
    _build_export_command,
    _build_train_command,
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


def test_build_train_command_uses_checkpoint_output() -> None:
    args = _make_args()

    command = _build_train_command(args, Path("artifacts/checkpoints/model.pt"))

    assert "--output" in command
    output_index = command.index("--output") + 1
    assert Path(command[output_index]) == Path("artifacts/checkpoints/model.pt")
    assert "--graph-contrastive-loss-weight" in command


def test_build_eval_command_points_to_best_checkpoint() -> None:
    args = _make_args()

    command = _build_eval_command(args, Path("artifacts/checkpoints/model.pt"), Path("artifacts/eval/run"))

    checkpoint_index = command.index("--checkpoint") + 1
    assert Path(command[checkpoint_index]) == Path("artifacts/checkpoints/model.best.pt")
    assert "--include-risk-reranker" in command
    assert "--include-topk-reranker" in command


def test_build_export_command_uses_eval_dir() -> None:
    command = _build_export_command(Path("artifacts/eval/run"))

    assert command[-2] == "--eval-dir"
    assert Path(command[-1]) == Path("artifacts/eval/run")


def test_resolve_minute_dir_discovers_sibling_folder(tmp_path: Path) -> None:
    daily_dir = tmp_path / "daily"
    minute_dir = tmp_path / "minute"
    daily_dir.mkdir()
    minute_dir.mkdir()
    (minute_dir / "000001.csv").write_text("datetime,open\n2026-01-01 09:30:00,10\n", encoding="utf-8")

    resolved = _resolve_minute_dir(str(daily_dir), None)

    assert Path(resolved) == minute_dir
