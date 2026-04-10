"""Run the full M3Net strategy research pipeline end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_m3net_full.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_m3net_full.py"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_m3net_strategy_profiles.py"


def _append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_bool_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _build_train_command(args: argparse.Namespace, checkpoint_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--daily-dir",
        args.daily_dir,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--min-stock-rows",
        str(args.min_stock_rows),
        "--rebalance-freq",
        args.rebalance_freq,
        "--train-lookback-periods",
        str(args.train_lookback_periods),
        "--rank-loss-weight",
        str(args.rank_loss_weight),
        "--listwise-loss-weight",
        str(args.listwise_loss_weight),
        "--top-pick-loss-weight",
        str(args.top_pick_loss_weight),
        "--weighted-return-alpha",
        str(args.weighted_return_alpha),
        "--top-pick-quantile",
        str(args.top_pick_quantile),
        "--listwise-topk-focus",
        str(args.listwise_topk_focus),
        "--graph-neighbor-k",
        str(args.graph_neighbor_k),
        "--graph-residual-weight",
        str(args.graph_residual_weight),
        "--graph-contrastive-loss-weight",
        str(args.graph_contrastive_loss_weight),
        "--graph-contrastive-neighbors",
        str(args.graph_contrastive_neighbors),
        "--output",
        str(checkpoint_path),
    ]
    _append_optional_arg(command, "--minute-dir", args.minute_dir)
    return command


def _build_eval_command(args: argparse.Namespace, checkpoint_path: Path, eval_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")),
        "--daily-dir",
        args.daily_dir,
        "--device",
        args.device,
        "--min-stock-rows",
        str(args.min_stock_rows),
        "--rebalance-freq",
        args.rebalance_freq,
        "--eval-lookback-periods",
        str(args.eval_lookback_periods),
        "--top-n",
        str(args.top_n),
        "--top-k-list",
        *[str(value) for value in args.top_k_list],
        "--include-baseline",
        "--include-reranker",
        "--include-topk-reranker",
        "--include-risk-reranker",
        "--reranker-train-min-periods",
        str(args.reranker_train_min_periods),
        "--reranker-candidate-pool",
        str(args.reranker_candidate_pool),
        "--topk-reranker-top-k-target",
        str(args.topk_reranker_top_k_target),
        "--topk-reranker-epochs",
        str(args.topk_reranker_epochs),
        "--topk-reranker-learning-rate",
        str(args.topk_reranker_learning_rate),
        "--risk-reranker-downside-penalty",
        str(args.risk_reranker_downside_penalty),
        "--risk-reranker-downside-power",
        str(args.risk_reranker_downside_power),
        "--risk-reranker-inference-risk-weight",
        str(args.risk_reranker_inference_risk_weight),
        "--baseline-backend",
        args.baseline_backend,
        "--baseline-train-lookback-days",
        str(args.baseline_train_lookback_days),
        "--output-dir",
        str(eval_dir),
    ]
    _append_optional_arg(command, "--minute-dir", args.minute_dir)
    return command


def _build_export_command(eval_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--eval-dir",
        str(eval_dir),
    ]


def _run_command(command: list[str]) -> None:
    print("[pipeline] running:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full M3Net strategy pipeline end-to-end.")
    parser.add_argument("--daily-dir", required=True)
    parser.add_argument("--minute-dir")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--train-lookback-periods", type=int, default=36)
    parser.add_argument("--eval-lookback-periods", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-k-list", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--rank-loss-weight", type=float, default=0.45)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.25)
    parser.add_argument("--top-pick-loss-weight", type=float, default=0.08)
    parser.add_argument("--weighted-return-alpha", type=float, default=1.0)
    parser.add_argument("--top-pick-quantile", type=float, default=0.9)
    parser.add_argument("--listwise-topk-focus", type=int, default=5)
    parser.add_argument("--graph-neighbor-k", type=int, default=8)
    parser.add_argument("--graph-residual-weight", type=float, default=0.4)
    parser.add_argument("--graph-contrastive-loss-weight", type=float, default=0.08)
    parser.add_argument("--graph-contrastive-neighbors", type=int, default=6)
    parser.add_argument("--reranker-train-min-periods", type=int, default=4)
    parser.add_argument("--reranker-candidate-pool", type=int, default=20)
    parser.add_argument("--topk-reranker-top-k-target", type=int, default=5)
    parser.add_argument("--topk-reranker-epochs", type=int, default=120)
    parser.add_argument("--topk-reranker-learning-rate", type=float, default=1e-3)
    parser.add_argument("--risk-reranker-downside-penalty", type=float, default=1.5)
    parser.add_argument("--risk-reranker-downside-power", type=float, default=1.5)
    parser.add_argument("--risk-reranker-inference-risk-weight", type=float, default=0.35)
    parser.add_argument("--baseline-backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--baseline-train-lookback-days", type=int, default=504)
    parser.add_argument("--output-root", default="artifacts/m3net_strategy_pipeline")
    parser.add_argument("--run-name", default="graph_contrastive")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    eval_dir = output_root / "eval" / args.run_name
    checkpoint_path = checkpoint_dir / f"m3net_full_{args.run_name}.pt"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        _run_command(_build_train_command(args, checkpoint_path))

    best_checkpoint = checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")

    if not args.skip_eval:
        _run_command(_build_eval_command(args, checkpoint_path, eval_dir))

    compare_summary = eval_dir / "compare_summary.csv"
    if not args.skip_export and not compare_summary.exists():
        raise FileNotFoundError(f"Evaluation output not found: {compare_summary}")

    if not args.skip_export:
        _run_command(_build_export_command(eval_dir))

    print(f"[pipeline] checkpoint dir: {checkpoint_dir}")
    print(f"[pipeline] eval dir: {eval_dir}")


if __name__ == "__main__":
    main()
