"""Run multi-seed stability checks for the M3Net backbone and reranker stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_m3net_full.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_m3net_full.py"

DEFAULT_MODELS = [
    "m3net_full",
    "m3net_reranker",
    "m3net_risk_reranker",
    "m3net_topk_reranker",
]

import pandas as pd


def _compute_max_drawdown(period_returns: pd.Series) -> float:
    clean_returns = pd.to_numeric(period_returns, errors="coerce").dropna()
    if clean_returns.empty:
        return float("nan")
    equity_curve = (1.0 + clean_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def _append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _run_command(command: list[str]) -> None:
    print("[stability] running:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _build_train_command(args: argparse.Namespace, seed: int, checkpoint_path: Path) -> list[str]:
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
        "--random-state",
        str(seed),
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


def _collect_seed_run(compare_summary_path: Path, rolling_periods_path: Path, seed: int, models: list[str]) -> pd.DataFrame:
    compare_summary = pd.read_csv(compare_summary_path)
    periods = pd.read_csv(rolling_periods_path)

    rows: list[dict[str, object]] = []
    summary_subset = compare_summary.loc[compare_summary["model"].isin(models)].copy()
    periods_subset = periods.loc[periods["model"].isin(models)].copy()
    for _, row in summary_subset.iterrows():
        model_name = str(row["model"])
        top_k = int(row["top_k"])
        top_periods = periods_subset.loc[
            (periods_subset["model"] == model_name)
            & (pd.to_numeric(periods_subset["top_k"], errors="coerce") == top_k)
        ]
        rows.append(
            {
                "seed": int(seed),
                "model": model_name,
                "top_k": top_k,
                "periods": int(row["periods"]),
                "avg_realized_return": float(row["avg_realized_return"]),
                "win_rate": float(row["win_rate"]),
                "cumulative_return_proxy": float(row["cumulative_return_proxy"]),
                "max_drawdown": _compute_max_drawdown(top_periods["avg_realized_return"]),
            }
        )
    return pd.DataFrame(rows)


def _build_stability_summary(seed_runs: pd.DataFrame) -> pd.DataFrame:
    if seed_runs.empty:
        return pd.DataFrame(
            columns=[
                "top_k",
                "model",
                "num_seeds",
                "mean_avg_realized_return",
                "std_avg_realized_return",
                "min_avg_realized_return",
                "max_avg_realized_return",
                "mean_win_rate",
                "mean_max_drawdown",
                "worst_max_drawdown",
            ]
        )

    rows: list[dict[str, object]] = []
    for (model_name, top_k), group in seed_runs.groupby(["model", "top_k"], dropna=False):
        rows.append(
            {
                "model": model_name,
                "top_k": int(top_k),
                "num_seeds": int(group["seed"].nunique()),
                "mean_avg_realized_return": float(group["avg_realized_return"].mean()),
                "std_avg_realized_return": float(group["avg_realized_return"].std(ddof=0)),
                "min_avg_realized_return": float(group["avg_realized_return"].min()),
                "max_avg_realized_return": float(group["avg_realized_return"].max()),
                "mean_win_rate": float(group["win_rate"].mean()),
                "mean_max_drawdown": float(group["max_drawdown"].mean()),
                "worst_max_drawdown": float(group["max_drawdown"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "top_k"]).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed stability checks for M3Net and rerankers.")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-root", default="artifacts/m3net_backbone_stability")
    parser.add_argument("--run-name", default="graph_contrastive")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    eval_root = output_root / "eval"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    collected_runs: list[pd.DataFrame] = []
    for seed in args.seeds:
        run_name = f"{args.run_name}_seed{seed}"
        checkpoint_path = checkpoint_dir / f"m3net_full_{run_name}.pt"
        best_checkpoint = checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")
        eval_dir = eval_root / run_name
        compare_summary_path = eval_dir / "compare_summary.csv"
        rolling_periods_path = eval_dir / "rolling_periods.csv"

        if not (args.skip_existing and best_checkpoint.exists()):
            _run_command(_build_train_command(args, seed, checkpoint_path))
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found for seed {seed}: {best_checkpoint}")

        if not (args.skip_existing and compare_summary_path.exists() and rolling_periods_path.exists()):
            _run_command(_build_eval_command(args, checkpoint_path, eval_dir))

        collected_runs.append(_collect_seed_run(compare_summary_path, rolling_periods_path, seed, args.models))

    seed_runs = pd.concat(collected_runs, ignore_index=True) if collected_runs else pd.DataFrame()
    stability_summary = _build_stability_summary(seed_runs)
    seed_runs.to_csv(output_root / "seed_runs.csv", index=False)
    stability_summary.to_csv(output_root / "stability_summary.csv", index=False)

    print(f"Saved seed runs to: {output_root / 'seed_runs.csv'}")
    print(f"Saved stability summary to: {output_root / 'stability_summary.csv'}")
    if not stability_summary.empty:
        print(stability_summary.to_string(index=False))


if __name__ == "__main__":
    main()
