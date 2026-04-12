"""Export a paper-trading plan from M3Net profile outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def _load_required_csv(eval_dir: Path, filename: str) -> pd.DataFrame:
    path = eval_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _build_trade_plan(
    profile_summary: pd.DataFrame,
    profile_risk_summary: pd.DataFrame,
    profile_latest_picks: pd.DataFrame,
    profile_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "profile" not in profile_risk_summary.columns:
        profile_risk_summary = pd.DataFrame(columns=["profile", "top_k", "model", "max_drawdown", "worst_period_return"])
    if "profile" not in profile_latest_picks.columns:
        profile_latest_picks = pd.DataFrame(columns=["profile", "model", "top_k", "trade_date", "symbol", "score"])

    summary = profile_summary.loc[profile_summary["profile"] == profile_name].copy()
    risk = profile_risk_summary.loc[profile_risk_summary["profile"] == profile_name].copy()
    latest = profile_latest_picks.loc[profile_latest_picks["profile"] == profile_name].copy()

    if summary.empty:
        raise ValueError(f"Profile '{profile_name}' not found in profile_summary.csv")

    summary["top_k"] = pd.to_numeric(summary["top_k"], errors="coerce")
    risk["top_k"] = pd.to_numeric(risk["top_k"], errors="coerce")
    latest["top_k"] = pd.to_numeric(latest["top_k"], errors="coerce")
    latest["trade_date"] = pd.to_datetime(latest["trade_date"], errors="coerce")

    strategy_snapshot = summary.merge(
        risk[["profile", "top_k", "model", "max_drawdown", "worst_period_return"]],
        on=["profile", "top_k", "model"],
        how="left",
        suffixes=("", "_risk"),
    ).sort_values("top_k")

    rows: list[pd.DataFrame] = []
    for top_k, group in latest.groupby("top_k", sort=True):
        ranked = group.sort_values("score", ascending=False).head(int(top_k)).copy()
        if ranked.empty:
            continue
        ranked["target_weight"] = 1.0 / float(len(ranked))
        ranked["position_rank"] = range(1, len(ranked) + 1)
        rows.append(ranked)

    if not rows:
        return strategy_snapshot.reset_index(drop=True), pd.DataFrame()

    trade_plan = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["top_k", "position_rank"])
        .reset_index(drop=True)
    )
    preferred_columns = [
        "profile",
        "model",
        "top_k",
        "trade_date",
        "position_rank",
        "symbol",
        "target_weight",
        "score",
        "pred_return",
        "pred_risk",
        "confidence",
        "top_pick_prob",
        "realized_return",
        "reranker_train_periods",
    ]
    available_columns = [column for column in preferred_columns if column in trade_plan.columns]
    return strategy_snapshot.reset_index(drop=True), trade_plan.loc[:, available_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a paper-trading plan from M3Net profile outputs.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing profile_summary.csv, profile_risk_summary.csv, and profile_latest_picks.csv.")
    parser.add_argument("--profile", default="default", help="Profile name to export. Defaults to 'default'.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to --eval-dir.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir or args.eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_summary = _load_required_csv(eval_dir, "profile_summary.csv")
    profile_risk_summary = _load_required_csv(eval_dir, "profile_risk_summary.csv")
    profile_latest_picks = _load_required_csv(eval_dir, "profile_latest_picks.csv")

    strategy_snapshot, trade_plan = _build_trade_plan(
        profile_summary=profile_summary,
        profile_risk_summary=profile_risk_summary,
        profile_latest_picks=profile_latest_picks,
        profile_name=args.profile,
    )

    snapshot_path = output_dir / f"{args.profile}_strategy_snapshot.csv"
    trade_plan_path = output_dir / f"{args.profile}_paper_trade_plan.csv"
    strategy_snapshot.to_csv(snapshot_path, index=False)
    trade_plan.to_csv(trade_plan_path, index=False)

    print(f"Saved strategy snapshot to: {snapshot_path}")
    print(f"Saved paper trade plan to: {trade_plan_path}")
    if not strategy_snapshot.empty:
        print(strategy_snapshot.to_string(index=False))
    if not trade_plan.empty:
        print()
        print(trade_plan.to_string(index=False))


if __name__ == "__main__":
    main()
