"""Build a historical paper-trade ledger from M3Net evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.evaluate_m3net_stage1 import _realized_forward_return
from scripts.train_m3net_stage1 import _load_price_folder


def _compute_max_drawdown(period_returns: pd.Series) -> float:
    clean_returns = pd.to_numeric(period_returns, errors="coerce").dropna()
    if clean_returns.empty:
        return float("nan")
    equity_curve = (1.0 + clean_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def _build_profile_mapping(profile_summary: pd.DataFrame, profile_name: str) -> dict[int, str]:
    subset = profile_summary.loc[profile_summary["profile"] == profile_name].copy()
    if subset.empty:
        raise ValueError(f"Profile '{profile_name}' not found in profile_summary.csv")
    subset["top_k"] = pd.to_numeric(subset["top_k"], errors="coerce")
    return {
        int(row["top_k"]): str(row["model"])
        for _, row in subset.dropna(subset=["top_k"]).iterrows()
    }


def _build_trade_history(
    rolling_picks: pd.DataFrame,
    profile_name: str,
    profile_mapping: dict[int, str],
    daily_data: dict[str, pd.DataFrame],
    label_horizon: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    picks = rolling_picks.copy()
    picks["top_k"] = pd.to_numeric(picks["top_k"], errors="coerce")
    picks["trade_date"] = pd.to_datetime(picks["trade_date"], errors="coerce")
    picks["score"] = pd.to_numeric(picks["score"], errors="coerce")

    for top_k, model_name in sorted(profile_mapping.items()):
        subset = picks.loc[
            (picks["model"] == model_name)
            & (picks["top_k"] == int(top_k))
        ].copy()
        if subset.empty:
            continue

        for trade_date, group in subset.groupby("trade_date", sort=True):
            ranked = group.sort_values("score", ascending=False).head(int(top_k)).copy()
            if ranked.empty:
                continue
            ranked.insert(0, "profile", profile_name)
            ranked["position_rank"] = range(1, len(ranked) + 1)
            ranked["target_weight"] = 1.0 / float(len(ranked))
            realized_returns: list[float | None] = []
            for _, row in ranked.iterrows():
                symbol = str(row["symbol"])
                frame = daily_data.get(symbol)
                realized = None
                if frame is not None:
                    realized = _realized_forward_return(frame, pd.Timestamp(trade_date), label_horizon)
                realized_returns.append(realized)
            ranked["realized_return"] = realized_returns
            rows.append(ranked)

    if not rows:
        return pd.DataFrame()

    history = pd.concat(rows, ignore_index=True)
    return history.sort_values(["top_k", "trade_date", "position_rank"]).reset_index(drop=True)


def _build_period_ledger(trade_history: pd.DataFrame) -> pd.DataFrame:
    if trade_history.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = trade_history.groupby(["profile", "top_k", "model", "trade_date"], dropna=False)
    for (profile_name, top_k, model_name, trade_date), group in grouped:
        realized = pd.to_numeric(group["realized_return"], errors="coerce")
        weights = pd.to_numeric(group["target_weight"], errors="coerce").fillna(0.0)
        settled_mask = realized.notna()
        settled_weight = float(weights.loc[settled_mask].sum())
        partial_weighted_return = float((weights.loc[settled_mask] * realized.loc[settled_mask]).sum()) if settled_mask.any() else float("nan")
        portfolio_realized_return = partial_weighted_return if settled_mask.all() else float("nan")
        rows.append(
            {
                "profile": profile_name,
                "top_k": int(top_k),
                "model": model_name,
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "selected_count": int(len(group)),
                "settled_count": int(settled_mask.sum()),
                "settled_weight": settled_weight,
                "partial_weighted_return": partial_weighted_return,
                "portfolio_realized_return": portfolio_realized_return,
            }
        )
    return pd.DataFrame(rows).sort_values(["profile", "top_k", "trade_date"]).reset_index(drop=True)


def _build_ledger_summary(period_ledger: pd.DataFrame) -> pd.DataFrame:
    if period_ledger.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = period_ledger.groupby(["profile", "top_k", "model"], dropna=False)
    for (profile_name, top_k, model_name), group in grouped:
        realized = pd.to_numeric(group["portfolio_realized_return"], errors="coerce")
        settled = realized.dropna()
        rows.append(
            {
                "profile": profile_name,
                "top_k": int(top_k),
                "model": model_name,
                "periods": int(len(group)),
                "settled_periods": int(settled.notna().sum()),
                "avg_realized_return": float(settled.mean()) if not settled.empty else float("nan"),
                "win_rate": float((settled > 0).mean()) if not settled.empty else float("nan"),
                "cumulative_return_proxy": float((1.0 + settled.fillna(0.0)).prod() - 1.0) if not settled.empty else float("nan"),
                "max_drawdown": _compute_max_drawdown(settled),
                "worst_period_return": float(settled.min()) if not settled.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["profile", "top_k"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update a historical paper-trade ledger from M3Net evaluation outputs.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing profile_summary.csv and rolling_picks.csv.")
    parser.add_argument("--daily-dir", required=True, help="Directory containing per-symbol daily CSV files.")
    parser.add_argument("--profile", default="default", help="Profile name to track. Defaults to 'default'.")
    parser.add_argument("--label-horizon", type=int, default=5, help="Holding horizon in trading days.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to --eval-dir.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir or args.eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_summary = pd.read_csv(eval_dir / "profile_summary.csv")
    rolling_picks = pd.read_csv(eval_dir / "rolling_picks.csv")
    daily_data = _load_price_folder(Path(args.daily_dir))

    profile_mapping = _build_profile_mapping(profile_summary, args.profile)
    trade_history = _build_trade_history(
        rolling_picks=rolling_picks,
        profile_name=args.profile,
        profile_mapping=profile_mapping,
        daily_data=daily_data,
        label_horizon=args.label_horizon,
    )
    period_ledger = _build_period_ledger(trade_history)
    ledger_summary = _build_ledger_summary(period_ledger)

    history_path = output_dir / f"{args.profile}_paper_trade_history.csv"
    periods_path = output_dir / f"{args.profile}_paper_trade_periods.csv"
    summary_path = output_dir / f"{args.profile}_paper_trade_summary.csv"
    trade_history.to_csv(history_path, index=False)
    period_ledger.to_csv(periods_path, index=False)
    ledger_summary.to_csv(summary_path, index=False)

    print(f"Saved paper trade history to: {history_path}")
    print(f"Saved paper trade periods to: {periods_path}")
    print(f"Saved paper trade summary to: {summary_path}")
    if not ledger_summary.empty:
        print(ledger_summary.to_string(index=False))


if __name__ == "__main__":
    main()
