"""Export ready-to-use strategy profiles from M3Net evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


DEFAULT_PROFILES: dict[str, dict[int, str]] = {
    "aggressive": {
        5: "m3net_reranker",
        10: "m3net_reranker",
        20: "m3net_full",
    },
    "risk_aware": {
        5: "m3net_risk_reranker",
        10: "m3net_risk_reranker",
        20: "m3net_full",
    },
    "topk_experimental": {
        5: "m3net_topk_reranker",
        10: "m3net_topk_reranker",
        20: "m3net_full",
    },
}


def _compute_max_drawdown(period_returns: pd.Series) -> float:
    clean_returns = pd.to_numeric(period_returns, errors="coerce").dropna()
    if clean_returns.empty:
        return float("nan")
    equity_curve = (1.0 + clean_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def _parse_profile_mapping(value: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("=", 1)
        mapping[int(left.strip())] = right.strip()
    return mapping


def _resolve_profiles(args: argparse.Namespace) -> dict[str, dict[int, str]]:
    profiles = {name: mapping.copy() for name, mapping in DEFAULT_PROFILES.items()}
    for profile_name in profiles:
        override = getattr(args, f"{profile_name}_mapping", None)
        if override:
            profiles[profile_name] = _parse_profile_mapping(override)
    return profiles


def _build_auto_best_profile(compare_summary: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if compare_summary.empty:
        return mapping

    frame = compare_summary.copy()
    frame["top_k"] = pd.to_numeric(frame["top_k"], errors="coerce")
    frame["avg_realized_return"] = pd.to_numeric(frame["avg_realized_return"], errors="coerce")
    frame["win_rate"] = pd.to_numeric(frame["win_rate"], errors="coerce")
    frame["cumulative_return_proxy"] = pd.to_numeric(frame["cumulative_return_proxy"], errors="coerce")

    for top_k, group in frame.dropna(subset=["top_k"]).groupby("top_k"):
        ranked = group.sort_values(
            ["avg_realized_return", "win_rate", "cumulative_return_proxy"],
            ascending=[False, False, False],
        )
        if ranked.empty:
            continue
        mapping[int(top_k)] = str(ranked.iloc[0]["model"])
    return mapping


def _build_profile_summary(compare_summary: pd.DataFrame, profiles: dict[str, dict[int, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile_name, mapping in profiles.items():
        for top_k, model_name in sorted(mapping.items()):
            matched = compare_summary.loc[
                (compare_summary["model"] == model_name)
                & (pd.to_numeric(compare_summary["top_k"], errors="coerce") == int(top_k))
            ]
            if matched.empty:
                rows.append(
                    {
                        "profile": profile_name,
                        "top_k": int(top_k),
                        "model": model_name,
                        "periods": 0,
                        "avg_realized_return": float("nan"),
                        "win_rate": float("nan"),
                        "cumulative_return_proxy": float("nan"),
                    }
                )
                continue

            best = matched.iloc[0]
            rows.append(
                {
                    "profile": profile_name,
                    "top_k": int(top_k),
                    "model": model_name,
                    "periods": int(best["periods"]),
                    "avg_realized_return": float(best["avg_realized_return"]),
                    "win_rate": float(best["win_rate"]),
                    "cumulative_return_proxy": float(best["cumulative_return_proxy"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["profile", "top_k"]).reset_index(drop=True)


def _build_profile_periods(aligned_periods: pd.DataFrame, profiles: dict[str, dict[int, str]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for profile_name, mapping in profiles.items():
        for top_k, model_name in mapping.items():
            matched = aligned_periods.loc[
                (aligned_periods["model"] == model_name)
                & (pd.to_numeric(aligned_periods["top_k"], errors="coerce") == int(top_k))
            ].copy()
            if matched.empty:
                continue
            matched.insert(0, "profile", profile_name)
            rows.append(matched)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["profile", "top_k", "trade_date"]).reset_index(drop=True)


def _build_profile_latest_picks(rolling_picks: pd.DataFrame, profiles: dict[str, dict[int, str]]) -> pd.DataFrame:
    if rolling_picks.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for profile_name, mapping in profiles.items():
        for top_k, model_name in mapping.items():
            subset = rolling_picks.loc[
                (rolling_picks["model"] == model_name)
                & (pd.to_numeric(rolling_picks["top_k"], errors="coerce") == int(top_k))
            ].copy()
            if subset.empty:
                continue
            latest_trade_date = str(pd.to_datetime(subset["trade_date"]).max().date())
            latest = subset.loc[subset["trade_date"].astype(str) == latest_trade_date].copy()
            latest.insert(0, "profile", profile_name)
            rows.append(latest)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["profile", "top_k", "score"], ascending=[True, True, False]).reset_index(drop=True)


def _build_profile_risk_summary(profile_periods: pd.DataFrame) -> pd.DataFrame:
    if profile_periods.empty:
        return pd.DataFrame(
            columns=[
                "profile",
                "top_k",
                "model",
                "periods",
                "max_drawdown",
                "worst_period_return",
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = profile_periods.sort_values("trade_date").groupby(["profile", "top_k", "model"], dropna=False)
    for (profile_name, top_k, model_name), group in grouped:
        realized = pd.to_numeric(group["avg_realized_return"], errors="coerce")
        rows.append(
            {
                "profile": profile_name,
                "top_k": int(top_k),
                "model": model_name,
                "periods": int(realized.notna().sum()),
                "max_drawdown": _compute_max_drawdown(realized),
                "worst_period_return": float(realized.min()) if realized.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["profile", "top_k"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export strategy profile summaries from M3Net evaluation outputs.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing compare_summary.csv, aligned_periods.csv, and rolling_picks.csv.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to --eval-dir.")
    parser.add_argument("--aggressive-mapping", help="Override like '5=m3net_reranker,10=m3net_reranker,20=m3net_full'.")
    parser.add_argument("--risk-aware-mapping", help="Override like '5=m3net_risk_reranker,10=m3net_risk_reranker,20=m3net_full'.")
    parser.add_argument("--topk-experimental-mapping", help="Override like '5=m3net_topk_reranker,10=m3net_topk_reranker,20=m3net_full'.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir or args.eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_summary = pd.read_csv(eval_dir / "compare_summary.csv")
    aligned_periods = pd.read_csv(eval_dir / "aligned_periods.csv")
    rolling_picks = pd.read_csv(eval_dir / "rolling_picks.csv")

    profiles = _resolve_profiles(args)
    auto_best_mapping = _build_auto_best_profile(compare_summary)
    if auto_best_mapping:
        profiles["auto_best"] = auto_best_mapping
    profile_periods = _build_profile_periods(aligned_periods, profiles)
    profile_summary = _build_profile_summary(compare_summary, profiles)
    profile_risk_summary = _build_profile_risk_summary(profile_periods)
    if not profile_risk_summary.empty:
        profile_summary = profile_summary.merge(
            profile_risk_summary[["profile", "top_k", "model", "max_drawdown", "worst_period_return"]],
            on=["profile", "top_k", "model"],
            how="left",
        )
    latest_picks = _build_profile_latest_picks(rolling_picks, profiles)

    profile_summary.to_csv(output_dir / "profile_summary.csv", index=False)
    profile_periods.to_csv(output_dir / "profile_periods.csv", index=False)
    profile_risk_summary.to_csv(output_dir / "profile_risk_summary.csv", index=False)
    latest_picks.to_csv(output_dir / "profile_latest_picks.csv", index=False)

    print(f"Saved profile summary to: {output_dir / 'profile_summary.csv'}")
    print(f"Saved profile periods to: {output_dir / 'profile_periods.csv'}")
    print(f"Saved profile risk summary to: {output_dir / 'profile_risk_summary.csv'}")
    print(f"Saved profile latest picks to: {output_dir / 'profile_latest_picks.csv'}")
    if not profile_summary.empty:
        print(profile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
