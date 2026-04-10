"""Evaluate a trained full M3Net checkpoint on rolling rebalance dates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from models.gradient_boosting_factor import GradientBoostingFactorModel, PanelFactorDatasetBuilder
from models.m3net_full import M3NetFullDatasetBuilder, M3NetFullModel
from scripts.evaluate_m3net_stage1 import _filter_stock_data_by_min_rows, _realized_forward_return, _slice_daily_history
from scripts.run_m3net_research import _build_rebalance_dates
from scripts.train_m3net_full import _choose_device
from scripts.train_m3net_stage1 import _load_price_folder

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _load_checkpoint(path: Path, device: "torch.device") -> tuple[M3NetFullModel, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = M3NetFullModel(config=config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def _parse_top_k_list(top_n: int, top_k_list: list[int] | None) -> list[int]:
    values = sorted({int(value) for value in (top_k_list or [top_n]) if int(value) > 0})
    if top_n not in values:
        values.append(top_n)
    return sorted(set(values))


def _parse_ensemble_weights(weights: list[float] | None) -> list[float]:
    if not weights:
        return []
    parsed = []
    for value in weights:
        weight = float(value)
        if 0.0 <= weight <= 1.0:
            parsed.append(weight)
    return sorted(set(parsed))


def _standardize_series(series: pd.Series, method: str = "zscore") -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.empty:
        return values

    filled = values.fillna(values.median(skipna=True)).fillna(0.0)
    if method == "percentile":
        ranked = filled.rank(method="average", ascending=False, pct=True)
        return (ranked - 0.5).astype(float)

    if method == "robust_zscore":
        median = float(filled.median())
        mad = float((filled - median).abs().median())
        scale = max(mad * 1.4826, 1e-6)
        return ((filled - median) / scale).clip(-5.0, 5.0).astype(float)

    mean = float(filled.mean())
    std = float(filled.std(ddof=0))
    scale = max(std, 1e-6)
    return ((filled - mean) / scale).clip(-5.0, 5.0).astype(float)


def _build_m3net_ranked_frame(
    symbols: list[str],
    pred_return: object,
    pred_risk: object,
    confidence: object,
    top_pick_prob: object,
    top_n: int,
    config: object,
) -> pd.DataFrame:
    ranked = pd.DataFrame(
        {
            "symbol": symbols,
            "pred_return": pred_return,
            "pred_risk": pred_risk,
            "confidence": confidence,
            "top_pick_prob": top_pick_prob,
        }
    )
    ranked["return_score"] = _standardize_series(ranked["pred_return"], method="zscore")
    ranked["risk_score"] = _standardize_series(ranked["pred_risk"], method="zscore")
    ranked["confidence_score"] = _standardize_series(ranked["confidence"], method="zscore")
    ranked["top_pick_score"] = _standardize_series(ranked["top_pick_prob"], method="zscore")
    ranked["score"] = (
        config.score_return_weight * ranked["return_score"]
        + config.score_top_pick_weight * ranked["top_pick_score"]
        + config.score_confidence_weight * ranked["confidence_score"]
        - config.score_risk_weight * ranked["risk_score"]
    )
    return ranked.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def _build_reranker_candidate_frame(
    m3net_ranked: pd.DataFrame,
    baseline_ranked: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    base = m3net_ranked.head(top_n).copy().reset_index(drop=True)
    base["m3net_rank"] = pd.Series(range(1, len(base) + 1), dtype=float)
    base["m3net_score"] = pd.to_numeric(base["score"], errors="coerce").fillna(0.0)
    base["m3net_rank_pct"] = base["m3net_score"].rank(method="average", ascending=False, pct=True)

    baseline = baseline_ranked[["symbol", "score"]].rename(columns={"score": "baseline_score"}).copy()
    candidate = base.merge(baseline, on="symbol", how="left")
    candidate["baseline_score"] = pd.to_numeric(candidate["baseline_score"], errors="coerce")
    candidate["baseline_score"] = candidate["baseline_score"].fillna(candidate["baseline_score"].median(skipna=True)).fillna(0.0)
    candidate["baseline_rank_pct"] = candidate["baseline_score"].rank(method="average", ascending=False, pct=True)

    candidate["pred_return_z"] = _standardize_series(candidate["pred_return"], method="zscore")
    candidate["pred_risk_z"] = _standardize_series(candidate["pred_risk"], method="zscore")
    candidate["confidence_z"] = _standardize_series(candidate["confidence"], method="zscore")
    candidate["top_pick_z"] = _standardize_series(candidate["top_pick_prob"], method="zscore")
    candidate["baseline_score_z"] = _standardize_series(candidate["baseline_score"], method="zscore")
    candidate["m3net_score_z"] = _standardize_series(candidate["score"], method="zscore")
    candidate["score_gap_z"] = candidate["m3net_score_z"] - candidate["baseline_score_z"]
    candidate["return_confidence"] = pd.to_numeric(candidate["pred_return"], errors="coerce").fillna(0.0) * pd.to_numeric(candidate["confidence"], errors="coerce").fillna(0.0)
    candidate["top_pick_confidence"] = pd.to_numeric(candidate["top_pick_prob"], errors="coerce").fillna(0.0) * pd.to_numeric(candidate["confidence"], errors="coerce").fillna(0.0)
    return candidate


def _build_reranker_training_rows(
    candidate_frame: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    horizon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in candidate_frame.iterrows():
        realized = _realized_forward_return(daily_data[row["symbol"]], pd.Timestamp(trade_date), horizon)
        if realized is None:
            continue
        rows.append(
            {
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "symbol": row["symbol"],
                "realized_return": float(realized),
                "m3net_score": float(row["score"]),
                "baseline_score": float(row["baseline_score"]),
                "pred_return": float(row["pred_return"]),
                "pred_risk": float(row["pred_risk"]),
                "confidence": float(row["confidence"]),
                "top_pick_prob": float(row["top_pick_prob"]),
                "m3net_rank_pct": float(row["m3net_rank_pct"]),
                "baseline_rank_pct": float(row["baseline_rank_pct"]),
                "pred_return_z": float(row["pred_return_z"]),
                "pred_risk_z": float(row["pred_risk_z"]),
                "confidence_z": float(row["confidence_z"]),
                "top_pick_z": float(row["top_pick_z"]),
                "baseline_score_z": float(row["baseline_score_z"]),
                "m3net_score_z": float(row["m3net_score_z"]),
                "score_gap_z": float(row["score_gap_z"]),
                "return_confidence": float(row["return_confidence"]),
                "top_pick_confidence": float(row["top_pick_confidence"]),
            }
        )
    return rows


def _reranker_feature_columns() -> list[str]:
    return [
        "m3net_score",
        "baseline_score",
        "pred_return",
        "pred_risk",
        "confidence",
        "top_pick_prob",
        "m3net_rank_pct",
        "baseline_rank_pct",
        "pred_return_z",
        "pred_risk_z",
        "confidence_z",
        "top_pick_z",
        "baseline_score_z",
        "m3net_score_z",
        "score_gap_z",
        "return_confidence",
        "top_pick_confidence",
    ]


def _build_reranker_ranked_frame(
    candidate_frame: pd.DataFrame,
    training_rows: list[dict[str, object]],
    train_min_periods: int,
    top_n: int,
) -> tuple[pd.DataFrame | None, int]:
    if not training_rows:
        return None, 0

    training = pd.DataFrame(training_rows)
    valid_periods = training["trade_date"].nunique()
    if valid_periods < train_min_periods:
        return None, int(valid_periods)

    feature_columns = _reranker_feature_columns()
    train_x = training[feature_columns].copy()
    train_x = train_x.fillna(train_x.median(numeric_only=True)).fillna(0.0)
    train_y = pd.to_numeric(training["realized_return"], errors="coerce").fillna(0.0)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=8,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(train_x, train_y)

    ranked = candidate_frame.copy()
    pred_x = ranked[feature_columns].copy()
    pred_x = pred_x.fillna(train_x.median(numeric_only=True)).fillna(0.0)
    ranked["score"] = model.predict(pred_x)
    return ranked.sort_values("score", ascending=False).head(top_n).reset_index(drop=True), int(valid_periods)


def _build_risk_aware_reranker_ranked_frame(
    candidate_frame: pd.DataFrame,
    training_rows: list[dict[str, object]],
    train_min_periods: int,
    top_n: int,
    downside_penalty: float,
    downside_power: float,
    inference_risk_weight: float,
) -> tuple[pd.DataFrame | None, int]:
    if not training_rows:
        return None, 0

    training = pd.DataFrame(training_rows)
    valid_periods = training["trade_date"].nunique()
    if valid_periods < train_min_periods:
        return None, int(valid_periods)

    feature_columns = _reranker_feature_columns()
    train_x = training[feature_columns].copy()
    feature_medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(feature_medians).fillna(0.0)
    realized = pd.to_numeric(training["realized_return"], errors="coerce").fillna(0.0)
    downside = (-realized).clip(lower=0.0)
    adjusted_target = realized - downside_penalty * downside.pow(max(downside_power, 1.0))
    sample_weight = 1.0 + downside_penalty * downside

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=250,
        max_depth=3,
        min_samples_leaf=8,
        l2_regularization=0.08,
        random_state=42,
    )
    model.fit(train_x, adjusted_target, sample_weight=sample_weight)

    ranked = candidate_frame.copy()
    pred_x = ranked[feature_columns].copy()
    pred_x = pred_x.fillna(feature_medians).fillna(0.0)
    predicted_adjusted = model.predict(pred_x)
    explicit_risk_penalty = inference_risk_weight * pd.to_numeric(ranked["pred_risk_z"], errors="coerce").fillna(0.0)
    ranked["score"] = predicted_adjusted - explicit_risk_penalty
    return ranked.sort_values("score", ascending=False).head(top_n).reset_index(drop=True), int(valid_periods)


if nn is not None:

    class _TopKReranker(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            return self.network(features).squeeze(-1)


def _build_topk_reranker_ranked_frame(
    candidate_frame: pd.DataFrame,
    training_rows: list[dict[str, object]],
    train_min_periods: int,
    top_n: int,
    top_k_target: int,
    device: "torch.device",
    epochs: int = 120,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.35,
) -> tuple[pd.DataFrame | None, int]:
    if torch is None:
        raise ImportError("PyTorch is required for the differentiable Top-K reranker.")
    if not training_rows:
        return None, 0

    training = pd.DataFrame(training_rows)
    valid_periods = training["trade_date"].nunique()
    if valid_periods < train_min_periods:
        return None, int(valid_periods)

    feature_columns = _reranker_feature_columns()
    feature_medians = training[feature_columns].median(numeric_only=True)
    train_x = training[feature_columns].copy().fillna(feature_medians).fillna(0.0)
    train_x = train_x.astype(float)
    mean = train_x.mean()
    std = train_x.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    train_x = (train_x - mean) / std

    group_frames = []
    for _, group in training.groupby("trade_date", sort=True):
        group_x = ((group[feature_columns].copy().fillna(feature_medians).fillna(0.0).astype(float) - mean) / std).astype(float)
        group_frames.append(
            (
                torch.tensor(group_x.to_numpy(), dtype=torch.float32, device=device),
                torch.tensor(pd.to_numeric(group["realized_return"], errors="coerce").fillna(0.0).to_numpy(), dtype=torch.float32, device=device),
            )
        )

    model = _TopKReranker(input_dim=len(feature_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    focus_k = max(1, int(top_k_target))
    temp = max(float(temperature), 1e-4)

    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        group_count = 0

        for group_x, group_y in group_frames:
            if group_x.size(0) < 2:
                continue
            logits = model(group_x)
            order = torch.argsort(group_y, descending=True)
            ordered_logits = logits[order]
            ordered_count = ordered_logits.size(0)

            listmle_terms = torch.logcumsumexp(torch.flip(ordered_logits, dims=[0]), dim=0)
            listmle_terms = torch.flip(listmle_terms, dims=[0]) - ordered_logits
            discounts = 1.0 / torch.log2(torch.arange(ordered_count, device=device, dtype=torch.float32) + 2.0)
            if focus_k < ordered_count:
                discounts[focus_k:] = discounts[focus_k:] * 0.2
            listmle_loss = (listmle_terms * discounts).sum() / discounts.sum().clamp_min(1e-6)

            k = min(focus_k, group_y.numel())
            threshold = torch.topk(group_y, k=k).values[-1]
            topk_target = (group_y >= threshold).float()
            topk_target = topk_target / topk_target.sum().clamp_min(1.0)
            pred_log_probs = torch.log_softmax(logits / temp, dim=0)
            topk_loss = F.kl_div(pred_log_probs, topk_target, reduction="batchmean")

            total_loss = total_loss + listmle_loss + topk_loss
            group_count += 1

        if group_count == 0:
            break
        total_loss = total_loss / group_count
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    pred_x = candidate_frame[feature_columns].copy().fillna(feature_medians).fillna(0.0).astype(float)
    pred_x = ((pred_x - mean) / std).astype(float)
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(pred_x.to_numpy(), dtype=torch.float32, device=device)).detach().cpu().numpy()

    ranked = candidate_frame.copy()
    ranked["score"] = scores
    return ranked.sort_values("score", ascending=False).head(top_n).reset_index(drop=True), int(valid_periods)


def _evaluate_ranked_frame(
    ranked: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    horizon: int,
    model_name: str,
    top_k_list: list[int],
    extra_fields: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    period_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []
    extra_fields = extra_fields or {}

    for top_k in top_k_list:
        top_frame = ranked.head(top_k).copy()
        realized_returns: list[float] = []
        for _, row in top_frame.iterrows():
            realized = _realized_forward_return(
                daily_data[row["symbol"]],
                pd.Timestamp(trade_date),
                horizon,
            )
            if realized is not None:
                realized_returns.append(realized)
            pick_rows.append(
                {
                    "model": model_name,
                    "top_k": int(top_k),
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "symbol": row["symbol"],
                    "score": float(row["score"]),
                    "pred_return": float(row["pred_return"]) if "pred_return" in row else float("nan"),
                    "pred_risk": float(row["pred_risk"]) if "pred_risk" in row else float("nan"),
                    "confidence": float(row["confidence"]) if "confidence" in row else float("nan"),
                    "top_pick_prob": float(row["top_pick_prob"]) if "top_pick_prob" in row else float("nan"),
                    "realized_return": realized,
                    **extra_fields,
                }
            )

        period_rows.append(
            {
                "model": model_name,
                "top_k": int(top_k),
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "selected_count": int(len(top_frame)),
                "avg_predicted_score": float(top_frame["score"].mean()) if not top_frame.empty else float("nan"),
                "avg_realized_return": float(pd.Series(realized_returns).mean()) if realized_returns else float("nan"),
                "win_rate": float((pd.Series(realized_returns) > 0).mean()) if realized_returns else float("nan"),
                **extra_fields,
            }
        )

    return period_rows, pick_rows


def _summarize_periods(periods: pd.DataFrame) -> pd.DataFrame:
    if periods.empty:
        return pd.DataFrame(columns=["model", "top_k", "periods", "avg_realized_return", "win_rate", "cumulative_return_proxy"])

    summary_rows: list[dict[str, object]] = []
    for (model_name, top_k), group in periods.groupby(["model", "top_k"], dropna=False):
        summary_rows.append(
            {
                "model": model_name,
                "top_k": int(top_k),
                "periods": int(len(group)),
                "avg_realized_return": float(group["avg_realized_return"].mean()),
                "win_rate": float((group["avg_realized_return"] > 0).mean()),
                "cumulative_return_proxy": float((1.0 + group["avg_realized_return"].fillna(0.0)).prod() - 1.0),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["model", "top_k"]).reset_index(drop=True)


def _align_common_valid_periods(periods: pd.DataFrame) -> pd.DataFrame:
    if periods.empty:
        return periods.copy()

    valid = periods.dropna(subset=["avg_realized_return"]).copy()
    if valid.empty or "model" not in valid.columns or "top_k" not in valid.columns or "trade_date" not in valid.columns:
        return valid

    model_names = sorted(valid["model"].dropna().unique().tolist())
    if len(model_names) < 2:
        return valid

    aligned_frames: list[pd.DataFrame] = []
    for top_k, group in valid.groupby("top_k", dropna=False):
        common_dates = None
        for model_name in model_names:
            dates = set(group.loc[group["model"] == model_name, "trade_date"].astype(str))
            common_dates = dates if common_dates is None else common_dates & dates
        if not common_dates:
            continue
        aligned_frames.append(group.loc[group["trade_date"].astype(str).isin(common_dates)].copy())

    if not aligned_frames:
        return valid.iloc[0:0].copy()
    return pd.concat(aligned_frames, ignore_index=True).sort_values(["top_k", "trade_date", "model"]).reset_index(drop=True)


def _build_baseline_ranked_frame(
    stock_data: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    backend: str,
    label_horizon: int,
    top_n: int,
) -> pd.DataFrame:
    builder = PanelFactorDatasetBuilder()
    train_panel = builder.build_dataset(
        stock_data,
        label_horizon=label_horizon,
        min_history=80,
        label_col="label",
        drop_na_label=True,
    )
    score_panel = builder.build_dataset(
        stock_data,
        label_horizon=label_horizon,
        min_history=80,
        label_col="label",
        drop_na_label=False,
    )
    model = GradientBoostingFactorModel(backend=backend)
    model.fit(train_panel, label_col="label", train_ratio=0.8)
    scored = model.predict(score_panel)
    ranked = (
        scored.loc[pd.to_datetime(scored["date"]) == pd.Timestamp(as_of_date), ["symbol", "score"]]
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked["pred_return"] = ranked["score"]
    ranked["pred_risk"] = float("nan")
    ranked["confidence"] = float("nan")
    ranked["top_pick_prob"] = float("nan")
    return ranked


def _build_ensemble_ranked_frame(
    baseline_ranked: pd.DataFrame,
    m3net_ranked: pd.DataFrame,
    baseline_weight: float,
    top_n: int,
) -> pd.DataFrame:
    baseline_frame = baseline_ranked[["symbol", "score"]].rename(columns={"score": "baseline_score"})
    m3net_frame = m3net_ranked[["symbol", "score", "pred_return", "pred_risk", "confidence", "top_pick_prob"]].rename(columns={"score": "m3net_score"})
    merged = baseline_frame.merge(m3net_frame, on="symbol", how="outer")
    merged["baseline_score"] = pd.to_numeric(merged["baseline_score"], errors="coerce")
    merged["m3net_score"] = pd.to_numeric(merged["m3net_score"], errors="coerce")
    merged["baseline_score"] = merged["baseline_score"].fillna(merged["baseline_score"].median(skipna=True))
    merged["m3net_score"] = merged["m3net_score"].fillna(merged["m3net_score"].median(skipna=True))
    merged["baseline_score"] = merged["baseline_score"].fillna(0.0)
    merged["m3net_score"] = merged["m3net_score"].fillna(0.0)

    baseline_rank = merged["baseline_score"].rank(method="average", ascending=False, pct=True)
    m3net_rank = merged["m3net_score"].rank(method="average", ascending=False, pct=True)
    merged["score"] = baseline_weight * baseline_rank + (1.0 - baseline_weight) * m3net_rank
    merged["pred_return"] = pd.to_numeric(merged["pred_return"], errors="coerce").fillna(merged["m3net_score"])
    merged["pred_risk"] = pd.to_numeric(merged["pred_risk"], errors="coerce")
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    merged["top_pick_prob"] = pd.to_numeric(merged["top_pick_prob"], errors="coerce")
    return merged.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def _build_standardized_ensemble_ranked_frame(
    baseline_ranked: pd.DataFrame,
    m3net_ranked: pd.DataFrame,
    baseline_weight: float,
    top_n: int,
    method: str = "zscore",
) -> pd.DataFrame:
    baseline_frame = baseline_ranked[["symbol", "score"]].rename(columns={"score": "baseline_score"})
    m3net_frame = m3net_ranked[["symbol", "score", "pred_return", "pred_risk", "confidence", "top_pick_prob"]].rename(columns={"score": "m3net_score"})
    merged = baseline_frame.merge(m3net_frame, on="symbol", how="outer")
    merged["baseline_score"] = pd.to_numeric(merged["baseline_score"], errors="coerce")
    merged["m3net_score"] = pd.to_numeric(merged["m3net_score"], errors="coerce")
    merged["baseline_std_score"] = _standardize_series(merged["baseline_score"], method=method)
    merged["m3net_std_score"] = _standardize_series(merged["m3net_score"], method=method)
    merged["score"] = baseline_weight * merged["baseline_std_score"] + (1.0 - baseline_weight) * merged["m3net_std_score"]
    merged["pred_return"] = pd.to_numeric(merged["pred_return"], errors="coerce").fillna(merged["m3net_score"])
    merged["pred_risk"] = pd.to_numeric(merged["pred_risk"], errors="coerce")
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    merged["top_pick_prob"] = pd.to_numeric(merged["top_pick_prob"], errors="coerce")
    return merged.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def _compute_dynamic_baseline_weight(
    periods: pd.DataFrame,
    trade_date: pd.Timestamp,
    reference_top_k: int,
    lookback_periods: int,
    min_weight: float = 0.1,
    max_weight: float = 0.9,
    temperature: float = 0.02,
) -> float:
    if periods.empty:
        return 0.5

    frame = periods.copy()
    frame = frame.loc[
        frame["model"].isin(["baseline_factor", "m3net_full"])
        & (pd.to_numeric(frame["top_k"], errors="coerce") == int(reference_top_k))
    ].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.loc[
        (frame["trade_date"] < pd.Timestamp(trade_date))
        & frame["avg_realized_return"].notna()
    ].sort_values("trade_date")
    if frame.empty:
        return 0.5

    baseline_tail = frame.loc[frame["model"] == "baseline_factor", "avg_realized_return"].tail(lookback_periods)
    m3net_tail = frame.loc[frame["model"] == "m3net_full", "avg_realized_return"].tail(lookback_periods)
    if baseline_tail.empty and m3net_tail.empty:
        return 0.5

    baseline_mean = float(baseline_tail.mean()) if not baseline_tail.empty else 0.0
    m3net_mean = float(m3net_tail.mean()) if not m3net_tail.empty else 0.0
    baseline_score = pd.Series([baseline_mean / max(temperature, 1e-6)]).clip(-20, 20).iloc[0]
    m3net_score = pd.Series([m3net_mean / max(temperature, 1e-6)]).clip(-20, 20).iloc[0]
    baseline_exp = float(math.exp(baseline_score))
    m3net_exp = float(math.exp(m3net_score))
    weight = baseline_exp / max(baseline_exp + m3net_exp, 1e-8)
    return float(min(max(weight, min_weight), max_weight))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained full M3Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--daily-dir", required=True)
    parser.add_argument("--minute-dir")
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--eval-lookback-periods", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-k-list", nargs="+", type=int, help="Optional list like 5 10 20 for layered evaluation.")
    parser.add_argument("--include-baseline", action="store_true", help="Also evaluate baseline_factor on the same dates.")
    parser.add_argument("--ensemble-weights", nargs="+", type=float, help="Optional baseline weights such as 0.2 0.5 0.8.")
    parser.add_argument("--include-standardized-ensemble", action="store_true", help="Evaluate standardized-score ensembles alongside rank ensembles.")
    parser.add_argument("--standardized-method", default="zscore", choices=["zscore", "robust_zscore", "percentile"])
    parser.add_argument("--include-dynamic-ensemble", action="store_true", help="Enable dynamic baseline/M3Net weighting based on recent realized returns.")
    parser.add_argument("--dynamic-lookback-periods", type=int, default=3, help="How many prior valid periods to use for dynamic weights.")
    parser.add_argument("--dynamic-reference-top-k", type=int, help="Use this top-k history to compute dynamic weights. Defaults to --top-n.")
    parser.add_argument("--include-reranker", action="store_true", help="Train a rolling second-stage reranker on prior Top-N candidates.")
    parser.add_argument("--reranker-train-min-periods", type=int, default=4, help="Minimum number of historical rebalance periods before enabling reranker.")
    parser.add_argument("--reranker-candidate-pool", type=int, default=20, help="How many M3Net candidates to keep for stage-two reranking.")
    parser.add_argument("--include-topk-reranker", action="store_true", help="Train a differentiable Top-K reranker on prior candidate pools.")
    parser.add_argument("--topk-reranker-epochs", type=int, default=120)
    parser.add_argument("--topk-reranker-learning-rate", type=float, default=1e-3)
    parser.add_argument("--topk-reranker-top-k-target", type=int, default=5)
    parser.add_argument("--include-risk-reranker", action="store_true", help="Train a downside-penalized reranker on prior candidate pools.")
    parser.add_argument("--risk-reranker-downside-penalty", type=float, default=1.5)
    parser.add_argument("--risk-reranker-downside-power", type=float, default=1.5)
    parser.add_argument("--risk-reranker-inference-risk-weight", type=float, default=0.35)
    parser.add_argument("--baseline-backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--baseline-train-lookback-days", type=int, default=504)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="artifacts/m3net_full_eval")
    args = parser.parse_args()

    if torch is None:
        raise ImportError("PyTorch is required to evaluate the full M3Net model.")

    device = _choose_device(args.device)
    model, config = _load_checkpoint(Path(args.checkpoint), device)
    daily_data = _filter_stock_data_by_min_rows(_load_price_folder(Path(args.daily_dir)), args.min_stock_rows)
    minute_data = _load_price_folder(Path(args.minute_dir)) if args.minute_dir else None
    if minute_data is not None:
        minute_data = {symbol: frame for symbol, frame in minute_data.items() if symbol in daily_data}

    rebalance_dates = _build_rebalance_dates(daily_data, args.rebalance_freq)
    if args.eval_lookback_periods > 0:
        rebalance_dates = rebalance_dates[-args.eval_lookback_periods :]
    top_k_list = _parse_top_k_list(args.top_n, args.top_k_list)
    ensemble_weights = _parse_ensemble_weights(args.ensemble_weights)
    dynamic_reference_top_k = int(args.dynamic_reference_top_k or args.top_n)
    candidate_pool_size = max(args.top_n, max(top_k_list), int(args.reranker_candidate_pool))

    builder = M3NetFullDatasetBuilder(config=config)
    samples = builder.build_samples(daily_data, minute_data=minute_data, rebalance_dates=rebalance_dates)
    m3net_ranked_by_date: dict[str, pd.DataFrame] = {}
    reranker_training_rows: list[dict[str, object]] = []

    period_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for sample in samples:
            output = model(
                sample.daily_sequence.to(device),
                sample.intraday_sequence.to(device),
                sample.factor_features.to(device),
                sample.memory_features.to(device),
            )
            ranked = _build_m3net_ranked_frame(
                symbols=sample.symbols,
                pred_return=output.return_pred.detach().cpu().numpy(),
                pred_risk=output.risk_pred.detach().cpu().numpy(),
                confidence=output.confidence.detach().cpu().numpy(),
                top_pick_prob=output.top_pick_prob.detach().cpu().numpy(),
                top_n=candidate_pool_size,
                config=config,
            )
            m3net_ranked_by_date[str(pd.Timestamp(sample.date).date())] = ranked.copy()

            rows, picks = _evaluate_ranked_frame(
                ranked=ranked,
                daily_data=daily_data,
                trade_date=pd.Timestamp(sample.date),
                horizon=config.label_horizon,
                model_name="m3net_full",
                top_k_list=top_k_list,
            )
            period_rows.extend(rows)
            pick_rows.extend(picks)

    if args.include_baseline:
        for trade_date in rebalance_dates:
            sliced_daily = _slice_daily_history(
                daily_data,
                pd.Timestamp(trade_date),
                min_history=80,
                max_history_days=args.baseline_train_lookback_days,
            )
            if len(sliced_daily) < max(args.top_n, 5):
                continue
            baseline_ranked = _build_baseline_ranked_frame(
                stock_data=sliced_daily,
                as_of_date=pd.Timestamp(trade_date),
                backend=args.baseline_backend,
                label_horizon=config.label_horizon,
                top_n=candidate_pool_size,
            )
            rows, picks = _evaluate_ranked_frame(
                ranked=baseline_ranked,
                daily_data=daily_data,
                trade_date=pd.Timestamp(trade_date),
                horizon=config.label_horizon,
                model_name="baseline_factor",
                top_k_list=top_k_list,
            )
            period_rows.extend(rows)
            pick_rows.extend(picks)

            m3net_ranked = m3net_ranked_by_date.get(str(pd.Timestamp(trade_date).date()))
            candidate_frame = None
            if m3net_ranked is not None:
                candidate_frame = _build_reranker_candidate_frame(
                    m3net_ranked=m3net_ranked,
                    baseline_ranked=baseline_ranked,
                    top_n=candidate_pool_size,
                )

            if m3net_ranked is not None and ensemble_weights:
                for baseline_weight in ensemble_weights:
                    ensemble_ranked = _build_ensemble_ranked_frame(
                        baseline_ranked=baseline_ranked,
                        m3net_ranked=m3net_ranked,
                        baseline_weight=baseline_weight,
                        top_n=max(top_k_list),
                    )
                    rows, picks = _evaluate_ranked_frame(
                        ranked=ensemble_ranked,
                        daily_data=daily_data,
                        trade_date=pd.Timestamp(trade_date),
                        horizon=config.label_horizon,
                        model_name=f"ensemble_b{baseline_weight:.1f}_m{1.0 - baseline_weight:.1f}",
                        top_k_list=top_k_list,
                    )
                    period_rows.extend(rows)
                    pick_rows.extend(picks)

                    if args.include_standardized_ensemble:
                        standardized_ranked = _build_standardized_ensemble_ranked_frame(
                            baseline_ranked=baseline_ranked,
                            m3net_ranked=m3net_ranked,
                            baseline_weight=baseline_weight,
                            top_n=max(top_k_list),
                            method=args.standardized_method,
                        )
                        rows, picks = _evaluate_ranked_frame(
                            ranked=standardized_ranked,
                            daily_data=daily_data,
                            trade_date=pd.Timestamp(trade_date),
                            horizon=config.label_horizon,
                            model_name=f"std_{args.standardized_method}_b{baseline_weight:.1f}_m{1.0 - baseline_weight:.1f}",
                            top_k_list=top_k_list,
                        )
                        period_rows.extend(rows)
                        pick_rows.extend(picks)

            if candidate_frame is not None and args.include_reranker:
                reranked, train_periods = _build_reranker_ranked_frame(
                    candidate_frame=candidate_frame,
                    training_rows=reranker_training_rows,
                    train_min_periods=args.reranker_train_min_periods,
                    top_n=max(top_k_list),
                )
                if reranked is not None:
                    rows, picks = _evaluate_ranked_frame(
                        ranked=reranked,
                        daily_data=daily_data,
                        trade_date=pd.Timestamp(trade_date),
                        horizon=config.label_horizon,
                        model_name="m3net_reranker",
                        top_k_list=top_k_list,
                        extra_fields={"reranker_train_periods": int(train_periods)},
                    )
                    period_rows.extend(rows)
                    pick_rows.extend(picks)

            if candidate_frame is not None and args.include_risk_reranker:
                risk_reranked, train_periods = _build_risk_aware_reranker_ranked_frame(
                    candidate_frame=candidate_frame,
                    training_rows=reranker_training_rows,
                    train_min_periods=args.reranker_train_min_periods,
                    top_n=max(top_k_list),
                    downside_penalty=args.risk_reranker_downside_penalty,
                    downside_power=args.risk_reranker_downside_power,
                    inference_risk_weight=args.risk_reranker_inference_risk_weight,
                )
                if risk_reranked is not None:
                    rows, picks = _evaluate_ranked_frame(
                        ranked=risk_reranked,
                        daily_data=daily_data,
                        trade_date=pd.Timestamp(trade_date),
                        horizon=config.label_horizon,
                        model_name="m3net_risk_reranker",
                        top_k_list=top_k_list,
                        extra_fields={"reranker_train_periods": int(train_periods)},
                    )
                    period_rows.extend(rows)
                    pick_rows.extend(picks)

            if candidate_frame is not None and args.include_topk_reranker:
                topk_reranked, train_periods = _build_topk_reranker_ranked_frame(
                    candidate_frame=candidate_frame,
                    training_rows=reranker_training_rows,
                    train_min_periods=args.reranker_train_min_periods,
                    top_n=max(top_k_list),
                    top_k_target=args.topk_reranker_top_k_target,
                    device=device,
                    epochs=args.topk_reranker_epochs,
                    learning_rate=args.topk_reranker_learning_rate,
                )
                if topk_reranked is not None:
                    rows, picks = _evaluate_ranked_frame(
                        ranked=topk_reranked,
                        daily_data=daily_data,
                        trade_date=pd.Timestamp(trade_date),
                        horizon=config.label_horizon,
                        model_name="m3net_topk_reranker",
                        top_k_list=top_k_list,
                        extra_fields={"reranker_train_periods": int(train_periods)},
                    )
                    period_rows.extend(rows)
                    pick_rows.extend(picks)

            if m3net_ranked is not None and args.include_dynamic_ensemble:
                realized_periods = pd.DataFrame(period_rows)
                dynamic_weight = _compute_dynamic_baseline_weight(
                    periods=realized_periods,
                    trade_date=pd.Timestamp(trade_date),
                    reference_top_k=dynamic_reference_top_k,
                    lookback_periods=args.dynamic_lookback_periods,
                )
                dynamic_ranked = _build_ensemble_ranked_frame(
                    baseline_ranked=baseline_ranked,
                    m3net_ranked=m3net_ranked,
                    baseline_weight=dynamic_weight,
                    top_n=max(top_k_list),
                )
                rows, picks = _evaluate_ranked_frame(
                    ranked=dynamic_ranked,
                    daily_data=daily_data,
                    trade_date=pd.Timestamp(trade_date),
                    horizon=config.label_horizon,
                    model_name="dynamic_ensemble",
                    top_k_list=top_k_list,
                    extra_fields={"baseline_weight": float(dynamic_weight), "m3net_weight": float(1.0 - dynamic_weight)},
                )
                period_rows.extend(rows)
                pick_rows.extend(picks)

            if candidate_frame is not None and args.include_reranker:
                reranker_training_rows.extend(
                    _build_reranker_training_rows(
                        candidate_frame=candidate_frame,
                        daily_data=daily_data,
                        trade_date=pd.Timestamp(trade_date),
                        horizon=config.label_horizon,
                    )
                )

    periods = pd.DataFrame(period_rows)
    picks = pd.DataFrame(pick_rows)
    summary = _summarize_periods(periods.dropna(subset=["avg_realized_return"]))
    aligned_periods = _align_common_valid_periods(periods)
    compare_summary = _summarize_periods(aligned_periods)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    periods.to_csv(output_dir / "rolling_periods.csv", index=False)
    picks.to_csv(output_dir / "rolling_picks.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    aligned_periods.to_csv(output_dir / "aligned_periods.csv", index=False)
    compare_summary.to_csv(output_dir / "compare_summary.csv", index=False)

    print(f"Saved periods to: {output_dir / 'rolling_periods.csv'}")
    print(f"Saved picks to: {output_dir / 'rolling_picks.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.csv'}")
    print(f"Saved aligned periods to: {output_dir / 'aligned_periods.csv'}")
    print(f"Saved compare summary to: {output_dir / 'compare_summary.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
