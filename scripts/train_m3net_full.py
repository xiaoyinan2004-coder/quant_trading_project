"""Cloud-ready training entrypoint for the full M3Net model."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from models.m3net_full import M3NetFullConfig, M3NetFullDatasetBuilder, M3NetFullModel
from scripts.evaluate_m3net_stage1 import _filter_stock_data_by_min_rows
from scripts.run_m3net_research import _build_rebalance_dates
from scripts.train_m3net_stage1 import _load_price_folder

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _choose_device(requested: str) -> "torch.device":
    if torch is None:
        raise ImportError("PyTorch is required. Install it on the cloud server before training M3Net full.")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_finite(name: str, tensor: "torch.Tensor", sample_date: object) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite values detected in {name} for sample {sample_date}.")


def _save_checkpoint(
    path: Path,
    config: M3NetFullConfig,
    model: "torch.nn.Module",
    epoch: int,
    train_loss: float,
    valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        path,
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the full GPU-first M3Net model.")
    parser.add_argument("--daily-dir", required=True, help="Folder of per-symbol daily OHLCV CSV files.")
    parser.add_argument("--minute-dir", help="Folder of per-symbol minute CSV files.")
    parser.add_argument("--min-stock-rows", type=int, default=1000)
    parser.add_argument("--rebalance-freq", default="M")
    parser.add_argument("--train-lookback-periods", type=int, default=36, help="Number of rebalance periods to retain.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="artifacts/m3net_full.pt")
    args = parser.parse_args()

    device = _choose_device(args.device)
    _set_random_seed(args.random_state)
    daily_data = _filter_stock_data_by_min_rows(_load_price_folder(Path(args.daily_dir)), args.min_stock_rows)
    minute_data = _load_price_folder(Path(args.minute_dir)) if args.minute_dir else None
    if minute_data is not None:
        minute_data = {symbol: frame for symbol, frame in minute_data.items() if symbol in daily_data}

    rebalance_dates = _build_rebalance_dates(daily_data, args.rebalance_freq)
    if args.train_lookback_periods > 0:
        rebalance_dates = rebalance_dates[-args.train_lookback_periods :]

    config = M3NetFullConfig(
        device=str(device),
        max_epochs=args.epochs,
        rank_loss_weight=args.rank_loss_weight,
        listwise_loss_weight=args.listwise_loss_weight,
        top_pick_loss_weight=args.top_pick_loss_weight,
        weighted_return_alpha=args.weighted_return_alpha,
        top_pick_quantile=args.top_pick_quantile,
        listwise_topk_focus=args.listwise_topk_focus,
        graph_neighbor_k=args.graph_neighbor_k,
        graph_residual_weight=args.graph_residual_weight,
        graph_contrastive_loss_weight=args.graph_contrastive_loss_weight,
        graph_contrastive_neighbors=args.graph_contrastive_neighbors,
        random_state=args.random_state,
    )
    builder = M3NetFullDatasetBuilder(config=config)
    samples = builder.build_samples(daily_data, minute_data=minute_data, rebalance_dates=rebalance_dates)
    if len(samples) < 2:
        raise ValueError("Not enough cross-sectional samples were built for training.")

    split_index = max(1, int(len(samples) * config.train_ratio))
    train_samples = samples[:split_index]
    valid_samples = samples[split_index:] or samples[-1:]

    model = M3NetFullModel(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_enabled = device.type == "cuda" and config.precision == "amp"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    output_path = Path(args.output)
    best_output_path = output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    history_path = output_path.with_name(f"{output_path.stem}.history.csv")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for sample in train_samples:
            _ensure_finite("daily_sequence", sample.daily_sequence, sample.date)
            _ensure_finite("intraday_sequence", sample.intraday_sequence, sample.date)
            _ensure_finite("factor_features", sample.factor_features, sample.date)
            _ensure_finite("memory_features", sample.memory_features, sample.date)
            _ensure_finite("future_return", sample.future_return, sample.date)
            _ensure_finite("future_risk", sample.future_risk, sample.date)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(
                    sample.daily_sequence.to(device),
                    sample.intraday_sequence.to(device),
                    sample.factor_features.to(device),
                    sample.memory_features.to(device),
                )
                loss = model.loss(output, sample.future_return.to(device), sample.future_risk.to(device))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss detected for sample {sample.date}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for sample in valid_samples:
                output = model(
                    sample.daily_sequence.to(device),
                    sample.intraday_sequence.to(device),
                    sample.factor_features.to(device),
                    sample.memory_features.to(device),
                )
                loss = model.loss(output, sample.future_return.to(device), sample.future_risk.to(device))
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite validation loss detected for sample {sample.date}.")
                valid_losses.append(float(loss.detach().cpu()))

        train_loss_mean = float(pd.Series(train_losses).mean())
        valid_loss_mean = float(pd.Series(valid_losses).mean())
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "valid_loss": valid_loss_mean,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss_mean:.6f} "
            f"valid_loss={valid_loss_mean:.6f} samples={len(train_samples)}/{len(valid_samples)}"
        )

        if valid_loss_mean < best_valid_loss:
            best_valid_loss = valid_loss_mean
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(best_output_path, config, model, epoch, train_loss_mean, valid_loss_mean)
            print(f"best checkpoint updated -> {best_output_path} (epoch={epoch}, valid_loss={valid_loss_mean:.6f})")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping triggered at epoch={epoch} best_epoch={best_epoch} best_valid_loss={best_valid_loss:.6f}")
            break

    _save_checkpoint(output_path, config, model, epoch, train_loss_mean, valid_loss_mean)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    print(f"Saved latest M3Net full checkpoint to: {output_path}")
    print(f"Saved best M3Net full checkpoint to: {best_output_path}")
    print(f"Saved training history to: {history_path}")
    print(f"Best epoch={best_epoch} best_valid_loss={best_valid_loss:.6f}")


if __name__ == "__main__":
    main()
