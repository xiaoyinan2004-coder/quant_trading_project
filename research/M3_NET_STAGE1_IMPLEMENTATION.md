# M3-Net Stage 1 Implementation

## What is implemented

The architecture document is now implemented as a runnable Stage 1 system that fits the current repository and the current machine:

- `models/m3net/model.py`
  - `M3NetStage1Model`
- `models/m3net/sequence.py`
  - `SequenceFeatureBuilder`
  - `SequenceExpertModel`
- `models/m3net/memory.py`
  - `MarketMemoryBank`
- `models/m3net/router.py`
  - `AdaptiveExpertRouter`
- `scripts/train_m3net_stage1.py`
  - local training entrypoint

## Stage 1 architecture mapping

This implementation covers the first practical slice of M3-Net:

1. Tabular expert
   - Reuses the existing `LightGBM / XGBoost` factor model.
   - Handles cross-sectional daily alpha.

2. Sequence expert
   - Builds time-series features from daily bars.
   - Optionally injects minute-level summaries such as intraday return, volatility, VWAP gap, and session strength.
   - Uses a CPU-friendly sklearn gradient boosting regressor.

3. Memory
   - Stores rolling market state summaries by date.
   - Provides regime features to the router.

4. Router
   - Fuses tabular and sequence outputs.
   - Gives more weight to sequence signals when minute features and intraday volatility are informative.

## Why this version first

This project is not yet at the stage where a full MLLM + MoE + World Model stack is the best first move. The implemented Stage 1 is intentionally:

- trainable on a normal laptop
- compatible with the existing factor pipeline
- able to absorb minute-level data now
- easy to benchmark against the pure `LightGBM / XGBoost` baseline

## Current limitations

The following architecture layers are still future work:

- text / MLLM branch
- true sparse MoE experts
- latent world model rollout
- causal modeling layer
- neuro-symbolic rule engine
- execution RL policy

## Recommended next implementation order

1. Add a dedicated minute alpha feature store.
2. Add rolling monthly retraining and model registry.
3. Add a text/event encoder branch.
4. Replace heuristic routing with a trainable gating model.
5. Add portfolio constraints and execution-aware backtest integration.
