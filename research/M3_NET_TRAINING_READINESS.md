# M3-Net Training Readiness

## Machine profile

Checked on `2026-04-01` in the current environment:

- Laptop: `RedmiBook 14`
- CPU: `Intel Core i5-10210U`
- Cores / Threads: `4 / 8`
- RAM: `7.83 GB`
- GPU 1: `Intel UHD Graphics`
- GPU 2: `NVIDIA GeForce MX250`
- Dedicated VRAM: about `2 GB`
- Free disk: `C: 29.48 GB`, `D: 278.65 GB`

## What the machine can support well

- `LightGBM` training
- `XGBoost` training
- sklearn sequence models
- daily factor research
- moderate minute-feature engineering
- small rolling backtests
- small PyTorch experiments if you later install `torch`

## What the machine does not support well

- large transformer pretraining
- large multimodal LLM training
- serious MoE training
- world-model training with large latent models
- BloombergGPT / FinGPT class full-model training
- long GPU-heavy reinforcement learning experiments

## Practical conclusion

Yes, this computer can support the currently implemented training path:

- `M3-Net Stage 1`
- `LightGBM / XGBoost` factor models
- CPU-friendly sequence expert
- minute-level summary features

But it should be treated as a research and prototyping machine, not a heavy deep learning training workstation.

## Suggested model tiers for this machine

### Safe to run locally

- daily cross-sectional `LightGBM`
- daily cross-sectional `XGBoost`
- sklearn sequence experts
- compact graph features after offline preprocessing
- small causal uplift / treatment-effect prototypes

### Possible with care

- small PyTorch `PatchTST` or `iTransformer` prototypes
- LoRA or parameter-efficient fine-tuning on tiny financial text models
- lightweight PPO policy experiments on reduced state spaces

### Better moved to cloud or a stronger workstation

- true multimodal LLM training
- MoE routing with large expert capacity
- world-model training over large minute data corpora
- end-to-end M3-Net full stack training

## Immediate recommendation

Use this laptop for:

1. feature engineering
2. baseline model training
3. Stage 1 fusion experiments
4. small ablations
5. backtests

If we later add `MLLM + MoE + World Model`, the clean path is to keep data prep and research local, and move heavy training to a cloud GPU environment.
