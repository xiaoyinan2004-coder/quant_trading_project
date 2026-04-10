# M3Net v2-lite Research Workflow

## Goal

This workflow keeps normal development on the local machine and uses the cloud only when real data preparation, rolling evaluation, or model training is needed.

## Local vs Cloud

### Run locally

- read and modify code
- add features
- add scripts
- add tests
- run unit tests on synthetic data
- inspect experiment outputs after they are copied back

### Run on cloud

- bulk download real A-share daily data
- bulk download real A-share minute data
- rolling research evaluation
- real model training and result generation

## Local checklist

From the project root:

```bash
pytest tests/test_prepare_m3net_data.py -q
pytest tests/test_m3net_stage1.py -q
pytest tests/test_run_m3net_research.py -q
pytest tests/test_gradient_boosting_factor.py -q
```

These checks validate:

- data preparation helpers
- M3Net v2-lite model wiring
- unified rolling research runner
- baseline factor model pipeline

## Cloud workflow

### 1. Enter project root

```bash
cd /workspace/quant_trading_project
```

### 2. Prepare research data

Start with a small symbol set first:

```bash
python scripts/prepare_m3net_data.py \
  --symbols 000001 000002 600000 600036 600519 601318 601888 000858 002415 300750 \
            688981 600276 000333 002594 002475 601012 600900 000651 300059 601166 \
  --daily-start 2022-01-01 \
  --daily-end 2026-04-01 \
  --minute-start "2026-01-01 09:30:00" \
  --minute-end "2026-04-01 15:00:00" \
  --minute-period 15 \
  --output-root data \
  --use-cache
```

Expected output folders:

- `data/daily`
- `data/minute`

Quick check:

```bash
find data/daily -maxdepth 1 -type f | head
find data/minute -maxdepth 1 -type f | head
```

### 3. Run unified rolling research

```bash
python scripts/run_m3net_research.py \
  --daily-dir data/daily \
  --minute-dir data/minute \
  --backend lightgbm \
  --rebalance-freq M \
  --top-n 20 \
  --output-dir artifacts/research
```

This writes:

- `artifacts/research/rolling_periods.csv`
- `artifacts/research/rolling_picks.csv`
- `artifacts/research/summary.csv`

## What to bring back from cloud

Copy or paste back:

- the printed `summary.csv` table
- the first 20 lines of `rolling_periods.csv`
- any error stack trace if the run fails

Helpful commands:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv('artifacts/research/summary.csv').to_string(index=False))
print()
print(pd.read_csv('artifacts/research/rolling_periods.csv').head(20).to_string(index=False))
PY
```

## Recommended experiment order

1. Small symbol set sanity check
2. Expand to a broader stock pool
3. Compare `baseline_factor` vs `m3net_v2_lite`
4. Check ablations:
   `m3net_no_learned_router`
   `m3net_no_minute`
5. Decide next feature or label upgrade based on the gaps

## How to interpret results

### Good signal

- `m3net_v2_lite` beats `baseline_factor` on average realized return
- `m3net_v2_lite` beats at least one ablation consistently
- win rate stays stable across rebalance periods
- fused RankIC is not worse than tabular RankIC

### Warning signal

- M3Net only wins in-sample metrics but not realized rolling returns
- learned router is not better than heuristic router
- removing minute features does not hurt results
- performance depends on only one or two periods

## Current next upgrades after first cloud results

When the first real rolling results come back, prioritize one of these:

1. better labels
2. stronger minute features
3. more robust router features
4. portfolio-level constraints and transaction-cost evaluation
