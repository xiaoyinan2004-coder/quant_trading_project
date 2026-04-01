#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/quant_trading_project}"
BRANCH="${BRANCH:-main}"
BACKEND="${BACKEND:-lightgbm}"
DAILY_DIR="${DAILY_DIR:-data/processed/daily}"
MINUTE_DIR="${MINUTE_DIR:-data/processed/minute}"
OUTPUT_PATH="${OUTPUT_PATH:-artifacts/m3net_stage1.joblib}"

cd "${PROJECT_DIR}"

if [ ! -d ".git" ]; then
  echo "Not a git repository: ${PROJECT_DIR}"
  exit 1
fi

echo "[1/4] Pulling latest code"
git pull origin "${BRANCH}"

echo "[2/4] Activating environment"
source .venv/bin/activate

echo "[3/4] Ensuring artifact directory exists"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "[4/4] Starting training"
python scripts/train_m3net_stage1.py \
  --daily-dir "${DAILY_DIR}" \
  --minute-dir "${MINUTE_DIR}" \
  --backend "${BACKEND}" \
  --output "${OUTPUT_PATH}"

