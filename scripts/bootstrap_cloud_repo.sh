#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:?Usage: bash scripts/bootstrap_cloud_repo.sh <repo_url> [target_dir]}"
TARGET_DIR="${2:-$HOME/quant_trading_project}"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "Repository already exists at ${TARGET_DIR}"
else
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

cd "${TARGET_DIR}"
bash scripts/setup_cloud_env.sh "${TARGET_DIR}"

echo "Cloud repository bootstrapped at ${TARGET_DIR}"
