#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/quant_trading_project}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/5] Checking project directory: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

echo "[2/5] Creating virtual environment: ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[3/5] Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

echo "[4/5] Installing base project dependencies"
pip install -r requirements.txt

echo "[5/5] Cloud environment ready"
echo "If you need GPU PyTorch, install the correct CUDA wheel separately."
echo "Example:"
echo "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"

