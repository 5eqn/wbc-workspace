#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (parent of this script's directory) and assets dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$PROJECT_ROOT/assets"
mkdir -p "$ASSETS_DIR"
cd "$ASSETS_DIR"

# --- hf (huggingface_hub) ---
if ! command -v hf &>/dev/null; then
    echo "[download_assets] 'hf' not found — attempting pip install huggingface_hub..."
    if pip install huggingface_hub &>/dev/null; then
        echo "[download_assets] huggingface_hub installed successfully."
    else
        echo "[download_assets] ERROR: failed to install huggingface_hub. Please install it manually:" >&2
        echo "  pip install huggingface_hub" >&2
        exit 1
    fi
fi

echo "[download_assets] Downloading HoloMotion_models..."
hf download HorizonRobotics/HoloMotion_models --local-dir HoloMotion_models

echo "[download_assets] Downloading GEAR-SONIC..."
hf download nvidia/GEAR-SONIC --local-dir GEAR-SONIC \
    --include "model_decoder.onnx" \
    --include "model_encoder.onnx" \
    --include "observation_config.yaml"

# --- gdown ---
if ! command -v gdown &>/dev/null; then
    echo "[download_assets] 'gdown' not found — attempting pip install gdown..."
    if pip install gdown &>/dev/null; then
        echo "[download_assets] gdown installed successfully."
    else
        echo "[download_assets] ERROR: failed to install gdown. Please install it manually:" >&2
        echo "  pip install gdown" >&2
        exit 1
    fi
fi

echo "[download_assets] Downloading Google Drive file..."
gdown https://drive.google.com/file/d/1p-JiYnfDlD9XnjaODpYqj1BLsgBPQjDQ/view?usp=share_link

echo "[download_assets] Done. Assets downloaded to $ASSETS_DIR"
