#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (parent of this script's directory) and assets dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$PROJECT_ROOT/assets"
THIRDPARTIES_DIR="$PROJECT_ROOT/thirdparties"
mkdir -p "$ASSETS_DIR"
mkdir -p "$THIRDPARTIES_DIR"

clone_or_update_repo() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local path="$THIRDPARTIES_DIR/$name"

    if [ ! -d "$path/.git" ]; then
        echo "[download_assets] Cloning $name..."
        git clone "$url" "$path"
    fi
    git -C "$path" fetch --all --tags
    git -C "$path" checkout "$commit"
}

clone_or_update_repo \
    "unitree_sim_isaaclab" \
    "https://github.com/unitreerobotics/unitree_sim_isaaclab.git" \
    "e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc"
git -C "$THIRDPARTIES_DIR/unitree_sim_isaaclab" submodule update --init --depth 1

clone_or_update_repo \
    "IsaacLab" \
    "https://github.com/isaac-sim/IsaacLab.git" \
    "54a65ea830c6002e17dc18c77831fa60e43937bc"

clone_or_update_repo \
    "cyclonedds" \
    "https://github.com/eclipse-cyclonedds/cyclonedds.git" \
    "5041f3560c088c99e5088b2b8520b69169621196"

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
