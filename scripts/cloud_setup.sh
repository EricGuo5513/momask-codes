#!/bin/bash
# RunPod setup script — run once after the pod starts
# Usage: bash scripts/cloud_setup.sh
set -e

echo "=== MoMask Semantic Spectrum — Cloud Setup ==="

# ── 1. System deps ────────────────────────────────────────────────────
echo "[1/4] Installing system dependencies..."
apt-get update -q && apt-get install -y -q ffmpeg rsync

# ── 2. Python deps ────────────────────────────────────────────────────
echo "[2/4] Installing Python dependencies..."
pip install -q \
    einops==0.6.1 \
    ftfy==6.1.1 \
    scipy \
    scikit-learn \
    trimesh \
    smplx==0.1.28 \
    vector-quantize-pytorch==1.6.30 \
    gdown==4.7.1 \
    tensorboard \
    tqdm

pip install -q git+https://github.com/openai/CLIP.git

# ── 3. Dataset symlink ────────────────────────────────────────────────
# Expects data to have been uploaded to /workspace/data/
echo "[3/4] Linking dataset..."
mkdir -p dataset
if [ -d "/workspace/data/HumanML3D" ]; then
    ln -sfn /workspace/data/HumanML3D dataset/HumanML3D
    echo "  Linked /workspace/data/HumanML3D -> dataset/HumanML3D"
else
    echo "  WARNING: /workspace/data/HumanML3D not found — upload data first"
fi

# ── 4. Checkpoint symlink ─────────────────────────────────────────────
echo "[4/4] Linking checkpoints..."
mkdir -p checkpoints/t2m
if [ -d "/workspace/data/checkpoints/t2m/RVQVAE" ]; then
    ln -sfn /workspace/data/checkpoints/t2m/RVQVAE checkpoints/t2m/RVQVAE
    echo "  Linked RVQVAE checkpoint"
else
    echo "  WARNING: /workspace/data/checkpoints/t2m/RVQVAE not found"
fi

echo ""
echo "=== Setup complete. Run: bash scripts/train_all.sh ==="
