#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_deps.sh — Install ML dependencies into /opt/pytorch venv
#
# The EC2 instance already has PyTorch 2.7.0+cu128 at /opt/pytorch.
# This script installs the remaining packages needed for DIA-GUARD training.
#
# Usage:
#   sudo bash install_deps.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PIP="/opt/pytorch/bin/pip"

echo "Installing DIA-GUARD training dependencies into /opt/pytorch..."
echo "PyTorch is already installed — skipping torch."

${PIP} install --no-cache-dir \
    transformers==4.51.0 \
    trl==0.17.0 \
    accelerate==1.0.0 \
    datasets==3.0.0 \
    peft==0.14.0 \
    deepspeed==0.14.4 \
    bitsandbytes==0.43.3 \
    sentencepiece==0.2.0 \
    safetensors>=0.4.3 \
    scipy==1.13.0 \
    scikit-learn==1.5.0 \
    pandas==2.2.2 \
    wandb \
    huggingface_hub>=0.30.0

echo ""
echo "Installing flash-attn (requires CUDA, may take a few minutes)..."
${PIP} install --no-cache-dir flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn install failed. Training will fall back to eager attention."

echo ""
echo "Done! Verify with:"
echo "  /opt/pytorch/bin/python -c \"import transformers, trl, peft, deepspeed; print('All OK')\""
