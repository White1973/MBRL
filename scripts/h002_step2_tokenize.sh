#!/bin/bash
# h002 Step 2: Precompute V-JEPA 2 visual tokens for collected episodes
# Input: data/sokoban_10k/raw/ (from step 1)
# Output: data/sokoban_10k/tokenized/ (~10000 .pt files + manifest.jsonl)
# Expected time: ~5-10 minutes on H200 (GPU)
# Supports resume: re-running skips already-tokenized episodes
set -euo pipefail

REPO_DIR="$HOME/MBRL-JEPA"
VENV="$HOME/venvs/mbrlppo"
DATA_DIR="$REPO_DIR/data/sokoban_10k"

cd "$REPO_DIR"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "=== h002 Step 2: Precomputing V-JEPA 2 tokens ==="
echo "Input: $DATA_DIR"
echo "Model: facebook/vjepa2-vitg-fpc64-384-ssv2"
echo "Batch size: 64"
echo ""

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_tokens.py \
    --raw-dir "$DATA_DIR" \
    --out-dir "$DATA_DIR" \
    --batch-size 64 \
    --device cuda \
    --hidden-dim 3584

echo ""
echo "=== Step 2 complete ==="
echo "Next: bash scripts/h002_step3_train.sh"
