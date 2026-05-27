#!/bin/bash
# h002 Step 3: Train Phase 1 world model on Sokoban 10k dataset
# Input: data/sokoban_10k/tokenized/ (from step 2)
# Output: checkpoints/h002/ (model checkpoints)
# Expected time: depends on --total-steps (default 10000)
# Supports resume: pass --resume <checkpoint_path>
set -euo pipefail

REPO_DIR="$HOME/MBRL-JEPA"
VENV="$HOME/venvs/mbrlppo"
DATA_DIR="$REPO_DIR/data/sokoban_10k/tokenized"
CKPT_DIR="$REPO_DIR/checkpoints/h002"

cd "$REPO_DIR"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Parse optional arguments
TOTAL_STEPS="${1:-10000}"
RESUME_PATH="${2:-}"

echo "=== h002 Step 3: Phase 1 World Model Training ==="
echo "Data: $DATA_DIR"
echo "Checkpoints: $CKPT_DIR"
echo "Total steps: $TOTAL_STEPS"
echo "Backbone: Qwen2.5-VL-7B + LoRA"
if [ -n "$RESUME_PATH" ]; then
    echo "Resuming from: $RESUME_PATH"
fi
echo ""

RESUME_FLAG=""
if [ -n "$RESUME_PATH" ]; then
    RESUME_FLAG="--resume $RESUME_PATH"
fi

CUDA_VISIBLE_DEVICES=0 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "$DATA_DIR" \
    --backbone qwen \
    --total-steps "$TOTAL_STEPS" \
    --checkpoint-dir "$CKPT_DIR" \
    --device cuda \
    --wandb-project mbrl-vlm \
    --wandb-run-name "h002-sokoban-phase1-${TOTAL_STEPS}steps" \
    $RESUME_FLAG

echo ""
echo "=== Step 3 complete ==="
echo "Checkpoints saved to: $CKPT_DIR"
