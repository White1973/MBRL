#!/bin/bash
# h002 Step 1: Collect 10000 Sokoban episodes with structured metadata
# Strategy mix: expert:3, noisy_expert:3, random:3, near_success:1
# Expected output: data/sokoban_10k/raw/ (~10000 .pt files + manifest.jsonl)
# Expected time: ~10 minutes on CPU
set -euo pipefail

REPO_DIR="$HOME/MBRL-JEPA"
VENV="$HOME/venvs/mbrlppo"
DATA_DIR="$REPO_DIR/data/sokoban_10k"

cd "$REPO_DIR"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "=== h002 Step 1: Collecting 10000 Sokoban episodes ==="
echo "Output: $DATA_DIR"
echo "Strategy mix: expert:3, noisy_expert:3, random:3, near_success:1"
echo "Env: 6x6, 1 box, max 25 steps, BFS depth 20"
echo ""

python scripts/collect_sokoban.py \
    --num_episodes 10000 \
    --output_dir "$DATA_DIR" \
    --seed 42 \
    --max_steps 25 \
    --split train \
    --dim_room 6 6 \
    --num_boxes 1 \
    --bfs_max_depth 20 \
    --noise_prob 0.3 \
    --near_success_remaining 3

echo ""
echo "=== Step 1 complete ==="
echo "Next: bash scripts/h002_step2_tokenize.sh"
