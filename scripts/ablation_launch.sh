#!/usr/bin/env bash
# Launch 6 ablation experiments on GPUs 0,1,4,5,6,7
# Each experiment runs as a background process with its own log file.
#
# Experiments (Round 1A):
#   GPU 0: default       — vjepa2 + bidirectional + sigreg (baseline)
#   GPU 1: causal        — vjepa2 + causal + sigreg
#   GPU 4: ema           — vjepa2 + bidirectional + ema_teacher
#   GPU 5: ema_var       — vjepa2 + bidirectional + ema_teacher + variance_reg
#   GPU 6: no_curriculum — vjepa2 + bidirectional + sigreg, fixed H=4
#   GPU 7: default_seed2 — vjepa2 + bidirectional + sigreg, seed=2

set -euo pipefail

REPO_DIR="${HOME}/MBRL-JEPA"
VENV="${HOME}/venvs/mbrlppo/bin/activate"
DATA_DIR="${REPO_DIR}/data/sokoban_10k/tokenized"
LOG_DIR="${REPO_DIR}/logs/ablation_round1"
TOTAL_STEPS=5000
PYTHONPATH="${REPO_DIR}"

cd "${REPO_DIR}"
source "${VENV}"
export PYTHONPATH
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Ablation Round 1A — 6 experiments"
echo "Data: ${DATA_DIR}"
echo "Steps: ${TOTAL_STEPS}"
echo "Logs: ${LOG_DIR}"
echo "============================================"

# --- GPU 0: Default baseline ---
echo "[GPU 0] Launching: default (bidirectional + sigreg)"
CUDA_VISIBLE_DEVICES=0 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps ${TOTAL_STEPS} \
    --checkpoint-dir "${REPO_DIR}/checkpoints/default" \
    --wandb-project mbrl-vlm \
    --wandb-run-name "default_bidir_sigreg" \
    --attention-mode bidirectional \
    --anti-collapse sigreg \
    --seed 1 \
    --val-every 200 \
    --device cuda \
    > "${LOG_DIR}/default.log" 2>&1 &
echo "  PID: $!"

# --- GPU 1: Causal attention ---
echo "[GPU 1] Launching: causal (causal + sigreg)"
CUDA_VISIBLE_DEVICES=1 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps ${TOTAL_STEPS} \
    --checkpoint-dir "${REPO_DIR}/checkpoints/causal" \
    --wandb-project mbrl-vlm \
    --wandb-run-name "causal_sigreg" \
    --attention-mode causal \
    --anti-collapse sigreg \
    --seed 1 \
    --val-every 200 \
    --device cuda \
    > "${LOG_DIR}/causal.log" 2>&1 &
echo "  PID: $!"

# --- GPU 4: EMA teacher ---
echo "[GPU 4] Launching: ema_teacher (bidirectional + ema)"
CUDA_VISIBLE_DEVICES=4 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps ${TOTAL_STEPS} \
    --checkpoint-dir "${REPO_DIR}/checkpoints/ema_teacher" \
    --wandb-project mbrl-vlm \
    --wandb-run-name "ema_teacher" \
    --attention-mode bidirectional \
    --anti-collapse ema_teacher \
    --seed 1 \
    --val-every 200 \
    --device cuda \
    > "${LOG_DIR}/ema_teacher.log" 2>&1 &
echo "  PID: $!"

# --- GPU 5: EMA teacher + variance reg ---
echo "[GPU 5] Launching: ema_teacher_var (bidirectional + ema + var)"
CUDA_VISIBLE_DEVICES=5 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps ${TOTAL_STEPS} \
    --checkpoint-dir "${REPO_DIR}/checkpoints/ema_teacher_var" \
    --wandb-project mbrl-vlm \
    --wandb-run-name "ema_teacher_var" \
    --attention-mode bidirectional \
    --anti-collapse ema_teacher_var \
    --seed 1 \
    --val-every 200 \
    --device cuda \
    > "${LOG_DIR}/ema_teacher_var.log" 2>&1 &
echo "  PID: $!"

# --- GPU 6: No curriculum (fixed H=4) ---
echo "[GPU 6] Launching: no_curriculum (bidirectional + sigreg, H=4 fixed)"
CUDA_VISIBLE_DEVICES=6 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps ${TOTAL_STEPS} \
    --checkpoint-dir "${REPO_DIR}/checkpoints/no_curriculum" \
    --wandb-project mbrl-vlm \
    --wandb-run-name "no_curriculum_h4" \
    --attention-mode bidirectional \
    --anti-collapse sigreg \
    --no-curriculum \
    --seed 1 \
    --val-every 200 \
    --device cuda \
    > "${LOG_DIR}/no_curriculum.log" 2>&1 &
echo "  PID: $!"

# --- GPU 7: Default with seed=2 ---
echo "[GPU 7] Launching: default_seed2 (bidirectional + sigreg, seed=2)"
CUDA_VISIBLE_DEVICES=7 python scripts/train_phase1.py \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps ${TOTAL_STEPS} \
    --checkpoint-dir "${REPO_DIR}/checkpoints/default_seed2" \
    --wandb-project mbrl-vlm \
    --wandb-run-name "default_bidir_sigreg_seed2" \
    --attention-mode bidirectional \
    --anti-collapse sigreg \
    --seed 2 \
    --val-every 200 \
    --device cuda \
    > "${LOG_DIR}/default_seed2.log" 2>&1 &
echo "  PID: $!"

echo ""
echo "All 6 experiments launched. Monitor with:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  nvidia-smi"
echo ""
echo "wandb dashboard: https://wandb.ai project=mbrl-vlm"
