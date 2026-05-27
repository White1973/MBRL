#!/usr/bin/env bash
# Unified Sokoban entrypoint:
#   1. optional offline WM warm-up on tokenized data
#   2. online alternating WM + policy training
#
# Examples:
#   OFFLINE_WARMUP_STEPS=2000 ./e2e_alternating_wm.sh
#   OFFLINE_WARMUP_STEPS=500 WM_REFRESH_DATA_MODE=offline_only ./e2e_alternating_wm.sh
#   OFFLINE_WARMUP_STEPS=0 RUN_SUFFIX=wm0 ./e2e_alternating_wm.sh
#   EXISTING_WM_CKPT=checkpoints/warmup-qwen-wm500/latest.pt RUN_SUFFIX=wm500-reuse ./e2e_alternating_wm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

# ── offline warm-up steps (0 = no offline warm-up) ──
OFFLINE_WARMUP_STEPS="${OFFLINE_WARMUP_STEPS:-${TOTAL_STEPS:-2000}}"

# ── naming ──
RUN_SUFFIX="${RUN_SUFFIX:-wm${OFFLINE_WARMUP_STEPS}}"
WARMUP_RUN_NAME="${WARMUP_RUN_NAME:-warmup-qwen-${RUN_SUFFIX}}"
ONLINE_RUN_NAME="${ONLINE_RUN_NAME:-online-alternating-${RUN_SUFFIX}}"
WANDB_GROUP="${WANDB_GROUP:-sokoban-${RUN_SUFFIX}}"
export WANDB_GROUP

WARMUP_CKPT_DIR="checkpoints/${WARMUP_RUN_NAME}"
ONLINE_CKPT_DIR="checkpoints/${ONLINE_RUN_NAME}"
EXISTING_WM_CKPT="${EXISTING_WM_CKPT:-}"

# ── Phase 2 settings ──
TOTAL_UPDATES="${TOTAL_UPDATES:-2000}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-8}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-8}"
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-50}"
USE_VJEPA_EVAL="${USE_VJEPA_EVAL:-1}"

WM_REFRESH_EVERY="${WM_REFRESH_EVERY:-1}"
WM_REFRESH_UPDATES="${WM_REFRESH_UPDATES:-4}"
WM_REFRESH_BATCH_SIZE="${WM_REFRESH_BATCH_SIZE:-4}"
WM_REFRESH_LR="${WM_REFRESH_LR:-1e-4}"
WM_REFRESH_HORIZON="${WM_REFRESH_HORIZON:-8}"
WM_REFRESH_DATA_MODE="${WM_REFRESH_DATA_MODE:-mixed}"

COLLECT_EVERY="${COLLECT_EVERY:-20}"
COLLECT_EPISODES="${COLLECT_EPISODES:-4}"
COLLECT_MAX_STEPS="${COLLECT_MAX_STEPS:-0}"

# ════════════════════════════════════════════
# Offline WM warm-up (skip if OFFLINE_WARMUP_STEPS=0)
# ════════════════════════════════════════════
if [[ -n "${EXISTING_WM_CKPT}" ]]; then
  echo "══ Reusing existing WM checkpoint: ${EXISTING_WM_CKPT} ══"
  WM_CKPT_ARG="--wm-checkpoint ${EXISTING_WM_CKPT}"
elif [[ "${OFFLINE_WARMUP_STEPS}" -gt 0 ]]; then
  echo "══ Offline WM warm-up: ${OFFLINE_WARMUP_STEPS} steps ══"
  LAUNCH_MODE=sync launch "${WARMUP_RUN_NAME}" python -u "${ROOT_DIR}/scripts/train_phase1.py" \
    --mode offline \
    --data-dir "${DATA_DIR}" \
    --backbone qwen \
    --total-steps "${OFFLINE_WARMUP_STEPS}" \
    --device cuda \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WARMUP_RUN_NAME}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-job-type offline_warmup \
    --checkpoint-dir "${WARMUP_CKPT_DIR}" \
    --encoder-type qwen \
    --action-conditioning-mode text \
    --use-sigreg \
    --use-ema-target \
    --reward-readout mean_pool \
    --reward-supervision posterior
  WM_CKPT_ARG="--wm-checkpoint ${WARMUP_CKPT_DIR}/latest.pt"
else
  echo "══ Skipping offline WM warm-up ══"
  WM_CKPT_ARG=""
fi

# ════════════════════════════════════════════
# Online alternating training
# ════════════════════════════════════════════
echo "══ Online alternating training: ${TOTAL_UPDATES} PPO updates ══"

cmd=(python -u "${ROOT_DIR}/scripts/train_phase2.py"
  --data-dir "${DATA_DIR}"
  ${WM_CKPT_ARG}
  --checkpoint-dir "${ONLINE_CKPT_DIR}"
  --device cuda
  --backbone qwen
  --action-conditioning-mode text
  --policy-family vlm_action_head
  --world-model-mode alternating_wm
  --total-updates "${TOTAL_UPDATES}"
  --rollout-batch-size "${ROLLOUT_BATCH}"
  --rollout-horizon "${ROLLOUT_HORIZON}"
  --no-normalize-advantages
  --wm-refresh-every "${WM_REFRESH_EVERY}"
  --wm-refresh-updates "${WM_REFRESH_UPDATES}"
  --wm-refresh-data-mode "${WM_REFRESH_DATA_MODE}"
  --wm-refresh-batch-size "${WM_REFRESH_BATCH_SIZE}"
  --wm-refresh-lr "${WM_REFRESH_LR}"
  --wm-refresh-horizon "${WM_REFRESH_HORIZON}"
  --collect-every "${COLLECT_EVERY}"
  --collect-episodes "${COLLECT_EPISODES}"
  --collect-max-steps "${COLLECT_MAX_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --eval-episodes "${EVAL_EPISODES}"
  --eval-max-steps "${EVAL_MAX_STEPS}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-run-name "${ONLINE_RUN_NAME}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-job-type online_alternating
)

if [[ "${USE_VJEPA_EVAL}" == "1" ]]; then
  cmd+=(--use-vjepa-eval)
fi

LAUNCH_MODE=sync launch "${ONLINE_RUN_NAME}" "${cmd[@]}"
