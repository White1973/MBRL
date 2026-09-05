#!/usr/bin/env bash
set -euo pipefail

# Qwen + V-JEPA formal PPO with conservative alternating WM refresh.
# This is a separate experiment branch; the frozen-WM runner and checkpoints
# remain unchanged. The first pilot stops at 150 accepted Actor updates.
ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
SEED="${SEED:-1}"

export WORLD_MODEL_MODE=alternating_wm
export ACTOR_LR="${ACTOR_LR:-2e-7}"
export ACTOR_UPDATES="${ACTOR_UPDATES:-500}"
export EVAL_EVERY="${EVAL_EVERY:-10}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"

# Refresh only the action-conditioned prior LoRA. The Qwen posterior and the
# released Reward Head retain their Stage-1/2 coordinate system.
export WM_REFRESH_EVERY="${WM_REFRESH_EVERY:-50}"
export WM_REFRESH_UPDATES="${WM_REFRESH_UPDATES:-1}"
export WM_REFRESH_BATCH_SIZE="${WM_REFRESH_BATCH_SIZE:-16}"
export WM_REFRESH_LR="${WM_REFRESH_LR:-1e-6}"
export WM_REFRESH_WARMUP_STEPS="${WM_REFRESH_WARMUP_STEPS:-0}"
export WM_REFRESH_GRAD_CLIP="${WM_REFRESH_GRAD_CLIP:-0.5}"
export WM_REFRESH_VALIDATION_BATCHES="${WM_REFRESH_VALIDATION_BATCHES:-8}"
export WM_OPEN_DYNAMICS_COEF="${WM_OPEN_DYNAMICS_COEF:-0.25}"
export WM_PRIOR_REWARD_COEF="${WM_PRIOR_REWARD_COEF:-0.1}"
# Reuse the immutable Stage-2 split: refresh only on reward-train episodes,
# monitor on reward-validation, and never consume calibration/official test.
export WM_REFRESH_USE_REWARD_SPLITS=1

export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage4_ppo_alternating_wm_slow_lr2e7}"
export WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}"
export WANDB_GROUP="${WANDB_GROUP:-qwen_vjepa_ppo_alternating_wm}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_vjepa_ppo_alternating_wm_slow_lr2e7_seed${SEED}}"

exec bash "${ROOT}/scripts/run_qwen_vjepa_ppo.sh"
