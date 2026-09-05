#!/usr/bin/env bash
set -euo pipefail

# FrozenLake Stage 2 configuration for the shared Qwen + V-JEPA-teacher
# Reward-Head trainer.  This script never regenerates replay or updates the WM.

ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
ENGINE_ROOT="${MBRL_ENGINE_ROOT:-/personal/jiayu2026/code/MBRL}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-11}"

STAGE1_DIR="${STAGE1_DIR:-${ROOT}/checkpoints/frozenlake/qwen_vjepa_teacher_seed${SEED}/stage1_prior_spatial_repair_v3}"
SOURCE_WM="${SOURCE_WM:-${STAGE1_DIR}/best.pt}"
STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT:-${STAGE1_DIR}/stage1_semantic_gate_report.json}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/frozenlake/vagen_official_qwen_vjepa_teacher_3B/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/frozenlake/qwen_vjepa_teacher_seed${SEED}/stage2_reward_head_v3}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${ROOT}/data/frozenlake/reward_head_feature_cache/qwen_vjepa_teacher_seed${SEED}_stage1_v3}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${SOURCE_WM}" ]] || { echo "Missing Stage-1 best checkpoint: ${SOURCE_WM}" >&2; exit 2; }
[[ -f "${STAGE1_GATE_REPORT}" ]] || { echo "Missing Stage-1 Gate report: ${STAGE1_GATE_REPORT}" >&2; exit 2; }
[[ -f "${DATA_DIR}/manifest.jsonl" ]] || { echo "Missing paired replay: ${DATA_DIR}" >&2; exit 2; }
if [[ -e "${OUTPUT_DIR}/latest.pt" || -e "${OUTPUT_DIR}/best.pt" ]]; then
  echo "Refusing to overwrite Reward-Head checkpoints: ${OUTPUT_DIR}" >&2
  echo "Choose a new OUTPUT_DIR. Existing compatible feature caches can be reused." >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
if [[ "${NO_WANDB:-0}" != "1" ]]; then
  export REQUIRE_WANDB=1
  export WANDB_MODE="${WANDB_MODE:-online}"
fi
mkdir -p "${OUTPUT_DIR}" "${FEATURE_CACHE_DIR}"

echo "========================================================"
echo " FrozenLake Stage 2: shared compact Reward Head (v3 WM)"
echo "========================================================"
echo "source best.pt : ${SOURCE_WM}"
echo "Stage-1 Gate   : ${STAGE1_GATE_REPORT}"
echo "paired replay  : ${DATA_DIR}"
echo "output          : ${OUTPUT_DIR}"
echo "feature cache   : ${FEATURE_CACHE_DIR}"
echo "trainable       : Reward Head only; WM and V-JEPA teacher frozen"
echo "W&B             : ${WANDB_PROJECT:-jiayu-mbrl}/${WANDB_GROUP:-frozenlake_qwen_vjepa_reward_head_v3}"

"${PYTHON_BIN}" -u "${ROOT}/scripts/validate_qwen_vjepa_dataset.py" \
  --data-dir "${DATA_DIR}" \
  --env-id frozenlake \
  --expected-episodes "${DATA_EPISODES:-10000}" \
  --expected-seed-start "${DATA_SEED_START:-1}" \
  --check-samples "${DATA_CHECK_SAMPLES:-16}" \
  --output "${OUTPUT_DIR}/dataset_contract_report.json" \
  --enforce

ENV_ID=frozenlake \
SOURCE_DIR="${STAGE1_DIR}" \
SOURCE_WM="${SOURCE_WM}" \
STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT}" \
DATA_DIR="${DATA_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
GPU_ID="${GPU_ID}" \
SEED="${SEED}" \
HORIZONS="${HORIZONS:-1 2 3 4 5 6 7 8}" \
VALIDATION_EPISODES="${VALIDATION_EPISODES:-1000}" \
CALIBRATION_EPISODES="${CALIBRATION_EPISODES:-1000}" \
TRAIN_WINDOWS_PER_EPISODE="${TRAIN_WINDOWS_PER_EPISODE:-2}" \
EVAL_WINDOWS_PER_EPISODE="${EVAL_WINDOWS_PER_EPISODE:-1}" \
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8}" \
EPOCHS="${EPOCHS:-50}" \
PATIENCE="${PATIENCE:-10}" \
BATCH_SIZE="${BATCH_SIZE:-256}" \
HEAD_LR="${HEAD_LR:-1e-4}" \
TARGET_CALIBRATION_PRECISION="${TARGET_CALIBRATION_PRECISION:-0.75}" \
MIN_CALIBRATION_RECALL="${MIN_CALIBRATION_RECALL:-0.10}" \
MIN_OFFICIAL_AUC="${MIN_OFFICIAL_AUC:-0.75}" \
MIN_OFFICIAL_AP="${MIN_OFFICIAL_AP:-0.35}" \
MIN_OFFICIAL_PRECISION="${MIN_OFFICIAL_PRECISION:-0.70}" \
MIN_OFFICIAL_RECALL="${MIN_OFFICIAL_RECALL:-0.10}" \
MAX_OFFICIAL_FPR="${MAX_OFFICIAL_FPR:-0.10}" \
MIN_HORIZON_AUC="${MIN_HORIZON_AUC:-0.65}" \
NO_WANDB="${NO_WANDB:-0}" \
WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
WANDB_GROUP="${WANDB_GROUP:-frozenlake_qwen_vjepa_reward_head_v3}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-frozenlake_qwen_vjepa_reward_head_v3_seed${SEED}}" \
  bash "${ROOT}/scripts/run_qwen_vjepa_reward_head.sh"
