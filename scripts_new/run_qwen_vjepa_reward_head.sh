#!/usr/bin/env bash
set -euo pipefail

# Stage 2: train the existing compact Reward Head on the frozen, gated
# Qwen + V-JEPA-teacher World Model.  No visual data or WM is regenerated.

ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
ENGINE_ROOT="${MBRL_ENGINE_ROOT:-/personal/jiayu2026/code/MBRL}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-20260902}"
ENV_ID="${ENV_ID:-sokoban}"

SOURCE_DIR="${SOURCE_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_spatial_v2}"
SOURCE_WM="${SOURCE_WM:-${SOURCE_DIR}/best.pt}"
STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT:-${SOURCE_DIR}/stage1_semantic_gate_report.json}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/sokoban_10k_qwen_vjepa_teacher_3B/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed1/stage2_reward_head}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${ROOT}/data/reward_head_feature_cache/qwen_vjepa_teacher_seed1}"

HORIZONS="${HORIZONS:-1 2 3 4 5 6 7 8}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-1000}"
CALIBRATION_EPISODES="${CALIBRATION_EPISODES:-1000}"
TRAIN_WINDOWS_PER_EPISODE="${TRAIN_WINDOWS_PER_EPISODE:-2}"
EVAL_WINDOWS_PER_EPISODE="${EVAL_WINDOWS_PER_EPISODE:-1}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HEAD_LR="${HEAD_LR:-1e-4}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${SOURCE_WM}" ]] || { echo "Missing gated Stage-1 best.pt: ${SOURCE_WM}" >&2; exit 2; }
[[ -f "${STAGE1_GATE_REPORT}" ]] || { echo "Missing Stage-1 Gate report: ${STAGE1_GATE_REPORT}" >&2; exit 2; }
[[ -f "${DATA_DIR}/manifest.jsonl" ]] || { echo "Missing paired replay: ${DATA_DIR}" >&2; exit 2; }
[[ -f "${ENGINE_ROOT}/scripts/inject_reward_head.py" ]] || {
  echo "Missing established shared Reward-Head utilities under ${ENGINE_ROOT}" >&2
  exit 2
}
if [[ -e "${OUTPUT_DIR}/latest.pt" || -e "${OUTPUT_DIR}/best.pt" ]]; then
  echo "Refusing to overwrite Reward-Head checkpoints: ${OUTPUT_DIR}" >&2
  echo "Use a new OUTPUT_DIR; cached frozen-WM features remain reusable." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${FEATURE_CACHE_DIR}"
export PYTHONPATH="${ROOT}:${ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

echo "======================================================="
echo " Qwen + V-JEPA teacher: frozen-WM Reward Head training"
echo "======================================================="
echo "source WM    : ${SOURCE_WM}"
echo "environment  : ${ENV_ID}"
echo "Stage-1 Gate : ${STAGE1_GATE_REPORT}"
echo "paired replay: ${DATA_DIR}"
echo "output       : ${OUTPUT_DIR}"
echo "feature cache: ${FEATURE_CACHE_DIR}"
echo "horizons     : ${HORIZONS}"
echo "split        : Stage-1 train -> train/val/calibration; official 1000 -> test only"
echo "GPU          : physical ${GPU_ID} (process device cuda:0)"
echo "W&B          : ${WANDB_PROJECT:-jiayu-mbrl}/${WANDB_GROUP:-qwen_vjepa_reward_head}"

extra_args=()
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  extra_args+=(--no-wandb)
fi

# Intentional word splitting passes HORIZONS as separate argparse integers.
# shellcheck disable=SC2086
"${PYTHON_BIN}" -u "${ROOT}/scripts/train_qwen_vjepa_reward_head.py" \
  --env-id "${ENV_ID}" \
  --wm-checkpoint "${SOURCE_WM}" \
  --stage1-gate-report "${STAGE1_GATE_REPORT}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --engine-root "${ENGINE_ROOT}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --validation-episodes "${VALIDATION_EPISODES}" \
  --calibration-episodes "${CALIBRATION_EPISODES}" \
  --horizons ${HORIZONS} \
  --train-windows-per-episode "${TRAIN_WINDOWS_PER_EPISODE}" \
  --eval-windows-per-episode "${EVAL_WINDOWS_PER_EPISODE}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --feature-cache-dir "${FEATURE_CACHE_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${HEAD_LR}" \
  --head-hidden-dim 0 \
  --target-calibration-precision "${TARGET_CALIBRATION_PRECISION:-0.75}" \
  --min-calibration-recall "${MIN_CALIBRATION_RECALL:-0.10}" \
  --min-official-auc "${MIN_OFFICIAL_AUC:-0.75}" \
  --min-official-ap "${MIN_OFFICIAL_AP:-0.35}" \
  --min-official-precision "${MIN_OFFICIAL_PRECISION:-0.70}" \
  --min-official-recall "${MIN_OFFICIAL_RECALL:-0.10}" \
  --max-official-fpr "${MAX_OFFICIAL_FPR:-0.10}" \
  --min-horizon-auc "${MIN_HORIZON_AUC:-0.65}" \
  --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
  --wandb-group "${WANDB_GROUP:-qwen_vjepa_reward_head}" \
  --wandb-run-name "${WANDB_RUN_NAME:-qwen_vjepa_reward_head_seed1}" \
  "${extra_args[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
