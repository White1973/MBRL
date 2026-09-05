#!/usr/bin/env bash
set -euo pipefail

ROOT="${MBRL0901_ROOT:-MBRL0901}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-20260902}"
PHASE="${PHASE:-all}"  # train | audit | all

SOURCE_WM="${SOURCE_WM:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_lora/best.pt}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/sokoban_10k_qwen_vjepa_teacher_3B/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed1/stage1_prior_repair_spatial_v2}"

REPAIR_STEPS="${REPAIR_STEPS:-1500}"
REPAIR_BATCH_SIZE="${REPAIR_BATCH_SIZE:-4}"
REPAIR_LR="${REPAIR_LR:-5e-6}"
REPAIR_EVAL_EVERY="${REPAIR_EVAL_EVERY:-250}"
REPAIR_EVAL_EPISODES="${REPAIR_EVAL_EPISODES:-64}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-1}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-4}"

case "${PHASE}" in
  train|audit|all) ;;
  *) echo "PHASE must be train, audit, or all; got ${PHASE}" >&2; exit 2 ;;
esac

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${DATA_DIR}/manifest.jsonl" ]] || { echo "Missing paired replay: ${DATA_DIR}" >&2; exit 2; }

export PYTHONPATH="${ROOT}:/personal/jiayu2026/code/MBRL${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
  [[ -f "${SOURCE_WM}" ]] || { echo "Missing v1 prior checkpoint: ${SOURCE_WM}" >&2; exit 2; }
  if [[ -e "${OUTPUT_DIR}/latest.pt" || -e "${OUTPUT_DIR}/best.pt" ]]; then
    echo "Refusing to overwrite existing v2 checkpoints: ${OUTPUT_DIR}" >&2
    echo "Choose a new OUTPUT_DIR or run PHASE=audit." >&2
    exit 2
  fi
  mkdir -p "${OUTPUT_DIR}"
  echo "=== Position-aware Qwen prior repair v2 ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/run_qwen_vjepa_prior_repair_v2.py" \
    --wm-checkpoint "${SOURCE_WM}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --seed "${SEED}" \
    --repair-steps "${REPAIR_STEPS}" \
    --batch-size "${REPAIR_BATCH_SIZE}" \
    --lr "${REPAIR_LR}" \
    --eval-every "${REPAIR_EVAL_EVERY}" \
    --eval-episodes "${REPAIR_EVAL_EPISODES}" \
    --teacher-epochs "${TEACHER_EPOCHS}" \
    --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
    --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
    --wandb-group "${WANDB_GROUP:-qwen_vjepa_prior_repair_v2}" \
    --wandb-run-name "${WANDB_RUN_NAME:-qwen_vjepa_prior_repair_v2_s1}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

if [[ "${PHASE}" == "audit" || "${PHASE}" == "all" ]]; then
  [[ -f "${OUTPUT_DIR}/best.pt" ]] || {
    echo "Missing v2 best checkpoint: ${OUTPUT_DIR}/best.pt" >&2; exit 2;
  }
  echo "=== Independent official Stage-1 semantic Gate ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/audit_qwen_vjepa_teacher_wm.py" \
    --wm-checkpoint "${OUTPUT_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_DIR}/stage1_semantic_gate_report.json" \
    --device cuda:0 \
    --batch-size "${AUDIT_BATCH_SIZE:-4}" \
    --probe-epochs "${AUDIT_PROBE_EPOCHS:-1}" \
    --enforce-gate \
    2>&1 | tee "${OUTPUT_DIR}/audit.log"
fi
