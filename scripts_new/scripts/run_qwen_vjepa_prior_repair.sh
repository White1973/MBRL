#!/usr/bin/env bash
set -euo pipefail

# Isolated Stage-1 prior repair for the released Qwen-native WM.
# Posterior grounding remains on the frozen `default` Qwen WM LoRA. Only an
# independent `wm_prior` LoRA is optimized using held-out-safe dynamics and
# frozen V-JEPA future-state/delta teacher targets.

ROOT="${MBRL0901_ROOT:-MBRL0901}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
PHASE="${PHASE:-all}"                 # train | audit | all

SOURCE_WM="${SOURCE_WM:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage1_wm/best.pt}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/sokoban_10k_qwen_vjepa_teacher_3B/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage1_prior_repair_lora}"

REPAIR_STEPS="${REPAIR_STEPS:-3000}"
REPAIR_LR="${REPAIR_LR:-1e-5}"
REPAIR_BATCH_SIZE="${REPAIR_BATCH_SIZE:-4}"
REPAIR_EVAL_EVERY="${REPAIR_EVAL_EVERY:-100}"
REPAIR_VAL_BATCHES="${REPAIR_VAL_BATCHES:-32}"
REPAIR_WARMUP_STEPS="${REPAIR_WARMUP_STEPS:-100}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-4}"
AUDIT_PROBE_EPOCHS="${AUDIT_PROBE_EPOCHS:-1}"

# Changed-state supervision is intentionally stronger than absolute-state
# alignment: the released checkpoint already passed posterior spatial grounding
# but retained only ~25% exact player+box accuracy after a real action.
VJEPA_PRIOR_COEF="${VJEPA_PRIOR_COEF:-0.50}"
VJEPA_DELTA_COEF="${VJEPA_DELTA_COEF:-1.00}"
LATENT_DELTA_COEF="${LATENT_DELTA_COEF:-0.50}"
OPEN_DYNAMICS_COEF="${OPEN_DYNAMICS_COEF:-0.25}"

case "${PHASE}" in
  train|audit|all) ;;
  *) echo "PHASE must be train, audit, or all; got ${PHASE}" >&2; exit 2 ;;
esac

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing Qwen model: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${DATA_DIR}/manifest.jsonl" ]] || { echo "Missing paired replay: ${DATA_DIR}/manifest.jsonl" >&2; exit 2; }

export PYTHONPATH="${ROOT}:/personal/jiayu2026/code/MBRL${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
  [[ -f "${SOURCE_WM}" ]] || { echo "Missing source WM: ${SOURCE_WM}" >&2; exit 2; }
  if [[ -e "${OUTPUT_DIR}/latest.pt" || -e "${OUTPUT_DIR}/best.pt" ]]; then
    echo "Refusing to overwrite existing prior-repair checkpoints: ${OUTPUT_DIR}" >&2
    echo "Choose a new OUTPUT_DIR or run PHASE=audit." >&2
    exit 2
  fi
  mkdir -p "${OUTPUT_DIR}"
  echo "=== Repair action-conditioned prior with an isolated Qwen LoRA ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full \
    --env-id sokoban \
    --data-dir "${DATA_DIR}" \
    --wm-checkpoint "${SOURCE_WM}" \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 \
    --belief-slots 36 \
    --encoder-type qwen \
    --action-conditioning-mode embedded \
    --posterior-grounding-mode visual_anchor \
    --posterior-action-free \
    --posterior-recurrent-residual-scale 0.25 \
    --posterior-observation-residual-scale 0 \
    --prior-isolation-mode lora \
    --observation-anchor-coef 0 \
    --observation-delta-anchor-coef 0 \
    --vjepa-teacher-prior-coef "${VJEPA_PRIOR_COEF}" \
    --vjepa-teacher-posterior-coef 0 \
    --vjepa-teacher-delta-coef "${VJEPA_DELTA_COEF}" \
    --world-model-mode alternating_wm \
    --wm-action-id-offset 0 \
    --wm-refresh-prior-lora-only \
    --wm-refresh-lr "${REPAIR_LR}" \
    --wm-refresh-warmup-steps "${REPAIR_WARMUP_STEPS}" \
    --wm-refresh-batch-size "${REPAIR_BATCH_SIZE}" \
    --wm-refresh-horizon 2 \
    --wm-open-loop-horizon 2 \
    --wm-open-dynamics-coef "${OPEN_DYNAMICS_COEF}" \
    --wm-prior-reward-coef 0 \
    --wm-refresh-reward-loss-coef 0 \
    --wm-refresh-freeze-reward-head \
    --wm-delta-cosine-coef "${LATENT_DELTA_COEF}" \
    --wm-inverse-action-coef 0 \
    --wm-only-refresh-steps "${REPAIR_STEPS}" \
    --wm-only-eval-every "${REPAIR_EVAL_EVERY}" \
    --wm-only-val-batches "${REPAIR_VAL_BATCHES}" \
    --wm-only-out-checkpoint "${OUTPUT_DIR}/latest.pt" \
    --checkpoint-dir "${OUTPUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
    --wandb-group "${WANDB_GROUP:-qwen_vjepa_prior_repair}" \
    --wandb-run-name "${WANDB_RUN_NAME:-qwen_vjepa_prior_repair_s${SEED}}" \
    --device cuda:0 \
    --seed "${SEED}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

if [[ "${PHASE}" == "audit" || "${PHASE}" == "all" ]]; then
  [[ -f "${OUTPUT_DIR}/best.pt" ]] || {
    echo "Missing repaired best checkpoint: ${OUTPUT_DIR}/best.pt" >&2; exit 2;
  }
  echo "=== Held-out Stage-1 spatial / counterfactual dynamics gate ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/audit_qwen_vjepa_teacher_wm.py" \
    --wm-checkpoint "${OUTPUT_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_DIR}/stage1_semantic_gate_report.json" \
    --device cuda:0 \
    --batch-size "${AUDIT_BATCH_SIZE}" \
    --probe-epochs "${AUDIT_PROBE_EPOCHS}" \
    --enforce-gate \
    2>&1 | tee "${OUTPUT_DIR}/audit.log"
fi
