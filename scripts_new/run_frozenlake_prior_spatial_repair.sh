#!/usr/bin/env bash
set -euo pipefail

ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-11}"
RESUME="${RESUME:-0}"
PHASE="${PHASE:-all}"
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
SOURCE_WM="${SOURCE_WM:-${ROOT}/checkpoints/frozenlake/qwen_vjepa_teacher_seed${SEED}/stage1_prior_repair_lora/best.pt}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/frozenlake/vagen_official_qwen_vjepa_teacher_3B/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/frozenlake/qwen_vjepa_teacher_seed${SEED}/stage1_prior_spatial_repair_v3}"
DATA_EPISODES="${DATA_EPISODES:-10000}"
DATA_SEED_START="${DATA_SEED_START:-1}"

[[ "${RESUME}" == "0" || "${RESUME}" == "1" ]] || { echo "RESUME must be 0 or 1" >&2; exit 2; }
[[ "${PHASE}" == "train" || "${PHASE}" == "audit" || "${PHASE}" == "all" ]] || {
  echo "PHASE must be train, audit, or all" >&2
  exit 2
}
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${SOURCE_WM}" ]] || { echo "Missing source prior checkpoint: ${SOURCE_WM}" >&2; exit 2; }
[[ -f "${DATA_DIR}/manifest.jsonl" ]] || { echo "Missing paired replay: ${DATA_DIR}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing Qwen model: ${MODEL_PATH}" >&2; exit 2; }

export PYTHONPATH="${ROOT}:/personal/jiayu2026/code/MBRL${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_DIR}"

resume_args=()
tee_args=()
if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
  if [[ "${RESUME}" == "1" ]]; then
    resume_args+=(--resume)
    tee_args+=(-a)
  elif [[ -e "${OUTPUT_DIR}/latest_adapter.pt" || -e "${OUTPUT_DIR}/latest.pt" ]]; then
    echo "Refusing to overwrite existing spatial-repair artifacts: ${OUTPUT_DIR}" >&2
    echo "Set RESUME=1 to continue adapter training, or choose a new OUTPUT_DIR." >&2
    exit 2
  fi
fi
wandb_args=()
if [[ "${NO_WANDB:-0}" == "1" ]]; then wandb_args+=(--no-wandb); fi

echo "========================================================"
echo " FrozenLake Qwen WM: audit-aligned prior-LoRA repair v3"
echo "========================================================"
echo "source          : ${SOURCE_WM}"
echo "data            : ${DATA_DIR}"
echo "output          : ${OUTPUT_DIR}"
echo "phase           : ${PHASE}"
echo "trainable       : wm_prior LoRA only"
echo "selection       : weakest(prior/${PRIOR_GATE:-0.75}, counterfactual/${COUNTERFACTUAL_GATE:-0.70})"
echo "probe           : frozen flattened-slot linear probe (same form as final audit)"
echo "checkpoints     : latest_adapter.pt + best_adapter.pt; one final full checkpoint"
echo "W&B             : ${WANDB_PROJECT:-jiayu-mbrl}/${WANDB_GROUP:-frozenlake_qwen_vjepa_spatial_prior_v3}"

"${PYTHON_BIN}" -u "${ROOT}/scripts/validate_qwen_vjepa_dataset.py" \
  --data-dir "${DATA_DIR}" \
  --env-id frozenlake \
  --expected-episodes "${DATA_EPISODES}" \
  --expected-seed-start "${DATA_SEED_START}" \
  --check-samples "${DATA_CHECK_SAMPLES:-16}" \
  --output "${OUTPUT_DIR}/dataset_contract_report.json" \
  --enforce

run_train() {
  "${PYTHON_BIN}" -u "${ROOT}/scripts/frozenlake/run_qwen_vjepa_spatial_prior_repair.py" \
  --wm-checkpoint "${SOURCE_WM}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 --seed "${SEED}" \
  --probe-steps "${PROBE_STEPS:-1000}" \
  --repair-steps "${REPAIR_STEPS:-3000}" \
  --eval-every "${EVAL_EVERY:-250}" \
  --probe-train-episodes "${PROBE_TRAIN_EPISODES:-1000}" \
  --probe-validation-episodes "${PROBE_VALIDATION_EPISODES:-500}" \
  --selection-validation-episodes "${SELECTION_VALIDATION_EPISODES:-256}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${PRIOR_LR:-3e-6}" \
  --posterior-gate "${POSTERIOR_GATE:-0.98}" \
  --prior-gate "${PRIOR_GATE:-0.75}" \
  --counterfactual-gate "${COUNTERFACTUAL_GATE:-0.70}" \
  --changed-gate "${CHANGED_GATE:-0.60}" \
  --noop-gate "${NOOP_GATE:-0.70}" \
  --minimum-action-gate "${MINIMUM_ACTION_GATE:-0.55}" \
  --actual-position-weight "${ACTUAL_POSITION_WEIGHT:-1.0}" \
  --counterfactual-position-weight "${COUNTERFACTUAL_POSITION_WEIGHT:-1.0}" \
  --posterior-latent-weight "${POSTERIOR_LATENT_WEIGHT:-1.0}" \
  --posterior-cosine-weight "${POSTERIOR_COSINE_WEIGHT:-1.0}" \
  --posterior-delta-weight "${POSTERIOR_DELTA_WEIGHT:-0.50}" \
  --vjepa-prior-weight "${VJEPA_PRIOR_WEIGHT:-0.25}" \
  --vjepa-delta-weight "${VJEPA_DELTA_WEIGHT:-0.50}" \
  --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
  --wandb-group "${WANDB_GROUP:-frozenlake_qwen_vjepa_spatial_prior_v3}" \
  --wandb-run-name "${WANDB_RUN_NAME:-frozenlake_spatial_prior_v3_seed${SEED}}" \
  "${resume_args[@]}" "${wandb_args[@]}" \
    2>&1 | tee "${tee_args[@]}" "${OUTPUT_DIR}/train.log"
}

run_audit() {
  [[ -f "${OUTPUT_DIR}/best.pt" ]] || {
    echo "Spatial repair did not release best.pt; inspect ${OUTPUT_DIR}/repair_report.json" >&2
    exit 1
  }
  echo "=== Independent held-out spatial/counterfactual audit ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full --env-id frozenlake \
    --data-dir "${DATA_DIR}" \
    --wm-checkpoint "${OUTPUT_DIR}/best.pt" \
    --world-model-mode frozen_wm \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 --belief-slots 36 --independent-backbone \
    --encoder-type qwen \
    --action-conditioning-mode embedded \
    --posterior-grounding-mode visual_anchor \
    --posterior-action-free \
    --posterior-recurrent-residual-scale 0.25 \
    --posterior-observation-residual-scale 0 \
    --prior-isolation-mode lora \
    --vjepa-teacher-prior-coef 0.50 \
    --vjepa-teacher-posterior-coef 0 \
    --vjepa-teacher-delta-coef 1.0 \
    --frozenlake-spatial-audit \
    --spatial-audit-train-episodes "${AUDIT_TRAIN_EPISODES:-1000}" \
    --spatial-audit-val-episodes "${AUDIT_VALIDATION_EPISODES:-500}" \
    --spatial-audit-probe-steps "${AUDIT_PROBE_STEPS:-1000}" \
    --spatial-audit-output "${OUTPUT_DIR}/stage1_semantic_gate_report.json" \
    --spatial-audit-enforce-gate \
    --spatial-audit-min-posterior "${AUDIT_POSTERIOR_GATE:-0.90}" \
    --spatial-audit-min-prior "${AUDIT_PRIOR_GATE:-0.75}" \
    --spatial-audit-min-counterfactual "${AUDIT_COUNTERFACTUAL_GATE:-0.70}" \
    --spatial-audit-min-baseline-multiple "${AUDIT_BASELINE_MULTIPLE:-2.0}" \
    --checkpoint-dir "${OUTPUT_DIR}" \
    --device cuda:0 --seed "${SEED}" --no-wandb \
    2>&1 | tee "${OUTPUT_DIR}/audit.log"
}

if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then run_train; fi
if [[ "${PHASE}" == "audit" || "${PHASE}" == "all" ]]; then run_audit; fi
