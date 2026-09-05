#!/usr/bin/env bash
set -euo pipefail

# Qwen-native semantic-belief WM recipe.
#
# Observation input: native Qwen2.5-VL visual embeddings.
# Teacher target: frozen compressed V-JEPA features stored separately in the
# paired replay.  This script never enables V-JEPA as the posterior input and
# does not add world queries, BEV heads, or multi-frame query tokens.

ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
PHASE="${PHASE:-all}"                 # prepare | train | audit | all

RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/data/sokoban_10k_raw}"
PAIRED_ROOT="${PAIRED_ROOT:-${ROOT}/data/sokoban_10k_qwen_vjepa_teacher_3B}"
DATA_DIR="${DATA_DIR:-${PAIRED_ROOT}/tokenized}"
# Empty by default: full preparation re-extracts native Qwen features from
# raw RGB. Set this only for an independently provenance-audited Qwen cache;
# the builder still verifies trajectory/frame/action/reward/seed alignment.
QWEN_TOKEN_SOURCE="${QWEN_TOKEN_SOURCE:-}"

WM_DIR="${WM_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage1_wm}"
WM_STEPS="${WM_STEPS:-2000}"
WM_LR="${WM_LR:-1e-5}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-4}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-50}"
WM_VAL_BATCHES="${WM_VAL_BATCHES:-32}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-8}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-4}"
AUDIT_PROBE_EPOCHS="${AUDIT_PROBE_EPOCHS:-1}"

# These are loss weights, not gates.  The teacher gate below checks that each
# enabled branch has held-out samples and a finite loss; checkpoint selection
# ranks the same weighted objective rather than latent dynamics alone.
VJEPA_PRIOR_COEF="${VJEPA_PRIOR_COEF:-0.50}"
VJEPA_POSTERIOR_COEF="${VJEPA_POSTERIOR_COEF:-0.10}"
VJEPA_DELTA_COEF="${VJEPA_DELTA_COEF:-0.25}"

case "${PHASE}" in
  prepare|train|audit|all) ;;
  *) echo "PHASE must be prepare, train, audit, or all; got ${PHASE}" >&2; exit 2 ;;
esac

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing Qwen model: ${MODEL_PATH}" >&2; exit 2; }
if [[ "${PHASE}" == "prepare" || "${PHASE}" == "all" ]]; then
  [[ -f "${RAW_DATA_DIR}/raw/manifest.jsonl" ]] || {
    echo "Missing raw RGB manifest: ${RAW_DATA_DIR}/raw/manifest.jsonl" >&2; exit 2;
  }
fi

export PYTHONPATH="${ROOT}:/personal/jiayu2026/code/MBRL${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ "${PHASE}" == "prepare" || "${PHASE}" == "all" ]]; then
  if [[ -f "${DATA_DIR}/manifest.jsonl" ]]; then
    echo "Paired replay already exists; preserving it: ${DATA_DIR}/manifest.jsonl"
  else
    prepare_args=(
      --raw-data-dir "${RAW_DATA_DIR}"
      --output-dir "${PAIRED_ROOT}"
      --backbone-model "${MODEL_PATH}"
      --hidden-dim 2048
      --belief-slots 36
      --device cuda:0
      --frame-batch-size "${TOKENIZE_BATCH_SIZE}"
    )
    if [[ -n "${QWEN_TOKEN_SOURCE}" ]]; then
      prepare_args+=(--qwen-token-dir "${QWEN_TOKEN_SOURCE}")
    fi
    echo "=== Build paired Qwen-input / V-JEPA-teacher replay ==="
    "${PYTHON_BIN}" -u "${ROOT}/scripts/build_qwen_vjepa_teacher_dataset.py" \
      "${prepare_args[@]}"
  fi
fi

if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
  [[ -f "${DATA_DIR}/manifest.jsonl" ]] || {
    echo "Missing paired replay: ${DATA_DIR}/manifest.jsonl" >&2; exit 2;
  }
  if [[ -e "${WM_DIR}/latest.pt" || -e "${WM_DIR}/best.pt" ]]; then
    echo "Refusing to overwrite existing WM checkpoints: ${WM_DIR}" >&2
    echo "Choose a new WM_DIR to preserve a reversible experiment history." >&2
    exit 2
  fi
  mkdir -p "${WM_DIR}"
  echo "=== Train Qwen-native WM with frozen V-JEPA future-state teacher ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full \
    --env-id sokoban \
    --data-dir "${DATA_DIR}" \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 \
    --belief-slots 36 \
    --encoder-type qwen \
    --action-conditioning-mode embedded \
    --posterior-grounding-mode visual_anchor \
    --posterior-action-free \
    --posterior-recurrent-residual-scale 0.25 \
    --observation-anchor-coef 0 \
    --observation-delta-anchor-coef 0 \
    --vjepa-teacher-prior-coef "${VJEPA_PRIOR_COEF}" \
    --vjepa-teacher-posterior-coef "${VJEPA_POSTERIOR_COEF}" \
    --vjepa-teacher-delta-coef "${VJEPA_DELTA_COEF}" \
    --world-model-mode alternating_wm \
    --wm-action-id-offset 0 \
    --wm-refresh-lr "${WM_LR}" \
    --wm-refresh-warmup-steps 100 \
    --wm-refresh-batch-size "${WM_BATCH_SIZE}" \
    --wm-refresh-horizon 2 \
    --wm-open-loop-horizon 0 \
    --wm-open-dynamics-coef 0 \
    --wm-prior-reward-coef 0 \
    --wm-refresh-reward-loss-coef 0 \
    --wm-refresh-freeze-reward-head \
    --wm-delta-cosine-coef 0 \
    --wm-inverse-action-coef 0 \
    --wm-only-refresh-steps "${WM_STEPS}" \
    --wm-only-eval-every "${WM_EVAL_EVERY}" \
    --wm-only-val-batches "${WM_VAL_BATCHES}" \
    --wm-only-out-checkpoint "${WM_DIR}/latest.pt" \
    --checkpoint-dir "${WM_DIR}" \
    --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
    --wandb-group "${WANDB_GROUP:-qwen_vjepa_teacher_wm}" \
    --wandb-run-name "${WANDB_RUN_NAME:-qwen_vjepa_teacher_s${SEED}}" \
    --device cuda:0 \
    --seed "${SEED}"
fi

if [[ "${PHASE}" == "audit" || "${PHASE}" == "all" ]]; then
  [[ -f "${DATA_DIR}/manifest.jsonl" ]] || {
    echo "Missing paired replay: ${DATA_DIR}/manifest.jsonl" >&2; exit 2;
  }
  [[ -f "${WM_DIR}/best.pt" ]] || {
    echo "Missing best WM checkpoint: ${WM_DIR}/best.pt" >&2; exit 2;
  }
  echo "=== Held-out Stage-1 spatial / counterfactual dynamics gate ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/audit_qwen_vjepa_teacher_wm.py" \
    --wm-checkpoint "${WM_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --output "${WM_DIR}/stage1_semantic_gate_report.json" \
    --device cuda:0 \
    --batch-size "${AUDIT_BATCH_SIZE}" \
    --probe-epochs "${AUDIT_PROBE_EPOCHS}" \
    --enforce-gate
fi
