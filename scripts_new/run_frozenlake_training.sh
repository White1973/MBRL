#!/usr/bin/env bash
set -euo pipefail

# FrozenLake task configuration for the same Qwen+V-JEPA-teacher MBRL stack
# used by Sokoban.  This file defines paths, gates, and hyperparameters only:
# it does not implement a FrozenLake-specific model or trainer.
#
# Model input:    native Qwen2.5-VL image embeddings in obs_tokens [T,36,2048]
# Teacher target: frozen V-JEPA features in semantic_teacher_tokens [T,36,1408]
# V-JEPA is never passed to the posterior as the observation input.

ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
ENGINE_ROOT="${MBRL_ENGINE_ROOT:-/personal/jiayu2026/code/MBRL}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
VAGEN_ROOT="${VAGEN_ROOT:-/personal/jiayu2026/code/VAGEN-new}"
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-11}"
PHASE="${PHASE:-all}"
WORLD_MODEL_MODE="${WORLD_MODEL_MODE:-alternating_wm}"

RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/data/frozenlake/vagen_official_rgb_seed1_10000}"
PAIRED_ROOT="${PAIRED_ROOT:-${ROOT}/data/frozenlake/vagen_official_qwen_vjepa_teacher_3B}"
DATA_DIR="${DATA_DIR:-${PAIRED_ROOT}/tokenized}"
DATA_REPORT="${DATA_REPORT:-${PAIRED_ROOT}/dataset_contract_report.json}"
EVAL_SEEDS_FILE="${EVAL_SEEDS_FILE:-${ROOT}/data/frozenlake/vagen_official_seeded/test_seeds.json}"

EXP_ROOT="${EXP_ROOT:-${ROOT}/checkpoints/frozenlake/qwen_vjepa_teacher_seed${SEED}}"
BASE_WM_DIR="${BASE_WM_DIR:-${EXP_ROOT}/stage1_wm}"
PRIOR_WM_DIR="${PRIOR_WM_DIR:-${EXP_ROOT}/stage1_prior_repair_lora}"
SPATIAL_PRIOR_DIR="${SPATIAL_PRIOR_DIR:-${EXP_ROOT}/stage1_prior_spatial_repair_v3}"
FINAL_WM_DIR="${FINAL_WM_DIR:-${SPATIAL_PRIOR_DIR}}"
STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT:-${FINAL_WM_DIR}/stage1_semantic_gate_report.json}"
REWARD_DIR="${REWARD_DIR:-${EXP_ROOT}/stage2_reward_head_v3}"
CRITIC_DIR="${CRITIC_DIR:-${EXP_ROOT}/stage3_critic_warmup}"
PPO_DIR="${PPO_DIR:-${EXP_ROOT}/stage4_ppo_${WORLD_MODEL_MODE}}"
H2_CACHE="${H2_CACHE:-${ROOT}/data/frozenlake/critic_h2_cache/qwen_vjepa_teacher_seed${SEED}/cache.pt}"
REWARD_FEATURE_CACHE="${REWARD_FEATURE_CACHE:-${ROOT}/data/frozenlake/reward_head_feature_cache/qwen_vjepa_teacher_seed${SEED}_stage1_v3}"

DATA_EPISODES="${DATA_EPISODES:-10000}"
DATA_SEED_START="${DATA_SEED_START:-1}"
WM_STEPS="${WM_STEPS:-2000}"
PRIOR_REPAIR_STEPS="${PRIOR_REPAIR_STEPS:-3000}"
ACTOR_UPDATES="${ACTOR_UPDATES:-1000}"

case "${PHASE}" in
  prepare|data_check|wm|prior|spatial_prior|wm_audit|reward|critic|ppo|all) ;;
  *)
    echo "PHASE must be prepare, data_check, wm, prior, spatial_prior, wm_audit, reward, critic, ppo, or all; got ${PHASE}" >&2
    exit 2
    ;;
esac
case "${WORLD_MODEL_MODE}" in
  frozen_wm|alternating_wm) ;;
  *) echo "WORLD_MODEL_MODE must be frozen_wm or alternating_wm" >&2; exit 2 ;;
esac

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing Qwen model: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${EVAL_SEEDS_FILE}" ]] || { echo "Missing VAGEN evaluation seeds: ${EVAL_SEEDS_FILE}" >&2; exit 2; }

export PYTHONPATH="${ROOT}:${ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VAGEN_ROOT
export PYTHONUNBUFFERED=1

wandb_args=()
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  wandb_args+=(--no-wandb)
else
  export REQUIRE_WANDB=1
  export WANDB_MODE="${WANDB_MODE:-online}"
fi

validate_data() {
  "${PYTHON_BIN}" -u "${ROOT}/scripts/validate_qwen_vjepa_dataset.py" \
    --data-dir "${DATA_DIR}" \
    --env-id frozenlake \
    --expected-episodes "${DATA_EPISODES}" \
    --expected-seed-start "${DATA_SEED_START}" \
    --check-samples "${DATA_CHECK_SAMPLES:-16}" \
    --output "${DATA_REPORT}" \
    --enforce
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing required artifact: $1" >&2; exit 2; }
}

reward_floor() {
  if [[ -n "${REWARD_CONFIDENCE_FLOOR:-}" ]]; then
    printf '%s\n' "${REWARD_CONFIDENCE_FLOOR}"
    return
  fi
  "${PYTHON_BIN}" -c 'import json,sys; print(float(json.load(open(sys.argv[1]))["decision_threshold"]))' \
    "${REWARD_DIR}/reward_head_gate_report.json"
}

run_prepare() {
  local paired_count
  if [[ -f "${DATA_DIR}/manifest.jsonl" ]]; then
    paired_count="$(wc -l < "${DATA_DIR}/manifest.jsonl")"
    if [[ "${paired_count}" == "${DATA_EPISODES}" ]]; then
      echo "Paired replay has ${paired_count} episodes; validating instead of regenerating."
      validate_data
      return
    fi
    echo "Paired replay is partial (${paired_count}/${DATA_EPISODES}); resuming preparation."
  fi

  echo "=== Collect/resume VAGEN-distribution FrozenLake RGB replay: seeds 1..10000 ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/collect_offline_dataset.py" \
    --env-id frozenlake \
    --output-dir "${RAW_DATA_DIR}" \
    --num-episodes "${DATA_EPISODES}" \
    --seed-start "${DATA_SEED_START}" \
    --max-steps 25 \
    --split train \
    --strategy value_iteration_expert:1 \
    --strategy random:1

  echo "=== Extract native Qwen observations and separate frozen V-JEPA teachers ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/build_qwen_vjepa_teacher_dataset.py" \
    --raw-data-dir "${RAW_DATA_DIR}" \
    --output-dir "${PAIRED_ROOT}" \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 \
    --belief-slots 36 \
    --device cuda:0 \
    --frame-batch-size "${TOKENIZE_BATCH_SIZE:-8}"
  validate_data
}

run_base_wm() {
  validate_data
  if [[ -e "${BASE_WM_DIR}/latest.pt" || -e "${BASE_WM_DIR}/best.pt" ]]; then
    echo "Refusing to overwrite base WM checkpoints: ${BASE_WM_DIR}" >&2
    echo "Choose a new BASE_WM_DIR, or continue with PHASE=wm_audit after prior repair." >&2
    exit 2
  fi
  mkdir -p "${BASE_WM_DIR}"
  echo "=== FrozenLake Stage 1A: shared Qwen-native WM + frozen V-JEPA teacher ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full --env-id frozenlake \
    --data-dir "${DATA_DIR}" \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 --belief-slots 36 --independent-backbone \
    --encoder-type qwen \
    --action-conditioning-mode embedded \
    --posterior-grounding-mode visual_anchor \
    --posterior-action-free \
    --posterior-recurrent-residual-scale 0.25 \
    --observation-anchor-coef 0 \
    --observation-delta-anchor-coef 0 \
    --vjepa-teacher-prior-coef "${VJEPA_PRIOR_COEF:-0.50}" \
    --vjepa-teacher-posterior-coef "${VJEPA_POSTERIOR_COEF:-0.10}" \
    --vjepa-teacher-delta-coef "${VJEPA_DELTA_COEF:-0.25}" \
    --world-model-mode alternating_wm \
    --wm-action-id-offset 0 \
    --wm-refresh-lr "${WM_LR:-1e-5}" \
    --wm-refresh-warmup-steps "${WM_WARMUP_STEPS:-100}" \
    --wm-refresh-batch-size "${WM_BATCH_SIZE:-4}" \
    --wm-refresh-horizon 2 \
    --wm-open-loop-horizon 0 \
    --wm-open-dynamics-coef 0 \
    --wm-prior-reward-coef 0 \
    --wm-refresh-reward-loss-coef 0 \
    --wm-refresh-freeze-reward-head \
    --wm-delta-cosine-coef 0 \
    --wm-inverse-action-coef 0 \
    --wm-only-refresh-steps "${WM_STEPS}" \
    --wm-only-eval-every "${WM_EVAL_EVERY:-50}" \
    --wm-only-val-batches "${WM_VAL_BATCHES:-32}" \
    --wm-only-out-checkpoint "${BASE_WM_DIR}/latest.pt" \
    --checkpoint-dir "${BASE_WM_DIR}" \
    --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
    --wandb-group "${WANDB_GROUP_WM:-frozenlake_qwen_vjepa_wm}" \
    --wandb-run-name "frozenlake_qwen_vjepa_wm_seed${SEED}" \
    --device cuda:0 --seed "${SEED}" \
    "${wandb_args[@]}" \
    2>&1 | tee "${BASE_WM_DIR}/train.log"

}

run_prior() {
  validate_data
  require_file "${BASE_WM_DIR}/best.pt"
  if [[ -e "${PRIOR_WM_DIR}/latest.pt" || -e "${PRIOR_WM_DIR}/best.pt" ]]; then
    echo "Refusing to overwrite prior-repair checkpoints: ${PRIOR_WM_DIR}" >&2
    exit 2
  fi
  mkdir -p "${PRIOR_WM_DIR}"
  echo "=== FrozenLake Stage 1B: repair only the shared WM prior LoRA ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full --env-id frozenlake \
    --data-dir "${DATA_DIR}" \
    --wm-checkpoint "${BASE_WM_DIR}/best.pt" \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 --belief-slots 36 --independent-backbone \
    --encoder-type qwen \
    --action-conditioning-mode embedded \
    --posterior-grounding-mode visual_anchor \
    --posterior-action-free \
    --posterior-recurrent-residual-scale 0.25 \
    --posterior-observation-residual-scale 0 \
    --prior-isolation-mode lora \
    --observation-anchor-coef 0 \
    --observation-delta-anchor-coef 0 \
    --vjepa-teacher-prior-coef "${PRIOR_VJEPA_COEF:-0.50}" \
    --vjepa-teacher-posterior-coef 0 \
    --vjepa-teacher-delta-coef "${PRIOR_VJEPA_DELTA_COEF:-1.0}" \
    --world-model-mode alternating_wm \
    --wm-action-id-offset 0 \
    --wm-refresh-prior-lora-only \
    --wm-refresh-lr "${PRIOR_REPAIR_LR:-1e-5}" \
    --wm-refresh-warmup-steps "${PRIOR_REPAIR_WARMUP_STEPS:-100}" \
    --wm-refresh-batch-size "${PRIOR_REPAIR_BATCH_SIZE:-4}" \
    --wm-refresh-horizon 2 \
    --wm-open-loop-horizon 2 \
    --wm-open-dynamics-coef "${PRIOR_OPEN_DYNAMICS_COEF:-0.25}" \
    --wm-prior-reward-coef 0 \
    --wm-refresh-reward-loss-coef 0 \
    --wm-refresh-freeze-reward-head \
    --wm-delta-cosine-coef "${PRIOR_LATENT_DELTA_COEF:-0.50}" \
    --wm-inverse-action-coef 0 \
    --wm-only-refresh-steps "${PRIOR_REPAIR_STEPS}" \
    --wm-only-eval-every "${PRIOR_REPAIR_EVAL_EVERY:-100}" \
    --wm-only-val-batches "${PRIOR_REPAIR_VAL_BATCHES:-32}" \
    --wm-only-out-checkpoint "${PRIOR_WM_DIR}/latest.pt" \
    --checkpoint-dir "${PRIOR_WM_DIR}" \
    --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
    --wandb-group "${WANDB_GROUP_PRIOR:-frozenlake_qwen_vjepa_prior_repair}" \
    --wandb-run-name "frozenlake_qwen_vjepa_prior_seed${SEED}" \
    --device cuda:0 --seed "${SEED}" \
    "${wandb_args[@]}" \
    2>&1 | tee "${PRIOR_WM_DIR}/train.log"
}

run_spatial_prior() {
  validate_data
  require_file "${PRIOR_WM_DIR}/best.pt"
  SOURCE_WM="${PRIOR_WM_DIR}/best.pt" \
  DATA_DIR="${DATA_DIR}" OUTPUT_DIR="${SPATIAL_PRIOR_DIR}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" PHASE=train \
  RESUME="${SPATIAL_PRIOR_RESUME:-0}" \
  NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_SPATIAL_PRIOR:-frozenlake_qwen_vjepa_spatial_prior_v3}" \
  WANDB_RUN_NAME="frozenlake_spatial_prior_v3_seed${SEED}" \
    bash "${ROOT}/scripts/run_frozenlake_prior_spatial_repair.sh"
}

run_wm_audit() {
  validate_data
  require_file "${FINAL_WM_DIR}/best.pt"
  echo "=== FrozenLake Stage-1 held-out spatial/counterfactual Gate ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full --env-id frozenlake \
    --data-dir "${DATA_DIR}" \
    --wm-checkpoint "${FINAL_WM_DIR}/best.pt" \
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
    --vjepa-teacher-prior-coef "${PRIOR_VJEPA_COEF:-0.50}" \
    --vjepa-teacher-posterior-coef 0 \
    --vjepa-teacher-delta-coef "${PRIOR_VJEPA_DELTA_COEF:-1.0}" \
    --frozenlake-spatial-audit \
    --spatial-audit-train-episodes "${SPATIAL_AUDIT_TRAIN_EPISODES:-1000}" \
    --spatial-audit-val-episodes "${SPATIAL_AUDIT_VAL_EPISODES:-500}" \
    --spatial-audit-probe-steps "${SPATIAL_AUDIT_PROBE_STEPS:-1000}" \
    --spatial-audit-output "${STAGE1_GATE_REPORT}" \
    --spatial-audit-enforce-gate \
    --spatial-audit-min-posterior "${SPATIAL_AUDIT_MIN_POSTERIOR:-0.90}" \
    --spatial-audit-min-prior "${SPATIAL_AUDIT_MIN_PRIOR:-0.75}" \
    --spatial-audit-min-counterfactual "${SPATIAL_AUDIT_MIN_COUNTERFACTUAL:-0.70}" \
    --spatial-audit-min-baseline-multiple "${SPATIAL_AUDIT_MIN_BASELINE_MULTIPLE:-2.0}" \
    --checkpoint-dir "${FINAL_WM_DIR}" \
    --device cuda:0 --seed "${SEED}" --no-wandb \
    2>&1 | tee "${FINAL_WM_DIR}/audit.log"
}

run_reward() {
  validate_data
  require_file "${FINAL_WM_DIR}/best.pt"
  require_file "${STAGE1_GATE_REPORT}"
  ENV_ID=frozenlake \
  SOURCE_DIR="${FINAL_WM_DIR}" \
  SOURCE_WM="${FINAL_WM_DIR}/best.pt" \
  STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT}" \
  DATA_DIR="${DATA_DIR}" \
  OUTPUT_DIR="${REWARD_DIR}" \
  FEATURE_CACHE_DIR="${REWARD_FEATURE_CACHE}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" \
  NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_REWARD:-frozenlake_qwen_vjepa_reward_head}" \
  WANDB_RUN_NAME="frozenlake_qwen_vjepa_reward_seed${SEED}" \
    bash "${ROOT}/scripts/run_qwen_vjepa_reward_head.sh"
}

run_critic() {
  validate_data
  require_file "${REWARD_DIR}/best.pt"
  require_file "${REWARD_DIR}/reward_head_gate_report.json"
  local floor
  floor="$(reward_floor)"
  ENV_ID=frozenlake PHASE=all RESUME="${CRITIC_RESUME:-0}" \
  SOURCE_WM="${REWARD_DIR}/best.pt" REWARD_DIR="${REWARD_DIR}" \
  DATA_DIR="${DATA_DIR}" H2_CACHE="${H2_CACHE}" OUTPUT_DIR="${CRITIC_DIR}" \
  REWARD_CONFIDENCE_FLOOR="${floor}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_CRITIC:-frozenlake_qwen_vjepa_critic_warmup}" \
  WANDB_RUN_NAME="frozenlake_qwen_vjepa_critic_seed${SEED}" \
    bash "${ROOT}/scripts/run_qwen_vjepa_critic_warmup.sh"
}

run_ppo() {
  validate_data
  require_file "${REWARD_DIR}/best.pt"
  require_file "${CRITIC_DIR}/best.pt"
  require_file "${CRITIC_DIR}/warmup_report.json"
  local floor
  floor="$(reward_floor)"
  ENV_ID=frozenlake WORLD_MODEL_MODE="${WORLD_MODEL_MODE}" RESUME="${PPO_RESUME:-0}" \
  REWARD_CKPT="${REWARD_DIR}/best.pt" \
  CRITIC_DIR="${CRITIC_DIR}" CRITIC_CKPT="${CRITIC_DIR}/best.pt" \
  CRITIC_REPORT="${CRITIC_DIR}/warmup_report.json" \
  DATA_DIR="${DATA_DIR}" EVAL_SEEDS_FILE="${EVAL_SEEDS_FILE}" \
  EVAL_EPISODES=128 OUTPUT_DIR="${PPO_DIR}" \
  REWARD_CONFIDENCE_FLOOR="${floor}" ACTOR_UPDATES="${ACTOR_UPDATES}" \
  TOTAL_UPDATES="${TOTAL_UPDATES:-2020}" EVAL_EVERY="${EVAL_EVERY:-10}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" ACTOR_LR="${ACTOR_LR:-2e-6}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_PPO:-frozenlake_qwen_vjepa_ppo}" \
  WANDB_RUN_NAME="frozenlake_qwen_vjepa_ppo_${WORLD_MODEL_MODE}_seed${SEED}" \
    bash "${ROOT}/scripts/run_qwen_vjepa_ppo.sh"
}

echo "========================================================"
echo " FrozenLake / shared Qwen2.5-VL + V-JEPA-teacher pipeline"
echo "========================================================"
echo "phase       : ${PHASE}"
echo "model input : native Qwen2.5-VL [36,2048]"
echo "teacher     : frozen V-JEPA [36,1408], supervision only"
echo "train data  : VAGEN map seeds 1..10000"
echo "evaluation  : VAGEN fixed seeds 0..127"
echo "checkpoints : ${EXP_ROOT}"

if [[ "${PHASE}" == "prepare" || "${PHASE}" == "all" ]]; then run_prepare; fi
if [[ "${PHASE}" == "data_check" ]]; then validate_data; fi
if [[ "${PHASE}" == "wm" || "${PHASE}" == "all" ]]; then run_base_wm; fi
if [[ "${PHASE}" == "prior" || "${PHASE}" == "all" ]]; then run_prior; fi
if [[ "${PHASE}" == "spatial_prior" || "${PHASE}" == "all" ]]; then run_spatial_prior; fi
if [[ "${PHASE}" == "wm_audit" || "${PHASE}" == "all" ]]; then run_wm_audit; fi
if [[ "${PHASE}" == "reward" || "${PHASE}" == "all" ]]; then run_reward; fi
if [[ "${PHASE}" == "critic" || "${PHASE}" == "all" ]]; then run_critic; fi
if [[ "${PHASE}" == "ppo" || "${PHASE}" == "all" ]]; then run_ppo; fi
