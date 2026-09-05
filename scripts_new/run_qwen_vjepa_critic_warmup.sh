#!/usr/bin/env bash
set -euo pipefail

ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
ENGINE_ROOT="${MBRL_ENGINE_ROOT:-/personal/jiayu2026/code/MBRL}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
ENV_ID="${ENV_ID:-sokoban}"
PHASE="${PHASE:-all}"  # cache | train | finalize | all
RESUME="${RESUME:-0}"  # 1 resumes OUTPUT_DIR/latest.pt

REWARD_DIR="${REWARD_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed1/stage2_reward_head}"
SOURCE_WM="${SOURCE_WM:-${REWARD_DIR}/best.pt}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/sokoban_10k_qwen_vjepa_teacher_3B/tokenized}"
H2_CACHE="${H2_CACHE:-${ROOT}/data/critic_h2_cache/qwen_vjepa_teacher_seed1/cache.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed1/stage3_critic_warmup}"

TRAIN_PER_BUCKET="${TRAIN_PER_BUCKET:-1024}"
VALIDATION_PER_BUCKET="${VALIDATION_PER_BUCKET:-32}"
POSTERIOR_BATCH_SIZE="${POSTERIOR_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
ROLLOUTS_PER_UPDATE="${ROLLOUTS_PER_UPDATE:-16}"
MAX_UPDATES="${MAX_UPDATES:-100}"
MIN_UPDATES="${MIN_UPDATES:-20}"
CRITIC_LR="${CRITIC_LR:-3e-5}"
REWARD_CONFIDENCE_FLOOR="${REWARD_CONFIDENCE_FLOOR:-0.4968768060207367}"

MIN_EV_EMA="${MIN_EV_EMA:-0.10}"
MIN_MSE_IMPROVEMENT="${MIN_MSE_IMPROVEMENT:-0.05}"
MIN_TOP1="${MIN_TOP1:-0.60}"
MIN_PAIRWISE="${MIN_PAIRWISE:-0.60}"
MIN_Q_MARGIN="${MIN_Q_MARGIN:-0.001}"

case "${PHASE}" in
  cache|train|finalize|all) ;;
  *) echo "PHASE must be cache, train, finalize, or all; got ${PHASE}" >&2; exit 2 ;;
esac
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${SOURCE_WM}" ]] || { echo "Missing Reward-Head best.pt: ${SOURCE_WM}" >&2; exit 2; }
[[ -f "${DATA_DIR}/manifest.jsonl" ]] || { echo "Missing paired replay: ${DATA_DIR}" >&2; exit 2; }

export PYTHONPATH="${ROOT}:${ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ "${PHASE}" == "cache" || "${PHASE}" == "all" ]]; then
  cache_validation_args=()
  if [[ -f "${H2_CACHE}" ]]; then
    echo "Validating existing split-safe Critic H2 cache: ${H2_CACHE}"
    cache_validation_args+=(--validate-existing)
  else
    mkdir -p "$(dirname "${H2_CACHE}")"
    echo "=== Build fixed 9000/1000 Qwen posterior H2 panels ==="
  fi
  "${PYTHON_BIN}" -u "${ROOT}/scripts/build_qwen_vjepa_critic_h2_cache.py" \
    --env-id "${ENV_ID}" \
    --wm-checkpoint "${SOURCE_WM}" \
    --data-dir "${DATA_DIR}" \
    --output "${H2_CACHE}" \
    --device cuda:0 \
    --seed "${SEED}" \
    --train-per-bucket "${TRAIN_PER_BUCKET}" \
    --validation-per-bucket "${VALIDATION_PER_BUCKET}" \
    --batch-size "${POSTERIOR_BATCH_SIZE}" \
    "${cache_validation_args[@]}"
fi

if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
  [[ -f "${H2_CACHE}" ]] || { echo "Missing H2 cache: ${H2_CACHE}" >&2; exit 2; }
  resume_args=()
  tee_args=()
  if [[ -e "${OUTPUT_DIR}/best.pt" ]]; then
    echo "Refusing to modify a released Critic best.pt: ${OUTPUT_DIR}/best.pt" >&2
    exit 2
  fi
  if [[ -e "${OUTPUT_DIR}/latest.pt" ]]; then
    if [[ "${RESUME}" != "1" ]]; then
      echo "Existing Critic latest.pt found: ${OUTPUT_DIR}/latest.pt" >&2
      echo "Set RESUME=1 to continue it, or choose a new OUTPUT_DIR." >&2
      exit 2
    fi
    resume_args+=(--resume "${OUTPUT_DIR}/latest.pt")
    tee_args+=(-a)
  elif [[ "${RESUME}" == "1" ]]; then
    echo "RESUME=1 requested but latest.pt is missing: ${OUTPUT_DIR}/latest.pt" >&2
    exit 2
  fi
  mkdir -p "${OUTPUT_DIR}"

  # Critic-only transaction: exact 16 two-action sequences from every start;
  # Actor, WM and Reward Head remain frozen. Targets come from the deployed
  # learned prior + calibrated Reward Head, never explicit environment physics.
  export CRITIC_PRETRAIN_ONLY=1
  export COUNTERFACTUAL_H2_PPO=1
  export CRITIC_WARMUP_ZERO_BOOTSTRAP=1
  export CRITIC_WARMUP_REQUIRE_RANKING=1
  export CRITIC_WARMUP_COUNTERFACTUAL_ACTIONS=1
  export CRITIC_TARGET_CONTINUATIONS=1
  export CRITIC_WARMUP_MAX_UPDATES="${MAX_UPDATES}"
  export CRITIC_WARMUP_TOP1_GATE="${MIN_TOP1}"
  export CRITIC_WARMUP_PAIRWISE_GATE="${MIN_PAIRWISE}"
  export CRITIC_WARMUP_Q_MARGIN_GATE="${MIN_Q_MARGIN}"
  export CRITIC_RELEASE_MIN_BUCKETS_PASSED=4
  export CRITIC_RELEASE_REQUIRE_BUCKET_EV=1
  export CRITIC_RELEASE_BUCKET_EV_GATE="${MIN_EV_EMA}"
  export CRITIC_RELEASE_BUCKET_TOP1_GATE="${MIN_TOP1}"
  export CRITIC_RELEASE_BUCKET_PAIRWISE_GATE="${MIN_PAIRWISE}"
  export CRITIC_RELEASE_BUCKET_Q_MARGIN_GATE="${MIN_Q_MARGIN}"
  export CRITIC_SAVE_CANDIDATE=0
  export CRITIC_REWARD_CONFIDENCE_FLOOR="${REWARD_CONFIDENCE_FLOOR}"
  export CRITIC_TARGET_SEMANTICS=learned_wm_reward_h2_counterfactual
  export CRITIC_STABILIZATION_LR_FACTOR="${CRITIC_STABILIZATION_LR_FACTOR:-0.25}"
  export IMAGINED_CRITIC_UPDATE=1
  export SKIP_BASELINE_EVAL=1
  export ACTOR_UPDATE_LIMIT=0
  export COLLECT_EVERY=0

  extra_args=()
  if [[ "${NO_WANDB:-0}" == "1" ]]; then
    extra_args+=(--no-wandb)
  fi

  echo "========================================================"
  echo " Qwen + V-JEPA WM: exact-H2 Critic-only warmup"
  echo "========================================================"
  echo "Reward WM       : ${SOURCE_WM}"
  echo "environment     : ${ENV_ID}"
  echo "H2 panel cache  : ${H2_CACHE}"
  echo "output          : ${OUTPUT_DIR}"
  echo "resume          : ${resume_args[*]:-fresh run}"
  echo "Actor           : qwen_slotwise/qwen, frozen"
  echo "Critic          : qwen_slotwise_q, trainable Qwen LoRA"
  echo "reward threshold: ${REWARD_CONFIDENCE_FLOOR} (calibration-only selection)"
  echo "Gate            : EV>=${MIN_EV_EMA}, top1>=${MIN_TOP1}, pairwise>=${MIN_PAIRWISE}, all 4 buckets"
  echo "W&B             : ${WANDB_PROJECT:-jiayu-mbrl}/${WANDB_GROUP:-qwen_vjepa_critic_warmup}"

  "${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
    --mode full --env-id "${ENV_ID}" \
    --data-dir "${DATA_DIR}" \
    --wm-checkpoint "${SOURCE_WM}" \
    --critic-h2-cache "${H2_CACHE}" \
    --world-model-mode frozen_wm \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 --belief-slots 36 --independent-backbone \
    --encoder-type qwen \
    --action-conditioning-mode embedded \
    --posterior-grounding-mode visual_anchor \
    --posterior-recurrent-residual-scale 0.25 \
    --posterior-action-free \
    --prior-isolation-mode lora \
    --wm-refresh-prior-lora-only \
    --posterior-observation-residual-scale 0 \
    --vjepa-teacher-prior-coef 0.5 \
    --vjepa-teacher-posterior-coef 0 \
    --vjepa-teacher-delta-coef 1.0 \
    --reward-head-hidden-dim 0 \
    --require-injected-reward-head \
    --wm-action-id-offset 0 \
    --total-updates "${MAX_UPDATES}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --rollouts-per-update "${ROLLOUTS_PER_UPDATE}" \
    --rollout-horizon 2 \
    --imagination-termination-mode predicted_success \
    --ppo-epochs 2 --minibatch-size 64 \
    --actor-source qwen_slotwise \
    --slotwise-actor-features qwen \
    --slotwise-behavior-scale 0 \
    --actor-slot-dim 64 --actor-hidden-dim 256 --actor-hidden-layers 1 \
    --critic-source qwen_slotwise_q --critic-slot-dim 32 \
    --actor-lr 1e-5 --critic-lr "${CRITIC_LR}" \
    --critic-warmup-min-updates "${MIN_UPDATES}" \
    --critic-warmup-ev-threshold "${MIN_EV_EMA}" \
    --critic-warmup-ev-patience "${EV_PATIENCE:-3}" \
    --critic-warmup-validation-size "$((VALIDATION_PER_BUCKET * 4 * 4))" \
    --critic-warmup-replay-capacity "${REPLAY_CAPACITY:-4096}" \
    --critic-warmup-train-samples "${TRAIN_SAMPLES:-512}" \
    --critic-warmup-ev-ema-alpha "${EV_EMA_ALPHA:-0.20}" \
    --critic-warmup-mse-improvement "${MIN_MSE_IMPROVEMENT}" \
    --offline-bc-steps 0 --offline-bc-strategies "" \
    --behavior-kl-coef 0 --behavior-bc-coef 0 \
    --clip-epsilon 0.1 --target-kl 0.01 \
    --entropy-coef 0 --reward-mapping per_transition_success_conservative \
    --reward-confidence-floor "${REWARD_CONFIDENCE_FLOOR}" \
    --reward-scale 0.1 --positive-value 10.9 \
    --reward-low-confidence-scale 0.1 \
    --collect-every 0 --online-ratio 0 \
    --eval-every 0 --eval-episodes 0 \
    --checkpoint-every "${CHECKPOINT_EVERY:-5}" \
    --checkpoint-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" --device cuda:0 \
    --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
    --wandb-group "${WANDB_GROUP:-qwen_vjepa_critic_warmup}" \
    --wandb-run-name "${WANDB_RUN_NAME:-qwen_vjepa_critic_warmup_seed1}" \
    --recompute-old-log-probs \
    "${resume_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "${tee_args[@]}" "${OUTPUT_DIR}/train.log"
fi

if [[ "${PHASE}" == "finalize" || "${PHASE}" == "all" || "${PHASE}" == "train" ]]; then
  echo "=== Finalize Critic warmup release Gate ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/finalize_qwen_vjepa_critic_warmup.py" \
    --output-dir "${OUTPUT_DIR}" \
    --source-wm "${SOURCE_WM}" \
    --min-updates "${MIN_UPDATES}" \
    --min-ev-ema "${MIN_EV_EMA}" \
    --min-mse-improvement "${MIN_MSE_IMPROVEMENT}" \
    --min-top1 "${MIN_TOP1}" \
    --min-pairwise "${MIN_PAIRWISE}" \
    --min-q-margin "${MIN_Q_MARGIN}" \
    --reward-confidence-floor "${REWARD_CONFIDENCE_FLOOR}"
fi
