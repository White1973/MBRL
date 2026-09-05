#!/usr/bin/env bash
set -euo pipefail

# Formal PPO from the released Qwen+V-JEPA Critic.
# The baseline is Actor update 0. Training, evaluation, W&B and checkpoint
# schedules advance only after an Actor PPO transaction is accepted.
ROOT="${MBRL0901_ROOT:-/personal/jiayu2026/code/MBRL0901}"
ENGINE_ROOT="${MBRL_ENGINE_ROOT:-/personal/jiayu2026/code/MBRL}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
RESUME="${RESUME:-0}"
WORLD_MODEL_MODE="${WORLD_MODEL_MODE:-frozen_wm}"
ENV_ID="${ENV_ID:-sokoban}"

REWARD_CKPT="${REWARD_CKPT:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage2_reward_head/best.pt}"
CRITIC_DIR="${CRITIC_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage3_critic_warmup}"
CRITIC_CKPT="${CRITIC_CKPT:-${CRITIC_DIR}/best.pt}"
CRITIC_REPORT="${CRITIC_REPORT:-${CRITIC_DIR}/warmup_report.json}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/sokoban_10k_qwen_vjepa_teacher_3B/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/qwen_vjepa_teacher_seed${SEED}/stage4_ppo_${WORLD_MODEL_MODE}_v2}"
ONLINE_REPLAY_ROOT="${ONLINE_REPLAY_ROOT:-${OUTPUT_DIR}/online_replay}"

ACTOR_UPDATES="${ACTOR_UPDATES:-1000}"
# Stage-3 starts at global update 20. This deliberately leaves room for
# rejected Actor transactions while ACTOR_UPDATES remains the true stop clock.
TOTAL_UPDATES="${TOTAL_UPDATES:-2020}"
EVAL_EVERY="${EVAL_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
REWARD_CONFIDENCE_FLOOR="${REWARD_CONFIDENCE_FLOOR:-0.4968768060207367}"

case "${ENV_ID}" in
  sokoban)
    EVAL_SOURCE="${VAGEN_LEVELS_FILE:-${ROOT}/data/vagen_mirror_testset/sokoban_256.json}"
    EVAL_EPISODES="${EVAL_EPISODES:-256}"
    eval_protocol="all 256 fixed VAGEN Sokoban levels"
    preflight_eval_args=(--eval-levels "${EVAL_SOURCE}")
    trainer_eval_args=(--eval-levels-file "${EVAL_SOURCE}")
    ;;
  frozenlake)
    EVAL_SOURCE="${EVAL_SEEDS_FILE:-${ROOT}/data/frozenlake/vagen_official_seeded/test_seeds.json}"
    EVAL_EPISODES="${EVAL_EPISODES:-128}"
    eval_protocol="fixed VAGEN FrozenLake seeds 0..127"
    preflight_eval_args=(--eval-seeds "${EVAL_SOURCE}")
    trainer_eval_args=(--eval-seeds-file "${EVAL_SOURCE}")
    ;;
  *)
    echo "ENV_ID must be sokoban or frozenlake; got ${ENV_ID}" >&2
    exit 2
    ;;
esac

[[ "${RESUME}" == "0" || "${RESUME}" == "1" ]] || {
  echo "RESUME must be 0 or 1; got ${RESUME}" >&2
  exit 2
}
[[ "${WORLD_MODEL_MODE}" == "frozen_wm" || "${WORLD_MODEL_MODE}" == "alternating_wm" ]] || {
  echo "WORLD_MODEL_MODE must be frozen_wm or alternating_wm; got ${WORLD_MODEL_MODE}" >&2
  exit 2
}
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }

export PYTHONPATH="${ROOT}:${ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

preflight_args=()
resume_args=()
tee_args=()
if [[ "${RESUME}" == "1" ]]; then
  preflight_args+=(--resume)
  resume_args+=(--resume "${OUTPUT_DIR}/latest.pt")
  tee_args+=(-a)
else
  # A fresh PPO branch starts from the released Stage-3 Actor/Critic state,
  # not from a newly assembled random policy on top of Stage-2 WM weights.
  resume_args+=(--resume "${CRITIC_CKPT}")
fi

wm_mode_args=(--world-model-mode "${WORLD_MODEL_MODE}")
if [[ "${WORLD_MODEL_MODE}" == "alternating_wm" ]]; then
  wm_mode_args+=(
    --wm-refresh-prior-lora-only
    --wm-refresh-every "${WM_REFRESH_EVERY:-50}"
    --wm-refresh-updates "${WM_REFRESH_UPDATES:-1}"
    --wm-refresh-batch-size "${WM_REFRESH_BATCH_SIZE:-16}"
    --wm-refresh-lr "${WM_REFRESH_LR:-1e-6}"
    --wm-refresh-base-lr-factor 0
    --wm-refresh-warmup-steps "${WM_REFRESH_WARMUP_STEPS:-0}"
    --wm-refresh-grad-clip "${WM_REFRESH_GRAD_CLIP:-0.5}"
    --wm-refresh-horizon 2
    --wm-refresh-validation-batches "${WM_REFRESH_VALIDATION_BATCHES:-8}"
    --wm-refresh-reward-loss-coef 0
    --wm-refresh-freeze-reward-head
    --wm-open-loop-horizon 2
    --wm-open-dynamics-coef "${WM_OPEN_DYNAMICS_COEF:-0.25}"
    --wm-prior-reward-coef "${WM_PRIOR_REWARD_COEF:-0.1}"
  )
fi

"${PYTHON_BIN}" -u "${ROOT}/scripts/preflight_qwen_vjepa_ppo.py" \
  --env-id "${ENV_ID}" \
  --critic-checkpoint "${CRITIC_CKPT}" \
  --critic-report "${CRITIC_REPORT}" \
  --reward-checkpoint "${REWARD_CKPT}" \
  --data-dir "${DATA_DIR}" \
  "${preflight_eval_args[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --reward-confidence-floor "${REWARD_CONFIDENCE_FLOOR}" \
  "${preflight_args[@]}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: checks passed; PPO training was not started."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

# This is the learned-WM exact-H2 PPO protocol already aligned with the
# released Critic target. It is not symbolic physics and does not read the
# official evaluation panel for training, selection or replay.
export CRITIC_PRETRAIN_ONLY=0
export REQUIRE_RELEASED_CRITIC_AT_START=1
export COUNTERFACTUAL_H2_PPO=1
export CRITIC_WARMUP_ZERO_BOOTSTRAP=1
export CRITIC_WARMUP_REQUIRE_RANKING=1
export CRITIC_WARMUP_COUNTERFACTUAL_ACTIONS=1
export CRITIC_TARGET_CONTINUATIONS=1
export CRITIC_REWARD_CONFIDENCE_FLOOR="${REWARD_CONFIDENCE_FLOOR}"
export CRITIC_TARGET_SEMANTICS=learned_wm_reward_h2_counterfactual
export IMAGINED_CRITIC_UPDATE=1

# Roll back the Actor, Critic and Adam state if a candidate Actor update
# regresses on the immutable level-disjoint H1/H2 ranking panel.
export TRANSACTIONAL_ACTOR_GATE=1
export COUNTERFACTUAL_ACTOR_TRANSACTION_GATE=1
export COUNTERFACTUAL_ACTOR_VALIDATION_EVERY=1
export COUNTERFACTUAL_ACTOR_TOP1_GATE="${COUNTERFACTUAL_ACTOR_TOP1_GATE:-0.60}"
export COUNTERFACTUAL_ACTOR_PAIRWISE_GATE="${COUNTERFACTUAL_ACTOR_PAIRWISE_GATE:-0.60}"
export COUNTERFACTUAL_ACTOR_SCORE_DROP_TOLERANCE="${COUNTERFACTUAL_ACTOR_SCORE_DROP_TOLERANCE:-0.05}"
export ACTOR_TRANSACTION_REJECT_PATIENCE="${ACTOR_TRANSACTION_REJECT_PATIENCE:-10}"
export REQUIRE_COUNTERFACTUAL_ACTOR_VALIDATION=1
export COUNTERFACTUAL_FIXED_REPLAY_FRACTION="${COUNTERFACTUAL_FIXED_REPLAY_FRACTION:-0.25}"

# Formal unified replay: 800 fixed offline episodes, then one real online
# episode after each accepted Actor update until the online pool reaches 800.
export FORMAL_UNIFIED_PPO=1
export UNIFIED_RANDOM_REPLAY=1
export UNIFIED_REPLAY_OFFLINE_EPISODES="${UNIFIED_REPLAY_OFFLINE_EPISODES:-800}"
export UNIFIED_REPLAY_ONLINE_TARGET="${UNIFIED_REPLAY_ONLINE_TARGET:-800}"
export UNIFIED_REPLAY_SEED="${UNIFIED_REPLAY_SEED:-20260901}"
export COLLECT_EVERY=1
export COLLECT_EPISODES=1

# The only learning clock exposed as the main metric axis is the number of
# accepted Actor PPO updates. Global pipeline update remains provenance only.
export ACTOR_UPDATE_LIMIT="${ACTOR_UPDATES}"
export MIN_ACTOR_PPO_UPDATES="${ACTOR_UPDATES}"
export REQUIRE_ACTOR_UPDATE_LIMIT_REACHED=1
export REQUIRE_ACTOR_SR_IMPROVEMENT_LEVELS="${REQUIRE_ACTOR_SR_IMPROVEMENT_LEVELS:-8}"
export SKIP_BASELINE_EVAL=0
export WANDB_ACTOR_STEP_AXIS=1
# Record the first accepted policy update explicitly, then keep the regular
# eval cadence at 10, 20, 30, ... accepted Actor updates.
export EVAL_AT_ACTOR_UPDATE_ONE=1
export PPO_PROTOCOL="qwen_vjepa_learned_wm_exact_h2_ppo_${WORLD_MODEL_MODE}"
export PPO_SOURCE_CHECKPOINT="${CRITIC_CKPT}"
export OVERRIDE_RESUME_ACTOR_LR="${ACTOR_LR:-2e-6}"

# Explicitly exclude legacy symbolic runtime shortcuts.
export SYMBOLIC_PHYSICS_TERMINAL=0
export CONTINUE_SYMBOLIC_H2_RUN=0

extra_args=()
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  extra_args+=(--no-wandb)
  export REQUIRE_WANDB=0
else
  export REQUIRE_WANDB=1
  export WANDB_MODE="${WANDB_MODE:-online}"
fi

echo "========================================================"
echo " Qwen + V-JEPA formal ${WORLD_MODEL_MODE} PPO"
echo "========================================================"
echo "environment      : ${ENV_ID}"
echo "Critic source    : ${CRITIC_CKPT}"
echo "WM/Reward source : ${REWARD_CKPT}"
echo "output           : ${OUTPUT_DIR}"
echo "Actor updates    : ${ACTOR_UPDATES} accepted updates"
echo "metric x-axis    : actor_ppo_update (baseline=0, first accepted update=1)"
echo "evaluation       : ${eval_protocol} at Actor 1, then every ${EVAL_EVERY} updates"
echo "checkpoint       : latest.pt every ${CHECKPOINT_EVERY}; best.pt by eval success"
if [[ "${WORLD_MODEL_MODE}" == "alternating_wm" ]]; then
  echo "WM refresh       : prior LoRA every ${WM_REFRESH_EVERY:-50} Actor updates; ${WM_REFRESH_UPDATES:-1} step(s), lr=${WM_REFRESH_LR:-1e-6}"
fi
echo "W&B              : ${WANDB_PROJECT:-jiayu-mbrl}/${WANDB_GROUP:-qwen_vjepa_ppo}"
echo "resume           : ${resume_args[*]:-fresh run}"

"${PYTHON_BIN}" -u "${ROOT}/scripts/train_mbrl.py" \
  --mode full --env-id "${ENV_ID}" \
  --data-dir "${DATA_DIR}" \
  --wm-checkpoint "${REWARD_CKPT}" \
  "${wm_mode_args[@]}" \
  --backbone-model "${MODEL_PATH}" \
  --hidden-dim 2048 --belief-slots 36 --independent-backbone \
  --encoder-type qwen \
  --action-conditioning-mode embedded \
  --posterior-grounding-mode visual_anchor \
  --posterior-recurrent-residual-scale 0.25 \
  --posterior-action-free \
  --prior-isolation-mode lora \
  --posterior-observation-residual-scale 0 \
  --vjepa-teacher-prior-coef 0.5 \
  --vjepa-teacher-posterior-coef 0 \
  --vjepa-teacher-delta-coef 1.0 \
  --reward-head-hidden-dim 0 \
  --require-injected-reward-head \
  --wm-action-id-offset 0 \
  --total-updates "${TOTAL_UPDATES}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}" \
  --rollouts-per-update "${ROLLOUTS_PER_UPDATE:-16}" \
  --rollout-horizon 2 \
  --no-value-bootstrap \
  --imagination-termination-mode predicted_success \
  --ppo-epochs "${PPO_EPOCHS:-2}" \
  --minibatch-size "${MINIBATCH_SIZE:-64}" \
  --actor-source qwen_slotwise \
  --slotwise-actor-features qwen \
  --slotwise-behavior-scale 0 \
  --actor-slot-dim 64 --actor-hidden-dim 256 --actor-hidden-layers 1 \
  --critic-source qwen_slotwise_q --critic-slot-dim 32 \
  --actor-lr "${ACTOR_LR:-2e-6}" \
  --critic-lr "${CRITIC_LR:-3e-5}" \
  --critic-warmup-min-updates 20 \
  --critic-warmup-ev-threshold 0.10 \
  --critic-warmup-ev-patience 3 \
  --critic-warmup-validation-size 512 \
  --critic-warmup-replay-capacity 4096 \
  --critic-warmup-train-samples 512 \
  --critic-warmup-ev-ema-alpha 0.20 \
  --critic-warmup-mse-improvement 0.05 \
  --offline-bc-steps 0 --offline-bc-strategies "" \
  --behavior-kl-coef 0 --behavior-bc-coef 0 \
  --clip-epsilon "${CLIP_EPSILON:-0.1}" \
  --target-kl "${TARGET_KL:-0.01}" \
  --entropy-coef "${ENTROPY_COEF:-0.005}" \
  --target-entropy "${TARGET_ENTROPY:-0.8}" \
  --entropy-floor-coef "${ENTROPY_FLOOR_COEF:-0.1}" \
  --reward-mapping per_transition_success_conservative \
  --reward-confidence-floor "${REWARD_CONFIDENCE_FLOOR}" \
  --reward-scale 0.1 --positive-value 10.9 \
  --reward-low-confidence-scale 0.1 \
  --collect-every 1 --collect-episodes 1 --collect-max-steps 25 \
  --online-ratio 0 \
  --online-replay-root "${ONLINE_REPLAY_ROOT}" \
  --online-replay-max "${UNIFIED_REPLAY_ONLINE_TARGET}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-episodes "${EVAL_EPISODES}" --eval-max-steps 25 \
  "${trainer_eval_args[@]}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --checkpoint-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" --device cuda:0 \
  --wandb-project "${WANDB_PROJECT:-jiayu-mbrl}" \
  --wandb-group "${WANDB_GROUP:-qwen_vjepa_ppo}" \
  --wandb-run-name "${WANDB_RUN_NAME:-qwen_vjepa_ppo_seed${SEED}}" \
  --recompute-old-log-probs \
  "${resume_args[@]}" \
  "${extra_args[@]}" \
  2>&1 | tee "${tee_args[@]}" "${OUTPUT_DIR}/train.log"
