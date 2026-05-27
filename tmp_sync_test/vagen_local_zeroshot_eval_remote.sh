#!/usr/bin/env bash
set -euo pipefail

HKUST_HOST="${HKUST_HOST:-hkust}"
GPU_IDS="${GPU_IDS:-4,6,7,3}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
EPISODES="${EPISODES:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
TP_SIZE="${TP_SIZE:-$(awk -F, '{print NF}' <<<"$GPU_IDS")}"
RUN_NAME="${RUN_NAME:-vagen-local-zeroshot-$(date -u +%Y%m%d-%H%M%S)}"
REMOTE_CODE_DIR="${REMOTE_CODE_DIR:-/home/jincai_guo/MBRL-JEPA}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/jincai_guo/mbrl_vlm_vagen_eval/local_runs}"
WANDB_PROJECT="${WANDB_PROJECT:-mbrl-vlm}"
WANDB_ENTITY="${WANDB_ENTITY:-qinglinh2003-usc}"

echo "Launching local-VAGEN zero-shot eval on ${HKUST_HOST}"
echo "  gpus:      ${GPU_IDS}"
echo "  model:     ${MODEL}"
echo "  episodes:  ${EPISODES}"
echo "  batch:     ${BATCH_SIZE}"
echo "  run_name:  ${RUN_NAME}"

ssh "${HKUST_HOST}" bash -s -- \
  "${GPU_IDS}" \
  "${MODEL}" \
  "${EPISODES}" \
  "${BATCH_SIZE}" \
  "${TP_SIZE}" \
  "${RUN_NAME}" \
  "${REMOTE_CODE_DIR}" \
  "${REMOTE_OUTPUT_ROOT}" \
  "${WANDB_PROJECT}" \
  "${WANDB_ENTITY}" <<'REMOTE_SCRIPT'
set -euo pipefail

GPU_IDS="$1"
MODEL="$2"
EPISODES="$3"
BATCH_SIZE="$4"
TP_SIZE="$5"
RUN_NAME="$6"
REMOTE_CODE_DIR="$7"
REMOTE_OUTPUT_ROOT="$8"
WANDB_PROJECT="$9"
WANDB_ENTITY="${10}"

OUTPUT_DIR="${REMOTE_OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_DIR}/run.log"
mkdir -p "${OUTPUT_DIR}"

eval "$(/apps/miniconda3/bin/conda shell.bash hook)"
conda activate vagen

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

cd /tmp

python "${REMOTE_CODE_DIR}/scripts/experiments/sokoban/vagen_zeroshot_eval.py" \
  --episodes "${EPISODES}" \
  --batch-size "${BATCH_SIZE}" \
  --model "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_PATH}"

echo "==== SUMMARY ${OUTPUT_DIR}/summary.json ===="
cat "${OUTPUT_DIR}/summary.json"
REMOTE_SCRIPT
