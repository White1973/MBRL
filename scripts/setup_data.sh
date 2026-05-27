#!/usr/bin/env bash
# Setup training data for MBRL-VLM experiments.
#
# Default location: data/sokoban_10k/tokenized (repo-root)
#
# Usage:
#   bash scripts/setup_data.sh              # check if data exists
#   bash scripts/setup_data.sh --download   # download from HF if missing
#   bash scripts/setup_data.sh --download --force  # re-download from scratch
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Default local data path
LOCAL_DATA="${ROOT_DIR}/data/sokoban_10k/tokenized"

# HF dataset (contains tar shards + manifest.jsonl)
HF_REPO="qinglinhou/sokoban-10k-vjepa2-tokenized-shards"

# Expected file count: 10000 .pt + 1 manifest.jsonl
EXPECTED_COUNT=10001

count_files() {
  local path="$1"
  find "${path}" -maxdepth 1 \( -name "*.pt" -o -name "manifest.jsonl" \) 2>/dev/null | wc -l
}

check_data() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    local count
    count=$(count_files "${path}")
    if [[ "${count}" -ge "${EXPECTED_COUNT}" ]]; then
      echo "[OK] Found complete data at ${path} (${count} files)"
      return 0
    elif [[ "${count}" -gt 0 ]]; then
      echo "[INCOMPLETE] Found ${count}/${EXPECTED_COUNT} files at ${path}"
      return 1
    fi
  fi
  echo "[MISSING] No data at ${path}"
  return 1
}

echo "Checking for tokenized Sokoban data..."

DOWNLOAD=0
FORCE=0
for arg in "$@"; do
  case "${arg}" in
    --download) DOWNLOAD=1 ;;
    --force) FORCE=1 ;;
  esac
done

if [[ "${DOWNLOAD}" -eq 0 ]]; then
  if check_data "${LOCAL_DATA}"; then
    echo ""
    echo "To use in experiments, set in _common.sh or env:"
    echo "  export DATA_DIR=${LOCAL_DATA}"
    exit 0
  fi

  echo ""
  echo "To download, run:"
  echo "  bash scripts/setup_data.sh --download"
  exit 1
fi

# --download path
if [[ "${FORCE}" -eq 1 ]]; then
  echo "[FORCE] Removing existing data at ${LOCAL_DATA}"
  rm -rf "${LOCAL_DATA}"
elif check_data "${LOCAL_DATA}" 2>/dev/null; then
  # Check if complete
  count=$(count_files "${LOCAL_DATA}")
  if [[ "${count}" -ge "${EXPECTED_COUNT}" ]]; then
    echo ""
    echo "[SKIP] Data already complete. To re-download:"
    echo "  bash scripts/setup_data.sh --download --force"
    exit 0
  fi
  echo "[INCOMPLETE] Will download missing shards and re-extract..."
fi

echo ""
echo "Downloading shards from HuggingFace: ${HF_REPO}"
echo "This downloads tar shards instead of 10K individual files."
echo ""

# Disable xet download path (triggers excessive API calls)
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

SHARD_DIR="${ROOT_DIR}/data/sokoban_10k/_shards"
mkdir -p "${SHARD_DIR}" "${LOCAL_DATA}"

python3 - <<PY
import os, sys, tarfile, glob
from huggingface_hub import snapshot_download

shard_dir = "${SHARD_DIR}"
out_dir = "${LOCAL_DATA}"
repo = "${HF_REPO}"

print(f"Downloading {repo} → {shard_dir}")
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=shard_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8,
)
print("Download complete. Extracting shards...")

shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.tar")))
if not shards:
    print("ERROR: No shard_*.tar files found after download!", file=sys.stderr)
    sys.exit(1)

for i, shard in enumerate(shards):
    print(f"  [{i+1}/{len(shards)}] Extracting {os.path.basename(shard)}...")
    with tarfile.open(shard, "r") as tar:
        tar.extractall(path=out_dir)

# Copy manifest if present
manifest = os.path.join(shard_dir, "manifest.jsonl")
if os.path.exists(manifest):
    import shutil
    shutil.copy2(manifest, os.path.join(out_dir, "manifest.jsonl"))

pt_count = len(glob.glob(os.path.join(out_dir, "*.pt")))
print(f"Extraction complete: {pt_count} .pt files in {out_dir}")
PY

echo ""
echo "Cleaning up shard cache..."
rm -rf "${SHARD_DIR}"

echo ""
check_data "${LOCAL_DATA}"
echo ""
echo "To use in experiments:"
echo "  export DATA_DIR=${LOCAL_DATA}"
