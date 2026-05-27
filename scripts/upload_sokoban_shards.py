#!/usr/bin/env python3
"""
Upload Sokoban shard folder to Hugging Face dataset repo.

Requires:
  export HF_TOKEN=...

Example:
  python scripts/upload_sokoban_shards.py \
    --repo qinglinhou/sokoban-10k-vjepa2-tokenized-shards \
    --path data/sokoban_10k/shards
"""
import argparse
import os
import sys

from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/name")
    p.add_argument("--path", required=True, help="Local folder containing shard_*.tar")
    p.add_argument("--private", action="store_true", help="Create repo as private")
    return p.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.path):
        print(f"ERROR: path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
        token=token,
    )

    print(f"Uploading folder: {args.path} → {args.repo}")
    api.upload_folder(
        repo_id=args.repo,
        folder_path=args.path,
        repo_type="dataset",
        token=token,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
