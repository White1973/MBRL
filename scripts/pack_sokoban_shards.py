#!/usr/bin/env python3
"""
Pack Sokoban tokenized episodes into tar shards to avoid 10k small files on HF.

Example:
  python scripts/pack_sokoban_shards.py \
    --input data/sokoban_10k/tokenized \
    --output data/sokoban_10k/shards \
    --shard-size 500 \
    --dtype bf16
"""
import argparse
import io
import os
import tarfile
from glob import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory with sokoban_*.pt files")
    p.add_argument("--output", required=True, help="Output directory for shard_*.tar")
    p.add_argument("--shard-size", type=int, default=500, help="Episodes per shard")
    p.add_argument("--pattern", default="sokoban_*.pt", help="Glob pattern for episode files")
    p.add_argument("--manifest", default="manifest.jsonl", help="Manifest filename to copy if present")
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Convert tensors before packing (default: bf16)",
    )
    return p.parse_args()


def _convert_dtype(obj, dtype):
    import torch

    if torch.is_tensor(obj):
        return obj.to(dtype=dtype)
    if isinstance(obj, dict):
        return {k: _convert_dtype(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_dtype(v, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_dtype(v, dtype) for v in obj)
    return obj


def main():
    args = parse_args()
    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(in_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files found: {os.path.join(in_dir, args.pattern)}")

    shard_size = max(1, args.shard_size)
    total = len(files)
    num_shards = (total + shard_size - 1) // shard_size

    dtype_map = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
    target_dtype = dtype_map[args.dtype]
    if args.dtype != "fp32":
        print(f"Converting tensors to {target_dtype} before packing")

    for i in range(num_shards):
        shard_files = files[i * shard_size : (i + 1) * shard_size]
        shard_name = f"shard_{i:03d}.tar"
        shard_path = os.path.join(out_dir, shard_name)
        print(f"[{i+1}/{num_shards}] {shard_name} ({len(shard_files)} files)")
        with tarfile.open(shard_path, "w") as tar:
            for f in shard_files:
                arcname = os.path.basename(f)
                if args.dtype == "fp32":
                    tar.add(f, arcname=arcname)
                    continue

                import torch

                obj = torch.load(f, map_location="cpu", weights_only=False)
                if target_dtype == "bfloat16":
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

                obj = _convert_dtype(obj, dtype)
                buf = io.BytesIO()
                torch.save(obj, buf)
                size = buf.getbuffer().nbytes
                buf.seek(0)
                info = tarfile.TarInfo(name=arcname)
                info.size = size
                tar.addfile(info, fileobj=buf)

    manifest_path = os.path.join(in_dir, args.manifest)
    if os.path.exists(manifest_path):
        import shutil

        shutil.copy2(manifest_path, os.path.join(out_dir, args.manifest))
        print(f"Copied manifest: {args.manifest}")

    print(f"Done. Wrote {num_shards} shards to {out_dir}")


if __name__ == "__main__":
    main()
