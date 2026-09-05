#!/usr/bin/env python3
"""Build split-safe near-terminal posterior panels for Critic H2 warmup."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sembelief_wm.data.storage import read_manifest  # noqa: E402
from sembelief_wm.model import QwenTransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.model.checkpoint_semantics import (  # noqa: E402
    validate_world_model_semantics,
)
from sembelief_wm.rl.critic_h2_cache import FORMAT  # noqa: E402
from sembelief_wm.types import BeliefState  # noqa: E402


BUCKET_NAMES = ["initial_h1", "initial_h2", "suffix_h1", "suffix_h2"]


def _digest(indices: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in indices).encode("ascii")
    ).hexdigest()


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _candidate_examples(
    entries: list[dict[str, Any]], indices: list[int]
) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {
        name: [] for name in BUCKET_NAMES
    }
    for manifest_index in indices:
        entry = entries[manifest_index]
        metadata = dict(entry.get("metadata", {}))
        if not bool(metadata.get("success")):
            continue
        length = int(entry["length"])
        if length == 1:
            result["initial_h1"].append((manifest_index, 0))
        if length == 2:
            result["initial_h2"].append((manifest_index, 0))
        if length > 1:
            result["suffix_h1"].append((manifest_index, length - 1))
        if length > 2:
            result["suffix_h2"].append((manifest_index, length - 2))
    return result


@torch.no_grad()
def _ground_examples(
    *,
    world_model: WorldModel,
    entries: list[dict[str, Any]],
    data_dir: Path,
    examples: list[tuple[int, int]],
    device: torch.device,
    batch_size: int,
    label: str,
) -> torch.Tensor:
    dtype = next(world_model.parameters()).dtype
    null_action_id = int(world_model.config.env.null_action_id)
    parts: list[torch.Tensor] = []
    started = time.monotonic()
    for begin in range(0, len(examples), batch_size):
        chunk = examples[begin : begin + batch_size]
        episodes = []
        for manifest_index, _ in chunk:
            path = data_dir / str(entries[manifest_index]["path"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            episodes.append(payload)
        belief = world_model.get_initial_belief(
            len(chunk), device=device, dtype=dtype
        )
        max_target = max(target for _, target in chunk)
        for timestep in range(max_target + 1):
            active = [
                local_id
                for local_id, (_, target) in enumerate(chunk)
                if target >= timestep
            ]
            observations = torch.stack([
                episodes[local_id]["obs_tokens"][timestep]
                for local_id in active
            ]).to(device=device, dtype=dtype)
            previous_actions = torch.tensor([
                null_action_id
                if timestep == 0
                else int(episodes[local_id]["actions"][timestep - 1].item())
                for local_id in active
            ], device=device, dtype=torch.long)
            active_belief = BeliefState(slots=belief.slots[active])
            updated = world_model.posterior_step(
                prev_belief=active_belief,
                prev_actions=previous_actions,
                observation_tokens=observations,
                env_ids=None,
            )
            if belief.slots.dtype != updated.slots.dtype:
                belief = BeliefState(slots=belief.slots.to(updated.slots.dtype))
            belief.slots[active] = updated.slots
        parts.append(belief.slots.detach().to(device="cpu", dtype=torch.bfloat16))
        completed = min(begin + len(chunk), len(examples))
        if completed == len(examples) or completed % max(100, batch_size) < batch_size:
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = completed / elapsed
            print(
                f"[{label}] {completed}/{len(examples)} "
                f"({100*completed/len(examples):.1f}%) rate={rate:.2f} states/s",
                flush=True,
            )
        del episodes
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--env-id", choices=("sokoban", "frozenlake"), default="sokoban"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--train-per-bucket", type=int, default=1024)
    parser.add_argument("--validation-per-bucket", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate an existing cache against the requested source and exit.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.wm_checkpoint).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        if not args.validate_existing:
            raise FileExistsError(
                f"refusing to overwrite Critic H2 cache: {output_path}"
            )
        cached = torch.load(output_path, map_location="cpu", weights_only=False)
        if not isinstance(cached, dict):
            raise RuntimeError("existing Critic H2 cache is not a dictionary")
        provenance = cached.get("provenance", {})
        source_stat = checkpoint_path.stat()
        checks = {
            "format": cached.get("format") == FORMAT,
            # v1 Sokoban caches created before multi-environment support did
            # not record env_id. Their exact checkpoint/data provenance still
            # makes them safe to reuse; all newly written caches record it.
            "env_id": provenance.get("env_id", args.env_id) == args.env_id,
            "source_checkpoint": Path(
                provenance.get("source_checkpoint", "/missing")
            ).resolve() == checkpoint_path,
            "source_checkpoint_size": int(
                provenance.get("source_checkpoint_size", -1)
            ) == source_stat.st_size,
            "source_checkpoint_mtime_ns": int(
                provenance.get("source_checkpoint_mtime_ns", -1)
            ) == source_stat.st_mtime_ns,
            "data_dir": Path(provenance.get("data_dir", "/missing")).resolve()
            == data_dir,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "existing Critic H2 cache has stale or cross-task provenance: "
                + ", ".join(failed)
            )
        print(f"Existing Critic H2 cache provenance: PASS ({output_path})")
        return
    if args.train_per_bucket <= 0 or args.validation_per_bucket <= 0:
        raise ValueError("bucket sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to ground Qwen posterior panels")

    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("Reward-Head checkpoint is not self-contained")
    injection = payload.get("reward_head_injection")
    if (
        not payload.get("injected_reward_head")
        or not isinstance(injection, dict)
        or not bool(injection.get("gate", {}).get("passed"))
    ):
        raise RuntimeError("Reward Head has not passed its release Gate")
    if not {1, 2}.issubset({int(value) for value in injection.get("horizons", ())}):
        raise RuntimeError("Reward Head does not cover H1/H2")
    config = payload["config"]
    checkpoint_env_ids = tuple(getattr(config.env, "env_ids", ()))
    if args.env_id not in checkpoint_env_ids:
        raise ValueError(
            f"checkpoint env_ids={checkpoint_env_ids}, not {args.env_id!r}"
        )
    if config.encoder.encoder_type != "qwen":
        raise ValueError("Critic H2 cache requires Qwen-native observations")
    refresh = payload.get("wm_only_refresh", {})
    train_indices = [int(value) for value in refresh.get("train_indices", ())]
    official_indices = [int(value) for value in refresh.get("val_indices", ())]
    if not train_indices or not official_indices:
        raise ValueError("checkpoint has no fixed Stage-1 split")
    if set(train_indices) & set(official_indices):
        raise RuntimeError("Stage-1 train and official validation overlap")

    entries = read_manifest(data_dir / "manifest.jsonl")
    replay_env_ids = {str(entry.get("env_id")) for entry in entries}
    if replay_env_ids != {args.env_id}:
        raise ValueError(
            f"replay env_ids={sorted(replay_env_ids)}, expected [{args.env_id!r}]"
        )
    if max(train_indices + official_indices) >= len(entries):
        raise IndexError("checkpoint split exceeds paired replay manifest")
    candidates = {
        "train": _candidate_examples(entries, train_indices),
        "validation": _candidate_examples(entries, official_indices),
    }
    selected: dict[str, dict[str, list[tuple[int, int]]]] = {
        "train": {}, "validation": {}
    }
    for split, requested in (
        ("train", args.train_per_bucket),
        ("validation", args.validation_per_bucket),
    ):
        for bucket_id, name in enumerate(BUCKET_NAMES):
            available = candidates[split][name]
            if len(available) < requested:
                raise RuntimeError(
                    f"{split}/{name} has {len(available)} states, needs {requested}"
                )
            order = list(available)
            random.Random(args.seed + 1009 * bucket_id + (0 if split == "train" else 97)).shuffle(order)
            selected[split][name] = order[:requested]
    print(
        "Critic H2 split verified: Stage-1 train supplies training buckets; "
        "official 1000 supplies validation buckets only; overlap=0",
        flush=True,
    )

    print(f"Loading frozen Reward-Head WM: {checkpoint_path}", flush=True)
    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    world_model = WorldModel(config, backbone).to(device)
    world_model.load_state_dict(payload["model"], strict=True)
    validate_world_model_semantics(
        payload,
        attention_mode=config.backbone.attention_mode,
        context=f"Critic H2 cache source {checkpoint_path}",
    )
    world_model.eval().requires_grad_(False)
    # The checkpoint carries another full copy of the Qwen/WM tensors.  Once
    # strict loading and semantic validation have completed, release that CPU
    # copy before accumulating the ~600 MB posterior cache.
    del payload
    gc.collect()

    grounded: dict[str, dict[str, torch.Tensor]] = {
        "train": {}, "validation": {}
    }
    for split in ("train", "validation"):
        for name in BUCKET_NAMES:
            grounded[split][name] = _ground_examples(
                world_model=world_model,
                entries=entries,
                data_dir=data_dir,
                examples=selected[split][name],
                device=device,
                batch_size=args.batch_size,
                label=f"critic-h2/{split}/{name}",
            )

    stat = checkpoint_path.stat()
    result = {
        "format": FORMAT,
        "bucket_names": list(BUCKET_NAMES),
        "train_buckets": grounded["train"],
        "validation_buckets": grounded["validation"],
        "selected_examples": selected,
        "provenance": {
            "env_id": args.env_id,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_size": stat.st_size,
            "source_checkpoint_mtime_ns": stat.st_mtime_ns,
            "data_dir": str(data_dir),
            "seed": args.seed,
            "stage1_train_indices_sha256": _digest(train_indices),
            "official_validation_indices_sha256": _digest(official_indices),
            "stage1_train_episodes": len(train_indices),
            "official_validation_episodes": len(official_indices),
            "official_validation_used_for_training": False,
            "reward_head_gate_passed": True,
            "reward_head_decision_threshold": injection.get("decision_threshold"),
        },
    }
    _atomic_save(result, output_path)
    print(f"Critic H2 posterior cache saved: {output_path}", flush=True)
    del world_model
    gc.collect()


if __name__ == "__main__":
    main()
