#!/usr/bin/env python3
"""Validate the shared Qwen-input / V-JEPA-teacher replay contract.

This is a data/provenance check only.  It deliberately knows nothing about a
task-specific world-model implementation: both Sokoban and FrozenLake must
store native Qwen2.5-VL observation tokens in ``obs_tokens`` and keep V-JEPA
features in the separate ``semantic_teacher_tokens`` field.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


EXPECTED_TOKENIZER = "qwen2.5_vl_native+vjepa2_teacher"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sample_indices(count: int, requested: int) -> list[int]:
    if count <= 0:
        return []
    requested = min(max(requested, 1), count)
    if requested == 1:
        return [0]
    return sorted(
        {round(index * (count - 1) / (requested - 1)) for index in range(requested)}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--env-id", choices=("sokoban", "frozenlake"), required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-seed-start", type=int, default=None)
    parser.add_argument("--belief-slots", type=int, default=36)
    parser.add_argument("--qwen-dim", type=int, default=2048)
    parser.add_argument("--teacher-dim", type=int, default=1408)
    parser.add_argument("--check-samples", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.jsonl"
    failures: list[str] = []
    entries: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        failures.append(f"missing manifest: {manifest_path}")
    else:
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                failures.append(f"manifest line {line_number} is invalid JSON: {exc}")
                break

    if len(entries) != args.expected_episodes:
        failures.append(
            f"episode_count={len(entries)} != expected={args.expected_episodes}"
        )

    seeds: list[int] = []
    for index, entry in enumerate(entries):
        prefix = f"manifest[{index}]"
        if entry.get("env_id") != args.env_id:
            failures.append(f"{prefix}.env_id={entry.get('env_id')!r}")
        if entry.get("tokenizer") != EXPECTED_TOKENIZER:
            failures.append(
                f"{prefix}.tokenizer={entry.get('tokenizer')!r}; expected "
                f"{EXPECTED_TOKENIZER!r}"
            )
        token_shape = entry.get("token_shape")
        teacher_shape = entry.get("semantic_teacher_shape")
        if not (
            isinstance(token_shape, list)
            and len(token_shape) == 3
            and token_shape[1:] == [args.belief_slots, args.qwen_dim]
        ):
            failures.append(f"{prefix}.token_shape={token_shape!r}")
        if not (
            isinstance(teacher_shape, list)
            and len(teacher_shape) == 3
            and teacher_shape[1:] == [args.belief_slots, args.teacher_dim]
        ):
            failures.append(f"{prefix}.semantic_teacher_shape={teacher_shape!r}")
        metadata = entry.get("metadata", {})
        if metadata.get("visual_input") != "qwen2.5_vl_native":
            failures.append(
                f"{prefix}.metadata.visual_input={metadata.get('visual_input')!r}"
            )
        teacher = metadata.get("semantic_teacher", {})
        if teacher.get("name") != "vjepa2_compressed_raw":
            failures.append(f"{prefix}.metadata.semantic_teacher.name is invalid")
        if teacher.get("input_is_teacher") is not False:
            failures.append(
                f"{prefix}.metadata.semantic_teacher.input_is_teacher must be false"
            )
        seed = metadata.get("seed")
        if isinstance(seed, int):
            seeds.append(seed)

        # Avoid producing thousands of copies of the same schema failure.
        if len(failures) >= 50:
            failures.append("additional manifest failures omitted")
            break

    if args.expected_seed_start is not None and len(entries) == args.expected_episodes:
        expected_seeds = list(
            range(
                args.expected_seed_start,
                args.expected_seed_start + args.expected_episodes,
            )
        )
        if seeds != expected_seeds:
            failures.append(
                "manifest seeds are not the required contiguous official sequence "
                f"{expected_seeds[0]}..{expected_seeds[-1]}"
            )

    checked_paths: list[str] = []
    if not failures:
        for index in _sample_indices(len(entries), args.check_samples):
            entry = entries[index]
            episode_path = args.data_dir / str(entry["path"])
            checked_paths.append(str(episode_path.resolve()))
            if not episode_path.is_file():
                failures.append(f"missing episode file: {episode_path}")
                continue
            payload = torch.load(
                episode_path, map_location="cpu", weights_only=False
            )
            observation = payload.get("obs_tokens")
            teacher = payload.get("semantic_teacher_tokens")
            if not isinstance(observation, torch.Tensor) or tuple(
                observation.shape[1:]
            ) != (args.belief_slots, args.qwen_dim):
                failures.append(f"invalid obs_tokens in {episode_path}")
                continue
            if not isinstance(teacher, torch.Tensor) or tuple(
                teacher.shape[1:]
            ) != (args.belief_slots, args.teacher_dim):
                failures.append(f"invalid semantic_teacher_tokens in {episode_path}")
                continue
            if observation.shape[0] != teacher.shape[0]:
                failures.append(f"Qwen/V-JEPA time axes differ in {episode_path}")
            if not bool(torch.isfinite(observation.float()).all()):
                failures.append(f"non-finite Qwen observation tokens in {episode_path}")
            if not bool(torch.isfinite(teacher.float()).all()):
                failures.append(f"non-finite V-JEPA teacher tokens in {episode_path}")
            metadata = payload.get("metadata", {})
            if metadata.get("visual_input") != "qwen2.5_vl_native":
                failures.append(f"payload visual_input is invalid in {episode_path}")
            if metadata.get("semantic_teacher", {}).get("input_is_teacher") is not False:
                failures.append(f"V-JEPA is marked as model input in {episode_path}")

    report = {
        "format": "qwen_vjepa_dataset_contract_v1",
        "passed": not failures,
        "failures": failures,
        "data_dir": str(args.data_dir.resolve()),
        "env_id": args.env_id,
        "episode_count": len(entries),
        "expected_episode_count": args.expected_episodes,
        "expected_seed_start": args.expected_seed_start,
        "observation_contract": {
            "source": "native_qwen2.5_vl",
            "slots": args.belief_slots,
            "dim": args.qwen_dim,
        },
        "teacher_contract": {
            "source": "frozen_vjepa2",
            "used_as_model_input": False,
            "slots": args.belief_slots,
            "dim": args.teacher_dim,
        },
        "sample_files_checked": checked_paths,
    }
    _atomic_json(report, args.output)
    print(
        "Qwen/V-JEPA dataset contract: "
        + ("PASS" if not failures else "FAIL: " + "; ".join(failures[:5])),
        flush=True,
    )
    print(f"Dataset contract report: {args.output}", flush=True)
    if args.enforce and failures:
        raise RuntimeError(f"dataset contract failed: {args.output}")


if __name__ == "__main__":
    main()
