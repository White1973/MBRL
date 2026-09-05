#!/usr/bin/env python3
"""Build paired Qwen-input / frozen-V-JEPA-teacher world-model episodes.

The resulting episode contains two intentionally different visual tensors:

* ``obs_tokens``: native Qwen2.5-VL image embeddings used by the posterior;
* ``semantic_teacher_tokens``: frozen compressed V-JEPA features used only
  as a training target for future prior beliefs.

No V-JEPA feature is injected into the Qwen observation path.  Existing Qwen
tokens may be reused only after this script verifies they came from the exact
same raw frames, actions, rewards, and episode length.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sembelief_wm.config import BackboneConfig, Config, EncoderConfig, TrainingConfig
from sembelief_wm.data.schema import TokenizedEpisode
from sembelief_wm.data.storage import (
    existing_episode_ids,
    load_tokenized_episode,
    read_manifest,
    save_tokenized_episode,
)
from sembelief_wm.data.tokenizers import ImageTokenizer, QwenVisionTokenizer


def _tokenized_root(path: Path) -> Path:
    root = path / "tokenized" if (path / "tokenized").is_dir() else path
    if not (root / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"Tokenized manifest not found under {path}.")
    return root


def _raw_root(path: Path) -> Path:
    root = path / "raw" if (path / "raw").is_dir() else path
    if not (root / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"Raw manifest not found under {path}.")
    return root


def _assert_reusable_qwen_episode(
    *,
    raw: dict,
    qwen: TokenizedEpisode,
    episode_id: str,
) -> None:
    num_steps = len(raw["model_actions"])
    if qwen.episode_length != num_steps:
        raise RuntimeError(
            f"{episode_id}: Qwen episode length {qwen.episode_length} does not "
            f"match raw length {num_steps}."
        )
    if qwen.obs_tokens.shape[0] != len(raw["observations"]):
        raise RuntimeError(
            f"{episode_id}: Qwen frame count {qwen.obs_tokens.shape[0]} does not "
            f"match raw observation count {len(raw['observations'])}."
        )
    expected_actions = torch.as_tensor(raw["model_actions"], dtype=torch.long)
    if not torch.equal(qwen.actions[:num_steps].cpu(), expected_actions):
        raise RuntimeError(
            f"{episode_id}: Qwen actions do not match raw model_actions; refusing "
            "to pair teacher targets from another trajectory."
        )
    expected_rewards = torch.as_tensor(raw["rewards"], dtype=torch.float32)
    if not torch.allclose(qwen.rewards[:num_steps].float().cpu(), expected_rewards, atol=1e-5, rtol=1e-5):
        raise RuntimeError(
            f"{episode_id}: Qwen rewards do not match the raw trajectory."
        )
    raw_seed = raw.get("metadata", {}).get("seed")
    qwen_seed = qwen.metadata.get("seed")
    if raw_seed is not None and qwen_seed is not None and raw_seed != qwen_seed:
        raise RuntimeError(
            f"{episode_id}: Qwen seed {qwen_seed} does not match raw seed {raw_seed}."
        )


def _build_config(args: argparse.Namespace) -> Config:
    return Config(
        hidden_dim=args.hidden_dim,
        encoder=EncoderConfig(
            encoder_type="qwen",
            compressed_tokens=args.belief_slots,
            semantic_teacher_type="vjepa2",
            semantic_teacher_tokens=args.belief_slots,
            semantic_teacher_dim=1408,
        ),
        backbone=BackboneConfig(model_name=args.backbone_model),
        training=TrainingConfig(dtype=args.dtype),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair native Qwen visual inputs with frozen V-JEPA teacher targets."
    )
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--qwen-token-dir",
        default=None,
        help="Existing Qwen-token dataset to verify and reuse. If omitted, Qwen image features are encoded from raw RGB.",
    )
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--hidden-dim", required=True, type=int)
    parser.add_argument("--belief-slots", default=36, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--frame-batch-size", default=16, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.frame_batch_size <= 0:
        parser.error("--frame-batch-size must be positive")
    raw_root = _raw_root(Path(args.raw_data_dir))
    output_root = Path(args.output_dir)
    output_tokenized = output_root / "tokenized"
    if output_tokenized.exists() and args.overwrite:
        parser.error(
            "--overwrite is intentionally unsupported for paired data. Choose "
            "a new output directory or remove a specifically verified artifact."
        )
    completed = (
        existing_episode_ids(read_manifest(output_tokenized / "manifest.jsonl"))
        if (output_tokenized / "manifest.jsonl").is_file()
        else set()
    )

    config = _build_config(args)
    source_root: Path | None = None
    source_entries: dict[str, dict] = {}
    qwen_tokenizer: QwenVisionTokenizer | None = None
    if args.qwen_token_dir:
        source_root = _tokenized_root(Path(args.qwen_token_dir))
        source_entries = {
            str(entry["episode_id"]): entry
            for entry in read_manifest(source_root / "manifest.jsonl")
        }
        if not source_entries:
            raise RuntimeError("The supplied Qwen-token manifest is empty.")
        print(
            f"Reusing verified Qwen inputs from {source_root} "
            f"({len(source_entries)} episodes).",
            flush=True,
        )
    else:
        qwen_tokenizer = QwenVisionTokenizer(config, device=args.device)
        print("Encoding native Qwen2.5-VL visual inputs from raw RGB.", flush=True)

    # The projector in ImageTokenizer is not consulted by this path. We use
    # only frozen V-JEPA encoder + deterministic spatial compression.
    vjepa_teacher = ImageTokenizer(config, device=args.device)
    raw_entries = read_manifest(raw_root / "manifest.jsonl")
    if args.limit is not None:
        raw_entries = raw_entries[:args.limit]
    written = skipped = 0
    for index, entry in enumerate(raw_entries, start=1):
        episode_id = str(entry["episode_id"])
        if episode_id in completed:
            skipped += 1
            continue
        raw_path = raw_root / str(entry["path"])
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        frames = raw["observations"]
        if source_root is not None:
            source_entry = source_entries.get(episode_id)
            if source_entry is None:
                raise RuntimeError(f"{episode_id}: missing from Qwen token source.")
            tokenizer_name = str(source_entry.get("tokenizer", ""))
            if not tokenizer_name.startswith("qwen"):
                raise RuntimeError(
                    f"{episode_id}: source tokenizer is {tokenizer_name!r}, not native Qwen."
                )
            qwen_episode = load_tokenized_episode(
                source_root / str(source_entry["path"])
            )
            _assert_reusable_qwen_episode(
                raw=raw, qwen=qwen_episode, episode_id=episode_id
            )
            qwen_tokens = qwen_episode.obs_tokens[: len(frames)].clone()
            actions = qwen_episode.actions.clone()
            rewards = qwen_episode.rewards.clone()
            metadata = copy.deepcopy(qwen_episode.metadata)
            split = qwen_episode.split
            env_id = qwen_episode.env_id
        else:
            assert qwen_tokenizer is not None
            qwen_parts = [
                qwen_tokenizer.batch_tokenize(frames[offset : offset + args.frame_batch_size])
                for offset in range(0, len(frames), args.frame_batch_size)
            ]
            qwen_tokens = torch.cat(qwen_parts, dim=0).cpu()
            num_steps = len(raw["model_actions"])
            actions = torch.full((num_steps + 1,), 4, dtype=torch.long)
            actions[:num_steps] = torch.as_tensor(raw["model_actions"], dtype=torch.long)
            rewards = torch.zeros(num_steps + 1, dtype=torch.float32)
            rewards[:num_steps] = torch.as_tensor(raw["rewards"], dtype=torch.float32)
            metadata = copy.deepcopy(raw.get("metadata", {}))
            split = str(entry.get("split", "train"))
            env_id = str(raw["env_id"])

        teacher_parts = [
            vjepa_teacher.batch_semantic_teacher_tokens(
                frames[offset : offset + args.frame_batch_size]
            )
            for offset in range(0, len(frames), args.frame_batch_size)
        ]
        teacher_tokens = torch.cat(teacher_parts, dim=0).cpu()
        metadata["visual_input"] = "qwen2.5_vl_native"
        # Preserve explicit environment state truth for held-out spatial audits.
        # It is never placed in obs_tokens, never passed to the WM, and never
        # used by the V-JEPA teacher loss; it is only a future read-only audit
        # label. Existing paired data remains auditable through paired_raw_episode.
        initial_label = raw.get("metadata", {}).get("initial_state_labels")
        transition_labels = [
            info.get("state_labels") for info in raw.get("infos", [])
        ]
        if initial_label is not None and all(label is not None for label in transition_labels):
            metadata["state_labels"] = [initial_label, *transition_labels]
            metadata["state_labels_source"] = f"raw_{env_id}_rollout"
        metadata["semantic_teacher"] = {
            "name": "vjepa2_compressed_raw",
            "tokens": int(teacher_tokens.shape[1]),
            "dim": int(teacher_tokens.shape[2]),
            "input_is_teacher": False,
        }
        metadata["paired_raw_episode"] = str(raw_path.resolve())
        episode = TokenizedEpisode(
            obs_tokens=qwen_tokens,
            semantic_teacher_tokens=teacher_tokens,
            actions=actions,
            rewards=rewards,
            episode_length=len(raw["model_actions"]),
            env_id=env_id,
            split=split,
            metadata=metadata,
        )
        save_tokenized_episode(
            episode,
            output_root,
            episode_id=episode_id,
            tokenizer="qwen2.5_vl_native+vjepa2_teacher",
        )
        written += 1
        if written % 25 == 0 or index == len(raw_entries):
            print(
                f"[{index}/{len(raw_entries)}] written={written} skipped={skipped}",
                flush=True,
            )
    print(
        f"Paired dataset complete: written={written} skipped={skipped} "
        f"output={output_root / 'tokenized'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
