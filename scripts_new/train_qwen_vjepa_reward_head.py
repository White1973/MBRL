#!/usr/bin/env python3
"""Train and gate the shared compact reward head on a frozen Qwen WM.

This is the Reward-Head stage of the existing Sokoban pipeline.  It reuses the
proven multi-horizon prior feature collector and the existing RewardHead; it
does not introduce another world model or another reward architecture.

Split contract:
  * checkpoint Stage-1 train split -> reward train/validation/calibration
  * checkpoint Stage-1 validation split -> final official held-out gate only

Only exact endpoint transition rewards are used as labels.  Episode success
metadata is deliberately ignored.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sembelief_wm.data.datasource import TokenizedEpisodeDataset  # noqa: E402
from sembelief_wm.data.schema import TokenizedEpisode  # noqa: E402
from sembelief_wm.data.storage import read_manifest  # noqa: E402
from sembelief_wm.model import QwenTransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.model.checkpoint_semantics import (  # noqa: E402
    validate_world_model_semantics,
)


def _load_reward_tools(engine_root: Path) -> Any:
    """Load the established Sokoban reward utilities without copying them."""
    path = engine_root / "scripts" / "inject_reward_head.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing established Reward-Head trainer: {path}")
    spec = importlib.util.spec_from_file_location("_sokoban_reward_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Reward-Head utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _index_digest(indices: list[int]) -> str:
    encoded = ",".join(str(value) for value in indices).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_reward_dataset(data_dir: Path) -> TokenizedEpisodeDataset:
    """Load Qwen replay while discarding unused V-JEPA teacher tensors early."""
    entries = read_manifest(data_dir / "manifest.jsonl")
    episodes: list[TokenizedEpisode] = []
    for number, entry in enumerate(entries, start=1):
        episode_path = data_dir / str(entry["path"])
        payload = torch.load(episode_path, map_location="cpu", weights_only=False)
        payload["semantic_teacher_tokens"] = None
        episodes.append(TokenizedEpisode(**payload))
        if number % 1000 == 0 or number == len(entries):
            print(
                f"Loaded Reward-Head replay {number}/{len(entries)} "
                "(V-JEPA targets discarded)",
                flush=True,
            )
    return TokenizedEpisodeDataset(episodes)


def _add_operating_metrics(
    values: dict[str, float], logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    result = dict(values)
    threshold = float(result["decision_threshold"])
    predicted = torch.sigmoid(logits) >= threshold
    truth = labels.bool()
    false_positive = int((predicted & ~truth).sum())
    true_negative = int((~predicted & ~truth).sum())
    result["false_positive_rate"] = false_positive / max(
        false_positive + true_negative, 1
    )
    return result


def _fold_standardization(head: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Fold train-only standardization into the classifier's first affine layer."""
    classifier = head.net if head.net is not None else head.compact_net
    assert classifier is not None
    layer = classifier if isinstance(classifier, torch.nn.Linear) else classifier[0]
    if not isinstance(layer, torch.nn.Linear):
        raise TypeError("Reward Head must begin with a linear layer")
    with torch.no_grad():
        original_weight = layer.weight.detach().clone()
        layer.weight.copy_(original_weight / std.unsqueeze(0))
        if layer.bias is None:
            raise ValueError("Reward Head affine layer requires a bias")
        layer.bias.sub_((original_weight * (mean / std).unsqueeze(0)).sum(dim=1))


def _build_gate(
    *,
    overall: dict[str, float],
    per_horizon: dict[str, dict[str, float]],
    horizons: list[int],
    min_auc: float,
    min_ap: float,
    min_precision: float,
    min_recall: float,
    max_fpr: float,
    min_horizon_auc: float,
) -> dict[str, Any]:
    failures: list[str] = []
    if not math.isfinite(overall["auc"]) or overall["auc"] < min_auc:
        failures.append(f"official/auc={overall['auc']:.4f} < {min_auc:.4f}")
    if (
        not math.isfinite(overall["average_precision"])
        or overall["average_precision"] < min_ap
    ):
        failures.append(
            "official/average_precision="
            f"{overall['average_precision']:.4f} < {min_ap:.4f}"
        )
    if overall["precision"] < min_precision:
        failures.append(
            f"official/precision={overall['precision']:.4f} < {min_precision:.4f}"
        )
    if overall["recall"] < min_recall:
        failures.append(
            f"official/recall={overall['recall']:.4f} < {min_recall:.4f}"
        )
    if overall["false_positive_rate"] > max_fpr:
        failures.append(
            "official/false_positive_rate="
            f"{overall['false_positive_rate']:.4f} > {max_fpr:.4f}"
        )
    if overall["brier"] >= overall["brier_baseline"]:
        failures.append(
            f"official/brier={overall['brier']:.5f} is not below "
            f"constant baseline={overall['brier_baseline']:.5f}"
        )
    for horizon in horizons:
        values = per_horizon.get(str(horizon))
        if values is None:
            failures.append(f"official/H{horizon} metrics are missing")
            continue
        if not math.isfinite(values["auc"]) or values["auc"] < min_horizon_auc:
            failures.append(
                f"official/H{horizon}/auc={values['auc']:.4f} "
                f"< {min_horizon_auc:.4f}"
            )
        if values["brier"] >= values["brier_baseline"]:
            failures.append(
                f"official/H{horizon}/brier={values['brier']:.5f} is not below "
                f"baseline={values['brier_baseline']:.5f}"
            )
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "minimum_official_auc": min_auc,
            "minimum_official_average_precision": min_ap,
            "minimum_official_precision": min_precision,
            "minimum_official_recall": min_recall,
            "maximum_official_false_positive_rate": max_fpr,
            "minimum_per_horizon_auc": min_horizon_auc,
            "require_brier_below_constant_baseline": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-Qwen multi-horizon Reward-Head training"
    )
    parser.add_argument(
        "--env-id", choices=("sokoban", "frozenlake"), default="sokoban"
    )
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--stage1-gate-report", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--engine-root", default="/personal/jiayu2026/code/MBRL"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--validation-episodes", type=int, default=1000)
    parser.add_argument("--calibration-episodes", type=int, default=1000)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument("--train-windows-per-episode", type=int, default=2)
    parser.add_argument("--eval-windows-per-episode", type=int, default=1)
    parser.add_argument("--terminal-sample-probability", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--calibration-steps", type=int, default=300)
    parser.add_argument("--calibration-lr", type=float, default=5e-2)
    parser.add_argument("--target-calibration-precision", type=float, default=0.75)
    parser.add_argument("--min-calibration-recall", type=float, default=0.10)
    parser.add_argument("--min-official-auc", type=float, default=0.75)
    parser.add_argument("--min-official-ap", type=float, default=0.35)
    parser.add_argument("--min-official-precision", type=float, default=0.70)
    parser.add_argument("--min-official-recall", type=float, default=0.10)
    parser.add_argument("--max-official-fpr", type=float, default=0.10)
    parser.add_argument("--min-horizon-auc", type=float, default=0.65)
    parser.add_argument("--wandb-project", default="jiayu-mbrl")
    parser.add_argument("--wandb-group", default="qwen_vjepa_reward_head")
    parser.add_argument("--wandb-run-name", default="qwen_vjepa_reward_head_s1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.head_hidden_dim != 0:
        raise ValueError(
            "This validated stage uses the existing compact linear Reward Head; "
            "--head-hidden-dim must be 0"
        )
    if len(set(args.horizons)) != len(args.horizons) or any(
        value <= 0 for value in args.horizons
    ):
        raise ValueError("--horizons must contain unique positive integers")
    if args.validation_episodes <= 0 or args.calibration_episodes <= 0:
        raise ValueError("Reward validation/calibration sizes must be positive")

    checkpoint_path = Path(args.wm_checkpoint).resolve()
    gate_report_path = Path(args.stage1_gate_report).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    report_path = output_dir / "reward_head_gate_report.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing Stage-1 checkpoint: {checkpoint_path}")
    if not gate_report_path.is_file():
        raise FileNotFoundError(f"missing Stage-1 Gate report: {gate_report_path}")
    if not (data_dir / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"missing paired replay: {data_dir}")
    if latest_path.exists() or best_path.exists():
        raise FileExistsError(
            f"refusing to overwrite Reward-Head checkpoints in {output_dir}"
        )

    stage1_gate = json.loads(gate_report_path.read_text(encoding="utf-8"))
    if not bool(stage1_gate.get("gate", {}).get("passed")):
        raise RuntimeError(
            "Stage-1 spatial/dynamics Gate has not passed; refusing Reward-Head training"
        )
    audited_checkpoint = Path(
        stage1_gate.get("checkpoint", {}).get("path", "")
    ).resolve()
    if audited_checkpoint != checkpoint_path:
        raise ValueError(
            "Stage-1 Gate report audits a different checkpoint: "
            f"report={audited_checkpoint}, requested={checkpoint_path}"
        )

    reward_tools = _load_reward_tools(Path(args.engine_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen Reward-Head feature collection")

    print(f"Loading frozen Stage-1 WM: {checkpoint_path}", flush=True)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("source checkpoint is not self-contained")
    source_config = payload["config"]
    if args.env_id not in source_config.env.env_ids:
        raise ValueError(
            f"Reward-Head source is for {source_config.env.env_ids}, "
            f"not requested env {args.env_id!r}"
        )
    if source_config.encoder.encoder_type != "qwen":
        raise ValueError("Reward-Head stage requires the Qwen-native Stage-1 WM")
    if source_config.training.prior_isolation_mode != "lora":
        raise ValueError("Reward-Head stage requires the repaired independent prior LoRA")
    refresh = payload.get("wm_only_refresh")
    if not isinstance(refresh, dict):
        raise ValueError("source checkpoint has no fixed Stage-1 split metadata")
    stage1_train = [int(value) for value in refresh.get("train_indices", ())]
    official_test_pool = [int(value) for value in refresh.get("val_indices", ())]
    upstream_selection = {
        int(value)
        for value in payload.get("spatial_prior_repair", {})
        .get("split", {})
        .get("selection_indices", ())
    }
    if upstream_selection - set(official_test_pool):
        raise RuntimeError(
            "upstream prior-selection indices are outside the Stage-1 validation split"
        )
    official_test = [
        index for index in official_test_pool if index not in upstream_selection
    ]
    if not stage1_train or not official_test:
        raise ValueError("source checkpoint has empty Stage-1 train/validation split")
    if set(stage1_train) & set(official_test):
        raise RuntimeError("source Stage-1 train and official validation splits overlap")

    shuffled = list(stage1_train)
    random.Random(args.seed).shuffle(shuffled)
    held_inside_train = args.validation_episodes + args.calibration_episodes
    if held_inside_train >= len(shuffled):
        raise ValueError("Reward validation/calibration consume the Stage-1 train split")
    reward_validation = shuffled[: args.validation_episodes]
    reward_calibration = shuffled[
        args.validation_episodes : held_inside_train
    ]
    reward_train = shuffled[held_inside_train:]
    split_indices = {
        "train": reward_train,
        "validation": reward_validation,
        "calibration": reward_calibration,
        "official_test": official_test,
    }
    split_sets = {name: set(values) for name, values in split_indices.items()}
    for left, left_values in split_sets.items():
        for right, right_values in split_sets.items():
            if left < right and left_values & right_values:
                raise RuntimeError(f"Reward split overlap: {left} vs {right}")
    print(
        "Reward split verified: "
        f"train={len(reward_train)} validation={len(reward_validation)} "
        f"calibration={len(reward_calibration)} "
        f"official_test={len(official_test)} "
        f"upstream_selection_excluded={len(upstream_selection)} overlap=0",
        flush=True,
    )

    config = copy.deepcopy(source_config)
    config.reward.head_hidden_dim = args.head_hidden_dim
    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    world_model = WorldModel(config, backbone).to(device)
    source_state = payload.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("source checkpoint has no model state")
    frozen_state = {
        key: value
        for key, value in source_state.items()
        if not key.startswith("reward_head.")
    }
    incompatible = world_model.load_state_dict(frozen_state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("reward_head.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "frozen WM mismatch: "
            f"missing={missing[:20]} unexpected={incompatible.unexpected_keys[:20]}"
        )
    validate_world_model_semantics(
        payload,
        attention_mode=config.backbone.attention_mode,
        context=f"Reward-Head source {checkpoint_path}",
    )
    world_model.eval().requires_grad_(False)
    world_model.reward_head.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in world_model.parameters()
        if parameter.requires_grad
    )
    expected_trainable = sum(
        parameter.numel() for parameter in world_model.reward_head.parameters()
    )
    if trainable != expected_trainable or trainable <= 0:
        raise RuntimeError("optimizer ownership is not isolated to Reward Head")
    print(
        f"Frozen WM verified; trainable Reward Head only: "
        f"{world_model.reward_head.spec}, params={trainable}",
        flush=True,
    )

    dataset = _load_reward_dataset(data_dir)
    wrong_env = sorted(
        {episode.env_id for episode in dataset.episodes if episode.env_id != args.env_id}
    )
    if wrong_env:
        raise ValueError(
            f"Reward-Head replay contains non-{args.env_id} episodes: {wrong_env}"
        )
    if max(stage1_train + official_test) >= len(dataset.episodes):
        raise IndexError("checkpoint split index exceeds paired replay length")
    dtype = next(world_model.parameters()).dtype
    null_action = torch.full(
        (1,), config.env.null_action_id, dtype=torch.long, device=device
    )

    cache_dir = Path(
        args.feature_cache_dir
        or output_dir.parent.parent.parent
        / "data"
        / "reward_head_feature_cache"
        / output_dir.name
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stat = checkpoint_path.stat()
    base_cache_spec = {
        "version": 1,
        "source_checkpoint": str(checkpoint_path),
        "source_size": checkpoint_stat.st_size,
        "source_mtime_ns": checkpoint_stat.st_mtime_ns,
        "horizons": args.horizons,
        "independent_horizon_starts": True,
        "success_threshold": args.success_threshold,
        "feature_batch_size": args.feature_batch_size,
    }

    def collect(
        name: str,
        *,
        terminal_probability: float,
        windows_per_episode: int,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_path = cache_dir / f"{name}.pt"
        spec = {
            **base_cache_spec,
            "split": name,
            "indices_sha256": _index_digest(split_indices[name]),
            "episodes": len(split_indices[name]),
            "sampling_seed": args.seed + seed_offset,
            "terminal_probability": terminal_probability,
            "windows_per_episode": windows_per_episode,
        }
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("spec") != spec:
                raise ValueError(f"incompatible feature cache: {cache_path}")
            print(f"Reusing {name} feature cache: {cache_path}", flush=True)
            return cached["features"]
        result = reward_tools.collect_multi_horizon_features(
            world_model=world_model,
            dataset=dataset,
            indices=split_indices[name],
            device=device,
            dtype=dtype,
            null_action=null_action,
            seed=args.seed + seed_offset,
            horizons=args.horizons,
            success_threshold=args.success_threshold,
            terminal_sample_probability=terminal_probability,
            windows_per_episode=windows_per_episode,
            independent_horizon_starts=True,
            feature_batch_size=args.feature_batch_size,
            progress_label=name,
        )
        temporary = cache_path.with_name(f".{cache_path.name}.tmp")
        torch.save({"spec": spec, "features": result}, temporary)
        os.replace(temporary, cache_path)
        print(f"Saved resumable {name} features: {cache_path}", flush=True)
        return result

    train_x, train_y, train_h = collect(
        "train",
        terminal_probability=args.terminal_sample_probability,
        windows_per_episode=args.train_windows_per_episode,
        seed_offset=0,
    )
    val_x, val_y, val_h = collect(
        "validation",
        terminal_probability=0.0,
        windows_per_episode=args.eval_windows_per_episode,
        seed_offset=1,
    )
    calibration_x, calibration_y, calibration_h = collect(
        "calibration",
        terminal_probability=0.0,
        windows_per_episode=args.eval_windows_per_episode,
        seed_offset=2,
    )
    test_x, test_y, test_h = collect(
        "official_test",
        terminal_probability=0.0,
        windows_per_episode=args.eval_windows_per_episode,
        seed_offset=3,
    )
    for name, labels, horizon_ids in (
        ("train", train_y, train_h),
        ("validation", val_y, val_h),
        ("calibration", calibration_y, calibration_h),
        ("official_test", test_y, test_h),
    ):
        reward_tools.require_two_classes(name, labels)
        print(
            f"{name}: samples={len(labels)} positive={float(labels.mean()):.4f}",
            flush=True,
        )
        for horizon in args.horizons:
            mask = horizon_ids == horizon
            reward_tools.require_two_classes(f"{name}/H{horizon}", labels[mask])

    # Feature collection is the expensive frozen-WM part.  Release the dataset
    # now; only compact pooled vectors are needed for head optimization.
    del dataset
    gc.collect()

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    calibration_x, calibration_y = calibration_x.to(device), calibration_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    calibration_h = calibration_h.to(device)
    test_h = test_h.to(device)
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-4)
    train_z = (train_x - feature_mean) / feature_std
    val_z = (val_x - feature_mean) / feature_std
    calibration_z = (calibration_x - feature_mean) / feature_std

    optimizer = torch.optim.AdamW(
        world_model.reward_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = torch.tensor(args.pos_weight, device=device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    best_state: dict[str, torch.Tensor] | None = None
    best_ap = -float("inf")
    stale_epochs = 0
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.wandb_run_name,
                job_type="reward_head",
                config={
                    "env_id": args.env_id,
                    "source_checkpoint": str(checkpoint_path),
                    "horizons": args.horizons,
                    "reward_train_episodes": len(reward_train),
                    "reward_validation_episodes": len(reward_validation),
                    "reward_calibration_episodes": len(reward_calibration),
                    "official_test_episodes": len(official_test),
                    "train_windows_per_episode": args.train_windows_per_episode,
                    "head_hidden_dim": args.head_hidden_dim,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                },
            )
        except Exception as exc:
            if os.environ.get("REQUIRE_WANDB", "0") == "1":
                raise
            print(f"Warning: W&B disabled ({type(exc).__name__}: {exc})", flush=True)

    try:
        for epoch in range(1, args.epochs + 1):
            world_model.reward_head.train()
            permutation = torch.randperm(len(train_y), generator=generator)
            total_loss = 0.0
            total_count = 0
            for start in range(0, len(permutation), args.batch_size):
                indices = permutation[start : start + args.batch_size].to(device)
                logits = world_model.reward_head.forward_pooled(train_z[indices])
                loss = F.binary_cross_entropy_with_logits(
                    logits, train_y[indices], pos_weight=pos_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.reward_head.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                total_count += len(indices)
            world_model.reward_head.eval()
            with torch.no_grad():
                train_values = reward_tools.metrics(
                    world_model.reward_head.forward_pooled(train_z), train_y
                )
                val_values = reward_tools.metrics(
                    world_model.reward_head.forward_pooled(val_z), val_y
                )
            improved = val_values["average_precision"] > best_ap + 1e-6
            if improved:
                best_ap = float(val_values["average_precision"])
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in world_model.reward_head.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"[reward-head epoch {epoch:02d}] "
                f"train/loss={total_loss/max(total_count, 1):.5f} "
                f"train/auc={train_values['auc']:.4f} "
                f"val/auc={val_values['auc']:.4f} "
                f"val/ap={val_values['average_precision']:.4f} "
                f"val/brier={val_values['brier']:.5f} best_ap={best_ap:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "reward_head/epoch": epoch,
                        "reward_head/train_loss": total_loss / max(total_count, 1),
                        "reward_head/train_auc": train_values["auc"],
                        "reward_head/val_auc": val_values["auc"],
                        "reward_head/val_average_precision": val_values[
                            "average_precision"
                        ],
                        "reward_head/val_brier": val_values["brier"],
                    },
                    step=epoch,
                )
            if stale_epochs >= args.patience:
                print(f"Reward Head early stopping at epoch {epoch}", flush=True)
                break

        if best_state is None:
            raise RuntimeError("no Reward-Head candidate was selected")
        world_model.reward_head.load_state_dict(best_state)
        world_model.reward_head.to(device).eval()
        _fold_standardization(world_model.reward_head, feature_mean, feature_std)
        scale, calibration_bias = reward_tools.calibrate(
            world_model.reward_head,
            calibration_x,
            calibration_y,
            steps=args.calibration_steps,
            lr=args.calibration_lr,
        )
        calibration_gate_failure: str | None = None
        with torch.no_grad():
            calibration_logits = world_model.reward_head.forward_pooled(calibration_x)
            try:
                decision_threshold, calibration_precision, calibration_recall = (
                    reward_tools.select_precision_threshold(
                        calibration_logits,
                        calibration_y,
                        target_precision=args.target_calibration_precision,
                        min_recall=args.min_calibration_recall,
                    )
                )
            except RuntimeError as exc:
                # Keep the fully trained candidate and diagnostics even when
                # calibration cannot find a deployable operating point.
                calibration_gate_failure = str(exc)
                decision_threshold = reward_tools.select_f1_threshold(
                    calibration_logits,
                    calibration_y,
                )
                diagnostic = reward_tools.metrics(
                    calibration_logits,
                    calibration_y,
                    decision_threshold=decision_threshold,
                )
                calibration_precision = diagnostic["precision"]
                calibration_recall = diagnostic["recall"]
            official_logits = world_model.reward_head.forward_pooled(test_x)
            calibration_values = reward_tools.metrics(
                calibration_logits,
                calibration_y,
                decision_threshold=decision_threshold,
            )
            official_values = _add_operating_metrics(
                reward_tools.metrics(
                    official_logits,
                    test_y,
                    decision_threshold=decision_threshold,
                ),
                official_logits,
                test_y,
            )
            official_by_horizon = reward_tools.metrics_by_horizon(
                official_logits,
                test_y,
                test_h,
                horizons=args.horizons,
                decision_threshold=decision_threshold,
            )
            official_by_horizon = {
                horizon: _add_operating_metrics(
                    values,
                    official_logits[test_h == int(horizon)],
                    test_y[test_h == int(horizon)],
                )
                for horizon, values in official_by_horizon.items()
            }

        gate = _build_gate(
            overall=official_values,
            per_horizon=official_by_horizon,
            horizons=args.horizons,
            min_auc=args.min_official_auc,
            min_ap=args.min_official_ap,
            min_precision=args.min_official_precision,
            min_recall=args.min_official_recall,
            max_fpr=args.max_official_fpr,
            min_horizon_auc=args.min_horizon_auc,
        )
        if calibration_gate_failure is not None:
            gate["failures"].insert(
                0, "calibration operating-point Gate failed: " + calibration_gate_failure
            )
            gate["passed"] = False
        report = {
            "format": "qwen_vjepa_reward_head_gate_v1",
            "source_checkpoint": str(checkpoint_path),
            "stage1_gate_report": str(gate_report_path),
            "stage1_gate_passed": True,
            "split": {
                "stage1_train_episodes": len(stage1_train),
                "reward_train_indices": reward_train,
                "reward_validation_indices": reward_validation,
                "reward_calibration_indices": reward_calibration,
                "official_test_indices": official_test,
                "official_test_pool_episodes": len(official_test_pool),
                "upstream_selection_indices_excluded": sorted(upstream_selection),
                "official_test_used_for_training_selection_or_calibration": False,
            },
            "head": {
                "readout": config.reward.readout,
                "hidden_dim": args.head_hidden_dim,
                "trainable_parameters": trainable,
                "horizons": args.horizons,
                "best_validation_average_precision": best_ap,
                "calibration_scale": scale,
                "calibration_bias": calibration_bias,
                "decision_threshold": decision_threshold,
                "calibration_threshold_precision": calibration_precision,
                "calibration_threshold_recall": calibration_recall,
            },
            "metrics": {
                "calibration": calibration_values,
                "official_test": official_values,
                "official_test_by_horizon": official_by_horizon,
            },
            "gate": gate,
        }

        new_state = {
            key: value
            for key, value in source_state.items()
            if not key.startswith("reward_head.")
        }
        for key, value in world_model.reward_head.state_dict().items():
            new_state[f"reward_head.{key}"] = value.detach().cpu()
        new_checkpoint = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "optimizer", "scheduler"}
        }
        new_checkpoint["config"] = config
        new_checkpoint["model"] = new_state
        new_checkpoint["injected_reward_head"] = True
        new_checkpoint["reward_head_injection"] = {
            "version": 6,
            "head_hidden_dim": args.head_hidden_dim,
            "horizon": None,
            "horizons": args.horizons,
            "independent_horizon_starts": True,
            "success_threshold": args.success_threshold,
            "decision_threshold": decision_threshold,
            "calibration_scale": scale,
            "calibration_bias": calibration_bias,
            "calibration_metrics": calibration_values,
            "test_metrics": official_values,
            "test_metrics_by_horizon": official_by_horizon,
            "split_indices": split_indices,
            "official_test_used_for_training_selection_or_calibration": False,
            "gate": gate,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(new_checkpoint, latest_path)
        _atomic_json(report, report_path)
        print(
            "Official Reward-Head metrics: "
            f"AUC={official_values['auc']:.4f} "
            f"AP={official_values['average_precision']:.4f} "
            f"precision={official_values['precision']:.4f} "
            f"recall={official_values['recall']:.4f} "
            f"FPR={official_values['false_positive_rate']:.4f} "
            f"Brier={official_values['brier']:.5f}/"
            f"{official_values['brier_baseline']:.5f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "reward_head/gate_passed": float(gate["passed"]),
                    "reward_head/official_auc": official_values["auc"],
                    "reward_head/official_average_precision": official_values[
                        "average_precision"
                    ],
                    "reward_head/official_precision": official_values["precision"],
                    "reward_head/official_recall": official_values["recall"],
                    "reward_head/official_false_positive_rate": official_values[
                        "false_positive_rate"
                    ],
                    "reward_head/official_brier": official_values["brier"],
                }
            )
        if gate["passed"]:
            os.link(latest_path, best_path)
            print(f"Reward Head Gate: PASS", flush=True)
            print(f"Reward Head latest: {latest_path}", flush=True)
            print(f"Reward Head best:   {best_path}", flush=True)
        else:
            print(
                "Reward Head Gate: FAIL: " + "; ".join(gate["failures"]),
                flush=True,
            )
            print(f"Rejected candidate retained as latest: {latest_path}", flush=True)
            raise RuntimeError(f"Reward Head Gate failed: {report_path}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
