"""Position-aware repair of an isolated Sokoban Qwen prior LoRA.

This module is deliberately training-only.  It does not add symbolic state to
the World Model input or change its inference contract.  A slotwise decoder is
fitted from training posterior beliefs, frozen, and then used together with
changed-slot V-JEPA targets to supervise only the independent ``wm_prior``
adapter.  The official Stage-1 audit still fits a fresh decoder afterwards.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.storage import load_tokenized_episode, read_manifest
from ..types import BeliefState
from .sokoban_stage1_audit import (
    NUM_CELLS,
    SlotwiseSokobanDecoder,
    _evaluate,
    _one_epoch_train,
    _selected_entries,
    _state_from_raw,
    _successor,
)


@dataclass(frozen=True)
class PriorRepairV2Config:
    seed: int = 20260902
    repair_validation_episodes: int = 500
    teacher_epochs: int = 1
    teacher_batch_size: int = 4
    teacher_lr: float = 1e-3
    teacher_min_player: float = 0.90
    teacher_min_box: float = 0.85
    teacher_min_target: float = 0.88
    teacher_min_wall_f1: float = 0.88
    repair_steps: int = 1500
    batch_size: int = 4
    lr: float = 5e-6
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_episodes: int = 64
    logged_fraction: float = 0.75
    push_fraction_within_logged: float = 0.45
    noop_fraction_within_logged: float = 0.10
    spatial_player_coef: float = 1.0
    spatial_box_coef: float = 1.5
    static_layout_coef: float = 0.15
    latent_dynamics_coef: float = 0.25
    vjepa_absolute_coef: float = 0.10
    vjepa_changed_future_coef: float = 0.25
    vjepa_changed_delta_coef: float = 0.50
    vjepa_change_map_coef: float = 0.50
    changed_slot_topk: int = 4
    change_map_temperature: float = 0.10

    def validate(self) -> None:
        if self.repair_validation_episodes <= 0:
            raise ValueError("repair_validation_episodes must be positive")
        if self.teacher_epochs <= 0 or self.teacher_batch_size <= 0:
            raise ValueError("teacher epochs and batch size must be positive")
        if self.repair_steps <= 0 or self.batch_size <= 0:
            raise ValueError("repair steps and batch size must be positive")
        if self.lr <= 0.0 or self.teacher_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.warmup_steps < 0 or self.eval_every <= 0:
            raise ValueError("warmup_steps must be non-negative and eval_every positive")
        if not 0.0 <= self.logged_fraction <= 1.0:
            raise ValueError("logged_fraction must be in [0, 1]")
        if not 0.0 <= self.push_fraction_within_logged <= 1.0:
            raise ValueError("push_fraction_within_logged must be in [0, 1]")
        if not 0.0 <= self.noop_fraction_within_logged <= 1.0:
            raise ValueError("noop_fraction_within_logged must be in [0, 1]")
        if self.push_fraction_within_logged + self.noop_fraction_within_logged > 1.0:
            raise ValueError("push and noop logged fractions cannot sum above 1")
        if not 1 <= self.changed_slot_topk <= NUM_CELLS:
            raise ValueError("changed_slot_topk must be between 1 and 36")
        if self.change_map_temperature <= 0.0:
            raise ValueError("change_map_temperature must be positive")


@dataclass(frozen=True)
class _EpisodeInfo:
    manifest_index: int
    tokenized_path: str
    actions: tuple[int, ...]
    players: tuple[int, ...]
    boxes: tuple[int, ...]
    target: int
    walls: Tensor  # (36,), bool, CPU; stored once per episode


@dataclass(frozen=True)
class _Example:
    episode_slot: int
    time_index: int
    action: int
    target_player: int
    target_box: int
    kind: str  # noop | walk | push
    counterfactual: bool


@dataclass
class _Batch:
    starts: Tensor
    actions: Tensor
    target_player: Tensor
    target_box: Tensor
    target_target: Tensor
    target_walls: Tensor
    kinds: list[str]
    counterfactual: Tensor
    current_teacher: Tensor
    future_teacher: Tensor
    teacher_valid: Tensor
    logged_next_posterior: Tensor


def _cell(position: Sequence[int]) -> int:
    row, col = int(position[0]), int(position[1])
    if not (0 <= row < 6 and 0 <= col < 6):
        raise ValueError(f"position outside 6x6 board: {position!r}")
    return row * 6 + col


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def split_repair_indices(
    train_indices: Sequence[int], *, validation_size: int, seed: int
) -> tuple[list[int], list[int]]:
    if validation_size <= 0 or validation_size >= len(train_indices):
        raise ValueError(
            f"validation_size={validation_size} is invalid for {len(train_indices)} indices"
        )
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("source training split contains duplicate indices")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    shuffled = [int(train_indices[index]) for index in order]
    validation = shuffled[:validation_size]
    training = shuffled[validation_size:]
    if set(training) & set(validation):
        raise AssertionError("repair training and validation splits overlap")
    return training, validation


def _build_episode_index(
    paired_data_dir: Path,
    manifest_indices: Sequence[int],
) -> tuple[list[_EpisodeInfo], dict[str, list[_Example]]]:
    entries = read_manifest(paired_data_dir / "manifest.jsonl")
    episodes: list[_EpisodeInfo] = []
    pools: dict[str, list[_Example]] = {"noop": [], "walk": [], "push": []}
    for slot, manifest_index in enumerate(manifest_indices):
        entry = entries[int(manifest_index)]
        tokenized_path = paired_data_dir / str(entry["path"])
        raw_location = dict(entry.get("metadata", {})).get("paired_raw_episode")
        if not raw_location:
            raise ValueError(f"{tokenized_path}: paired_raw_episode is missing")
        raw = torch.load(raw_location, map_location="cpu", weights_only=False)
        metadata = dict(raw.get("metadata", {}))
        initial = metadata.get("initial_state_labels")
        infos = list(raw.get("infos", ()))
        actions = tuple(int(action) for action in raw.get("model_actions", ()))
        if initial is None or len(infos) != len(actions):
            raise ValueError(f"{raw_location}: state/action alignment is invalid")
        labels = [_state_from_raw(initial, metadata)]
        labels.extend(
            _state_from_raw(dict(info)["state_labels"], metadata)
            for info in infos
        )
        info = _EpisodeInfo(
            manifest_index=int(manifest_index),
            tokenized_path=str(tokenized_path),
            actions=actions,
            players=tuple(state.player for state in labels),
            boxes=tuple(state.box for state in labels),
            target=labels[0].target,
            walls=labels[0].walls.clone(),
        )
        episodes.append(info)
        for time_index, action in enumerate(actions):
            player_changed = info.players[time_index] != info.players[time_index + 1]
            box_changed = info.boxes[time_index] != info.boxes[time_index + 1]
            kind = "push" if box_changed else ("walk" if player_changed else "noop")
            pools[kind].append(
                _Example(
                    episode_slot=slot,
                    time_index=time_index,
                    action=action,
                    target_player=info.players[time_index + 1],
                    target_box=info.boxes[time_index + 1],
                    kind=kind,
                    counterfactual=False,
                )
            )
    if any(not pools[name] for name in pools):
        sizes = {name: len(pool) for name, pool in pools.items()}
        raise ValueError(
            "repair training needs non-empty noop/walk/push pools; "
            f"sizes={sizes}"
        )
    return episodes, pools


def _state(info: _EpisodeInfo, time_index: int):
    # Reuse the audited symbolic transition implementation without exposing
    # symbols to the World Model input.
    from .sokoban_stage1_audit import _StateLabel

    return _StateLabel(
        player=info.players[time_index],
        box=info.boxes[time_index],
        target=info.target,
        walls=info.walls,
    )


class _ExampleSampler:
    def __init__(
        self,
        episodes: Sequence[_EpisodeInfo],
        pools: dict[str, list[_Example]],
        config: PriorRepairV2Config,
    ) -> None:
        self.episodes = episodes
        self.pools = pools
        self.config = config
        self.rng = random.Random(config.seed)
        self.source_states = [
            (episode_slot, time_index)
            for episode_slot, episode in enumerate(episodes)
            for time_index in range(len(episode.actions))
        ]

    def _logged(self) -> _Example:
        draw = self.rng.random()
        if draw < self.config.push_fraction_within_logged:
            kind = "push"
        elif draw < (
            self.config.push_fraction_within_logged
            + self.config.noop_fraction_within_logged
        ):
            kind = "noop"
        else:
            kind = "walk"
        return self.rng.choice(self.pools[kind])

    def _counterfactual_changed(self) -> _Example:
        for _ in range(1000):
            episode_slot, time_index = self.rng.choice(self.source_states)
            info = self.episodes[episode_slot]
            source = _state(info, time_index)
            action = self.rng.randrange(4)
            if action == info.actions[time_index]:
                continue
            target = _successor(source, action)
            if target.player == source.player and target.box == source.box:
                continue
            kind = "push" if target.box != source.box else "walk"
            return _Example(
                episode_slot=episode_slot,
                time_index=time_index,
                action=action,
                target_player=target.player,
                target_box=target.box,
                kind=kind,
                counterfactual=True,
            )
        raise RuntimeError("could not sample a changed counterfactual transition")

    def sample(self, batch_size: int) -> list[_Example]:
        return [
            self._logged()
            if self.rng.random() < self.config.logged_fraction
            else self._counterfactual_changed()
            for _ in range(batch_size)
        ]


@lru_cache(maxsize=64)
def _cached_tokenized_episode(path: str):
    return load_tokenized_episode(path)


@torch.no_grad()
def _posterior_starts(
    *,
    world_model: Any,
    episode_infos: Sequence[_EpisodeInfo],
    examples: Sequence[_Example],
    device: torch.device,
) -> _Batch:
    tokenized = [
        _cached_tokenized_episode(episode_infos[example.episode_slot].tokenized_path)
        for example in examples
    ]
    max_time = max(example.time_index for example in examples)
    sample_tokens = tokenized[0].obs_tokens
    tokens = torch.zeros(
        len(examples),
        max_time + 2,
        sample_tokens.shape[1],
        sample_tokens.shape[2],
        dtype=sample_tokens.dtype,
        device=device,
    )
    replay_actions = torch.full(
        (len(examples), max_time + 1),
        world_model.config.env.null_action_id,
        dtype=torch.long,
        device=device,
    )
    current_teacher = torch.zeros(
        len(examples),
        sample_tokens.shape[1],
        world_model.config.encoder.semantic_teacher_dim,
        dtype=tokenized[0].semantic_teacher_tokens.dtype,
        device=device,
    )
    future_teacher = torch.zeros_like(current_teacher)
    teacher_valid = torch.zeros(len(examples), dtype=torch.bool, device=device)
    for row, (example, episode) in enumerate(zip(examples, tokenized, strict=True)):
        if episode.semantic_teacher_tokens is None:
            raise ValueError("prior repair v2 requires paired V-JEPA teacher tokens")
        end = example.time_index + 2
        tokens[row, :end] = episode.obs_tokens[:end].to(device)
        replay_actions[row, : example.time_index + 1] = episode.actions[
            : example.time_index + 1
        ].to(device=device, dtype=torch.long)
        current_teacher[row] = episode.semantic_teacher_tokens[
            example.time_index
        ].to(device)
        if not example.counterfactual:
            future_teacher[row] = episode.semantic_teacher_tokens[
                example.time_index + 1
            ].to(device)
            teacher_valid[row] = True

    belief = world_model.get_initial_belief(
        len(examples), device=device, dtype=tokens.dtype
    )
    previous_actions = torch.full(
        (len(examples),),
        world_model.config.env.null_action_id,
        dtype=torch.long,
        device=device,
    )
    starts: list[Tensor | None] = [None] * len(examples)
    for time_index in range(max_time + 1):
        posterior = world_model.posterior_step(
            belief, previous_actions, tokens[:, time_index]
        )
        for row, example in enumerate(examples):
            if example.time_index == time_index:
                starts[row] = posterior.slots[row]
        belief = posterior
        previous_actions = replay_actions[:, time_index]
    if any(start is None for start in starts):
        raise AssertionError("failed to capture a requested posterior start state")
    start_slots = torch.stack([start for start in starts if start is not None])

    logged = teacher_valid.nonzero(as_tuple=False).flatten()
    logged_next = torch.zeros_like(start_slots)
    if logged.numel() > 0:
        next_posterior = world_model.posterior_step(
            BeliefState(start_slots[logged]),
            torch.tensor(
                [examples[int(row)].action for row in logged.tolist()],
                dtype=torch.long,
                device=device,
            ),
            tokens[logged, torch.tensor(
                [examples[int(row)].time_index + 1 for row in logged.tolist()],
                dtype=torch.long,
                device=device,
            )],
        )
        logged_next[logged] = next_posterior.slots

    infos = [episode_infos[example.episode_slot] for example in examples]
    return _Batch(
        starts=start_slots,
        actions=torch.tensor(
            [example.action for example in examples], dtype=torch.long, device=device
        ),
        target_player=torch.tensor(
            [example.target_player for example in examples], dtype=torch.long, device=device
        ),
        target_box=torch.tensor(
            [example.target_box for example in examples], dtype=torch.long, device=device
        ),
        target_target=torch.tensor(
            [info.target for info in infos], dtype=torch.long, device=device
        ),
        target_walls=torch.stack([info.walls for info in infos]).to(device),
        kinds=[example.kind for example in examples],
        counterfactual=torch.tensor(
            [example.counterfactual for example in examples],
            dtype=torch.bool,
            device=device,
        ),
        current_teacher=current_teacher,
        future_teacher=future_teacher,
        teacher_valid=teacher_valid,
        logged_next_posterior=logged_next,
    )


def position_aware_prior_losses(
    *,
    world_model: Any,
    decoder: SlotwiseSokobanDecoder,
    batch: _Batch,
    config: PriorRepairV2Config,
) -> tuple[Tensor, dict[str, float]]:
    prior = world_model.prior_step(BeliefState(batch.starts), batch.actions)
    logits = decoder(prior.slots)
    player_loss = F.cross_entropy(
        logits["player"], batch.target_player, reduction="none"
    )
    box_loss = F.cross_entropy(logits["box"], batch.target_box, reduction="none")
    target_loss = F.cross_entropy(
        logits["target"], batch.target_target, reduction="none"
    )
    wall_loss = F.binary_cross_entropy_with_logits(
        logits["wall"], batch.target_walls.float(), reduction="none"
    ).mean(dim=-1)
    spatial = (
        config.spatial_player_coef * player_loss
        + config.spatial_box_coef * box_loss
    ).mean()
    static_layout = (target_loss + wall_loss).mean()

    logged = batch.teacher_valid
    zero = prior.slots.new_zeros((), dtype=torch.float32)
    latent_dynamics = zero
    vjepa_absolute = zero
    changed_future = zero
    changed_delta = zero
    change_map = zero
    if bool(logged.any()):
        assert world_model.vjepa_teacher_head is not None
        predicted_teacher = world_model.vjepa_teacher_head(prior.slots[logged])
        start_teacher = world_model.vjepa_teacher_head(
            batch.starts[logged]
        ).detach()
        target_current = batch.current_teacher[logged].to(
            dtype=predicted_teacher.dtype
        )
        target_future = batch.future_teacher[logged].to(
            dtype=predicted_teacher.dtype
        )
        predicted_delta = predicted_teacher.float() - start_teacher.float()
        target_delta = target_future.float() - target_current.float()

        vjepa_absolute = (
            1.0
            - F.cosine_similarity(
                predicted_teacher.float(), target_future.float(), dim=-1, eps=1e-6
            )
        ).mean()
        logged_rows = logged.nonzero(as_tuple=False).flatten().tolist()
        changed_rows = torch.tensor(
            [batch.kinds[row] != "noop" for row in logged_rows],
            dtype=torch.bool,
            device=prior.slots.device,
        )
        if bool(changed_rows.any()):
            changed_predicted = predicted_teacher.float()[changed_rows]
            changed_future_target = target_future.float()[changed_rows]
            changed_predicted_delta = predicted_delta[changed_rows]
            changed_target_delta = target_delta[changed_rows]
            target_energy = changed_target_delta.pow(2).mean(dim=-1).sqrt()
            predicted_energy = changed_predicted_delta.pow(2).mean(dim=-1).sqrt()
            topk = min(config.changed_slot_topk, target_energy.shape[-1])
            changed_mask = torch.zeros_like(target_energy)
            changed_mask.scatter_(
                1, target_energy.topk(topk, dim=-1).indices, 1.0
            )
            changed_weight = changed_mask / changed_mask.sum(dim=-1, keepdim=True)
            future_per_slot = 1.0 - F.cosine_similarity(
                changed_predicted,
                changed_future_target,
                dim=-1,
                eps=1e-6,
            )
            delta_per_slot = 1.0 - F.cosine_similarity(
                changed_predicted_delta,
                changed_target_delta,
                dim=-1,
                eps=1e-6,
            )
            changed_future = (
                future_per_slot * changed_weight
            ).sum(dim=-1).mean()
            changed_delta = (
                delta_per_slot * changed_weight
            ).sum(dim=-1).mean()
            temperature = config.change_map_temperature
            target_distribution = F.softmax(
                target_energy / temperature, dim=-1
            )
            predicted_log_distribution = F.log_softmax(
                predicted_energy / temperature, dim=-1
            )
            change_map = F.kl_div(
                predicted_log_distribution,
                target_distribution,
                reduction="batchmean",
            )
        latent_dynamics = F.mse_loss(
            prior.slots[logged].float(),
            batch.logged_next_posterior[logged].float(),
        )

    total = (
        spatial
        + config.static_layout_coef * static_layout
        + config.latent_dynamics_coef * latent_dynamics
        + config.vjepa_absolute_coef * vjepa_absolute
        + config.vjepa_changed_future_coef * changed_future
        + config.vjepa_changed_delta_coef * changed_delta
        + config.vjepa_change_map_coef * change_map
    )
    with torch.no_grad():
        predicted_player = logits["player"].argmax(dim=-1)
        predicted_box = logits["box"].argmax(dim=-1)
        joint = (
            (predicted_player == batch.target_player)
            & (predicted_box == batch.target_box)
        ).float().mean()
        push_mask = torch.tensor(
            [kind == "push" for kind in batch.kinds],
            dtype=torch.bool,
            device=prior.slots.device,
        )
        push_box = (
            (predicted_box[push_mask] == batch.target_box[push_mask]).float().mean()
            if bool(push_mask.any())
            else zero
        )
    metrics = {
        "loss/total": float(total.detach().item()),
        "loss/spatial": float(spatial.detach().item()),
        "loss/player": float(player_loss.mean().detach().item()),
        "loss/box": float(box_loss.mean().detach().item()),
        "loss/static_layout": float(static_layout.detach().item()),
        "loss/latent_dynamics": float(latent_dynamics.detach().item()),
        "loss/vjepa_absolute": float(vjepa_absolute.detach().item()),
        "loss/vjepa_changed_future": float(changed_future.detach().item()),
        "loss/vjepa_changed_delta": float(changed_delta.detach().item()),
        "loss/vjepa_change_map": float(change_map.detach().item()),
        "metric/joint_accuracy": float(joint.item()),
        "metric/push_box_accuracy": float(push_box.item()),
        "metric/counterfactual_fraction": float(batch.counterfactual.float().mean().item()),
    }
    return total, metrics


@torch.no_grad()
def _posterior_teacher_metrics(
    *,
    world_model: Any,
    decoder: SlotwiseSokobanDecoder,
    paths: Sequence[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    from .sokoban_stage1_audit import _collate, _iter_episode_batches, _labels_at

    correct = {"player": 0.0, "box": 0.0, "target": 0.0}
    wall_tp = wall_tn = wall_fp = wall_fn = 0.0
    states = 0.0
    decoder.eval()
    for episodes in _iter_episode_batches(paths, batch_size):
        tokens, actions, lengths = _collate(episodes, device)
        belief = world_model.get_initial_belief(
            len(episodes), device=device, dtype=tokens.dtype
        )
        previous_actions = torch.full(
            (len(episodes),),
            world_model.config.env.null_action_id,
            dtype=torch.long,
            device=device,
        )
        for time_index in range(tokens.shape[1]):
            posterior = world_model.posterior_step(
                belief, previous_actions, tokens[:, time_index]
            )
            valid = lengths > time_index
            if bool(valid.any()):
                labels = _labels_at(episodes, time_index, valid, device)
                logits = decoder(posterior.slots[valid])
                for name in correct:
                    correct[name] += float(
                        (logits[name].argmax(dim=-1) == labels[name]).sum().item()
                    )
                wall = logits["wall"].sigmoid() >= 0.5
                target_wall = labels["walls"].bool()
                wall_tp += float((wall & target_wall).sum().item())
                wall_tn += float(((~wall) & (~target_wall)).sum().item())
                wall_fp += float((wall & (~target_wall)).sum().item())
                wall_fn += float(((~wall) & target_wall).sum().item())
                states += float(labels["player"].numel())
            belief = posterior
            if time_index < actions.shape[1]:
                previous_actions = actions[:, time_index]
    positive_f1 = 2.0 * wall_tp / max(2.0 * wall_tp + wall_fp + wall_fn, 1.0)
    negative_f1 = 2.0 * wall_tn / max(2.0 * wall_tn + wall_fp + wall_fn, 1.0)
    return {
        "states": states,
        "player_accuracy": correct["player"] / max(states, 1.0),
        "box_accuracy": correct["box"] / max(states, 1.0),
        "target_accuracy": correct["target"] / max(states, 1.0),
        "wall_macro_f1": 0.5 * (positive_f1 + negative_f1),
    }


def _teacher_gate(
    metrics: dict[str, float], config: PriorRepairV2Config
) -> tuple[bool, list[str]]:
    thresholds = {
        "player_accuracy": config.teacher_min_player,
        "box_accuracy": config.teacher_min_box,
        "target_accuracy": config.teacher_min_target,
        "wall_macro_f1": config.teacher_min_wall_f1,
    }
    failures = [
        f"{name}={metrics.get(name, float('nan')):.4f} < {threshold:.4f}"
        for name, threshold in thresholds.items()
        if not math.isfinite(metrics.get(name, float("nan")))
        or metrics[name] < threshold
    ]
    return not failures, failures


def _selection_score(metrics: dict[str, dict[str, float]]) -> float:
    logged = metrics["logged_prior_changed_only"]["joint_accuracy"] / 0.35
    counterfactual = (
        metrics["counterfactual_changed_only"]["joint_accuracy"] / 0.30
    )
    # Maximize the weaker normalized Gate first; the mean breaks close ties.
    return min(logged, counterfactual) + 0.01 * (logged + counterfactual)


def _checkpoint_payload(
    *,
    source_checkpoint: Path,
    source_payload: dict[str, Any],
    world_model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    base_step: int,
    repair_step: int,
    config: PriorRepairV2Config,
    repair_train_indices: Sequence[int],
    repair_validation_indices: Sequence[int],
    teacher_metrics: dict[str, float],
    selection_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in source_payload.items()
        if key not in {"model", "optimizer", "scheduler", "step", "prior_repair_v2"}
    }
    payload.update(
        {
            "model": world_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": base_step + repair_step,
            "config": world_model.config,
            "prior_repair_v2": {
                "format": "qwen_position_aware_prior_lora_v2",
                "source_checkpoint": str(source_checkpoint.resolve()),
                "repair_step": repair_step,
                "config": config.__dict__,
                "repair_train_indices": list(repair_train_indices),
                "repair_validation_indices": list(repair_validation_indices),
                "official_validation_indices": list(
                    source_payload["wm_only_refresh"]["val_indices"]
                ),
                "teacher_metrics": dict(teacher_metrics),
                "selection_metrics": selection_metrics,
                "official_validation_used_for_training_or_selection": False,
            },
        }
    )
    return payload


def run_prior_repair_v2(
    *,
    world_model: Any,
    source_checkpoint: str | Path,
    source_payload: dict[str, Any],
    paired_data_dir: str | Path,
    output_dir: str | Path,
    device: torch.device | str,
    config: PriorRepairV2Config,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    config.validate()
    source_checkpoint = Path(source_checkpoint)
    paired_data_dir = Path(paired_data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    refresh = source_payload.get("wm_only_refresh", {})
    original_train = list(refresh.get("train_indices", ()))
    official_validation = list(refresh.get("val_indices", ()))
    if not original_train or not official_validation:
        raise ValueError("source checkpoint has no fixed Stage-1 split")
    repair_train, repair_validation = split_repair_indices(
        original_train,
        validation_size=config.repair_validation_episodes,
        seed=config.seed,
    )
    if set(repair_train) & set(official_validation) or set(repair_validation) & set(
        official_validation
    ):
        raise RuntimeError("repair split overlaps the official validation Gate")

    world_model.eval().requires_grad_(False)
    prior_name = world_model.transition.prior_lora_adapter_name
    if world_model.transition.prior_isolation_mode != "lora":
        raise ValueError("prior repair v2 requires prior_isolation_mode='lora'")
    backbone = world_model.transition.backbone
    backbone.set_lora_adapter_trainable(prior_name, True)
    trainable = list(backbone.lora_adapter_parameters(prior_name))
    expected_ids = {id(parameter) for parameter in trainable}
    actual_ids = {
        id(parameter) for parameter in world_model.parameters() if parameter.requires_grad
    }
    if not expected_ids or actual_ids != expected_ids:
        raise RuntimeError("prior repair v2 optimizer ownership is not isolated")
    if world_model.vjepa_teacher_head is None:
        raise ValueError("source World Model has no V-JEPA teacher projection")
    print(
        "Prior v2 isolation verified: optimizer will own only "
        f"{prior_name!r} ({len(trainable)} tensors); posterior/default LoRA, "
        "Reward Head, and Spatial Teacher are frozen.",
        flush=True,
    )
    print(
        "Repair split verified: "
        f"train={len(repair_train)} internal_validation={len(repair_validation)} "
        f"official_gate={len(official_validation)} overlap=0",
        flush=True,
    )

    train_paths = _selected_entries(
        paired_data_dir=paired_data_dir, indices=repair_train, maximum=0
    )
    validation_paths = _selected_entries(
        paired_data_dir=paired_data_dir, indices=repair_validation, maximum=0
    )
    teacher_path = output / "spatial_teacher.pt"
    decoder = SlotwiseSokobanDecoder(world_model.config.hidden_dim).to(device)
    teacher_history: list[dict[str, float]] = []
    if teacher_path.is_file():
        teacher_payload = torch.load(
            teacher_path, map_location=device, weights_only=False
        )
        if teacher_payload.get("repair_train_indices") != repair_train:
            raise ValueError("existing spatial teacher was fitted on a different split")
        decoder.load_state_dict(teacher_payload["model"], strict=True)
        teacher_metrics = dict(teacher_payload["validation_metrics"])
        print(f"Loaded frozen spatial teacher: {teacher_path}", flush=True)
    else:
        teacher_optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=config.teacher_lr, weight_decay=1e-4
        )
        print(
            "=== Fit training-only slotwise Spatial Teacher ===\n"
            f"  repair_train={len(repair_train)} repair_validation={len(repair_validation)}\n"
            f"  official_gate_validation={len(official_validation)} (untouched)",
            flush=True,
        )
        for epoch in range(1, config.teacher_epochs + 1):
            history, _, _ = _one_epoch_train(
                world_model=world_model,
                decoder=decoder,
                optimizer=teacher_optimizer,
                paths=train_paths,
                batch_size=config.teacher_batch_size,
                device=device,
                epoch=epoch,
            )
            teacher_history.append(history)
        teacher_metrics = _posterior_teacher_metrics(
            world_model=world_model,
            decoder=decoder,
            paths=validation_paths,
            batch_size=config.teacher_batch_size,
            device=device,
        )
        teacher_passed, teacher_failures = _teacher_gate(teacher_metrics, config)
        _atomic_torch_save(
            {
                "format": "sokoban_slotwise_spatial_teacher_v1",
                "model": decoder.state_dict(),
                "source_checkpoint": str(source_checkpoint.resolve()),
                "repair_train_indices": repair_train,
                "repair_validation_indices": repair_validation,
                "official_validation_indices": official_validation,
                "history": teacher_history,
                "validation_metrics": teacher_metrics,
                "gate": {"passed": teacher_passed, "failures": teacher_failures},
            },
            teacher_path,
        )
        if not teacher_passed:
            raise RuntimeError(
                "training-only Spatial Teacher gate failed: "
                + "; ".join(teacher_failures)
            )
        print(
            "Spatial Teacher gate: PASS | "
            + " | ".join(f"{key}={value:.4f}" for key, value in teacher_metrics.items()),
            flush=True,
        )
    teacher_passed, teacher_failures = _teacher_gate(teacher_metrics, config)
    if not teacher_passed:
        raise RuntimeError(
            "loaded Spatial Teacher gate failed: " + "; ".join(teacher_failures)
        )
    decoder.eval().requires_grad_(False)

    episode_infos, pools = _build_episode_index(paired_data_dir, repair_train)
    print(
        "Repair transition pools: "
        + " | ".join(f"{name}={len(pool)}" for name, pool in pools.items()),
        flush=True,
    )
    sampler = _ExampleSampler(episode_infos, pools, config)
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)

    def lr_factor(step: int) -> float:
        if config.warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(config.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    base_step = int(source_payload.get("step", 0))
    eval_paths = validation_paths[: min(config.eval_episodes, len(validation_paths))]
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"

    def validate_and_save(step: int) -> dict[str, dict[str, float]]:
        world_model.eval()
        metrics, _, _ = _evaluate(
            world_model=world_model,
            decoder=decoder,
            paths=eval_paths,
            batch_size=config.teacher_batch_size,
            device=device,
        )
        score = _selection_score(metrics)
        logged = metrics["logged_prior_changed_only"]
        counterfactual = metrics["counterfactual_changed_only"]
        summary = {
            "step": step,
            "selection_score": score,
            "logged_changed_joint": logged["joint_accuracy"],
            "logged_changed_player": logged["player_accuracy"],
            "logged_changed_box": logged["box_accuracy"],
            "counterfactual_changed_joint": counterfactual["joint_accuracy"],
            "counterfactual_changed_player": counterfactual["player_accuracy"],
            "counterfactual_changed_box": counterfactual["box_accuracy"],
        }
        history.append(summary)
        payload = _checkpoint_payload(
            source_checkpoint=source_checkpoint,
            source_payload=source_payload,
            world_model=world_model,
            optimizer=optimizer,
            scheduler=scheduler,
            base_step=base_step,
            repair_step=step,
            config=config,
            repair_train_indices=repair_train,
            repair_validation_indices=repair_validation,
            teacher_metrics=teacher_metrics,
            selection_metrics=metrics,
        )
        _atomic_torch_save(payload, latest_path)
        nonlocal best_score
        if score > best_score:
            best_score = score
            _atomic_torch_save(payload, best_path)
            print(f"Prior v2 best updated: {best_path} score={score:.5f}", flush=True)
        print(
            f"[prior-v2 eval {step}] "
            f"logged_changed_joint={logged['joint_accuracy']:.4f} | "
            f"cf_changed_joint={counterfactual['joint_accuracy']:.4f} | "
            f"score={score:.5f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {f"repair_val/{key}": value for key, value in summary.items() if key != "step"},
                step=base_step + step,
            )
        return metrics

    validate_and_save(0)
    for step in range(1, config.repair_steps + 1):
        examples = sampler.sample(config.batch_size)
        batch = _posterior_starts(
            world_model=world_model,
            episode_infos=episode_infos,
            examples=examples,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        total, train_metrics = position_aware_prior_losses(
            world_model=world_model,
            decoder=decoder,
            batch=batch,
            config=config,
        )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            train_metrics["metric/grad_norm"] = float(grad_norm)
            print(
                f"[prior-v2 train {step}] "
                + " | ".join(
                    f"{key}={value:.6g}" for key, value in sorted(train_metrics.items())
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_metrics.items()},
                    step=base_step + step,
                )
        if step % config.eval_every == 0 or step == config.repair_steps:
            validate_and_save(step)

    report = {
        "format": "qwen_position_aware_prior_lora_v2_report",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "latest_checkpoint": str(latest_path.resolve()),
        "repair_train_episodes": len(repair_train),
        "repair_validation_episodes": len(repair_validation),
        "official_validation_episodes": len(official_validation),
        "official_validation_used_for_training_or_selection": False,
        "transition_pools": {name: len(pool) for name, pool in pools.items()},
        "teacher_metrics": teacher_metrics,
        "history": history,
        "best_selection_score": best_score,
    }
    _atomic_json_dump(report, output / "repair_report.json")
    return report
