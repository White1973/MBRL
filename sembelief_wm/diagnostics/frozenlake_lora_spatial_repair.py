"""Audit-aligned spatial repair for the existing Qwen ``wm_prior`` LoRA.

This module adds no alternative world-model architecture.  It loads the shared
``WorldModel``, freezes every parameter, re-enables exactly the named prior
LoRA, and uses the same flattened-slot linear probe as the final audit for
future-position supervision and checkpoint selection.  Actual transitions are
also distilled back into the frozen posterior representation geometry.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from torch import Tensor, nn
import torch.nn.functional as F

from ..data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from ..model import QwenTransitionBackbone, WorldModel
from ..model.checkpoint_semantics import validate_world_model_semantics
from ..types import BeliefState
from .frozenlake_prior_repair import _posterior_rows


@dataclass(frozen=True)
class SpatialRepairGates:
    posterior_accuracy: float = 0.98
    prior_accuracy: float = 0.75
    counterfactual_accuracy: float = 0.70
    changed_accuracy: float = 0.60
    noop_accuracy: float = 0.70
    minimum_action_accuracy: float = 0.55


@dataclass(frozen=True)
class SpatialRepairWeights:
    actual_position: float = 1.0
    counterfactual_position: float = 1.0
    posterior_latent: float = 1.0
    posterior_cosine: float = 1.0
    posterior_delta: float = 0.50
    vjepa_prior: float = 0.25
    vjepa_delta: float = 0.50


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _indices_digest(indices: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in indices).encode("ascii")
    ).hexdigest()


@lru_cache(maxsize=4096)
def _transition_table(rows_key: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    env = FrozenLakeEnv(desc=list(rows_key), is_slippery=False)
    try:
        table = []
        for state in range(16):
            row = []
            for action in range(4):
                transitions = env.P[state][action]
                if (
                    len(transitions) != 1
                    or abs(float(transitions[0][0]) - 1.0) > 1e-8
                ):
                    raise RuntimeError(
                        "FrozenLake spatial repair requires deterministic VAGEN maps"
                    )
                row.append(int(transitions[0][1]))
            table.append(tuple(row))
        return tuple(table)
    finally:
        env.close()


def _states(episode: Any) -> list[int]:
    table = _transition_table(tuple(episode.metadata["map_rows"]))
    state = int(episode.metadata["start_state"])
    result = [state]
    for action in episode.actions[: episode.episode_length].tolist():
        state = table[state][int(action)]
        result.append(state)
    return result


def _balanced_mean(values: Tensor, changed: Tensor) -> Tensor:
    parts = []
    for mask in (changed, ~changed):
        if bool(mask.any()):
            parts.append(values[mask].mean())
    if not parts:
        raise RuntimeError("spatial repair batch has no valid targets")
    return torch.stack(parts).mean()


def _prior_state(model: WorldModel) -> dict[str, Tensor]:
    marker = f".{model.transition.prior_lora_adapter_name}."
    result = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if marker in key and "lora_" in key.lower()
    }
    if not result:
        raise RuntimeError(f"checkpoint contains no prior LoRA tensors for {marker}")
    return result


def _prior_keys(model: WorldModel) -> set[str]:
    marker = f".{model.transition.prior_lora_adapter_name}."
    return {
        key
        for key in model.state_dict()
        if marker in key and "lora_" in key.lower()
    }


def _restore_prior_state(model: WorldModel, state: dict[str, Tensor]) -> None:
    expected = _prior_keys(model)
    if set(state) != expected:
        raise RuntimeError(
            "prior adapter state mismatch: "
            f"missing={sorted(expected-set(state))[:10]}, "
            f"unexpected={sorted(set(state)-expected)[:10]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected prior adapter keys: {incompatible.unexpected_keys}")




@torch.no_grad()
def _linear_probe_accuracy(
    probe: nn.Linear,
    slots: Tensor,
    labels: Tensor,
    device: torch.device,
) -> float:
    correct = 0
    for begin in range(0, len(labels), 256):
        features = slots[begin : begin + 256].to(device).flatten(1).float()
        prediction = probe(features).argmax(-1).cpu()
        correct += int((prediction == labels[begin : begin + 256]).sum())
    return correct / max(len(labels), 1)


def _train_or_load_linear_probe(
    *,
    model: WorldModel,
    config: Any,
    dataset: TokenizedEpisodeDataset,
    source_checkpoint: Path,
    train_indices: list[int],
    validation_indices: list[int],
    output: Path,
    device: torch.device,
    seed: int,
    train_episodes: int,
    validation_episodes: int,
    steps: int,
    eval_every: int,
    posterior_gate: float,
) -> tuple[nn.Linear, dict[str, float], dict[str, list[int]]]:
    """Train the exact flattened-slot linear readout used by final audit."""
    probe_path = output / "spatial_linear_probe.pt"
    train_pool = list(train_indices)
    validation_pool = list(validation_indices)
    random.Random(seed).shuffle(train_pool)
    random.Random(seed + 1).shuffle(validation_pool)
    selected_train = train_pool[:train_episodes]
    selected_validation = validation_pool[:validation_episodes]
    if len(selected_train) != train_episodes or len(selected_validation) != validation_episodes:
        raise ValueError("fixed Stage-1 split is smaller than the linear-probe request")

    input_dim = config.belief.num_slots * config.hidden_dim
    probe = nn.Linear(input_dim, 16).to(device)
    if probe_path.is_file():
        saved = torch.load(probe_path, map_location=device, weights_only=False)
        if saved.get("format") != "frozenlake_audit_linear_probe_v3":
            raise RuntimeError(f"incompatible audit-aligned probe: {probe_path}")
        if Path(saved.get("source_checkpoint", "")).resolve() != source_checkpoint:
            raise RuntimeError("audit-aligned probe belongs to a different source WM")
        if saved.get("train_indices") != selected_train:
            raise RuntimeError("audit-aligned probe uses a different training panel")
        if saved.get("validation_indices") != selected_validation:
            raise RuntimeError("audit-aligned probe uses a different validation panel")
        probe.load_state_dict(saved["model"], strict=True)
        metrics = dict(saved["validation_metrics"])
        if metrics["posterior_accuracy"] < posterior_gate:
            raise RuntimeError("reused audit-aligned probe fails posterior Gate")
        probe.requires_grad_(False).eval()
        print(f"Reusing audit-aligned linear spatial probe: {probe_path}", flush=True)
        return probe, metrics, {
            "probe_train_indices": selected_train,
            "probe_validation_indices": selected_validation,
        }

    print(
        "Extracting frozen posterior states for audit-aligned linear probe: "
        f"train={train_episodes} validation={validation_episodes}",
        flush=True,
    )
    train_eps = [dataset.episodes[index] for index in selected_train]
    validation_eps = [dataset.episodes[index] for index in selected_validation]
    train_slots, train_labels = _posterior_rows(model, train_eps, config, device)
    validation_slots, validation_labels = _posterior_rows(
        model, validation_eps, config, device
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    probe.train()
    for step in range(1, steps + 1):
        ids = torch.randint(
            0,
            len(train_slots),
            (min(256, len(train_slots)),),
            generator=generator,
        )
        features = train_slots[ids].to(device).flatten(1).float()
        loss = F.cross_entropy(
            probe(features), train_labels["player"][ids].to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            accuracy = _linear_probe_accuracy(
                probe, validation_slots, validation_labels["player"], device
            )
            print(
                f"[audit-linear-probe {step}] loss={float(loss.detach()):.5f} | "
                f"validation/posterior_accuracy={accuracy:.4f}",
                flush=True,
            )
    probe.eval()
    metrics = {
        "posterior_accuracy": _linear_probe_accuracy(
            probe, validation_slots, validation_labels["player"], device
        )
    }
    _atomic_torch_save(
        {
            "format": "frozenlake_audit_linear_probe_v3",
            "source_checkpoint": str(source_checkpoint),
            "model": probe.state_dict(),
            "input_dim": input_dim,
            "validation_metrics": metrics,
            "train_indices": selected_train,
            "validation_indices": selected_validation,
            "steps": steps,
            "seed": seed,
        },
        probe_path,
    )
    del train_slots, validation_slots
    gc.collect()
    if metrics["posterior_accuracy"] < posterior_gate:
        raise RuntimeError(
            "audit-aligned linear probe Gate failed: "
            f"posterior_accuracy={metrics['posterior_accuracy']:.4f} < {posterior_gate:.4f}"
        )
    probe.requires_grad_(False).eval()
    return probe, metrics, {
        "probe_train_indices": selected_train,
        "probe_validation_indices": selected_validation,
    }


def _selection_key(metrics: dict[str, float], gates: SpatialRepairGates) -> tuple[float, ...]:
    # The weakest normalized release metric dominates.  Average accuracy and
    # minimum per-action accuracy break ties without requiring monotonic latent loss.
    weakest = min(
        metrics["prior_accuracy"] / gates.prior_accuracy,
        metrics["counterfactual_accuracy"] / gates.counterfactual_accuracy,
    )
    return (
        weakest,
        0.5 * (metrics["prior_accuracy"] + metrics["counterfactual_accuracy"]),
        metrics["minimum_action_accuracy"],
        metrics["changed_accuracy"],
    )


def run_lora_spatial_prior_repair(
    *,
    checkpoint: str,
    dataset: TokenizedEpisodeDataset,
    output_dir: str,
    device: torch.device,
    seed: int,
    probe_steps: int = 1000,
    repair_steps: int = 3000,
    eval_every: int = 250,
    probe_train_episodes: int = 1000,
    probe_validation_episodes: int = 500,
    selection_validation_episodes: int = 256,
    batch_size: int = 4,
    lr: float = 3e-6,
    gates: SpatialRepairGates = SpatialRepairGates(),
    weights: SpatialRepairWeights = SpatialRepairWeights(),
    resume: bool = False,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint).resolve()
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = copy.deepcopy(payload["config"])
    if config.encoder.encoder_type != "qwen":
        raise ValueError("spatial prior repair requires Qwen-native observations")
    if "frozenlake" not in config.env.env_ids:
        raise ValueError(f"checkpoint env_ids={config.env.env_ids}, not FrozenLake")
    if config.training.prior_isolation_mode != "lora":
        raise ValueError("spatial prior repair requires prior_isolation_mode='lora'")
    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    model = WorldModel(config, backbone).to(device)
    model.load_state_dict(payload["model"], strict=True)
    validate_world_model_semantics(
        payload,
        attention_mode=config.backbone.attention_mode,
        context=f"FrozenLake spatial prior source {checkpoint_path}",
    )
    model.requires_grad_(False)
    model.eval()

    split = payload.get("wm_only_refresh", {})
    train_indices = [int(value) for value in split.get("train_indices", ())]
    validation_indices = [int(value) for value in split.get("val_indices", ())]
    if not train_indices or not validation_indices:
        raise ValueError("source checkpoint has no fixed Stage-1 train/validation split")
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Stage-1 train and validation splits overlap")
    if max(train_indices + validation_indices) >= len(dataset.episodes):
        raise IndexError("checkpoint split exceeds paired replay")
    selection_pool = list(validation_indices)
    random.Random(seed + 404).shuffle(selection_pool)
    selection_indices = selection_pool[:selection_validation_episodes]
    if len(selection_indices) != selection_validation_episodes:
        raise ValueError("selection validation panel is smaller than requested")

    position_probe, probe_metrics, probe_split = _train_or_load_linear_probe(
        model=model,
        config=config,
        dataset=dataset,
        source_checkpoint=checkpoint_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        output=output,
        device=device,
        seed=seed,
        train_episodes=probe_train_episodes,
        validation_episodes=probe_validation_episodes,
        steps=probe_steps,
        eval_every=max(1, min(eval_every, 250)),
        posterior_gate=gates.posterior_accuracy,
    )

    prior_name = model.transition.prior_lora_adapter_name
    set_trainable = getattr(model.transition.backbone, "set_lora_adapter_trainable", None)
    adapter_parameters = getattr(model.transition.backbone, "lora_adapter_parameters", None)
    if not callable(set_trainable) or not callable(adapter_parameters):
        raise TypeError("Qwen backbone does not expose named LoRA ownership")
    set_trainable(prior_name, True)
    parameters = list(adapter_parameters(prior_name))
    expected_ids = {id(parameter) for parameter in parameters}
    actual_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if not expected_ids or expected_ids != actual_ids:
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        raise RuntimeError(f"prior-only optimizer ownership violated: {names[:20]}")
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)

    train_eps = [dataset.episodes[index] for index in train_indices]
    selection_eps = [dataset.episodes[index] for index in selection_indices]
    # Collation is stateless and identical for train and held-out episodes.
    # Use one collator so split selection never depends on object identity.
    collator = OfflineDataSource(TokenizedEpisodeDataset(train_eps), config)
    state_cache = {id(episode): _states(episode) for episode in train_eps + selection_eps}
    teacher_head = model.vjepa_teacher_head
    if teacher_head is None:
        raise RuntimeError("source WM has no frozen V-JEPA teacher projection")
    teacher_head.requires_grad_(False).eval()
    teacher_dtype = next(teacher_head.parameters()).dtype
    transition_rng = random.Random(seed + 505)

    def process(
        episodes: list[Any], *, sample_one_transition: bool = False
    ) -> tuple[Tensor, dict[str, float]]:
        batch = collator._collate(episodes)
        obs = batch.obs_tokens.to(device)
        actions = batch.actions.to(device)
        lengths = batch.episode_lengths.to(device)
        env_ids = batch.env_ids.to(device)
        teacher_tokens = batch.semantic_teacher_tokens
        if teacher_tokens is None:
            raise RuntimeError("paired V-JEPA teacher tokens are missing")
        teacher_tokens = teacher_tokens.to(device)
        belief = model.get_initial_belief(len(episodes), device=device, dtype=obs.dtype)
        null = torch.full(
            (len(episodes),), config.env.null_action_id, device=device, dtype=torch.long
        )
        states = [state_cache[id(episode)] for episode in episodes]
        sampled_times: Tensor | None = None
        last_time = obs.shape[1] - 2
        if sample_one_transition:
            if any(int(episode.episode_length) <= 0 for episode in episodes):
                raise RuntimeError("spatial repair cannot sample an empty episode")
            sampled_times = torch.tensor(
                [
                    transition_rng.randrange(int(episode.episode_length))
                    for episode in episodes
                ],
                device=device,
                dtype=torch.long,
            )
            last_time = int(sampled_times.max().item())
        totals: dict[str, float] = {
            "actual_correct": 0.0, "actual_total": 0.0,
            "cf_correct": 0.0, "cf_total": 0.0,
            "changed_correct": 0.0, "changed_total": 0.0,
            "noop_correct": 0.0, "noop_total": 0.0,
            "geometry_cosine_sum": 0.0,
            "geometry_l1_sum": 0.0,
            "geometry_total": 0.0,
        }
        for action in range(4):
            totals[f"action_{action}_correct"] = 0.0
            totals[f"action_{action}_total"] = 0.0
        losses = []
        with torch.no_grad():
            current = model.posterior_step(belief, null, obs[:, 0], env_ids)
        for time in range(last_time + 1):
            valid = lengths > (time + 1)
            if sampled_times is not None:
                valid = valid & (sampled_times == time)
            with torch.no_grad():
                target = model.posterior_step(
                    current, actions[:, time], obs[:, time + 1], env_ids
                )
            if bool(valid.any()):
                active_rows = [row for row in range(len(episodes)) if bool(valid[row])]
                start = BeliefState(current.slots[valid].detach())
                active_env = env_ids[valid]
                actual_actions = actions[valid, time]
                predicted = model.prior_step(start, actual_actions, active_env)
                current_states = torch.tensor(
                    [states[row][time] for row in active_rows],
                    device=device,
                    dtype=torch.long,
                )
                next_states = torch.tensor(
                    [states[row][time + 1] for row in active_rows],
                    device=device,
                    dtype=torch.long,
                )
                actual_logits = position_probe(predicted.slots.flatten(1).float())
                actual_ce = F.cross_entropy(
                    actual_logits, next_states, reduction="none"
                )
                actual_changed = next_states != current_states
                actual_position = _balanced_mean(actual_ce, actual_changed)

                repeated_slots = start.slots.repeat_interleave(4, dim=0)
                cf_actions = torch.arange(4, device=device).repeat(start.batch_size)
                repeated_env = active_env.repeat_interleave(4)
                cf_predicted = model.prior_step(
                    BeliefState(repeated_slots), cf_actions, repeated_env
                )
                cf_next = torch.tensor(
                    [
                        _transition_table(tuple(episodes[row].metadata["map_rows"]))[
                            int(current_states[local])
                        ][action]
                        for local, row in enumerate(active_rows)
                        for action in range(4)
                    ],
                    device=device,
                    dtype=torch.long,
                )
                cf_current = current_states.repeat_interleave(4)
                cf_logits = position_probe(cf_predicted.slots.flatten(1).float())
                cf_ce = F.cross_entropy(cf_logits, cf_next, reduction="none")
                cf_changed = cf_next != cf_current
                cf_position = _balanced_mean(cf_ce, cf_changed)

                predicted_slots = predicted.slots.float()
                target_slots = target.slots[valid].detach().float()
                start_slots = start.slots.float()
                normalized_predicted = F.layer_norm(
                    predicted_slots, (predicted_slots.shape[-1],)
                )
                normalized_target = F.layer_norm(
                    target_slots, (target_slots.shape[-1],)
                )
                posterior_latent = F.smooth_l1_loss(
                    normalized_predicted, normalized_target
                )
                posterior_cosine = (
                    1.0
                    - F.cosine_similarity(
                        normalized_predicted, normalized_target, dim=-1
                    )
                ).mean()
                predicted_posterior_delta = normalized_predicted - F.layer_norm(
                    start_slots, (start_slots.shape[-1],)
                )
                target_posterior_delta = normalized_target - F.layer_norm(
                    start_slots, (start_slots.shape[-1],)
                )
                posterior_delta = F.smooth_l1_loss(
                    predicted_posterior_delta, target_posterior_delta
                )
                per_transition_cosine = F.cosine_similarity(
                    normalized_predicted.flatten(1),
                    normalized_target.flatten(1),
                    dim=-1,
                )
                per_transition_l1 = (
                    normalized_predicted - normalized_target
                ).abs().flatten(1).mean(-1)
                projected = teacher_head(predicted.slots.to(dtype=teacher_dtype)).float()
                projected_start = teacher_head(
                    start.slots.to(dtype=teacher_dtype)
                ).float()
                teacher_next = teacher_tokens[valid, time + 1].float()
                teacher_current = teacher_tokens[valid, time].float()
                vjepa_prior = (
                    1.0
                    - F.cosine_similarity(projected, teacher_next, dim=-1)
                ).mean()
                predicted_delta = projected - projected_start
                teacher_delta = teacher_next - teacher_current
                valid_delta = teacher_delta.square().mean(dim=-1).sqrt() >= 1e-4
                if bool(valid_delta.any()):
                    vjepa_delta = (
                        1.0
                        - F.cosine_similarity(
                            predicted_delta[valid_delta], teacher_delta[valid_delta], dim=-1
                        )
                    ).mean()
                else:
                    vjepa_delta = projected.sum() * 0.0
                losses.append(
                    weights.actual_position * actual_position
                    + weights.counterfactual_position * cf_position
                    + weights.posterior_latent * posterior_latent
                    + weights.posterior_cosine * posterior_cosine
                    + weights.posterior_delta * posterior_delta
                    + weights.vjepa_prior * vjepa_prior
                    + weights.vjepa_delta * vjepa_delta
                )

                actual_prediction = actual_logits.argmax(-1)
                cf_prediction = cf_logits.argmax(-1)
                totals["actual_correct"] += float((actual_prediction == next_states).sum())
                totals["actual_total"] += float(len(next_states))
                totals["geometry_cosine_sum"] += float(
                    per_transition_cosine.detach().sum()
                )
                totals["geometry_l1_sum"] += float(per_transition_l1.detach().sum())
                totals["geometry_total"] += float(len(next_states))
                totals["cf_correct"] += float((cf_prediction == cf_next).sum())
                totals["cf_total"] += float(len(cf_next))
                totals["changed_correct"] += float(
                    ((cf_prediction == cf_next) & cf_changed).sum()
                )
                totals["changed_total"] += float(cf_changed.sum())
                totals["noop_correct"] += float(
                    ((cf_prediction == cf_next) & ~cf_changed).sum()
                )
                totals["noop_total"] += float((~cf_changed).sum())
                for action in range(4):
                    mask = cf_actions == action
                    totals[f"action_{action}_correct"] += float(
                        ((cf_prediction == cf_next) & mask).sum()
                    )
                    totals[f"action_{action}_total"] += float(mask.sum())
            current = target
        if not losses:
            raise RuntimeError("sampled repair batch contains no real transitions")
        return torch.stack(losses).mean(), totals

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        sums: dict[str, float] = {}
        for begin in range(0, len(selection_eps), batch_size):
            _, values = process(selection_eps[begin : begin + batch_size])
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value
        result = {
            "prior_accuracy": sums["actual_correct"] / max(sums["actual_total"], 1.0),
            "counterfactual_accuracy": sums["cf_correct"] / max(sums["cf_total"], 1.0),
            "changed_accuracy": sums["changed_correct"] / max(sums["changed_total"], 1.0),
            "noop_accuracy": sums["noop_correct"] / max(sums["noop_total"], 1.0),
            "posterior_geometry_cosine": sums["geometry_cosine_sum"]
            / max(sums["geometry_total"], 1.0),
            "posterior_geometry_l1": sums["geometry_l1_sum"]
            / max(sums["geometry_total"], 1.0),
        }
        for action in range(4):
            result[f"action_{action}_accuracy"] = sums[f"action_{action}_correct"] / max(
                sums[f"action_{action}_total"], 1.0
            )
        result["minimum_action_accuracy"] = min(
            result[f"action_{action}_accuracy"] for action in range(4)
        )
        return result

    latest_adapter = output / "latest_adapter.pt"
    best_adapter = output / "best_adapter.pt"
    optimizer_step = 0
    best_key = (-float("inf"),) * 4
    best_metrics: dict[str, float] | None = None
    best_step = 0
    if resume:
        if not latest_adapter.is_file():
            raise FileNotFoundError(f"RESUME requested but missing {latest_adapter}")
        resumed = torch.load(latest_adapter, map_location="cpu", weights_only=False)
        if resumed.get("format") != "qwen_prior_lora_spatial_resume_v3":
            raise RuntimeError("resume adapter was produced by an incompatible objective")
        if Path(resumed["source_checkpoint"]).resolve() != checkpoint_path:
            raise RuntimeError("resume adapter belongs to a different source checkpoint")
        _restore_prior_state(model, resumed["prior_state"])
        optimizer.load_state_dict(resumed["optimizer"])
        optimizer_step = int(resumed["step"])
        if "transition_rng_state" in resumed:
            transition_rng.setstate(resumed["transition_rng_state"])
        if best_adapter.is_file():
            prior_best = torch.load(best_adapter, map_location="cpu", weights_only=False)
            if prior_best.get("format") != "qwen_prior_lora_spatial_best_v3":
                raise RuntimeError("best adapter was produced by an incompatible objective")
            best_key = tuple(float(value) for value in prior_best["selection_key"])
            best_metrics = dict(prior_best["metrics"])
            best_step = int(prior_best["step"])
        print(f"Resumed spatial prior repair from step {optimizer_step}", flush=True)
    elif latest_adapter.exists() or (output / "latest.pt").exists():
        raise FileExistsError(
            f"refusing to overwrite spatial prior repair artifacts in {output}"
        )

    initial_metrics = evaluate()
    initial_key = _selection_key(initial_metrics, gates)
    print(
        "[spatial-prior step 0] "
        + " | ".join(f"val/{k}={v:.4f}" for k, v in initial_metrics.items()),
        flush=True,
    )
    if best_metrics is None or initial_key > best_key:
        best_key = initial_key
        best_metrics = initial_metrics
        best_step = optimizer_step
        _atomic_torch_save(
            {
                "format": "qwen_prior_lora_spatial_best_v3",
                "source_checkpoint": str(checkpoint_path),
                "prior_state": _prior_state(model),
                "metrics": best_metrics,
                "selection_key": best_key,
                "step": best_step,
            },
            best_adapter,
        )

    order = list(range(len(train_eps)))
    random.Random(seed + 303).shuffle(order)
    cursor = (optimizer_step * batch_size) % len(order)
    for step in range(optimizer_step + 1, repair_steps + 1):
        if cursor + batch_size > len(order):
            random.Random(seed + 303 + step).shuffle(order)
            cursor = 0
        selected = [train_eps[index] for index in order[cursor : cursor + batch_size]]
        cursor += batch_size
        # Optimization samples one genuine transition from every episode.
        # Validation below still evaluates every valid transition and every
        # counterfactual action, so this speedup does not weaken checkpoint gates.
        loss, _ = process(selected, sample_one_transition=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == repair_steps:
            metrics = evaluate()
            key = _selection_key(metrics, gates)
            print(
                f"[spatial-prior step {step}] loss={float(loss.detach()):.5f} "
                f"grad_norm={float(grad_norm):.4f} weakest_gate={key[0]:.4f} | "
                + " | ".join(f"val/{k}={v:.4f}" for k, v in metrics.items()),
                flush=True,
            )
            _atomic_torch_save(
                {
                    "format": "qwen_prior_lora_spatial_resume_v3",
                    "source_checkpoint": str(checkpoint_path),
                    "prior_state": _prior_state(model),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "selection_key": key,
                    "step": step,
                    "transition_rng_state": transition_rng.getstate(),
                },
                latest_adapter,
            )
            if key > best_key:
                best_key = key
                best_metrics = metrics
                best_step = step
                _atomic_torch_save(
                    {
                        "format": "qwen_prior_lora_spatial_best_v3",
                        "source_checkpoint": str(checkpoint_path),
                        "prior_state": _prior_state(model),
                        "metrics": best_metrics,
                        "selection_key": best_key,
                        "step": best_step,
                    },
                    best_adapter,
                )
                print(f"Spatial best adapter updated at step {step}", flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": float(loss.detach()),
                        "train/grad_norm": float(grad_norm),
                        "selection/weakest_normalized_gate": key[0],
                        **{f"validation/{k}": v for k, v in metrics.items()},
                    },
                    step=step,
                )

    if best_metrics is None:
        raise RuntimeError("spatial prior repair produced no validated checkpoint")
    saved_best = torch.load(best_adapter, map_location="cpu", weights_only=False)
    _restore_prior_state(model, saved_best["prior_state"])
    checks = {
        "prior_accuracy": best_metrics["prior_accuracy"] >= gates.prior_accuracy,
        "counterfactual_accuracy": best_metrics["counterfactual_accuracy"]
        >= gates.counterfactual_accuracy,
        "changed_accuracy": best_metrics["changed_accuracy"] >= gates.changed_accuracy,
        "noop_accuracy": best_metrics["noop_accuracy"] >= gates.noop_accuracy,
        "minimum_action_accuracy": best_metrics["minimum_action_accuracy"]
        >= gates.minimum_action_accuracy,
    }
    passed = all(checks.values())
    metadata = {
        "format": "qwen_prior_lora_spatial_repair_v3",
        "source_checkpoint": str(checkpoint_path),
        "posterior_frozen": True,
        "teacher_projection_frozen": True,
        "reward_head_frozen": True,
        "prior_adapter": prior_name,
        "trainable_parameter_tensors": len(parameters),
        "best_step": best_step,
        "selection_key": list(best_key),
        "validation_metrics": best_metrics,
        "audit_linear_probe_metrics": probe_metrics,
        "checks": checks,
        "gates": asdict(gates),
        "loss_weights": asdict(weights),
        "split": {
            "train_indices_sha256": _indices_digest(train_indices),
            "validation_indices_sha256": _indices_digest(validation_indices),
            "selection_indices": selection_indices,
            **probe_split,
        },
    }
    assembled = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "scheduler", "config", "step"}
    }
    assembled.update(
        {
            "model": model.state_dict(),
            "config": config,
            "step": int(payload.get("step", 0)) + best_step,
            "spatial_prior_repair": metadata,
        }
    )
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    _atomic_torch_save(assembled, latest_path)
    for linked in (best_path if passed else None,):
        if linked is None:
            continue
        if linked.exists():
            linked.unlink()
        os.link(latest_path, linked)
    report = {
        "format": "qwen_prior_lora_spatial_repair_v3",
        "passed": passed,
        "checks": checks,
        "metrics": best_metrics,
        "audit_linear_probe_metrics": probe_metrics,
        "gates": asdict(gates),
        "loss_weights": asdict(weights),
        "best_step": best_step,
        "source_checkpoint": str(checkpoint_path),
        "latest_checkpoint": str(latest_path.resolve()),
        "release_checkpoint": str(best_path.resolve()) if passed else None,
    }
    _atomic_json(report, output / "repair_report.json")
    print(
        "Spatial prior repair Gate: "
        + ("PASS" if passed else "FAIL: " + ", ".join(k for k, v in checks.items() if not v)),
        flush=True,
    )
    if not passed:
        raise RuntimeError(f"spatial prior repair gate failed: {output/'repair_report.json'}")
    return report
