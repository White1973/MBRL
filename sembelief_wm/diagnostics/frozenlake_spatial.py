"""Read-only spatial audit for a frozen FrozenLake world model."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from ..model.world_model import WorldModel


@dataclass
class SpatialAuditResult:
    posterior_accuracy: float
    prior_accuracy: float
    counterfactual_accuracy: float
    majority_baseline: float
    train_states: int
    validation_states: int
    prior_transitions: int
    counterfactual_transitions: int


def _next_state(state: int, action: int, rows: list[str]) -> int:
    size = len(rows)
    row, col = divmod(state, size)
    drdc = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    dr, dc = drdc[action]
    nr = min(max(row + dr, 0), size - 1)
    nc = min(max(col + dc, 0), size - 1)
    return nr * size + nc


def _state_sequence(episode) -> list[int]:
    rows = list(episode.metadata["map_rows"])
    states = [int(episode.metadata["start_state"])]
    state = states[0]
    for action in episode.actions[: episode.episode_length].tolist():
        state = _next_state(state, int(action), rows)
        states.append(state)
    return states


@torch.no_grad()
def _extract(
    world_model: WorldModel,
    episodes: list,
    config,
    device: torch.device,
    *,
    batch_size: int,
    counterfactual: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    posterior_x, posterior_y = [], []
    prior_x, prior_y = [], []
    cf_x, cf_y = [], []
    source = OfflineDataSource(TokenizedEpisodeDataset(episodes), config)
    for offset in range(0, len(episodes), batch_size):
        selected = episodes[offset : offset + batch_size]
        batch = source._collate(selected)
        obs = batch.obs_tokens.to(device)
        actions = batch.actions.to(device)
        lengths = batch.episode_lengths.to(device)
        env_ids = batch.env_ids.to(device)
        belief = world_model.get_initial_belief(
            len(selected), device=device, dtype=obs.dtype
        )
        state_lists = [_state_sequence(ep) for ep in selected]
        null_actions = torch.full(
            (len(selected),), config.env.null_action_id,
            device=device, dtype=torch.long,
        )
        for step in range(obs.shape[1]):
            valid = lengths > step
            previous = belief
            previous_actions = null_actions if step == 0 else actions[:, step - 1]
            if step > 0:
                predicted = world_model.prior_step(previous, previous_actions, env_ids)
                prior_x.append(predicted.slots[valid].flatten(1).float().cpu())
                prior_y.append(torch.tensor([
                    state_lists[row][step]
                    for row in range(len(selected)) if bool(valid[row])
                ], dtype=torch.long))
            belief = world_model.posterior_step(
                previous, previous_actions, obs[:, step], env_ids
            )
            posterior_x.append(belief.slots[valid].flatten(1).float().cpu())
            posterior_y.append(torch.tensor([
                state_lists[row][step]
                for row in range(len(selected)) if bool(valid[row])
            ], dtype=torch.long))
            if counterfactual:
                can_step = lengths > (step + 1)
                if bool(can_step.any()):
                    active = belief.slots[can_step]
                    active_env = env_ids[can_step]
                    active_rows = [
                        row for row in range(len(selected)) if bool(can_step[row])
                    ]
                    num_actions = config.env.num_actions
                    repeated_slots = active.repeat_interleave(num_actions, dim=0)
                    repeated_env = active_env.repeat_interleave(num_actions)
                    action_tensor = torch.arange(
                        num_actions, device=device, dtype=torch.long
                    ).repeat(active.shape[0])
                    imagined = world_model.prior_step(
                        type(belief)(slots=repeated_slots),
                        action_tensor,
                        repeated_env,
                    )
                    cf_x.append(imagined.slots.flatten(1).float().cpu())
                    cf_y.append(torch.tensor([
                        _next_state(
                            state_lists[row][step], action,
                            list(selected[row].metadata["map_rows"]),
                        )
                        for row in active_rows
                        for action in range(num_actions)
                    ], dtype=torch.long))
    empty_x = torch.empty(0, config.belief.num_slots * config.hidden_dim)
    empty_y = torch.empty(0, dtype=torch.long)
    cat = lambda values, empty: torch.cat(values) if values else empty
    return (
        cat(posterior_x, empty_x), cat(posterior_y, empty_y),
        cat(prior_x, empty_x), cat(prior_y, empty_y),
        cat(cf_x, empty_x), cat(cf_y, empty_y),
    )


def _accuracy(probe: nn.Module, x: Tensor, y: Tensor, device: torch.device) -> float:
    correct = 0
    for start in range(0, len(y), 256):
        xb = x[start : start + 256].to(device)
        pred = probe(xb).argmax(-1).cpu()
        correct += int((pred == y[start : start + 256]).sum())
    return correct / max(len(y), 1)


def run_frozenlake_spatial_audit(
    *, world_model: WorldModel, dataset: TokenizedEpisodeDataset, config,
    device: torch.device, seed: int, train_episodes: int = 400,
    validation_episodes: int = 200, probe_steps: int = 400,
    train_indices: list[int] | None = None,
    validation_indices: list[int] | None = None,
    validation_exclude_indices: list[int] | None = None,
) -> SpatialAuditResult:
    world_model.eval()
    rng = random.Random(seed)
    if train_indices is None or validation_indices is None:
        indices = list(range(len(dataset.episodes)))
        rng.shuffle(indices)
        selected_train = indices[:train_episodes]
        selected_validation = indices[
            train_episodes : train_episodes + validation_episodes
        ]
        split_protocol = "legacy_random_split"
    else:
        if set(train_indices) & set(validation_indices):
            raise RuntimeError("FrozenLake spatial audit train/validation overlap")
        excluded = set(validation_exclude_indices or ())
        if excluded - set(validation_indices):
            raise RuntimeError(
                "FrozenLake spatial audit exclusions are outside the validation split"
            )
        selected_train = list(train_indices)
        selected_validation = [
            index for index in validation_indices if index not in excluded
        ]
        rng.shuffle(selected_train)
        rng.shuffle(selected_validation)
        selected_train = selected_train[:train_episodes]
        selected_validation = selected_validation[:validation_episodes]
        split_protocol = (
            "checkpoint_fixed_stage1_split"
            if not excluded
            else "checkpoint_fixed_stage1_split_excluding_checkpoint_selection_panel"
        )
    if len(selected_train) < train_episodes or len(selected_validation) < validation_episodes:
        raise ValueError(
            "FrozenLake spatial audit split is smaller than requested: "
            f"train={len(selected_train)}/{train_episodes}, "
            f"validation={len(selected_validation)}/{validation_episodes}"
        )
    print(
        f"FrozenLake spatial audit split: {split_protocol}; "
        f"train={len(selected_train)} validation={len(selected_validation)} overlap=0",
        flush=True,
    )
    train_eps = [dataset.episodes[i] for i in selected_train]
    val_eps = [dataset.episodes[i] for i in selected_validation]
    train_x, train_y, _, _, _, _ = _extract(
        world_model, train_eps, config, device, batch_size=16, counterfactual=False
    )
    val_x, val_y, prior_x, prior_y, cf_x, cf_y = _extract(
        world_model, val_eps, config, device, batch_size=16, counterfactual=True
    )
    probe = nn.Linear(train_x.shape[1], 16).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    probe.train()
    for _ in range(probe_steps):
        choice = torch.randint(0, len(train_y), (min(256, len(train_y)),), generator=generator)
        logits = probe(train_x[choice].to(device))
        loss = nn.functional.cross_entropy(logits, train_y[choice].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    counts = torch.bincount(val_y, minlength=16)
    result = SpatialAuditResult(
        posterior_accuracy=_accuracy(probe, val_x, val_y, device),
        prior_accuracy=_accuracy(probe, prior_x, prior_y, device),
        counterfactual_accuracy=_accuracy(probe, cf_x, cf_y, device),
        majority_baseline=float(counts.max().item() / max(len(val_y), 1)),
        train_states=len(train_y), validation_states=len(val_y),
        prior_transitions=len(prior_y), counterfactual_transitions=len(cf_y),
    )
    print("=== FrozenLake frozen-WM spatial audit ===", flush=True)
    for key, value in vars(result).items():
        print(f"  {key}: {value}", flush=True)
    return result
