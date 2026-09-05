"""Held-out Stage-1 spatial and action-dynamics audit for Sokoban.

The WM is deliberately not trained or changed in this module.  A compact
slot-wise decoder is fitted on *training* episodes only, then applied without
adaptation to the checkpoint's held-out validation episodes.  Ground truth is
read from the original Sokoban rollout metadata, never from V-JEPA features.

This is consequently an acceptance test for a Qwen-native World Model, rather
than another latent-alignment loss disguised as an evaluation metric.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.storage import load_tokenized_episode, read_manifest
from ..types import BeliefState


GRID_SIZE = 6
NUM_CELLS = GRID_SIZE * GRID_SIZE
_DIRECTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # model action 0 == Sokoban push up
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class AuditThresholds:
    """Deliberately moderate Stage-1 release criteria.

    Posterior grounding needs to be reliable before a value/reward head is
    fitted.  Prior and counterfactual requirements are lower because they test
    action-conditioned prediction, not direct observation recognition.  The
    changed-only checks prevent a copy/no-op model from passing merely because
    Sokoban has many blocked moves.
    """

    posterior_player_accuracy: float = 0.75
    posterior_box_accuracy: float = 0.75
    posterior_target_accuracy: float = 0.75
    posterior_wall_macro_f1: float = 0.75
    posterior_baseline_margin: float = 0.25
    logged_changed_joint_accuracy: float = 0.35
    counterfactual_changed_joint_accuracy: float = 0.30
    dynamics_copy_margin: float = 0.20
    minimum_validation_states: int = 500
    minimum_counterfactual_changed_states: int = 200


@dataclass(frozen=True)
class _StateLabel:
    player: int
    box: int
    target: int
    walls: Tensor  # (36,), bool on CPU


@dataclass(frozen=True)
class _AuditEpisode:
    tokens: Tensor
    actions: Tensor
    labels: tuple[_StateLabel, ...]
    source_path: str

    @property
    def length(self) -> int:
        return len(self.labels)


class SlotwiseSokobanDecoder(nn.Module):
    """Decode objects from their corresponding ordered spatial slot.

    This deliberately has no global pooling and no learned cross-slot query.
    The input slot at a grid location must carry information about that same
    location for the decoder to succeed.  It is therefore a stronger spatial
    test than a mean-pooled linear probe while remaining a read-only audit of
    the World Model.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.player = nn.Linear(hidden_dim, 1)
        self.box = nn.Linear(hidden_dim, 1)
        self.target = nn.Linear(hidden_dim, 1)
        self.wall = nn.Linear(hidden_dim, 1)

    def forward(self, slots: Tensor) -> dict[str, Tensor]:
        if slots.ndim != 3 or slots.shape[1] != NUM_CELLS:
            raise ValueError(
                "SlotwiseSokobanDecoder requires (N, 36, D) beliefs, got "
                f"{tuple(slots.shape)}."
            )
        features = self.norm(slots.float())
        return {
            "player": self.player(features).squeeze(-1),
            "box": self.box(features).squeeze(-1),
            "target": self.target(features).squeeze(-1),
            "wall": self.wall(features).squeeze(-1),
        }

    @staticmethod
    def loss(logits: dict[str, Tensor], labels: dict[str, Tensor]) -> Tensor:
        classification = sum(
            F.cross_entropy(logits[name], labels[name])
            for name in ("player", "box", "target")
        ) / 3.0
        wall_targets = labels["walls"].float()
        # Wall cells are common in a 6x6 Sokoban board.  Balance the two
        # classes so an all-wall decoder cannot look good from raw accuracy.
        positive_rate = wall_targets.mean().clamp(0.05, 0.95)
        positive_weight = 0.5 / positive_rate
        negative_weight = 0.5 / (1.0 - positive_rate)
        wall_loss = F.binary_cross_entropy_with_logits(
            logits["wall"], wall_targets, reduction="none"
        )
        wall_weight = torch.where(
            wall_targets.bool(), positive_weight, negative_weight
        )
        return classification + (wall_loss * wall_weight).mean()


def _cell(position: Sequence[int]) -> int:
    row, col = int(position[0]), int(position[1])
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        raise ValueError(f"Sokoban position outside 6x6 board: {position!r}")
    return row * GRID_SIZE + col


def _state_from_raw(label: dict[str, Any], metadata: dict[str, Any]) -> _StateLabel:
    player = label.get("player_pos")
    boxes = label.get("box_positions") or []
    targets = label.get("target_positions") or metadata.get("target_positions") or []
    walls = metadata.get("wall_positions") or []
    if player is None or len(boxes) != 1 or len(targets) != 1:
        raise ValueError(
            "This Stage-1 audit currently requires one labelled player, one "
            f"box, and one target; got player={player!r}, boxes={boxes!r}, "
            f"targets={targets!r}."
        )
    wall_mask = torch.zeros(NUM_CELLS, dtype=torch.bool)
    for position in walls:
        wall_mask[_cell(position)] = True
    return _StateLabel(
        player=_cell(player),
        box=_cell(boxes[0]),
        target=_cell(targets[0]),
        walls=wall_mask,
    )


def _load_audit_episode(tokenized_path: Path) -> _AuditEpisode:
    episode = load_tokenized_episode(tokenized_path)
    raw_location = episode.metadata.get("paired_raw_episode")
    if not raw_location:
        raise ValueError(
            f"{tokenized_path}: paired_raw_episode provenance is required for "
            "the Stage-1 semantic audit."
        )
    raw_path = Path(str(raw_location))
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{tokenized_path}: original rollout declared by paired replay is "
            f"missing: {raw_path}"
        )
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    actions = torch.as_tensor(raw.get("model_actions", ()), dtype=torch.long)
    expected_length = int(episode.episode_length)
    if actions.numel() != expected_length:
        raise ValueError(
            f"{tokenized_path}: raw action count {actions.numel()} does not "
            f"match paired episode length {expected_length}."
        )
    if not torch.equal(episode.actions[:expected_length].cpu(), actions):
        raise ValueError(
            f"{tokenized_path}: paired actions differ from its declared raw "
            "rollout; refusing to audit mismatched trajectories."
        )
    if len(raw.get("observations", ())) != expected_length + 1:
        raise ValueError(
            f"{raw_path}: expected T+1 raw observations for T={expected_length}."
        )
    metadata = dict(raw.get("metadata", {}))
    initial = metadata.get("initial_state_labels")
    infos = list(raw.get("infos", ()))
    if initial is None or len(infos) != expected_length:
        raise ValueError(f"{raw_path}: missing aligned Sokoban state labels.")
    labels = [_state_from_raw(initial, metadata)]
    for info in infos:
        state_label = dict(info).get("state_labels")
        if state_label is None:
            raise ValueError(f"{raw_path}: a transition has no state_labels.")
        labels.append(_state_from_raw(state_label, metadata))
    if len(labels) != expected_length + 1:
        raise AssertionError("Sokoban label alignment invariant was broken.")
    return _AuditEpisode(
        tokens=episode.obs_tokens[: expected_length + 1].cpu(),
        actions=actions.cpu(),
        labels=tuple(labels),
        source_path=str(raw_path),
    )


def _selected_entries(
    *,
    paired_data_dir: Path,
    indices: Sequence[int],
    maximum: int,
) -> list[Path]:
    entries = read_manifest(paired_data_dir / "manifest.jsonl")
    if len(set(indices)) != len(indices):
        raise ValueError("Checkpoint Stage-1 split contains duplicate indices.")
    if not indices or min(indices) < 0 or max(indices) >= len(entries):
        raise ValueError(
            "Checkpoint Stage-1 split is incompatible with the paired replay "
            f"manifest ({len(entries)} episodes)."
        )
    chosen = list(indices if maximum <= 0 else indices[:maximum])
    return [paired_data_dir / str(entries[index]["path"]) for index in chosen]


def _iter_episode_batches(
    paths: Sequence[Path], batch_size: int
) -> Iterable[list[_AuditEpisode]]:
    for start in range(0, len(paths), batch_size):
        yield [_load_audit_episode(path) for path in paths[start : start + batch_size]]


def _collate(episodes: Sequence[_AuditEpisode], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    if not episodes:
        raise ValueError("Cannot collate an empty audit batch.")
    seq_lengths = torch.tensor([episode.length for episode in episodes], dtype=torch.long)
    max_length = int(seq_lengths.max().item())
    sample = episodes[0].tokens
    if sample.ndim != 3 or sample.shape[1] != NUM_CELLS:
        raise ValueError(
            "Sokoban Stage-1 audit requires Qwen tokens shaped (T, 36, D), "
            f"got {tuple(sample.shape)}."
        )
    tokens = torch.zeros(
        len(episodes), max_length, sample.shape[1], sample.shape[2], dtype=sample.dtype
    )
    actions = torch.full(
        (len(episodes), max_length), 4, dtype=torch.long
    )
    for row, episode in enumerate(episodes):
        if tuple(episode.tokens.shape[1:]) != tuple(sample.shape[1:]):
            raise ValueError("All audit episodes must share Qwen token geometry.")
        tokens[row, : episode.length] = episode.tokens
        actions[row, : episode.length - 1] = episode.actions
    return tokens.to(device), actions.to(device), seq_lengths.to(device)


def _labels_at(episodes: Sequence[_AuditEpisode], time_index: int, valid: Tensor, device: torch.device) -> dict[str, Tensor]:
    selected = [episode.labels[time_index] for row, episode in enumerate(episodes) if bool(valid[row])]
    if not selected:
        return {
            "player": torch.empty(0, dtype=torch.long, device=device),
            "box": torch.empty(0, dtype=torch.long, device=device),
            "target": torch.empty(0, dtype=torch.long, device=device),
            "walls": torch.empty(0, NUM_CELLS, dtype=torch.bool, device=device),
        }
    return {
        "player": torch.tensor([state.player for state in selected], dtype=torch.long, device=device),
        "box": torch.tensor([state.box for state in selected], dtype=torch.long, device=device),
        "target": torch.tensor([state.target for state in selected], dtype=torch.long, device=device),
        "walls": torch.stack([state.walls for state in selected]).to(device),
    }


def _successor(state: _StateLabel, action: int) -> _StateLabel:
    if action not in _DIRECTIONS:
        raise ValueError(f"Unexpected model action {action}; expected 0..3.")
    row, col = divmod(state.player, GRID_SIZE)
    dr, dc = _DIRECTIONS[action]
    next_row, next_col = row + dr, col + dc
    if not (0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE):
        return state
    candidate = next_row * GRID_SIZE + next_col
    if bool(state.walls[candidate]):
        return state
    if candidate != state.box:
        return _StateLabel(candidate, state.box, state.target, state.walls)
    box_row, box_col = divmod(state.box, GRID_SIZE)
    box_next_row, box_next_col = box_row + dr, box_col + dc
    if not (0 <= box_next_row < GRID_SIZE and 0 <= box_next_col < GRID_SIZE):
        return state
    box_next = box_next_row * GRID_SIZE + box_next_col
    if bool(state.walls[box_next]):
        return state
    return _StateLabel(candidate, box_next, state.target, state.walls)


def _metric_counts() -> dict[str, float]:
    return {
        "states": 0.0,
        "player_correct": 0.0,
        "box_correct": 0.0,
        "target_correct": 0.0,
        "wall_tp": 0.0,
        "wall_tn": 0.0,
        "wall_fp": 0.0,
        "wall_fn": 0.0,
        "joint_correct": 0.0,
        "copy_player_correct": 0.0,
        "copy_box_correct": 0.0,
        "copy_joint_correct": 0.0,
    }


def _accumulate(
    counts: dict[str, float],
    logits: dict[str, Tensor],
    labels: dict[str, Tensor],
    *,
    copy_labels: dict[str, Tensor] | None = None,
) -> None:
    if labels["player"].numel() == 0:
        return
    player = logits["player"].argmax(dim=-1)
    box = logits["box"].argmax(dim=-1)
    target = logits["target"].argmax(dim=-1)
    wall = logits["wall"].sigmoid() >= 0.5
    target_wall = labels["walls"].bool()
    counts["states"] += float(player.numel())
    counts["player_correct"] += float((player == labels["player"]).sum().item())
    counts["box_correct"] += float((box == labels["box"]).sum().item())
    counts["target_correct"] += float((target == labels["target"]).sum().item())
    counts["joint_correct"] += float(
        ((player == labels["player"]) & (box == labels["box"])).sum().item()
    )
    counts["wall_tp"] += float((wall & target_wall).sum().item())
    counts["wall_tn"] += float(((~wall) & (~target_wall)).sum().item())
    counts["wall_fp"] += float((wall & (~target_wall)).sum().item())
    counts["wall_fn"] += float(((~wall) & target_wall).sum().item())
    if copy_labels is not None:
        copy_joint = (
            (copy_labels["player"] == labels["player"])
            & (copy_labels["box"] == labels["box"])
        )
        counts["copy_player_correct"] += float(
            (copy_labels["player"] == labels["player"]).sum().item()
        )
        counts["copy_box_correct"] += float(
            (copy_labels["box"] == labels["box"]).sum().item()
        )
        counts["copy_joint_correct"] += float(copy_joint.sum().item())


def _finalize(counts: dict[str, float]) -> dict[str, float]:
    n = max(counts["states"], 1.0)
    positive_f1 = 2.0 * counts["wall_tp"] / max(
        2.0 * counts["wall_tp"] + counts["wall_fp"] + counts["wall_fn"], 1.0
    )
    negative_f1 = 2.0 * counts["wall_tn"] / max(
        2.0 * counts["wall_tn"] + counts["wall_fp"] + counts["wall_fn"], 1.0
    )
    return {
        "states": counts["states"],
        "player_accuracy": counts["player_correct"] / n,
        "box_accuracy": counts["box_correct"] / n,
        "target_accuracy": counts["target_correct"] / n,
        "joint_accuracy": counts["joint_correct"] / n,
        "wall_macro_f1": 0.5 * (positive_f1 + negative_f1),
        "copy_player_accuracy": counts["copy_player_correct"] / n,
        "copy_box_accuracy": counts["copy_box_correct"] / n,
        "copy_joint_accuracy": counts["copy_joint_correct"] / n,
    }


def _majority_baselines(labels: dict[str, Counter[int]], total: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ("player", "box", "target"):
        counts = labels[name]
        result[name] = max(counts.values(), default=0) / max(total, 1)
    return result


def _one_epoch_train(
    *,
    world_model: Any,
    decoder: SlotwiseSokobanDecoder,
    optimizer: torch.optim.Optimizer,
    paths: Sequence[Path],
    batch_size: int,
    device: torch.device,
    epoch: int,
) -> tuple[dict[str, float], dict[str, Counter[int]], int]:
    losses: list[float] = []
    majority: dict[str, Counter[int]] = {
        "player": Counter(), "box": Counter(), "target": Counter(),
    }
    total_states = 0
    for batch_index, episodes in enumerate(_iter_episode_batches(paths, batch_size), start=1):
        tokens, actions, lengths = _collate(episodes, device)
        belief = world_model.get_initial_belief(
            len(episodes), device=device, dtype=tokens.dtype
        )
        previous_actions = torch.full(
            (len(episodes),), world_model.config.env.null_action_id,
            device=device, dtype=torch.long,
        )
        slots: list[Tensor] = []
        labels_by_time: list[dict[str, Tensor]] = []
        for time_index in range(tokens.shape[1]):
            posterior = world_model.posterior_step(
                belief, previous_actions, tokens[:, time_index]
            )
            valid = lengths > time_index
            if bool(valid.any()):
                slots.append(posterior.slots[valid].detach())
                labels = _labels_at(episodes, time_index, valid, device)
                labels_by_time.append(labels)
                total_states += int(labels["player"].numel())
                majority["player"].update(labels["player"].detach().cpu().tolist())
                majority["box"].update(labels["box"].detach().cpu().tolist())
                majority["target"].update(labels["target"].detach().cpu().tolist())
            belief = posterior
            if time_index < actions.shape[1]:
                previous_actions = actions[:, time_index]
        if not slots:
            continue
        flat_slots = torch.cat(slots, dim=0)
        flat_labels = {
            name: torch.cat([label[name] for label in labels_by_time], dim=0)
            for name in ("player", "box", "target", "walls")
        }
        optimizer.zero_grad(set_to_none=True)
        loss = SlotwiseSokobanDecoder.loss(decoder(flat_slots), flat_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if batch_index % 100 == 0 or batch_index == (len(paths) + batch_size - 1) // batch_size:
            print(
                f"[stage1-spatial-probe epoch {epoch} batch {batch_index}] "
                f"loss={losses[-1]:.5f} train_states={total_states}",
                flush=True,
            )
    return {
        "mean_loss": sum(losses) / max(len(losses), 1),
        "updates": float(len(losses)),
        "states": float(total_states),
    }, majority, total_states


@torch.no_grad()
def _evaluate(
    *,
    world_model: Any,
    decoder: SlotwiseSokobanDecoder,
    paths: Sequence[Path],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, Counter[int]], int]:
    posterior_counts = _metric_counts()
    logged_all_counts = _metric_counts()
    logged_changed_counts = _metric_counts()
    counterfactual_all_counts = _metric_counts()
    counterfactual_changed_counts = _metric_counts()
    majority: dict[str, Counter[int]] = {
        "player": Counter(), "box": Counter(), "target": Counter(),
    }
    total_posterior_states = 0
    decoder.eval()
    for batch_index, episodes in enumerate(_iter_episode_batches(paths, batch_size), start=1):
        tokens, actions, lengths = _collate(episodes, device)
        belief = world_model.get_initial_belief(
            len(episodes), device=device, dtype=tokens.dtype
        )
        previous_actions = torch.full(
            (len(episodes),), world_model.config.env.null_action_id,
            device=device, dtype=torch.long,
        )
        for time_index in range(tokens.shape[1]):
            prior = world_model.prior_step(belief, previous_actions)
            posterior = world_model.posterior_step(belief, previous_actions, tokens[:, time_index])
            valid = lengths > time_index
            if bool(valid.any()):
                labels = _labels_at(episodes, time_index, valid, device)
                post_logits = decoder(posterior.slots[valid])
                _accumulate(posterior_counts, post_logits, labels)
                total_posterior_states += int(labels["player"].numel())
                majority["player"].update(labels["player"].detach().cpu().tolist())
                majority["box"].update(labels["box"].detach().cpu().tolist())
                majority["target"].update(labels["target"].detach().cpu().tolist())

            # Prior at time t predicts state_t from state_{t-1} and action_{t-1}.
            if time_index > 0:
                transition_valid = valid
                if bool(transition_valid.any()):
                    next_labels = _labels_at(episodes, time_index, transition_valid, device)
                    previous = _labels_at(episodes, time_index - 1, transition_valid, device)
                    prior_logits = decoder(prior.slots[transition_valid])
                    _accumulate(logged_all_counts, prior_logits, next_labels, copy_labels=previous)
                    changed = (
                        (previous["player"] != next_labels["player"])
                        | (previous["box"] != next_labels["box"])
                    )
                    if bool(changed.any()):
                        _accumulate(
                            logged_changed_counts,
                            {name: value[changed] for name, value in prior_logits.items()},
                            {name: value[changed] for name, value in next_labels.items()},
                            copy_labels={name: value[changed] for name, value in previous.items()},
                        )

            # Four-action counterfactual evaluation starts from a grounded
            # posterior state. Only nonterminal source states have a valid
            # environment transition. Truth is computed from raw Sokoban
            # layout labels, not from another learned model.
            source_valid = lengths > (time_index + 1)
            if bool(source_valid.any()):
                source_rows = source_valid.nonzero(as_tuple=False).flatten().tolist()
                source_states = [episodes[row].labels[time_index] for row in source_rows]
                expanded = BeliefState(
                    slots=posterior.slots[source_valid]
                    .unsqueeze(1)
                    .expand(-1, 4, -1, -1)
                    .reshape(-1, posterior.slots.shape[1], posterior.slots.shape[2])
                )
                all_actions = torch.arange(4, device=device, dtype=torch.long).repeat(len(source_states))
                predicted = world_model.prior_step(expanded, all_actions)
                cf_logits = decoder(predicted.slots)
                target_states = [
                    _successor(state, action)
                    for state in source_states
                    for action in range(4)
                ]
                copied_states = [state for state in source_states for _ in range(4)]
                target_labels = {
                    "player": torch.tensor([state.player for state in target_states], device=device),
                    "box": torch.tensor([state.box for state in target_states], device=device),
                    "target": torch.tensor([state.target for state in target_states], device=device),
                    "walls": torch.stack([state.walls for state in target_states]).to(device),
                }
                copied_labels = {
                    "player": torch.tensor([state.player for state in copied_states], device=device),
                    "box": torch.tensor([state.box for state in copied_states], device=device),
                    "target": torch.tensor([state.target for state in copied_states], device=device),
                    "walls": torch.stack([state.walls for state in copied_states]).to(device),
                }
                _accumulate(counterfactual_all_counts, cf_logits, target_labels, copy_labels=copied_labels)
                changed = (
                    (target_labels["player"] != copied_labels["player"])
                    | (target_labels["box"] != copied_labels["box"])
                )
                if bool(changed.any()):
                    _accumulate(
                        counterfactual_changed_counts,
                        {name: value[changed] for name, value in cf_logits.items()},
                        {name: value[changed] for name, value in target_labels.items()},
                        copy_labels={name: value[changed] for name, value in copied_labels.items()},
                    )
            belief = posterior
            if time_index < actions.shape[1]:
                previous_actions = actions[:, time_index]
        if batch_index % 100 == 0 or batch_index == (len(paths) + batch_size - 1) // batch_size:
            print(
                f"[stage1-semantic-audit batch {batch_index}] "
                f"validation_states={int(posterior_counts['states'])} "
                f"counterfactual_changed={int(counterfactual_changed_counts['states'])}",
                flush=True,
            )
    return {
        "posterior": _finalize(posterior_counts),
        "logged_prior_all": _finalize(logged_all_counts),
        "logged_prior_changed_only": _finalize(logged_changed_counts),
        "counterfactual_all_actions": _finalize(counterfactual_all_counts),
        "counterfactual_changed_only": _finalize(counterfactual_changed_counts),
    }, majority, total_posterior_states


def _gate(
    *,
    metrics: dict[str, dict[str, float]],
    train_majority: dict[str, float],
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    failures: list[str] = []
    posterior = metrics["posterior"]
    logged = metrics["logged_prior_changed_only"]
    counterfactual = metrics["counterfactual_changed_only"]
    if posterior["states"] < thresholds.minimum_validation_states:
        failures.append(
            f"posterior states={posterior['states']:.0f} < "
            f"{thresholds.minimum_validation_states}"
        )
    if counterfactual["states"] < thresholds.minimum_counterfactual_changed_states:
        failures.append(
            f"changed counterfactual states={counterfactual['states']:.0f} < "
            f"{thresholds.minimum_counterfactual_changed_states}"
        )
    absolute = {
        "posterior.player_accuracy": thresholds.posterior_player_accuracy,
        "posterior.box_accuracy": thresholds.posterior_box_accuracy,
        "posterior.target_accuracy": thresholds.posterior_target_accuracy,
        "posterior.wall_macro_f1": thresholds.posterior_wall_macro_f1,
        "logged_prior_changed_only.joint_accuracy": thresholds.logged_changed_joint_accuracy,
        "counterfactual_changed_only.joint_accuracy": thresholds.counterfactual_changed_joint_accuracy,
    }
    groups = {
        "posterior": posterior,
        "logged_prior_changed_only": logged,
        "counterfactual_changed_only": counterfactual,
    }
    for key, threshold in absolute.items():
        group, metric = key.split(".", maxsplit=1)
        value = groups[group][metric]
        if value < threshold:
            failures.append(f"{key}={value:.4f} < {threshold:.4f}")
    for label_name, metric_name in (("player", "player_accuracy"), ("box", "box_accuracy"), ("target", "target_accuracy")):
        required = train_majority[label_name] + thresholds.posterior_baseline_margin
        value = posterior[metric_name]
        if value < required:
            failures.append(
                f"posterior.{metric_name}={value:.4f} < train-majority "
                f"{train_majority[label_name]:.4f} + "
                f"{thresholds.posterior_baseline_margin:.4f}"
            )
    for name, result in (("logged_prior_changed_only", logged), ("counterfactual_changed_only", counterfactual)):
        improvement = result["joint_accuracy"] - result["copy_joint_accuracy"]
        if improvement < thresholds.dynamics_copy_margin:
            failures.append(
                f"{name}.joint_vs_copy={improvement:.4f} < "
                f"{thresholds.dynamics_copy_margin:.4f}"
            )
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            field: getattr(thresholds, field)
            for field in thresholds.__dataclass_fields__
        },
        "train_majority_baseline": train_majority,
    }


def run_sokoban_stage1_semantic_audit(
    *,
    world_model: Any,
    paired_data_dir: str | Path,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    device: torch.device | str,
    batch_size: int = 4,
    probe_epochs: int = 1,
    probe_lr: float = 1e-3,
    max_train_episodes: int = 0,
    max_validation_episodes: int = 0,
    thresholds: AuditThresholds = AuditThresholds(),
) -> dict[str, Any]:
    """Run the frozen Stage-1 spatial / action-dynamics acceptance test."""
    if batch_size <= 0 or probe_epochs <= 0 or probe_lr <= 0.0:
        raise ValueError("batch_size, probe_epochs, and probe_lr must be positive.")
    paired_root = Path(paired_data_dir)
    if not (paired_root / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"Paired replay manifest missing: {paired_root}")
    train_set = set(train_indices)
    validation_set = set(validation_indices)
    if train_set.intersection(validation_set):
        raise ValueError("Stage-1 train/validation episode split overlaps.")
    train_paths = _selected_entries(
        paired_data_dir=paired_root, indices=train_indices, maximum=max_train_episodes
    )
    validation_paths = _selected_entries(
        paired_data_dir=paired_root, indices=validation_indices, maximum=max_validation_episodes
    )
    if not train_paths or not validation_paths:
        raise ValueError("Stage-1 semantic audit requires non-empty train and validation splits.")
    target_device = torch.device(device)
    decoder = SlotwiseSokobanDecoder(world_model.config.hidden_dim).to(target_device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=probe_lr, weight_decay=1e-4)

    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    training_history: list[dict[str, float]] = []
    train_counts: dict[str, Counter[int]] = {
        "player": Counter(), "box": Counter(), "target": Counter(),
    }
    total_train_states = 0
    for epoch in range(1, probe_epochs + 1):
        decoder.train()
        epoch_result, epoch_counts, epoch_states = _one_epoch_train(
            world_model=world_model,
            decoder=decoder,
            optimizer=optimizer,
            paths=train_paths,
            batch_size=batch_size,
            device=target_device,
            epoch=epoch,
        )
        training_history.append(epoch_result)
        for name, counts in epoch_counts.items():
            train_counts[name].update(counts)
        total_train_states += epoch_states
    # The majority baseline must be calculated from probe-training labels only.
    # Repeated epochs duplicate each label equally, so it remains a valid ratio.
    train_majority = _majority_baselines(train_counts, total_train_states)
    metrics, validation_counts, validation_states = _evaluate(
        world_model=world_model,
        decoder=decoder,
        paths=validation_paths,
        batch_size=batch_size,
        device=target_device,
    )
    validation_majority = _majority_baselines(validation_counts, validation_states)
    gate = _gate(metrics=metrics, train_majority=train_majority, thresholds=thresholds)
    return {
        "protocol": "sokoban_stage1_slotwise_spatial_and_counterfactual_v1",
        "read_only_world_model": True,
        "teacher_features_used_as_audit_labels": False,
        "decoder": "slotwise_linear_no_mean_pooling",
        "data": {
            "paired_data_dir": str(paired_root.resolve()),
            "train_episode_count": len(train_paths),
            "validation_episode_count": len(validation_paths),
            "train_states_seen_by_probe": total_train_states,
            "validation_posterior_states": validation_states,
            "validation_majority_baseline": validation_majority,
        },
        "probe_training": {
            "epochs": probe_epochs,
            "learning_rate": probe_lr,
            "history": training_history,
        },
        "metrics": metrics,
        "gate": gate,
    }
