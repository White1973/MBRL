"""FrozenLake Stage-1 spatial teacher and isolated prior repair.

The shared posterior/Qwen transition is frozen.  Only the existing generic
``PriorResidualAdapter`` is optimized, so this does not introduce a second
FrozenLake world-model architecture.
"""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from torch import Tensor, nn
import torch.nn.functional as F

from ..data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from ..model import QwenTransitionBackbone, WorldModel
from ..types import BeliefState


class FrozenLakeSpatialDecoder(nn.Module):
    """Position-aware readout from the shared 36-slot belief representation."""

    def __init__(self, hidden_dim: int, model_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.input = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, model_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, 37, model_dim))
        layer = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim * 4, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(model_dim))
        self.player = nn.Linear(model_dim, 16)
        self.goal = nn.Linear(model_dim, 16)
        self.holes = nn.Linear(model_dim, 16)
        self.state_type = nn.Linear(model_dim, 3)  # safe/start, hole, goal

    def forward(self, slots: Tensor) -> dict[str, Tensor]:
        x = self.input(slots.float())
        cls = self.cls.expand(len(x), -1, -1)
        pooled = self.encoder(torch.cat((cls, x), 1) + self.position)[:, 0]
        return {
            "player": self.player(pooled), "goal": self.goal(pooled),
            "holes": self.holes(pooled), "state_type": self.state_type(pooled),
        }


@dataclass(frozen=True)
class RepairGates:
    posterior_accuracy: float = 0.90
    goal_accuracy: float = 0.90
    hole_macro_f1: float = 0.85
    state_type_accuracy: float = 0.90
    prior_accuracy: float = 0.75
    counterfactual_accuracy: float = 0.70


def _labels(episode: Any, state: int) -> tuple[int, int, Tensor, int]:
    rows = list(episode.metadata["map_rows"])
    flat = "".join(rows)
    goal = flat.index("G")
    holes = torch.tensor([1.0 if cell == "H" else 0.0 for cell in flat])
    cell = flat[state]
    state_type = 1 if cell == "H" else 2 if cell == "G" else 0
    return state, goal, holes, state_type


def _next_state_official(rows: list[str], state: int, action: int) -> int:
    """Query the exact deterministic Gym transition used by VAGEN."""
    env = FrozenLakeEnv(desc=rows, is_slippery=False)
    transitions = env.P[state][action]
    if len(transitions) != 1 or abs(float(transitions[0][0]) - 1.0) > 1e-8:
        env.close()
        raise RuntimeError("VAGEN FrozenLake repair expects deterministic transitions")
    result = int(transitions[0][1])
    env.close()
    return result


def _state_sequence(episode: Any) -> list[int]:
    state = int(episode.metadata["start_state"])
    states = [state]
    rows = list(episode.metadata["map_rows"])
    for action in episode.actions[:episode.episode_length].tolist():
        state = _next_state_official(rows, state, int(action))
        states.append(state)
    return states


@torch.no_grad()
def _posterior_rows(model, episodes, config, device) -> tuple[Tensor, dict[str, Tensor]]:
    slots, players, goals, holes, types = [], [], [], [], []
    source = OfflineDataSource(TokenizedEpisodeDataset(episodes), config)
    for begin in range(0, len(episodes), 8):
        selected = episodes[begin:begin + 8]
        batch = source._collate(selected)
        obs, actions = batch.obs_tokens.to(device), batch.actions.to(device)
        lengths, env_ids = batch.episode_lengths.to(device), batch.env_ids.to(device)
        belief = model.get_initial_belief(len(selected), device=device, dtype=obs.dtype)
        states = [_state_sequence(ep) for ep in selected]
        null = torch.full((len(selected),), config.env.null_action_id, device=device)
        for step in range(obs.shape[1]):
            valid = lengths > step
            previous_action = null if step == 0 else actions[:, step - 1]
            belief = model.posterior_step(belief, previous_action, obs[:, step], env_ids)
            slots.append(belief.slots[valid].half().cpu())
            for row in range(len(selected)):
                if bool(valid[row]):
                    player, goal, hole, kind = _labels(selected[row], states[row][step])
                    players.append(player); goals.append(goal); holes.append(hole); types.append(kind)
    return torch.cat(slots), {
        "player": torch.tensor(players), "goal": torch.tensor(goals),
        "holes": torch.stack(holes), "state_type": torch.tensor(types),
    }


def _decoder_loss(
    logits: dict[str, Tensor], labels: dict[str, Tensor], hole_pos_weight: Tensor,
) -> Tensor:
    return (
        F.cross_entropy(logits["player"], labels["player"])
        + 0.5 * F.cross_entropy(logits["goal"], labels["goal"])
        + F.binary_cross_entropy_with_logits(
            logits["holes"], labels["holes"], pos_weight=hole_pos_weight
        )
        + 0.5 * F.cross_entropy(logits["state_type"], labels["state_type"])
    )


@torch.no_grad()
def _decoder_metrics(
    decoder, slots, labels, device, *, hole_threshold: float = 0.5,
) -> dict[str, float]:
    predictions = {key: [] for key in ("player", "goal", "holes", "state_type")}
    for begin in range(0, len(slots), 256):
        out = decoder(slots[begin:begin + 256].to(device))
        predictions["player"].append(out["player"].argmax(-1).cpu())
        predictions["goal"].append(out["goal"].argmax(-1).cpu())
        predictions["holes"].append(
            (out["holes"].sigmoid() >= hole_threshold).cpu()
        )
        predictions["state_type"].append(out["state_type"].argmax(-1).cpu())
    pred = {key: torch.cat(value) for key, value in predictions.items()}
    hole_truth, hole_pred = labels["holes"].bool(), pred["holes"].bool()
    f1s = []
    for cell in range(16):
        tp = (hole_truth[:, cell] & hole_pred[:, cell]).sum().float()
        fp = (~hole_truth[:, cell] & hole_pred[:, cell]).sum().float()
        fn = (hole_truth[:, cell] & ~hole_pred[:, cell]).sum().float()
        f1s.append(float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item()))
    return {
        "posterior_accuracy": float((pred["player"] == labels["player"]).float().mean()),
        "goal_accuracy": float((pred["goal"] == labels["goal"]).float().mean()),
        "hole_macro_f1": sum(f1s) / len(f1s),
        "state_type_accuracy": float((pred["state_type"] == labels["state_type"]).float().mean()),
    }


@torch.no_grad()
def _best_hole_threshold(decoder, slots, labels, device) -> float:
    probabilities = []
    for begin in range(0, len(slots), 256):
        probabilities.append(
            decoder(slots[begin:begin + 256].to(device))["holes"].sigmoid().cpu()
        )
    probability = torch.cat(probabilities)
    truth = labels["holes"].bool()
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in torch.linspace(0.05, 0.95, 37).tolist():
        prediction = probability >= threshold
        f1s = []
        for cell in range(16):
            tp = (truth[:, cell] & prediction[:, cell]).sum().float()
            fp = (~truth[:, cell] & prediction[:, cell]).sum().float()
            fn = (truth[:, cell] & ~prediction[:, cell]).sum().float()
            f1s.append(float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item()))
        score = sum(f1s) / len(f1s)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold


def _teacher_passes(metrics: dict[str, float], gates: RepairGates) -> bool:
    return all(metrics[key] >= getattr(gates, key) for key in (
        "posterior_accuracy", "goal_accuracy", "hole_macro_f1", "state_type_accuracy"
    ))


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_prior_repair(
    *, checkpoint: str, dataset: TokenizedEpisodeDataset, output_dir: str,
    device: torch.device, seed: int, decoder_steps: int = 1000,
    repair_steps: int = 1000, eval_every: int = 100,
    train_episodes: int = 8000, calibration_episodes: int = 1000,
    validation_episodes: int = 1000,
    batch_size: int = 4, lr: float = 3e-5,
    gates: RepairGates = RepairGates(),
) -> dict[str, Any]:
    random.seed(seed); torch.manual_seed(seed)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    config.training = replace(config.training, prior_isolation_mode="residual")
    backbone = QwenTransitionBackbone.from_config(config, device_map={"": str(device)})
    model = WorldModel(config, backbone).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    expected_missing = {key for key in incompatible.missing_keys if key.startswith("transition.prior_residual.")}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model.requires_grad_(False); model.eval()

    indices = list(range(len(dataset.episodes))); random.Random(seed).shuffle(indices)
    train_eps = [dataset.episodes[i] for i in indices[:train_episodes]]
    calibration_start = train_episodes
    validation_start = calibration_start + calibration_episodes
    calibration_eps = [
        dataset.episodes[i]
        for i in indices[calibration_start:validation_start]
    ]
    val_eps = [
        dataset.episodes[i]
        for i in indices[validation_start:validation_start + validation_episodes]
    ]
    train_slots, train_labels = _posterior_rows(model, train_eps, config, device)
    calibration_slots, calibration_labels = _posterior_rows(
        model, calibration_eps, config, device
    )
    val_slots, val_labels = _posterior_rows(model, val_eps, config, device)
    decoder = FrozenLakeSpatialDecoder(config.hidden_dim).to(device)
    decoder_path = output / "spatial_decoder.pt"
    start_decoder_step = 0
    resumed_decoder_optimizer = None
    if decoder_path.exists():
        decoder_payload = torch.load(decoder_path, map_location=device, weights_only=False)
        if decoder_payload.get("format") != "frozenlake_spatial_decoder_v1":
            raise RuntimeError(f"incompatible decoder artifact: {decoder_path}")
        if Path(decoder_payload["base_checkpoint"]).resolve() != Path(checkpoint).resolve():
            raise RuntimeError("decoder artifact belongs to a different base WM")
        decoder.load_state_dict(decoder_payload["model"])
        resumed_decoder_optimizer = decoder_payload.get("optimizer")
        start_decoder_step = int(decoder_payload.get("decoder_step", 0))
        if start_decoder_step == 0 and (output / "train.log").exists():
            import re
            matches = re.findall(
                r"\[spatial-teacher (\d+)\]",
                (output / "train.log").read_text(encoding="utf-8"),
            )
            start_decoder_step = max(map(int, matches), default=0)
        print(
            f"Resuming spatial decoder from step {start_decoder_step}: {decoder_path}",
            flush=True,
        )
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=1e-2)
    if resumed_decoder_optimizer is not None:
        decoder_optimizer.load_state_dict(resumed_decoder_optimizer)
    positives = train_labels["holes"].sum(0).clamp_min(1)
    negatives = len(train_labels["holes"]) - positives
    hole_pos_weight = (negatives / positives).clamp(1.0, 10.0).to(device)
    initial_threshold = _best_hole_threshold(
        decoder, calibration_slots, calibration_labels, device
    )
    initial_metrics = _decoder_metrics(
        decoder, calibration_slots, calibration_labels, device,
        hole_threshold=initial_threshold,
    )
    best_teacher = sum(initial_metrics.values())
    best_teacher_state = copy.deepcopy(decoder.state_dict())
    best_hole_threshold = initial_threshold
    for step in range(start_decoder_step + 1, decoder_steps + 1):
        ids = torch.randint(0, len(train_slots), (min(256, len(train_slots)),))
        labels = {key: value[ids].to(device) for key, value in train_labels.items()}
        loss = _decoder_loss(
            decoder(train_slots[ids].to(device)), labels, hole_pos_weight
        )
        decoder_optimizer.zero_grad(set_to_none=True); loss.backward(); decoder_optimizer.step()
        if step == 1 or step % eval_every == 0 or step == decoder_steps:
            decoder.eval()
            threshold = _best_hole_threshold(
                decoder, calibration_slots, calibration_labels, device
            )
            metrics = _decoder_metrics(
                decoder, calibration_slots, calibration_labels, device,
                hole_threshold=threshold,
            )
            score = sum(metrics.values())
            print(
                f"[spatial-teacher {step}] calibration/hole_threshold={threshold:.3f} | "
                + " | ".join(f"calibration/{k}={v:.4f}" for k,v in metrics.items()),
                flush=True,
            )
            if score > best_teacher:
                best_teacher, best_teacher_state = score, copy.deepcopy(decoder.state_dict())
                best_hole_threshold = threshold
            decoder.train()
    assert best_teacher_state is not None
    decoder.load_state_dict(best_teacher_state); decoder.requires_grad_(False).eval()
    teacher_metrics = _decoder_metrics(
        decoder, val_slots, val_labels, device,
        hole_threshold=best_hole_threshold,
    )
    _atomic_save({
        "format": "frozenlake_spatial_decoder_v1",
        "model": decoder.state_dict(),
        "optimizer": decoder_optimizer.state_dict(),
        "hidden_dim": config.hidden_dim,
        "model_dim": decoder.model_dim,
        "base_checkpoint": str(Path(checkpoint).resolve()),
        "validation_metrics": teacher_metrics,
        "hole_threshold": best_hole_threshold,
        "decoder_step": decoder_steps,
        "split": {
            "train_episodes": train_episodes,
            "calibration_episodes": calibration_episodes,
            "validation_episodes": validation_episodes,
            "seed": seed,
        },
        "gates": asdict(gates),
    }, output / "spatial_decoder.pt")
    if not _teacher_passes(teacher_metrics, gates):
        report = {
            "passed": False, "stage": "spatial_teacher",
            "metrics": teacher_metrics,
            "calibrated_hole_threshold": best_hole_threshold,
            "gates": asdict(gates),
        }
        (output / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"posterior spatial teacher gate failed: {output/'repair_report.json'}")

    adapter = model.transition.prior_residual
    assert adapter is not None
    adapter.requires_grad_(True)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(
        not name.startswith("transition.prior_residual.") for name in trainable_names
    ):
        raise RuntimeError(f"posterior freeze contract violated: {trainable_names}")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    source = OfflineDataSource(TokenizedEpisodeDataset(train_eps), config)
    best_score, best_adapter, best_metrics = -1.0, None, None

    def repair_batch(episodes: list[Any], train: bool) -> tuple[Tensor, dict[str, float]]:
        batch = OfflineDataSource(TokenizedEpisodeDataset(episodes), config)._collate(episodes)
        obs, actions = batch.obs_tokens.to(device), batch.actions.to(device)
        lengths, env_ids = batch.episode_lengths.to(device), batch.env_ids.to(device)
        belief = model.get_initial_belief(len(episodes), device=device, dtype=obs.dtype)
        null = torch.full((len(episodes),), config.env.null_action_id, device=device)
        states = [_state_sequence(ep) for ep in episodes]
        actual_correct = actual_total = cf_correct = cf_total = 0
        losses = []
        with torch.no_grad():
            current = model.posterior_step(belief, null, obs[:, 0], env_ids)
        for time in range(obs.shape[1] - 1):
            valid = lengths > (time + 1)
            with torch.no_grad():
                target = model.posterior_step(current, actions[:, time], obs[:, time + 1], env_ids)
            if bool(valid.any()):
                start = BeliefState(current.slots[valid])
                actual_action = actions[valid, time]
                predicted = model.prior_step(start, actual_action, env_ids[valid])
                actual_target = target.slots[valid].detach()
                actual_labels = torch.tensor([states[row][time + 1] for row in range(len(episodes)) if bool(valid[row])], device=device)
                actual_logits = decoder(predicted.slots)["player"]
                latent = F.smooth_l1_loss(predicted.slots.float(), actual_target.float())
                spatial = F.cross_entropy(actual_logits, actual_labels)
                repeat_slots = start.slots.repeat_interleave(4, 0)
                cf_actions = torch.arange(4, device=device).repeat(start.batch_size)
                cf_pred = model.prior_step(BeliefState(repeat_slots), cf_actions, env_ids[valid].repeat_interleave(4))
                active_rows = [row for row in range(len(episodes)) if bool(valid[row])]
                cf_labels = torch.tensor([
                    _next_state_official(list(episodes[row].metadata["map_rows"]), states[row][time], action)
                    for row in active_rows for action in range(4)
                ], device=device)
                cf_logits = decoder(cf_pred.slots)["player"]
                cf_spatial = F.cross_entropy(cf_logits, cf_labels)
                losses.append(latent + spatial + cf_spatial)
                actual_correct += int((actual_logits.argmax(-1) == actual_labels).sum()); actual_total += len(actual_labels)
                cf_correct += int((cf_logits.argmax(-1) == cf_labels).sum()); cf_total += len(cf_labels)
            current = target
        loss = torch.stack(losses).mean()
        return loss, {"prior_accuracy": actual_correct/max(actual_total,1), "counterfactual_accuracy": cf_correct/max(cf_total,1)}

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        totals = {"prior_accuracy": 0.0, "counterfactual_accuracy": 0.0}; batches = 0
        for begin in range(0, len(val_eps), batch_size):
            _, metrics = repair_batch(val_eps[begin:begin + batch_size], False)
            for key in totals: totals[key] += metrics[key]
            batches += 1
        return {key: value/max(batches,1) for key,value in totals.items()}

    training_order = list(range(len(train_eps)))
    random.Random(seed + 17).shuffle(training_order)
    training_cursor = 0
    for step in range(1, repair_steps + 1):
        if training_cursor + batch_size > len(training_order):
            random.shuffle(training_order)
            training_cursor = 0
        selected = [
            train_eps[index]
            for index in training_order[training_cursor:training_cursor + batch_size]
        ]
        training_cursor += batch_size
        loss, train_metrics = repair_batch(selected, True)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
        if step == 1 or step % eval_every == 0 or step == repair_steps:
            metrics = evaluate(); score = metrics["prior_accuracy"] + metrics["counterfactual_accuracy"]
            print(f"[prior-repair {step}] loss={float(loss):.5f} | " + " | ".join(f"val/{k}={v:.4f}" for k,v in metrics.items()), flush=True)
            if score > best_score:
                best_score, best_adapter, best_metrics = score, copy.deepcopy(adapter.state_dict()), metrics
    assert best_adapter is not None and best_metrics is not None
    adapter.load_state_dict(best_adapter)
    passed = best_metrics["prior_accuracy"] >= gates.prior_accuracy and best_metrics["counterfactual_accuracy"] >= gates.counterfactual_accuracy
    saved = {key:value for key,value in payload.items() if key not in {"model","optimizer","scheduler","config"}}
    saved.update({"model": model.state_dict(), "config": config, "step": int(payload.get("step",0)),
                  "prior_repair": {"base_checkpoint": str(Path(checkpoint).resolve()), "teacher_metrics": teacher_metrics,
                                   "validation_metrics": best_metrics, "gates": asdict(gates), "passed": passed,
                                   "trainable_parameters": trainable_names,
                                   "spatial_decoder": str((output / "spatial_decoder.pt").resolve())}})
    _atomic_save(saved, output / "latest.pt")
    if passed:
        try: os.link(output / "latest.pt", output / "best.pt")
        except FileExistsError: os.replace(output / "latest.pt", output / "best.pt"); os.link(output / "best.pt", output / "latest.pt")
    report = {"passed": passed, "stage": "prior_repair", "teacher_metrics": teacher_metrics,
              "calibrated_hole_threshold": best_hole_threshold,
              "metrics": best_metrics, "gates": asdict(gates), "checkpoint": str(output/"latest.pt")}
    (output / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"prior repair gate failed: {output/'repair_report.json'}")
    return report
