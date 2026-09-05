"""Solver-supervised H3 ranking repair for the compact terminal Reward Head."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.adapters.sokoban import _DIRECTIONS, _bfs_solve
from ..types import BeliefState


def _step(player, boxes, walls, action: int):
    dr, dc = _DIRECTIONS[action + 1]
    nxt = player[0] + dr, player[1] + dc
    if not (0 <= nxt[0] < 6 and 0 <= nxt[1] < 6) or nxt in walls:
        return player, boxes
    if nxt in boxes:
        pushed = nxt[0] + dr, nxt[1] + dc
        if not (0 <= pushed[0] < 6 and 0 <= pushed[1] < 6) or pushed in walls or pushed in boxes:
            return player, boxes
        return nxt, (boxes - {nxt}) | {pushed}
    return nxt, boxes


def _noop_action(player, boxes, walls) -> int:
    for action in range(4):
        if _step(player, boxes, walls, action) == (player, boxes):
            return action
    return 0


def _targets(label: dict[str, Any], metadata: dict[str, Any]):
    player = tuple(label["player_pos"])
    boxes = frozenset(map(tuple, label["box_positions"]))
    targets = frozenset(map(tuple, label["target_positions"]))
    walls = frozenset(map(tuple, metadata["wall_positions"]))
    distances, sequences = [], []
    for action in range(4):
        pos, bxs = _step(player, boxes, walls, action)
        path = _bfs_solve(pos, bxs, targets, walls, (6, 6), 25)
        distances.append(10_000 if path is None else 1 + len(path))
        seq = [action]
        for env_action in (path or [])[:2]:
            follow = env_action - 1
            seq.append(follow)
            pos, bxs = _step(pos, bxs, walls, follow)
        while len(seq) < 3:
            stay = _noop_action(pos, bxs, walls)
            seq.append(stay)
            pos, bxs = _step(pos, bxs, walls, stay)
        sequences.append(seq)
    best = min(distances)
    if best >= 10_000:
        return None
    return (
        torch.tensor([d == best for d in distances]),
        torch.tensor(sequences),
        torch.tensor(distances, dtype=torch.float32),
    )


@torch.no_grad()
def _ground(world_model, episode, timestep: int, device, dtype, null_action_id: int):
    belief = world_model.get_initial_belief(1, device=device, dtype=dtype)
    for t in range(timestep + 1):
        action = torch.tensor(
            [null_action_id if t == 0 else int(episode.actions[t - 1])],
            device=device, dtype=torch.long,
        )
        belief = world_model.posterior_step(
            prev_belief=belief, prev_actions=action,
            observation_tokens=episode.obs_tokens[t:t + 1].to(device, dtype),
            env_ids=torch.zeros(1, device=device, dtype=torch.long),
        )
    return belief


@torch.no_grad()
def _build_split(world_model, episodes, count, device, null_action_id, seed):
    rng = random.Random(seed)
    features, masks, distances = [], [], []
    dtype = next(world_model.parameters()).dtype
    attempts = 0
    while len(features) < count:
        attempts += 1
        if attempts > count * 30:
            raise RuntimeError(f"could only build {len(features)}/{count} ranking groups")
        episode = rng.choice(episodes)
        labels = episode.metadata.get("state_labels") or []
        max_t = min(int(episode.episode_length) - 1, len(labels) - 1)
        if max_t < 0:
            continue
        # Half reset states, half early trajectory states.
        t = 0 if rng.random() < 0.5 else rng.randint(0, min(max_t, 4))
        target = _targets(labels[t], episode.metadata)
        if target is None:
            continue
        optimal_mask, sequences, action_distances = target
        start = _ground(world_model, episode, t, device, dtype, null_action_id)
        repeated = BeliefState(start.slots.repeat_interleave(4, 0))
        trajectory = world_model.rollout_prior(
            repeated, sequences.to(device=device, dtype=torch.long)
        )
        # Preserve first-action differences at H1 while still teaching the
        # deployed H3 endpoint. Shape per group: (H=3, A=4, D).
        pooled_horizons = torch.stack([
            world_model.reward_head.readout(
                trajectory.beliefs[:, horizon]
            ).float().cpu()
            for horizon in range(3)
        ])
        features.append(pooled_horizons)
        masks.append(optimal_mask)
        distances.append(action_distances)
        if len(features) % 100 == 0 or len(features) == count:
            print(f"ranking cache: {len(features)}/{count}", flush=True)
    return torch.stack(features), torch.stack(masks), torch.stack(distances)


def _listwise_loss(logits: Tensor, masks: Tensor) -> Tensor:
    masked = logits.masked_fill(~masks, -1e9)
    return -(
        torch.logsumexp(masked, -1) - torch.logsumexp(logits, -1)
    ).mean()


def _distance_pairwise_loss(logits: Tensor, distances: Tensor) -> Tensor:
    """Distance-aware ranking: larger BFS disadvantage gets larger margin."""
    losses = []
    finite = distances < 10_000
    for better in range(4):
        for worse in range(4):
            gap = distances[:, worse] - distances[:, better]
            valid = finite[:, better] & (gap > 0)
            if not bool(valid.any()):
                continue
            # One-step alternatives need only a small separation; deadlocks
            # and long detours receive a substantially stronger margin.
            margin = torch.where(
                finite[:, worse],
                (0.10 + 0.05 * gap.clamp(max=6)),
                torch.full_like(gap, 0.75),
            )
            losses.append(F.relu(
                margin[valid] - (logits[valid, better] - logits[valid, worse])
            ).mean())
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def _ranking_metrics(logits: Tensor, masks: Tensor) -> dict[str, float]:
    order = logits.argsort(-1, descending=True)
    top1 = masks.gather(1, order[:, :1]).any(1).float().mean()
    top2 = masks.gather(1, order[:, :2]).any(1).float().mean()
    return {"top1": float(top1), "top2": float(top2)}


def _terminal_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    probabilities = logits.sigmoid()
    brier = (probabilities - labels).pow(2).mean()
    # Rank-based AUC without sklearn dependency.
    positive, negative = probabilities[labels > .5], probabilities[labels <= .5]
    auc = ((positive[:, None] > negative).float() + .5 * (positive[:, None] == negative).float()).mean()
    return {"brier": float(brier), "auc": float(auc)}


def _calibrate_and_fold(classifier, features: Tensor, labels: Tensor, device):
    with torch.no_grad():
        raw = classifier.forward_pooled(features.to(device)).float()
    scale = torch.ones((), device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    targets = labels.to(device)
    optimizer = torch.optim.LBFGS([scale, bias], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(scale * raw + bias, targets)
        loss.backward(); return loss
    optimizer.step(closure)
    scale_value, bias_value = float(scale.detach()), float(bias.detach())
    last = classifier.compact_net[-1] if isinstance(classifier.compact_net, torch.nn.Sequential) else classifier.compact_net
    with torch.no_grad():
        last.weight.mul_(scale_value)
        last.bias.mul_(scale_value).add_(bias_value)
    return scale_value, bias_value


def _precision_threshold(probabilities: Tensor, labels: Tensor, target=.75, min_recall=.10):
    candidates = torch.unique(probabilities).sort(descending=True).values
    best = None
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = (predicted & (labels > .5)).sum().item()
        fp = (predicted & (labels <= .5)).sum().item()
        fn = ((~predicted) & (labels > .5)).sum().item()
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        if precision >= target and recall >= min_recall:
            if best is None or recall > best[2]:
                best = (float(threshold), precision, recall)
    if best is None:
        raise RuntimeError(
            f"calibration has no threshold with precision>={target} and recall>={min_recall}"
        )
    return best


def train_ranking_reward_head(
    *, world_model, dataset, source_checkpoint, output_path, cache_path,
    terminal_cache_path, train_states, validation_states, test_states,
    epochs, learning_rate, ranking_loss_weight, pairwise_loss_weight,
    terminal_loss_weight, device, null_action_id,
):
    if world_model.reward_head.head_hidden_dim is None:
        raise ValueError("ranking repair requires a compact Reward Head")
    world_model.eval().requires_grad_(False)
    world_model.reward_head.requires_grad_(True)
    episodes = [e for e in dataset.episodes if e.metadata.get("strategy") == "expert"]
    rng = random.Random(20260812); rng.shuffle(episodes)
    n = len(episodes); train_eps = episodes[:int(.7*n)]
    val_eps = episodes[int(.7*n):int(.85*n)]; test_eps = episodes[int(.85*n):]
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.is_file():
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        requested = {"train": train_states, "validation": validation_states, "test": test_states}
        if cache.get("version") != 2 or cache.get("sizes") != requested:
            raise ValueError(
                "ranking cache mismatch: this trainer requires version=2 "
                f"and sizes={requested}; cached version={cache.get('version')} "
                f"sizes={cache.get('sizes')}. Use a new RANK_CACHE path."
            )
        print(f"Loading ranking cache: {cache_file}", flush=True)
    else:
        cache = {
            "version": 2,
            "sizes": {"train": train_states, "validation": validation_states, "test": test_states},
            "train": _build_split(world_model, train_eps, train_states, device, null_action_id, 1),
            "validation": _build_split(world_model, val_eps, validation_states, device, null_action_id, 2),
            "test": _build_split(world_model, test_eps, test_states, device, null_action_id, 3),
        }
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True); torch.save(cache, cache_file)
            print(f"Saved ranking cache: {cache_file}", flush=True)
    terminal = torch.load(terminal_cache_path, map_location="cpu", weights_only=False, mmap=True)
    classifier = world_model.reward_head
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=1e-2)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(epochs):
        classifier.train(); rank_x, rank_mask, rank_distance = cache["train"]
        term_x, term_y, _ = terminal["train"]
        steps = max(1, len(rank_x) // 128)
        for _ in range(steps):
            ri = torch.randint(len(rank_x), (128,)); ti = torch.randint(len(term_x), (256,))
            rx, rm = rank_x[ri].to(device), rank_mask[ri].to(device)
            rd = rank_distance[ri].to(device)
            tx, ty = term_x[ti].to(device), term_y[ti].to(device)
            horizon_logits = classifier.forward_pooled(
                rx.flatten(0, 2)
            ).reshape(-1, 3, 4)
            horizon_weights = (0.20, 0.30, 0.50)
            rank_loss = sum(
                weight * _listwise_loss(horizon_logits[:, horizon], rm)
                for horizon, weight in enumerate(horizon_weights)
            )
            pairwise_loss = sum(
                weight * _distance_pairwise_loss(
                    horizon_logits[:, horizon], rd
                )
                for horizon, weight in enumerate(horizon_weights)
            )
            terminal_loss = F.binary_cross_entropy_with_logits(classifier.forward_pooled(tx), ty)
            loss = (
                terminal_loss_weight * terminal_loss
                + ranking_loss_weight * rank_loss
                + pairwise_loss_weight * pairwise_loss
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0); optimizer.step()
        classifier.eval()
        with torch.no_grad():
            vx, vm, _ = cache["validation"]
            # Formal deployment Gate remains H3-only.
            vl = classifier.forward_pooled(
                vx[:, 2].to(device).flatten(0, 1)
            ).reshape(-1, 4).cpu()
            rank_metrics = _ranking_metrics(vl, vm)
            tx, ty, _ = terminal["validation"]
            terminal_metrics = _terminal_metrics(classifier.forward_pooled(tx.to(device)).cpu(), ty)
        score = rank_metrics["top1"] if terminal_metrics["brier"] < .1523 else -1.0
        print(f"epoch {epoch:3d}: val top1={rank_metrics['top1']:.3f} top2={rank_metrics['top2']:.3f} terminal AUC={terminal_metrics['auc']:.3f} Brier={terminal_metrics['brier']:.4f}", flush=True)
        if score > best:
            best, patience = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            patience += 1
            if patience >= 12: break
    if best_state is None: raise RuntimeError("no ranking checkpoint retained terminal Brier gate")
    classifier.load_state_dict(best_state); classifier.eval()
    calibration_x, calibration_y, _ = terminal["calibration"]
    platt_scale, platt_bias = _calibrate_and_fold(
        classifier, calibration_x, calibration_y, device
    )
    with torch.no_grad():
        calibration_probabilities = classifier.forward_pooled(
            calibration_x.to(device)
        ).sigmoid().cpu()
    decision_threshold, calibration_precision, calibration_recall = _precision_threshold(
        calibration_probabilities, calibration_y
    )
    with torch.no_grad():
        test_x, test_mask, _ = cache["test"]
        test_logits = classifier.forward_pooled(
            test_x[:, 2].to(device).flatten(0, 1)
        ).reshape(-1, 4).cpu()
        ranking_test = _ranking_metrics(test_logits, test_mask)
        tx, ty, _ = terminal["test"]
        terminal_test = _terminal_metrics(classifier.forward_pooled(tx.to(device)).cpu(), ty)
    with torch.no_grad():
        test_probabilities = classifier.forward_pooled(tx.to(device)).sigmoid().cpu()
    predicted = test_probabilities >= decision_threshold
    positive = ty > .5
    tp = (predicted & positive).sum().item(); fp = (predicted & ~positive).sum().item()
    fn = ((~predicted) & positive).sum().item()
    heldout_precision = tp / max(tp + fp, 1); heldout_recall = tp / max(tp + fn, 1)
    report = {
        "ranking_test": ranking_test, "terminal_test": terminal_test,
        "platt_scale": platt_scale, "platt_bias": platt_bias,
        "decision_threshold": decision_threshold,
        "calibration_precision": calibration_precision,
        "calibration_recall": calibration_recall,
        "heldout_precision": heldout_precision,
        "heldout_recall": heldout_recall,
        "training": {
            "ranking_loss_weight": ranking_loss_weight,
            "pairwise_loss_weight": pairwise_loss_weight,
            "terminal_loss_weight": terminal_loss_weight,
            "horizon_weights": [0.20, 0.30, 0.50],
            "cache_version": 2,
        },
    }
    if (ranking_test["top1"] < .55 or ranking_test["top2"] < .80
            or terminal_test["brier"] >= .1566 or heldout_precision < .70
            or heldout_recall < .10):
        raise RuntimeError(f"ranking Reward Head quality gate failed: {report}")
    checkpoint = dict(source_checkpoint)
    state = dict(checkpoint.get("model", checkpoint))
    state.update({f"reward_head.{k}": v for k, v in classifier.state_dict().items()})
    checkpoint["model"] = state
    injection = dict(checkpoint.get("reward_head_injection", {}))
    injection.update({
        "solver_ranking_repair": True, "ranking_horizon": 3,
        "decision_threshold": decision_threshold, **report,
    })
    checkpoint["reward_head_injection"] = injection
    checkpoint["injected_reward_head"] = True
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved solver-ranked Reward Head: {output}\n{json.dumps(report, indent=2)}", flush=True)
