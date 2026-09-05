"""Le-WM-owned trajectory contract and real-return conversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class LeWMCriticBatch:
    states: Tensor
    actions: Tensor
    advantages: Tensor
    returns: Tensor
    old_log_probs: Tensor
    old_values: Tensor


def real_collection_to_critic_batch(
    result: Any, *, gamma: float, device: torch.device,
    reward_scale: float,
) -> LeWMCriticBatch | None:
    states, actions, log_probs = [], [], []
    old_values, returns, advantages = [], [], []
    for episode in result.episodes:
        trajectory = episode.info.get("_policy_trajectory")
        if not trajectory:
            continue
        rewards = trajectory["rewards"].float() * reward_scale
        return_to_go = torch.empty_like(rewards)
        running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            return_to_go[index] = running
        values = trajectory["values"].float().reshape(-1)
        states.append(trajectory["states"])
        actions.append(trajectory["actions"].reshape(-1))
        log_probs.append(trajectory["log_probs"].reshape(-1))
        old_values.append(values)
        returns.append(return_to_go)
        advantages.append(return_to_go - values)
    if not states:
        return None
    return LeWMCriticBatch(
        states=torch.cat(states).to(device),
        actions=torch.cat(actions).to(device),
        advantages=torch.cat(advantages).to(device),
        returns=torch.cat(returns).to(device),
        old_log_probs=torch.cat(log_probs).to(device),
        old_values=torch.cat(old_values).to(device),
    )
