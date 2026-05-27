"""Real environment collector for online MBRL.

Runs a policy in a real environment, collecting episodes for:
  1. Eval (success rate, avg return)
  2. Online replay (tokenized episodes for WM refresh / belief sampling)

This module bridges three abstractions:
  - EnvProtocol (real environment)
  - Policy (LLM or MLP actor-critic)
  - WorldModel posterior (obs → belief for policy conditioning)
  - ObservationTokenizer (raw obs → visual tokens)

It does NOT import rl/ or pipelines/. It produces data, not gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from sembelief_wm.data.schema import TokenizedEpisode


# ---------------------------------------------------------------------------
# Protocols — what the collector needs from its dependencies
# ---------------------------------------------------------------------------

class EnvProtocol(Protocol):
    def reset(self, *, seed: int | None = None) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]: ...


class TokenizerProtocol(Protocol):
    def tokenize(self, obs: Any) -> Tensor:
        """Raw observation → (K, D) tokens."""
        ...


class PosteriorProtocol(Protocol):
    def get_initial_belief(
        self, batch_size: int, *, device: Any, dtype: Any
    ) -> Any: ...

    def posterior_step(
        self,
        *,
        prev_belief: Any,
        prev_actions: Tensor,
        observation_tokens: Tensor,
        env_ids: Tensor | None = None,
    ) -> Any: ...


class PolicyProtocol(Protocol):
    def act(self, states: Any, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Any]:
        """(action, log_prob, value, info)"""
        ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RealCollectorConfig:
    max_steps: int = 25
    deterministic: bool = False
    seed_offset: int = 10000


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Result of one real-env episode."""
    reward: float
    length: int
    success: bool
    info: dict[str, Any] = field(default_factory=dict)
    tokenized: TokenizedEpisode | None = None


@dataclass
class CollectionResult:
    """Aggregated result of collecting multiple episodes."""
    episodes: list[EpisodeResult]
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Real collector
# ---------------------------------------------------------------------------

class RealCollector:
    """Collects episodes by running a policy in a real environment.

    The collector handles the full obs → tokenize → posterior → belief → act
    loop. It does NOT do any gradient computation or optimizer steps.

    Usage:
        collector = RealCollector(
            env_factory=lambda seed: SokobanEnv(seed=seed),
            tokenizer=image_tokenizer,
            world_model=world_model,
            policy=llm_policy,
            config=RealCollectorConfig(max_steps=25),
        )
        result = collector.collect(num_episodes=128, update_id=0)
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[int | None], EnvProtocol],
        tokenizer: TokenizerProtocol,
        world_model: PosteriorProtocol,
        policy: PolicyProtocol,
        config: RealCollectorConfig | None = None,
        # Environment-specific action mapping
        model_to_env_action: Callable[[int], Any] | None = None,
        env_to_model_action: Callable[[Any], int] | None = None,
        null_action_fn: Callable[[int, Any, Any], Tensor] | None = None,
        env_id_tensor: Tensor | None = None,
        env_id: str = "sokoban",
        replay_buffer: Any | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.env_factory = env_factory
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.policy = policy
        self.config = config or RealCollectorConfig()
        self.model_to_env_action = model_to_env_action or (lambda a: a + 1)
        self.env_to_model_action = env_to_model_action or (lambda a: a - 1)
        self.null_action_fn = null_action_fn
        self.env_id_tensor = env_id_tensor
        self.env_id = env_id
        self.replay_buffer = replay_buffer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

    def _get_null_action(self, batch_size: int, device: Any, dtype: Any) -> Tensor:
        if self.null_action_fn is not None:
            return self.null_action_fn(batch_size, device, dtype)
        # Default: action 4 (null/noop) for discrete Sokoban
        return torch.full((batch_size,), 4, device=device, dtype=torch.long)

    @torch.no_grad()
    def collect(
        self,
        num_episodes: int,
        *,
        update_id: int = 0,
        collect_tokenized: bool = False,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> CollectionResult:
        """Run policy in real env for num_episodes, return results + metrics.

        Args:
            num_episodes: How many episodes to collect.
            update_id: Current training update (used for seeding).
            collect_tokenized: If True, also produce TokenizedEpisode for replay.
            device: Device for model inference.
            dtype: Dtype for model inference.
        """
        device = torch.device(device) if isinstance(device, str) else device
        results: list[EpisodeResult] = []

        for ep_idx in range(num_episodes):
            seed = self.config.seed_offset + update_id * num_episodes + ep_idx
            result = self._run_episode(
                seed=seed,
                ep_idx=ep_idx,
                update_id=update_id,
                collect_tokenized=collect_tokenized,
                device=device,
                dtype=dtype,
            )
            results.append(result)

        metrics = self._compute_metrics(results)
        return CollectionResult(episodes=results, metrics=metrics)

    def _run_episode(
        self,
        *,
        seed: int,
        ep_idx: int,
        update_id: int,
        collect_tokenized: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> EpisodeResult:
        env = self.env_factory(seed)

        obs = env.reset(seed=seed)
        done = False
        ep_return = 0.0
        ep_steps = 0

        # For building TokenizedEpisode
        obs_tokens_list: list[Tensor] = []
        actions_list: list[int] = []
        rewards_list: list[float] = []

        # Tokenize initial observation and get initial belief
        obs_tokens = self.tokenizer.tokenize(obs)  # (K, D)
        if collect_tokenized:
            obs_tokens_list.append(obs_tokens.detach().cpu())

        belief = self.world_model.get_initial_belief(
            1, device=device, dtype=dtype
        )
        null_action = self._get_null_action(1, device, dtype)
        belief = self.world_model.posterior_step(
            prev_belief=belief,
            prev_actions=null_action,
            observation_tokens=obs_tokens.unsqueeze(0).to(device, dtype),
            env_ids=self.env_id_tensor,
        )

        info: dict[str, Any] = {}
        while not done and ep_steps < self.config.max_steps:
            # Get belief slots for policy
            belief_slots = belief.slots if hasattr(belief, 'slots') else belief

            # Policy acts on belief
            action, _log_prob, _value, _info = self.policy.act(
                belief_slots, env_ids=self.env_id_tensor
            )

            # Convert model action to env action
            action_int = int(action.item())
            env_action = self.model_to_env_action(action_int)

            obs, reward, done, info = env.step(env_action)
            ep_return += float(reward)
            ep_steps += 1

            if collect_tokenized:
                actions_list.append(action_int)
                rewards_list.append(float(reward))

            # Update belief with new observation (unless episode ended)
            if not done or collect_tokenized:
                next_obs_tokens = self.tokenizer.tokenize(obs)
                if collect_tokenized:
                    obs_tokens_list.append(next_obs_tokens.detach().cpu())
                if not done:
                    belief = self.world_model.posterior_step(
                        prev_belief=belief,
                        prev_actions=action.to(device),
                        observation_tokens=next_obs_tokens.unsqueeze(0).to(device, dtype),
                        env_ids=self.env_id_tensor,
                    )

        success = info.get("success", False) or (done and ep_return > 0.0)

        # Build tokenized episode for replay
        tokenized: TokenizedEpisode | None = None
        if collect_tokenized and ep_steps > 0:
            tokenized = self._build_tokenized_episode(
                obs_tokens_list, actions_list, rewards_list,
                ep_steps, update_id, ep_idx, success,
            )

        # Close env if possible
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

        return EpisodeResult(
            reward=ep_return,
            length=ep_steps,
            success=success,
            info=info,
            tokenized=tokenized,
        )

    def _build_tokenized_episode(
        self,
        obs_tokens_list: list[Tensor],
        actions_list: list[int],
        rewards_list: list[float],
        ep_steps: int,
        update_id: int,
        ep_idx: int,
        success: bool,
    ) -> TokenizedEpisode:
        """Build a TokenizedEpisode from collected data."""
        seq_len = ep_steps + 1  # T+1 observations

        obs_tokens = torch.stack(obs_tokens_list[:seq_len], dim=0)  # (T+1, K, D)

        # Actions: T+1 length, last is null/padding
        action_tensor = torch.full((seq_len,), 4, dtype=torch.long)  # null action
        action_tensor[:ep_steps] = torch.tensor(actions_list[:ep_steps], dtype=torch.long)

        # Rewards: T+1 length, last is zero
        reward_tensor = torch.zeros(seq_len, dtype=torch.float32)
        reward_tensor[:ep_steps] = torch.tensor(rewards_list[:ep_steps], dtype=torch.float32)

        return TokenizedEpisode(
            obs_tokens=obs_tokens,
            actions=action_tensor,
            rewards=reward_tensor,
            episode_length=ep_steps,
            env_id=self.env_id,
            split="train",
            metadata={
                "source": "online_collect",
                "update": update_id,
                "episode": ep_idx,
                "success": success,
            },
        )

    @staticmethod
    def _compute_metrics(results: list[EpisodeResult]) -> dict[str, float]:
        n = max(1, len(results))
        successes = sum(1 for r in results if r.success)
        total_return = sum(r.reward for r in results)
        total_steps = sum(r.length for r in results)
        returns = torch.tensor([r.reward for r in results], dtype=torch.float32)
        lengths = torch.tensor([r.length for r in results], dtype=torch.float32)

        return {
            "success_rate": successes / n,
            "avg_return": total_return / n,
            "avg_episode_length": total_steps / n,
            "return_std": float(returns.std().item()) if len(results) > 1 else 0.0,
            "length_std": float(lengths.std().item()) if len(results) > 1 else 0.0,
            "num_success": float(successes),
            "num_episodes": float(len(results)),
        }

    def collect_and_store(
        self,
        num_episodes: int,
        update_id: int = 0,
        *,
        replay_buffer: Any | None = None,
    ) -> CollectionResult:
        """Collect episodes and store tokenized versions in replay buffer.

        Uses self.replay_buffer if replay_buffer arg is not provided.
        This method signature matches what MBRLPipeline expects:
            collect_and_store(num_episodes, update_id)
        """
        buffer = replay_buffer or self.replay_buffer
        collect_tokenized = buffer is not None

        result = self.collect(
            num_episodes,
            update_id=update_id,
            collect_tokenized=collect_tokenized,
            device=self.device,
            dtype=self.dtype,
        )

        if buffer is not None:
            for i, ep_result in enumerate(result.episodes):
                if ep_result.tokenized is not None:
                    episode_id = f"u{update_id:06d}_online_ep{i:05d}"
                    buffer.add_episode(episode_id, ep_result.tokenized)

            result.metrics["replay/saved_episodes"] = sum(
                1 for r in result.episodes if r.tokenized is not None
            )
            result.metrics["replay/buffer_size"] = float(buffer.size)

        return result
