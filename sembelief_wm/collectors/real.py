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

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from sembelief_wm.data.schema import TokenizedEpisode


def _fixed_sokoban_solution_length(level: dict[str, Any]) -> int | None:
    """Return the exact shortest path length for a serialized Sokoban level."""
    from sembelief_wm.data.adapters.sokoban import _bfs_solve

    if "room_state" not in level or "room_fixed" not in level:
        return None
    room_state = level["room_state"]
    room_fixed = level["room_fixed"]
    rows, cols = len(room_state), len(room_state[0])
    player = next(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if room_state[row][col] in (5, 6)
    )
    boxes = frozenset(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if room_state[row][col] in (3, 4)
    )
    targets = frozenset(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if room_fixed[row][col] == 2
    )
    walls = frozenset(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if room_fixed[row][col] == 0
    )
    path = _bfs_solve(
        player, boxes, targets, walls, (rows, cols), max_depth=100
    )
    return None if path is None else len(path)


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
    exploration_epsilon: float = 0.0
    capture_policy_trajectory: bool = False


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
        wm_action_id_offset: int = 0,
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
        if not 0.0 <= self.config.exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1]")
        self.model_to_env_action = model_to_env_action or (lambda a: a + 1)
        if wm_action_id_offset < 0:
            raise ValueError("wm_action_id_offset must be non-negative")
        self.wm_action_id_offset = wm_action_id_offset
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
        seeds: list[int] | None = None,
        levels: list[dict] | None = None,
        forced_first_actions: list[int] | None = None,
        forced_action_sequences: list[list[int]] | None = None,
    ) -> CollectionResult:
        """Run policy in real env for num_episodes, return results + metrics.

        Args:
            num_episodes: How many episodes to collect.
            update_id: Current training update (used for seeding).
            collect_tokenized: If True, also produce TokenizedEpisode for replay.
            device: Device for model inference.
            dtype: Dtype for model inference.
            seeds: Optional fixed seed list for reproducible evaluation (e.g. a
                held-out test set). If given, len(seeds) overrides num_episodes
                and each episode uses the corresponding seed verbatim (no
                update_id mixing). Useful for VAGEN-style fixed test sets.
            levels: Optional list of precomputed layouts ({room_state, room_fixed}).
                Takes precedence over seeds — each episode injects the layout via
                env.reset_with_room(), giving 100% reproducible levels regardless
                of RNG/generate_room divergence. len(levels) overrides num_episodes.
        """
        stabilize = getattr(
            self.policy,
            "set_deterministic_forward_mode",
            None,
        )
        if callable(stabilize):
            stabilize()
        device = torch.device(device) if isinstance(device, str) else device
        results: list[EpisodeResult] = []

        if levels is not None:
            episode_seeds = list(range(len(levels)))
            num_episodes = len(levels)
        elif seeds is not None:
            episode_seeds = list(seeds)
            num_episodes = len(episode_seeds)
        else:
            episode_seeds = [
                self.config.seed_offset + update_id * num_episodes + ep_idx
                for ep_idx in range(num_episodes)
            ]
        if forced_first_actions is not None:
            if len(forced_first_actions) != num_episodes:
                raise ValueError(
                    "forced_first_actions must match the collected episode count"
                )
            if any(action < 0 or action >= 4 for action in forced_first_actions):
                raise ValueError("forced first actions must be in [0, 3]")
        if forced_action_sequences is not None:
            if forced_first_actions is not None:
                raise ValueError(
                    "forced_first_actions and forced_action_sequences are mutually exclusive"
                )
            if len(forced_action_sequences) != num_episodes:
                raise ValueError(
                    "forced_action_sequences must match the collected episode count"
                )
            if any(
                action < 0 or action >= 4
                for sequence in forced_action_sequences for action in sequence
            ):
                raise ValueError("forced action sequence entries must be in [0, 3]")

        for ep_idx, seed in enumerate(episode_seeds):
            lvl = levels[ep_idx] if levels is not None else None
            result = self._run_episode(
                seed=seed,
                ep_idx=ep_idx,
                update_id=update_id,
                collect_tokenized=collect_tokenized,
                device=device,
                dtype=dtype,
                level=lvl,
                forced_first_action=(
                    None if forced_first_actions is None
                    else forced_first_actions[ep_idx]
                ),
                forced_action_sequence=(
                    None if forced_action_sequences is None
                    else forced_action_sequences[ep_idx]
                ),
            )
            results.append(result)
            progress_every = int(os.environ.get(
                "REAL_COLLECT_PROGRESS_EVERY", "0"
            ))
            if (
                progress_every > 0
                and ((ep_idx + 1) % progress_every == 0 or ep_idx + 1 == num_episodes)
            ):
                print(
                    f"  real collection progress: {ep_idx + 1}/{num_episodes}",
                    flush=True,
                )

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
        level: dict | None = None,
        forced_first_action: int | None = None,
        forced_action_sequence: list[int] | None = None,
    ) -> EpisodeResult:
        env = self.env_factory(seed)

        if level is not None:
            # Inject a precomputed layout (e.g. VagenMirror0416 exported level)
            # for 100% reproducible eval. Bypasses generate_room entirely.
            obs = env.reset_with_room(level["room_state"], level["room_fixed"])
        else:
            obs = env.reset(seed=seed)
        done = False
        ep_return = 0.0
        ep_steps = 0

        # For building TokenizedEpisode
        obs_tokens_list: list[Tensor] = []
        semantic_teacher_list: list[Tensor] = []
        semantic_teacher_tokenize = getattr(
            self.tokenizer, "semantic_teacher_tokenize", None
        )
        has_semantic_teacher = callable(semantic_teacher_tokenize)
        actions_list: list[int] = []
        action_counts = [0, 0, 0, 0]
        rewards_list: list[float] = []

        # Tokenize initial observation and get initial belief
        obs_tokens = self.tokenizer.tokenize(obs)  # (K, D)
        if collect_tokenized:
            obs_tokens_list.append(obs_tokens.detach().cpu())
            if has_semantic_teacher:
                semantic_teacher_list.append(
                    semantic_teacher_tokenize(obs).detach().cpu()
                )

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
        initial_actor_logits: Tensor | None = None
        initial_actor_action: int | None = None
        exploration_steps = 0
        policy_states: list[Tensor] = []
        policy_next_states: list[Tensor] = []
        policy_board_states: list[Tensor] = []
        policy_next_board_states: list[Tensor] = []
        policy_room_fixed: Tensor | None = None
        policy_actions: list[Tensor] = []
        policy_log_probs: list[Tensor] = []
        policy_values: list[Tensor] = []
        policy_rewards: list[float] = []
        policy_dones: list[bool] = []
        while not done and ep_steps < self.config.max_steps:
            # Get belief slots for policy
            belief_slots = belief.slots if hasattr(belief, 'slots') else belief

            # Policy acts on belief
            action, _log_prob, _entropy, _value = self.policy.act(
                belief_slots,
                env_ids=self.env_id_tensor,
                deterministic=self.config.deterministic,
            )
            if ep_steps == 0 and forced_first_action is not None:
                action = torch.full_like(action, forced_first_action)
            if (
                forced_action_sequence is not None
                and ep_steps < len(forced_action_sequence)
            ):
                action = torch.full_like(action, forced_action_sequence[ep_steps])
            if (
                not self.config.deterministic
                and self.config.exploration_epsilon > 0.0
                and bool(torch.rand(()) < self.config.exploration_epsilon)
            ):
                action = torch.randint(
                    0, 4, action.shape, device=action.device,
                    dtype=action.dtype,
                )
                exploration_steps += 1
            if ep_steps == 0:
                initial_actor_action = int(action.item())
                actor_logits = getattr(self.policy, "actor_logits", None)
                if callable(actor_logits):
                    initial_actor_logits = actor_logits(
                        belief_slots
                    ).detach().float().cpu().squeeze(0)

            # Convert model action to env action
            action_int = int(action.item())
            if 0 <= action_int < 4:
                action_counts[action_int] += 1
            env_action = self.model_to_env_action(action_int)

            if self.config.capture_policy_trajectory:
                policy_states.append(belief_slots.squeeze(0).detach().cpu())
                room_state = getattr(env, "room_state", None)
                room_fixed = getattr(env, "room_fixed", None)
                if room_state is not None and room_fixed is not None:
                    policy_board_states.append(
                        torch.as_tensor(room_state).detach().cpu().clone()
                    )
                    if policy_room_fixed is None:
                        policy_room_fixed = (
                            torch.as_tensor(room_fixed).detach().cpu().clone()
                        )
                policy_actions.append(action.squeeze(0).detach().cpu())
                policy_log_probs.append(_log_prob.squeeze(0).detach().float().cpu())
                policy_values.append(_value.squeeze(0).detach().float().cpu())

            obs, reward, done, info = env.step(env_action)
            if self.config.capture_policy_trajectory:
                policy_rewards.append(float(reward))
                policy_dones.append(bool(done))
                next_room_state = getattr(env, "room_state", None)
                if next_room_state is not None:
                    policy_next_board_states.append(
                        torch.as_tensor(next_room_state).detach().cpu().clone()
                    )
            ep_return += float(reward)
            ep_steps += 1

            if collect_tokenized:
                actions_list.append(action_int + self.wm_action_id_offset)
                rewards_list.append(float(reward))

            # A transition-level audit needs the posterior after the terminal
            # observation as well. Normal acting still stops at ``done``; the
            # extra posterior is captured read-only and never fed back to the
            # environment policy.
            if not done or collect_tokenized or self.config.capture_policy_trajectory:
                next_obs_tokens = self.tokenizer.tokenize(obs)
                if collect_tokenized:
                    obs_tokens_list.append(next_obs_tokens.detach().cpu())
                    if has_semantic_teacher:
                        semantic_teacher_list.append(
                            semantic_teacher_tokenize(obs).detach().cpu()
                        )
                if not done or self.config.capture_policy_trajectory:
                    next_belief = self.world_model.posterior_step(
                        prev_belief=belief,
                        prev_actions=(
                            action.to(device) + self.wm_action_id_offset
                        ),
                        observation_tokens=next_obs_tokens.unsqueeze(0).to(device, dtype),
                        env_ids=self.env_id_tensor,
                    )
                    if self.config.capture_policy_trajectory:
                        next_slots = (
                            next_belief.slots
                            if hasattr(next_belief, "slots") else next_belief
                        )
                        policy_next_states.append(
                            next_slots.squeeze(0).detach().cpu()
                        )
                    if not done:
                        belief = next_belief

        success = info.get("success", False) or (done and ep_return > 0.0)
        # Private diagnostics are consumed only by RealEnvEvaluator. They make
        # the Actor -> fixed real posterior -> eval action link observable.
        info = dict(info)
        info["_actor_initial_action"] = initial_actor_action
        info["_actor_initial_logits"] = initial_actor_logits
        info["_actor_action_counts"] = action_counts
        info["_exploration_steps"] = exploration_steps
        if self.config.capture_policy_trajectory and policy_states:
            info["_policy_trajectory"] = {
                "states": torch.stack(policy_states),
                "next_states": torch.stack(policy_next_states),
                "actions": torch.stack(policy_actions).long(),
                "log_probs": torch.stack(policy_log_probs),
                "values": torch.stack(policy_values),
                "rewards": torch.tensor(policy_rewards, dtype=torch.float32),
                "dones": torch.tensor(policy_dones, dtype=torch.bool),
            }
            if (
                len(policy_board_states) == len(policy_states)
                and len(policy_next_board_states) == len(policy_states)
                and policy_room_fixed is not None
            ):
                info["_policy_trajectory"].update({
                    "board_states": torch.stack(policy_board_states),
                    "next_board_states": torch.stack(policy_next_board_states),
                    "room_fixed": policy_room_fixed,
                })

        # Build tokenized episode for replay
        tokenized: TokenizedEpisode | None = None
        if collect_tokenized and ep_steps > 0:
            tokenized = self._build_tokenized_episode(
                obs_tokens_list, semantic_teacher_list, actions_list, rewards_list,
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
        semantic_teacher_list: list[Tensor],
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
        semantic_teacher_tokens = (
            None
            if not semantic_teacher_list
            else torch.stack(semantic_teacher_list[:seq_len], dim=0)
        )
        if semantic_teacher_tokens is not None and semantic_teacher_tokens.shape[0] != seq_len:
            raise RuntimeError(
                "Semantic teacher collection is incomplete: expected "
                f"{seq_len} frames, got {semantic_teacher_tokens.shape[0]}."
            )

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
            semantic_teacher_tokens=semantic_teacher_tokens,
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

        action_counts = [0, 0, 0, 0]
        for result in results:
            for action, count in enumerate(
                result.info.get("_actor_action_counts", [0, 0, 0, 0])
            ):
                action_counts[action] += int(count)
        action_total = max(1, sum(action_counts))

        metrics = {
            "success_rate": successes / n,
            "avg_return": total_return / n,
            "avg_episode_length": total_steps / n,
            "return_std": float(returns.std().item()) if len(results) > 1 else 0.0,
            "length_std": float(lengths.std().item()) if len(results) > 1 else 0.0,
            "num_success": float(successes),
            "num_episodes": float(len(results)),
        }
        for action, count in enumerate(action_counts):
            metrics[f"action_{action}_fraction"] = count / action_total
        metrics["action_max_fraction"] = max(action_counts) / action_total
        metrics["action_num_covered"] = float(sum(count > 0 for count in action_counts))
        metrics["exploration_fraction"] = sum(
            int(result.info.get("_exploration_steps", 0)) for result in results
        ) / action_total
        return metrics

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


class RealEnvEvaluator:
    """Wrap a `RealCollector` into the `Evaluator` protocol consumed by MBRLPipeline.

    Differences from online collection:
      - deterministic policy (argmax) for reproducible eval
      - never stores episodes into the replay buffer (collect_tokenized=False)
      - prefixes metrics with `eval/` so they land at e.g. `eval/success_rate`,
        which is the key MBRLPipeline logs and prints (mbrl_train.py).
      - optionally runs on a fixed seed list (held-out test set) so success_rate
        is reproducible across updates and comparable to VAGEN's test set.

    The wrapped RealCollector is shared with online collection; we only flip the
    `deterministic` flag during evaluation and restore it afterwards.
    """

    def __init__(
        self,
        real_collector: RealCollector,
        eval_seeds: list[int] | None = None,
        eval_levels: list[dict] | None = None,
    ) -> None:
        self.collector = real_collector
        self.eval_seeds = eval_seeds
        # eval_levels (precomputed layouts) takes precedence over eval_seeds:
        # it gives 100% reproducible levels by injecting room_state directly,
        # bypassing generate_room RNG divergence (e.g. VagenMirror test set).
        self.eval_levels = eval_levels
        self._solution_lengths = (
            [_fixed_sokoban_solution_length(level) for level in eval_levels]
            if eval_levels is not None else None
        )
        self._reference_initial_logits: Tensor | None = None
        self._reference_initial_actions: Tensor | None = None

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        prev_det = self.collector.config.deterministic
        self.collector.config.deterministic = True
        try:
            kwargs: dict = {}
            if self.eval_levels is not None:
                # ``RealCollector.collect`` treats an explicit level list as
                # authoritative and ignores num_episodes, so slice here to
                # preserve Evaluator.evaluate(N)'s public contract.
                kwargs["levels"] = self.eval_levels[:num_episodes]
            elif self.eval_seeds is not None:
                kwargs["seeds"] = self.eval_seeds[:num_episodes]
            result = self.collector.collect(
                num_episodes,
                collect_tokenized=False,
                device=self.collector.device,
                dtype=self.collector.dtype,
                **kwargs,
            )
        finally:
            self.collector.config.deterministic = prev_det
        metrics = {f"eval/{k}": v for k, v in result.metrics.items()}
        if self._solution_lengths is not None:
            lengths = self._solution_lengths[:len(result.episodes)]
            for solution_length in sorted({
                value for value in lengths if value is not None
            }):
                selected = [
                    episode for episode, value in zip(
                        result.episodes, lengths, strict=True
                    ) if value == solution_length
                ]
                successes = sum(episode.success for episode in selected)
                metrics.update({
                    f"eval/num_levels_len_{solution_length}": float(len(selected)),
                    f"eval/num_success_len_{solution_length}": float(successes),
                    f"eval/success_rate_len_{solution_length}": (
                        successes / len(selected)
                    ),
                })
        logits_values = [
            episode.info.get("_actor_initial_logits")
            for episode in result.episodes
        ]
        action_values = [
            episode.info.get("_actor_initial_action")
            for episode in result.episodes
        ]
        if logits_values and all(value is not None for value in logits_values):
            logits = torch.stack(logits_values)
            actions = torch.tensor(action_values, dtype=torch.long)
            if self._reference_initial_logits is None:
                self._reference_initial_logits = logits.clone()
                self._reference_initial_actions = actions.clone()
            elif self._reference_initial_logits.shape == logits.shape:
                reference = self._reference_initial_logits
                reference_prob = reference.softmax(-1)
                current_prob = logits.softmax(-1)
                metrics.update({
                    "eval_actor/logit_delta_mean_abs": float(
                        (logits - reference).abs().mean()
                    ),
                    "eval_actor/kl_from_actor_start": float((
                        reference_prob * (
                            reference_prob.clamp_min(1e-8).log()
                            - current_prob.clamp_min(1e-8).log()
                        )
                    ).sum(-1).mean()),
                    "eval_actor/initial_action_flip_rate": float(
                        (actions != self._reference_initial_actions).float().mean()
                    ),
                })
            for action_id in range(4):
                metrics[f"eval_actor/initial_action_{action_id}_fraction"] = float(
                    (actions == action_id).float().mean()
                )
        return metrics

    @torch.no_grad()
    def evaluate_initial_policy(self, batch_size: int = 32) -> dict[str, float]:
        """Fast hard gate on the exact fixed-eval initial observations.

        This intentionally stops before any environment step: it audits the
        online RGB/tokenizer/posterior/Actor chain without paying for full
        episodes and catches the historical all-action-3 collapse before PPO.
        """
        if self.eval_levels is None:
            raise RuntimeError(
                "initial online Actor gate requires fixed eval levels"
            )
        frames = []
        for index, level in enumerate(self.eval_levels):
            env = self.collector.env_factory(index)
            frames.append(env.reset_with_room(
                level["room_state"], level["room_fixed"]
            ))
            close = getattr(env, "close", None)
            if callable(close):
                close()
        device = self.collector.device
        dtype = self.collector.dtype
        tokens = self.collector.tokenizer.batch_tokenize(frames).to(
            device=device, dtype=dtype
        )
        count = len(tokens)
        belief = self.collector.world_model.get_initial_belief(
            count, device=device, dtype=dtype
        )
        env_ids = self.collector.env_id_tensor
        if env_ids is not None and env_ids.numel() == 1 and count > 1:
            env_ids = env_ids.expand(count)
        belief = self.collector.world_model.posterior_step(
            prev_belief=belief,
            prev_actions=self.collector._get_null_action(
                count, device, dtype
            ),
            observation_tokens=tokens,
            env_ids=env_ids,
        )
        logits = []
        self.collector.policy.set_deterministic_forward_mode()
        for begin in range(0, count, batch_size):
            logits.append(self.collector.policy.actor_logits(
                belief.slots[begin:begin + batch_size]
            ).float().cpu())
        actions = torch.cat(logits).argmax(-1)
        counts = torch.bincount(actions, minlength=4)
        result = {
            f"online_actor_gate/action_{action}_fraction": float(
                counts[action].float() / count
            )
            for action in range(4)
        }
        result.update({
            "online_actor_gate/num_states": float(count),
            "online_actor_gate/num_predicted_actions": float(
                (counts > 0).sum()
            ),
            "online_actor_gate/max_action_fraction": float(
                counts.max().float() / count
            ),
        })
        return result
