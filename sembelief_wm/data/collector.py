"""Generic episode collection with pluggable strategy scheduling."""
from __future__ import annotations

from itertools import cycle
from typing import Any

from .adapters import EnvironmentAdapter
from .schema import CollectorConfig, RawEpisode, StrategySpec


def strategy_schedule(strategies: list[StrategySpec]) -> list[StrategySpec]:
    """Expand strategy weights into one deterministic round-robin cycle."""

    if not strategies:
        raise ValueError("strategy_schedule requires at least one strategy.")
    schedule: list[StrategySpec] = []
    for spec in strategies:
        schedule.extend([spec] * spec.weight)
    return schedule


def validate_strategies(adapter: EnvironmentAdapter, config: CollectorConfig) -> None:
    """Fail fast when a requested strategy is unsupported by the adapter."""

    available = set(adapter.available_strategies())
    requested = {spec.name for spec in config.strategies}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"Adapter {adapter.env_id!r} does not support strategies {missing}; "
            f"available strategies are {sorted(available)}."
        )


def collect_episodes(
    adapter: EnvironmentAdapter,
    config: CollectorConfig,
) -> list[RawEpisode]:
    """Collect episodes according to a weighted strategy schedule."""

    validate_strategies(adapter, config)
    episodes: list[RawEpisode] = []
    for index, strategy in zip(
        range(config.num_episodes),
        cycle(strategy_schedule(config.strategies)),
        strict=False,
    ):
        seed = config.seed + index
        episodes.append(
            collect_one_episode(
                adapter=adapter,
                strategy=strategy,
                max_steps=config.max_steps,
                seed=seed,
                split=config.split,
            )
        )
    return episodes


def collect_one_episode(
    *,
    adapter: EnvironmentAdapter,
    strategy: StrategySpec,
    max_steps: int,
    seed: int,
    split: str,
) -> RawEpisode:
    """Collect one episode from a single adapter and strategy."""

    env = adapter.make_env(seed=seed)
    obs = env.reset(seed=seed)

    # Collect episode-level metadata from env (e.g. initial room layout, optimal solution)
    initial_meta: dict[str, Any] = {}
    if hasattr(env, "get_initial_metadata"):
        initial_meta = env.get_initial_metadata()

    # Also collect initial state labels (before any action)
    initial_state_labels: dict[str, Any] = {}
    if hasattr(env, "_get_state_labels"):
        initial_state_labels = env._get_state_labels()

    policy = adapter.make_policy(
        strategy.name,
        env,
        params=strategy.params,
        seed=seed,
    )

    observations = [obs]
    actions: list[Any] = []
    model_actions: list[Any] = []
    rewards: list[float] = []
    dones: list[bool] = []
    infos: list[dict[str, Any]] = []

    for _ in range(max_steps):
        env_action = policy(obs)
        next_obs, reward, done, info = env.step(env_action)
        observations.append(next_obs)
        actions.append(env_action)
        model_actions.append(adapter.map_action(env_action))
        rewards.append(float(reward))
        dones.append(bool(done))
        infos.append(dict(info))
        obs = next_obs
        if done:
            break

    # Determine success from terminal info or done+reward heuristic
    success = False
    if infos and infos[-1].get("success"):
        success = True
    elif dones and dones[-1] and rewards and rewards[-1] > 0:
        success = True

    return RawEpisode(
        env_id=adapter.env_id,
        observations=observations,
        actions=actions,
        model_actions=model_actions,
        rewards=rewards,
        dones=dones,
        infos=infos,
        metadata={
            "strategy": strategy.name,
            "strategy_params": strategy.params,
            "seed": seed,
            "split": split,
            "success": success,
            "initial_state_labels": initial_state_labels,
            **initial_meta,
        },
    )
