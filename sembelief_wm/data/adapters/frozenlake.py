"""FrozenLake environment adapter for SemBelief-WM.

FrozenLake is the simplest sanity-check environment: a 4x4 grid with
4 discrete actions (left, down, right, up). The agent starts at (0,0)
and must reach the goal at (3,3) while avoiding holes.

This adapter supports two strategies:
- random: uniform random policy
- value_iteration_expert: optimal policy via dynamic programming
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import ActionSpaceSpec, ObservationSpaceSpec, Observation
from .base import EnvProtocol, EnvironmentAdapter, Policy


# ---------------------------------------------------------------------------
# Minimal FrozenLake env (no gymnasium dependency)
# ---------------------------------------------------------------------------

# Default 4x4 map from OpenAI Gym
_DEFAULT_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


class FrozenLakeEnv:
    """Deterministic FrozenLake 4x4 without gymnasium dependency."""

    def __init__(self, lake_map: list[str] | None = None, seed: int | None = None) -> None:
        self.map = lake_map or _DEFAULT_MAP
        self.nrow = len(self.map)
        self.ncol = len(self.map[0])
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self._rng = np.random.default_rng(seed)
        self.state = 0

        # Precompute goal and hole positions
        self._goals: set[int] = set()
        self._holes: set[int] = set()
        for r, row in enumerate(self.map):
            for c, ch in enumerate(row):
                s = r * self.ncol + c
                if ch == "G":
                    self._goals.add(s)
                elif ch == "H":
                    self._holes.add(s)

    def reset(self, *, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        row, col = divmod(self.state, self.ncol)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)

        self.state = row * self.ncol + col

        if self.state in self._goals:
            return self.state, 1.0, True, {"success": True}
        if self.state in self._holes:
            return self.state, 0.0, True, {"success": False}
        return self.state, 0.0, False, {}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class RandomPolicy:
    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        self._n_actions = n_actions
        self._rng = rng

    def __call__(self, observation: Observation) -> int:
        return int(self._rng.integers(0, self._n_actions))


class ValueIterationExpert:
    """Optimal policy via value iteration on the deterministic FrozenLake."""

    def __init__(self, env: FrozenLakeEnv) -> None:
        self._policy = self._solve(env)

    def __call__(self, observation: Observation) -> int:
        return self._policy.get(int(observation), 0)

    @staticmethod
    def _solve(env: FrozenLakeEnv, gamma: float = 0.99, n_iters: int = 200) -> dict[int, int]:
        n_states = env.n_states
        n_actions = env.n_actions
        values = np.zeros(n_states, dtype=np.float64)

        for _ in range(n_iters):
            new_values = np.zeros(n_states)
            for s in range(n_states):
                if s in env._goals or s in env._holes:
                    continue
                best_val = -np.inf
                for a in range(n_actions):
                    ns = ValueIterationExpert._next_state(env, s, a)
                    r = 1.0 if ns in env._goals else 0.0
                    val = r + gamma * values[ns]
                    if val > best_val:
                        best_val = val
                new_values[s] = best_val
            values = new_values

        policy: dict[int, int] = {}
        for s in range(n_states):
            if s in env._goals or s in env._holes:
                policy[s] = 0
                continue
            best_a, best_val = 0, -np.inf
            for a in range(n_actions):
                ns = ValueIterationExpert._next_state(env, s, a)
                r = 1.0 if ns in env._goals else 0.0
                val = r + gamma * values[ns]
                if val > best_val:
                    best_a, best_val = a, val
            policy[s] = best_a
        return policy

    @staticmethod
    def _next_state(env: FrozenLakeEnv, state: int, action: int) -> int:
        row, col = divmod(state, env.ncol)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, env.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, env.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return row * env.ncol + col


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FrozenLakeAdapter:
    """EnvironmentAdapter implementation for FrozenLake 4x4."""

    env_id: str = "frozenlake"

    def __init__(self, lake_map: list[str] | None = None) -> None:
        self._lake_map = lake_map or _DEFAULT_MAP
        self._n_states = len(self._lake_map) * len(self._lake_map[0])

    def make_env(self, seed: int | None = None) -> EnvProtocol:
        return FrozenLakeEnv(lake_map=self._lake_map, seed=seed)

    def action_spec(self) -> ActionSpaceSpec:
        return ActionSpaceSpec(type="discrete", num_actions=4)

    def observation_spec(self) -> ObservationSpaceSpec:
        return ObservationSpaceSpec(
            modality="discrete_state",
            shape=(self._n_states,),
            dtype="int64",
        )

    def available_strategies(self) -> list[str]:
        return ["random", "value_iteration_expert"]

    def make_policy(
        self,
        strategy: str,
        env: EnvProtocol,
        *,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Policy:
        rng = np.random.default_rng(seed)
        if strategy == "random":
            return RandomPolicy(4, rng)
        if strategy == "value_iteration_expert":
            if not isinstance(env, FrozenLakeEnv):
                raise TypeError("value_iteration_expert requires FrozenLakeEnv")
            return ValueIterationExpert(env)
        raise ValueError(f"Unknown strategy: {strategy!r}")

    def map_action(self, env_action: Any) -> int:
        return int(env_action)

    def action_to_text(self, action_id: int) -> str:
        texts = {
            LEFT: "move left",
            DOWN: "move down",
            RIGHT: "move right",
            UP: "move up",
        }
        if action_id not in texts:
            raise ValueError(f"Unsupported FrozenLake action_id: {action_id}")
        return texts[action_id]
