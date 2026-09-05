"""Sokoban environment adapter for SemBelief-WM.

Sokoban is a 6x6 grid puzzle with 1 box. The agent must push the box
onto the target position. This adapter provides four sampling strategies:

- expert: BFS solver, optimal path
- noisy_expert: BFS + 30% random noise, re-solve after each deviation
- random: uniform random policy
- near_success: expert until 3 steps from goal, then random

The env renders 96x96 RGB frames and has 4 discrete push actions
(up/down/left/right). Env actions are 1-4; model actions are 0-3.
"""
from __future__ import annotations

from collections import deque
from typing import Any
import random

import numpy as np

from ..schema import ActionSpaceSpec, ObservationSpaceSpec, Observation
from .base import EnvProtocol, EnvironmentAdapter, Policy


# ---------------------------------------------------------------------------
# BFS Solver
# ---------------------------------------------------------------------------

# Sokoban push directions: (dr, dc) for each env action (1=up,2=down,3=left,4=right)
_DIRECTIONS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _bfs_solve(
    player: tuple[int, int],
    boxes: frozenset[tuple[int, int]],
    targets: frozenset[tuple[int, int]],
    walls: frozenset[tuple[int, int]],
    dim_room: tuple[int, int],
    max_depth: int = 20,
) -> list[int] | None:
    """BFS solver for single-box Sokoban. Returns list of env actions (1-4) or None."""
    initial_state = (player, boxes)
    if boxes == targets:
        return []

    visited: set[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = {initial_state}
    queue: deque[tuple[tuple[int, int], frozenset[tuple[int, int]], list[int]]] = deque()
    queue.append((player, boxes, []))

    while queue:
        pos, bxs, path = queue.popleft()
        if len(path) >= max_depth:
            continue

        for action, (dr, dc) in _DIRECTIONS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            new_pos = (nr, nc)

            # Out of bounds or wall
            if nr < 0 or nr >= dim_room[0] or nc < 0 or nc >= dim_room[1]:
                continue
            if new_pos in walls:
                continue

            new_boxes = bxs
            if new_pos in bxs:
                # Push the box
                box_nr, box_nc = nr + dr, nc + dc
                box_new = (box_nr, box_nc)
                if box_nr < 0 or box_nr >= dim_room[0] or box_nc < 0 or box_nc >= dim_room[1]:
                    continue
                if box_new in walls or box_new in bxs:
                    continue
                new_boxes = (bxs - {new_pos}) | {box_new}

            state = (new_pos, new_boxes)
            if state in visited:
                continue
            visited.add(state)

            new_path = path + [action]
            if new_boxes == targets:
                return new_path
            queue.append((new_pos, new_boxes, new_path))

    return None


# ---------------------------------------------------------------------------
# Sokoban Env Wrapper
# ---------------------------------------------------------------------------

class SokobanEnv:
    """Minimal Sokoban wrapper around gym_sokoban.

    Requires gym-sokoban to be installed. Falls back to a mock env
    if gym_sokoban is not available (for testing).

    Every step/reset returns structured state labels in the info dict
    for downstream probing evaluation.
    """

    def __init__(
        self,
        dim_room: tuple[int, int] = (6, 6),
        num_boxes: int = 1,
        max_steps: int = 25,
        seed: int | None = None,
        bfs_max_depth: int = 20,
        require_real: bool = False,
    ) -> None:
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        self._seed = seed
        self._bfs_max_depth = bfs_max_depth
        self._env = None
        self._step_count = 0
        self._use_mock = False
        self._prev_boxes: frozenset[tuple[int, int]] | None = None
        self._prev_player: tuple[int, int] | None = None

        try:
            from gym_sokoban.envs import SokobanEnv as _GymSokobanEnv
            self._env = _GymSokobanEnv(
                dim_room=dim_room,
                num_boxes=num_boxes,
                max_steps=max_steps,
            )
            self._use_mock = False
        except Exception as exc:
            if require_real:
                raise RuntimeError(
                    "A real gym_sokoban environment is required for this "
                    "run, but gym_sokoban could not be imported/constructed. "
                    "Install the locked project dependencies with `uv sync`."
                ) from exc
            self._use_mock = True

    def _get_state_labels(self) -> dict[str, Any]:
        """Extract structured state labels for probing eval."""
        state = _extract_sokoban_state(self)
        if state is None:
            return {}
        player, boxes, targets, walls = state
        boxes_list = sorted(boxes)
        targets_list = sorted(targets)
        boxes_on_target = [b in targets for b in boxes_list]
        # Solvability check
        solution = _bfs_solve(player, boxes, targets, walls, self.dim_room, self._bfs_max_depth)
        return {
            "player_pos": player,
            "box_positions": boxes_list,
            "boxes_on_target": boxes_on_target,
            "num_boxes_on_target": sum(boxes_on_target),
            "is_solvable": solution is not None,
            "distance_to_goal": len(solution) if solution is not None else -1,
            "target_positions": targets_list,
        }

    def _compute_action_effect(self, player: tuple[int, int] | None, boxes: frozenset[tuple[int, int]] | None) -> str:
        """Compare prev and current state to determine action effect."""
        if self._prev_player is None or player is None or self._prev_boxes is None or boxes is None:
            return "unknown"
        if player == self._prev_player:
            return "noop"
        if boxes != self._prev_boxes:
            return "push"
        return "move"

    def reset(
        self,
        *,
        seed: int | None = None,
        min_solution_steps: tuple[int, int] | None = None,
        max_tries: int = 10000,
    ) -> np.ndarray:
        """Reset the environment.

        Args:
            seed: RNG seed for room generation.
            min_solution_steps: If given, reject levels whose shortest solution
                length is outside [lo, hi] and retry with a hash-derived seed
                (mirrors VagenMirror0416's PatchedSokobanEnv.reset filtering,
                so MBRL eval levels match VagenMirror0416's test set exactly).
            max_tries: Max retry attempts before giving up and using the last
                generated level.
        """
        s = seed if seed is not None else self._seed
        self._step_count = 0

        if self._use_mock:
            self._prev_player = None
            self._prev_boxes = None
            rng = np.random.default_rng(s)
            return rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)

        # VagenMirror0416-compatible generation: seed → generate_room → BFS
        # shortest solution → if length not in [lo,hi], hash-retry with new seed.
        cur_seed = s
        found = min_solution_steps is None
        for _ in range(max_tries):
            if cur_seed is not None:
                # gym_sokoban's generate_room uses BOTH the python `random`
                # module and np.random. Seed both to make level generation
                # deterministic (matches VAGEN/VagenMirror set_seed behavior).
                random.seed(cur_seed)
                np.random.seed(cur_seed)
            obs = self._env.reset()
            if min_solution_steps is None:
                found = True
                break
            state = _extract_sokoban_state(self)
            if state is not None:
                player, boxes, targets, walls = state
                sol = _bfs_solve(player, boxes, targets, walls, self.dim_room, self._bfs_max_depth)
                sol_len = len(sol) if sol is not None else -1
                lo, hi = min_solution_steps
                if lo <= sol_len <= hi:
                    found = True
                    break
            # else: state extraction failed or solution out of range → retry
            cur_seed = abs(hash(str(cur_seed))) % (2**32) if cur_seed is not None else None
        if not found:
            # Use the last generated level even if it didn't satisfy the filter.
            pass

        # Cache state for action_effect computation
        state = _extract_sokoban_state(self)
        if state is not None:
            self._prev_player = state[0]
            self._prev_boxes = state[1]
        else:
            self._prev_player = None
            self._prev_boxes = None

        return np.asarray(obs, dtype=np.uint8)

    def reset_with_room(self, room_state: np.ndarray, room_fixed: np.ndarray) -> np.ndarray:
        """Reset to a precomputed layout (skip generate_room entirely).

        Used to load VagenMirror0416's exported test levels so MBRL eval runs on
        the exact same levels as VagenMirror — bypassing any seed/reset/filter
        divergence. 100% reproducible.

        Args:
            room_state: (H, W) int array, gym_sokoban encoding (0=wall, 1=floor,
                2=target, 3=box-on-target, 4=box, 5=player, 6=player-on-target).
            room_fixed: (H, W) int array, the fixed walls/targets layout.
        """
        self._step_count = 0
        if self._use_mock or self._env is None:
            raise RuntimeError(
                "reset_with_room requires a real gym_sokoban env (got mock). "
                "Install gym_sokoban into the active environment."
            )
        self._env.room_state = np.array(room_state, dtype=np.int64)
        self._env.room_fixed = np.array(room_fixed, dtype=np.int64)
        # gym_sokoban uses 5 for player-on-floor and 6 for
        # player-on-target.  Counterfactual successors can legitimately place
        # the player on a target, so looking for code 5 alone makes a valid
        # room appear to contain no player.
        player_positions = np.argwhere(
            (self._env.room_state == 5) | (self._env.room_state == 6)
        )
        if len(player_positions) != 1:
            raise ValueError(
                "reset_with_room expects exactly one player encoded as 5 or 6, "
                f"found {len(player_positions)}"
            )
        self._env.player_position = player_positions[0]
        self._env.num_env_steps = 0
        self._env.reward_last = 0
        self._env.boxes_on_target = 0
        obs = self._env.render("rgb_array")

        state = _extract_sokoban_state(self)
        if state is not None:
            self._prev_player = state[0]
            self._prev_boxes = state[1]
        else:
            self._prev_player = None
            self._prev_boxes = None
        return np.asarray(obs, dtype=np.uint8)
        """Get episode-level metadata after reset (initial layout, optimal solution)."""
        room_state = self.room_state
        room_fixed = self.room_fixed
        meta: dict[str, Any] = {}
        if room_state is not None:
            meta["initial_room_state"] = room_state.copy()
        if room_fixed is not None:
            meta["room_fixed"] = room_fixed.copy()

        state = _extract_sokoban_state(self)
        if state is not None:
            player, boxes, targets, walls = state
            solution = _bfs_solve(player, boxes, targets, walls, self.dim_room, self._bfs_max_depth)
            meta["optimal_solution_length"] = len(solution) if solution is not None else -1
            meta["target_positions"] = sorted(targets)
            meta["wall_positions"] = sorted(walls)
        return meta

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self._step_count += 1

        if self._use_mock:
            rng = np.random.default_rng(self._seed * 1000 + self._step_count if self._seed else self._step_count)
            obs = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
            done = self._step_count >= self.max_steps
            return obs, -0.1, done, {"success": False}

        obs, reward, done, info = self._env.step(action)
        # Structured success judgment (aligned with VAGEN): all boxes on targets.
        # This is independent of reward magnitude, so it stays correct if the env's
        # reward shaping changes. gym_sokoban terminates with done=True when boxes
        # are all on targets, but we check the state explicitly to be robust.
        state = _extract_sokoban_state(self)
        if state is not None:
            _player, boxes, targets, _walls = state
            success = bool(boxes) and boxes <= targets
        else:
            # Fallback for mock/unavailable state: reuse the legacy reward-threshold rule.
            success = bool(done and reward > 1.0)
        info["success"] = success

        # Structured state labels
        state_labels = self._get_state_labels()
        state = _extract_sokoban_state(self)
        cur_player = state[0] if state else None
        cur_boxes = state[1] if state else None
        state_labels["action_effect"] = self._compute_action_effect(cur_player, cur_boxes)
        info["state_labels"] = state_labels

        # Update prev state
        self._prev_player = cur_player
        self._prev_boxes = cur_boxes

        return np.asarray(obs, dtype=np.uint8), float(reward), bool(done), info

    @property
    def room_state(self) -> np.ndarray | None:
        if self._use_mock:
            return None
        return getattr(self._env, "room_state", None)

    @property
    def room_fixed(self) -> np.ndarray | None:
        if self._use_mock:
            return None
        return getattr(self._env, "room_fixed", None)


def _extract_sokoban_state(
    env: SokobanEnv,
) -> tuple[
    tuple[int, int],
    frozenset[tuple[int, int]],
    frozenset[tuple[int, int]],
    frozenset[tuple[int, int]],
] | None:
    """Extract (player, boxes, targets, walls) from gym_sokoban internal state.

    gym_sokoban encoding (0 = wall/outside, 1 = floor, 2 = target):
      room_state: 0=wall, 1=floor, 2=target, 3=box on floor,
                  4=box on target, 5=player, 6=player on target
      room_fixed: 0=wall/outside, 1=floor, 2=target
    """
    room_state = env.room_state
    room_fixed = env.room_fixed
    if room_state is None or room_fixed is None:
        return None

    player = None
    boxes: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()

    for r in range(room_state.shape[0]):
        for c in range(room_state.shape[1]):
            cell = int(room_state[r, c])
            fixed = int(room_fixed[r, c])
            if cell in (5, 6):
                player = (r, c)
            if cell in (3, 4):
                boxes.add((r, c))
            if fixed == 2:
                targets.add((r, c))
            if fixed == 0:
                walls.add((r, c))

    if player is None:
        return None
    return player, frozenset(boxes), frozenset(targets), frozenset(walls)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class RandomSokobanPolicy:
    """Uniform random over 4 push actions (env actions 1-4)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def __call__(self, observation: Observation) -> int:
        return int(self._rng.integers(1, 5))


class ExpertPolicy:
    """BFS expert: solve once at init, execute the plan step by step.

    If env state extraction fails (e.g. mock env), falls back to random.
    """

    def __init__(
        self,
        env: SokobanEnv,
        *,
        max_depth: int = 20,
        rng: np.random.Generator,
    ) -> None:
        self._plan: list[int] = []
        self._step = 0
        self._fallback = RandomSokobanPolicy(rng)

        state = _extract_sokoban_state(env)
        if state is not None:
            player, boxes, targets, walls = state
            solution = _bfs_solve(player, boxes, targets, walls, env.dim_room, max_depth)
            if solution is not None:
                self._plan = solution

    def __call__(self, observation: Observation) -> int:
        if self._step < len(self._plan):
            action = self._plan[self._step]
            self._step += 1
            return action
        return self._fallback(observation)


class NoisyExpertPolicy:
    """BFS expert with noise_prob chance of random action at each step.

    After a random action, re-solves BFS from the new state. If BFS fails
    (dead state), continues with random until episode ends.
    """

    def __init__(
        self,
        env: SokobanEnv,
        *,
        noise_prob: float = 0.3,
        max_depth: int = 20,
        rng: np.random.Generator,
    ) -> None:
        self._env = env
        self._noise_prob = noise_prob
        self._max_depth = max_depth
        self._rng = rng
        self._plan: list[int] = []
        self._plan_step = 0
        self._dead = False
        self._random = RandomSokobanPolicy(rng)

        self._replan()

    def _replan(self) -> None:
        state = _extract_sokoban_state(self._env)
        if state is None:
            self._dead = True
            return
        player, boxes, targets, walls = state
        solution = _bfs_solve(player, boxes, targets, walls, self._env.dim_room, self._max_depth)
        if solution is None:
            self._dead = True
        else:
            self._plan = solution
            self._plan_step = 0

    def __call__(self, observation: Observation) -> int:
        # Re-plan if previous action was random (plan was cleared)
        if not self._plan and not self._dead:
            self._replan()

        if self._dead or not self._plan:
            return self._random(observation)

        if self._rng.random() < self._noise_prob:
            # Random action — next call will trigger replan
            self._plan = []
            self._plan_step = 0
            return self._random(observation)

        if self._plan_step < len(self._plan):
            action = self._plan[self._plan_step]
            self._plan_step += 1
            return action
        return self._random(observation)


class NearSuccessPolicy:
    """Expert until remaining_steps from goal, then random.

    If BFS solution has <= remaining_steps, immediately goes random
    (already near success from the start).
    """

    def __init__(
        self,
        env: SokobanEnv,
        *,
        remaining_steps: int = 3,
        max_depth: int = 20,
        rng: np.random.Generator,
    ) -> None:
        self._remaining_steps = remaining_steps
        self._rng = rng
        self._random = RandomSokobanPolicy(rng)
        self._plan: list[int] = []
        self._step = 0
        self._switch_point = 0

        state = _extract_sokoban_state(env)
        if state is not None:
            player, boxes, targets, walls = state
            solution = _bfs_solve(player, boxes, targets, walls, env.dim_room, max_depth)
            if solution is not None:
                self._plan = solution
                # Execute expert until remaining_steps from end
                self._switch_point = max(0, len(solution) - remaining_steps)

    def __call__(self, observation: Observation) -> int:
        if self._step < self._switch_point:
            action = self._plan[self._step]
            self._step += 1
            return action
        return self._random(observation)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SokobanAdapter:
    """EnvironmentAdapter implementation for Sokoban.

    Supports four strategies: expert, noisy_expert, random, near_success.
    """

    env_id: str = "sokoban"

    def __init__(
        self,
        dim_room: tuple[int, int] = (6, 6),
        num_boxes: int = 1,
        max_steps: int = 25,
        bfs_max_depth: int = 20,
        require_real: bool = False,
    ) -> None:
        self._dim_room = dim_room
        self._num_boxes = num_boxes
        self._max_steps = max_steps
        self._bfs_max_depth = bfs_max_depth
        self._require_real = require_real

    def make_env(self, seed: int | None = None) -> EnvProtocol:
        return SokobanEnv(
            dim_room=self._dim_room,
            num_boxes=self._num_boxes,
            max_steps=self._max_steps,
            seed=seed,
            bfs_max_depth=self._bfs_max_depth,
            require_real=self._require_real,
        )

    def action_spec(self) -> ActionSpaceSpec:
        return ActionSpaceSpec(type="discrete", num_actions=4)

    def observation_spec(self) -> ObservationSpaceSpec:
        return ObservationSpaceSpec(
            modality="rgb",
            shape=(96, 96, 3),
            dtype="uint8",
        )

    def available_strategies(self) -> list[str]:
        return ["expert", "noisy_expert", "random", "near_success"]

    def make_policy(
        self,
        strategy: str,
        env: EnvProtocol,
        *,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Policy:
        rng = np.random.default_rng(seed)
        params = params or {}

        if not isinstance(env, SokobanEnv):
            raise TypeError(f"Sokoban policies require SokobanEnv, got {type(env)}")

        if strategy == "random":
            return RandomSokobanPolicy(rng)
        if strategy == "expert":
            return ExpertPolicy(
                env,
                max_depth=params.get("bfs_max_depth", self._bfs_max_depth),
                rng=rng,
            )
        if strategy == "noisy_expert":
            return NoisyExpertPolicy(
                env,
                noise_prob=params.get("noise_prob", 0.3),
                max_depth=params.get("bfs_max_depth", self._bfs_max_depth),
                rng=rng,
            )
        if strategy == "near_success":
            return NearSuccessPolicy(
                env,
                remaining_steps=params.get("remaining_steps", 3),
                max_depth=params.get("bfs_max_depth", self._bfs_max_depth),
                rng=rng,
            )
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Available: {self.available_strategies()}"
        )

    def map_action(self, env_action: Any) -> int:
        """Map Sokoban env action (1-4) to model action (0-3)."""
        return int(env_action) - 1

    def action_to_text(self, action_id: int) -> str:
        texts = {
            0: "push up",
            1: "push down",
            2: "push left",
            3: "push right",
        }
        if action_id not in texts:
            raise ValueError(f"Unsupported Sokoban action_id: {action_id}")
        return texts[action_id]
