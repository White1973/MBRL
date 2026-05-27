"""Fixed step-based curriculum utilities for Phase 1 training."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class CurriculumState:
    """Resolved curriculum state for one global optimization step."""

    horizon: int
    stage_index: int
    stage_start_step: int


class CurriculumManager:
    """Pure step-based curriculum schedule.

    This intentionally has no metric-triggered state transitions.
    The active horizon is a pure function of `global_step`.
    """

    def __init__(self, config: Config) -> None:
        self.horizons = tuple(config.curriculum.horizons)
        self.switch_steps = tuple(config.curriculum.switch_steps)
        if len(self.horizons) != len(self.switch_steps):
            raise ValueError(
                "curriculum.horizons and curriculum.switch_steps must have the same length, "
                f"got {len(self.horizons)} and {len(self.switch_steps)}."
            )
        if not self.horizons:
            raise ValueError("curriculum.horizons must not be empty.")
        if self.switch_steps[0] != 0:
            raise ValueError(
                f"curriculum.switch_steps must start at 0, got {self.switch_steps[0]}."
            )
        if any(step_b <= step_a for step_a, step_b in zip(self.switch_steps, self.switch_steps[1:])):
            raise ValueError(
                "curriculum.switch_steps must be strictly increasing."
            )

    def state_at(self, global_step: int) -> CurriculumState:
        """Return the resolved curriculum state for `global_step`."""

        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}.")

        stage_index = 0
        for index, switch_step in enumerate(self.switch_steps):
            if global_step >= switch_step:
                stage_index = index
            else:
                break
        return CurriculumState(
            horizon=self.horizons[stage_index],
            stage_index=stage_index,
            stage_start_step=self.switch_steps[stage_index],
        )

    def horizon_at(self, global_step: int) -> int:
        """Convenience wrapper returning only the active horizon."""

        return self.state_at(global_step).horizon

    def did_switch(self, prev_step: int, current_step: int) -> bool:
        """Return whether the curriculum stage changed between two steps."""

        if current_step < prev_step:
            raise ValueError(
                f"current_step must be >= prev_step, got {prev_step} -> {current_step}."
            )
        return self.state_at(prev_step).stage_index != self.state_at(current_step).stage_index
