"""Base protocols for environment adapters."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..schema import ActionSpaceSpec, Observation, ObservationSpaceSpec

StepResult = tuple[Observation, float, bool, dict[str, Any]]


@runtime_checkable
class EnvProtocol(Protocol):
    """Project-local minimal environment handle.

    Concrete adapters may wrap Gym, Gymnasium, ALFWorld, or custom embodied
    environments behind this interface.
    """

    def reset(self, *, seed: int | None = None) -> Observation: ...

    def step(self, action: Any) -> StepResult: ...


class Policy(Protocol):
    """Callable policy used by collectors."""

    def __call__(self, observation: Observation) -> Any: ...


class EnvironmentAdapter(Protocol):
    """Adapter from a concrete environment family to SemBelief-WM data contracts."""

    env_id: str

    def make_env(self, seed: int | None = None) -> EnvProtocol: ...

    def action_spec(self) -> ActionSpaceSpec: ...

    def observation_spec(self) -> ObservationSpaceSpec: ...

    def available_strategies(self) -> list[str]: ...

    def make_policy(
        self,
        strategy: str,
        env: EnvProtocol,
        *,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Policy: ...

    def map_action(self, env_action: Any) -> int:
        """Map a concrete env action to the global discrete model action id."""

    def action_to_text(self, action_id: int) -> str:
        """Render a model action id as a natural-language action string."""
