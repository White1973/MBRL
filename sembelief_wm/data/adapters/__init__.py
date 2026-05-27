"""Environment adapter interfaces and implementations."""

from .base import EnvironmentAdapter, EnvProtocol, Policy, StepResult
from .frozenlake import FrozenLakeAdapter
from .sokoban import SokobanAdapter


def make_default_adapter(env_id: str) -> EnvironmentAdapter:
    """Instantiate the default adapter for a known env id."""
    if env_id == "sokoban":
        return SokobanAdapter()
    if env_id == "frozenlake":
        return FrozenLakeAdapter()
    raise ValueError(
        f"No default adapter is registered for env_id={env_id!r}. "
        "Text action conditioning currently requires a known adapter."
    )

__all__ = [
    "EnvironmentAdapter",
    "EnvProtocol",
    "FrozenLakeAdapter",
    "make_default_adapter",
    "Policy",
    "SokobanAdapter",
    "StepResult",
]
