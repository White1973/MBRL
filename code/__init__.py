"""Environment adapter interfaces.

Avoid importing concrete adapters at package import time so environment-specific
dependencies stay optional.
"""

from __future__ import annotations

from .base import EnvironmentAdapter, EnvProtocol, Policy, StepResult

try:
    from .base import GymnasiumEnvWrapper, wrap_gymnasium_env
except ImportError:  # pragma: no cover - compatibility with older remote checkouts
    GymnasiumEnvWrapper = None  # type: ignore[assignment]
    wrap_gymnasium_env = None  # type: ignore[assignment]


def make_default_adapter(env_id: str, **kwargs: object) -> EnvironmentAdapter:
    """Construct the default adapter for a known environment id.

    Imports stay local so optional environment dependencies remain lazy.
    """

    normalized = str(env_id).strip().lower()

    if normalized == "sokoban":
        from .sokoban import SokobanAdapter

        return SokobanAdapter(**kwargs)
    if normalized == "frozenlake":
        if kwargs:
            raise TypeError(f"Adapter kwargs are not supported for env_id={env_id!r}: {sorted(kwargs)}")
        from .frozenlake import FrozenLakeAdapter

        return FrozenLakeAdapter()
    if normalized == "minigrid":
        if kwargs:
            raise TypeError(f"Adapter kwargs are not supported for env_id={env_id!r}: {sorted(kwargs)}")
        from .minigrid import MiniGridAdapter

        return MiniGridAdapter()
    if normalized == "pendulum":
        if kwargs:
            raise TypeError(f"Adapter kwargs are not supported for env_id={env_id!r}: {sorted(kwargs)}")
        from .pendulum import PendulumAdapter

        return PendulumAdapter()
    if normalized == "maniskill":
        if kwargs:
            raise TypeError(f"Adapter kwargs are not supported for env_id={env_id!r}: {sorted(kwargs)}")
        from .maniskill import ManiSkillAdapter

        return ManiSkillAdapter()
    if normalized in {"habitat_pointnav", "pointnav"}:
        if kwargs:
            raise TypeError(f"Adapter kwargs are not supported for env_id={env_id!r}: {sorted(kwargs)}")
        from .habitat import HabitatAdapter

        return HabitatAdapter(task_type="pointnav")
    if normalized in {"habitat_objectnav", "objectnav"}:
        if kwargs:
            raise TypeError(f"Adapter kwargs are not supported for env_id={env_id!r}: {sorted(kwargs)}")
        from .habitat import HabitatAdapter

        return HabitatAdapter(task_type="objectnav")

    raise ValueError(f"Unsupported env_id for default adapter: {env_id!r}")

__all__ = [
    "EnvironmentAdapter",
    "EnvProtocol",
    "GymnasiumEnvWrapper",
    "Policy",
    "StepResult",
    "make_default_adapter",
    "wrap_gymnasium_env",
]
