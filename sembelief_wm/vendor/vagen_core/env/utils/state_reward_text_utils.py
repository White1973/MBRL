from __future__ import annotations


def env_state_reward_wrapper(step_func):
    """Minimal local wrapper.

    The native VAGEN env decorates `step` so grounding/world-model process rewards
    can be attached later via an external judge service. The internal vendored
    zeroshot path does not use that service, so we keep the decorator surface but
    make it a no-op unless someone explicitly enables `use_state_reward`.
    """

    def wrapped_step(self, action_str):
        if hasattr(self, "config") and self.config.get("use_state_reward", False):
            raise NotImplementedError(
                "Vendored vagen_core does not support external state-reward judges."
            )
        return step_func(self, action_str)

    return wrapped_step
