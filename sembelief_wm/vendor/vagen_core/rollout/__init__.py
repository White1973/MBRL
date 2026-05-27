__all__ = ["MultiTurnRolloutBuilder"]


def __getattr__(name):
    if name == "MultiTurnRolloutBuilder":
        from sembelief_wm.vendor.vagen_core.rollout.multimodal import MultiTurnRolloutBuilder

        return MultiTurnRolloutBuilder
    raise AttributeError(name)
