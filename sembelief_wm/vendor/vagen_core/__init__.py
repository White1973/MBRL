__all__ = ["SokobanEnv", "SokobanEnvConfig"]


def __getattr__(name):
    if name == "SokobanEnv":
        from sembelief_wm.vendor.vagen_core.env.sokoban.env import SokobanEnv

        return SokobanEnv
    if name == "SokobanEnvConfig":
        from sembelief_wm.vendor.vagen_core.env.sokoban.env_config import SokobanEnvConfig

        return SokobanEnvConfig
    raise AttributeError(name)
