__all__ = ["VLLMMultiModalModel", "VLLMRunnerConfig"]


def __getattr__(name):
    if name in {"VLLMMultiModalModel", "VLLMRunnerConfig"}:
        from sembelief_wm.vendor.vagen_core.model.vllm_runner import (
            VLLMMultiModalModel,
            VLLMRunnerConfig,
        )

        return {
            "VLLMMultiModalModel": VLLMMultiModalModel,
            "VLLMRunnerConfig": VLLMRunnerConfig,
        }[name]
    raise AttributeError(name)
