from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class VLLMRunnerConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    enforce_eager: bool = False
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    seed: int = 0
    max_images_per_prompt: int = 4


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class VLLMMultiModalModel:
    """vLLM backend using VAGEN-style prompt text + generate()."""

    def __init__(self, config: VLLMRunnerConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=config.trust_remote_code,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            limit_mm_per_prompt={"image": config.max_images_per_prompt},
        )

    @staticmethod
    def format_vllm_prompt(prompt_text: str) -> str:
        return prompt_text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

    def generate(
        self,
        prompt_batches: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p if top_p is None else top_p,
            top_k=self.config.top_k if top_k is None else top_k,
            # No seed — random sampling matches native VAGEN behavior.
            # seed=0 forces a deterministic path that breaks XML format.
        )

        prompts: List[Dict[str, Any]] = []
        for batch_item in prompt_batches:
            prompt_text = self.format_vllm_prompt(batch_item["prompt_text"])
            prompt_obj: Dict[str, Any] = {"prompt": prompt_text}
            images = list(batch_item.get("images", []))
            if images:
                prompt_obj["multi_modal_data"] = {"image": images}
            prompts.append(prompt_obj)

        outputs = self.model.generate(prompts, sampling_params)

        results: List[Dict[str, Any]] = []
        for output in outputs:
            completion = output.outputs[0]
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(completion.token_ids)
            results.append(
                {
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        return results
