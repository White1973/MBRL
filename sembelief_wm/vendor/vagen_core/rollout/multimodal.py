from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class PromptBundle:
    chat_messages: List[Dict[str, str]]
    images_by_turn: List[List[Image.Image]]
    prompt_text: str = ""
    flat_images: List[Image.Image] = field(default_factory=list)


class MultiTurnRolloutBuilder:
    """Minimal VAGEN-style multi-turn recorder and prompt builder.

    This keeps the parts that matter for zeroshot Sokoban:
    system -> user(obs) -> assistant(raw response) -> user(next obs) ...
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def record(self, env: Any, recorder: List[Dict[str, Any]], obs: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> None:
        entry: Dict[str, Any] = {
            "done": bool(done),
            "reward": float(reward),
            "info": info,
            "obs_str": obs["obs_str"],
        }
        image_placeholder = env.config.get("image_placeholder", "<image>")
        if "multi_modal_data" in obs and image_placeholder in obs["multi_modal_data"]:
            entry["image_data"] = list(obs["multi_modal_data"][image_placeholder])
        recorder.append(entry)

    def build_prompt_bundle(
        self,
        env: Any,
        recorder: List[Dict[str, Any]],
        step: int,
        window_size: Optional[int] = None,
    ) -> PromptBundle:
        start_step = max(0, step - window_size) if window_size is not None else 0
        history = recorder[start_step : step + 1]
        if len(history) != step + 1 - start_step:
            raise ValueError("Recording history is shorter than requested step window.")

        chat: List[Dict[str, str]] = [{"role": "system", "content": env.system_prompt()}]
        images_by_turn: List[List[Image.Image]] = []
        flat_images: List[Image.Image] = []
        for index, record in enumerate(history):
            if index > 0:
                raw_response = str(record["info"].get("llm_raw_response", "")).replace("<image>", "")
                chat.append({"role": "assistant", "content": raw_response})
            chat.append({"role": "user", "content": record["obs_str"]})
            turn_images = list(record.get("image_data", []))
            images_by_turn.append(turn_images)
            flat_images.extend(turn_images)

        prompt_text = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        return PromptBundle(
            chat_messages=chat,
            images_by_turn=images_by_turn,
            prompt_text=prompt_text,
            flat_images=flat_images,
        )
