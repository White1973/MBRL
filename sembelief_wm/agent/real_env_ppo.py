"""Real-environment PPO baseline with Qwen2.5-VL freeform action generation.

This baseline removes the world model entirely:

- input: raw RGB observations from the real environment
- policy: Qwen2.5-VL + freeform text action generation
- reward: real environment reward
- update: PPO on generated response tokens
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from PIL import Image
except ImportError:  # pragma: no cover - remote env may omit Pillow
    Image = None

from ..config import Config
from ..data.adapters import EnvironmentAdapter, make_default_adapter


class RealEnvLogger(Protocol):
    """Minimal logger contract for real-environment PPO."""

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        """Record scalar metrics for one PPO update."""


def _normalize_action_text(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    return normalized.strip(" \t\r\n.,;:!?\"'`()[]{}")


@dataclass(frozen=True)
class RealEnvPolicyStep:
    """Policy outputs for one environment step."""

    actions: Tensor
    values: Tensor
    response_token_ids: Tensor
    response_mask: Tensor
    token_log_probs: Tensor
    token_entropies: Tensor
    parse_success: Tensor
    raw_texts: list[str]
    parsed_texts: list[str]
    action_texts: list[str | None]
    action_sequences: list[list[str]]
    prompt_texts: list[str]


@dataclass(frozen=True)
class RealEnvRolloutBatch:
    """Flattened real-environment PPO batch."""

    observations: list[Any]
    env_ids: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    returns: Tensor
    advantages: Tensor
    response_token_ids: Tensor
    response_mask: Tensor
    old_token_log_probs: Tensor
    parse_success: Tensor
    response_lengths: Tensor
    trajectory_ids: Tensor
    step_ids: Tensor
    raw_texts: list[str]
    parsed_texts: list[str]
    prompt_texts: list[str]


class _ActionParser:
    """Parse generated text into legal discrete actions."""

    def __init__(
        self,
        *,
        answer_open_tag: str,
        answer_close_tag: str,
        action_texts_by_env: list[list[str]],
        action_separator: str,
        max_actions_per_turn: int,
    ) -> None:
        open_tag = re.escape(answer_open_tag)
        close_tag = re.escape(answer_close_tag)
        self._pattern = re.compile(f"{open_tag}(.*?){close_tag}", flags=re.IGNORECASE | re.DOTALL)
        self._separator = action_separator
        self._max_actions = max_actions_per_turn
        self._normalized_actions = [
            [_normalize_action_text(text) for text in env_texts]
            for env_texts in action_texts_by_env
        ]

    def parse_with_success(
        self,
        texts: list[str],
        env_ids: Tensor,
    ) -> tuple[Tensor, Tensor, list[str], list[str | None], list[list[str]]]:
        parsed_actions: list[int] = []
        matched: list[bool] = []
        parsed_texts: list[str] = []
        action_texts: list[str | None] = []
        action_sequences: list[list[str]] = []

        for index, raw_text in enumerate(texts):
            env_index = int(env_ids[index].item())
            normalized_actions = self._normalized_actions[env_index]
            candidate, has_answer_tag = self._extract_answer(raw_text)
            parsed_texts.append(candidate)
            action_ids, normalized_sequence = self._match_action_sequence(candidate, normalized_actions)
            is_match = has_answer_tag and len(action_ids) > 0
            if not action_ids:
                action_id = 0
                action_texts.append(None)
            else:
                action_id = action_ids[0]
                action_texts.append(self._separator.join(normalized_sequence))
            parsed_actions.append(action_id)
            matched.append(is_match)
            action_sequences.append(normalized_sequence)

        return (
            torch.tensor(parsed_actions, dtype=torch.long, device=env_ids.device),
            torch.tensor(matched, dtype=torch.bool, device=env_ids.device),
            parsed_texts,
            action_texts,
            action_sequences,
        )

    def _extract_answer(self, raw_text: str) -> tuple[str, bool]:
        match = self._pattern.search(raw_text)
        if match is not None:
            return _normalize_action_text(match.group(1)), True
        return _normalize_action_text(raw_text), False

    def _match_action_sequence(
        self,
        candidate: str,
        normalized_actions: list[str],
    ) -> tuple[list[int], list[str]]:
        if not candidate:
            return [], []
        chunks = [
            _normalize_action_text(part)
            for part in candidate.split(self._separator)
            if _normalize_action_text(part)
        ]
        if not chunks:
            chunks = [_normalize_action_text(candidate)]
        action_ids: list[int] = []
        normalized_sequence: list[str] = []
        for chunk in chunks[: self._max_actions]:
            action_id = self._match_single_action(chunk, normalized_actions)
            if action_id is None:
                break
            action_ids.append(action_id)
            normalized_sequence.append(normalized_actions[action_id])
        return action_ids, normalized_sequence

    @staticmethod
    def _match_single_action(candidate: str, normalized_actions: list[str]) -> int | None:
        for action_id, action_text in enumerate(normalized_actions):
            if candidate == action_text:
                return action_id
        for action_id, action_text in enumerate(normalized_actions):
            if action_text in candidate or candidate in action_text:
                return action_id
        matches = difflib.get_close_matches(candidate, normalized_actions, n=1, cutoff=0.6)
        if matches:
            return normalized_actions.index(matches[0])
        return None


def _import_vlm_dependencies() -> tuple[Any, Any | None]:
    try:
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Real-env VLM PPO requires optional VLM dependencies. "
            "Install them with `uv sync --extra vlm`."
        ) from exc
    try:
        import peft  # type: ignore[import-not-found]
    except ImportError:
        peft = None
    return transformers, peft


class _ManualQwenVLProcessor:
    """Lightweight Qwen-VL processor that bypasses AutoProcessor (no torchvision needed).

    Combines ``Qwen2VLImageProcessor`` for vision and ``AutoTokenizer`` for text,
    replicating the two call-sites used by ``RGBFreeformPolicy``:
    ``apply_chat_template(...)`` and ``__call__(images=..., text=..., ...)``.
    """

    IMAGE_PAD = "<|image_pad|>"

    def __init__(self, tokenizer: Any, image_processor: Any) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    # -- public API used by RGBFreeformPolicy ----------------------------------

    def apply_chat_template(
        self,
        conversations: list,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> list[str] | list[list[int]]:
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(
        self,
        *,
        images: list | None = None,
        text: list[str] | str | None = None,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict:
        import torch

        # 1. Process images → pixel_values, image_grid_thw
        img_out = self.image_processor(images=images, return_tensors="pt") if images else {}

        # 2. Expand <|image_pad|> placeholders to match actual image token counts.
        #    We must use sentinels to avoid re-expanding already-expanded pads.
        if isinstance(text, str):
            text = [text]
        expanded_texts: list[str] = []
        grid_thw = img_out.get("image_grid_thw")  # (N_images, 3)
        merge_length = int(getattr(self.image_processor, "merge_size", 1) ** 2)
        img_idx = 0
        _SENTINEL = "\x00IMG_EXPANDED\x00"
        for t in text:
            result = t
            # First pass: replace each single <|image_pad|> with a unique sentinel
            # carrying the correct repeat count.
            while self.IMAGE_PAD in result and grid_thw is not None and img_idx < grid_thw.shape[0]:
                g = grid_thw[img_idx]
                n_tokens = int(g[0].item() * g[1].item() * g[2].item()) // merge_length
                result = result.replace(self.IMAGE_PAD, f"{_SENTINEL}{n_tokens}{_SENTINEL}", 1)
                img_idx += 1
            # Second pass: expand sentinels to actual repeated pad tokens.
            import re as _re
            def _expand_sentinel(m: "_re.Match[str]") -> str:
                return self.IMAGE_PAD * int(m.group(1))
            result = _re.sub(
                _re.escape(_SENTINEL) + r"(\d+)" + _re.escape(_SENTINEL),
                _expand_sentinel,
                result,
            )
            expanded_texts.append(result)

        # 3. Tokenize
        tok_out = self.tokenizer(
            expanded_texts,
            padding=padding,
            return_tensors=return_tensors,
        )

        # 4. Merge
        batch: dict = dict(tok_out)
        if "pixel_values" in img_out:
            batch["pixel_values"] = img_out["pixel_values"]
        if "image_grid_thw" in img_out:
            batch["image_grid_thw"] = img_out["image_grid_thw"]
        return batch


def _build_qwen_processor(transformers: Any, model_name: str) -> Any:
    """Load a Qwen VL processor, falling back to a manual shim if torchvision is missing."""
    try:
        return transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
        return _ManualQwenVLProcessor(tokenizer=tokenizer, image_processor=image_processor)


class RGBFreeformPolicy(nn.Module):
    """Qwen2.5-VL freeform policy on raw RGB observations."""

    def __init__(self, config: Config, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.ff_cfg = config.phase2.freeform_policy
        self.env_ids = config.env.env_ids

        transformers, peft = _import_vlm_dependencies()
        model_name = self.ff_cfg.model_name or config.backbone.model_name

        self.processor = _build_qwen_processor(transformers, model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if config.training.dtype == "bf16" else torch.float32,
            "trust_remote_code": True,
        }
        if device is not None:
            model_kwargs["device_map"] = {"": str(device)}

        qwen_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")
        self.model = qwen_cls.from_pretrained(model_name, **model_kwargs)
        if peft is not None:
            lora_config = peft.LoraConfig(
                r=self.ff_cfg.lora_rank,
                lora_alpha=self.ff_cfg.lora_alpha,
                lora_dropout=self.ff_cfg.lora_dropout,
                target_modules=self.ff_cfg.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = peft.get_peft_model(self.model, lora_config)
        self.model.config.use_cache = True

        self.value_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 1),
        )

        self.action_texts_by_env = self._build_action_texts()
        self.system_prompts_by_env = self._build_system_prompts()
        self.prompts_by_env = self._build_prompts()
        self.parser = _ActionParser(
            answer_open_tag=self.ff_cfg.answer_open_tag,
            answer_close_tag=self.ff_cfg.answer_close_tag,
            action_texts_by_env=self.action_texts_by_env,
            action_separator=self.ff_cfg.action_separator,
            max_actions_per_turn=self.ff_cfg.max_actions_per_turn,
        )

    def act(
        self,
        observations: list[Any],
        env_ids: Tensor,
        *,
        deterministic: bool = False,
        prompt_texts: list[str] | None = None,
        conversations: list[list[dict[str, Any]]] | None = None,
    ) -> RealEnvPolicyStep:
        prompt_batch = self._prepare_prompt_batch(
            observations,
            env_ids,
            prompt_texts=prompt_texts,
            conversations=conversations,
        )
        values = self._value_from_prompt_batch(prompt_batch)
        generated_ids, texts = self._generate(prompt_batch, deterministic=deterministic)
        response_token_ids, response_mask = self._pack_generated_tokens(generated_ids, prompt_batch["prompt_lengths"])
        token_log_probs, token_entropies = self.response_token_stats(
            observations,
            env_ids,
            prompt_texts=prompt_batch["prompt_texts"],
            conversations=conversations,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
        )
        actions, parse_success, parsed_texts, action_texts, action_sequences = self.parser.parse_with_success(texts, env_ids)
        return RealEnvPolicyStep(
            actions=actions,
            values=values,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
            token_log_probs=token_log_probs,
            token_entropies=token_entropies,
            parse_success=parse_success,
            raw_texts=texts,
            parsed_texts=parsed_texts,
            action_texts=action_texts,
            action_sequences=action_sequences,
            prompt_texts=prompt_batch["prompt_texts"],
        )

    def evaluate_actions(
        self,
        observations: list[Any],
        env_ids: Tensor,
        *,
        prompt_texts: list[str],
        conversations: list[list[dict[str, Any]]] | None = None,
        response_token_ids: Tensor,
        response_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values = self._value_from_prompt_batch(
            self._prepare_prompt_batch(
                observations,
                env_ids,
                prompt_texts=prompt_texts,
                conversations=conversations,
            )
        )
        token_log_probs, token_entropies = self.response_token_stats(
            observations,
            env_ids,
            prompt_texts=prompt_texts,
            conversations=conversations,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
        )
        return token_log_probs, token_entropies, values

    def response_token_stats(
        self,
        observations: list[Any],
        env_ids: Tensor,
        *,
        prompt_texts: list[str],
        conversations: list[list[dict[str, Any]]] | None = None,
        response_token_ids: Tensor,
        response_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prompt_batch = self._prepare_prompt_batch(
            observations,
            env_ids,
            prompt_texts=prompt_texts,
            conversations=conversations,
        )
        inputs = self._concat_prompt_and_response(
            prompt_batch=prompt_batch,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
        )
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        token_log_probs = torch.zeros_like(response_token_ids, dtype=logits.dtype)
        token_entropies = torch.zeros_like(response_token_ids, dtype=logits.dtype)
        batch_size = response_token_ids.shape[0]
        for i in range(batch_size):
            prompt_len = int(prompt_batch["prompt_lengths"][i].item())
            resp_len = int(response_mask[i].sum().item())
            if resp_len == 0:
                continue
            token_logits = logits[i, prompt_len - 1:prompt_len + resp_len - 1, :]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            token_ids = response_token_ids[i, :resp_len]
            gathered = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            token_log_probs[i, :resp_len] = gathered
            token_entropies[i, :resp_len] = entropy
        mask_f = response_mask.to(dtype=token_log_probs.dtype)
        return token_log_probs * mask_f, token_entropies * mask_f

    def _value_from_prompt_batch(self, prompt_batch: dict[str, Tensor]) -> Tensor:
        outputs = self.model(
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            pixel_values=prompt_batch.get("pixel_values"),
            image_grid_thw=prompt_batch.get("image_grid_thw"),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        batch_size = hidden.shape[0]
        prompt_lengths = prompt_batch["prompt_lengths"]
        pooled = []
        for i in range(batch_size):
            last_idx = int(prompt_lengths[i].item()) - 1
            pooled.append(hidden[i, last_idx, :])
        pooled_hidden = torch.stack(pooled, dim=0).to(dtype=next(self.value_head.parameters()).dtype)
        return self.value_head(pooled_hidden).squeeze(-1)

    def _generate(
        self,
        prompt_batch: dict[str, Tensor],
        *,
        deterministic: bool,
    ) -> tuple[Tensor, list[str]]:
        generation_kwargs = {
            "max_new_tokens": self.ff_cfg.max_new_tokens,
            "do_sample": False if deterministic else self.ff_cfg.do_sample,
            "temperature": None if deterministic else self.ff_cfg.temperature,
            "top_p": None if deterministic else self.ff_cfg.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        cleaned_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        sequences = self.model.generate(
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            pixel_values=prompt_batch.get("pixel_values"),
            image_grid_thw=prompt_batch.get("image_grid_thw"),
            **cleaned_kwargs,
        )
        prompt_lengths = prompt_batch["prompt_lengths"]
        responses: list[Tensor] = []
        texts: list[str] = []
        for i in range(sequences.shape[0]):
            prompt_len = int(prompt_lengths[i].item())
            response_ids = sequences[i, prompt_len:]
            responses.append(response_ids)
            texts.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        return sequences, texts

    def _pack_generated_tokens(
        self,
        sequences: Tensor,
        prompt_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        responses: list[Tensor] = []
        max_len = 1
        for i in range(sequences.shape[0]):
            response_ids = sequences[i, int(prompt_lengths[i].item()):]
            if response_ids.numel() == 0:
                response_ids = torch.full(
                    (1,),
                    fill_value=self.tokenizer.pad_token_id or 0,
                    device=sequences.device,
                    dtype=torch.long,
                )
                mask = torch.zeros(1, device=sequences.device, dtype=torch.long)
            else:
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    eos_pos = (response_ids == eos_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        response_ids = response_ids[:int(eos_pos[0].item()) + 1]
                mask = torch.ones(response_ids.shape[0], device=sequences.device, dtype=torch.long)
            max_len = max(max_len, response_ids.shape[0])
            responses.append((response_ids, mask))

        padded_ids: list[Tensor] = []
        padded_masks: list[Tensor] = []
        pad_id = self.tokenizer.pad_token_id or 0
        for response_ids, mask in responses:
            pad = max_len - response_ids.shape[0]
            if pad > 0:
                response_ids = F.pad(response_ids, (0, pad), value=pad_id)
                mask = F.pad(mask, (0, pad), value=0)
            padded_ids.append(response_ids)
            padded_masks.append(mask)
        return torch.stack(padded_ids, dim=0), torch.stack(padded_masks, dim=0)

    def _prepare_prompt_batch(
        self,
        observations: list[Any],
        env_ids: Tensor,
        *,
        prompt_texts: list[str] | None = None,
        conversations: list[list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        if conversations is None:
            resolved_prompt_texts = []
            built_conversations: list[list[dict[str, Any]]] = []
            for batch_idx, env_idx in enumerate(env_ids.tolist()):
                prompt = self.prompts_by_env[int(env_idx)] if prompt_texts is None else prompt_texts[batch_idx]
                resolved_prompt_texts.append(prompt)
                built_conversations.append(
                    [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": self.system_prompts_by_env[int(env_idx)]}],
                        },
                        {
                            "role": "user",
                            "content": self._prompt_to_content(prompt, self._to_pil_image(observations[batch_idx])),
                        },
                    ]
                )
            conversations = built_conversations
        else:
            resolved_prompt_texts = list(prompt_texts or [""] * len(conversations))

        template_conversations, flat_images = self._serialize_conversations(conversations)
        texts = self.processor.apply_chat_template(
            template_conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = self.processor(
            images=flat_images,
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        device = self._model_device()
        prompt_lengths = batch["attention_mask"].sum(dim=1).to(device=device, dtype=torch.long)
        prompt_batch: dict[str, Tensor] = {
            "input_ids": batch["input_ids"].to(device=device, dtype=torch.long),
            "attention_mask": batch["attention_mask"].to(device=device, dtype=torch.long),
            "prompt_lengths": prompt_lengths,
        }
        if "pixel_values" in batch:
            prompt_batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=self._model_dtype())
        if "image_grid_thw" in batch:
            prompt_batch["image_grid_thw"] = batch["image_grid_thw"].to(device=device, dtype=torch.long)
        prompt_batch["prompt_texts"] = resolved_prompt_texts
        return prompt_batch

    def _concat_prompt_and_response(
        self,
        *,
        prompt_batch: dict[str, Tensor],
        response_token_ids: Tensor,
        response_mask: Tensor,
    ) -> dict[str, Tensor]:
        prompt_ids = prompt_batch["input_ids"]
        prompt_mask = prompt_batch["attention_mask"]
        prompt_lengths = prompt_batch["prompt_lengths"]
        batch_size = prompt_ids.shape[0]
        total_lengths = []
        sequences: list[Tensor] = []
        masks: list[Tensor] = []
        for i in range(batch_size):
            prompt_len = int(prompt_lengths[i].item())
            resp_len = int(response_mask[i].sum().item())
            seq = torch.cat(
                [prompt_ids[i, :prompt_len], response_token_ids[i, :resp_len]],
                dim=0,
            )
            mask = torch.ones(seq.shape[0], dtype=torch.long, device=seq.device)
            sequences.append(seq)
            masks.append(mask)
            total_lengths.append(seq.shape[0])
        max_len = max(total_lengths)
        pad_id = self.tokenizer.pad_token_id or 0
        padded_ids: list[Tensor] = []
        padded_masks: list[Tensor] = []
        for seq, mask in zip(sequences, masks, strict=True):
            pad = max_len - seq.shape[0]
            if pad > 0:
                seq = F.pad(seq, (0, pad), value=pad_id)
                mask = F.pad(mask, (0, pad), value=0)
            padded_ids.append(seq)
            padded_masks.append(mask)
        out = {
            "input_ids": torch.stack(padded_ids, dim=0),
            "attention_mask": torch.stack(padded_masks, dim=0),
        }
        if "pixel_values" in prompt_batch:
            out["pixel_values"] = prompt_batch["pixel_values"]
        if "image_grid_thw" in prompt_batch:
            out["image_grid_thw"] = prompt_batch["image_grid_thw"]
        return out

    def _build_action_texts(self) -> list[list[str]]:
        all_texts: list[list[str]] = []
        for env_id in self.env_ids:
            adapter = make_default_adapter(env_id)
            all_texts.append([adapter.action_to_text(action_id) for action_id in range(self.config.env.num_actions)])
        return all_texts

    def _build_prompts(self) -> list[str]:
        prompts: list[str] = []
        for _env_name, _action_texts in zip(self.env_ids, self.action_texts_by_env, strict=True):
            prompts.append(
                "[Initial Observation]:\n"
                "<image>\n"
                "Decide your next action(s).\n"
                f"{self._build_user_format_suffix()}"
            )
        return prompts

    def _build_system_prompts(self) -> list[str]:
        prompts: list[str] = []
        for _action_texts in self.action_texts_by_env:
            prompts.append(self._build_vagen_system_prompt())
        return prompts

    def _build_vagen_system_prompt(self) -> str:
        return (
            "You are a Sokoban solver.\n"
            "Sokoban Quick Guide\n"
            "Goal: Push all boxes onto targets.\n"
            "Symbols (If image is provided there are no symbols):\n"
            "# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target\n"
            "Rules:\n"
            "1. Push boxes (can't pull).\n"
            "2. Avoid walls.\n"
            "Actions you can take: Left, Down, Right, Up."
        )

    def _build_user_format_suffix(self) -> str:
        max_actions = self.ff_cfg.max_actions_per_turn
        sep = self.ff_cfg.action_separator
        if self.ff_cfg.prompt_format == "wm":
            suffix = (
                f"You can take up to {max_actions} action(s) at a time, separated by {sep}.\n"
                "Your response must be in the format of:\n"
                "<observation>...</observation><think>...</think>"
                f"{self.ff_cfg.answer_open_tag}...{self.ff_cfg.answer_close_tag}<prediction>...</prediction>.\n\n"
                "Rules for <observation> and <prediction>:\n"
                "- You must strictly describe the relative position of the `target` and any visible `box` objects relative to the player.\n"
                "- For each object, include exactly one vertical relationship: `above`, `below`, or `same row`.\n"
                "- For each object, include exactly one horizontal relationship: `left`, `right`, or `same column`.\n"
                "- Use only the terms: `above`, `below`, `same row`, `left`, `right`, `same column`.\n"
                "- Always use the phrasing pattern: \"X is <vertical> and <horizontal> of the player\".\n"
                "- Do not include extra information.\n\n"
                f"Rules for {self.ff_cfg.answer_open_tag}...{self.ff_cfg.answer_close_tag}:\n"
                f"- Output 1 to {max_actions} action(s).\n"
                "- Valid actions are: Up, Down, Left, Right.\n"
                f"- Separate multiple actions with `{sep}`."
            )
            if self.ff_cfg.use_prompt_examples:
                suffix += (
                    "\n\nExample 1:\n"
                    "<observation>The box is below and right of the player, and the target is below and right of the player</observation>\n"
                    "<think>I should move right to align my column with the box and the target</think>\n"
                    f"{self.ff_cfg.answer_open_tag}Right{self.ff_cfg.answer_close_tag}\n"
                    "<prediction>The box will be below and same column of the player, and the target will be below and same column of the player</prediction>\n\n"
                    "Example 2:\n"
                    "<observation>The box is above and left of the player, and the target is above and same column of the player</observation>\n"
                    "<think>I should move up to align my row with the box and reach the target's row position</think>\n"
                    f"{self.ff_cfg.answer_open_tag}Up{self.ff_cfg.answer_close_tag}\n"
                    "<prediction>The box will be same row and left of the player, and the target will be same row and same column of the player</prediction>\n\n"
                    "Example 3:\n"
                    "<observation>The box is same row and right of the player, and the target is same row and left of the player</observation>\n"
                    "<think>I should move right to push the box right while keeping the target on my left</think>\n"
                    f"{self.ff_cfg.answer_open_tag}Right{self.ff_cfg.answer_close_tag}\n"
                    "<prediction>The box will be same row and right of the player, and the target will be same row and left of the player</prediction>"
                )
        else:
            suffix = (
                f"You can take up to {max_actions} action(s) at a time, separated by {sep}.\n"
                "You should first give your reasoning, and then your answer.\n"
                f"Your response should be in the format of:\n<think>...</think>{self.ff_cfg.answer_open_tag}...{self.ff_cfg.answer_close_tag}"
            )
            if self.ff_cfg.use_prompt_examples:
                suffix += (
                    "\n\nExample 1:\n"
                    "<think>The box is one step below me, and the target is two steps below me. I should go down to reach the box and then push it down to the target.</think>\n"
                    f"{self.ff_cfg.answer_open_tag}Down{self.ff_cfg.answer_close_tag}\n\n"
                    "Example 2:\n"
                    "<think>The box is to the right of me, and the target is further to the right. I need to move right to get behind the box and push it toward the target.</think>\n"
                    f"{self.ff_cfg.answer_open_tag}Right{self.ff_cfg.answer_close_tag}\n\n"
                    "Example 3:\n"
                    "<think>The box is above me, and the target is above the box. I should move up to reach the box and then push it upward to the target.</think>\n"
                    f"{self.ff_cfg.answer_open_tag}Up{self.ff_cfg.answer_close_tag}"
                )
        return suffix


    @staticmethod
    def _to_pil_image(observation: Any) -> Any:
        if Image is None:
            return observation
        if isinstance(observation, Image.Image):
            return observation.convert("RGB")
        return Image.fromarray(observation).convert("RGB")

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @staticmethod
    def _prompt_to_content(prompt: str, image: Any) -> list[dict[str, Any]]:
        parts = prompt.split("<image>")
        content: list[dict[str, Any]] = []
        for idx, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if idx < len(parts) - 1:
                content.append({"type": "image", "image": image})
        if not content:
            content.append({"type": "image", "image": image})
        return content

    def _serialize_conversations(
        self,
        conversations: list[list[dict[str, Any]]],
    ) -> tuple[list[list[dict[str, Any]]], list[Any]]:
        serialized: list[list[dict[str, Any]]] = []
        flat_images: list[Any] = []
        for conversation in conversations:
            serialized_messages: list[dict[str, Any]] = []
            for message in conversation:
                serialized_content: list[dict[str, Any]] = []
                for item in message["content"]:
                    item_type = item.get("type")
                    if item_type == "image":
                        flat_images.append(item.get("image"))
                        serialized_content.append({"type": "image"})
                    else:
                        serialized_content.append(item)
                serialized_messages.append(
                    {
                        "role": message["role"],
                        "content": serialized_content,
                    }
                )
            serialized.append(serialized_messages)
        return serialized, flat_images


def compute_gae_with_dones(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE on flattened real trajectories with terminal masks."""

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * values[step + 1] * not_done - values[step]
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
        advantages[step] = next_advantage
    returns = advantages + values[:-1]
    return advantages, returns


def compute_gae_with_termination(
    rewards: Tensor, values: Tensor, terminated: Tensor, truncated: Tensor, *,
    gamma: float, gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """GAE that bootstraps time-limit truncations but not true terminals."""
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap_mask = 1.0 - terminated[index]
        boundary_mask = 1.0 - torch.maximum(terminated[index], truncated[index])
        delta = rewards[index] + gamma * values[index + 1] * bootstrap_mask - values[index]
        next_advantage = delta + gamma * gae_lambda * boundary_mask * next_advantage
        advantages[index] = next_advantage
    return advantages, advantages + values[:-1]


class RealEnvPPOTrainer:
    """PPO trainer for raw-RGB freeform VLM policies in the real environment."""

    def __init__(
        self,
        *,
        config: Config,
        policy: RGBFreeformPolicy,
        device: torch.device,
        logger: RealEnvLogger | None = None,
        checkpoint_dir: str | Path | None = None,
        adapter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy.to(device)
        self.device = device
        self.logger = logger
        self.adapter_kwargs = dict(adapter_kwargs or {})
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.checkpoint_dir / "freeform_traces.jsonl"
        else:
            self.trace_path = None

        policy_params = [p for p in self.policy.model.parameters() if p.requires_grad]
        value_params = [p for p in self.policy.value_head.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": policy_params, "lr": config.phase2.ppo.actor_lr},
                {"params": value_params, "lr": config.phase2.ppo.critic_lr},
            ],
            weight_decay=config.phase2.ppo.weight_decay,
        )

    def train(self, *, start_update: int = 0) -> None:
        total_updates = self.config.phase2.ppo.total_updates
        for update in range(start_update + 1, total_updates + 1):
            rollout = self.collect_rollout(update=update)
            metrics = self.update_policy(rollout)
            metrics.update(self._rollout_metrics(rollout))
            metrics["phase2/update"] = float(update)
            if self.logger is not None:
                self.logger.log_scalars(update, metrics)
            if self._should_eval(update):
                eval_metrics = self.evaluate(num_episodes=self.config.phase2.ppo.eval_episodes)
                if self.logger is not None:
                    self.logger.log_scalars(update, eval_metrics)
                metrics.update(eval_metrics)
            if self.checkpoint_dir is not None and self._should_save(update):
                self.save_checkpoint(self.checkpoint_dir / "latest.pt", update=update)
            sr = metrics.get("eval/success_rate", -1.0)
            sr_str = f" eval_sr={sr:.2f}" if sr >= 0 else ""
            print(
                f"[update {update}/{total_updates}] "
                f"ret={metrics['phase2/return_mean']:.4f} "
                f"ent={metrics['phase2/entropy']:.4f} "
                f"parse={metrics['phase2/parse_success_rate']:.3f}{sr_str}",
                flush=True,
            )

    @torch.no_grad()
    def collect_rollout(self, *, update: int) -> RealEnvRolloutBatch:
        adapter = self._adapter()
        num_episodes = self.config.phase2.ppo.rollout_batch_size
        collect_max_steps = getattr(self.config.phase2.ppo, "collect_max_steps", 0)
        max_steps = collect_max_steps or self.config.phase2.ppo.eval_max_steps
        observations: list[Any] = []
        env_ids_list: list[int] = []
        actions: list[Tensor] = []
        rewards: list[float] = []
        dones: list[float] = []
        terminated_flags: list[float] = []
        truncated_flags: list[float] = []
        values: list[float] = []
        response_token_ids: list[Tensor] = []
        response_masks: list[Tensor] = []
        token_log_probs: list[Tensor] = []
        parse_success: list[Tensor] = []
        response_lengths: list[int] = []
        trajectory_ids: list[int] = []
        step_ids: list[int] = []
        raw_texts: list[str] = []
        parsed_texts: list[str] = []
        prompt_texts: list[str] = []
        bootstrap_values: list[float] = []

        env_id = self._env_index(adapter.env_id)
        self.policy.eval()
        for episode_idx in range(num_episodes):
            env = adapter.make_env(seed=10000 + episode_idx)
            try:
                obs = env.reset(seed=10000 + episode_idx)
                done = False
                step_idx = 0
                turn_history: list[dict[str, str]] = []
                while not done and step_idx < max_steps:
                    obs_list = [obs]
                    env_tensor = torch.tensor([env_id], device=self.device, dtype=torch.long)
                    prompt = self._build_single_turn_prompt(obs, turn_history)
                    step = self.policy.act(
                        obs_list,
                        env_tensor,
                        deterministic=False,
                        prompt_texts=[prompt],
                    )
                    next_obs, reward, done, _info = self._step_env_with_action_sequence(
                        env=env,
                        adapter=adapter,
                        current_obs=obs,
                        action_sequence=step.action_sequences[0],
                    )

                    observations.append(obs)
                    env_ids_list.append(env_id)
                    actions.append(step.actions[0].detach().cpu())
                    rewards.append(float(reward))
                    success = bool(_info.get("success", False) or _info.get("all_boxes_on_target", False))
                    time_limit = bool(done and not success)
                    collector_limit = bool((step_idx + 1) >= max_steps and not success)
                    terminated_flags.append(float(success))
                    truncated_flags.append(float(time_limit or collector_limit))
                    dones.append(float(success or time_limit or collector_limit))
                    values.append(float(step.values[0].item()))
                    response_token_ids.append(step.response_token_ids[0].detach().cpu())
                    response_masks.append(step.response_mask[0].detach().cpu())
                    token_log_probs.append(step.token_log_probs[0].detach().cpu())
                    parse_success.append(step.parse_success[0].detach().cpu())
                    response_lengths.append(int(step.response_mask[0].sum().item()))
                    trajectory_ids.append(episode_idx)
                    step_ids.append(step_idx)
                    raw_texts.append(step.raw_texts[0])
                    parsed_texts.append(step.parsed_texts[0])
                    prompt_texts.append(step.prompt_texts[0])

                    action_text = step.action_texts[0] if bool(step.parse_success[0].item()) else "invalid"
                    turn_history.append({"response": step.raw_texts[0], "action": action_text})
                    obs = next_obs
                    step_idx += 1
                if terminated_flags and terminated_flags[-1] and trajectory_ids[-1] == episode_idx:
                    bootstrap_values.append(0.0)
                else:
                    final_prompt = self._build_single_turn_prompt(obs, turn_history)
                    final = self.policy.act(
                        [obs], torch.tensor([env_id], device=self.device),
                        deterministic=True, prompt_texts=[final_prompt],
                    )
                    bootstrap_values.append(float(final.values[0].item()))
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        self.policy.train()

        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32)
        terminated_t = torch.tensor(terminated_flags, device=self.device, dtype=torch.float32)
        truncated_t = torch.tensor(truncated_flags, device=self.device, dtype=torch.float32)
        values_t = torch.tensor(values + bootstrap_values[:1], device=self.device, dtype=torch.float32)
        values_ext = self._expand_bootstrap_values(values_t, trajectory_ids, values, bootstrap_values)
        advantages, returns = compute_gae_with_termination(
            rewards_t,
            values_ext,
            terminated_t, truncated_t,
            gamma=self.config.phase2.ppo.gamma,
            gae_lambda=self.config.phase2.ppo.gae_lambda,
        )

        response_token_ids_t, response_mask_t, old_token_log_probs_t = self._pad_token_lists(
            response_token_ids,
            response_masks,
            token_log_probs,
        )
        parse_success_t = torch.stack(parse_success, dim=0).to(dtype=torch.bool)
        batch = RealEnvRolloutBatch(
            observations=observations,
            env_ids=torch.tensor(env_ids_list, dtype=torch.long),
            actions=torch.stack(actions, dim=0).to(dtype=torch.long),
            rewards=rewards_t.cpu(),
            dones=dones_t.cpu(),
            values=values_ext[:-1].cpu(),
            returns=returns.cpu(),
            advantages=advantages.cpu(),
            response_token_ids=response_token_ids_t,
            response_mask=response_mask_t,
            old_token_log_probs=old_token_log_probs_t,
            parse_success=parse_success_t,
            response_lengths=torch.tensor(response_lengths, dtype=torch.long),
            trajectory_ids=torch.tensor(trajectory_ids, dtype=torch.long),
            step_ids=torch.tensor(step_ids, dtype=torch.long),
            raw_texts=raw_texts,
            parsed_texts=parsed_texts,
            prompt_texts=prompt_texts,
        )
        self._write_traces(update=update, rollout=batch)
        return batch

    def update_policy(self, rollout: RealEnvRolloutBatch) -> dict[str, float]:
        ppo_cfg = self.config.phase2.ppo
        advantages = rollout.advantages.to(self.device)
        if ppo_cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        returns = rollout.returns.to(self.device)
        env_ids = rollout.env_ids.to(self.device)
        response_token_ids = rollout.response_token_ids.to(self.device)
        response_mask = rollout.response_mask.to(self.device)
        old_token_log_probs = rollout.old_token_log_probs.to(self.device)
        old_values = rollout.values.to(self.device)

        indices = torch.arange(returns.shape[0], device=self.device)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_kl = 0.0
        num_minibatches = 0

        for _ in range(ppo_cfg.epochs_per_update):
            perm = indices[torch.randperm(indices.shape[0], device=self.device)]
            for start in range(0, perm.shape[0], ppo_cfg.minibatch_size):
                mb_idx = perm[start:start + ppo_cfg.minibatch_size]
                if mb_idx.numel() == 0:
                    continue
                mb_obs = [rollout.observations[int(i)] for i in mb_idx.detach().cpu().tolist()]
                mb_env_ids = env_ids[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_old_token_log_probs = old_token_log_probs[mb_idx]
                mb_response_token_ids = response_token_ids[mb_idx]
                mb_response_mask = response_mask[mb_idx]
                mb_prompt_texts = [rollout.prompt_texts[int(i)] for i in mb_idx.detach().cpu().tolist()]

                new_token_log_probs, token_entropies, values = self.policy.evaluate_actions(
                    mb_obs,
                    mb_env_ids,
                    prompt_texts=mb_prompt_texts,
                    response_token_ids=mb_response_token_ids,
                    response_mask=mb_response_mask,
                )
                token_mask = mb_response_mask.to(dtype=new_token_log_probs.dtype)
                adv_matrix = mb_adv.unsqueeze(1).expand_as(new_token_log_probs)
                ratio = torch.exp(new_token_log_probs - mb_old_token_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - ppo_cfg.clip_epsilon, 1.0 + ppo_cfg.clip_epsilon)
                token_objective = torch.min(ratio * adv_matrix, clipped_ratio * adv_matrix)
                valid_tokens = token_mask.sum().clamp_min(1.0)
                policy_loss = -(token_objective * token_mask).sum() / valid_tokens
                entropy_mean = (token_entropies * token_mask).sum() / valid_tokens
                clipfrac = (((torch.abs(ratio - 1.0) > ppo_cfg.clip_epsilon).float()) * token_mask).sum() / valid_tokens
                kl_div = ((mb_old_token_log_probs - new_token_log_probs) * token_mask).sum() / valid_tokens
                value_loss = F.mse_loss(values, mb_returns)
                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy_mean
                    + ppo_cfg.kl_coef * kl_div
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), ppo_cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.detach().item())
                total_value_loss += float(value_loss.detach().item())
                total_entropy += float(entropy_mean.detach().item())
                total_clipfrac += float(clipfrac.detach().item())
                total_kl += float(kl_div.detach().item())
                num_minibatches += 1

        updated_values = []
        with torch.no_grad():
            for i in range(len(rollout.observations)):
                obs = [rollout.observations[i]]
                env_id = rollout.env_ids[i:i + 1].to(self.device)
                prompt_batch = self.policy._prepare_prompt_batch(
                    obs,
                    env_id,
                    prompt_texts=[rollout.prompt_texts[i]],
                )
                updated_values.append(self.policy._value_from_prompt_batch(prompt_batch)[0])
        updated_values_t = torch.stack(updated_values, dim=0)
        explained_variance = _explained_variance(updated_values_t.cpu(), returns.cpu())
        value_delta = (updated_values_t - old_values).abs().mean()
        return {
            "phase2/policy_loss": total_policy_loss / max(1, num_minibatches),
            "phase2/value_loss": total_value_loss / max(1, num_minibatches),
            "phase2/entropy": total_entropy / max(1, num_minibatches),
            "phase2/clipfrac": total_clipfrac / max(1, num_minibatches),
            "phase2/kl_divergence": total_kl / max(1, num_minibatches),
            "phase2/explained_variance": explained_variance,
            "phase2/value_delta": float(value_delta.item()),
        }

    @torch.no_grad()
    def evaluate(self, *, num_episodes: int) -> dict[str, float]:
        metrics, _ = self.evaluate_with_traces(num_episodes=num_episodes, trace_path=None)
        return metrics

    @torch.no_grad()
    def evaluate_with_traces(
        self,
        *,
        num_episodes: int,
        trace_path: str | Path | None,
    ) -> tuple[dict[str, float], list[dict[str, object]]]:
        adapter = self._adapter()
        max_steps = self.config.phase2.ppo.eval_max_steps
        successes = 0
        returns: list[float] = []
        lengths: list[int] = []
        parse_successes: list[float] = []
        trace_records: list[dict[str, object]] = []
        self.policy.eval()
        for episode_idx in range(num_episodes):
            env = adapter.make_env(seed=20000 + episode_idx)
            try:
                obs = env.reset(seed=20000 + episode_idx)
                done = False
                ep_return = 0.0
                ep_len = 0
                env_tensor = torch.tensor([self._env_index(adapter.env_id)], device=self.device, dtype=torch.long)
                turn_history: list[dict[str, str]] = []
                while not done and ep_len < max_steps:
                    prompt = self._build_single_turn_prompt(obs, turn_history)
                    step = self.policy.act(
                        [obs],
                        env_tensor,
                        deterministic=False,
                        prompt_texts=[prompt],
                    )
                    next_obs, reward, done, info = self._step_env_with_action_sequence(
                        env=env,
                        adapter=adapter,
                        current_obs=obs,
                        action_sequence=step.action_sequences[0],
                    )
                    parse_successes.append(float(step.parse_success[0].item()))
                    ep_return += float(reward)
                    trace_records.append(
                        {
                            "episode": episode_idx,
                            "step_id": ep_len,
                            "prompt_text": step.prompt_texts[0],
                            "raw_text": step.raw_texts[0],
                            "parsed_text": step.parsed_texts[0],
                            "action_id": int(step.actions[0].item()),
                            "action_sequence": step.action_sequences[0],
                            "parse_success": bool(step.parse_success[0].item()),
                            "reward": float(reward),
                            "done": bool(done),
                            "success": bool(info.get("success", False)),
                        }
                    )
                    action_text = step.action_texts[0] if bool(step.parse_success[0].item()) else "invalid"
                    turn_history.append({"response": step.raw_texts[0], "action": action_text})
                    obs = next_obs
                    ep_len += 1
                if info.get("success", False) or (done and ep_return > 0.0):
                    successes += 1
                returns.append(ep_return)
                lengths.append(ep_len)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        self.policy.train()
        metrics = {
            "eval/success_rate": successes / max(1, num_episodes),
            "eval/avg_return": sum(returns) / max(1, len(returns)),
            "eval/avg_episode_length": sum(lengths) / max(1, len(lengths)),
            "eval/parse_success_rate": sum(parse_successes) / max(1, len(parse_successes)),
        }
        if trace_path is not None:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for record in trace_records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return metrics, trace_records

    def save_checkpoint(self, path: str | Path, *, update: int) -> None:
        torch.save(
            {
                "update": update,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint.get("update", 0))

    def _rollout_metrics(self, rollout: RealEnvRolloutBatch) -> dict[str, float]:
        rewards = rollout.rewards
        values = rollout.values
        metrics = {
            "phase2/reward_mean": float(rewards.mean().item()),
            "phase2/reward_std": float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
            "phase2/reward_positive_rate": float((rewards > 0).float().mean().item()),
            "phase2/return_mean": float(rollout.returns.mean().item()),
            "phase2/return_std": float(rollout.returns.std().item()) if rollout.returns.numel() > 1 else 0.0,
            "phase2/value_mean": float(values.mean().item()),
            "phase2/adv_mean": float(rollout.advantages.mean().item()),
            "phase2/adv_std": float(rollout.advantages.std().item()) if rollout.advantages.numel() > 1 else 0.0,
            "phase2/parse_success_rate": float(rollout.parse_success.float().mean().item()),
            "phase2/response_length_mean": float(rollout.response_lengths.float().mean().item()),
        }
        counts = torch.bincount(rollout.actions, minlength=self.config.env.num_actions).float()
        action_frac = counts / counts.sum().clamp_min(1.0)
        for action_id in range(self.config.env.num_actions):
            metrics[f"phase2/action_{action_id}_frac"] = float(action_frac[action_id].item())
        return metrics

    def _write_traces(self, *, update: int, rollout: RealEnvRolloutBatch) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as handle:
            for i in range(len(rollout.raw_texts)):
                record = {
                    "update": update,
                    "trajectory_id": int(rollout.trajectory_ids[i].item()),
                    "step_id": int(rollout.step_ids[i].item()),
                    "prompt_text": rollout.prompt_texts[i],
                    "raw_text": rollout.raw_texts[i],
                    "parsed_text": rollout.parsed_texts[i],
                    "action_id": int(rollout.actions[i].item()),
                    "action_sequence": rollout.action_texts[i].split(self.config.phase2.freeform_policy.action_separator)
                    if rollout.action_texts[i]
                    else [],
                    "parse_success": bool(rollout.parse_success[i].item()),
                    "reward": float(rollout.rewards[i].item()),
                    "value": float(rollout.values[i].item()),
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _pad_token_lists(
        response_token_ids: list[Tensor],
        response_masks: list[Tensor],
        token_log_probs: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        max_len = max(t.shape[0] for t in response_token_ids)
        pad_id = 0
        padded_ids = []
        padded_masks = []
        padded_log_probs = []
        for token_ids, mask, log_probs in zip(response_token_ids, response_masks, token_log_probs, strict=True):
            pad = max_len - token_ids.shape[0]
            if pad > 0:
                token_ids = F.pad(token_ids, (0, pad), value=pad_id)
                mask = F.pad(mask, (0, pad), value=0)
                log_probs = F.pad(log_probs, (0, pad), value=0.0)
            padded_ids.append(token_ids)
            padded_masks.append(mask)
            padded_log_probs.append(log_probs)
        return (
            torch.stack(padded_ids, dim=0),
            torch.stack(padded_masks, dim=0),
            torch.stack(padded_log_probs, dim=0),
        )

    @staticmethod
    def _expand_bootstrap_values(values_t: Tensor, trajectory_ids: list[int], values: list[float], bootstrap_values: list[float] | None = None) -> Tensor:
        # Build per-step next values; boundary values may bootstrap truncations.
        n = len(values)
        out = torch.zeros(n + 1, device=values_t.device, dtype=values_t.dtype)
        out[:n] = values_t[:-1]
        for i in range(n):
            if i + 1 < n and trajectory_ids[i + 1] == trajectory_ids[i]:
                out[i + 1] = values_t[i + 1]
            else:
                trajectory = trajectory_ids[i]
                out[i + 1] = 0.0 if bootstrap_values is None else bootstrap_values[trajectory]
        return out

    def _adapter(self) -> EnvironmentAdapter:
        return make_default_adapter(self.config.env.env_ids[0], **self.adapter_kwargs)

    def _build_single_turn_prompt(
        self,
        obs: Any,
        history: list[dict[str, str]],
    ) -> str:
        """Build a single-turn user prompt with text history + current image.

        Always produces exactly one ``<image>`` placeholder so the processor
        only needs to handle a single image — avoiding all multi-image bugs.

        ``history`` is a list of dicts with keys ``action`` and optionally
        ``response`` (the raw model output from the previous turn).
        """
        parts: list[str] = []
        if history:
            parts.append("[Previous turns]\n")
            for idx, h in enumerate(history[-5:], 1):  # keep last 5 turns
                resp = h.get("response", "")
                act_text = h.get("action", "unknown")
                if resp:
                    # Include the model's own reasoning so it can build on it
                    parts.append(f"Turn {idx}: {resp}\nExtracted action: {act_text}\n\n")
                else:
                    parts.append(f"Turn {idx}: action was {act_text}\n\n")
            parts.append("[Current Observation]:\n<image>\nDecide your next action(s).\n")
        else:
            parts.append("[Initial Observation]:\n<image>\nDecide your next action(s).\n")
        parts.append(self.policy._build_user_format_suffix())
        return "".join(parts)

    def _env_index(self, env_id: str) -> int:
        return self.config.env.env_ids.index(env_id)

    @staticmethod
    def _to_env_action(adapter: EnvironmentAdapter, action_id: int) -> Any:
        converter = getattr(adapter, "to_env_action", None)
        if callable(converter):
            return converter(action_id)
        if getattr(adapter, "env_id", "") == "sokoban":
            return int(action_id) + 1
        return int(action_id)

    def _step_env_with_action_sequence(
        self,
        *,
        env: Any,
        adapter: EnvironmentAdapter,
        current_obs: Any,
        action_sequence: list[str],
    ) -> tuple[Any, float, bool, dict[str, Any]]:
        if not action_sequence:
            return current_obs, 0.0, False, {"success": False, "executed_actions": []}

        total_reward = 0.0
        obs = current_obs
        done = False
        last_info: dict[str, Any] = {"success": False}
        executed_actions: list[str] = []
        normalized_lookup = {
            _normalize_action_text(text): idx
            for idx, text in enumerate(self.policy.action_texts_by_env[0])
        }
        for action_text in action_sequence:
            normalized_action = _normalize_action_text(action_text)
            if normalized_action not in normalized_lookup:
                break
            action_id = normalized_lookup[normalized_action]
            env_action = self._to_env_action(adapter, action_id)
            obs, reward, done, info = env.step(env_action)
            total_reward += float(reward)
            last_info = dict(info)
            executed_actions.append(self.policy.action_texts_by_env[0][action_id])
            if done:
                break
        last_info["executed_actions"] = executed_actions
        return obs, total_reward, done, last_info

    def _should_eval(self, next_update: int) -> bool:
        freq = self.config.phase2.ppo.eval_every
        return freq > 0 and next_update % freq == 0

    def _should_save(self, next_update: int) -> bool:
        freq = self.config.phase2.ppo.checkpoint_every
        return freq > 0 and next_update % freq == 0


def _explained_variance(values: Tensor, targets: Tensor) -> float:
    target_var = torch.var(targets)
    if not torch.isfinite(target_var) or target_var.item() < 1e-8:
        return 0.0
    residual = torch.var(targets - values)
    return float((1.0 - residual / target_var).item())
