"""Phase 2 latent-space RL scaffolding for SemBelief-WM.

This module implements the Phase 2 baseline and Route B policy family:

- Route A: latent actor-critic heads on top of frozen world-model beliefs
- Route B: free-decoding VLM policy conditioned on latent beliefs
- PPO + GAE optimization on imagined rewards from the frozen simulator
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from ..config import Config
from ..data.datasource import OfflineDataSource
from ..model.belief import BeliefReadout
from ..model.world_model import WorldModel
from ..types import BeliefState, SequenceBatch
from .planner import LookaheadPlanner, make_planner


class Phase2Logger(Protocol):
    """Minimal logger contract for Phase 2 training."""

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        """Record a set of scalar metrics for one PPO update."""


@dataclass(frozen=True)
class RolloutBatch:
    """Imagined PPO batch collected from the frozen world model."""

    beliefs: Tensor          # (B, H, K, D)
    env_ids: Tensor | None   # (B,)
    actions: Tensor          # (B, H)
    rewards: Tensor          # (B, H)
    values: Tensor           # (B, H + 1)
    advantages: Tensor       # (B, H)
    returns: Tensor          # (B, H)
    old_log_probs: Tensor | None = None            # (B, H) for scalar-action PPO
    response_token_ids: Tensor | None = None       # (B, H, L) for token-level freeform PPO
    response_mask: Tensor | None = None            # (B, H, L)
    old_token_log_probs: Tensor | None = None      # (B, H, L)
    parse_success: Tensor | None = None            # (B, H)

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class FreeformGeneration:
    """Decoded policy response plus token-level statistics for PPO."""

    texts: list[str]
    token_ids: Tensor       # (B, L)
    token_mask: Tensor      # (B, L)
    token_log_probs: Tensor # (B, L)
    token_entropies: Tensor # (B, L)


@dataclass(frozen=True)
class FreeformPolicyStep:
    """Freeform policy output for one rollout step."""

    actions: Tensor
    log_probs: Tensor
    entropy: Tensor
    response_token_ids: Tensor
    response_mask: Tensor
    old_token_log_probs: Tensor
    parse_success: Tensor


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def _normalize_action_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


class ActionParser:
    """Parse free-decoded text into a legal environment action."""

    def __init__(self, config: Config, action_texts_by_env: list[list[str]]) -> None:
        self._num_actions = config.env.num_actions
        open_tag = re.escape(config.phase2.freeform_policy.answer_open_tag)
        close_tag = re.escape(config.phase2.freeform_policy.answer_close_tag)
        self._pattern = re.compile(f"{open_tag}(.*?){close_tag}", flags=re.IGNORECASE | re.DOTALL)
        self._normalized_actions = [
            [_normalize_action_text(text) for text in env_texts]
            for env_texts in action_texts_by_env
        ]

    def parse(
        self,
        texts: list[str],
        env_ids: Tensor | None,
        *,
        fallback_actions: Tensor | None = None,
    ) -> Tensor:
        actions, _ = self.parse_with_success(
            texts,
            env_ids,
            fallback_actions=fallback_actions,
        )
        return actions

    def parse_with_success(
        self,
        texts: list[str],
        env_ids: Tensor | None,
        *,
        fallback_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if env_ids is None:
            device = fallback_actions.device if fallback_actions is not None else torch.device("cpu")
            env_ids = torch.zeros(len(texts), dtype=torch.long, device=device)
        parsed: list[int] = []
        matched: list[bool] = []
        for index, raw_text in enumerate(texts):
            env_index = int(env_ids[index].item())
            normalized_actions = self._normalized_actions[env_index]
            candidate = self._extract_answer(raw_text)
            action_id = self._match_action(candidate, normalized_actions)
            is_match = action_id is not None
            if action_id is None:
                if fallback_actions is not None:
                    action_id = int(fallback_actions[index].item())
                else:
                    action_id = 0
            parsed.append(action_id)
            matched.append(is_match)
        return (
            torch.tensor(parsed, dtype=torch.long, device=env_ids.device),
            torch.tensor(matched, dtype=torch.bool, device=env_ids.device),
        )

    def _extract_answer(self, raw_text: str) -> str:
        match = self._pattern.search(raw_text)
        if match is not None:
            return _normalize_action_text(match.group(1))
        return _normalize_action_text(raw_text)

    @staticmethod
    def _match_action(candidate: str, normalized_actions: list[str]) -> int | None:
        if not candidate:
            return None
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


class BeliefProjector(nn.Module):
    """Project latent belief slots into a short prefix token sequence for the policy LLM."""

    def __init__(self, hidden_dim: int, num_prefix_tokens: int, num_heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_prefix_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, belief: BeliefState | Tensor) -> Tensor:
        slots = belief.slots if isinstance(belief, BeliefState) else belief
        queries = self.queries.unsqueeze(0).expand(slots.shape[0], -1, -1)
        attended, _ = self.cross_attn(queries, slots, slots, need_weights=False)
        return self.norm(queries + attended)


def _import_policy_dependencies() -> tuple[Any, Any]:
    try:
        import peft  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Route B VLM policy requires the optional VLM dependencies. "
            "Install them with `uv sync --extra vlm`."
        ) from exc
    return transformers, peft


def _load_policy_model(
    *,
    transformers: Any,
    model_name: str,
    model_kwargs: dict[str, Any],
) -> torch.nn.Module:
    qwen_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    if qwen_cls is not None:
        return qwen_cls.from_pretrained(model_name, **model_kwargs)
    auto_cls = getattr(transformers, "AutoModelForCausalLM")
    return auto_cls.from_pretrained(model_name, **model_kwargs)


class VLMFreeformPolicy(nn.Module):
    """Route B policy: free decoding from a separate policy LLM."""

    def __init__(self, config: Config, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_actions = config.env.num_actions
        ff_cfg = config.phase2.freeform_policy

        self.projector = BeliefProjector(
            hidden_dim=config.hidden_dim,
            num_prefix_tokens=ff_cfg.projector_tokens,
            num_heads=ff_cfg.projector_heads,
        )

        transformers, peft = _import_policy_dependencies()
        model_name = ff_cfg.model_name or config.backbone.model_name
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if config.training.dtype == "bf16" else torch.float32,
            "trust_remote_code": True,
        }
        if device is not None:
            model_kwargs["device_map"] = {"": str(device)}
        self.policy_model = _load_policy_model(
            transformers=transformers,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_config = peft.LoraConfig(
            r=ff_cfg.lora_rank,
            lora_alpha=ff_cfg.lora_alpha,
            lora_dropout=ff_cfg.lora_dropout,
            target_modules=ff_cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.policy_model = peft.get_peft_model(self.policy_model, lora_config)
        self.policy_model.config.use_cache = True

        self.action_texts_by_env = self._build_action_texts(config)
        self.prompt_token_ids_table, self.prompt_attention_mask_table = self._build_prompt_tables(config)
        self.response_token_ids_table, self.response_attention_mask_table = self._build_response_tables(config)
        self.parser = ActionParser(config, self.action_texts_by_env)

    def act(
        self,
        belief: BeliefState | Tensor,
        env_ids: Tensor | None,
    ) -> FreeformPolicyStep:
        update_mode = self.config.phase2.freeform_policy.update_mode
        batch_size = belief.slots.shape[0] if isinstance(belief, BeliefState) else belief.shape[0]
        parse_env_ids = env_ids
        if parse_env_ids is None:
            parse_env_ids = torch.zeros(batch_size, dtype=torch.long, device=self._model_device())
        with torch.no_grad():
            generated = self.generate_responses(belief, env_ids)

        fallback_actions: Tensor | None = None
        dist: Categorical | None = None
        scalar_entropy: Tensor | None = None
        if update_mode == "hybrid_scoring":
            action_logits = self.action_logits(belief, env_ids)
            dist = Categorical(logits=action_logits)
            fallback_actions = torch.argmax(action_logits, dim=-1)
            scalar_entropy = dist.entropy()

        actions, parse_success = self.parser.parse_with_success(
            generated.texts,
            parse_env_ids,
            fallback_actions=fallback_actions,
        )

        if update_mode == "hybrid_scoring":
            assert dist is not None
            assert scalar_entropy is not None
            assert fallback_actions is not None
            scalar_log_probs = dist.log_prob(actions)
            entropy = scalar_entropy
            log_probs = scalar_log_probs
        else:
            token_mask = generated.token_mask.to(dtype=generated.token_log_probs.dtype)
            log_probs = (generated.token_log_probs * token_mask).sum(dim=1)
            denom = token_mask.sum(dim=1).clamp_min(1.0)
            entropy = (generated.token_entropies * token_mask).sum(dim=1) / denom

        return FreeformPolicyStep(
            actions=actions,
            log_probs=log_probs,
            entropy=entropy,
            response_token_ids=generated.token_ids,
            response_mask=generated.token_mask,
            old_token_log_probs=generated.token_log_probs,
            parse_success=parse_success,
        )

    def evaluate_actions(
        self,
        belief: BeliefState | Tensor,
        actions: Tensor | None,
        env_ids: Tensor | None,
        *,
        response_token_ids: Tensor | None = None,
        response_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        update_mode = self.config.phase2.freeform_policy.update_mode
        if update_mode == "hybrid_scoring":
            if actions is None:
                raise ValueError("hybrid_scoring requires discrete actions.")
            logits = self.action_logits(belief, env_ids)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy()

        if response_token_ids is None or response_mask is None:
            raise ValueError("token_ppo requires response_token_ids and response_mask.")
        return self.response_token_stats(
            belief,
            env_ids,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
        )

    def action_logits(self, belief: BeliefState | Tensor, env_ids: Tensor | None) -> Tensor:
        prefix_tokens = self.projector(belief)
        prefix_tokens = prefix_tokens.to(dtype=self._model_dtype(), device=self._model_device())
        if env_ids is None:
            env_ids = torch.zeros(prefix_tokens.shape[0], dtype=torch.long, device=prefix_tokens.device)
        else:
            env_ids = env_ids.to(device=prefix_tokens.device, dtype=torch.long)

        batch_size = prefix_tokens.shape[0]
        prompt_ids = self.prompt_token_ids_table[env_ids]
        prompt_mask = self.prompt_attention_mask_table[env_ids]
        response_ids = self.response_token_ids_table[env_ids]
        response_mask = self.response_attention_mask_table[env_ids]

        num_actions = response_ids.shape[1]
        prefix_tokens = prefix_tokens.unsqueeze(1).expand(-1, num_actions, -1, -1)
        prompt_ids = prompt_ids.unsqueeze(1).expand(-1, num_actions, -1)
        prompt_mask = prompt_mask.unsqueeze(1).expand(-1, num_actions, -1)

        flat_prefix = prefix_tokens.reshape(batch_size * num_actions, prefix_tokens.shape[-2], prefix_tokens.shape[-1])
        flat_prompt_ids = prompt_ids.reshape(batch_size * num_actions, prompt_ids.shape[-1])
        flat_prompt_mask = prompt_mask.reshape(batch_size * num_actions, prompt_mask.shape[-1])
        flat_response_ids = response_ids.reshape(batch_size * num_actions, response_ids.shape[-1])
        flat_response_mask = response_mask.reshape(batch_size * num_actions, response_mask.shape[-1])

        prompt_embeds = self._embed_token_ids(flat_prompt_ids)
        response_embeds = self._embed_token_ids(flat_response_ids)

        prefix_mask = torch.ones(
            flat_prefix.shape[0],
            flat_prefix.shape[1],
            dtype=torch.long,
            device=flat_prefix.device,
        )
        attention_mask = torch.cat([prefix_mask, flat_prompt_mask, flat_response_mask], dim=1)
        inputs_embeds = torch.cat([flat_prefix, prompt_embeds, response_embeds], dim=1)

        prefix_and_prompt = flat_prefix.shape[1] + flat_prompt_ids.shape[1]
        labels = torch.full(
            (flat_response_ids.shape[0], prefix_and_prompt + flat_response_ids.shape[1]),
            fill_value=-100,
            dtype=torch.long,
            device=flat_response_ids.device,
        )
        candidate_labels = flat_response_ids.masked_fill(flat_response_mask == 0, -100)
        labels[:, prefix_and_prompt:] = candidate_labels

        outputs = self.policy_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        token_losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.shape[0], -1)
        sequence_log_probs = -token_losses.sum(dim=1)
        return sequence_log_probs.view(batch_size, num_actions)

    def generate_responses(
        self,
        belief: BeliefState | Tensor,
        env_ids: Tensor | None,
    ) -> FreeformGeneration:
        ff_cfg = self.config.phase2.freeform_policy
        prefix_tokens = self.projector(belief)
        prefix_tokens = prefix_tokens.to(dtype=self._model_dtype(), device=self._model_device())
        if env_ids is None:
            env_ids = torch.zeros(prefix_tokens.shape[0], dtype=torch.long, device=prefix_tokens.device)
        else:
            env_ids = env_ids.to(device=prefix_tokens.device, dtype=torch.long)

        prompt_ids = self.prompt_token_ids_table[env_ids]
        prompt_mask = self.prompt_attention_mask_table[env_ids]
        prompt_embeds = self._embed_token_ids(prompt_ids)
        prefix_mask = torch.ones(
            prefix_tokens.shape[0],
            prefix_tokens.shape[1],
            dtype=torch.long,
            device=prefix_tokens.device,
        )
        attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        inputs_embeds = torch.cat([prefix_tokens, prompt_embeds], dim=1)

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id
        generated_ids: list[Tensor] = []
        token_log_probs: list[Tensor] = []
        token_entropies: list[Tensor] = []
        token_masks: list[Tensor] = []
        finished = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)

        outputs = self.policy_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        for _ in range(ff_cfg.max_new_tokens):
            next_logits = outputs.logits[:, -1, :]
            active_mask = ~finished
            step_log_probs = torch.log_softmax(next_logits, dim=-1)
            step_entropies = -(step_log_probs.exp() * step_log_probs).sum(dim=-1)
            sampled_token = self._sample_next_token(next_logits)
            next_token = sampled_token
            if eos_token_id is not None:
                filler = torch.full_like(next_token, pad_token_id if pad_token_id is not None else eos_token_id)
                next_token = torch.where(active_mask, next_token, filler)
            gathered_log_probs = step_log_probs.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1)
            generated_ids.append(next_token)
            token_log_probs.append(torch.where(active_mask, gathered_log_probs, torch.zeros_like(gathered_log_probs)))
            token_entropies.append(torch.where(active_mask, step_entropies, torch.zeros_like(step_entropies)))
            token_masks.append(active_mask.to(dtype=torch.long))
            if eos_token_id is not None:
                finished = finished | (sampled_token == eos_token_id)

            next_input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0],
                        1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            outputs = self.policy_model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            if finished.all():
                break

        if not generated_ids:
            empty = torch.full(
                (inputs_embeds.shape[0], 1),
                fill_value=pad_token_id if pad_token_id is not None else 0,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            zero = torch.zeros_like(empty, dtype=inputs_embeds.dtype)
            mask = torch.zeros_like(empty, dtype=torch.long)
            return FreeformGeneration(
                texts=self.tokenizer.batch_decode(empty, skip_special_tokens=True),
                token_ids=empty,
                token_mask=mask,
                token_log_probs=zero,
                token_entropies=zero,
            )
        stacked_ids = torch.stack(generated_ids, dim=1)
        stacked_log_probs = torch.stack(token_log_probs, dim=1)
        stacked_entropies = torch.stack(token_entropies, dim=1)
        stacked_mask = torch.stack(token_masks, dim=1)
        return FreeformGeneration(
            texts=self.tokenizer.batch_decode(stacked_ids, skip_special_tokens=True),
            token_ids=stacked_ids,
            token_mask=stacked_mask,
            token_log_probs=stacked_log_probs,
            token_entropies=stacked_entropies,
        )

    def response_token_stats(
        self,
        belief: BeliefState | Tensor,
        env_ids: Tensor | None,
        *,
        response_token_ids: Tensor,
        response_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prefix_tokens = self.projector(belief)
        prefix_tokens = prefix_tokens.to(dtype=self._model_dtype(), device=self._model_device())
        if env_ids is None:
            env_ids = torch.zeros(prefix_tokens.shape[0], dtype=torch.long, device=prefix_tokens.device)
        else:
            env_ids = env_ids.to(device=prefix_tokens.device, dtype=torch.long)

        response_token_ids = response_token_ids.to(device=prefix_tokens.device, dtype=torch.long)
        response_mask = response_mask.to(device=prefix_tokens.device, dtype=torch.long)
        prompt_ids = self.prompt_token_ids_table[env_ids]
        prompt_mask = self.prompt_attention_mask_table[env_ids]
        prompt_embeds = self._embed_token_ids(prompt_ids)
        response_embeds = self._embed_token_ids(response_token_ids)

        prefix_mask = torch.ones(
            prefix_tokens.shape[0],
            prefix_tokens.shape[1],
            dtype=torch.long,
            device=prefix_tokens.device,
        )
        attention_mask = torch.cat([prefix_mask, prompt_mask, response_mask], dim=1)
        inputs_embeds = torch.cat([prefix_tokens, prompt_embeds, response_embeds], dim=1)

        outputs = self.policy_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        response_start = prefix_tokens.shape[1] + prompt_ids.shape[1]
        token_logits = logits[:, response_start - 1:-1, :]
        log_probs = torch.log_softmax(token_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=response_token_ids.unsqueeze(-1)).squeeze(-1)
        token_entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
        token_mask_f = response_mask.to(dtype=token_log_probs.dtype)
        return token_log_probs * token_mask_f, token_entropies * token_mask_f

    def _sample_next_token(self, logits: Tensor) -> Tensor:
        ff_cfg = self.config.phase2.freeform_policy
        if not ff_cfg.do_sample:
            return torch.argmax(logits, dim=-1)

        temperature = max(ff_cfg.temperature, 1e-5)
        scaled_logits = logits / temperature
        if ff_cfg.top_p < 1.0:
            scaled_logits = self._top_p_filtering(scaled_logits, ff_cfg.top_p)
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _top_p_filtering(logits: Tensor, top_p: float) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)
        return filtered

    def _embed_token_ids(self, token_ids: Tensor) -> Tensor:
        embedding_layer = self.policy_model.get_input_embeddings()
        return embedding_layer(token_ids.to(device=self._model_device(), dtype=torch.long)).to(dtype=self._model_dtype())

    def _model_device(self) -> torch.device:
        return next(self.policy_model.parameters()).device

    def _model_dtype(self) -> torch.dtype:
        return next(self.policy_model.parameters()).dtype

    def _build_action_texts(self, config: Config) -> list[list[str]]:
        from ..data.adapters import make_default_adapter

        all_texts: list[list[str]] = []
        for env_id in config.env.env_ids:
            adapter = make_default_adapter(env_id)
            all_texts.append([adapter.action_to_text(action_id) for action_id in range(config.env.num_actions)])
        return all_texts

    def _build_prompt_tables(self, config: Config) -> tuple[Tensor, Tensor]:
        ff_cfg = config.phase2.freeform_policy
        prompts: list[str] = []
        for env_name, action_texts in zip(config.env.env_ids, self.action_texts_by_env, strict=True):
            action_list = "\n".join(f"- {text}" for text in action_texts)
            prompts.append(
                ff_cfg.prompt_template.format(
                    env_name=env_name,
                    action_list=action_list,
                    open_tag=ff_cfg.answer_open_tag,
                    close_tag=ff_cfg.answer_close_tag,
                )
            )
        encoded = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.register_buffer("prompt_token_ids_table", encoded["input_ids"], persistent=False)
        self.register_buffer("prompt_attention_mask_table", encoded["attention_mask"], persistent=False)
        return self.prompt_token_ids_table, self.prompt_attention_mask_table

    def _build_response_tables(self, config: Config) -> tuple[Tensor, Tensor]:
        ff_cfg = config.phase2.freeform_policy
        responses = [
            [
                f"{ff_cfg.answer_open_tag}{action_text}{ff_cfg.answer_close_tag}"
                for action_text in action_texts
            ]
            for action_texts in self.action_texts_by_env
        ]
        flat_responses = [text for env_responses in responses for text in env_responses]
        encoded = self.tokenizer(
            flat_responses,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        seq_len = encoded["input_ids"].shape[1]
        self.register_buffer(
            "response_token_ids_table",
            encoded["input_ids"].view(len(responses), config.env.num_actions, seq_len),
            persistent=False,
        )
        self.register_buffer(
            "response_attention_mask_table",
            encoded["attention_mask"].view(len(responses), config.env.num_actions, seq_len),
            persistent=False,
        )
        return self.response_token_ids_table, self.response_attention_mask_table


class LatentActorCritic(nn.Module):
    """Unified Phase 2 actor/value interface across policy families."""

    def __init__(self, config: Config, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.policy_family = config.phase2.actor_critic.policy_family
        self.freeform_update_mode = config.phase2.freeform_policy.update_mode
        ac_cfg = config.phase2.actor_critic

        self.value_readout = BeliefReadout.from_config(config, mode=ac_cfg.readout)
        self.value_net = _build_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=ac_cfg.hidden_dim,
            hidden_layers=ac_cfg.hidden_layers,
            activation=ac_cfg.activation,
            output_dim=1,
        )

        self.policy_readout: BeliefReadout | None = None
        self.policy_net: nn.Module | None = None
        self.freeform_policy: VLMFreeformPolicy | None = None

        if self.policy_family == "mlp":
            self.policy_readout = BeliefReadout.from_config(config, mode=ac_cfg.readout)
            self.policy_net = _build_mlp(
                input_dim=config.hidden_dim,
                hidden_dim=ac_cfg.hidden_dim,
                hidden_layers=ac_cfg.hidden_layers,
                activation=ac_cfg.activation,
                output_dim=config.env.num_actions,
            )
        elif self.policy_family == "vlm_freeform":
            self.freeform_policy = VLMFreeformPolicy(config, device=device)
        else:
            raise ValueError(f"Unsupported policy family: {self.policy_family}")

    def act(
        self,
        belief: BeliefState | Tensor,
        env_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor] | None]:
        value = self.value(belief)
        if self.policy_family == "mlp":
            logits = self.policy_logits(belief)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy, value, None

        assert self.freeform_policy is not None
        step = self.freeform_policy.act(belief, env_ids)
        extras = {
            "response_token_ids": step.response_token_ids,
            "response_mask": step.response_mask,
            "old_token_log_probs": step.old_token_log_probs,
            "parse_success": step.parse_success,
        }
        return step.actions, step.log_probs, step.entropy, value, extras

    def evaluate_actions(
        self,
        belief: BeliefState | Tensor,
        actions: Tensor | None,
        env_ids: Tensor | None = None,
        *,
        response_token_ids: Tensor | None = None,
        response_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        value = self.value(belief)
        if self.policy_family == "mlp":
            if actions is None:
                raise ValueError("MLP policy requires discrete actions.")
            logits = self.policy_logits(belief)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), value

        assert self.freeform_policy is not None
        log_prob, entropy = self.freeform_policy.evaluate_actions(
            belief,
            actions,
            env_ids,
            response_token_ids=response_token_ids,
            response_mask=response_mask,
        )
        return log_prob, entropy, value

    def policy_logits(self, belief: BeliefState | Tensor) -> Tensor:
        if self.policy_readout is None or self.policy_net is None:
            raise RuntimeError("policy_logits is only available for the latent MLP policy family.")
        pooled = self.policy_readout(belief)
        return self.policy_net(pooled)

    def action_scores(
        self,
        belief: BeliefState | Tensor,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        """Return action scores (B, num_actions) for any policy family.

        Used by the lookahead planner for expansion selection and greedy rollout.
        - MLP: delegates to policy_logits (fast, single MLP forward)
        - VLM freeform: delegates to freeform_policy.action_logits (teacher-forcing scoring)
        """
        if self.policy_family == "mlp":
            return self.policy_logits(belief)
        assert self.freeform_policy is not None
        return self.freeform_policy.action_logits(belief, env_ids)

    def value(self, belief: BeliefState | Tensor) -> Tensor:
        pooled = self.value_readout(belief)
        return self.value_net(pooled).squeeze(-1)

    def policy_parameters(self) -> list[nn.Parameter]:
        if self.policy_family == "mlp":
            assert self.policy_readout is not None and self.policy_net is not None
            params = list(self.policy_readout.parameters()) + list(self.policy_net.parameters())
            return [param for param in params if param.requires_grad]

        assert self.freeform_policy is not None
        return [param for param in self.freeform_policy.parameters() if param.requires_grad]

    def value_parameters(self) -> list[nn.Parameter]:
        params = list(self.value_readout.parameters()) + list(self.value_net.parameters())
        return [param for param in params if param.requires_grad]


def map_reward_logits_to_scalar(
    reward_logits: Tensor,
    env_ids: Tensor | None,
    config: Config,
) -> Tensor:
    """Convert reward logits to scalar imagined rewards.

    Mapping mode (config.phase2.ppo.reward_mapping):
        sigmoid_affine: r = negative + sigmoid(logit) * (positive - negative)
        raw_sigmoid:    r = sigmoid(logit)   (range [0, 1])
        clipped_logit:  r = clamp(logit, -5, 5)
    """
    mapping = config.phase2.ppo.reward_mapping

    if mapping == "raw_sigmoid":
        return torch.sigmoid(reward_logits)

    if mapping == "clipped_logit":
        return torch.clamp(reward_logits, -5.0, 5.0)

    # sigmoid_affine (default)
    probs = torch.sigmoid(reward_logits)
    if env_ids is None:
        spec = config.env.reward_spec_for(config.env.env_ids[0])
        positive = reward_logits.new_full(probs.shape, spec.positive_value)
        negative = reward_logits.new_full(probs.shape, spec.negative_value)
        return negative + probs * (positive - negative)

    if env_ids.ndim != 1 or env_ids.shape[0] != reward_logits.shape[0]:
        raise ValueError(
            "env_ids must have shape (B,) matching reward logits batch size, got "
            f"env_ids={tuple(env_ids.shape)} reward_logits={tuple(reward_logits.shape)}."
        )

    positives = torch.empty_like(probs)
    negatives = torch.empty_like(probs)
    for index, env_name in enumerate(config.env.env_ids):
        spec = config.env.reward_spec_for(env_name)
        mask = env_ids == index
        if mask.any():
            positives[mask] = spec.positive_value
            negatives[mask] = spec.negative_value
    return negatives + probs * (positives - negatives)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute generalized advantage estimation for truncated imagined rollouts."""

    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (B, H), got {tuple(rewards.shape)}.")
    if values.ndim != 2 or values.shape[0] != rewards.shape[0] or values.shape[1] != rewards.shape[1] + 1:
        raise ValueError(
            "values must have shape (B, H + 1) aligned with rewards, got "
            f"rewards={tuple(rewards.shape)} values={tuple(values.shape)}."
        )

    batch_size, horizon = rewards.shape
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)

    for step in range(horizon - 1, -1, -1):
        delta = rewards[:, step] + gamma * values[:, step + 1] - values[:, step]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[:, step] = next_advantage

    returns = advantages + values[:, :-1]
    return advantages, returns


class Phase2Trainer:
    """Frozen-WM latent PPO trainer."""

    def __init__(
        self,
        *,
        config: Config,
        world_model: WorldModel,
        actor_critic: LatentActorCritic,
        data_source: OfflineDataSource,
        device: torch.device,
        logger: Phase2Logger | None = None,
    ) -> None:
        if config.phase2.algorithm != "ppo":
            raise ValueError(f"Unsupported Phase 2 algorithm: {config.phase2.algorithm}")
        if config.phase2.world_model_mode != "frozen_wm":
            raise NotImplementedError("Phase 2 currently implements frozen_wm only.")

        self.config = config
        self.world_model = world_model.to(device)
        self.actor_critic = actor_critic.to(device)
        self.data_source = data_source
        self.device = device
        self.logger = logger

        self.world_model.eval()
        self.world_model.requires_grad_(False)

        # Lookahead planner (inference-time only, does not affect PPO gradients)
        self.planner = make_planner(
            config,
            simulator=self.world_model,
            policy=self.actor_critic,
            reward_mapper=map_reward_logits_to_scalar,
        )

        ppo_cfg = config.phase2.ppo
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.actor_critic.policy_parameters(), "lr": ppo_cfg.actor_lr},
                {"params": self.actor_critic.value_parameters(), "lr": ppo_cfg.critic_lr},
            ],
            weight_decay=ppo_cfg.weight_decay,
        )

    def train(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        start_update: int = 0,
        tokenizer: Any | None = None,
    ) -> None:
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)

        import time as _time
        for update in range(start_update, self.config.phase2.ppo.total_updates):
            t0 = _time.time()
            rollout = self.collect_rollout()
            t_rollout = _time.time() - t0
            metrics = self.update_actor_critic(rollout)
            metrics.update(self._rollout_metrics(rollout))
            # Periodic real-environment evaluation
            if self._should_eval(update + 1):
                eval_metrics = self.evaluate_in_env(tokenizer=tokenizer)
                metrics.update(eval_metrics)
            if self.logger is not None:
                self.logger.log_scalars(update, metrics)
            if checkpoint_root is not None and self._should_save_checkpoint(update + 1):
                self.save_checkpoint(checkpoint_root / "latest.pt", update=update + 1)
            elapsed = _time.time() - t0
            rew = metrics.get("phase2/reward_mean", 0)
            sr = metrics.get("eval/success_rate", -1)
            sr_str = f" | eval_sr={sr:.2f}" if sr >= 0 else ""
            print(f"[update {update+1}/{self.config.phase2.ppo.total_updates}] "
                  f"rollout={t_rollout:.1f}s total={elapsed:.1f}s "
                  f"rew={rew:.4f} ploss={metrics.get('phase2/policy_loss', 0):.4f}"
                  f"{sr_str}", flush=True)

    @torch.no_grad()
    def collect_rollout(self) -> RolloutBatch:
        """Ground a batch of real observations and imagine a short PPO trajectory."""

        ppo_cfg = self.config.phase2.ppo
        batch = self.data_source.sample_batch(ppo_cfg.rollout_batch_size)
        batch = self._to_device(batch)

        belief = self._ground_initial_belief(batch)
        beliefs: list[Tensor] = []
        actions: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        response_token_ids: list[Tensor] = []
        response_masks: list[Tensor] = []
        old_token_log_probs: list[Tensor] = []
        parse_successes: list[Tensor] = []
        rewards: list[Tensor] = []
        values: list[Tensor] = []

        for _ in range(ppo_cfg.rollout_horizon):
            # If lookahead planner is active, it overrides action selection.
            # We still call actor_critic.act() to get log_prob/value/extras
            # for PPO — the planner only decides *which* action to execute.
            if self.planner is not None and self.planner.enabled:
                planned_action = self.planner.select_actions(belief, batch.env_ids)
                action, log_prob, _entropy, value, extras = self.actor_critic.act(belief, batch.env_ids)
                action = planned_action  # override with planner's choice
            else:
                action, log_prob, _entropy, value, extras = self.actor_critic.act(belief, batch.env_ids)
            next_belief = self.world_model.prior_step(
                prev_belief=belief,
                prev_actions=action,
                env_ids=batch.env_ids,
            )
            reward_logits = self.world_model.predict_reward(next_belief)
            reward = map_reward_logits_to_scalar(reward_logits, batch.env_ids, self.config)

            beliefs.append(belief.slots.detach())
            actions.append(action.detach())
            old_log_probs.append(log_prob.detach())
            if extras is not None:
                response_token_ids.append(extras["response_token_ids"].detach())
                response_masks.append(extras["response_mask"].detach())
                old_token_log_probs.append(extras["old_token_log_probs"].detach())
                parse_successes.append(extras["parse_success"].detach())
            rewards.append(reward.detach())
            values.append(value.detach())
            belief = next_belief

        bootstrap_value = self.actor_critic.value(belief).detach()
        value_tensor = torch.stack(values + [bootstrap_value], dim=1)
        reward_tensor = torch.stack(rewards, dim=1)
        advantage_tensor, return_tensor = compute_gae(
            reward_tensor,
            value_tensor,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        return RolloutBatch(
            beliefs=torch.stack(beliefs, dim=1),
            env_ids=None if batch.env_ids is None else batch.env_ids.detach(),
            actions=torch.stack(actions, dim=1),
            rewards=reward_tensor,
            values=value_tensor,
            advantages=advantage_tensor.detach(),
            returns=return_tensor.detach(),
            old_log_probs=torch.stack(old_log_probs, dim=1),
            response_token_ids=self._pad_and_stack(response_token_ids) if response_token_ids else None,
            response_mask=self._pad_and_stack(response_masks) if response_masks else None,
            old_token_log_probs=self._pad_and_stack(old_token_log_probs, pad_value=0.0) if old_token_log_probs else None,
            parse_success=torch.stack(parse_successes, dim=1) if parse_successes else None,
        )

    def update_actor_critic(self, rollout: RolloutBatch) -> dict[str, float]:
        """Run PPO updates against one imagined rollout batch."""

        ppo_cfg = self.config.phase2.ppo
        beliefs = rollout.beliefs.reshape(-1, rollout.beliefs.shape[-2], rollout.beliefs.shape[-1])
        actions = rollout.actions.reshape(-1)
        advantages = rollout.advantages.reshape(-1)
        returns = rollout.returns.reshape(-1)
        old_values = rollout.values[:, :-1].reshape(-1)
        env_ids = None if rollout.env_ids is None else rollout.env_ids.unsqueeze(1).expand(-1, rollout.horizon).reshape(-1)
        use_token_ppo = (
            self.actor_critic.policy_family == "vlm_freeform"
            and self.actor_critic.freeform_update_mode == "token_ppo"
        )
        old_log_probs = None if rollout.old_log_probs is None else rollout.old_log_probs.reshape(-1)
        response_token_ids = None
        response_mask = None
        old_token_log_probs = None
        if use_token_ppo:
            if rollout.response_token_ids is None or rollout.response_mask is None or rollout.old_token_log_probs is None:
                raise ValueError("token_ppo rollout is missing response token statistics.")
            response_token_ids = rollout.response_token_ids.reshape(-1, rollout.response_token_ids.shape[-1])
            response_mask = rollout.response_mask.reshape(-1, rollout.response_mask.shape[-1])
            old_token_log_probs = rollout.old_token_log_probs.reshape(-1, rollout.old_token_log_probs.shape[-1])

        if ppo_cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_kl = 0.0
        num_minibatches = 0

        total_items = actions.shape[0]
        minibatch_size = min(ppo_cfg.minibatch_size, total_items)

        for _ in range(ppo_cfg.epochs_per_update):
            permutation = torch.randperm(total_items, device=beliefs.device)
            for start in range(0, total_items, minibatch_size):
                indices = permutation[start:start + minibatch_size]
                mb_beliefs = beliefs[indices]
                mb_actions = actions[indices]
                mb_advantages = advantages[indices]
                mb_returns = returns[indices]
                mb_old_values = old_values[indices]
                mb_env_ids = None if env_ids is None else env_ids[indices]
                if use_token_ppo:
                    assert response_token_ids is not None and response_mask is not None and old_token_log_probs is not None
                    mb_response_token_ids = response_token_ids[indices]
                    mb_response_mask = response_mask[indices]
                    mb_old_token_log_probs = old_token_log_probs[indices]
                    new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                        BeliefState(slots=mb_beliefs),
                        None,
                        mb_env_ids,
                        response_token_ids=mb_response_token_ids,
                        response_mask=mb_response_mask,
                    )
                    token_mask = mb_response_mask.to(dtype=new_log_probs.dtype)
                    adv_matrix = mb_advantages.unsqueeze(1).expand_as(new_log_probs)
                    ratio = torch.exp(new_log_probs - mb_old_token_log_probs)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - ppo_cfg.clip_epsilon,
                        1.0 + ppo_cfg.clip_epsilon,
                    )
                    token_objective = torch.min(
                        ratio * adv_matrix,
                        clipped_ratio * adv_matrix,
                    )
                    valid_tokens = token_mask.sum().clamp_min(1.0)
                    policy_loss = -(token_objective * token_mask).sum() / valid_tokens
                    entropy_mean = (entropy * token_mask).sum() / valid_tokens
                    clipfrac = (((torch.abs(ratio - 1.0) > ppo_cfg.clip_epsilon).float()) * token_mask).sum() / valid_tokens
                    # KL(old || new) ≈ mean of (old_logp - new_logp) over valid tokens
                    kl_div = ((mb_old_token_log_probs - new_log_probs) * token_mask).sum() / valid_tokens
                else:
                    assert old_log_probs is not None
                    mb_old_log_probs = old_log_probs[indices]
                    new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                        BeliefState(slots=mb_beliefs),
                        mb_actions,
                        mb_env_ids,
                    )
                    entropy_mean = entropy.mean()

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - ppo_cfg.clip_epsilon,
                        1.0 + ppo_cfg.clip_epsilon,
                    )
                    policy_loss = -torch.min(
                        ratio * mb_advantages,
                        clipped_ratio * mb_advantages,
                    ).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > ppo_cfg.clip_epsilon).float().mean()
                    # KL(old || new) ≈ mean of (old_logp - new_logp)
                    kl_div = (mb_old_log_probs - new_log_probs).mean()
                value_loss = F.mse_loss(values, mb_returns)

                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy_mean
                    + ppo_cfg.kl_coef * kl_div
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), ppo_cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.detach().item())
                total_value_loss += float(value_loss.detach().item())
                total_entropy += float(entropy_mean.detach().item())
                total_clipfrac += float(clipfrac.detach().item())
                total_kl += float(kl_div.detach().item())
                num_minibatches += 1

        updated_values = self.actor_critic.value(BeliefState(slots=beliefs)).detach()
        explained_variance = _explained_variance(updated_values, returns)
        value_delta = (updated_values - old_values).abs().mean()

        return {
            "phase2/policy_loss": total_policy_loss / max(1, num_minibatches),
            "phase2/value_loss": total_value_loss / max(1, num_minibatches),
            "phase2/entropy": total_entropy / max(1, num_minibatches),
            "phase2/clipfrac": total_clipfrac / max(1, num_minibatches),
            "phase2/kl_divergence": total_kl / max(1, num_minibatches),
            "phase2/explained_variance": explained_variance,
            "phase2/value_delta": float(value_delta.item()),
        }

    def save_checkpoint(self, path: str | Path, *, update: int) -> None:
        checkpoint = {
            "update": update,
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, Path(path))

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint.get("update", 0))

    def _ground_initial_belief(self, batch: SequenceBatch) -> BeliefState:
        batch_size = batch.obs_tokens.shape[0]
        param_dtype = next(self.world_model.parameters()).dtype
        initial_belief = self.world_model.get_initial_belief(
            batch_size,
            device=self.device,
            dtype=param_dtype,
        )
        null_actions = torch.full(
            (batch_size,),
            fill_value=self.config.env.null_action_id,
            device=self.device,
            dtype=torch.long,
        )
        return self.world_model.posterior_step(
            prev_belief=initial_belief,
            prev_actions=null_actions,
            observation_tokens=batch.obs_tokens[:, 0],
            env_ids=batch.env_ids,
        )

    def _to_device(self, batch: SequenceBatch) -> SequenceBatch:
        return SequenceBatch(
            obs_tokens=batch.obs_tokens.to(self.device),
            actions=batch.actions.to(self.device),
            rewards=batch.rewards.to(self.device),
            episode_lengths=batch.episode_lengths.to(self.device),
            env_ids=None if batch.env_ids is None else batch.env_ids.to(self.device),
        )

    @staticmethod
    def _pad_and_stack(tensors: list[Tensor], pad_value: float = 0) -> Tensor:
        """Pad variable-length tensors along the last dim, then stack along dim=1."""
        max_len = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            pad_size = max_len - t.shape[-1]
            if pad_size > 0:
                t = torch.nn.functional.pad(t, (0, pad_size), value=pad_value)
            padded.append(t)
        return torch.stack(padded, dim=1)

    def _rollout_metrics(self, rollout: RolloutBatch) -> dict[str, float]:
        metrics = {
            "phase2/reward_mean": float(rollout.rewards.mean().item()),
            "phase2/reward_std": float(rollout.rewards.std().item()),
            "phase2/adv_mean": float(rollout.advantages.mean().item()),
            "phase2/return_mean": float(rollout.returns.mean().item()),
            "phase2/value_mean": float(rollout.values[:, :-1].mean().item()),
        }
        if rollout.parse_success is not None:
            metrics["phase2/parse_success_rate"] = float(rollout.parse_success.float().mean().item())
        return metrics

    def _should_save_checkpoint(self, next_update: int) -> bool:
        frequency = self.config.phase2.ppo.checkpoint_every
        return frequency > 0 and next_update % frequency == 0

    def _should_eval(self, next_update: int) -> bool:
        frequency = self.config.phase2.ppo.eval_every
        return frequency > 0 and next_update % frequency == 0

    @torch.no_grad()
    def evaluate_in_env(
        self,
        *,
        num_episodes: int | None = None,
        max_steps: int | None = None,
        tokenizer: Any | None = None,
    ) -> dict[str, float]:
        """Run the current policy in a real Sokoban environment.

        If *tokenizer* is None, observations are tokenized using a simple
        learned linear projection from flattened RGB — this is a lightweight
        fallback when V-JEPA 2 is not loaded.  For accurate evaluation,
        pass an ``ImageTokenizer`` instance.

        Returns metrics dict with keys:
            eval/success_rate, eval/avg_return, eval/avg_episode_length,
            eval/num_success, eval/num_episodes
        """
        from ..data.adapters.sokoban import SokobanEnv

        ppo_cfg = self.config.phase2.ppo
        n_episodes = num_episodes or ppo_cfg.eval_episodes
        max_ep_steps = max_steps or ppo_cfg.eval_max_steps

        env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=max_ep_steps)

        successes = 0
        total_return = 0.0
        total_steps = 0

        self.actor_critic.eval()

        for ep in range(n_episodes):
            if ep % 16 == 0:
                print(f"  [eval] episode {ep}/{n_episodes}, successes so far: {successes}", flush=True)
            obs = env.reset(seed=ep + 10000)  # deterministic seeds for reproducibility
            done = False
            ep_return = 0.0
            ep_steps = 0

            # Get initial belief by tokenizing first observation
            obs_tokens = self._tokenize_obs(obs, tokenizer)
            param_dtype = next(self.world_model.parameters()).dtype
            belief = self.world_model.get_initial_belief(
                1, device=self.device, dtype=param_dtype,
            )
            null_action = torch.tensor(
                [self.config.env.null_action_id], device=self.device, dtype=torch.long,
            )
            belief = self.world_model.posterior_step(
                prev_belief=belief,
                prev_actions=null_action,
                observation_tokens=obs_tokens.unsqueeze(0).to(self.device, param_dtype),
                env_ids=None,
            )

            while not done and ep_steps < max_ep_steps:
                # Select action
                if self.planner is not None and self.planner.enabled:
                    action = self.planner.select_actions(belief, None)
                else:
                    action, _, _, _, _ = self.actor_critic.act(belief, None)

                # Step real environment (model actions 0-3 → env actions 1-4)
                env_action = int(action.item()) + 1
                obs, reward, done, info = env.step(env_action)
                ep_return += reward
                ep_steps += 1

                if not done:
                    # Update belief with new observation
                    obs_tokens = self._tokenize_obs(obs, tokenizer)
                    belief = self.world_model.posterior_step(
                        prev_belief=belief,
                        prev_actions=action,
                        observation_tokens=obs_tokens.unsqueeze(0).to(self.device, param_dtype),
                        env_ids=None,
                    )

            if info.get("success", False):
                successes += 1
            total_return += ep_return
            total_steps += ep_steps

        self.actor_critic.train()

        return {
            "eval/success_rate": successes / max(1, n_episodes),
            "eval/avg_return": total_return / max(1, n_episodes),
            "eval/avg_episode_length": total_steps / max(1, n_episodes),
            "eval/num_success": float(successes),
            "eval/num_episodes": float(n_episodes),
        }

    def _tokenize_obs(self, obs: Any, tokenizer: Any | None) -> Tensor:
        """Tokenize a single RGB observation into (K, D) tensor.

        If a real tokenizer (ImageTokenizer) is provided, use it.
        Otherwise, use a simple resize + linear projection as fallback.
        """
        if tokenizer is not None:
            return tokenizer.tokenize(obs)

        # Lightweight fallback: resize to small grid, flatten, project to D
        # This won't match precomputed V-JEPA tokens but allows eval without
        # loading V-JEPA 2 (~5GB GPU).
        import numpy as np
        from PIL import Image

        img = Image.fromarray(obs).resize((16, 16))
        flat = torch.from_numpy(np.array(img, dtype=np.float32)).flatten() / 255.0
        # Project to (K, D) with a cached fixed random projection
        K = self.config.encoder.compressed_tokens  # 36
        D = self.config.hidden_dim  # 3584
        if not hasattr(self, "_eval_proj"):
            gen = torch.Generator().manual_seed(42)
            self._eval_proj = torch.randn(flat.shape[0], K * D, generator=gen) * 0.01
        tokens = (flat @ self._eval_proj).reshape(K, D)
        return tokens


def _explained_variance(values: Tensor, targets: Tensor) -> float:
    target_var = torch.var(targets)
    if not torch.isfinite(target_var) or target_var.item() < 1e-8:
        return 0.0
    residual = torch.var(targets - values)
    return float((1.0 - residual / target_var).item())
