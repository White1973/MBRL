"""Qwen-backed transition backbone.

This module provides the concrete `TransitionBackbone` implementation for the
shared VLM backbone used by posterior and prior transitions. It keeps
`transformers` and `peft` as optional runtime dependencies so the lightweight
CPU smoke path can still import the package without the VLM extras installed.
"""
from __future__ import annotations

import copy
import inspect
from typing import Any, Literal

import torch
from torch import Tensor

from ..config import Config
from .transition import TransitionBackbone


QwenDType = Literal["bf16", "fp32"]
WM_LORA_ADAPTER = "default"


def training_dtype(name: QwenDType) -> torch.dtype:
    """Resolve the configured training dtype to a torch dtype."""

    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported training dtype: {name}")


def _resolve_attention_implementation(
    *,
    attention_mode: str,
    requested: str | None,
) -> str | None:
    """Choose a backend compatible with the configured attention semantics.

    FlashAttention2 only accepts its specialized 2-D padding/varlen masks. It
    cannot consume the explicit 4-D additive mask required to bypass Qwen's
    internally-created causal mask. SDPA supports that mask and still uses
    fused CUDA kernels where possible.
    """
    if attention_mode == "bidirectional" and requested not in {"sdpa", "eager"}:
        return "sdpa"
    return requested


class QwenTransitionBackbone(TransitionBackbone):
    """Qwen2.5-VL language backbone adapted to token-to-token transitions.

    The wrapper accepts already embedded world-model tokens `(B, S, D)` via
    `inputs_embeds` and returns the final hidden states with the same shape.
    Native Qwen visual preprocessing happens once in ``QwenVisionTokenizer``
    and its image embeddings are supplied here as ``inputs_embeds``. Keeping
    extraction outside the recurrent transition avoids re-running the vision
    tower at every BPTT step while retaining Qwen2.5-VL—not V-JEPA—as the
    observation representation.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        hidden_dim: int,
        compute_dtype: torch.dtype = torch.bfloat16,
        attention_mode: Literal["bidirectional", "causal"] = "bidirectional",
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.transformer = _inner_transformer(model)
        # Keep ``transformer`` registered exactly as in historical
        # checkpoints. The mask-aware decoder is only a non-owning execution
        # reference; registering the inner language model here would rename
        # thousands of state-dict keys and break Phase-1/Phase-2 checkpoints.
        object.__setattr__(
            self,
            "_forward_transformer",
            _mask_aware_transformer(self.transformer),
        )
        self.hidden_dim = hidden_dim
        self.compute_dtype = compute_dtype
        self.attention_mode = attention_mode
        self._lora_parameter_cache: dict[
            str, list[torch.nn.Parameter]
        ] = {}
        self._all_lora_parameter_cache: list[
            torch.nn.Parameter
        ] | None = None

        if attention_mode == "bidirectional":
            _patch_bidirectional_attention(self.transformer)
        elif attention_mode != "causal":
            raise ValueError(f"Unsupported attention mode: {attention_mode}")

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        device_map: str | dict[str, int | str] | None = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
    ) -> "QwenTransitionBackbone":
        """Load Qwen2.5-VL with frozen base weights and LoRA adapters."""

        transformers, peft = _import_vlm_dependencies()
        dtype = training_dtype(config.training.dtype)

        # Use config-level attn_implementation if caller didn't override
        if attn_implementation is None:
            attn_implementation = config.backbone.attn_implementation
        attn_implementation = _resolve_attention_implementation(
            attention_mode=config.backbone.attention_mode,
            requested=attn_implementation,
        )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Optional 4-bit quantization (QLoRA)
        if config.backbone.quantization == "4bit":
            model_kwargs["quantization_config"] = _make_bnb_config(dtype)

        model = _load_qwen_model(
            transformers=transformers,
            model_name=config.backbone.model_name,
            model_kwargs=model_kwargs,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.backbone.model_name,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        _validate_hidden_size(model, config.hidden_dim)

        lora_config = peft.LoraConfig(
            r=config.backbone.lora_rank,
            lora_alpha=config.backbone.lora_alpha,
            lora_dropout=config.backbone.lora_dropout,
            target_modules=config.backbone.lora_target_modules,
            bias="none",
            task_type=_lora_task_type(peft),
        )
        # Keep the historical ``default`` name for Phase-1 checkpoint
        # compatibility. Phase 2 adds actor/critic adapters to this same PEFT
        # container instead of loading additional Qwen base models.
        model = peft.get_peft_model(
            model,
            lora_config,
            adapter_name=WM_LORA_ADAPTER,
        )
        model.config.use_cache = False

        # Gradient checkpointing: recompute activations to save ~40-60 GB VRAM
        if config.backbone.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        return cls(
            model=model,
            tokenizer=tokenizer,
            hidden_dim=config.hidden_dim,
            compute_dtype=dtype,
            attention_mode=config.backbone.attention_mode,
        )

    @property
    def lora_adapter_names(self) -> tuple[str, ...]:
        configs = getattr(self.model, "peft_config", {})
        return tuple(configs.keys())

    def add_lora_adapter(
        self,
        adapter_name: str,
        *,
        initialize_from: str = WM_LORA_ADAPTER,
        lora_dropout: float | None = None,
    ) -> None:
        """Add an independent LoRA set while retaining one frozen Qwen base.

        Phase-2 policy adapters use ``lora_dropout=0``. PPO requires the old
        and current policy log probabilities to be comparable, while active
        LoRA dropout otherwise makes an unchanged actor look as if it already
        moved. The adapter remains fully trainable; only stochastic masking is
        disabled.
        """
        configs = getattr(self.model, "peft_config", None)
        add_adapter = getattr(self.model, "add_adapter", None)
        if configs is None or not callable(add_adapter):
            raise TypeError("Qwen backbone is not a multi-adapter PEFT model")
        if adapter_name in configs:
            raise ValueError(f"LoRA adapter already exists: {adapter_name!r}")
        if initialize_from not in configs:
            raise ValueError(
                f"Cannot initialize {adapter_name!r}: source adapter "
                f"{initialize_from!r} does not exist"
            )

        # PEFT state helpers normalize away the source adapter name, allowing
        # the same tensors to be installed under the new destination name.
        from peft import (  # type: ignore[import-not-found]
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )

        source_state = get_peft_model_state_dict(
            self.model,
            adapter_name=initialize_from,
        )
        adapter_config = copy.deepcopy(configs[initialize_from])
        if lora_dropout is not None:
            if not 0.0 <= lora_dropout < 1.0:
                raise ValueError("lora_dropout must be in [0, 1)")
            adapter_config.lora_dropout = float(lora_dropout)
        add_adapter(adapter_name, adapter_config)
        set_peft_model_state_dict(
            self.model,
            source_state,
            adapter_name=adapter_name,
        )
        self._lora_parameter_cache = {}
        self._all_lora_parameter_cache = None
        # Adding an adapter must not implicitly alter the Phase-1 ownership
        # mask. Assembly explicitly enables only the PPO-owned adapter params.
        self.set_lora_adapter_trainable(adapter_name, False)

    def copy_lora_adapter(self, source: str, destination: str) -> None:
        """Copy adapter tensors after loading a legacy single-adapter model."""
        configs = getattr(self.model, "peft_config", {})
        if source not in configs or destination not in configs:
            raise ValueError(
                f"Cannot copy LoRA adapter {source!r}->{destination!r}; "
                f"available={tuple(configs)}"
            )
        from peft import (  # type: ignore[import-not-found]
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        state = get_peft_model_state_dict(self.model, adapter_name=source)
        set_peft_model_state_dict(
            self.model, state, adapter_name=destination
        )

    def lora_adapter_parameters(
        self,
        adapter_name: str,
    ) -> list[torch.nn.Parameter]:
        if adapter_name not in self.lora_adapter_names:
            raise ValueError(f"Unknown LoRA adapter: {adapter_name!r}")
        cache = getattr(self, "_lora_parameter_cache", None)
        if cache is None:
            cache = {}
            self._lora_parameter_cache = cache
        if adapter_name in cache:
            return list(cache[adapter_name])
        marker = f".{adapter_name}."
        parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if "lora_" in name.lower() and marker in name
        ]
        if not parameters:
            raise RuntimeError(
                f"PEFT adapter {adapter_name!r} has no LoRA parameters"
            )
        cache[adapter_name] = parameters
        return list(parameters)

    def _all_lora_parameters(self) -> list[torch.nn.Parameter]:
        cached = getattr(self, "_all_lora_parameter_cache", None)
        if cached is not None:
            return list(cached)
        parameters: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for adapter_name in self.lora_adapter_names:
            for parameter in self.lora_adapter_parameters(adapter_name):
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        self._all_lora_parameter_cache = parameters
        return list(parameters)

    def set_lora_adapter_trainable(
        self,
        adapter_name: str,
        trainable: bool,
    ) -> None:
        for parameter in self.lora_adapter_parameters(adapter_name):
            parameter.requires_grad_(trainable)

    def _activate_lora_adapter(self, adapter_name: str) -> None:
        """Select an adapter without letting PEFT mutate ownership masks.

        ``PeftModel.set_adapter`` makes the selected adapter trainable as a
        side effect. That is unsafe for frozen WM rollouts and frozen_vlm
        branches, so preserve every adapter's current requires-grad state.
        """
        set_adapter = getattr(self.model, "set_adapter", None)
        if not callable(set_adapter):
            raise TypeError("Qwen backbone does not support PEFT adapters")
        adapter_parameters = self._all_lora_parameters()
        trainability = [
            parameter.requires_grad for parameter in adapter_parameters
        ]
        set_adapter(adapter_name)
        for parameter, requires_grad in zip(
            adapter_parameters,
            trainability,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

    def forward_with_adapter(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        adapter_name: str,
    ) -> Tensor:
        self._activate_lora_adapter(adapter_name)
        return self._forward_hidden(tokens, attention_mask=attention_mask)

    def forward(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.forward_with_adapter(
            tokens,
            attention_mask=attention_mask,
            adapter_name=WM_LORA_ADAPTER,
        )

    def _forward_hidden(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"QwenTransitionBackbone expects tokens with shape (B, S, D), got {tuple(tokens.shape)}."
            )
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dim {self.hidden_dim}, got {tokens.shape[-1]}."
            )

        input_dtype = tokens.dtype
        token_device = next(self.model.parameters()).device
        inputs_embeds = tokens.to(device=token_device, dtype=self.compute_dtype)

        batch_size, seq_len, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        else:
            attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.long)
        transformer_attention_mask: Tensor | dict[str, Tensor]
        if self.attention_mode == "bidirectional":
            # Qwen2.5-VL 4.54 constructs a causal 4-D mask inside the text
            # model whenever it receives a conventional 2-D padding mask.
            # Flipping ``self_attn.is_causal`` is therefore insufficient: the
            # already-created additive mask still prevents an earlier belief
            # token from attending to the action tokens appended after it.
            # Passing the mask mapping expected by Qwen bypasses that internal
            # causal-mask construction and makes the configured attention mode
            # real rather than cosmetic.
            transformer_attention_mask = _bidirectional_mask_mapping(
                attention_mask,
                dtype=inputs_embeds.dtype,
            )
        else:
            transformer_attention_mask = attention_mask
        outputs = self._forward_transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=transformer_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = _last_hidden_state(outputs)
        if hidden.shape != inputs_embeds.shape:
            raise ValueError(
                "Qwen backbone must preserve token sequence shape, "
                f"got hidden={tuple(hidden.shape)} inputs={tuple(inputs_embeds.shape)}."
            )
        return hidden.to(dtype=input_dtype)

    def tokenize_texts(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def embed_text_token_ids(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                f"embed_text_token_ids expects token ids with shape (B, L), got {tuple(token_ids.shape)}."
            )
        token_device = next(self.model.parameters()).device
        embedding_layer = self.model.get_input_embeddings()
        embedded = embedding_layer(token_ids.to(device=token_device, dtype=torch.long))
        return embedded.to(dtype=self.compute_dtype)


def _import_vlm_dependencies() -> tuple[Any, Any]:
    try:
        import peft  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "QwenTransitionBackbone requires the optional VLM dependencies. "
            "Install them with `uv sync --extra vlm`."
        ) from exc
    return transformers, peft


def _make_bnb_config(compute_dtype: torch.dtype) -> Any:
    """Create BitsAndBytes 4-bit quantization config for QLoRA."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "4-bit quantization requires bitsandbytes. "
            "Install with: pip install bitsandbytes"
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _load_qwen_model(
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


def _lora_task_type(peft: Any) -> Any:
    task_type = getattr(peft, "TaskType", None)
    if task_type is not None and hasattr(task_type, "FEATURE_EXTRACTION"):
        return task_type.FEATURE_EXTRACTION

    # Older PEFT releases may not expose FEATURE_EXTRACTION. CAUSAL_LM is a
    # fallback wrapper only; target_modules is explicit, and the model is still
    # called through inputs_embeds for token-to-token feature extraction.
    return "CAUSAL_LM"


def _validate_hidden_size(model: torch.nn.Module, expected_hidden_dim: int) -> None:
    actual_hidden_dim = _model_hidden_size(getattr(model, "config", None))
    if actual_hidden_dim is None:
        return
    if actual_hidden_dim != expected_hidden_dim:
        raise ValueError(
            "Qwen hidden size must match Config.hidden_dim, "
            f"got model hidden_size={actual_hidden_dim} config hidden_dim={expected_hidden_dim}."
        )


def _model_hidden_size(config: Any) -> int | None:
    if config is None:
        return None
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size

    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", None)
    if isinstance(text_hidden_size, int):
        return text_hidden_size
    return None


def _inner_transformer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the hidden-state transformer under PEFT/Qwen wrapper layers."""

    candidates: list[tuple[str, torch.nn.Module]] = []

    def add_candidate(path: str, value: Any) -> None:
        if isinstance(value, torch.nn.Module):
            candidates.append((path, value))

    add_candidate("model", model)
    if hasattr(model, "get_base_model"):
        try:
            add_candidate("model.get_base_model()", model.get_base_model())
        except Exception:
            pass

    base_model = getattr(model, "base_model", None)
    add_candidate("model.base_model", base_model)

    peft_model = getattr(base_model, "model", None)
    add_candidate("model.base_model.model", peft_model)

    nested_model = getattr(peft_model, "model", None)
    add_candidate("model.base_model.model.model", nested_model)

    deeper_model = getattr(nested_model, "model", None)
    add_candidate("model.base_model.model.model.model", deeper_model)

    for path, candidate in reversed(candidates):
        if _looks_like_hidden_transformer(candidate):
            return candidate

    candidate_types = ", ".join(
        f"{path}: {type(candidate).__name__}" for path, candidate in candidates
    )
    raise TypeError(
        "Could not locate a Qwen hidden-state transformer that accepts "
        f"`inputs_embeds`. Checked candidates: {candidate_types}."
    )


def _mask_aware_transformer(transformer: torch.nn.Module) -> torch.nn.Module:
    """Select Qwen's decoder while preserving historical module ownership."""
    language_model = getattr(transformer, "language_model", None)
    if isinstance(language_model, torch.nn.Module):
        return language_model
    return transformer


def _looks_like_hidden_transformer(module: torch.nn.Module) -> bool:
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return False

    params = signature.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if not accepts_kwargs and "inputs_embeds" not in params:
        return False

    if hasattr(module, "lm_head"):
        return False
    return True


def _patch_bidirectional_attention(module: torch.nn.Module) -> None:
    """Best-effort patch for decoder modules that expose causal flags.

    Decoder-only Transformer implementations differ in where they store causal
    state. This intentionally avoids monkey-patching forward methods; it only
    flips conventional boolean flags when present. A real-load smoke should
    still verify the attention semantics for the installed transformers version.
    """

    for child in module.modules():
        for attr in ("is_causal", "causal"):
            if hasattr(child, attr):
                try:
                    setattr(child, attr, False)
                except Exception:
                    pass

        config = getattr(child, "config", None)
        if config is not None and hasattr(config, "is_causal"):
            try:
                setattr(config, "is_causal", False)
            except Exception:
                pass


def _bidirectional_mask_mapping(
    attention_mask: Tensor,
    *,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Build Qwen's explicit non-causal additive attention masks.

    The returned tensors have shape ``(B, 1, Q, K)``. Every non-padding query
    can attend to every non-padding key, including keys to its right. Qwen's
    text model accepts a mapping directly and consequently skips
    ``create_causal_mask``. Both keys are supplied because Qwen2.5-VL may mix
    full- and sliding-attention decoder layers.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            "Bidirectional Qwen attention expects a 2-D padding mask, got "
            f"shape {tuple(attention_mask.shape)}"
        )
    batch_size, seq_len = attention_mask.shape
    valid_keys = attention_mask.to(dtype=torch.bool)[:, None, None, :]
    additive = torch.zeros(
        batch_size,
        1,
        seq_len,
        seq_len,
        dtype=dtype,
        device=attention_mask.device,
    )
    additive.masked_fill_(~valid_keys, torch.finfo(dtype).min)
    return {
        "full_attention": additive,
        "sliding_attention": additive,
    }


def _last_hidden_state(outputs: Any) -> Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise TypeError(
        "Qwen transformer output did not expose `last_hidden_state` or tuple[0]."
    )
