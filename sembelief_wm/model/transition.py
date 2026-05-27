"""Transition core for SemBelief-WM.

This module defines the latent transition contract of the world model:
- posterior_step(prev_belief, prev_action, observation_tokens) -> next belief
- prior_step(prev_belief, prev_action) -> next belief
- rollout_prior(start_belief, action_seq) -> belief trajectory

The real Qwen backbone wrapper is intentionally not implemented here.
At this stage the transition core depends only on an abstract backbone
that maps token sequences (B, S, D) -> hidden states (B, S, D).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from ..types import BeliefState, BeliefTrajectory


class TransitionBackbone(nn.Module, ABC):
    """Abstract token-to-token backbone used by posterior and prior transitions."""

    @abstractmethod
    def forward(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Map input tokens of shape (B, S, D) to hidden states of shape (B, S, D)."""

    def tokenize_texts(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support tokenizer-backed text actions."
        )

    def embed_text_token_ids(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not support token-id text embeddings."
        )


class TokenTypeEmbedding(nn.Module):
    """Add learned token-type embeddings for visual, belief, and action tokens."""

    VISUAL = 0
    BELIEF = 1
    ACTION = 2
    NUM_TOKEN_TYPES = 3

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(self.NUM_TOKEN_TYPES, hidden_dim)
        # Small-variance init matching main branch
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, token_type: int) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"TokenTypeEmbedding expects (B, S, D), got {tuple(tokens.shape)}."
            )
        type_ids = torch.full(
            (tokens.shape[0], tokens.shape[1]),
            fill_value=token_type,
            dtype=torch.long,
            device=tokens.device,
        )
        return tokens + self.embedding(type_ids)


class ActionEncoder(nn.Module):
    """Encode discrete actions to hidden vectors."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._null_action_id = config.env.null_action_id
        num_action_ids = config.env.null_action_id + 1
        self.embedding = nn.Embedding(num_action_ids, config.hidden_dim)
        # Initialize null action embedding to zero (matches main branch's
        # zero-vector first action at step 0)
        with torch.no_grad():
            self.embedding.weight[config.env.null_action_id].zero_()

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 1:
            raise ValueError(
                f"ActionEncoder expects actions with shape (B,), got {tuple(actions.shape)}."
            )
        result = self.embedding(actions)
        # Null actions always produce zero vector (not trainable)
        null_mask = (actions == self._null_action_id).unsqueeze(-1)
        return result * (~null_mask).to(dtype=result.dtype)


@dataclass(frozen=True)
class ActionCondition:
    """Conditioning tokens and mask to append to the transition input."""

    tokens: Tensor
    attention_mask: Tensor


class TextActionConditioner(nn.Module):
    """Map discrete actions to pre-tokenized natural-language action tokens."""

    def __init__(self, config: Config, backbone: TransitionBackbone) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_envs = len(config.env.env_ids)
        self.num_action_ids = config.env.null_action_id + 1
        self.null_action_id = config.env.null_action_id
        # Keep a direct backbone reference for text-token embedding without
        # re-registering the backbone as a child module of the conditioner.
        # TransitionCore already owns the backbone; duplicating registration
        # here would make parameter traversal and state dict structure messy.
        object.__setattr__(self, "_backbone_ref", backbone)

        action_texts_by_env = self._build_action_texts(config)
        flat_texts = [
            text
            for texts in action_texts_by_env
            for text in texts
        ]
        token_ids, attention_mask = backbone.tokenize_texts(flat_texts)
        if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
            raise ValueError(
                "tokenize_texts must return token_ids and attention_mask with shape (N, L), "
                f"got token_ids={tuple(token_ids.shape)} attention_mask={tuple(attention_mask.shape)}."
            )
        seq_len = token_ids.shape[1]
        self.register_buffer(
            "token_ids_table",
            token_ids.reshape(self.num_envs, self.num_action_ids, seq_len),
            persistent=False,
        )
        self.register_buffer(
            "attention_mask_table",
            attention_mask.reshape(self.num_envs, self.num_action_ids, seq_len),
            persistent=False,
        )

    def forward(self, actions: Tensor, env_ids: Tensor | None) -> ActionCondition:
        if actions.ndim != 1:
            raise ValueError(
                f"TextActionConditioner expects actions with shape (B,), got {tuple(actions.shape)}."
            )
        if env_ids is None:
            env_ids = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
        if env_ids.ndim != 1 or env_ids.shape[0] != actions.shape[0]:
            raise ValueError(
                "env_ids must have shape (B,) matching actions batch size, got "
                f"env_ids={tuple(env_ids.shape)} actions={tuple(actions.shape)}."
            )
        if env_ids.dtype != torch.long:
            env_ids = env_ids.to(dtype=torch.long)
        if actions.dtype != torch.long:
            actions = actions.to(dtype=torch.long)

        table_device = self.token_ids_table.device
        env_ids = env_ids.to(device=table_device)
        actions = actions.to(device=table_device)

        token_ids = self.token_ids_table[env_ids, actions]
        attention_mask = self.attention_mask_table[env_ids, actions]
        tokens = self._embed_text_token_ids(token_ids)
        attention_mask = attention_mask.to(device=tokens.device)
        tokens = tokens * attention_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return ActionCondition(tokens=tokens, attention_mask=attention_mask)

    def _embed_text_token_ids(self, token_ids: Tensor) -> Tensor:
        flat_ids = token_ids.reshape(-1, token_ids.shape[-1])
        flat_tokens = self._backbone_ref.embed_text_token_ids(flat_ids)
        if flat_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected text action hidden dim {self.hidden_dim}, got {flat_tokens.shape[-1]}."
            )
        return flat_tokens.reshape(*token_ids.shape, self.hidden_dim)

    def _build_action_texts(self, config: Config) -> list[list[str]]:
        from ..data.adapters import make_default_adapter

        all_texts: list[list[str]] = []
        for env_id in config.env.env_ids:
            adapter = make_default_adapter(env_id)
            texts = [adapter.action_to_text(action_id) for action_id in range(config.env.num_actions)]
            texts.append(config.env.null_action_text)
            all_texts.append(texts)
        return all_texts


class EnvEmbedding(nn.Module):
    """Add a learned environment embedding to every token in a sequence."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_envs = len(config.env.env_ids)
        self.embedding = nn.Embedding(self.num_envs, config.hidden_dim)

    def forward(self, tokens: Tensor, env_ids: Tensor | None) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"EnvEmbedding expects (B, S, D), got {tuple(tokens.shape)}.")
        if env_ids is None:
            env_ids = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        if env_ids.ndim != 1 or env_ids.shape[0] != tokens.shape[0]:
            raise ValueError(
                "env_ids must have shape (B,) matching tokens batch size, got "
                f"env_ids={tuple(env_ids.shape)} tokens={tuple(tokens.shape)}."
            )
        if env_ids.dtype != torch.long:
            env_ids = env_ids.to(dtype=torch.long)
        return tokens + self.embedding(env_ids).unsqueeze(1)


class BeliefUpdate(nn.Module):
    """Gated residual belief update followed by RecurrenceNorm."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        # Match main branch: zero-init weight so gate starts as constant
        # sigmoid(bias) ≈ 0.12, independent of delta content.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, config.belief.gate_bias_init)

    def forward(self, prev_belief: BeliefState, delta: Tensor) -> BeliefState:
        if delta.shape != prev_belief.slots.shape:
            raise ValueError(
                "BeliefUpdate requires delta and belief slots to have identical shapes, "
                f"got delta={tuple(delta.shape)} belief={tuple(prev_belief.slots.shape)}."
            )

        gate = torch.sigmoid(self.gate(delta))
        next_slots = prev_belief.slots + gate * delta
        next_slots = self.norm(next_slots)
        return BeliefState(slots=next_slots)


class TransitionCore(nn.Module):
    """Shared transition core for posterior and prior belief updates."""

    def __init__(self, config: Config, backbone: TransitionBackbone) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.action_conditioning_mode = config.backbone.action_conditioning_mode
        self.backbone = backbone
        self.token_types = TokenTypeEmbedding(config.hidden_dim)
        self.env_embedding = EnvEmbedding(config)
        self.action_encoder = ActionEncoder(config)
        self.text_action_conditioner: TextActionConditioner | None = None
        if self.action_conditioning_mode == "text":
            self.text_action_conditioner = TextActionConditioner(config, backbone)
        elif self.action_conditioning_mode != "embedded":
            raise ValueError(
                f"Unsupported action_conditioning_mode: {self.action_conditioning_mode}"
            )
        self.belief_update = BeliefUpdate(config)

    def posterior_step(
        self,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        observation_tokens: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefState:
        """Compute Z_t from (Z_{t-1}, a_{t-1}, o_t)."""
        return self._transition_step(
            prev_belief=prev_belief,
            prev_actions=prev_actions,
            observation_tokens=observation_tokens,
            env_ids=env_ids,
        )

    def prior_step(
        self,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefState:
        """Predict the next belief from (Z_{t-1}, a_{t-1}) only."""
        return self._transition_step(
            prev_belief=prev_belief,
            prev_actions=prev_actions,
            observation_tokens=None,
            env_ids=env_ids,
        )

    def rollout_prior(
        self,
        start_belief: BeliefState,
        action_seq: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefTrajectory:
        """Unroll prior_step over an action sequence.

        This function performs a pure model rollout. Detach behavior at BPTT
        boundaries is a trainer concern and is intentionally not handled here.
        """
        if action_seq.ndim != 2:
            raise ValueError(
                f"rollout_prior expects action_seq with shape (B, T), got {tuple(action_seq.shape)}."
            )
        if action_seq.shape[0] != start_belief.batch_size:
            raise ValueError(
                "rollout_prior requires matching batch size between start_belief and action_seq, "
                f"got belief batch={start_belief.batch_size} action batch={action_seq.shape[0]}."
            )

        beliefs: list[Tensor] = []
        belief = start_belief
        horizon = action_seq.shape[1]

        for step in range(horizon):
            belief = self.prior_step(belief, action_seq[:, step], env_ids=env_ids)
            beliefs.append(belief.slots)

        stacked_beliefs = torch.stack(beliefs, dim=1)
        return BeliefTrajectory(beliefs=stacked_beliefs, actions=action_seq)

    def _transition_step(
        self,
        *,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        observation_tokens: Tensor | None,
        env_ids: Tensor | None,
    ) -> BeliefState:
        belief_tokens = self._with_token_type(
            prev_belief.slots,
            TokenTypeEmbedding.BELIEF,
        )
        action_tokens, action_mask = self._encode_actions(prev_actions, env_ids)

        input_segments: list[Tensor] = []
        mask_segments: list[Tensor] = []
        belief_start_index = 0

        if observation_tokens is not None:
            visual_tokens = self._with_token_type(
                observation_tokens,
                TokenTypeEmbedding.VISUAL,
            )
            input_segments.append(visual_tokens)
            mask_segments.append(self._full_attention_mask(visual_tokens))
            belief_start_index = visual_tokens.shape[1]

        input_segments.append(belief_tokens)
        mask_segments.append(self._full_attention_mask(belief_tokens))
        input_segments.append(action_tokens)
        mask_segments.append(action_mask)
        input_tokens = torch.cat(input_segments, dim=1)
        attention_mask = torch.cat(mask_segments, dim=1)
        input_tokens = self.env_embedding(input_tokens, env_ids)

        hidden = self.backbone(input_tokens, attention_mask=attention_mask)
        self._validate_hidden(hidden, input_tokens)

        belief_end_index = belief_start_index + prev_belief.num_slots
        belief_delta = hidden[:, belief_start_index:belief_end_index, :]
        return self.belief_update(prev_belief, belief_delta)

    def _encode_actions(
        self,
        prev_actions: Tensor,
        env_ids: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.action_conditioning_mode == "embedded":
            action_vectors = self.action_encoder(prev_actions)
            action_tokens = action_vectors.unsqueeze(1)
            action_tokens = self._with_token_type(action_tokens, TokenTypeEmbedding.ACTION)
            return action_tokens, self._full_attention_mask(action_tokens)

        assert self.text_action_conditioner is not None
        condition = self.text_action_conditioner(prev_actions, env_ids)
        action_tokens = self._with_token_type(condition.tokens, TokenTypeEmbedding.ACTION)
        return action_tokens, condition.attention_mask

    def _with_token_type(self, tokens: Tensor, token_type: int) -> Tensor:
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected token hidden dim {self.hidden_dim}, got {tokens.shape[-1]}."
            )
        return self.token_types(tokens, token_type)

    @staticmethod
    def _full_attention_mask(tokens: Tensor) -> Tensor:
        return torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.long, device=tokens.device)

    @staticmethod
    def _validate_hidden(hidden: Tensor, reference: Tensor) -> None:
        if hidden.shape != reference.shape:
            raise ValueError(
                "Transition backbone must preserve token sequence shape, "
                f"got hidden={tuple(hidden.shape)} reference={tuple(reference.shape)}."
            )
