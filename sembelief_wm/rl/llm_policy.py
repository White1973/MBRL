"""LLM-based actor-critic policy.

Uses an LLM backbone (e.g. Qwen2.5-VL + LoRA) to process belief slots
as soft prefix tokens, then delegates action output to an environment-
specific ActionAdapter.

This module implements the Policy protocol from rl/policy.py.
It does NOT know about specific environments — that knowledge lives
entirely in the ActionAdapter.

Belief slots are fed directly into the LLM backbone as soft prefix tokens.
No separate projection layer is needed because belief_dim is always aligned
with llm_hidden_dim (both = backbone hidden_size). The LoRA adapters in the
backbone learn to read the latent slots during SFT/PPO training.

Two output modes:
  - Action head: belief → LLM hidden → adapter.forward_logits → discrete dist
  - Free decoding: belief → LLM generate → adapter.decode_text → action index
    (not yet implemented — requires token-level PPO, deferred to later sprint)

Critic branch is selectable via LLMPolicyConfig.critic_source:
  - "qwen_pooled"   : critic eats Qwen+LoRA pooled hidden, then a 2-layer MLP
                      value_head. Pipeline assembly gives it an independent
                      Qwen+LoRA when the actor is also qwen_pooled.
  - "latent_belief"  : critic eats the raw belief slots via an independent
                      BeliefReadout (mean_pool) + a fresh MLP value_net,
                      WITHOUT running them through the VLM. Decouples the critic
                      from the shared backbone; mirrors MBRL-MLP's critic.
  - "frozen_vlm"     : critic runs the belief slots through the Qwen backbone
                      UNDER torch.no_grad() (so the LoRA is NOT updated by the
                      critic loss — it stays frozen w.r.t. the value objective),
                      then feeds the pooled hidden to a fresh MLP value_head.
                      The actor path is unchanged and follows actor_source.
                      Cost: one extra (no-grad) backbone forward vs qwen_pooled;
                      gain: critic uses the VLM representation without adding
                      critic gradients to its LoRA.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from .action_adapter import ActionAdapter
from ..model.belief import BeliefReadout


def project_batch_global_action_gradient(logits: Tensor) -> Tensor:
    """Keep Actor logits unchanged while removing batch-global action gradient.

    The detached correction makes the forward value exactly ``logits``. On
    backward, each action's gradient is centered across states, so PPO can
    still learn state-dependent action preferences but cannot turn a marginal
    batch imbalance into a policy-wide direction bias.
    """
    action_mean = logits.mean(dim=0, keepdim=True)
    return logits - action_mean + action_mean.detach()


# ---------------------------------------------------------------------------
# Local MLP helpers (kept here so rl/ does not reverse-import agent/).
# Mirrors sembelief_wm/agent/phase2.py:_build_mlp / _activation.
# ---------------------------------------------------------------------------

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


class TwoStageSlotActionDecoder(nn.Module):
    """Historical ordered-slot decoder used by the proven Sokoban actor."""

    def __init__(
        self,
        *,
        num_slots: int,
        hidden_dim: int,
        slot_dim: int,
        decoder_hidden_dim: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, slot_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(num_slots * slot_dim),
            nn.Linear(num_slots * slot_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden_dim, num_actions),
        )

    def forward(self, slots: Tensor) -> Tensor:
        if slots.ndim != 3 or slots.shape[1] != self.num_slots:
            raise ValueError(
                "TwoStageSlotActionDecoder expects ordered (B,K,D), got "
                f"{tuple(slots.shape)}"
            )
        dtype = next(self.parameters()).dtype
        encoded = self.slot_encoder(slots.to(dtype=dtype))
        return self.decoder(encoded.flatten(start_dim=1))


@dataclass
class LLMPolicyConfig:
    """Configuration for LLM actor-critic."""

    hidden_dim: int = 3584            # belief slot dim == LLM hidden dim (always aligned)
    num_slots: int = 36               # belief slot count
    output_mode: str = "action_head"  # "action_head" or "free_decoding"

    # ── Actor branch selection ───────────────────────────────────────────
    actor_source: Literal[
        "qwen_pooled", "qwen_slotwise", "latent_belief", "frozen_vlm"
    ] = "qwen_pooled"
    """Which input feeds the action head.

    "qwen_pooled"  : belief slots → Qwen+LoRA (grad enabled) → mean-pool →
                     action_head. PPO policy loss trains the actor-owned LoRA.
    "qwen_slotwise": belief slots → Qwen+LoRA → shared per-slot projection →
                     ordered flatten → action MLP. Preserves the 6x6 spatial
                     token identity while training the actor-owned LoRA.
    "latent_belief": belief slots → action_readout (mean_pool, no VLM forward)
                     → action_mlp (MLP D→1024→1024→num_actions) → Categorical.
                     Decouples the actor from the backbone — actor never touches
                     Qwen/LoRA, so policy loss cannot update any LoRA
                     through the actor (mirrors latent_belief critic).
    "frozen_vlm"  : belief slots → Qwen+LoRA UNDER torch.no_grad() → mean-pool
                     → action_head. The LoRA is frozen w.r.t. the policy loss
                     (only action_head trains on the actor side). Useful as an
                     ablation: "is the WM-trained LoRA representation already
                     good enough for action selection, with only a linear head
                     on top?" ``frozen_vlm`` freezes it only for this branch.
    """

    # qwen_pooled / frozen_vlm use the env ActionAdapter as the action head.
    # latent_belief actor branch parameters (independent MLP on belief slots).
    actor_hidden_dim: int = 1024
    actor_hidden_layers: int = 2
    actor_slot_dim: int = 64
    slotwise_actor_features: Literal["qwen", "raw"] = "qwen"
    """Ordered slot features used by qwen_slotwise: Qwen tokens or raw beliefs."""
    slotwise_behavior_scale: float = 1.0
    """Scale of the legacy raw-slot behavior logits in qwen_slotwise mode.

    Set to zero for a direct Qwen actor.  The legacy default remains one so
    old checkpoints retain their exact forward semantics.
    """
    actor_activation: Literal["gelu", "relu", "tanh"] = "gelu"
    actor_readout: Literal["mean_pool", "attention_pool", "learned_query"] = "mean_pool"

    # ── Critic branch selection ──────────────────────────────────────────
    critic_source: Literal[
        "qwen_pooled", "qwen_slotwise_q", "latent_belief",
        "latent_ordered_v", "frozen_vlm"
    ] = "qwen_pooled"
    """Which input feeds the value head.

    "qwen_pooled" : Qwen+LoRA pooled hidden fed to value_head. With an injected
                    critic_backbone, it trains a critic-owned LoRA; otherwise
                    it reuses the actor forward for legacy low-level callers.
    "latent_belief": the raw belief slots are pooled by an independent
                    BeliefReadout (default mean_pool, no VLM forward) and then
                    passed to a fresh value_net MLP. Decouples the critic from
                    the backbone — it never touches Qwen/LoRA at all.
    "latent_ordered_v": all raw belief slots retain their order and feed an
                    action-independent scalar V(s) head. It is intended for
                    explicit real-return supervision, never imagined Q labels.
    "frozen_vlm" : the belief slots are run through the Qwen backbone under
                    torch.no_grad(), so the critic loss does NOT update the
                    selected backbone LoRA (frozen for this objective only).
                    The pooled hidden feeds a fresh value_head MLP — the ONLY
                    trainable part of this branch. The actor path is unaffected
                    and follows actor_source.
    """

    # qwen_pooled / frozen_vlm shared value_head parameters
    value_hidden_dim: int = 256       # value head hidden layer
    critic_slot_dim: int = 32         # ordered Q critic per-token projection

    # latent_belief branch parameters (independent MLP critic on belief slots)
    critic_hidden_dim: int = 1024
    critic_hidden_layers: int = 2
    critic_activation: Literal["gelu", "relu", "tanh"] = "gelu"
    critic_readout: Literal["mean_pool", "attention_pool", "learned_query"] = "mean_pool"


class LLMActorCritic(nn.Module):
    """LLM-based actor-critic that reads belief slots.

    Qwen actor path:
        belief_slots (B, K, D)
            → LLM backbone forward (with LoRA) → hidden_states (B, K, D)
            → qwen_pooled: mean-pool + action adapter, or
              qwen_slotwise: ordered per-slot projection + action MLP
            → (B, num_actions)

    Critic path (selectable via cfg.critic_source):
        "qwen_pooled"  : reuse the actor's Qwen pooled hidden (B, D)
                         → value_head (Linear→GELU→Linear) → (B,)
        "frozen_vlm"   : re-run Qwen backbone under no_grad (B, D)
                         → value_head (same MLP, independently trained)
                         → (B,). LoRA frozen w.r.t. critic; actor unaffected.
        "latent_belief" : belief_slots → value_readout (mean_pool, no VLM)
                         → value_net (MLP D→1024→1024→1) → (B,)

    Belief slots are fed directly as soft prefix tokens — no projection
    needed since belief_dim == llm_hidden_dim by design. LoRA learns to
    read the latent slots during SFT/PPO.

    The LLM backbone is injected. Normal pipeline assembly provides a
    policy-owned backbone; read-only compatibility configurations may inject
    a non-owning wrapper around the WM backbone.

    Implements rl.policy.Policy protocol (compatible via adapter).
    """

    def __init__(
        self,
        backbone: nn.Module,
        action_adapter: ActionAdapter,
        config: LLMPolicyConfig | None = None,
        critic_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = config or LLMPolicyConfig()

        self.backbone = backbone
        # Optional independent LoRA view for the qwen_pooled critic branch.
        # Actor and critic wrappers select different adapters in one shared
        # frozen Qwen base; their LoRA Parameter objects remain disjoint.
        self.critic_backbone = critic_backbone
        self.action_adapter = action_adapter
        self.config = cfg
        self.critic_source = cfg.critic_source
        self.actor_source = cfg.actor_source
        # Multiple LoRA adapters share one Qwen base and PEFT selects adapters
        # through model-level state. With gradient checkpointing, the actor
        # backward must finish before a critic forward switches that state.
        self.requires_separate_actor_critic_backward = (
            critic_backbone is not None
            and cfg.actor_source in {"qwen_pooled", "qwen_slotwise"}
            and not (
                cfg.actor_source == "qwen_slotwise"
                and cfg.slotwise_actor_features == "raw"
            )
            and cfg.critic_source in {
                "qwen_pooled", "qwen_slotwise_q", "frozen_vlm"
            }
        )

        # Optional frozen snapshot fitted by offline latent behavior cloning.
        # It contains actor heads/readouts only; the Qwen/LoRA representation
        # remains frozen in the supported behavior-cloning modes.
        self.behavior_action_adapter: nn.Module | None = None
        self.behavior_action_readout: nn.Module | None = None
        self.behavior_action_mlp: nn.Module | None = None
        self.behavior_slotwise_head: nn.Module | None = None
        # qwen_pooled needs a stationary copy of both pieces of the actor:
        # the action head *and* the actor-owned LoRA representation.  Keeping
        # only a head snapshot is not a behavior policy once PPO moves LoRA.
        self.behavior_actor_adapter_name: str | None = None
        num_actor_actions = getattr(action_adapter, "num_actions", None)
        if num_actor_actions is None:
            raise ValueError("action_adapter.num_actions is required")
        # Non-persistent for backward checkpoint compatibility. The pipeline
        # serializes it explicitly so old P0 checkpoints load with zero
        # correction and new Stage-2 checkpoints resume exactly.
        self.register_buffer(
            "_actor_logit_correction",
            torch.zeros(int(num_actor_actions), dtype=torch.float32),
            persistent=False,
        )

        # ── Actor branch construction ──
        # qwen_pooled / frozen_vlm: use the env ActionAdapter as the action
        # head, fed by the Qwen+LoRA pooled hidden (grad or no_grad).
        # latent_belief: independent BeliefReadout + MLP on belief slots, no VLM.
        self.action_readout: BeliefReadout | None = None
        self.action_mlp: nn.Module | None = None
        self.slotwise_head: nn.Module | None = None
        self.slotwise_behavior_prior: TwoStageSlotActionDecoder | None = None
        if cfg.actor_source == "latent_belief":
            self.action_readout = BeliefReadout(
                mode=cfg.actor_readout,
                num_slots=cfg.num_slots,
                hidden_dim=cfg.hidden_dim,
            )
            # action_mlp outputs num_actions logits directly (bypasses
            # action_adapter.forward_logits, since there's no VLM hidden).
            num_actions = getattr(action_adapter, "num_actions", None)
            if num_actions is None:
                raise ValueError(
                    "latent_belief actor needs action_adapter.num_actions to size "
                    "the action MLP; the supplied adapter exposes none."
                )
            self.action_mlp = _build_mlp(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.actor_hidden_dim,
                hidden_layers=cfg.actor_hidden_layers,
                activation=cfg.actor_activation,
                output_dim=num_actions,
            )
        elif cfg.actor_source == "qwen_slotwise":
            num_actions = getattr(action_adapter, "num_actions", None)
            if num_actions is None:
                raise ValueError(
                    "qwen_slotwise actor needs action_adapter.num_actions"
                )
            self.slotwise_behavior_prior = TwoStageSlotActionDecoder(
                num_slots=cfg.num_slots,
                hidden_dim=cfg.hidden_dim,
                slot_dim=cfg.actor_slot_dim,
                decoder_hidden_dim=cfg.actor_hidden_dim,
                num_actions=num_actions,
            )
            self.slotwise_behavior_prior.requires_grad_(False)
            self.slotwise_head = TwoStageSlotActionDecoder(
                num_slots=cfg.num_slots,
                hidden_dim=cfg.hidden_dim,
                slot_dim=cfg.actor_slot_dim,
                decoder_hidden_dim=cfg.actor_hidden_dim,
                num_actions=num_actions,
            )
            # The Qwen branch is a residual policy.  Zeroing only its final
            # projection preserves useful upstream gradients after the first
            # optimizer step while making initial logits exactly equal to the
            # historical behavior prior.
            final = self.slotwise_head.decoder[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        # qwen_pooled / frozen_vlm keep using action_adapter.forward_logits.

        # ── Critic branch construction ──
        self.q_head: TwoStageSlotActionDecoder | None = None
        self.ordered_value_head: TwoStageSlotActionDecoder | None = None
        if cfg.critic_source in ("qwen_pooled", "frozen_vlm"):
            # value_head on the Qwen pooled hidden (B, D).
            # qwen_pooled : uses critic_backbone when injected; otherwise it
            #               reuses the actor backbone for legacy callers.
            # frozen_vlm  : re-runs the backbone under no_grad → LoRA is NOT in
            #               the critic's autograd graph; only value_head trains.
            self.value_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.value_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.value_hidden_dim, 1),
            )
            self.value_readout: BeliefReadout | None = None
            self.value_net: nn.Module | None = None
        elif cfg.critic_source == "qwen_slotwise_q":
            num_actions = getattr(action_adapter, "num_actions", None)
            if num_actions is None:
                raise ValueError(
                    "qwen_slotwise_q critic needs action_adapter.num_actions"
                )
            self.q_head = TwoStageSlotActionDecoder(
                num_slots=cfg.num_slots,
                hidden_dim=cfg.hidden_dim,
                slot_dim=cfg.critic_slot_dim,
                decoder_hidden_dim=cfg.value_hidden_dim,
                num_actions=num_actions,
            )
            self.value_head = None
            self.value_readout = None
            self.value_net = None
        elif cfg.critic_source == "latent_ordered_v":
            self.ordered_value_head = TwoStageSlotActionDecoder(
                num_slots=cfg.num_slots,
                hidden_dim=cfg.hidden_dim,
                slot_dim=cfg.critic_slot_dim,
                decoder_hidden_dim=cfg.value_hidden_dim,
                num_actions=1,
            )
            self.value_head = None
            self.value_readout = None
            self.value_net = None
        elif cfg.critic_source == "latent_belief":
            # Independent MLP critic on belief slots — no VLM forward, no
            # coupling to a Qwen LoRA. Mirrors MBRL-MLP's critic.
            self.value_readout = BeliefReadout(
                mode=cfg.critic_readout,
                num_slots=cfg.num_slots,
                hidden_dim=cfg.hidden_dim,
            )
            self.value_net = _build_mlp(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.critic_hidden_dim,
                hidden_layers=cfg.critic_hidden_layers,
                activation=cfg.critic_activation,
                output_dim=1,
            )
            self.value_head = None
        else:
            raise ValueError(f"Unsupported critic_source: {cfg.critic_source!r}")

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def set_deterministic_forward_mode(self) -> None:
        """Disable stochastic layers on every policy forward dependency.

        Actor and critic wrappers keep the shared Qwen backbone as a
        non-owning reference, so the normal recursive ``self.eval()`` does not
        reach Qwen/PEFT/LoRA dropout modules. PPO still computes gradients in
        eval mode: this method does *not* change ``requires_grad`` and does not
        freeze actor or critic LoRA parameters.
        """
        self.eval()
        seen: set[int] = set()
        for wrapper in (self.backbone, self.critic_backbone):
            if wrapper is None:
                continue
            underlying = getattr(wrapper, "_backbone", wrapper)
            if isinstance(underlying, nn.Module) and id(underlying) not in seen:
                underlying.eval()
                seen.add(id(underlying))

    def forward_mode_diagnostics(self) -> dict[str, float]:
        """Report dropout state including non-owning shared Qwen modules."""
        modules: list[nn.Module] = list(self.modules())
        for wrapper in (self.backbone, self.critic_backbone):
            if wrapper is None:
                continue
            underlying = getattr(wrapper, "_backbone", wrapper)
            if isinstance(underlying, nn.Module):
                modules.extend(underlying.modules())
        unique: list[nn.Module] = []
        seen: set[int] = set()
        for module in modules:
            if id(module) not in seen:
                unique.append(module)
                seen.add(id(module))
        dropouts = [
            module
            for module in unique
            if isinstance(module, nn.modules.dropout._DropoutNd)
        ]
        active = [
            module
            for module in dropouts
            if module.training and float(module.p) > 0.0
        ]
        return {
            "dropout_modules": float(len(dropouts)),
            "active_dropout_modules": float(len(active)),
            "max_active_dropout_p": max(
                (float(module.p) for module in active),
                default=0.0,
            ),
        }

    def _slots_from_states(self, states: Tensor) -> Tensor:
        """Reshape flat (B, K*D) into slots (B, K, D); pass through if already 3D."""
        cfg = self.config
        if states.ndim == 2:
            return states.view(-1, cfg.num_slots, cfg.hidden_dim)
        return states  # already (B, K, D)

    def _actor_hidden(self, states: Tensor) -> Tensor:
        """Actor path: slots → Qwen+LoRA → mean-pool → (B, D).

        Also reused by the "qwen_pooled" critic branch, so callers should pass
        the returned hidden into _critic_value(..., hidden=hidden) to avoid a
        duplicate backbone forward.
        """
        slots = self._slots_from_states(states)

        # Forward through backbone — belief slots as soft prefix tokens
        hidden = self.backbone(slots)  # expected: (B, K, D) or (B, D)

        # Pool to (B, D) if backbone returns per-token hidden states
        if hidden.ndim == 3:
            hidden = hidden.mean(dim=1)

        # Cast to the heads' dtype. The Qwen backbone runs in bf16, but the
        # action/value heads are float32 parameters; matmuls require matching
        # dtypes. Aligning here keeps act()/evaluate_actions() dtype-agnostic.
        head_dtype = next(self.action_adapter.parameters()).dtype
        if hidden.dtype != head_dtype:
            hidden = hidden.to(head_dtype)
        return hidden

    def _actor_token_hidden(self, states: Tensor) -> Tensor:
        """Actor Qwen output without spatial pooling: (B, K, D)."""
        slots = self._slots_from_states(states)
        hidden = self.backbone(slots)
        if hidden.ndim != 3:
            raise RuntimeError(
                "qwen_slotwise requires per-token Qwen hidden states, got "
                f"shape={tuple(hidden.shape)}"
            )
        assert self.slotwise_head is not None
        head_dtype = next(self.slotwise_head.parameters()).dtype
        return hidden.to(dtype=head_dtype)

    def _frozen_critic_hidden(self, states: Tensor) -> Tensor:
        """frozen_vlm branch: run the backbone under no_grad to get pooled hidden.

        This re-runs the Qwen+LoRA forward with gradients DISABLED, so the
        critic loss cannot backprop into the selected LoRA — only value_head
        receives gradients. The actor path continues to use its own (grad-
        enabled) backbone forward to keep training the LoRA via policy loss.

        Extra cost vs qwen_pooled: one more backbone forward (no autograd).
        """
        slots = self._slots_from_states(states)
        backbone = (
            self.critic_backbone
            if self.critic_backbone is not None
            else self.backbone
        )
        with torch.no_grad():
            hidden = backbone(slots)
            if hidden.ndim == 3:
                hidden = hidden.mean(dim=1)
        head_dtype = next(self.value_head.parameters()).dtype
        if hidden.dtype != head_dtype:
            hidden = hidden.to(head_dtype)
        return hidden

    def _critic_hidden_independent(self, states: Tensor) -> Tensor:
        """Decoupled qwen_pooled critic: forward through `self.critic_backbone`.

        The critic owns its own LoRA (separate parameter objects from the
        actor's) while sharing the frozen Qwen base. Gradients update only the
        critic LoRA + value_head; PPO completes actor backward before switching
        adapters so gradient-checkpoint recomputation stays correct.
        """
        slots = self._slots_from_states(states)
        hidden = self.critic_backbone(slots)
        if hidden.ndim == 3:
            hidden = hidden.mean(dim=1)
        head_dtype = next(self.value_head.parameters()).dtype
        if hidden.dtype != head_dtype:
            hidden = hidden.to(head_dtype)
        return hidden

    def _critic_token_hidden_independent(self, states: Tensor) -> Tensor:
        """Critic-owned Qwen output preserving all ordered slot tokens."""
        slots = self._slots_from_states(states)
        hidden = self.critic_backbone(slots)
        if hidden.ndim != 3:
            raise RuntimeError(
                "qwen_slotwise_q requires per-token critic hidden states, got "
                f"shape={tuple(hidden.shape)}"
            )
        assert self.q_head is not None
        return hidden.to(dtype=next(self.q_head.parameters()).dtype)

    def _critic_q_values(self, states: Tensor) -> Tensor:
        """Return ordered-slot action values Q(s, ·), without mean pooling."""
        if self.critic_source != "qwen_slotwise_q":
            raise RuntimeError("Q values are only defined for qwen_slotwise_q")
        return self.q_head(self._critic_token_hidden_independent(states))

    def _critic_value(
        self, states: Tensor, hidden: Tensor | None = None,
        actions: Tensor | None = None,
    ) -> Tensor:
        """Compute V(s) according to the configured critic branch.

        Args:
            states: belief slots (B, K, D) or flat (B, K*D).
            hidden: optional precomputed actor Qwen pooled hidden (B, D).
                    Used only by the "qwen_pooled" branch — passing it avoids
                    a redundant backbone forward. Ignored by "latent_belief"
                    and "frozen_vlm" (the latter needs its own no_grad forward).
        """
        if self.critic_source == "qwen_slotwise_q":
            if actions is None:
                raise ValueError("qwen_slotwise_q requires the evaluated actions")
            q_values = self._critic_q_values(states)
            return q_values.gather(1, actions.long().view(-1, 1)).squeeze(1)

        if self.critic_source == "qwen_pooled":
            if self.critic_backbone is not None:
                # Decoupled critic: run the critic adapter on the shared Qwen
                # (grad-enabled) so the value loss updates the critic's own
                # LoRA, never the actor's. The pooled hidden feeds value_head.
                h = self._critic_hidden_independent(states)
            else:
                h = hidden if hidden is not None else self._actor_hidden(states)
            return self.value_head(h).squeeze(-1)

        if self.critic_source == "frozen_vlm":
            # Re-run the backbone under no_grad → LoRA frozen w.r.t. critic.
            pooled = self._frozen_critic_hidden(states)
            return self.value_head(pooled).squeeze(-1)

        if self.critic_source == "latent_ordered_v":
            assert self.ordered_value_head is not None
            return self.ordered_value_head(
                self._slots_from_states(states)
            ).squeeze(-1)

        # latent_belief: pool the raw slots WITHOUT touching the VLM.
        slots = self._slots_from_states(states)
        pooled = self.value_readout(slots)  # (B, D); mean_pool is parameter-free

        # slots are typically bf16 (from the WM posterior); value_net is fp32.
        net_dtype = next(self.value_net.parameters()).dtype
        if pooled.dtype != net_dtype:
            pooled = pooled.to(dtype=net_dtype)
        return self.value_net(pooled).squeeze(-1)

    def _actor_logits(self, states: Tensor, hidden: Tensor | None = None) -> Tensor:
        """Compute action logits (B, num_actions) per the actor branch.

        Args:
            states: belief slots (B, K, D) or flat (B, K*D).
            hidden: optional precomputed Qwen pooled hidden (B, D). Used only by
                    qwen_pooled / frozen_vlm to avoid a redundant backbone
                    forward. Ignored by latent_belief.
        """
        cfg = self.config
        if cfg.actor_source == "latent_belief":
            slots = self._slots_from_states(states)
            pooled = self.action_readout(slots)  # (B, D); mean_pool is parameter-free
            mlp_dtype = next(self.action_mlp.parameters()).dtype
            if pooled.dtype != mlp_dtype:
                pooled = pooled.to(dtype=mlp_dtype)
            return self._correct_actor_logits(self.action_mlp(pooled))

        if cfg.actor_source == "qwen_slotwise":
            assert self.slotwise_head is not None
            assert self.slotwise_behavior_prior is not None
            if cfg.slotwise_actor_features == "raw":
                token_hidden = self._slots_from_states(states)
            else:
                token_hidden = (
                    hidden if hidden is not None else self._actor_token_hidden(states)
                )
            actor_logits = self.slotwise_head(token_hidden)
            if cfg.slotwise_behavior_scale == 0.0:
                return self._correct_actor_logits(actor_logits)
            base_logits = self.slotwise_behavior_prior(
                self._slots_from_states(states)
            )
            return self._correct_actor_logits(
                cfg.slotwise_behavior_scale * base_logits + actor_logits
            )

        # qwen_pooled: reuse the (grad-enabled) actor hidden if provided.
        # frozen_vlm : re-run the backbone under no_grad so policy loss cannot
        #               update the actor LoRA through this path.
        if cfg.actor_source == "frozen_vlm":
            h = self._frozen_actor_hidden(states)
        else:  # qwen_pooled
            h = hidden if hidden is not None else self._actor_hidden(states)
        return self._correct_actor_logits(
            self.action_adapter.forward_logits(h)
        )

    def _correct_actor_logits(self, logits: Tensor) -> Tensor:
        correction = self._actor_logit_correction.to(
            device=logits.device, dtype=logits.dtype
        )
        return logits - correction

    @torch.no_grad()
    def adjust_actor_logit_correction(self, action_shift: Tensor) -> None:
        """Remove an action-wise global shift measured on fixed probes."""
        shift = action_shift.detach().to(
            device=self._actor_logit_correction.device,
            dtype=self._actor_logit_correction.dtype,
        )
        if shift.shape != self._actor_logit_correction.shape:
            raise ValueError("Actor logit correction shape mismatch")
        # A scalar shared by all actions is softmax-invariant and needlessly
        # consumes numerical range, so retain only relative action bias.
        self._actor_logit_correction.add_(shift - shift.mean())

    @torch.no_grad()
    def actor_logit_correction(self) -> Tensor:
        return self._actor_logit_correction.detach().clone()

    @torch.no_grad()
    def restore_actor_logit_correction(self, value: Tensor) -> None:
        self._actor_logit_correction.copy_(value.to(
            device=self._actor_logit_correction.device,
            dtype=self._actor_logit_correction.dtype,
        ))

    def actor_logits(self, states: Tensor) -> Tensor:
        """Actor logits for offline BC with a fixed Qwen representation.

        Offline BC intentionally initializes only the action head.  Detaching
        the qwen_pooled hidden prevents unused LoRA gradients from accumulating
        over hundreds of BC steps; PPO later updates the actor LoRA normally
        through ``evaluate_actor_actions``.
        """
        if self.actor_source == "qwen_pooled":
            with torch.no_grad():
                hidden = self._actor_hidden(states)
            return self._actor_logits(states, hidden=hidden.detach())
        if self.actor_source == "qwen_slotwise":
            if self.config.slotwise_actor_features == "raw":
                return self._actor_logits(states)
            with torch.no_grad():
                hidden = self._actor_token_hidden(states)
            return self._actor_logits(states, hidden=hidden.detach())
        return self._actor_logits(states)

    def actor_logits_trainable(self, states: Tensor) -> Tensor:
        """Expert-rehearsal logits with gradients through the current actor.

        For ``qwen_pooled`` this intentionally differs from ``actor_logits``:
        rehearsal must protect both the actor-owned Qwen LoRA and the action
        head, while one-time offline BC remains a cheap head-only warm start.
        """
        hidden = None
        if self.actor_source == "qwen_pooled":
            hidden = self._actor_hidden(states)
        elif self.actor_source == "qwen_slotwise":
            if self.config.slotwise_actor_features != "raw":
                hidden = self._actor_token_hidden(states)
        return self._actor_logits(states, hidden=hidden)

    def actor_base_logits(self, states: Tensor) -> Tensor:
        """Frozen historical behavior logits for residual-policy protection."""
        if self.actor_source != "qwen_slotwise":
            raise RuntimeError("actor_base_logits requires qwen_slotwise")
        assert self.slotwise_behavior_prior is not None
        return self.slotwise_behavior_prior(self._slots_from_states(states))

    def load_slotwise_behavior_prior(self, policy_state: dict[str, Tensor]) -> None:
        """Load and freeze the proven legacy ``slot_action_decoder`` weights."""
        if self.actor_source != "qwen_slotwise":
            raise RuntimeError("slotwise behavior prior requires qwen_slotwise")
        assert self.slotwise_behavior_prior is not None
        prefix = "slot_action_decoder."
        mapped = {
            key[len(prefix):]: value
            for key, value in policy_state.items()
            if key.startswith(prefix)
        }
        expected = set(self.slotwise_behavior_prior.state_dict())
        if set(mapped) != expected:
            raise RuntimeError(
                "historical slotwise policy is incompatible: "
                f"missing={sorted(expected-set(mapped))}, "
                f"unexpected={sorted(set(mapped)-expected)}"
            )
        self.slotwise_behavior_prior.load_state_dict(mapped, strict=True)
        self.slotwise_behavior_prior.eval()
        self.slotwise_behavior_prior.requires_grad_(False)

    def actor_parameters(self) -> list[nn.Parameter]:
        """Return only trainable actor-head parameters, never Qwen/LoRA params."""
        if self.actor_source == "latent_belief":
            assert self.action_readout is not None
            assert self.action_mlp is not None
            return list(self.action_readout.parameters()) + list(
                self.action_mlp.parameters()
            )
        if self.actor_source == "qwen_slotwise":
            assert self.slotwise_head is not None
            return list(self.slotwise_head.parameters())
        return list(self.action_adapter.parameters())

    def critic_parameters(self) -> list[nn.Parameter]:
        """Return critic-owned head/readout parameters, never actor/WM LoRA."""
        if self.critic_source == "qwen_slotwise_q":
            assert self.q_head is not None
            return list(self.q_head.parameters())
        if self.critic_source == "latent_belief":
            assert self.value_readout is not None
            assert self.value_net is not None
            return list(self.value_readout.parameters()) + list(
                self.value_net.parameters()
            )
        if self.critic_source == "latent_ordered_v":
            assert self.ordered_value_head is not None
            return list(self.ordered_value_head.parameters())
        assert self.value_head is not None
        return list(self.value_head.parameters())

    def capture_behavior_reference(self) -> None:
        """Freeze a complete stationary actor snapshot for behavior KL.

        For ``qwen_pooled`` this snapshots both the action head and the
        actor-owned LoRA into a frozen PEFT adapter.  This is deliberately a
        policy reference, not a WM reference: PPO can keep training the actor
        LoRA while KL is measured against the immutable BC policy.
        """
        if (
            self.actor_source == "qwen_slotwise"
            and self.config.slotwise_actor_features == "raw"
        ):
            self.behavior_slotwise_head = copy.deepcopy(self.slotwise_head)
            modules = [self.behavior_slotwise_head]
        elif self.actor_source in {"qwen_pooled", "qwen_slotwise"}:
            underlying = getattr(self.backbone, "_backbone", None)
            source_name = getattr(self.backbone, "adapter_name", None)
            if underlying is None or not source_name:
                raise TypeError(
                    "qwen_pooled behavior reference requires a named, "
                    "multi-adapter Qwen policy backbone"
                )
            behavior_name = f"{source_name}_behavior"
            available = tuple(getattr(underlying, "lora_adapter_names", ()))
            if behavior_name not in available:
                add_adapter = getattr(underlying, "add_lora_adapter", None)
                if not callable(add_adapter):
                    raise TypeError(
                        "Qwen backbone cannot create a frozen behavior adapter"
                    )
                add_adapter(
                    behavior_name,
                    initialize_from=source_name,
                    lora_dropout=0.0,
                )
            else:
                copy_adapter = getattr(underlying, "copy_lora_adapter", None)
                if not callable(copy_adapter):
                    raise TypeError(
                        "Qwen backbone cannot refresh the behavior adapter"
                    )
                copy_adapter(source_name, behavior_name)
            freeze_adapter = getattr(
                underlying, "set_lora_adapter_trainable", None
            )
            if callable(freeze_adapter):
                freeze_adapter(behavior_name, False)
            self.behavior_actor_adapter_name = behavior_name
            if self.actor_source == "qwen_slotwise":
                self.behavior_slotwise_head = copy.deepcopy(self.slotwise_head)
                modules = [self.behavior_slotwise_head]
            else:
                self.behavior_action_adapter = copy.deepcopy(self.action_adapter)
                modules = [self.behavior_action_adapter]
        elif self.actor_source == "latent_belief":
            self.behavior_action_readout = copy.deepcopy(self.action_readout)
            self.behavior_action_mlp = copy.deepcopy(self.action_mlp)
            modules = [
                self.behavior_action_readout,
                self.behavior_action_mlp,
            ]
        else:
            self.behavior_action_adapter = copy.deepcopy(self.action_adapter)
            modules = [self.behavior_action_adapter]
        for module in modules:
            assert module is not None
            module.eval()
            module.requires_grad_(False)

    def _actor_and_behavior_logits(
        self, states: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute current and frozen behavior logits on identical features."""
        if self.actor_source == "latent_belief":
            if (
                self.behavior_action_readout is None
                or self.behavior_action_mlp is None
            ):
                raise RuntimeError("behavior reference has not been captured")
            slots = self._slots_from_states(states)
            pooled = self.action_readout(slots)
            current_dtype = next(self.action_mlp.parameters()).dtype
            current_logits = self.action_mlp(pooled.to(dtype=current_dtype))
            with torch.no_grad():
                behavior_pooled = self.behavior_action_readout(slots)
                behavior_dtype = next(
                    self.behavior_action_mlp.parameters()
                ).dtype
                behavior_logits = self.behavior_action_mlp(
                    behavior_pooled.to(dtype=behavior_dtype)
                )
            return self._correct_actor_logits(current_logits), behavior_logits

        if (
            self.actor_source == "qwen_slotwise"
            and self.config.slotwise_actor_features == "raw"
        ):
            if self.behavior_slotwise_head is None:
                raise RuntimeError("behavior reference has not been captured")
            slots = self._slots_from_states(states)
            dtype = next(self.slotwise_head.parameters()).dtype
            current_logits = self.slotwise_head(slots.to(dtype=dtype))
            with torch.no_grad():
                behavior_logits = self.behavior_slotwise_head(slots.to(dtype=dtype))
            return self._correct_actor_logits(current_logits), behavior_logits

        if self.actor_source in {"qwen_pooled", "qwen_slotwise"}:
            if (
                (
                    self.behavior_slotwise_head is None
                    if self.actor_source == "qwen_slotwise"
                    else self.behavior_action_adapter is None
                )
                or self.behavior_actor_adapter_name is None
            ):
                raise RuntimeError("Qwen behavior reference has not been captured")
            underlying = getattr(self.backbone, "_backbone", None)
            forward_with_adapter = getattr(
                underlying, "forward_with_adapter", None
            )
            if not callable(forward_with_adapter):
                raise TypeError(
                    "Qwen behavior reference requires adapter-selectable forward"
                )
            slots = self._slots_from_states(states)
            # Evaluate the frozen behavior adapter first.  The current actor is
            # evaluated last so gradient-checkpoint recomputation sees the
            # actor adapter, not the behavior adapter, during backward.
            with torch.no_grad():
                behavior_hidden = forward_with_adapter(
                    slots,
                    adapter_name=self.behavior_actor_adapter_name,
                )
                if self.actor_source == "qwen_slotwise":
                    if behavior_hidden.ndim != 3:
                        raise RuntimeError("behavior Qwen lost slot dimension")
                    behavior_dtype = next(
                        self.behavior_slotwise_head.parameters()
                    ).dtype
                    behavior_logits = self.behavior_slotwise_head(
                        behavior_hidden.to(dtype=behavior_dtype)
                    )
                else:
                    if behavior_hidden.ndim == 3:
                        behavior_hidden = behavior_hidden.mean(dim=1)
                    behavior_dtype = next(
                        self.behavior_action_adapter.parameters()
                    ).dtype
                    behavior_logits = self.behavior_action_adapter.forward_logits(
                        behavior_hidden.to(dtype=behavior_dtype)
                    )
            if self.actor_source == "qwen_slotwise":
                current_hidden = self._actor_token_hidden(states)
                scale = self.config.slotwise_behavior_scale
                current_logits = self.slotwise_head(current_hidden)
                if scale != 0.0:
                    base_logits = self.actor_base_logits(states)
                    behavior_logits = scale * base_logits + behavior_logits
                    current_logits = scale * base_logits + current_logits
            else:
                current_hidden = self._actor_hidden(states)
                current_logits = self.action_adapter.forward_logits(current_hidden)
            return self._correct_actor_logits(current_logits), behavior_logits

        if self.actor_source != "frozen_vlm":
            raise RuntimeError(
                "behavior logits require qwen_pooled, frozen_vlm, or "
                "latent_belief actor"
            )
        if self.behavior_action_adapter is None:
            raise RuntimeError("behavior reference has not been captured")
        # One frozen Qwen forward supplies identical fixed features to both
        # heads, avoiding a second expensive VLM call per PPO minibatch.
        hidden = self._frozen_actor_hidden(states)
        current_logits = self.action_adapter.forward_logits(hidden)
        with torch.no_grad():
            behavior_logits = self.behavior_action_adapter.forward_logits(hidden)
        return self._correct_actor_logits(current_logits), behavior_logits

    def _frozen_actor_hidden(self, states: Tensor) -> Tensor:
        """frozen_vlm actor branch: backbone forward under no_grad → pooled.

        The LoRA stays frozen w.r.t. the policy loss (only action_head trains
        on the actor side). The critic path is unaffected — if critic_source
        is qwen_pooled/frozen_vlm it still drives (or not) the LoRA via its own
        forward/grad.
        """
        slots = self._slots_from_states(states)
        with torch.no_grad():
            hidden = self.backbone(slots)
            if hidden.ndim == 3:
                hidden = hidden.mean(dim=1)
        head_dtype = next(self.action_adapter.parameters()).dtype
        if hidden.dtype != head_dtype:
            hidden = hidden.to(head_dtype)
        return hidden

    # ------------------------------------------------------------------
    # Policy protocol
    # ------------------------------------------------------------------

    @torch.no_grad()
    def bootstrap_value(self, states: Tensor) -> Tensor:
        """Return the exact current-policy value of fragment leaves.

        The slotwise Q critic represents ``Q(s, a)``.  Sampling one action via
        ``act`` is an unbiased but unnecessarily noisy estimate of ``V^pi``;
        Sokoban has only four actions, so compute the categorical expectation
        exactly.  Scalar-V critic branches retain their direct value output.
        """
        if self.critic_source == "qwen_slotwise_q":
            probabilities = self._actor_logits(states).float().softmax(-1)
            q_values = self._critic_q_values(states).float()
            return (probabilities * q_values).sum(-1)
        return self._critic_value(states)

    def act(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select actions given belief states.

        Returns: (actions, log_probs, entropy, values)
        """
        # For qwen_pooled actor + qwen_pooled critic, compute the shared Qwen
        # hidden once and reuse it for both paths. Other actor_source values
        # don't use the actor hidden (latent_belief, frozen_vlm) or other
        # critic_source values don't reuse it (latent_belief, frozen_vlm).
        hidden = None
        if self.actor_source == "qwen_pooled":
            hidden = self._actor_hidden(states)
        elif self.actor_source == "qwen_slotwise":
            hidden = self._actor_token_hidden(states)
        logits = self._actor_logits(states, hidden=hidden)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()

        # Value estimate (hidden reused only when both sides are qwen_pooled).
        values = self._critic_value(states, hidden=hidden, actions=actions)

        return actions, dist.log_prob(actions), dist.entropy(), values

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate given (state, action) pairs — used by PPO re-evaluation.

        Returns: (log_probs, entropy, values)
        """
        hidden = None
        if self.actor_source == "qwen_pooled":
            hidden = self._actor_hidden(states)
        elif self.actor_source == "qwen_slotwise":
            hidden = self._actor_token_hidden(states)
        logits = self._actor_logits(states, hidden=hidden)
        dist = Categorical(logits=logits)

        values = self._critic_value(states, hidden=hidden, actions=actions)

        return dist.log_prob(actions), dist.entropy(), values

    def evaluate_actor_actions(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Actor-only PPO forward, allowing backward before adapter switch."""
        hidden = None
        if self.actor_source == "qwen_pooled":
            hidden = self._actor_hidden(states)
        elif self.actor_source == "qwen_slotwise":
            hidden = self._actor_token_hidden(states)
        logits = self._actor_logits(states, hidden=hidden)
        if getattr(self, "project_global_action_gradient", False):
            logits = project_batch_global_action_gradient(logits)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def evaluate_values(
        self, states: Tensor, actions: Tensor | None = None,
    ) -> Tensor:
        """Critic-only PPO forward through the critic-owned adapter/head."""
        return self._critic_value(states, hidden=None, actions=actions)

    def evaluate_actions_with_behavior(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Evaluate PPO actions plus exact categorical behavior-reference KL.

        Returns current chosen-action log-probability, entropy, value, all
        current action log-probabilities, and all frozen behavior
        log-probabilities.
        """
        logits, behavior_logits = self._actor_and_behavior_logits(states)
        if getattr(self, "project_global_action_gradient", False):
            logits = project_batch_global_action_gradient(logits)
        dist = Categorical(logits=logits)
        values = self._critic_value(states, actions=actions)
        return (
            dist.log_prob(actions),
            dist.entropy(),
            values,
            torch.log_softmax(logits, dim=-1),
            torch.log_softmax(behavior_logits, dim=-1),
        )

    def evaluate_actor_actions_with_behavior(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Actor-only behavior-KL forward for decoupled PPO optimizers."""
        logits, behavior_logits = self._actor_and_behavior_logits(states)
        dist = Categorical(logits=logits)
        return (
            dist.log_prob(actions),
            dist.entropy(),
            torch.log_softmax(logits, dim=-1),
            torch.log_softmax(behavior_logits, dim=-1),
        )
