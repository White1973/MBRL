"""Single-source configuration for SemBelief-WM.

All hyperparameters live here. No other module should define its own
parameter entry points. Training, evaluation, and data preprocessing
all read from the same config tree.

Reference: world-model-architecture-final.md §4, §9.3, §12, §13
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class BeliefConfig:
    """Belief state configuration (§7, §10)."""
    num_slots: int = 36                 # K_belief — decoupled from K_vis
    gate_bias_init: float = -2.0        # sigmoid(-2) ≈ 0.12


@dataclass
class EncoderConfig:
    """Visual encoder configuration (§8)."""
    encoder_type: Literal["vjepa2", "qwen"] = "vjepa2"
    vjepa2_model: str = "facebook/vjepa2-vitg-fpc64-384-ssv2"
    vjepa2_raw_tokens: int = 576
    vjepa2_raw_dim: int = 1408
    compressed_tokens: int = 36         # K_vis after compression
    # V-JEPA is optional when it is the input encoder, but is explicitly kept
    # as the frozen semantic teacher in the Qwen-native WM recipe. Teacher
    # features are compressed raw V-JEPA tokens (no learned projector), so
    # their dimensionality remains 1408.
    semantic_teacher_type: Literal["none", "vjepa2"] = "none"
    semantic_teacher_tokens: int = 36
    semantic_teacher_dim: int = 1408


@dataclass
class BackboneConfig:
    """Shared VLM backbone configuration (§9)."""
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    attention_mode: Literal["bidirectional", "causal"] = "bidirectional"
    action_conditioning_mode: Literal["embedded", "text", "continuous"] = "embedded"
    # LoRA (§9.3) — single source, referenced by §13.1
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Memory optimization — enables H20 (96 GB) compatibility
    gradient_checkpointing: bool = True
    attn_implementation: str | None = "flash_attention_2"
    quantization: Literal["none", "4bit"] = "none"


@dataclass
class RewardConfig:
    """Reward head configuration (§11, §12.3)."""
    readout: Literal["mean_pool", "attention_pool", "learned_query"] = "mean_pool"
    pos_weight: float = 12.0
    supervision_source: Literal["posterior", "prior", "both"] = "posterior"
    # Binary reward target threshold.  Reward is labeled positive only when
    # raw reward > threshold.  Set to 1.0 for Sokoban to ignore small positive
    # shaping rewards and label only terminal success (+10.9) as positive.
    success_reward_threshold: float = 0.0
    # Optional extra BCE on the terminal transition.  Sparse positives are
    # already handled by ``pos_weight``; keeping this at zero avoids counting
    # the same successful terminal transition twice with a very large weight.
    terminal_aux_weight: float = 0.0
    # Reward classifier capacity. None preserves the legacy D->D->1 MLP.
    # 0 selects a linear D->1 classifier; a positive value selects a compact
    # D->hidden->1 MLP. Sparse terminal reward prediction should generally use
    # 0 or a small bottleneck rather than the legacy multi-million-param head.
    head_hidden_dim: int | None = None


@dataclass
class SIGRegConfig:
    """Anti-collapse regularization (§12.2)."""
    lambda_ep: float = 0.05
    lambda_var: float = 0.005
    lambda_cov: float = 0.0
    num_projections: int = 1024
    integration_range: float = 5.0
    num_quadrature: int = 17
    buffer_size: int = 256
    flush_on_curriculum_switch: bool = True


@dataclass
class EMAConfig:
    """EMA teacher target-stabilization (ablation §2.3)."""
    decay: float = 0.996
    lambda_var: float = 0.01          # only used when anti_collapse.use_ema_variance is enabled


@dataclass
class AntiCollapseConfig:
    """Anti-collapse and target-stabilization switches."""
    use_sigreg: bool = True
    use_ema_target: bool = False
    use_ema_variance: bool = False


@dataclass
class CurriculumConfig:
    """Fixed step-based curriculum (§13.2)."""
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    switch_steps: list[int] = field(default_factory=lambda: [0, 400, 800, 1400])
    horizon_decay: float = 0.9          # w_t = 0.9^(t_local)


@dataclass
class TrainingConfig:
    """Phase 1 training parameters (§13)."""
    total_steps: int = 2000
    episodes_per_step: int = 4
    lr: float = 1e-4
    base_lr_factor: float = 0.01       # base VLM weights LR = lr * factor
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    checkpoint_every: int = 1000
    lambda_dynamics: float = 1.0
    lambda_reward: float = 0.1
    # ``mean`` keeps SIGReg invariant to the number/length of sampled
    # episodes. ``n_scaled`` preserves the legacy EP*N / variance*N behavior.
    sigreg_scale_mode: Literal["mean", "n_scaled"] = "mean"
    # Optional open-loop auxiliaries.  Phase 1 keeps these disabled by
    # default; Phase-2 alternating refresh explicitly enables them through
    # ``WorldModelRefreshConfig``.
    open_dynamics_coef: float = 0.0
    prior_reward_coef: float = 0.0
    open_loop_horizon: int = 0
    open_dynamics_decay: float = 0.9
    prior_reward_decay: float = 1.0
    # Action-aware transition auxiliaries.  Absolute belief MSE admits an
    # identity/average-transition shortcut on slowly changing environments.
    # Delta cosine supervises the direction of the actual state change, while
    # inverse-action CE forces logged actions to remain identifiable in both
    # posterior and prior deltas.
    delta_cosine_coef: float = 0.0
    inverse_action_coef: float = 0.0
    inverse_action_mode: Literal["joint", "prior_frozen"] = "joint"
    inverse_action_lr: float | None = None
    delta_min_rms: float = 1e-4
    # Optional parameter isolation between grounded posterior inference and
    # action-only prior imagination.  ``lora`` selects a second Qwen adapter
    # for prior calls; ``residual`` keeps the shared transition frozen and
    # adds a small action-conditioned correction after the base prior step;
    # ``state_action`` constrains that correction to the pooled latent change
    # learned from a fixed PCA state x discrete-action interaction probe.
    prior_isolation_mode: Literal[
        "shared", "lora", "residual", "state_action"
    ] = "shared"
    prior_lora_adapter_name: str = "wm_prior"
    # An isolated prior repair keeps every frozen module in eval mode while
    # still allowing gradients through the selected prior LoRA.  This makes
    # the released posterior a deterministic target (notably disabling its
    # frozen LoRA dropout) without changing ordinary Stage-1 training.
    isolated_prior_repair: bool = False
    prior_residual_rank: int = 64
    # Anchor grounded posterior slots to the frozen V-JEPA observation tokens.
    # The state term preserves observation identity/spatial content; the delta
    # term specifically preserves action-induced visual change.
    observation_anchor_coef: float = 0.0
    observation_delta_anchor_coef: float = 0.0
    observation_delta_min_rms: float = 1e-4
    observation_anchor_projection_trainable: bool = True
    # Frozen V-JEPA spatial-semantic teacher.  Qwen observation tokens enter
    # the posterior; these losses supervise belief states through a separate
    # predictor and therefore cannot leak V-JEPA features into the input.
    # The prior coefficient is the principal future-state constraint. The
    # posterior term only grounds the Qwen-observed belief; the delta term
    # compares action-conditioned slotwise changes, never mean-pooled states.
    vjepa_teacher_prior_coef: float = 0.0
    vjepa_teacher_posterior_coef: float = 0.0
    vjepa_teacher_delta_coef: float = 0.0
    vjepa_teacher_delta_min_rms: float = 1e-4
    # Fixed slot-aligned visual skip used only by posterior grounding. A
    # non-zero value makes observation information impossible for the shared
    # posterior/prior objective to erase from both branches simultaneously.
    posterior_observation_residual_scale: float = 0.0
    posterior_grounding_mode: Literal["legacy_residual", "visual_anchor"] = (
        "legacy_residual"
    )
    posterior_recurrent_residual_scale: float = 0.25
    posterior_action_free: bool = False
    # Validation safety guard.  A run is stopped only after the configured
    # number of consecutive validation failures, and an emergency checkpoint
    # is written first.
    validation_guard_enabled: bool = False
    validation_guard_patience: int = 2
    validation_reward_predicted_positive_max: float = 0.98
    validation_dynamics_degradation_factor: float = 2.0
    dtype: Literal["bf16", "fp32"] = "bf16"


@dataclass
class ActorCriticConfig:
    """Phase 2 actor/value heads on top of latent beliefs."""
    policy_family: Literal["mlp", "vlm_action_head"] = "vlm_action_head"
    readout: Literal["mean_pool", "attention_pool", "learned_query"] = "mean_pool"
    hidden_dim: int = 1024
    hidden_layers: int = 2
    activation: Literal["gelu", "relu", "tanh"] = "gelu"


@dataclass
class FreeformPolicyConfig:
    """Route B free-decoding VLM policy configuration."""
    update_mode: Literal["token_ppo", "hybrid_scoring"] = "token_ppo"
    model_name: str | None = None
    projector_tokens: int = 8
    projector_heads: int = 8
    max_new_tokens: int = 512
    max_actions_per_turn: int = 1
    action_separator: str = ","
    prompt_format: Literal["free_think", "wm"] = "free_think"
    use_prompt_examples: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    answer_open_tag: str = "<answer>"
    answer_close_tag: str = "</answer>"
    prompt_template: str = (
        "You are an agent acting in the environment {env_name}. "
        "Choose exactly one action from the allowed actions below.\n"
        "{action_list}\n"
        "Respond using only {open_tag}ACTION{close_tag}."
    )
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class LookaheadConfig:
    """Lookahead planning wrapper for action selection (inference-time only)."""
    enabled: bool = False
    expansion_depth: int = 1       # how many layers of exhaustive action expansion
    expansion_width: int = 4       # actions to expand per node (num_actions = exhaustive)
    rollout_horizon: int = 4       # policy rollout steps after expansion to estimate value
    discount: float = 0.99         # gamma for cumulative reward scoring
    use_value_bootstrap: bool = True  # use V(Z_leaf) at the end of rollout


@dataclass
class PPOConfig:
    """Phase 2 PPO hyperparameters for latent-space control."""
    total_updates: int = 2000
    rollout_source: Literal["imagined", "real_env"] = "imagined"  # where PPO trajectories come from
    rollout_batch_size: int = 32
    rollout_horizon: int = 8
    # A rollout horizon truncates an imagined fragment; it is not necessarily
    # an environment terminal.  When enabled, non-terminal leaves bootstrap
    # from the Critic while genuinely terminated samples still use zero.
    use_value_bootstrap: bool = False
    # Keep rollout termination independent from reward shaping.  Until the
    # prior reward head is calibrated as a per-transition termination model,
    # fixed-horizon imagination prevents false positives from shrinking an
    # H-step PPO batch to a single step.
    imagination_termination_mode: Literal[
        "fixed_horizon", "predicted_success"
    ] = "fixed_horizon"
    # Collect this many detached rollout chunks before one PPO update. This
    # increases the advantage sample size without increasing Qwen forward
    # memory for an individual imagined rollout.
    rollouts_per_update: int = 1
    epochs_per_update: int = 1
    minibatch_size: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    critic_warmup_min_updates: int = 0
    critic_warmup_ev_threshold: float = 0.2
    critic_warmup_ev_patience: int = 3
    critic_warmup_validation_fraction: float = 0.2
    critic_warmup_validation_size: int = 256
    critic_warmup_replay_capacity: int = 4096
    critic_warmup_train_samples: int = 512
    critic_warmup_ev_ema_alpha: float = 0.2
    critic_warmup_mse_improvement: float = 0.05
    weight_decay: float = 0.0
    recompute_old_log_probs: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.0               # KL penalty against old policy (0 = disabled)
    target_kl: float | None = None      # early-stop PPO minibatches above this KL
    behavior_kl_coef: float = 0.0       # KL(pi || frozen offline-BC policy)
    behavior_bc_coef: float = 0.0       # expert CE on real posterior beliefs
    behavior_bc_batch_size: int = 32
    offline_bc_steps: int = 0           # pre-PPO offline latent BC updates
    offline_bc_batch_size: int = 32
    offline_bc_cache_size: int = 2048
    offline_bc_strategies: tuple[str, ...] = ()
    offline_bc_lr: float = 1e-4
    # Compatibility for legacy tokenized Sokoban data/checkpoints trained on
    # env action ids 1..4. Policy ids stay 0..3; imagined actions are shifted
    # before the frozen WM, and offline BC labels are shifted back.
    wm_action_id_offset: int = 0
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    # Entropy floor (anti-collapse); target_entropy=None disables.
    target_entropy: float | None = None
    entropy_floor_coef: float = 0.1
    reward_mapping: Literal[
        "sigmoid_affine",
        "raw_sigmoid",
        "clipped_logit",
        "terminal_success",
        "terminal_success_scaled",
        "terminal_success_conservative",
        "per_transition_success_conservative",
    ] = "raw_sigmoid"
    reward_scale: float = 1.0
    reward_confidence_floor: float = 0.5
    reward_low_confidence_scale: float = 0.1
    checkpoint_every: int = 100
    eval_every: int = 50                 # run real-env eval every N updates (0 = disabled)
    eval_episodes: int = 20              # episodes per eval round
    eval_max_steps: int = 25             # max steps per eval episode
    collect_every: int = 20              # run real-env data collection every N updates (0 = disabled)
    collect_episodes: int = 4            # episodes per collection round
    collect_max_steps: int = 0           # max steps per collected episode (<=0 uses adapter default)
    # Model-free real-env rollout settings (used when rollout_source="real_env")
    real_rollout_episodes: int = 16      # episodes per PPO update in real-env mode
    real_rollout_max_steps: int = 25     # max steps per real-env rollout episode


@dataclass
class WorldModelRefreshConfig:
    """Periodic supervised world-model refresh during Phase 2.

    This does not backpropagate RL gradients through the world model.
    Instead, it reuses Phase 1 supervision on offline data between PPO
    updates, which is closer to Dreamer-style alternating optimization.
    """

    warmup_ratio: float = 0.0           # offline WM warm-up before PPO, as fraction of training.total_steps
    refresh_every: int = 1               # run WM refresh every N PPO updates (0 = disabled)
    updates_per_refresh: int = 4         # number of supervised WM steps per refresh
    data_mix_mode: Literal["offline_only", "mixed", "online_only", "custom"] = "mixed"
    online_data_ratio: float = 0.5       # fraction of online replay in each WM refresh batch
    batch_size: int = 4                  # episodes per supervised WM step
    lr: float = 1e-4
    base_lr_factor: float = 0.01
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    horizon: int = 8                     # fixed BPTT horizon for WM refresh
    # Reward optimization is independent from the original Phase-1 settings.
    # ``None`` preserves the checkpoint/config value; dynamics-only refreshes
    # set reward_loss_coef=0 and freeze_reward_head=True.
    reward_pos_weight: float | None = None
    reward_loss_coef: float | None = None
    freeze_reward_head: bool = False
    validation_batches: int = 0         # fixed held-out batches per refresh; 0 disables
    # Open-loop refresh starts from the grounded posterior at the first
    # observation in a window, then consumes actions only.  Dynamics gradients
    # train the WM; prior-reward gradients are stopped at the imagined belief
    # and train only the reward head.
    open_dynamics_coef: float = 0.25
    prior_reward_coef: float = 0.5
    open_loop_horizon: int = 4
    open_dynamics_decay: float = 0.9
    prior_reward_decay: float = 1.0
    delta_cosine_coef: float = 0.0
    inverse_action_coef: float = 0.0
    inverse_action_mode: Literal["joint", "prior_frozen"] = "joint"
    inverse_action_lr: float | None = None


@dataclass
class Phase2Config:
    """Phase 2 latent RL configuration."""
    algorithm: Literal["ppo"] = "ppo"
    world_model_mode: Literal["frozen_wm", "alternating_wm", "joint_wm"] = "alternating_wm"
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    freeform_policy: FreeformPolicyConfig = field(default_factory=FreeformPolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    lookahead: LookaheadConfig = field(default_factory=LookaheadConfig)
    wm_refresh: WorldModelRefreshConfig = field(default_factory=WorldModelRefreshConfig)


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    enabled: bool = True
    project: str = "mbrl-vlm"
    entity: str | None = None           # None = use logged-in account
    run_name: str | None = None
    group: str | None = None            # groups Phase 1 + Phase 2 runs together
    job_type: str | None = None         # "phase1" or "phase2"
    log_interval: int = 1


@dataclass
class EnvRewardSpec:
    """Per-environment reward semantics for Phase 2 reward mapping."""
    positive_value: float = 10.9        # reward when task-relevant event happens
    negative_value: float = -0.1        # step penalty / default reward
    positive_threshold: float = 0.0     # binary target: 1[r > threshold]


@dataclass
class EnvironmentConfig:
    """Environment and multi-env conditioning parameters."""
    num_actions: int = 4                # global max discrete actions across all envs
    null_action_id: int = 4             # reserved start-of-window action, not an env action
    null_action_text: str = "start"
    env_ids: list[str] = field(default_factory=lambda: ["default"])
    unified_obs_tokens: int = 36
    action_type: Literal["discrete", "continuous"] = "discrete"
    action_dim: int | None = None
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    # Per-env reward semantics; keys must be subset of env_ids
    reward_specs: dict[str, EnvRewardSpec] = field(default_factory=dict)

    def reward_spec_for(self, env_id: str) -> EnvRewardSpec:
        """Get reward spec for an environment, falling back to defaults."""
        return self.reward_specs.get(env_id, EnvRewardSpec())


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Single-source config for the entire system.

    Usage:
        cfg = Config()                   # all defaults
        cfg = Config(belief=BeliefConfig(num_slots=48))  # override one field
    """
    hidden_dim: int = 3584              # D — aligned with Qwen2.5-VL-7B
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    anti_collapse: AntiCollapseConfig = field(default_factory=AntiCollapseConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self) -> None:
        if self.env.action_type == "discrete" and self.env.null_action_id != self.env.num_actions:
            raise ValueError(
                "null_action_id must equal num_actions so it occupies one extra "
                "reserved id outside the environment action set."
            )
        if not self.env.env_ids:
            raise ValueError("env_ids must contain at least one environment id.")
        if len(set(self.env.env_ids)) != len(self.env.env_ids):
            raise ValueError(f"env_ids must be unique, got {self.env.env_ids}.")
        if self.env.action_type == "continuous" and self.env.action_dim is None:
            raise ValueError("continuous actions require env.action_dim.")
        if self.env.action_type == "continuous":
            if self.env.action_low is not None and len(self.env.action_low) != self.env.action_dim:
                raise ValueError("continuous actions require action_low to match action_dim.")
            if self.env.action_high is not None and len(self.env.action_high) != self.env.action_dim:
                raise ValueError("continuous actions require action_high to match action_dim.")
        if self.anti_collapse.use_ema_variance and not self.anti_collapse.use_ema_target:
            raise ValueError(
                "anti_collapse.use_ema_variance requires anti_collapse.use_ema_target."
            )
