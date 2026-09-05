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
    dtype: Literal["bf16", "fp32"] = "bf16"


@dataclass
class ActorCriticConfig:
    """Phase 2 actor/value heads on top of latent beliefs."""
    policy_family: Literal["mlp", "vlm_freeform", "vlm_action_head"] = "vlm_action_head"
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
    rollout_batch_size: int = 32
    rollout_horizon: int = 8
    epochs_per_update: int = 1
    minibatch_size: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.5
    kl_coef: float = 0.0               # KL penalty against old policy (0 = disabled)
    target_kl: float | None = None
    max_grad_norm: float = 0.5
    normalize_advantages: bool = False
    reward_mapping: Literal[
        "sigmoid_affine",
        "raw_sigmoid",
        "clipped_logit",
        "terminal_success",
        "terminal_success_scaled",
        "terminal_success_conservative",
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
    reward_pos_weight: float | None = None
    reward_loss_coef: float | None = None
    freeze_reward_head: bool = False
    validation_batches: int = 0


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
