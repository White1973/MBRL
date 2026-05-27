"""Train a real-environment Qwen2.5-VL freeform PPO baseline.

This baseline removes world modeling and imagination entirely:
raw RGB -> prompted Qwen2.5-VL -> freeform text action -> env reward PPO
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sembelief_wm.agent.real_env_ppo import RGBFreeformPolicy, RealEnvPPOTrainer  # noqa: E402
from sembelief_wm.config import (  # noqa: E402
    ActorCriticConfig,
    BackboneConfig,
    Config,
    EnvironmentConfig,
    FreeformPolicyConfig,
    Phase2Config,
    PPOConfig,
    TrainingConfig,
    WandbConfig,
)


class PrintLogger:
    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
        print(f"[update {step:>5d}] {' | '.join(parts)}")


class WandbLogger:
    def __init__(self, project: str, run_name: str | None, *, config: dict[str, object]) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(project=project, name=run_name, job_type="real_env_ppo", config=config)

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        self._wandb.log(metrics, step=step)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        hidden_dim=3584,
        backbone=BackboneConfig(
            model_name=args.backbone_model_name,
            attention_mode="causal",
            action_conditioning_mode="text",
        ),
        training=TrainingConfig(dtype=args.dtype),
        phase2=Phase2Config(
            world_model_mode="frozen_wm",
            actor_critic=ActorCriticConfig(policy_family="vlm_freeform"),
            freeform_policy=FreeformPolicyConfig(
                update_mode="token_ppo",
                model_name=args.backbone_model_name,
                max_new_tokens=args.max_new_tokens,
                max_actions_per_turn=args.max_actions_per_turn,
                action_separator=args.action_separator,
                prompt_format=args.prompt_format,
                use_prompt_examples=not args.no_prompt_examples,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=not args.greedy_rollout,
            ),
            ppo=PPOConfig(
                total_updates=args.total_updates,
                rollout_batch_size=args.rollout_batch_size,
                rollout_horizon=0,
                epochs_per_update=args.epochs_per_update,
                minibatch_size=args.minibatch_size,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                weight_decay=args.weight_decay,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                kl_coef=args.kl_coef,
                max_grad_norm=args.max_grad_norm,
                normalize_advantages=args.normalize_advantages,
                checkpoint_every=args.checkpoint_every,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                eval_max_steps=args.max_turns,
            ),
        ),
        wandb=WandbConfig(
            enabled=not args.no_wandb,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        ),
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
            env_ids=["sokoban"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-environment RGB freeform PPO baseline")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/real-env-ppo")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone-model-name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--total-updates", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=8, help="Episodes per PPO update")
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum policy turns per episode")
    parser.add_argument("--env-max-steps", type=int, default=100, help="Primitive env-step cap inside Sokoban")
    parser.add_argument("--min-solution-steps", type=int, default=1, help="Reject Sokoban resets shorter than this BFS solution length")
    parser.add_argument("--max-solution-steps", type=int, default=5, help="Reject Sokoban resets longer than this BFS solution length")
    parser.add_argument("--max-reset-attempts", type=int, default=128, help="Max reset resamples when enforcing solution-length filters")
    parser.add_argument("--epochs-per-update", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-actions-per-turn", type=int, default=3)
    parser.add_argument("--action-separator", type=str, default=",")
    parser.add_argument("--prompt-format", choices=["free_think", "wm"], default="wm")
    parser.add_argument("--no-prompt-examples", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy-rollout", action="store_true")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-trace-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="mbrl-vlm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    config = build_config(args)
    policy = RGBFreeformPolicy(config, device=device)

    logger = PrintLogger()
    if config.wandb.enabled:
        try:
            logger = WandbLogger(
                config.wandb.project,
                config.wandb.run_name,
                config={
                    "baseline": "real_env_rgb_freeform_ppo",
                    "model_name": args.backbone_model_name,
                    "rollout_batch_size": args.rollout_batch_size,
                    "max_turns": args.max_turns,
                    "env_max_steps": args.env_max_steps,
                    "epochs_per_update": args.epochs_per_update,
                    "minibatch_size": args.minibatch_size,
                    "actor_lr": args.actor_lr,
                    "critic_lr": args.critic_lr,
                    "entropy_coef": args.entropy_coef,
                    "prompt_format": args.prompt_format,
                    "max_actions_per_turn": args.max_actions_per_turn,
                },
            )
        except Exception as exc:
            print(f"W&B disabled for this run: {exc}")

    trainer = RealEnvPPOTrainer(
        config=config,
        policy=policy,
        device=device,
        logger=logger,
        checkpoint_dir=args.checkpoint_dir,
        adapter_kwargs={
            "max_steps": args.env_max_steps,
            "min_solution_steps": args.min_solution_steps,
            "max_solution_steps": args.max_solution_steps,
            "max_reset_attempts": args.max_reset_attempts,
        },
    )
    start_update = 0
    if args.resume:
        start_update = trainer.load_checkpoint(args.resume)
        print(f"Resumed checkpoint from update {start_update}")
    if args.eval_only:
        trace_path = args.eval_trace_path
        if trace_path is None:
            trace_path = str(Path(args.checkpoint_dir) / "eval_only_traces.jsonl")
        print(f"Running eval-only on {device}")
        print(f"  Episodes: {args.eval_episodes}")
        print(f"  Max turns: {args.max_turns}")
        print(f"  Env max steps: {args.env_max_steps}")
        print(f"  Solution length filter: [{args.min_solution_steps}, {args.max_solution_steps}]")
        print(f"  Trace log: {trace_path}")
        metrics, _ = trainer.evaluate_with_traces(
            num_episodes=args.eval_episodes,
            trace_path=trace_path,
        )
        if logger is not None:
            logger.log_scalars(0, metrics)
        for key, value in sorted(metrics.items()):
            print(f"{key}={value:.4f}")
        return
    print(f"Training real-env PPO on {device}")
    print(f"  Backbone: {args.backbone_model_name}")
    print("  Observation: raw RGB")
    print("  Action interface: freeform text -> parser -> discrete action")
    print("  Reward source: real environment reward")
    print(f"  Rollout batch size: {args.rollout_batch_size} episodes")
    print(f"  Max turns per episode: {args.max_turns}")
    print(f"  Env max steps: {args.env_max_steps}")
    print(f"  Prompt format: {args.prompt_format}")
    print(f"  Max actions per turn: {args.max_actions_per_turn}")
    print(f"  Solution length filter: [{args.min_solution_steps}, {args.max_solution_steps}]")
    print(f"  Trace log: {Path(args.checkpoint_dir) / 'freeform_traces.jsonl'}")
    trainer.train(start_update=start_update)


if __name__ == "__main__":
    main()
