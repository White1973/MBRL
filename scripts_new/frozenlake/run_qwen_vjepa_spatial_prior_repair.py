#!/usr/bin/env python3
"""Repair only the shared Qwen WM prior LoRA using FrozenLake spatial labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sembelief_wm.data.datasource import TokenizedEpisodeDataset  # noqa: E402
from sembelief_wm.diagnostics.frozenlake_lora_spatial_repair import (  # noqa: E402
    SpatialRepairGates,
    SpatialRepairWeights,
    run_lora_spatial_prior_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--repair-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--probe-train-episodes", type=int, default=1000)
    parser.add_argument("--probe-validation-episodes", type=int, default=500)
    parser.add_argument("--selection-validation-episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--posterior-gate", type=float, default=0.98)
    parser.add_argument("--prior-gate", type=float, default=0.75)
    parser.add_argument("--counterfactual-gate", type=float, default=0.70)
    parser.add_argument("--changed-gate", type=float, default=0.60)
    parser.add_argument("--noop-gate", type=float, default=0.70)
    parser.add_argument("--minimum-action-gate", type=float, default=0.55)

    parser.add_argument("--actual-position-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-position-weight", type=float, default=1.0)
    parser.add_argument("--posterior-latent-weight", type=float, default=1.0)
    parser.add_argument("--posterior-cosine-weight", type=float, default=1.0)
    parser.add_argument("--posterior-delta-weight", type=float, default=0.50)
    parser.add_argument("--vjepa-prior-weight", type=float, default=0.25)
    parser.add_argument("--vjepa-delta-weight", type=float, default=0.50)

    parser.add_argument("--wandb-project", default="jiayu-mbrl")
    parser.add_argument("--wandb-group", default="frozenlake_qwen_vjepa_spatial_prior_v3")
    parser.add_argument("--wandb-run-name", default="frozenlake_spatial_prior_v3_seed11")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    positive_ints = {
        "probe_steps": args.probe_steps,
        "repair_steps": args.repair_steps,
        "eval_every": args.eval_every,
        "probe_train_episodes": args.probe_train_episodes,
        "probe_validation_episodes": args.probe_validation_episodes,
        "selection_validation_episodes": args.selection_validation_episodes,
        "batch_size": args.batch_size,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        parser.error("positive values required for: " + ", ".join(invalid))
    if args.lr <= 0:
        parser.error("--lr must be positive")

    gates = SpatialRepairGates(
        posterior_accuracy=args.posterior_gate,
        prior_accuracy=args.prior_gate,
        counterfactual_accuracy=args.counterfactual_gate,
        changed_accuracy=args.changed_gate,
        noop_accuracy=args.noop_gate,
        minimum_action_accuracy=args.minimum_action_gate,
    )
    weights = SpatialRepairWeights(
        actual_position=args.actual_position_weight,
        counterfactual_position=args.counterfactual_position_weight,
        posterior_latent=args.posterior_latent_weight,
        posterior_cosine=args.posterior_cosine_weight,
        posterior_delta=args.posterior_delta_weight,
        vjepa_prior=args.vjepa_prior_weight,
        vjepa_delta=args.vjepa_delta_weight,
    )
    if any(value < 0 for value in vars(weights).values()):
        parser.error("all loss weights must be non-negative")

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            job_type="wm_prior_spatial_repair",
            config={**vars(args), "gates": vars(gates), "weights": vars(weights)},
        )
    try:
        dataset = TokenizedEpisodeDataset.from_directory(args.data_dir)
        report = run_lora_spatial_prior_repair(
            checkpoint=args.wm_checkpoint,
            dataset=dataset,
            output_dir=args.output_dir,
            device=torch.device(args.device),
            seed=args.seed,
            probe_steps=args.probe_steps,
            repair_steps=args.repair_steps,
            eval_every=args.eval_every,
            probe_train_episodes=args.probe_train_episodes,
            probe_validation_episodes=args.probe_validation_episodes,
            selection_validation_episodes=args.selection_validation_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            gates=gates,
            weights=weights,
            resume=args.resume,
            wandb_run=wandb_run,
        )
        print("SPATIAL_PRIOR_REPAIR_PASS " + json.dumps(report, sort_keys=True))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
