#!/usr/bin/env python3
"""Run position-aware v2 repair of the released independent Qwen prior LoRA."""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sembelief_wm.diagnostics.sokoban_prior_repair_v2 import (  # noqa: E402
    PriorRepairV2Config,
    run_prior_repair_v2,
)
from sembelief_wm.model import QwenTransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.model.checkpoint_semantics import (  # noqa: E402
    validate_world_model_semantics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-only Spatial Teacher + changed-slot V-JEPA prior repair."
    )
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--repair-validation-episodes", type=int, default=500)
    parser.add_argument("--teacher-epochs", type=int, default=1)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--repair-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--wandb-project", default="jiayu-mbrl")
    parser.add_argument("--wandb-group", default="qwen_vjepa_prior_repair_v2")
    parser.add_argument("--wandb-run-name", default="qwen_vjepa_prior_repair_v2")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.wm_checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {checkpoint_path}")
    if not (data_dir / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"paired replay is missing: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "latest.pt").exists() or (output_dir / "best.pt").exists():
        raise FileExistsError(
            f"refusing to overwrite v2 checkpoints in {output_dir}; "
            "choose another output directory"
        )

    # Read architecture/split metadata without materializing tensor storage.
    metadata = torch.load(
        checkpoint_path, map_location="meta", weights_only=False, mmap=True
    )
    if not isinstance(metadata, dict) or "config" not in metadata:
        raise ValueError("source is not a self-contained Phase-1 checkpoint")
    config = metadata["config"]
    if config.encoder.encoder_type != "qwen":
        raise ValueError("prior repair v2 requires a Qwen-native World Model")
    if config.belief.num_slots != 36 or config.encoder.compressed_tokens != 36:
        raise ValueError("prior repair v2 requires the ordered 6x6 slot geometry")
    if config.training.prior_isolation_mode != "lora":
        raise ValueError("source checkpoint has no independent prior LoRA")
    if not metadata.get("prior_repair"):
        raise ValueError("source checkpoint is not an isolated prior-repair artifact")
    del metadata

    device = torch.device(args.device)
    print(f"Loading Qwen prior-repair checkpoint: {checkpoint_path}", flush=True)
    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    world_model = WorldModel(config, backbone).to(device)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("source checkpoint has no model state")
    world_model.load_state_dict(state, strict=True)
    validate_world_model_semantics(
        payload,
        attention_mode=config.backbone.attention_mode,
        context=f"prior repair v2 source {checkpoint_path}",
    )
    # Retain only metadata needed by the v2 artifact.  The source model and
    # optimizer would otherwise duplicate several GB of CPU memory.
    source_metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "scheduler"}
    }
    del state, payload
    gc.collect()

    repair_config = PriorRepairV2Config(
        seed=args.seed,
        repair_validation_episodes=args.repair_validation_episodes,
        teacher_epochs=args.teacher_epochs,
        teacher_batch_size=args.teacher_batch_size,
        teacher_lr=args.teacher_lr,
        repair_steps=args.repair_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.wandb_run_name,
                job_type="wm_prior_repair_v2",
                config={
                    "source_checkpoint": str(checkpoint_path.resolve()),
                    **repair_config.__dict__,
                },
            )
        except Exception as exc:
            if os.environ.get("REQUIRE_WANDB", "0") == "1":
                raise
            print(
                f"Warning: W&B disabled ({type(exc).__name__}: {exc})",
                flush=True,
            )
    try:
        report = run_prior_repair_v2(
            world_model=world_model,
            source_checkpoint=checkpoint_path,
            source_payload=source_metadata,
            paired_data_dir=data_dir,
            output_dir=output_dir,
            device=device,
            config=repair_config,
            wandb_run=wandb_run,
        )
        print(
            "Prior repair v2 complete: " + str(report["best_checkpoint"]),
            flush=True,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
