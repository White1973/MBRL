#!/usr/bin/env python3
"""Run the real Stage-1 acceptance gate for Qwen + V-JEPA-teacher WM."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sembelief_wm.diagnostics.sokoban_stage1_audit import (  # noqa: E402
    AuditThresholds,
    run_sokoban_stage1_semantic_audit,
)
from sembelief_wm.model import QwenTransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.model.checkpoint_semantics import (  # noqa: E402
    validate_world_model_semantics,
)


def _metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="meta", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError(f"{path} is not a self-contained Phase-1 WM checkpoint.")
    refresh = payload.get("wm_only_refresh")
    if not isinstance(refresh, dict):
        raise ValueError(
            "Checkpoint has no wm_only_refresh split metadata. Refusing to "
            "invent a new audit split after training."
        )
    if not refresh.get("train_indices") or not refresh.get("val_indices"):
        raise ValueError("Checkpoint is missing its fixed train/validation indices.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out slot-wise Sokoban Stage-1 semantic / dynamics gate."
    )
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="Paired tokenized replay directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-epochs", type=int, default=1)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--max-train-episodes", type=int, default=0, help="0 audits all checkpoint training episodes")
    parser.add_argument("--max-validation-episodes", type=int, default=0, help="0 audits all checkpoint validation episodes")
    parser.add_argument("--min-posterior-player", type=float, default=0.75)
    parser.add_argument("--min-posterior-box", type=float, default=0.75)
    parser.add_argument("--min-posterior-target", type=float, default=0.75)
    parser.add_argument("--min-posterior-wall-macro-f1", type=float, default=0.75)
    parser.add_argument("--min-posterior-baseline-margin", type=float, default=0.25)
    parser.add_argument("--min-logged-changed-joint", type=float, default=0.35)
    parser.add_argument("--min-counterfactual-changed-joint", type=float, default=0.30)
    parser.add_argument("--min-dynamics-copy-margin", type=float, default=0.20)
    parser.add_argument("--min-validation-states", type=int, default=500)
    parser.add_argument("--min-counterfactual-changed-states", type=int, default=200)
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.wm_checkpoint)
    output_path = Path(args.output)
    metadata = _metadata(checkpoint_path)
    config = metadata["config"]
    if config.encoder.encoder_type != "qwen":
        raise ValueError("This audit is only valid for a Qwen-native World Model.")
    if config.encoder.compressed_tokens != 36 or config.belief.num_slots != 36:
        raise ValueError("This Sokoban audit requires 36 ordered spatial slots.")
    device = torch.device(args.device)
    print(f"Loading Qwen-native WM checkpoint: {checkpoint_path}", flush=True)
    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    world_model = WorldModel(config, backbone).to(device)
    # The full checkpoint includes optimizer state for resumability. Load it on
    # CPU so the audit does not duplicate the optimizer on GPU.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Phase-1 checkpoint has no model state dictionary.")
    world_model.load_state_dict(model_state, strict=True)
    validate_world_model_semantics(
        payload,
        attention_mode=config.backbone.attention_mode,
        context=f"Stage-1 audit checkpoint {checkpoint_path}",
    )
    checkpoint_step = int(payload.get("step", 0))
    refresh = payload["wm_only_refresh"]
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    thresholds = AuditThresholds(
        posterior_player_accuracy=args.min_posterior_player,
        posterior_box_accuracy=args.min_posterior_box,
        posterior_target_accuracy=args.min_posterior_target,
        posterior_wall_macro_f1=args.min_posterior_wall_macro_f1,
        posterior_baseline_margin=args.min_posterior_baseline_margin,
        logged_changed_joint_accuracy=args.min_logged_changed_joint,
        counterfactual_changed_joint_accuracy=args.min_counterfactual_changed_joint,
        dynamics_copy_margin=args.min_dynamics_copy_margin,
        minimum_validation_states=args.min_validation_states,
        minimum_counterfactual_changed_states=args.min_counterfactual_changed_states,
    )
    report = run_sokoban_stage1_semantic_audit(
        world_model=world_model,
        paired_data_dir=args.data_dir,
        train_indices=list(refresh["train_indices"]),
        validation_indices=list(refresh["val_indices"]),
        device=device,
        batch_size=args.batch_size,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        max_train_episodes=args.max_train_episodes,
        max_validation_episodes=args.max_validation_episodes,
        thresholds=thresholds,
    )
    report["checkpoint"] = {"path": str(checkpoint_path.resolve()), "step": checkpoint_step}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate = report["gate"]
    print(
        "Sokoban Stage-1 semantic gate: " + (
            "PASS" if gate["passed"] else "FAIL: " + "; ".join(gate["failures"])
        ),
        flush=True,
    )
    print(f"Stage-1 semantic audit report: {output_path}", flush=True)
    if args.enforce_gate and not gate["passed"]:
        raise RuntimeError(f"Stage-1 semantic gate failed: {output_path}")


if __name__ == "__main__":
    main()
