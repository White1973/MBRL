#!/usr/bin/env python3
"""Read-only release checks before starting Qwen+V-JEPA formal PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-report", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--env-id", choices=("sokoban", "frozenlake"), default="sokoban"
    )
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-levels", type=Path)
    evaluation.add_argument("--eval-seeds", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reward-confidence-floor", type=float, required=True)
    args = parser.parse_args()

    evaluation_path = args.eval_levels or args.eval_seeds
    assert evaluation_path is not None
    required_files = (
        args.critic_checkpoint,
        args.critic_report,
        args.reward_checkpoint,
        args.data_dir / "manifest.jsonl",
        evaluation_path,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError("missing PPO prerequisite(s): " + ", ".join(missing))

    report = json.loads(args.critic_report.read_text())
    if not report.get("gate", {}).get("passed", False):
        raise RuntimeError("Stage-3 Critic warmup report did not pass its Gate")
    if not _same_path(
        report.get("best_checkpoint", ""), args.critic_checkpoint
    ):
        raise RuntimeError("Critic report best_checkpoint does not match PPO source")
    if not _same_path(
        report.get("source_reward_head_checkpoint", ""),
        args.reward_checkpoint,
    ):
        raise RuntimeError("Critic and PPO use different WM/Reward-Head sources")
    if report.get("target_semantics") != "learned_wm_reward_h2_counterfactual":
        raise RuntimeError("Critic target semantics do not match exact-H2 PPO")

    critic = _load(args.critic_checkpoint)
    critic_checks = {
        "critic_warmup_complete": critic.get("critic_warmup_complete") is True,
        "actor_was_frozen": int(critic.get("actor_ppo_updates", -1)) == 0,
        "critic_has_four_bucket_validation": len(
            critic.get("critic_warmup_validation_bucket_names", [])
        ) == 4,
        "critic_target_semantics": critic.get("critic_target_semantics")
        == "learned_wm_reward_h2_counterfactual",
        "critic_is_stage3_release": critic.get("training_stage")
        == "critic_pretrain",
    }
    failed = [name for name, passed in critic_checks.items() if not passed]
    if failed:
        raise RuntimeError("invalid Critic release checkpoint: " + ", ".join(failed))
    saved_floor = float(critic.get("critic_reward_confidence_floor", float("nan")))
    if abs(saved_floor - args.reward_confidence_floor) > 1e-8:
        raise RuntimeError(
            "Reward confidence floor differs from Critic warmup: "
            f"checkpoint={saved_floor}, requested={args.reward_confidence_floor}"
        )

    reward = _load(args.reward_checkpoint)
    reward_config = reward.get("config")
    reward_env_ids = tuple(getattr(getattr(reward_config, "env", None), "env_ids", ()))
    if args.env_id not in reward_env_ids:
        raise RuntimeError(
            f"Reward Head is for env_ids={reward_env_ids}, not {args.env_id!r}"
        )
    injection = reward.get("reward_head_injection", {})
    trained_horizons = {int(value) for value in injection.get("horizons", [])}
    if not {1, 2}.issubset(trained_horizons):
        raise RuntimeError("Reward Head does not cover both H1 and H2")
    if not injection.get("independent_horizon_starts", False):
        raise RuntimeError("Reward Head lacks independent-horizon training")
    reward_gate = injection.get("gate", {})
    if reward_gate and not reward_gate.get("passed", False):
        raise RuntimeError("Reward Head checkpoint contains a failed quality Gate")

    manifest_entries = [
        json.loads(line)
        for line in (args.data_dir / "manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    manifest_env_ids = {entry.get("env_id") for entry in manifest_entries}
    manifest_tokenizers = {entry.get("tokenizer") for entry in manifest_entries}
    if manifest_env_ids != {args.env_id}:
        raise RuntimeError(
            f"PPO replay env_ids={sorted(manifest_env_ids)}, expected {args.env_id!r}"
        )
    if manifest_tokenizers != {"qwen2.5_vl_native+vjepa2_teacher"}:
        raise RuntimeError(
            "PPO replay is not native-Qwen input plus separate V-JEPA teacher: "
            f"tokenizers={sorted(str(value) for value in manifest_tokenizers)}"
        )

    eval_payload = json.loads(evaluation_path.read_text())
    if args.env_id == "sokoban":
        if args.eval_levels is None:
            raise RuntimeError("Sokoban PPO requires --eval-levels")
        levels = eval_payload.get("levels", [])
        if len(levels) != 256 or int(eval_payload.get("count", 256)) != 256:
            raise RuntimeError(
                "formal Sokoban evaluation must use all 256 fixed VAGEN levels, "
                f"got {len(levels)}"
            )
        eval_protocol = "fixed VAGEN Sokoban 256-level panel"
    else:
        if args.eval_seeds is None:
            raise RuntimeError("FrozenLake PPO requires --eval-seeds")
        seeds = [int(value) for value in eval_payload.get("seeds", [])]
        if seeds != list(range(128)):
            raise RuntimeError(
                "formal FrozenLake evaluation must use VAGEN seeds 0..127 exactly"
            )
        eval_protocol = "fixed VAGEN FrozenLake seeds 0..127"

    latest = args.output_dir / "latest.pt"
    best = args.output_dir / "best.pt"
    resume_summary: dict[str, int] = {}
    if args.resume:
        if not latest.is_file():
            raise RuntimeError(f"RESUME=1 but latest.pt is missing: {latest}")
        resumed = _load(latest)
        if resumed.get("training_stage") != "ppo":
            raise RuntimeError("resume checkpoint is not a PPO checkpoint")
        if resumed.get("metric_step_axis") != "actor_ppo_update":
            raise RuntimeError("resume checkpoint did not use the Actor PPO step axis")
        resumed_actor = int(resumed.get("actor_ppo_updates", -1))
        online_target = int(
            resumed.get("unified_replay", {}).get("online_target", 800)
        )
        expected_visible = min(resumed_actor, online_target)
        manifest = args.output_dir / "online_replay/tokenized/manifest.jsonl"
        stored_online = (
            sum(1 for line in manifest.open() if line.strip())
            if manifest.is_file()
            else 0
        )
        if stored_online < expected_visible:
            raise RuntimeError(
                "online replay is behind the resume checkpoint: "
                f"stored={stored_online}, expected={expected_visible}"
            )
        resume_summary = {
            "resume_global_update": int(resumed.get("update", -1)),
            "resume_actor_ppo_update": resumed_actor,
            "stored_online_episodes": stored_online,
            "initial_visible_online_episodes": expected_visible,
        }
    elif latest.exists() or best.exists():
        raise RuntimeError(
            f"refusing to overwrite an existing PPO run: {args.output_dir}"
        )

    result = {
        "critic_gate": "PASS",
        "critic_update": int(critic.get("update", -1)),
        "critic_warmup_updates": int(critic.get("critic_warmup_updates", -1)),
        "source_actor_ppo_updates": int(critic.get("actor_ppo_updates", -1)),
        "reward_horizons": sorted(trained_horizons),
        "reward_confidence_floor": args.reward_confidence_floor,
        "env_id": args.env_id,
        "eval_protocol": eval_protocol,
        "metric_step_axis": "accepted actor_ppo_update",
        "resume": args.resume,
        **resume_summary,
    }
    print("=== Qwen + V-JEPA formal PPO preflight: PASS ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
