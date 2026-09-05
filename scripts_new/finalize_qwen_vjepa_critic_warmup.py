#!/usr/bin/env python3
"""Validate a Critic-only checkpoint and publish best.pt plus JSON report."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-wm", required=True)
    parser.add_argument("--min-updates", type=int, default=20)
    parser.add_argument("--min-ev-ema", type=float, default=0.10)
    parser.add_argument("--min-mse-improvement", type=float, default=0.05)
    parser.add_argument("--min-top1", type=float, default=0.60)
    parser.add_argument("--min-pairwise", type=float, default=0.60)
    parser.add_argument("--min-q-margin", type=float, default=0.001)
    parser.add_argument("--reward-confidence-floor", type=float, required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    latest = output / "latest.pt"
    best = output / "best.pt"
    report_path = output / "warmup_report.json"
    if not latest.is_file():
        raise FileNotFoundError(f"Critic warmup latest.pt is missing: {latest}")
    if best.exists():
        raise FileExistsError(f"refusing to overwrite Critic best.pt: {best}")
    payload = torch.load(latest, map_location="meta", weights_only=False, mmap=True)
    metrics = dict(payload.get("critic_warmup_release_metrics", {}))
    cache = dict(payload.get("critic_h2_cache_metadata", {}))
    failures: list[str] = []
    expected_source = Path(args.source_wm).resolve()

    def require_metric(name: str, minimum: float) -> None:
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) < minimum:
            failures.append(f"{name}={value} < {minimum}")

    if not bool(payload.get("critic_warmup_complete")):
        failures.append("critic_warmup_complete is false")
    if payload.get("training_stage") != "critic_pretrain":
        failures.append(f"training_stage={payload.get('training_stage')!r}")
    if int(payload.get("actor_ppo_updates", -1)) != 0:
        failures.append(
            f"Actor changed during warmup: actor_ppo_updates={payload.get('actor_ppo_updates')}"
        )
    if int(payload.get("critic_warmup_updates", 0)) < args.min_updates:
        failures.append(
            f"critic_warmup_updates={payload.get('critic_warmup_updates')} "
            f"< {args.min_updates}"
        )
    if cache.get("official_validation_used_for_training") is not False:
        failures.append("official H2 validation was not isolated from training")
    cached_source = cache.get("source_checkpoint")
    if not cached_source or Path(cached_source).resolve() != expected_source:
        failures.append(
            "H2 cache/source mismatch: "
            f"cache={cached_source!r}, expected={str(expected_source)!r}"
        )
    saved_floor = payload.get("critic_reward_confidence_floor")
    if (
        saved_floor is None
        or not math.isfinite(float(saved_floor))
        or not math.isclose(
            float(saved_floor), args.reward_confidence_floor,
            rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        failures.append(
            "reward confidence floor mismatch: "
            f"checkpoint={saved_floor}, expected={args.reward_confidence_floor}"
        )
    target_semantics = payload.get("critic_target_semantics")
    if target_semantics != "learned_wm_reward_h2_counterfactual":
        failures.append(f"unexpected Critic target semantics: {target_semantics!r}")
    require_metric("critic_warmup/heldout_ev_ema", args.min_ev_ema)
    require_metric("critic_warmup/mse_improvement", args.min_mse_improvement)
    require_metric("critic_warmup/heldout_top1_accuracy", args.min_top1)
    require_metric("critic_warmup/heldout_pairwise_accuracy", args.min_pairwise)
    require_metric("critic_warmup/heldout_q_margin", args.min_q_margin)
    require_metric("critic_warmup/ranking_gate_passed", 1.0)
    require_metric("critic_warmup/level_disjoint_bucket_gate_passed", 1.0)
    for bucket in ("initial_h1", "initial_h2", "suffix_h1", "suffix_h2"):
        require_metric(f"critic_warmup/bucket_{bucket}/passed", 1.0)

    report = {
        "format": "qwen_vjepa_critic_warmup_gate_v1",
        "source_reward_head_checkpoint": str(Path(args.source_wm).resolve()),
        "latest_checkpoint": str(latest),
        "best_checkpoint": str(best) if not failures else None,
        "actor_frozen": int(payload.get("actor_ppo_updates", -1)) == 0,
        "critic_warmup_updates": int(payload.get("critic_warmup_updates", 0)),
        "critic_h2_cache": cache,
        "reward_confidence_floor": saved_floor,
        "target_semantics": target_semantics,
        "metrics": metrics,
        "gate": {
            "passed": not failures,
            "failures": failures,
            "thresholds": {
                "minimum_updates": args.min_updates,
                "minimum_heldout_ev_ema": args.min_ev_ema,
                "minimum_mse_improvement": args.min_mse_improvement,
                "minimum_top1_accuracy": args.min_top1,
                "minimum_pairwise_accuracy": args.min_pairwise,
                "minimum_q_margin": args.min_q_margin,
                "require_all_four_level_disjoint_buckets": True,
            },
        },
    }
    _atomic_json(report, report_path)
    if failures:
        raise RuntimeError(
            "Critic warmup release Gate failed: " + "; ".join(failures)
        )
    os.link(latest, best)
    print("Critic warmup release Gate: PASS", flush=True)
    print(f"Critic latest: {latest}", flush=True)
    print(f"Critic best:   {best}", flush=True)
    print(f"Report:        {report_path}", flush=True)


if __name__ == "__main__":
    main()
