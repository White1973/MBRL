#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[3]
VAGEN_ROOT = CODE_ROOT / "VAGEN"
if str(VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(VAGEN_ROOT))

if TYPE_CHECKING:
    from vagen.env.sokoban.env import SokobanEnv
    from vagen.env.sokoban.env_config import SokobanEnvConfig
    from vagen.inference.model_interface.vllm.model import VLLMModelInterface
    from vagen.inference.model_interface.vllm.model_config import VLLMModelConfig

try:
    import wandb
except Exception:  # noqa: BLE001
    wandb = None


LOGGER = logging.getLogger("vagen_zeroshot_eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sokoban zero-shot eval using the local VAGEN env and vLLM model interface."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--dim-room", type=int, nargs=2, default=(6, 6))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-boxes", type=int, default=1)
    parser.add_argument("--min-actions-to-succeed", type=int, default=5)
    parser.add_argument("--max-actions-per-step", type=int, default=3)
    parser.add_argument("--prompt-format", type=str, default="grounding_worldmodeling")
    parser.add_argument("--render-mode", type=str, default="vision", choices=["vision", "text"])
    parser.add_argument("--action-sep", type=str, default=",")
    parser.add_argument("--format-reward", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-name", type=str, default="eval_traces.jsonl")
    parser.add_argument("--summary-name", type=str, default="summary.json")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_env_config(args: argparse.Namespace) -> "SokobanEnvConfig":
    from vagen.env.sokoban.env_config import SokobanEnvConfig

    return SokobanEnvConfig(
        dim_room=tuple(args.dim_room),
        max_steps=args.max_steps,
        num_boxes=args.num_boxes,
        render_mode=args.render_mode,
        min_actions_to_succeed=args.min_actions_to_succeed,
        max_actions_per_step=args.max_actions_per_step,
        prompt_format=args.prompt_format,
        action_sep=args.action_sep,
        format_reward=args.format_reward,
    )


def _build_model_config(args: argparse.Namespace) -> "VLLMModelConfig":
    from vagen.inference.model_interface.vllm.model_config import VLLMModelConfig

    return VLLMModelConfig(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )


def _make_user_message(obs: Dict[str, Any]) -> Dict[str, Any]:
    message = {"role": "user", "content": obs["obs_str"]}
    if "multi_modal_data" in obs:
        message["multi_modal_data"] = obs["multi_modal_data"]
    return message


def _make_initial_messages(env: "SokobanEnv", obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": env.system_prompt()},
        _make_user_message(obs),
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_trace(trace_path: Path, row: Dict[str, Any]) -> None:
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _init_wandb(args: argparse.Namespace, config_payload: Dict[str, Any]) -> Any:
    if not args.wandb_project or wandb is None:
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=config_payload,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _chunked(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _seed_everything(args.seed)

    output_dir = args.output_dir.resolve()
    trace_path = output_dir / args.trace_name
    summary_path = output_dir / args.summary_name
    _ensure_output_dir(output_dir)
    if trace_path.exists():
        trace_path.unlink()

    env_cfg = _build_env_config(args)
    model_cfg = _build_model_config(args)
    config_payload = {
        "episodes": args.episodes,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "env_config": asdict(env_cfg),
        "model_config": asdict(model_cfg),
        "output_dir": str(output_dir),
        "vagen_root": str(VAGEN_ROOT),
    }

    run = _init_wandb(args, config_payload)

    from vagen.env.sokoban.env import SokobanEnv
    from vagen.inference.model_interface.vllm.model import VLLMModelInterface

    LOGGER.info("Loading vLLM model: %s", args.model)
    model = VLLMModelInterface(model_cfg)

    LOGGER.info("Preparing %d Sokoban environments", args.episodes)
    seed_pool = env_cfg.generate_seeds(size=args.episodes, seed=args.seed)
    env_specs = [{"episode_id": idx, "seed": seed} for idx, seed in enumerate(seed_pool)]

    episode_rows: List[Dict[str, Any]] = []
    trace_count = 0
    started_at = time.time()

    for batch in _chunked(env_specs, max(1, args.batch_size)):
        envs: List[SokobanEnv] = []
        states: List[Dict[str, Any]] = []
        try:
            for spec in batch:
                env = SokobanEnv(env_cfg)
                obs, reset_info = env.reset(seed=spec["seed"])
                envs.append(env)
                states.append(
                    {
                        "episode_id": spec["episode_id"],
                        "seed": spec["seed"],
                        "env": env,
                        "messages": _make_initial_messages(env, obs),
                        "done": False,
                        "return": 0.0,
                        "steps": 0,
                        "parse_successes": 0,
                        "format_correct_count": 0,
                        "valid_action_count": 0,
                        "effective_action_count": 0,
                        "final_info": reset_info,
                    }
                )

            while True:
                active_states = [state for state in states if not state["done"]]
                if not active_states:
                    break

                prompts = [state["messages"] for state in active_states]
                responses = model.generate(
                    prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=args.seed,
                )

                for state, response in zip(active_states, responses):
                    raw_text = response["text"]
                    obs, reward, done, info = state["env"].step(raw_text)
                    state["return"] += float(reward)
                    state["steps"] += 1
                    state["done"] = bool(done)
                    state["final_info"] = info

                    format_correct = bool(info.get("format_correct", False))
                    actions = list(info.get("actions", []))
                    valid = bool(info.get("metrics", {}).get("turn_metrics", {}).get("action_is_valid", False))
                    effective = bool(info.get("metrics", {}).get("turn_metrics", {}).get("action_is_effective", False))

                    state["parse_successes"] += int(format_correct and len(actions) > 0)
                    state["format_correct_count"] += int(format_correct)
                    state["valid_action_count"] += int(valid)
                    state["effective_action_count"] += int(effective)

                    trace_row = {
                        "episode_id": state["episode_id"],
                        "seed": state["seed"],
                        "turn_index": state["steps"] - 1,
                        "raw_text": raw_text,
                        "llm_response": info.get("llm_response"),
                        "actions": actions,
                        "action_content": info.get("action_content"),
                        "format_correct": format_correct,
                        "action_is_valid": valid,
                        "action_is_effective": effective,
                        "reward": reward,
                        "done": done,
                        "finish_reason": response.get("finish_reason"),
                        "usage": response.get("usage", {}),
                        "obs_str": obs["obs_str"],
                    }
                    _append_trace(trace_path, trace_row)
                    trace_count += 1

                    state["messages"].append({"role": "assistant", "content": raw_text})
                    if not done:
                        state["messages"].append(_make_user_message(obs))

            for state in states:
                final_metrics = state["final_info"].get("metrics", {})
                success = bool(final_metrics.get("traj_metrics", {}).get("success", False))
                steps = max(1, state["steps"])
                episode_rows.append(
                    {
                        "episode_id": state["episode_id"],
                        "seed": state["seed"],
                        "success": success,
                        "return": state["return"],
                        "steps": state["steps"],
                        "parse_success_rate": state["parse_successes"] / steps,
                        "format_correct_rate": state["format_correct_count"] / steps,
                        "valid_action_rate": state["valid_action_count"] / steps,
                        "effective_action_rate": state["effective_action_count"] / steps,
                        "final_info": _jsonable(state["final_info"]),
                    }
                )
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Failed to close env cleanly")

    episode_rows.sort(key=lambda row: row["episode_id"])
    elapsed = time.time() - started_at
    episode_count = max(1, len(episode_rows))
    total_steps = sum(row["steps"] for row in episode_rows)
    successes = sum(int(row["success"]) for row in episode_rows)
    summary = {
        "episodes": len(episode_rows),
        "success_rate": successes / episode_count,
        "avg_return": sum(row["return"] for row in episode_rows) / episode_count,
        "avg_episode_length": total_steps / episode_count,
        "avg_parse_success_rate": sum(row["parse_success_rate"] for row in episode_rows) / episode_count,
        "avg_format_correct_rate": sum(row["format_correct_rate"] for row in episode_rows) / episode_count,
        "avg_valid_action_rate": sum(row["valid_action_rate"] for row in episode_rows) / episode_count,
        "avg_effective_action_rate": sum(row["effective_action_rate"] for row in episode_rows) / episode_count,
        "trace_count": trace_count,
        "elapsed_sec": elapsed,
        "config": config_payload,
        "per_episode": episode_rows,
    }
    _write_json(summary_path, summary)

    LOGGER.info(
        "Finished eval: success_rate=%.4f avg_return=%.4f avg_parse_success_rate=%.4f",
        summary["success_rate"],
        summary["avg_return"],
        summary["avg_parse_success_rate"],
    )

    if run is not None:
        run.log(
            {
                "eval/success_rate": summary["success_rate"],
                "eval/avg_return": summary["avg_return"],
                "eval/avg_episode_length": summary["avg_episode_length"],
                "eval/avg_parse_success_rate": summary["avg_parse_success_rate"],
                "eval/avg_format_correct_rate": summary["avg_format_correct_rate"],
                "eval/avg_valid_action_rate": summary["avg_valid_action_rate"],
                "eval/avg_effective_action_rate": summary["avg_effective_action_rate"],
                "eval/elapsed_sec": summary["elapsed_sec"],
            }
        )
        run.summary.update(
            {
                "trace_path": str(trace_path),
                "summary_path": str(summary_path),
            }
        )
        run.finish()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
