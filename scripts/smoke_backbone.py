"""Real-load smoke test for the Qwen transition backbone.

This script intentionally checks only the dependency boundary:
- optional VLM dependencies can import
- Qwen + LoRA can load
- the wrapper accepts `(B, S, D)` tokens and returns the same shape
- bidirectional attention lets early tokens depend on later tokens
"""
from __future__ import annotations

import argparse

import torch

from sembelief_wm import QwenTransitionBackbone
from sembelief_wm.config import BackboneConfig, Config, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to from_pretrained; use 'none' to omit device_map.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional transformers attention implementation, e.g. sdpa.",
    )
    parser.add_argument(
        "--skip-bidirectional-check",
        action="store_true",
        help="Only check load and shape, not future-token sensitivity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_map: str | None = None if args.device_map == "none" else args.device_map
    config = Config(
        backbone=BackboneConfig(
            model_name=args.model_name,
            attention_mode="bidirectional",
        ),
        training=TrainingConfig(dtype=args.dtype),
    )

    backbone = QwenTransitionBackbone.from_config(
        config,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    backbone.eval()

    tokens = torch.randn(args.batch_size, args.seq_len, config.hidden_dim)
    with torch.no_grad():
        outputs = backbone(tokens)

    expected = tuple(tokens.shape)
    actual = tuple(outputs.shape)
    if actual != expected:
        raise AssertionError(f"Unexpected backbone output shape: {actual} != {expected}.")
    if not torch.isfinite(outputs).all():
        raise AssertionError("Backbone output contains non-finite values.")

    if not args.skip_bidirectional_check and args.seq_len > 1:
        altered = tokens.clone()
        altered[:, -1] = altered[:, -1] + torch.randn_like(altered[:, -1]) * 0.25
        with torch.no_grad():
            baseline = backbone(tokens)
            future_changed = backbone(altered)
        first_token_delta = (
            future_changed[:, 0].float() - baseline[:, 0].float()
        ).abs().max().item()
        if first_token_delta <= 1e-6:
            raise AssertionError(
                "Future-token perturbation did not affect token 0 output. "
                "The Qwen wrapper may still be using causal attention."
            )
        print(f"Bidirectional sensitivity check passed: delta={first_token_delta:.6g}")

    print(f"Qwen backbone smoke passed: output_shape={actual}")


if __name__ == "__main__":
    main()
