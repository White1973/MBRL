"""Evaluate belief probing on a trained Phase 1 world model checkpoint.

Usage:
    python scripts/eval_probing.py \
        --checkpoint checkpoints/step_01000.pt \
        --data-dir data/sokoban_10k/tokenized \
        --num-episodes 200 \
        --device cuda:0

This script:
1. Loads the world model from a checkpoint
2. Runs posterior rollout on tokenized episodes to extract beliefs
3. Loads structured state labels from raw episodes
4. Trains linear probes on frozen beliefs
5. Reports probe accuracy vs baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sembelief_wm.config import Config
from sembelief_wm.train import (
    BeliefProbe,
    ProbeMetrics,
    build_sokoban_probes,
    extract_sokoban_labels,
    num_windows,
    slice_training_window,
)
from sembelief_wm.model import WorldModel
from sembelief_wm.types import BeliefState, SequenceBatch


def load_checkpoint(path: str, device: str) -> tuple[WorldModel, Config]:
    """Load world model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Build world model with mock backbone for probing (no need for real Qwen)
    from sembelief_wm.model import TransitionBackbone

    class IdentityBackbone(TransitionBackbone):
        def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
            return x

    wm = WorldModel(config, IdentityBackbone())
    wm.load_state_dict(checkpoint["model"], strict=False)
    wm.to(device)
    wm.eval()
    return wm, config


def extract_beliefs(
    wm: WorldModel,
    episodes: list[dict],
    config: Config,
    device: str,
) -> tuple[Tensor, list[dict]]:
    """Run posterior rollout and extract beliefs + state labels.

    Returns:
        beliefs: (N_total, K, D) — all posterior beliefs across all timesteps
        all_state_labels: list of state_label dicts, one per timestep
    """
    all_beliefs = []
    all_state_labels = []

    with torch.no_grad():
        for ep in episodes:
            obs_tokens = ep["obs_tokens"].to(device)  # (T, K, D)
            actions = ep["actions"].to(device)          # (T,)
            T = obs_tokens.shape[0]

            # Run posterior rollout
            belief = BeliefState(
                slots=torch.zeros(1, config.belief.num_slots, config.hidden_dim, device=device)
            )
            null_action = torch.tensor([config.env.null_action_id], device=device)
            prev_actions = null_action

            for t in range(T):
                obs_t = obs_tokens[t:t+1]  # (1, K, D)
                belief = wm.posterior_step(
                    prev_belief=belief,
                    prev_actions=prev_actions,
                    observation_tokens=obs_t,
                )
                all_beliefs.append(belief.slots.squeeze(0).cpu())  # (K, D)

                if t < T - 1:
                    prev_actions = actions[t:t+1]

            # Collect state labels
            if "metadata" in ep and "state_labels" in ep["metadata"]:
                labels = ep["metadata"]["state_labels"]
                # Align: state_labels has one entry per step
                for sl in labels[:T]:
                    all_state_labels.append(sl)
            elif "state_labels" in ep:
                labels = ep["state_labels"]
                for sl in labels[:T]:
                    all_state_labels.append(sl)

    beliefs = torch.stack(all_beliefs, dim=0)  # (N, K, D)
    return beliefs, all_state_labels


def load_episodes(data_dir: str, num_episodes: int) -> list[dict]:
    """Load tokenized episodes with metadata."""
    data_path = Path(data_dir)

    # Try manifest first
    manifest = data_path / "manifest.jsonl"
    if manifest.exists():
        entries = []
        with open(manifest) as f:
            for line in f:
                entries.append(json.loads(line))
        entries = entries[:num_episodes]
        episodes = []
        for entry in entries:
            pt_path = data_path / entry["path"]
            if pt_path.exists():
                ep = torch.load(pt_path, weights_only=False)
                episodes.append(ep)
        return episodes

    # Fallback: load .pt files directly
    pt_files = sorted(data_path.glob("*.pt"))[:num_episodes]
    return [torch.load(f, weights_only=False) for f in pt_files]


def load_raw_episodes(raw_dir: str, num_episodes: int) -> list[dict]:
    """Load raw episodes to get state labels."""
    raw_path = Path(raw_dir)
    manifest = raw_path / "manifest.jsonl"
    if manifest.exists():
        entries = []
        with open(manifest) as f:
            for line in f:
                entries.append(json.loads(line))
        entries = entries[:num_episodes]
        episodes = []
        for entry in entries:
            pt_path = raw_path / entry["path"]
            if pt_path.exists():
                ep = torch.load(pt_path, weights_only=False)
                episodes.append(ep)
        return episodes
    pt_files = sorted(raw_path.glob("*.pt"))[:num_episodes]
    return [torch.load(f, weights_only=False) for f in pt_files]


def main():
    parser = argparse.ArgumentParser(description="Belief probing evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", required=True, help="Path to tokenized data dir")
    parser.add_argument("--raw-dir", default=None, help="Path to raw data dir (for state labels)")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--grid-size", type=int, default=6)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    wm, config = load_checkpoint(args.checkpoint, args.device)

    print(f"Loading {args.num_episodes} tokenized episodes from {args.data_dir}")
    episodes = load_episodes(args.data_dir, args.num_episodes)

    # Get state labels from raw episodes if separate raw dir provided
    if args.raw_dir:
        print(f"Loading raw episodes for state labels from {args.raw_dir}")
        raw_episodes = load_raw_episodes(args.raw_dir, args.num_episodes)
        # Merge state_labels into tokenized episodes
        for tok_ep, raw_ep in zip(episodes, raw_episodes):
            if "metadata" in raw_ep and "state_labels" in raw_ep["metadata"]:
                if "metadata" not in tok_ep:
                    tok_ep["metadata"] = {}
                tok_ep["metadata"]["state_labels"] = raw_ep["metadata"]["state_labels"]

    print("Extracting posterior beliefs...")
    beliefs, state_labels = extract_beliefs(wm, episodes, config, args.device)
    print(f"  Total beliefs: {beliefs.shape[0]}, shape: {beliefs.shape}")

    if not state_labels:
        print("ERROR: No state labels found. Cannot run probing.")
        print("  Make sure episodes contain metadata.state_labels")
        return

    print(f"  Total state labels: {len(state_labels)}")

    # Align beliefs and labels (take the min)
    N = min(beliefs.shape[0], len(state_labels))
    beliefs = beliefs[:N]
    state_labels = state_labels[:N]

    # Extract probe labels
    labels = extract_sokoban_labels(state_labels, grid_size=args.grid_size)
    print(f"  Label keys: {list(labels.keys())}")

    # Train/val split
    split_idx = int(N * args.train_split)
    train_beliefs = beliefs[:split_idx]
    val_beliefs = beliefs[split_idx:]
    train_labels = {k: v[:split_idx] for k, v in labels.items()}
    val_labels = {k: v[split_idx:] for k, v in labels.items()}

    print(f"\nTrain: {train_beliefs.shape[0]}, Val: {val_beliefs.shape[0]}")

    # Build and train probes
    probe = build_sokoban_probes(
        hidden_dim=config.hidden_dim,
        num_belief_slots=config.belief.num_slots,
        grid_size=args.grid_size,
        device=args.device,
    )

    print(f"\nTraining probes for {args.probe_epochs} epochs...")
    history = probe.fit(
        train_beliefs,
        train_labels,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
    )

    # Print training convergence
    for name, losses in history.items():
        print(f"  {name}: loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Evaluate on val set
    print("\n--- Probe Evaluation Results ---")
    results = probe.evaluate(val_beliefs, val_labels)

    for r in results:
        if r.accuracy > 0:
            delta = r.accuracy - r.baseline_acc
            print(
                f"  {r.name}: accuracy={r.accuracy:.4f} "
                f"(baseline={r.baseline_acc:.4f}, delta=+{delta:.4f})"
            )
        else:
            improvement = (r.baseline_mse - r.mse) / max(r.baseline_mse, 1e-8) * 100
            print(
                f"  {r.name}: MSE={r.mse:.4f} "
                f"(baseline={r.baseline_mse:.4f}, improvement={improvement:.1f}%)"
            )

    # Summary
    print("\n--- Summary ---")
    for r in results:
        if r.accuracy > 0:
            status = "PASS" if r.accuracy > r.baseline_acc + 0.05 else "MARGINAL" if r.accuracy > r.baseline_acc else "FAIL"
            print(f"  {r.name}: {status} (acc={r.accuracy:.4f})")

    # Optionally log to wandb
    try:
        import wandb
        if wandb.run is not None:
            for r in results:
                if r.accuracy > 0:
                    wandb.log({
                        f"probe/{r.name}_acc": r.accuracy,
                        f"probe/{r.name}_baseline": r.baseline_acc,
                    })
                else:
                    wandb.log({
                        f"probe/{r.name}_mse": r.mse,
                        f"probe/{r.name}_baseline_mse": r.baseline_mse,
                    })
    except ImportError:
        pass


if __name__ == "__main__":
    main()
