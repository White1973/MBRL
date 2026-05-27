"""Tests for WorldModelRefresher."""

import pytest
from dataclasses import dataclass
from typing import Any

import torch

from sembelief_wm.trainers.wm_refresher import WorldModelRefresher, WMRefreshConfig


# --- Fakes ---


class FakeWorldModel:
    """Minimal world model stub for testing."""

    def requires_grad_(self, val: bool) -> None:
        self._grad = val

    def eval(self) -> None:
        self._eval = True

    def train(self) -> None:
        self._eval = False


@dataclass
class FakeStepMetrics:
    dynamics_loss: float = 0.5
    reward_loss: float = 0.1
    total_loss: float = 0.6

    def as_dict(self) -> dict[str, float]:
        # Match real Phase1Trainer key format: "loss/dynamics", "loss/reward", "loss/total"
        return {
            "loss/dynamics": self.dynamics_loss,
            "loss/reward": self.reward_loss,
            "loss/total": self.total_loss,
        }


class FakePhase1Trainer:
    """Minimal Phase1Trainer stub."""

    def __init__(self) -> None:
        self.world_model = FakeWorldModel()
        self.optimizer = FakeOptimizer()
        self.scheduler = FakeScheduler()
        self.step_count = 0

    def train_one_step(self, *, global_step: int, batch: Any) -> FakeStepMetrics:
        self.step_count += 1
        return FakeStepMetrics()


class FakeOptimizer:
    def state_dict(self) -> dict:
        return {"fake": True}

    def load_state_dict(self, d: dict) -> None:
        pass


class FakeScheduler:
    def state_dict(self) -> dict:
        return {"fake": True}

    def load_state_dict(self, d: dict) -> None:
        pass


# --- Tests ---


def test_should_refresh():
    config = WMRefreshConfig(refresh_every=5)
    refresher = WorldModelRefresher(FakePhase1Trainer(), config)

    assert not refresher.should_refresh(0)  # never at step 0
    assert not refresher.should_refresh(3)
    assert refresher.should_refresh(5)
    assert refresher.should_refresh(10)
    assert not refresher.should_refresh(7)


def test_should_refresh_disabled():
    config = WMRefreshConfig(refresh_every=0)
    refresher = WorldModelRefresher(FakePhase1Trainer(), config)
    assert not refresher.should_refresh(5)
    assert not refresher.should_refresh(10)


def test_refresh_runs_correct_steps():
    trainer = FakePhase1Trainer()
    config = WMRefreshConfig(updates_per_refresh=3, batch_size=4)
    refresher = WorldModelRefresher(trainer, config)

    fake_batch = "dummy_batch"
    sample_fn = lambda bs: fake_batch

    metrics = refresher.refresh(sample_fn)

    assert trainer.step_count == 3
    assert metrics.num_steps == 3
    assert refresher.refresh_step == 3
    assert abs(metrics.avg_dynamics_loss - 0.5) < 1e-6
    assert abs(metrics.avg_reward_loss - 0.1) < 1e-6


def test_refresh_restores_eval_mode():
    trainer = FakePhase1Trainer()
    config = WMRefreshConfig(updates_per_refresh=2)
    refresher = WorldModelRefresher(trainer, config)

    refresher.refresh(lambda bs: "batch")

    # After refresh, WM should be back to eval mode with no grads
    assert not trainer.world_model._grad
    assert trainer.world_model._eval


def test_state_dict_roundtrip():
    trainer = FakePhase1Trainer()
    config = WMRefreshConfig(updates_per_refresh=2)
    refresher = WorldModelRefresher(trainer, config)

    refresher.refresh(lambda bs: "batch")
    state = refresher.state_dict()
    assert state["refresh_step"] == 2

    # New refresher, load state
    trainer2 = FakePhase1Trainer()
    refresher2 = WorldModelRefresher(trainer2, config)
    refresher2.load_state_dict(state)
    assert refresher2.refresh_step == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
