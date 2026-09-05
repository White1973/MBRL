"""Typed, fail-closed configuration for Le-WM orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os


class LeWMStage(str, Enum):
    REAL_CRITIC_PROBE = "real_critic_probe"
    REAL_CRITIC_VALIDATE = "real_critic_validate"
    ACTOR_PPO = "actor_ppo"


@dataclass(frozen=True)
class LeWMOrchestrationConfig:
    stage: LeWMStage
    collect_episodes: int
    replay_capacity: int
    validation_capacity: int
    train_samples: int
    critic_updates_per_collection: int
    reward_scale: float
    ev_threshold: float
    mse_improvement_threshold: float
    gate_patience: int
    ev_ema_alpha: float

    @classmethod
    def from_environment(cls) -> "LeWMOrchestrationConfig":
        stage_value = os.environ.get(
            "LEWM_STAGE", LeWMStage.REAL_CRITIC_PROBE.value
        )
        try:
            stage = LeWMStage(stage_value)
        except ValueError as error:
            raise ValueError(f"unsupported LEWM_STAGE={stage_value!r}") from error
        result = cls(
            stage=stage,
            collect_episodes=int(os.environ.get("COLLECT_EPISODES", "32")),
            replay_capacity=int(os.environ.get(
                "LEWM_REAL_CRITIC_REPLAY_CAPACITY", "8192"
            )),
            validation_capacity=int(os.environ.get(
                "LEWM_REAL_CRITIC_VALIDATION_SIZE", "2048"
            )),
            train_samples=int(os.environ.get(
                "LEWM_REAL_CRITIC_TRAIN_SAMPLES", "1024"
            )),
            critic_updates_per_collection=int(os.environ.get(
                "LEWM_REAL_CRITIC_UPDATES", "10"
            )),
            reward_scale=float(os.environ.get(
                "LEWM_REAL_RETURN_REWARD_SCALE", "0.1"
            )),
            ev_threshold=float(os.environ.get(
                "LEWM_REAL_CRITIC_EV_THRESHOLD", "0.10"
            )),
            mse_improvement_threshold=float(os.environ.get(
                "LEWM_REAL_CRITIC_MSE_IMPROVEMENT", "0.05"
            )),
            gate_patience=int(os.environ.get(
                "LEWM_REAL_CRITIC_GATE_PATIENCE", "3"
            )),
            ev_ema_alpha=float(os.environ.get(
                "LEWM_REAL_CRITIC_EV_EMA_ALPHA", "0.20"
            )),
        )
        result.validate()
        return result

    def validate(self) -> None:
        positive_ints = {
            "collect_episodes": self.collect_episodes,
            "replay_capacity": self.replay_capacity,
            "validation_capacity": self.validation_capacity,
            "train_samples": self.train_samples,
            "critic_updates_per_collection": self.critic_updates_per_collection,
            "gate_patience": self.gate_patience,
        }
        invalid = {name: value for name, value in positive_ints.items() if value <= 0}
        if invalid:
            raise ValueError(f"Le-WM positive integer contract failed: {invalid}")
        if not 0.0 < self.ev_ema_alpha <= 1.0:
            raise ValueError("ev_ema_alpha must be in (0,1]")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
