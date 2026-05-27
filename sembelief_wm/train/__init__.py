"""World model training components for SemBelief-WM."""

from __future__ import annotations

from .curriculum import CurriculumManager, CurriculumState
from .losses import (
    Phase1Losses,
    binary_reward_targets,
    compose_phase1_losses,
    compute_phase1_loss,
    compute_pos_weight,
    covariance_loss,
    dynamics_loss,
    dynamics_loss_sum,
    local_time_weights,
    reward_bce_loss,
    reward_bce_loss_sum,
    valid_time_mask,
)
from .probing import (
    BeliefProbe,
    ProbeMetrics,
    ProbeSpec,
    build_sokoban_probes,
    extract_sokoban_labels,
)
from .sigreg import (
    RunningBuffer,
    SIGRegLoss,
    SIGRegTerms,
    epps_pulley_loss,
    flatten_posterior_beliefs,
    vicreg_variance_loss,
)
from .targets import (
    dynamics_mask_from_window,
    posterior_valid_mask_from_window,
    reward_targets_from_window,
    reward_valid_mask_from_window,
)
from .windowing import WindowCursor, num_windows, slice_training_window

__all__ = [
    "BeliefProbe",
    "CurriculumManager",
    "CurriculumState",
    "DataSource",
    "Logger",
    "Phase1Losses",
    "Phase1Trainer",
    "ProbeMetrics",
    "ProbeSpec",
    "RunningBuffer",
    "SIGRegLoss",
    "SIGRegTerms",
    "WindowCursor",
    "binary_reward_targets",
    "build_sokoban_probes",
    "compose_phase1_losses",
    "compute_phase1_loss",
    "compute_pos_weight",
    "covariance_loss",
    "dynamics_loss",
    "dynamics_loss_sum",
    "dynamics_mask_from_window",
    "epps_pulley_loss",
    "extract_sokoban_labels",
    "flatten_posterior_beliefs",
    "local_time_weights",
    "num_windows",
    "posterior_valid_mask_from_window",
    "reward_bce_loss",
    "reward_bce_loss_sum",
    "reward_targets_from_window",
    "reward_valid_mask_from_window",
    "slice_training_window",
    "valid_time_mask",
    "vicreg_variance_loss",
    "DataSource",
    "Logger",
    "Phase1Trainer",
    "TrainStepMetrics",
]


def __getattr__(name: str):
    if name in {"DataSource", "Logger", "Phase1Trainer", "TrainStepMetrics"}:
        from .trainer import DataSource, Logger, Phase1Trainer, TrainStepMetrics

        exports = {
            "DataSource": DataSource,
            "Logger": Logger,
            "Phase1Trainer": Phase1Trainer,
            "TrainStepMetrics": TrainStepMetrics,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
