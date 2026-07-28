"""
mT5-FT baseline (Table 3): "Standard fine-tuning without fairness
intervention."

This is not new code -- it's simply model/mt5.py's
build_mt5_sentiment_model / build_mt5_ner_model, trained with
optimization/trainer.py's task loss ALONE (no BTS term, no fairness
controller). The cleanest way to run this baseline is to use
ADAPTBTSTrainer with lambda permanently pinned at 0 rather than writing
a separate training loop, since that guarantees identical
optimizer/scheduler/AMP/logging behavior to the full ADAPT-BTS runs --
exactly the "same dataset, same preprocessing, same evaluation" fairness
of comparison Sec 5.2 calls for.
"""

from __future__ import annotations

from fairness.fairness_controller import ControllerConfig
from optimization.trainer import TrainerConfig


def mt5_ft_controller_config() -> ControllerConfig:
    """Pins lambda at 0 throughout training and disables the controller
    update (eta_lambda=0), so the composite loss (Eq. 12) collapses to
    L_task alone -- i.e. no fairness intervention."""
    return ControllerConfig(
        tau=0.0, eta_lambda=0.0, lambda_init=0.0, lambda_min=0.0, lambda_max=0.0,
    )


def mt5_ft_trainer_config(**kwargs) -> TrainerConfig:
    """Convenience TrainerConfig for the mT5-FT baseline: same defaults
    as ADAPT-BTS's TrainerConfig, but with the controller pinned at 0
    (see mt5_ft_controller_config) and IBADR refresh effectively disabled
    (selection_ratio=0.0, so Algorithm 3 selects zero samples each
    interval)."""
    defaults = dict(
        tau=0.0, eta_lambda=0.0, lambda_init=0.0, lambda_min=0.0, lambda_max=0.0,
        ibadr_selection_ratio=0.0,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)
