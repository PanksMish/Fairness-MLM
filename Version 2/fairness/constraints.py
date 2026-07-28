"""
Thin re-export module: the constraints governing whether a counterfactual
candidate is usable (Eq. 3's semantic-preservation gate, Eq. 9's
acceptance gate, and the fairness controller's lambda bounds, Eq. 14) are
implemented in their natural homes (semantic_validation.py,
fairness_controller.py) rather than duplicated here. This module exists
so other code (e.g. optimization/trainer.py, evaluation scripts) can do
a single `from fairness.constraints import ...` without needing to know
which specific submodule a given constraint check lives in.
"""

from __future__ import annotations

from fairness.semantic_validation import (
    semantic_preservation_ok,   # Eq. 3
    accept_candidate,           # Eq. 9
)
from fairness.fairness_controller import ControllerConfig  # Eq. 14 (lambda_min/lambda_max)
from fairness.morphology import length_ratio_check          # structural sanity gate

__all__ = [
    "semantic_preservation_ok",
    "accept_candidate",
    "ControllerConfig",
    "length_ratio_check",
]
