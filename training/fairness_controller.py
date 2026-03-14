"""
fairness_controller.py
======================
Fairness-Aware Proportional Control (FAPC) module.

Implements the feedback-controlled optimization mechanism described in
Sections 4.2 and 4.4 of the ADAPT-BTS paper.

The controller regulates the fairness regularization weight λ_t based on
the deviation between the observed BTS and the desired fairness threshold τ:

    λ_{t+1} = λ_t + η_λ · (BTS_t − τ)    [Equation 20 / 27]

followed by projection to [λ_min, λ_max]:

    λ_{t+1} = clip(λ_{t+1}, λ_min, λ_max)   [Equation 28]

This resembles a proportional (P) controller in control theory, where:
  - BTS_t − τ  is the error signal
  - η_λ        is the proportional gain
  - λ_t        is the control output (fairness weight)

When BTS > τ: λ increases → stronger fairness enforcement
When BTS < τ: λ decreases → relax constraint, focus on predictive accuracy

Stability Condition (Section 4.6):
    0 < η_λ < 2 / L_BTS
where L_BTS is the Lipschitz constant of the BTS function.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """Snapshot of controller state at a given training step."""
    step: int
    epoch: int
    lambda_t: float
    bts_t: float
    error: float            # BTS_t - τ
    update: float           # η_λ * error


class FairnessProportionalController:
    """
    Proportional feedback controller for λ_t.

    The controller is stateful and tracks the full trajectory of λ_t
    and BTS_t across training for analysis and debugging.

    Parameters
    ----------
    tau : float
        Desired fairness tolerance τ (paper: 0.40).
    lambda_init : float
        Initial regularization weight λ_0 (paper: 0.1).
    lambda_min : float
        Lower bound for λ after projection (paper: 0.0).
    lambda_max : float
        Upper bound for λ after projection (paper: 10.0).
    eta_lambda : float
        Proportional gain η_λ (paper: 0.01).
    """

    def __init__(
        self,
        tau: float = 0.40,
        lambda_init: float = 0.1,
        lambda_min: float = 0.0,
        lambda_max: float = 10.0,
        eta_lambda: float = 0.01,
    ):
        self.tau = tau
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.eta_lambda = eta_lambda

        # Current state
        self._lambda = float(lambda_init)
        self._step = 0
        self._epoch = 0

        # History for analysis
        self.history: List[ControllerState] = []

        logger.info(
            f"[FAPC] Initialized: τ={tau}, λ_0={lambda_init}, "
            f"λ∈[{lambda_min}, {lambda_max}], η_λ={eta_lambda}"
        )

    # ------------------------------------------------------------------
    # Core Update (Equations 27–28)
    # ------------------------------------------------------------------

    def step(self, bts_batch: float, epoch: Optional[int] = None) -> float:
        """
        Perform one proportional control update.

        λ_{t+1} = clip(λ_t + η_λ · (BTS_t − τ), λ_min, λ_max)

        Parameters
        ----------
        bts_batch : float
            BTS measured on the current mini-batch.
        epoch : int, optional
            Current epoch (for logging).

        Returns
        -------
        lambda_new : float
            Updated λ_{t+1} to use for the next training step.
        """
        if epoch is not None:
            self._epoch = epoch

        error = bts_batch - self.tau
        update = self.eta_lambda * error
        lambda_new = self._lambda + update

        # Project to feasible range (Equation 28)
        lambda_new = float(np.clip(lambda_new, self.lambda_min, self.lambda_max))

        # Record state
        self.history.append(ControllerState(
            step=self._step,
            epoch=self._epoch,
            lambda_t=self._lambda,
            bts_t=bts_batch,
            error=error,
            update=update,
        ))

        self._lambda = lambda_new
        self._step += 1

        return lambda_new

    def batch_update(self, bts_batch: float) -> float:
        """
        Alias for step(), used during training batch loops.
        Implements Equation 40 (clip version):
            λ_t = clip(λ_{t-1} + η(BTS_batch − τ), λ_min, λ_max)
        """
        return self.step(bts_batch)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lambda_current(self) -> float:
        """Current regularization weight λ_t."""
        return self._lambda

    @property
    def is_fairness_satisfied(self) -> bool:
        """
        Returns True if the most recent BTS ≤ τ (fairness constraint met).
        """
        if not self.history:
            return False
        return self.history[-1].bts_t <= self.tau

    @property
    def convergence_trend(self) -> float:
        """
        Approximate convergence speed: mean |BTS_t - τ| over last 10 steps.
        Smaller value = closer to desired fairness level.
        """
        if len(self.history) < 2:
            return float("inf")
        recent = self.history[-min(10, len(self.history)):]
        return float(np.mean([abs(s.error) for s in recent]))

    # ------------------------------------------------------------------
    # Stability Analysis
    # ------------------------------------------------------------------

    def check_stability_condition(self, lipschitz_bts: float = 1.0) -> bool:
        """
        Verify the stability condition from Section 4.6:
            0 < η_λ < 2 / L_BTS

        Parameters
        ----------
        lipschitz_bts : float
            Estimated Lipschitz constant of BTS (default: 1.0, conservative bound
            since BTS ∈ [0,1] implies L_BTS ≤ 1).

        Returns
        -------
        stable : bool
        """
        upper_bound = 2.0 / max(lipschitz_bts, 1e-9)
        stable = 0 < self.eta_lambda < upper_bound
        if not stable:
            logger.warning(
                f"[FAPC] Stability condition violated: η_λ={self.eta_lambda:.4f} "
                f"must be in (0, {upper_bound:.4f})"
            )
        return stable

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "lambda": self._lambda,
            "step": self._step,
            "epoch": self._epoch,
            "tau": self.tau,
            "eta_lambda": self.eta_lambda,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "history_len": len(self.history),
        }

    def load_state_dict(self, state: dict):
        """Restore controller state from a checkpoint."""
        self._lambda = state["lambda"]
        self._step = state["step"]
        self._epoch = state["epoch"]
        self.tau = state["tau"]
        self.eta_lambda = state["eta_lambda"]
        self.lambda_min = state["lambda_min"]
        self.lambda_max = state["lambda_max"]

    def reset(self, lambda_init: Optional[float] = None):
        """Reset controller to initial state (e.g., for a new seed run)."""
        self._lambda = lambda_init if lambda_init is not None else 0.1
        self._step = 0
        self._epoch = 0
        self.history.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of controller state."""
        lines = [
            "═══ FAPC Controller Summary ═══",
            f"  Current λ:       {self._lambda:.4f}",
            f"  Desired τ:       {self.tau:.4f}",
            f"  Gain η_λ:        {self.eta_lambda:.4f}",
            f"  λ range:         [{self.lambda_min}, {self.lambda_max}]",
            f"  Steps taken:     {self._step}",
            f"  Fairness met:    {self.is_fairness_satisfied}",
            f"  Convergence:     {self.convergence_trend:.4f} (mean |error|)",
        ]
        if self.history:
            last = self.history[-1]
            lines.append(f"  Last BTS:        {last.bts_t:.4f}")
            lines.append(f"  Last error:      {last.error:+.4f}")
        return "\n".join(lines)

    def get_lambda_trajectory(self) -> List[float]:
        """Return the full λ trajectory for plotting."""
        return [s.lambda_t for s in self.history]

    def get_bts_trajectory(self) -> List[float]:
        """Return the full BTS trajectory for plotting."""
        return [s.bts_t for s in self.history]
