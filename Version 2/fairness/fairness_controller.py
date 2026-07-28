"""
Adaptive Fairness Controller.

Implements Eq. (13)/(14) (equivalently Eq. 17, the `clip` form) and the
controller-update lines of Algorithm 2:

    lambda_{t+1} = lambda_t + eta_lambda * (BTS_t - tau)
    lambda_{t+1} = Proj_[lambda_min, lambda_max](lambda_{t+1})

This is a proportional feedback controller: when BTS exceeds the target
tau (fairness constraint violated), lambda increases, placing more weight
on the fairness term in the composite loss L = L_task + lambda * BTS
(Eq. 5/12/18). When BTS is below tau, lambda decays back down, avoiding
over-regularization.

Pure NumPy / stdlib -- no torch dependency, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControllerConfig:
    tau: float = 0.40            # fairness target / tolerance, Eq. 11
    eta_lambda: float = 0.05     # controller learning rate, Eq. 13
    lambda_init: float = 0.20
    lambda_min: float = 0.05
    lambda_max: float = 1.00


@dataclass
class ControllerState:
    lam: float
    step: int = 0
    history: list = field(default_factory=list)  # list of (step, bts, lam, error)


class FairnessController:
    """
    Stateful proportional controller. One instance should be created per
    training run (or per resource-group, matching Fig. 8 right panel,
    which tracks separate lambda trajectories for HR/MR/LR subsets).
    """

    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.state = ControllerState(lam=self.config.lambda_init)

    @property
    def lam(self) -> float:
        return self.state.lam

    def project(self, value: float) -> float:
        """Eq. (14): Proj_[lambda_min, lambda_max]."""
        return min(max(value, self.config.lambda_min), self.config.lambda_max)

    def update(self, bts_batch: float) -> float:
        """
        Single controller step, Eq. (13)+(14) / Eq. (17):

            error_t = BTS_batch - tau
            lambda_{t+1} = clip(lambda_t + eta * error_t, lambda_min, lambda_max)

        Args:
            bts_batch: the BTS score estimated on the current mini-batch
                (Algorithm 2, line 6).

        Returns:
            The updated lambda_{t+1}.
        """
        error = bts_batch - self.config.tau
        raw_update = self.state.lam + self.config.eta_lambda * error
        new_lam = self.project(raw_update)

        self.state.history.append(
            {"step": self.state.step, "bts": bts_batch, "lambda": new_lam, "error": error}
        )
        self.state.lam = new_lam
        self.state.step += 1
        return new_lam

    def composite_loss(self, task_loss: float, bts_batch: float) -> float:
        """
        Eq. (12)/(18): L(theta, lambda_t) = L_task(theta) + lambda_t * (BTS(theta) - tau)

        Note: the manuscript's Eq. 5/18 uses the simpler `lambda * BTS`
        form (without subtracting tau) for the objective actually
        minimized, while Eq. 12 (the Lagrangian relaxation of the
        constrained problem) uses `lambda * (BTS - tau)`. We expose both;
        this method implements Eq. 12's Lagrangian form and
        `composite_loss_unconstrained` implements Eq. 5/18.
        """
        return task_loss + self.state.lam * (bts_batch - self.config.tau)

    def composite_loss_unconstrained(self, task_loss: float, bts_batch: float) -> float:
        """Eq. (5)/(18): L = L_task + lambda * BTS."""
        return task_loss + self.state.lam * bts_batch

    def is_converged(self, window: int = 10, tol: float = 1e-3) -> bool:
        """Convergence monitoring: lambda has stabilized over the last
        `window` steps (used for logging / early-stopping diagnostics,
        not part of the manuscript's formal convergence proof in
        Appendix C.1, which instead bounds eta_lambda by 2 / L_BTS)."""
        if len(self.state.history) < window:
            return False
        recent = [h["lambda"] for h in self.state.history[-window:]]
        return (max(recent) - min(recent)) < tol

    def reset(self) -> None:
        self.state = ControllerState(lam=self.config.lambda_init)


def max_stable_learning_rate(l_bts_lipschitz: float) -> float:
    """
    Appendix C.1: stability condition for the dual update.

        0 < eta_lambda < 2 / L_BTS

    Args:
        l_bts_lipschitz: an estimated/assumed Lipschitz constant of
            BTS(theta) with respect to theta (must be estimated
            empirically for a real model; not provided by the manuscript
            as a closed-form value).
    """
    if l_bts_lipschitz <= 0:
        raise ValueError("Lipschitz constant must be positive")
    return 2.0 / l_bts_lipschitz
