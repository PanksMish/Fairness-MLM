import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fairness.fairness_controller import (
    FairnessController,
    ControllerConfig,
    max_stable_learning_rate,
)


def test_lambda_increases_when_bts_exceeds_tau():
    cfg = ControllerConfig(tau=0.4, eta_lambda=0.1, lambda_init=0.2, lambda_min=0.0, lambda_max=1.0)
    ctrl = FairnessController(cfg)
    new_lam = ctrl.update(bts_batch=0.7)  # violation: 0.7 > tau=0.4
    # lambda_1 = 0.2 + 0.1*(0.7-0.4) = 0.23
    assert new_lam == pytest.approx(0.23)
    assert new_lam > cfg.lambda_init


def test_lambda_decreases_when_bts_below_tau():
    cfg = ControllerConfig(tau=0.4, eta_lambda=0.1, lambda_init=0.5, lambda_min=0.0, lambda_max=1.0)
    ctrl = FairnessController(cfg)
    new_lam = ctrl.update(bts_batch=0.1)  # 0.1 < tau=0.4
    # lambda_1 = 0.5 + 0.1*(0.1-0.4) = 0.47
    assert new_lam == pytest.approx(0.47)
    assert new_lam < cfg.lambda_init


def test_lambda_projected_to_bounds_eq14():
    cfg = ControllerConfig(tau=0.0, eta_lambda=10.0, lambda_init=0.5, lambda_min=0.1, lambda_max=0.9)
    ctrl = FairnessController(cfg)
    new_lam = ctrl.update(bts_batch=1.0)  # huge push upward
    assert new_lam == cfg.lambda_max
    ctrl.reset()
    new_lam = ctrl.update(bts_batch=-1.0)  # huge push downward (won't happen in practice, BTS>=0, but tests projection)
    assert new_lam == cfg.lambda_min


def test_composite_loss_lagrangian_form_eq12():
    cfg = ControllerConfig(tau=0.4, lambda_init=0.3)
    ctrl = FairnessController(cfg)
    loss = ctrl.composite_loss(task_loss=2.0, bts_batch=0.6)
    # 2.0 + 0.3*(0.6-0.4) = 2.06
    assert loss == pytest.approx(2.06)


def test_composite_loss_unconstrained_form_eq18():
    cfg = ControllerConfig(lambda_init=0.3)
    ctrl = FairnessController(cfg)
    loss = ctrl.composite_loss_unconstrained(task_loss=2.0, bts_batch=0.6)
    # 2.0 + 0.3*0.6 = 2.18
    assert loss == pytest.approx(2.18)


def test_convergence_detection():
    cfg = ControllerConfig(tau=0.4, eta_lambda=0.0)  # eta=0 -> lambda never moves -> should converge fast
    ctrl = FairnessController(cfg)
    assert not ctrl.is_converged(window=5)
    for _ in range(10):
        ctrl.update(bts_batch=0.4)
    assert ctrl.is_converged(window=5, tol=1e-6)


def test_history_recorded_per_step():
    ctrl = FairnessController(ControllerConfig())
    for i in range(5):
        ctrl.update(bts_batch=0.3 + 0.01 * i)
    assert len(ctrl.state.history) == 5
    assert ctrl.state.step == 5
    assert [h["step"] for h in ctrl.state.history] == [0, 1, 2, 3, 4]


def test_max_stable_learning_rate_appendix_c1():
    # eta_lambda < 2 / L_BTS
    eta_max = max_stable_learning_rate(l_bts_lipschitz=4.0)
    assert eta_max == pytest.approx(0.5)
    with pytest.raises(ValueError):
        max_stable_learning_rate(l_bts_lipschitz=-1.0)


def test_lambda_never_leaves_bounds_under_random_walk():
    import random
    random.seed(42)
    cfg = ControllerConfig(tau=0.4, eta_lambda=0.2, lambda_min=0.05, lambda_max=1.0)
    ctrl = FairnessController(cfg)
    for _ in range(200):
        bts = random.uniform(0.0, 1.0)
        lam = ctrl.update(bts)
        assert cfg.lambda_min <= lam <= cfg.lambda_max


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
