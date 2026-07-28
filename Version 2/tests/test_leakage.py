import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from evaluation.leakage import compute_leakage, leakage_above_chance


def test_leakage_high_when_representations_perfectly_separable():
    rng = np.random.default_rng(0)
    n = 200
    # class 0 clustered around +5, class 1 clustered around -5 -> trivially separable
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    reps = np.vstack([
        rng.normal(5.0, 0.5, size=(n // 2, 8)),
        rng.normal(-5.0, 0.5, size=(n // 2, 8)),
    ])
    result = compute_leakage(reps, labels, seed=0)
    assert result.probe_accuracy > 0.9
    assert leakage_above_chance(result)


def test_leakage_near_chance_when_representations_uninformative():
    rng = np.random.default_rng(1)
    n = 200
    labels = rng.integers(0, 2, size=n)
    reps = rng.normal(0.0, 1.0, size=(n, 8))  # pure noise, unrelated to labels
    result = compute_leakage(reps, labels, seed=1)
    # Should be close to chance (0.5 for 2 classes); allow some slack for
    # finite-sample noise in a small synthetic test set
    assert abs(result.probe_accuracy - 0.5) < 0.20


def test_leakage_chance_accuracy_matches_num_classes():
    rng = np.random.default_rng(2)
    n = 300
    labels = rng.integers(0, 3, size=n)  # 3 classes
    reps = rng.normal(0.0, 1.0, size=(n, 8))
    result = compute_leakage(reps, labels, seed=2)
    assert abs(result.chance_accuracy - (1.0 / 3.0)) < 1e-9


def test_leakage_rejects_single_class():
    import pytest
    labels = np.zeros(50)
    reps = np.random.randn(50, 4)
    with pytest.raises(ValueError):
        compute_leakage(reps, labels)


def test_leakage_rejects_shape_mismatch():
    import pytest
    with pytest.raises(ValueError):
        compute_leakage(np.random.randn(10, 4), np.array([0, 1, 0]))


def test_leakage_above_chance_respects_margin():
    from evaluation.leakage import LeakageResult
    result = LeakageResult(probe_accuracy=0.55, probe_macro_f1=0.5, chance_accuracy=0.5, n_train=10, n_test=10)
    assert not leakage_above_chance(result, margin=0.10)  # 0.55 - 0.5 = 0.05 < 0.10
    assert leakage_above_chance(result, margin=0.02)


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
