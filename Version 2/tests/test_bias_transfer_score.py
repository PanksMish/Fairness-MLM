import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairness.bias_transfer_score import (
    total_variation_distance,
    compute_bts,
    instance_weighted_global_bts,
    semantic_preservation_score,
    _validate_distributions,
)


def test_tv_distance_identical_distributions_is_zero():
    p = np.array([[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    delta = total_variation_distance(p, p.copy())
    assert np.allclose(delta, 0.0)


def test_tv_distance_disjoint_supports_is_one():
    p_a = np.array([[1.0, 0.0, 0.0]])
    p_b = np.array([[0.0, 1.0, 0.0]])
    delta = total_variation_distance(p_a, p_b)
    assert np.allclose(delta, 1.0)


def test_tv_distance_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_classes = rng.integers(2, 10)
        p_a = rng.dirichlet(np.ones(n_classes), size=8)
        p_b = rng.dirichlet(np.ones(n_classes), size=8)
        delta = total_variation_distance(p_a, p_b)
        assert np.all(delta >= 0.0) and np.all(delta <= 1.0)


def test_tv_distance_symmetric():
    rng = np.random.default_rng(1)
    p_a = rng.dirichlet(np.ones(4), size=10)
    p_b = rng.dirichlet(np.ones(4), size=10)
    assert np.allclose(total_variation_distance(p_a, p_b), total_variation_distance(p_b, p_a))


def test_validate_rejects_non_normalized_rows():
    p_a = np.array([[0.5, 0.6]])  # sums to 1.1
    p_b = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError):
        _validate_distributions(p_a, p_b)


def test_validate_rejects_shape_mismatch():
    p_a = np.array([[0.5, 0.5]])
    p_b = np.array([[0.3, 0.3, 0.4]])
    with pytest.raises(ValueError):
        total_variation_distance(p_a, p_b)


def test_compute_bts_matches_manual_expectation():
    p_a = np.array([[0.6, 0.4], [0.5, 0.5], [1.0, 0.0]])
    p_b = np.array([[0.2, 0.8], [0.5, 0.5], [0.0, 1.0]])
    result = compute_bts(p_a, p_b)
    manual = np.array([0.4, 0.0, 1.0])
    assert np.allclose(result.per_instance, manual)
    assert np.isclose(result.mean, manual.mean())
    assert result.n == 3


def test_compute_bts_weighted_matches_manual():
    p_a = np.array([[1.0, 0.0], [0.5, 0.5]])
    p_b = np.array([[0.0, 1.0], [0.5, 0.5]])
    weights = np.array([9.0, 1.0])  # instance 0 dominates
    result = compute_bts(p_a, p_b, weights=weights)
    # instance-weighted mean: (9*1.0 + 1*0.0)/10 = 0.9
    assert np.isclose(result.mean, 0.9)


def test_bts_bounds_hold_range_0_to_1():
    rng = np.random.default_rng(2)
    p_a = rng.dirichlet(np.ones(5), size=200)
    p_b = rng.dirichlet(np.ones(5), size=200)
    result = compute_bts(p_a, p_b)
    assert 0.0 <= result.mean <= 1.0


def test_instance_weighted_global_bts_matches_eq16_style_aggregation():
    per_lang_bts = {"en": 0.30, "sw": 0.60}
    per_lang_n = {"en": 100, "sw": 900}
    global_bts = instance_weighted_global_bts(per_lang_bts, per_lang_n)
    expected = (100 * 0.30 + 900 * 0.60) / 1000
    assert np.isclose(global_bts, expected)
    # should be pulled toward the larger-n language
    assert global_bts > (0.30 + 0.60) / 2


def test_instance_weighted_global_bts_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        instance_weighted_global_bts({"en": 0.1}, {"fr": 10})


def test_semantic_preservation_threshold_eq3():
    cos_sim = np.array([0.90, 0.84, 0.85, 0.10])
    mask = semantic_preservation_score(cos_sim, threshold=0.85)
    assert list(mask) == [True, False, True, False]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
