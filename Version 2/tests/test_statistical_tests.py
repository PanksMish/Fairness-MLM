import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from evaluation.statistical_tests import (
    cohens_d_paired,
    bootstrap_ci,
    paired_comparison,
    pearson_and_spearman,
    ols_regression,
    anova_across_groups,
    kruskal_wallis_across_groups,
)


def test_cohens_d_zero_when_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d_paired(a, a) == 0.0


def test_cohens_d_known_value():
    # diff is constant [2,2,2,2] -> std=0 -> handled as 0.0 special case
    a = np.array([3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d_paired(a, b) == 0.0  # zero variance in diff


def test_cohens_d_nonzero_realistic():
    rng = np.random.default_rng(0)
    a = rng.normal(85, 2, size=101)
    b = rng.normal(84, 2, size=101)
    d = cohens_d_paired(a, b)
    assert isinstance(d, float)
    assert d != 0.0


def test_bootstrap_ci_contains_true_mean_diff_usually():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.8, 0.5, size=101)  # e.g. Macro-F1 gain per language
    lo, hi = bootstrap_ci(diffs, n_boot=2000, seed=1)
    assert lo < diffs.mean() < hi


def test_paired_comparison_matches_known_significant_difference():
    rng = np.random.default_rng(2)
    n = 101
    b = rng.normal(84.0, 3.0, size=n)   # baseline per-language F1
    a = b + rng.normal(0.8, 0.3, size=n)  # ADAPT-BTS: consistent +0.8 improvement
    result = paired_comparison(a, b, metric_name="Macro-F1", n_boot=2000, seed=2)
    assert result.n == 101
    assert result.mean_diff > 0.5
    assert result.t_pvalue < 0.001  # should be highly significant given consistent gain
    assert result.cohens_d > 0.5    # medium-large effect


def test_paired_comparison_rejects_shape_mismatch():
    import pytest
    with pytest.raises(ValueError):
        paired_comparison(np.array([1, 2, 3]), np.array([1, 2]))


def test_pearson_spearman_perfect_correlation():
    x = np.arange(1, 20).astype(float)
    y = 2 * x + 1
    corr = pearson_and_spearman(x, y)
    assert corr["pearson_r"] > 0.999
    assert corr["spearman_r"] > 0.999


def test_ols_regression_recovers_known_line():
    x = np.linspace(0, 1, 50)
    y = 3.0 * x + 0.5  # no noise
    result = ols_regression(x, y)
    assert abs(result["slope"] - 3.0) < 1e-6
    assert abs(result["intercept"] - 0.5) < 1e-6
    assert result["r_squared"] > 0.999


def test_ols_regression_with_noise_gives_reasonable_r2():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 0.9, size=101)  # e.g. per-language BTS
    y = 0.837 * x + 0.02 + rng.normal(0, 0.02, size=101)  # roughly matches DPG-BTS relationship shape
    result = ols_regression(x, y)
    assert result["r_squared"] > 0.5  # strong but not perfect due to noise


def test_anova_detects_group_difference():
    rng = np.random.default_rng(4)
    hr = rng.normal(0.35, 0.02, size=18)
    mr = rng.normal(0.35, 0.02, size=37)
    lr = rng.normal(0.45, 0.02, size=46)  # clearly different mean
    result = anova_across_groups(hr, mr, lr)
    assert result["p_value"] < 0.001


def test_kruskal_wallis_detects_group_difference():
    rng = np.random.default_rng(5)
    hr = rng.normal(0.35, 0.02, size=18)
    mr = rng.normal(0.35, 0.02, size=37)
    lr = rng.normal(0.45, 0.02, size=46)
    result = kruskal_wallis_across_groups(hr, mr, lr)
    assert result["p_value"] < 0.001


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
