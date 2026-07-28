import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from appendix.regression_analysis import analyze_relationship, per_language_improvement_summary


def test_analyze_relationship_without_output_dir_returns_numbers_only():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.9, 101)
    y = 0.8 * x + 0.05 + rng.normal(0, 0.02, 101)
    result = analyze_relationship(x, y, "BTS", "DPG")
    assert result.scatter_plot_path is None
    assert result.residual_plot_path is None
    assert result.summary_table_path is None
    assert result.r_squared > 0.5  # strong linear relationship by construction
    assert result.pearson_r > 0.5


def test_analyze_relationship_writes_real_files_when_output_dir_given():
    tmpdir = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(1)
        x = rng.uniform(0.2, 0.9, 101)
        y = 0.8 * x + 0.05 + rng.normal(0, 0.02, 101)
        result = analyze_relationship(x, y, "BTS", "DPG", output_dir=tmpdir, file_prefix="bts_dpg")

        assert os.path.exists(result.scatter_plot_path)
        assert os.path.exists(result.residual_plot_path)
        assert os.path.exists(result.summary_table_path)

        # sanity-check the CSV table actually has the expected rows
        content = open(result.summary_table_path).read()
        assert "Pearson correlation" in content
        assert "Coefficient of determination" in content
    finally:
        shutil.rmtree(tmpdir)


def test_analyze_relationship_negative_correlation():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.2, 0.9, 101)
    y = -0.8 * x + 0.05 + rng.normal(0, 0.02, 101)
    result = analyze_relationship(x, y, "x", "y")
    assert result.slope < 0
    assert result.pearson_r < 0


def test_per_language_improvement_summary_basic():
    base = {"en": 78.4, "sw": 73.4, "de": 80.0}
    ours = {"en": 87.5, "sw": 83.6, "de": 79.9}  # de: negligible regression
    summary = per_language_improvement_summary(base, ours)
    assert summary["n_languages"] == 3
    assert summary["n_positive"] == 2  # en, sw
    assert summary["n_negligible"] == 1  # de (delta -0.1, within 0.5)
    assert summary["n_regression"] == 0
    assert abs(summary["deltas"]["en"] - 9.1) < 1e-9


def test_per_language_improvement_summary_detects_real_regression():
    base = {"en": 80.0}
    ours = {"en": 78.0}  # -2.0, a real regression
    summary = per_language_improvement_summary(base, ours)
    assert summary["n_regression"] == 1
    assert summary["n_positive"] == 0


def test_per_language_improvement_summary_rejects_mismatched_languages():
    import pytest
    with pytest.raises(ValueError):
        per_language_improvement_summary({"en": 1.0}, {"fr": 2.0})


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
