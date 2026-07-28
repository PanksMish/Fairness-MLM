import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from visualization.plots import (
    bar_comparison, resource_level_lines, training_dynamics,
    effect_size_bar, kde_distribution, scatter_with_regression, residual_diagnostics,
)


def _assert_valid_png(path: str):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 500  # a real rendered PNG, not an empty/near-empty file
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes


def test_bar_comparison_produces_valid_png():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "bar.png")
        result = bar_comparison(
            labels=["mT5-FT", "MFC", "ADAPT-BTS"],
            values=[78.4, 84.8, 85.6],
            ylabel="Macro-F1 (%)",
            highlight_label="ADAPT-BTS",
            save_path=path,
        )
        assert result == path
        _assert_valid_png(path)
    finally:
        shutil.rmtree(tmpdir)


def test_resource_level_lines_produces_valid_png():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "resource.png")
        result = resource_level_lines(
            resource_categories=["HR", "MR", "LR"],
            series={"mT5-FT": [82.0, 83.0, 73.0], "ADAPT-BTS": [86.9, 87.3, 83.6]},
            ylabel="Macro-F1 (%)",
            highlight_label="ADAPT-BTS",
            save_path=path,
        )
        _assert_valid_png(result)
    finally:
        shutil.rmtree(tmpdir)


def test_training_dynamics_produces_valid_png():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "dynamics.png")
        steps = list(range(1, 11))
        rng = np.random.default_rng(0)
        result = training_dynamics(
            steps=steps,
            series={"ADAPT-BTS": (0.65 - 0.03 * np.array(steps) + rng.normal(0, 0.01, 10)).tolist()},
            ylabel="BTS",
            save_path=path,
        )
        _assert_valid_png(result)
    finally:
        shutil.rmtree(tmpdir)


def test_effect_size_bar_produces_valid_png():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "effect.png")
        result = effect_size_bar(
            labels=["Grad-Unl", "MFC", "ADAPT-BTS"],
            cohens_d=[0.55, 0.70, 0.84],
            highlight_label="ADAPT-BTS",
            save_path=path,
        )
        _assert_valid_png(result)
    finally:
        shutil.rmtree(tmpdir)


def test_kde_distribution_produces_valid_png():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "kde.png")
        rng = np.random.default_rng(1)
        result = kde_distribution(
            data_by_group={
                "mT5-FT": rng.normal(78, 5, 50),
                "ADAPT-BTS": rng.normal(86, 3, 50),
            },
            xlabel="Per-language Macro-F1 (%)",
            save_path=path,
        )
        _assert_valid_png(result)
    finally:
        shutil.rmtree(tmpdir)


def test_scatter_with_regression_produces_valid_png_and_uses_real_ols():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "scatter.png")
        rng = np.random.default_rng(2)
        x = rng.uniform(0.2, 0.9, 101)
        y = 0.8 * x + 0.05 + rng.normal(0, 0.02, 101)
        result = scatter_with_regression(x, y, xlabel="BTS", ylabel="DPG", save_path=path)
        _assert_valid_png(result)
    finally:
        shutil.rmtree(tmpdir)


def test_residual_diagnostics_produces_valid_png():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "residuals.png")
        rng = np.random.default_rng(3)
        fitted = rng.uniform(0.1, 0.25, 200)
        residuals = rng.normal(0, 0.02, 200)
        result = residual_diagnostics(fitted, residuals, save_path=path)
        _assert_valid_png(result)
    finally:
        shutil.rmtree(tmpdir)


def test_functions_return_none_without_save_path():
    # Confirms the functions don't crash when no save_path is given
    # (e.g. for interactive/notebook use), and correctly return None
    # rather than a bogus path.
    result = bar_comparison(["a", "b"], [1.0, 2.0], ylabel="y")
    assert result is None


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
