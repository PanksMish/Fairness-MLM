

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend; caller can switch backends before importing if interactive display is wanted
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, save_path: Optional[str | Path]) -> Optional[str]:
    if save_path is None:
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def bar_comparison(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str = "",
    lower_is_better: bool = False,
    highlight_label: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Single-metric bar chart across methods, matching the style of
    Fig. 4 (Macro-F1/Span-F1 comparison) or Fig. 5 top-left (BTS
    comparison). `highlight_label`, if given, draws that bar in a
    distinct color (e.g. to call out ADAPT-BTS among baselines).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#c0392b" if lbl == highlight_label else "#3477a8" for lbl in labels]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel + (" \u2193 (lower is better)" if lower_is_better else " \u2191"))
    if title:
        ax.set_title(title)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, save_path)


def resource_level_lines(
    resource_categories: list[str],  # e.g. ["HR", "MR", "LR"]
    series: dict[str, list[float]],  # method_name -> values aligned with resource_categories
    ylabel: str,
    highlight_label: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Multi-method line chart across resource tiers, matching Fig. 6's
    style (Macro-F1 or BTS vs. HR/MR/LR)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, values in series.items():
        is_highlight = name == highlight_label
        ax.plot(
            resource_categories, values, marker="o",
            linewidth=2.5 if is_highlight else 1.2,
            color="#c0392b" if is_highlight else None,
            label=name, zorder=3 if is_highlight else 2,
        )
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, save_path)


def training_dynamics(
    steps: list[int],
    series: dict[str, list[float]],
    ylabel: str,
    save_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Convergence-over-training-steps chart, matching Fig. 8's BTS
    convergence / lambda-trajectory style. `series` maps a line label
    (e.g. method name, or resource-tier name for a lambda trajectory) to
    a list of values aligned with `steps`."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, values in series.items():
        ax.plot(steps, values, marker="o", markersize=3, label=name)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def effect_size_bar(
    labels: list[str],
    cohens_d: list[float],
    highlight_label: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Cohen's d comparison with conventional small/medium/large
    reference lines, matching Fig. 11(a)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#c0392b" if lbl == highlight_label else "#555555" for lbl in labels]
    bars = ax.bar(labels, cohens_d, color=colors)
    for threshold, style in [(0.2, ":"), (0.5, "--"), (0.8, "-")]:
        ax.axhline(threshold, color="gray", linestyle=style, linewidth=0.8)
    ax.set_ylabel("Cohen's d")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, save_path)


def kde_distribution(
    data_by_group: dict[str, np.ndarray],
    xlabel: str,
    save_path: Optional[str | Path] = None,
    bins: int = 20,
) -> Optional[str]:
    """Overlaid histogram + KDE per group, matching Fig. 10/14's
    distributional comparisons. Uses scipy's gaussian_kde directly
    (avoids a seaborn dependency this repo doesn't otherwise need)."""
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, values in data_by_group.items():
        values = np.asarray(values)
        ax.hist(values, bins=bins, density=True, alpha=0.35, label=f"{name} (hist)")
        if len(values) > 1 and np.std(values) > 0:
            kde = gaussian_kde(values)
            x_grid = np.linspace(values.min(), values.max(), 200)
            ax.plot(x_grid, kde(x_grid), label=f"{name} (KDE)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, save_path)


def scatter_with_regression(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Per-language correlation scatter with an OLS fit line, matching
    Fig. 9/15's style. Regression is computed via
    evaluation/statistical_tests.py's real ols_regression (not
    re-derived here), so the fitted line matches whatever
    Table 7-style statistics the caller separately reports."""
    from evaluation.statistical_tests import ols_regression

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, alpha=0.6, s=20)
    result = ols_regression(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = result["slope"] * x_line + result["intercept"]
    ax.plot(x_line, y_line, color="black", linestyle="--",
            label=f"OLS: R\u00b2={result['r_squared']:.2f}, p={result['p_value']:.3g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def residual_diagnostics(
    fitted: np.ndarray,
    residuals: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Residuals-vs-fitted + Q-Q plot side by side, matching Fig. 16.
    Q-Q reference line computed via scipy.stats.probplot (real
    normal-quantile computation, not a hand-rolled approximation)."""
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(fitted, residuals, alpha=0.6, s=15)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    scipy_stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot of Residuals")

    fig.tight_layout()
    return _save(fig, save_path)
