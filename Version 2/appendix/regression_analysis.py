"""
Appendix A.3-A.6-style analyses: BTS-vs-DPG regression (Fig. 15, Table
7), residual diagnostics (Fig. 16), and per-language improvement
distributions (Fig. 17). Thin orchestration over already-tested pieces
(evaluation/statistical_tests.py's ols_regression,
visualization/plots.py's scatter_with_regression and
residual_diagnostics, evaluation/report.py's regression_summary_table)
-- no new statistics or plotting logic lives here, just wiring, so it's
testable end-to-end with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from evaluation.statistical_tests import ols_regression, pearson_and_spearman
from evaluation.report import regression_summary_table, save_table
from visualization.plots import scatter_with_regression, residual_diagnostics


@dataclass
class RegressionAnalysisResult:
    r_squared: float
    pearson_r: float
    slope: float
    p_value: float
    scatter_plot_path: Optional[str]
    residual_plot_path: Optional[str]
    summary_table_path: Optional[str]


def analyze_relationship(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    output_dir: Optional[str | Path] = None,
    file_prefix: str = "relationship",
) -> RegressionAnalysisResult:
    """
    Full Appendix A.3-A.5 pipeline for one pair of per-language metric
    arrays (e.g. BTS vs. DPG, or any other pair a real evaluation run
    produces per language). Writes a scatter+regression PNG, a
    residual-diagnostics PNG, and a Table-7-shaped CSV, if `output_dir`
    is given; returns the underlying numbers regardless.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    reg = ols_regression(x, y)
    corr = pearson_and_spearman(x, y)

    scatter_path = residual_path = table_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        scatter_path = scatter_with_regression(
            x, y, xlabel=x_label, ylabel=y_label,
            save_path=output_dir / f"{file_prefix}_scatter.png",
        )
        residual_path = residual_diagnostics(
            reg["fitted"], reg["residuals"],
            save_path=output_dir / f"{file_prefix}_residuals.png",
        )
        table_df = regression_summary_table({**reg, "pearson_r": corr["pearson_r"]})
        table_path = save_table(table_df, output_dir / f"{file_prefix}_regression_summary.csv", fmt="csv")

    return RegressionAnalysisResult(
        r_squared=reg["r_squared"], pearson_r=corr["pearson_r"],
        slope=reg["slope"], p_value=reg["p_value"],
        scatter_plot_path=scatter_path, residual_plot_path=residual_path,
        summary_table_path=table_path,
    )


def per_language_improvement_summary(base_values: dict[str, float], ours_values: dict[str, float]) -> dict:
    """
    Appendix B's "Across 101 languages: N exhibit positive delta, M show
    negligible change, K exhibit minor regression" style summary --
    computed directly from real per-language dicts rather than the
    manuscript's specific counts.
    """
    if set(base_values) != set(ours_values):
        raise ValueError("base_values and ours_values must cover the same languages")

    deltas = {lang: ours_values[lang] - base_values[lang] for lang in base_values}
    n_positive = sum(1 for d in deltas.values() if d > 0.5)
    n_negligible = sum(1 for d in deltas.values() if abs(d) <= 0.5)
    n_regression = sum(1 for d in deltas.values() if d < -0.5)

    return {
        "deltas": deltas,
        "n_languages": len(deltas),
        "n_positive": n_positive,
        "n_negligible": n_negligible,
        "n_regression": n_regression,
        "mean_delta": float(np.mean(list(deltas.values()))),
        "median_delta": float(np.median(list(deltas.values()))),
    }
