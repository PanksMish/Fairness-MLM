"""
Formats already-computed metrics (from evaluation/evaluator.py's
EvaluationReport, evaluation/statistical_tests.py's PairedTestResult, or
plain dicts) into Table 5/6/7/8-shaped pandas DataFrames, with
markdown/CSV export. No metric computation happens here -- this is
formatting only, matching the module's stated scope in the repo
structure brief ("evaluation/report.py").

Fully testable with synthetic data (pandas is available in this
sandbox); the functions here have actually been executed against
example inputs, see tests/test_report.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def global_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Table 5's shape: one row per model/method, columns for each metric.

    Args:
        results: {method_name: {"Macro-F1": ..., "BTS": ..., "CCR": ...,
            "DPG": ..., "Leakage": ...}} -- exactly what
            EvaluationReport.as_table_row() produces per method; the
            caller collects one such dict per baseline/method and passes
            them all in here keyed by method name.

    Returns:
        DataFrame indexed by method name, in insertion order (so the
        caller controls row ordering, e.g. baselines first, ADAPT-BTS
        last, matching Table 5's layout).
    """
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Model"
    return df


def per_language_table(per_language_results: dict[str, dict]) -> pd.DataFrame:
    """
    Table 8's shape: one row per language, with base/ours columns and a
    delta. `per_language_results` maps language -> {"base_f1": ...,
    "ours_f1": ..., "base_bts": ..., "ours_bts": ..., "resource": ...}.
    """
    rows = []
    for lang, vals in per_language_results.items():
        row = {"Language": lang}
        row.update(vals)
        if "base_f1" in vals and "ours_f1" in vals:
            row["\u0394F1"] = vals["ours_f1"] - vals["base_f1"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Language")


def statistical_summary_table(paired_results: list) -> pd.DataFrame:
    """
    Table 6's shape: one row per metric, columns Improvement/p-value/
    Cohen's d, built from a list of
    evaluation.statistical_tests.PairedTestResult objects (or any object
    with the same attribute names).
    """
    rows = []
    for r in paired_results:
        rows.append({
            "Metric": r.metric_name,
            "Improvement": r.mean_diff,
            "p-value": r.t_pvalue,
            "Cohen's d": r.cohens_d,
        })
    return pd.DataFrame(rows).set_index("Metric")


def regression_summary_table(regression_result: dict) -> pd.DataFrame:
    """Table 7's shape, built from evaluation.statistical_tests.ols_regression's
    output plus a Pearson r (pass pearson_r explicitly since ols_regression
    doesn't compute it itself -- use evaluation.statistical_tests.pearson_and_spearman)."""
    rows = [
        {"Statistic": "Pearson correlation (r)", "Value": regression_result.get("pearson_r")},
        {"Statistic": "Coefficient of determination (R\u00b2)", "Value": regression_result["r_squared"]},
        {"Statistic": "Relationship", "Value": "Positive" if regression_result["slope"] > 0 else "Negative"},
    ]
    return pd.DataFrame(rows).set_index("Statistic")


def dataset_statistics_table(per_category: dict[str, dict]) -> pd.DataFrame:
    """
    Table 2's shape: resource category -> {n_languages, avg_tokens,
    avg_train_size, tasks}. Purely a formatting convenience over
    whatever real counts the caller measured from their actual corpus
    (this function does not itself compute token counts -- see
    evaluation/fairness_metrics.resource_category for the categorization
    logic, and datasets/build_sentiment.py for where real counts would
    come from).
    """
    df = pd.DataFrame.from_dict(per_category, orient="index")
    df.index.name = "Category"
    return df


def save_table(df: pd.DataFrame, path: str | Path, fmt: str = "csv") -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path)
    elif fmt == "markdown":
        with open(path, "w") as f:
            f.write(df.to_markdown())
    elif fmt == "json":
        df.to_json(path, orient="index", indent=2)
    else:
        raise ValueError(f"Unknown format '{fmt}', expected 'csv', 'markdown', or 'json'")
    return str(path)
