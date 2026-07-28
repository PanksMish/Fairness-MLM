import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from evaluation.report import (
    global_comparison_table, per_language_table, statistical_summary_table,
    regression_summary_table, dataset_statistics_table, save_table,
)


def test_global_comparison_table_shape_and_content():
    results = {
        "mT5-FT": {"Macro-F1": 78.4, "BTS": 0.77, "CCR": 73.8},
        "ADAPT-BTS": {"Macro-F1": 85.6, "BTS": 0.36, "CCR": 88.3},
    }
    df = global_comparison_table(results)
    assert list(df.index) == ["mT5-FT", "ADAPT-BTS"]  # order preserved
    assert df.loc["ADAPT-BTS", "Macro-F1"] == 85.6
    assert df.index.name == "Model"


def test_per_language_table_computes_delta():
    per_lang = {
        "en": {"base_f1": 85.2, "ours_f1": 87.5, "base_bts": 0.66, "ours_bts": 0.35},
        "sw": {"base_f1": 78.0, "ours_f1": 81.2, "base_bts": 0.80, "ours_bts": 0.45},
    }
    df = per_language_table(per_lang)
    assert abs(df.loc["en", "\u0394F1"] - 2.3) < 1e-9
    assert abs(df.loc["sw", "\u0394F1"] - 3.2) < 1e-9


def test_statistical_summary_table_from_paired_results():
    class FakePairedResult:
        def __init__(self, name, diff, p, d):
            self.metric_name = name
            self.mean_diff = diff
            self.t_pvalue = p
            self.cohens_d = d

    results = [
        FakePairedResult("Macro-F1", 0.8, 0.0001, 0.84),
        FakePairedResult("BTS", -0.05, 0.0002, 0.91),
    ]
    df = statistical_summary_table(results)
    assert df.loc["Macro-F1", "Improvement"] == 0.8
    assert df.loc["BTS", "Cohen's d"] == 0.91


def test_regression_summary_table():
    regression_result = {"slope": 0.837, "r_squared": 0.80, "pearson_r": 0.89}
    df = regression_summary_table(regression_result)
    assert df.loc["Relationship", "Value"] == "Positive"
    assert df.loc["Coefficient of determination (R\u00b2)", "Value"] == 0.80


def test_regression_summary_table_negative_slope():
    regression_result = {"slope": -0.5, "r_squared": 0.3, "pearson_r": -0.55}
    df = regression_summary_table(regression_result)
    assert df.loc["Relationship", "Value"] == "Negative"


def test_dataset_statistics_table():
    per_category = {
        "HR": {"n_languages": 18, "avg_tokens": "2B+"},
        "LR": {"n_languages": 46, "avg_tokens": "38M"},
    }
    df = dataset_statistics_table(per_category)
    assert df.loc["HR", "n_languages"] == 18


def test_save_table_csv_roundtrip():
    tmpdir = tempfile.mkdtemp()
    try:
        df = global_comparison_table({"A": {"F1": 1.0}, "B": {"F1": 2.0}})
        path = save_table(df, os.path.join(tmpdir, "table.csv"), fmt="csv")
        loaded = pd.read_csv(path, index_col=0)
        assert loaded.loc["B", "F1"] == 2.0
    finally:
        shutil.rmtree(tmpdir)


def test_save_table_markdown():
    tmpdir = tempfile.mkdtemp()
    try:
        df = global_comparison_table({"A": {"F1": 1.0}})
        path = save_table(df, os.path.join(tmpdir, "table.md"), fmt="markdown")
        content = open(path).read()
        assert "F1" in content and "A" in content
    finally:
        shutil.rmtree(tmpdir)


def test_save_table_rejects_unknown_format():
    import pytest
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        save_table(df, "/tmp/x", fmt="xyz")


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
