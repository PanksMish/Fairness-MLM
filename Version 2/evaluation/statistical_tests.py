"""
Statistical analysis over per-language paired observations (n=101), as
described in Section 5.2:

    "the statistical significance of results is determined through paired
    language-level tests ... n = 101 paired samples ... Cohen's d is used
    to assess practical significance"

All functions operate on two arrays of per-language metric values (one
per system being compared) -- e.g. ADAPT-BTS's per-language Macro-F1 vs.
MFC's per-language Macro-F1, matching Table 6 / Table 8's structure.
Nothing here hardcodes the manuscript's reported p-values or effect
sizes; they are recomputed from whatever per-language arrays are passed
in from real evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PairedTestResult:
    metric_name: str
    mean_diff: float
    t_stat: float
    t_pvalue: float
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    cohens_d: float
    ci_95: tuple[float, float]  # bootstrap CI on the mean difference
    n: int


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired samples: d = mean(diff) / std(diff).
    Matches the "moderate-to-large" effect sizes reported in Table 6 /
    Fig. 11(a), recomputed rather than looked up.
    """
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    statistic=np.mean,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a 1D array of paired
    differences (or any statistic array)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)
    lower = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return lower, upper


def paired_comparison(
    system_a: np.ndarray,
    system_b: np.ndarray,
    metric_name: str = "metric",
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedTestResult:
    """
    Full paired comparison between two systems' per-language metric
    arrays, matching the analysis reported in Table 6 (Macro-F1, BTS,
    CCR, DPG improvements of ADAPT-BTS vs. MFC across n=101 languages).

    Args:
        system_a: per-language values for system A (e.g. ADAPT-BTS), shape (n,)
        system_b: per-language values for system B (e.g. MFC), shape (n,)
    """
    a = np.asarray(system_a, dtype=np.float64)
    b = np.asarray(system_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must match shape: {a.shape} vs {b.shape}")
    diff = a - b

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # Wilcoxon undefined if all diffs are zero
        w_stat, w_p = float("nan"), float("nan")

    d = cohens_d_paired(a, b)
    ci_low, ci_high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)

    return PairedTestResult(
        metric_name=metric_name,
        mean_diff=float(diff.mean()),
        t_stat=float(t_stat),
        t_pvalue=float(t_p),
        wilcoxon_stat=float(w_stat),
        wilcoxon_pvalue=float(w_p),
        cohens_d=d,
        ci_95=(ci_low, ci_high),
        n=len(a),
    )


def anova_across_groups(*groups: np.ndarray) -> dict:
    """One-way ANOVA across resource groups (HR/MR/LR) or typological
    categories (analytic/fusional/agglutinative/inflectional), as in
    Fig. 6 / Fig. 12."""
    f_stat, p_val = stats.f_oneway(*groups)
    return {"f_stat": float(f_stat), "p_value": float(p_val)}


def kruskal_wallis_across_groups(*groups: np.ndarray) -> dict:
    """Non-parametric analogue of ANOVA for the same group comparisons."""
    h_stat, p_val = stats.kruskal(*groups)
    return {"h_stat": float(h_stat), "p_value": float(p_val)}


def pearson_and_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Correlation between two per-language metric arrays, e.g. BTS vs. DPG
    (Table 7 / Fig. 15: reports r=0.89, R^2=0.80 -- recomputed here from
    real per-language data rather than asserted).
    """
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
    }


def ols_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Simple OLS regression y ~ x, matching Fig. 15 (BTS vs DPG linear fit)
    and Table 7 (R^2). Returns slope, intercept, R^2, and residuals for
    downstream residual diagnostics (Fig. 16).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_hat = slope * x + intercept
    residuals = y - y_hat
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "fitted": y_hat,
        "residuals": residuals,
    }


def shapiro_normality_check(residuals: np.ndarray) -> dict:
    """Used in residual diagnostics (Fig. 16 QQ-plot companion check) to
    verify the regression's normality assumption is not badly violated."""
    stat, p_value = stats.shapiro(residuals)
    return {"stat": float(stat), "p_value": float(p_value)}
