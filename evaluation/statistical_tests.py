"""
statistical_tests.py
====================
Statistical validation tests for ADAPT-BTS experimental results.

Implements the statistical testing protocol from Section 5.9 and
reported in the paper's results (p < 0.001, Cohen's d = 0.84):

  1. One-way ANOVA — test for significant differences in F1 distributions
     across methods.
  2. Kruskal–Wallis test — non-parametric test for BTS distributions.
  3. Paired per-language t-test — pairwise comparison between ADAPT-BTS
     and each baseline for each metric.
  4. Mann–Whitney U test — non-parametric pairwise comparisons.
  5. Cohen's d — effect size for practical significance.

These tests establish that the reported improvements are:
  (a) statistically significant (p < 0.001)
  (b) practically meaningful (Cohen's d = 0.84, large effect)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import (
    f_oneway,
    kruskal,
    mannwhitneyu,
    ttest_rel,
    shapiro,
    levene,
)

logger = logging.getLogger(__name__)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    d = (μ_a - μ_b) / s_pooled

    where s_pooled = sqrt(((n_a-1)·s_a² + (n_b-1)·s_b²) / (n_a+n_b-2))

    Interpretation (Cohen, 1988):
      |d| < 0.2  → small
      0.2 ≤ |d| < 0.5 → small-medium
      0.5 ≤ |d| < 0.8 → medium
      |d| ≥ 0.8  → large

    Parameters
    ----------
    group_a, group_b : 1D arrays

    Returns
    -------
    d : float (signed)
    """
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return 0.0

    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if s_pooled == 0:
        return 0.0

    return float((mean_a - mean_b) / s_pooled)


def paired_ttest(
    scores_model: np.ndarray, scores_baseline: np.ndarray
) -> Tuple[float, float]:
    """
    Paired per-language t-test comparing a model against a baseline.
    Used when each language provides one measurement per method.

    Returns
    -------
    t_statistic : float
    p_value : float
    """
    if len(scores_model) != len(scores_baseline):
        raise ValueError("Paired t-test requires equal-length arrays.")
    t_stat, p_val = ttest_rel(scores_model, scores_baseline)
    return float(t_stat), float(p_val)


def mann_whitney_u(
    scores_model: np.ndarray, scores_baseline: np.ndarray
) -> Tuple[float, float]:
    """
    Mann–Whitney U test for non-parametric pairwise comparison.
    Used when normality assumptions may not hold.

    Returns
    -------
    u_statistic : float
    p_value : float
    """
    u_stat, p_val = mannwhitneyu(
        scores_model, scores_baseline, alternative="two-sided"
    )
    return float(u_stat), float(p_val)


def one_way_anova(*groups: np.ndarray) -> Tuple[float, float]:
    """
    One-way ANOVA for testing differences in F1 distributions
    across multiple methods (Section 5.9).

    Parameters
    ----------
    *groups : variable number of 1D arrays (one per method)

    Returns
    -------
    f_statistic : float
    p_value : float
    """
    f_stat, p_val = f_oneway(*groups)
    return float(f_stat), float(p_val)


def kruskal_wallis(*groups: np.ndarray) -> Tuple[float, float]:
    """
    Kruskal–Wallis test for BTS distributions across methods.
    Non-parametric alternative to one-way ANOVA.

    Returns
    -------
    h_statistic : float
    p_value : float
    """
    h_stat, p_val = kruskal(*groups)
    return float(h_stat), float(p_val)


def normality_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Shapiro–Wilk normality test. Used to validate Q-Q plot assumptions.

    Returns
    -------
    w_statistic : float
    p_value : float  (p < 0.05 → reject normality)
    """
    if len(data) > 5000:
        # Shapiro–Wilk is not reliable for very large samples
        data = np.random.choice(data, size=5000, replace=False)
    w_stat, p_val = shapiro(data)
    return float(w_stat), float(p_val)


def interpret_effect_size(d: float) -> str:
    """Return a human-readable interpretation of Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def full_comparison_report(
    adapt_bts_scores: Dict[str, np.ndarray],
    baseline_scores: Dict[str, Dict[str, np.ndarray]],
    alpha: float = 0.05,
) -> Dict:
    """
    Generate a complete statistical comparison report between ADAPT-BTS
    and all baseline methods across all metrics.

    Parameters
    ----------
    adapt_bts_scores : Dict[metric → 1D array of per-language scores]
    baseline_scores : Dict[method_name → Dict[metric → 1D array]]
    alpha : float
        Significance level.

    Returns
    -------
    report : Dict
        Structured report with statistical test results for each
        (baseline, metric) pair.
    """
    report = {}

    for baseline_name, baseline_metric_scores in baseline_scores.items():
        report[baseline_name] = {}

        for metric, adapt_vals in adapt_bts_scores.items():
            base_vals = baseline_metric_scores.get(metric)
            if base_vals is None or len(base_vals) != len(adapt_vals):
                continue

            # Paired t-test
            t_stat, p_val_t = paired_ttest(adapt_vals, base_vals)

            # Mann–Whitney U
            u_stat, p_val_u = mann_whitney_u(adapt_vals, base_vals)

            # Cohen's d
            d = cohens_d(adapt_vals, base_vals)

            # Normality of differences
            diffs = adapt_vals - base_vals
            w_stat, p_normal = normality_test(diffs)

            report[baseline_name][metric] = {
                "mean_adapt": float(np.mean(adapt_vals)),
                "mean_baseline": float(np.mean(base_vals)),
                "mean_diff": float(np.mean(diffs)),
                "t_statistic": t_stat,
                "p_value_ttest": p_val_t,
                "u_statistic": u_stat,
                "p_value_mwu": p_val_u,
                "cohens_d": d,
                "effect_size_interpretation": interpret_effect_size(d),
                "significant_ttest": p_val_t < alpha,
                "significant_mwu": p_val_u < alpha,
                "normality_w": w_stat,
                "p_normality": p_normal,
            }

            # Log summary
            sig_str = "***" if p_val_t < 0.001 else ("**" if p_val_t < 0.01 else ("*" if p_val_t < 0.05 else "ns"))
            logger.info(
                f"ADAPT-BTS vs {baseline_name} | {metric}: "
                f"Δ={np.mean(diffs):+.4f}, "
                f"p={p_val_t:.4f}{sig_str}, "
                f"d={d:.3f} ({interpret_effect_size(d)})"
            )

    return report


def compute_anova_across_methods(
    method_scores: Dict[str, np.ndarray], metric: str
) -> Dict:
    """
    Run one-way ANOVA and Kruskal-Wallis across all methods for a given metric.

    Parameters
    ----------
    method_scores : Dict[method_name → 1D array of per-language scores]
    metric : str (for logging)

    Returns
    -------
    Dict with ANOVA and Kruskal-Wallis results.
    """
    groups = list(method_scores.values())
    method_names = list(method_scores.keys())

    f_stat, p_anova = one_way_anova(*groups)
    h_stat, p_kruskal = kruskal_wallis(*groups)

    logger.info(
        f"[{metric}] One-way ANOVA: F={f_stat:.4f}, p={p_anova:.6f} | "
        f"Kruskal-Wallis: H={h_stat:.4f}, p={p_kruskal:.6f}"
    )

    return {
        "metric": metric,
        "methods": method_names,
        "anova_f": f_stat,
        "anova_p": p_anova,
        "kruskal_h": h_stat,
        "kruskal_p": p_kruskal,
        "significant_anova": p_anova < 0.05,
        "significant_kruskal": p_kruskal < 0.05,
    }


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic=np.mean,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a given statistic.

    Parameters
    ----------
    data : 1D array
    n_bootstrap : number of bootstrap samples
    confidence : confidence level
    statistic : function to compute (default: mean)

    Returns
    -------
    (lower_bound, upper_bound) : Tuple[float, float]
    """
    bootstrap_stats = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return lower, upper
