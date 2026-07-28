"""
Fairness metrics from Table 4 / Section 5.2:

  - CCR  (Counterfactual Consistency Rate), Eq. (19)
  - DPG  (Demographic Parity Gap)
  - instance-weighted global aggregation, Eq. (16)/(20)

Representation Leakage (linear-probe accuracy on frozen encoder
representations) requires a trained probe classifier and therefore lives
in `evaluation/evaluator.py` (torch-dependent), not here.
"""

from __future__ import annotations

import numpy as np


def counterfactual_consistency_rate(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    """
    Eq. (19):

        CCR = (1/N) * sum_i  1[ argmax P(y|x_i^(a)) == argmax P(y|x_i^(b)) ]

    Args:
        preds_a, preds_b: (N,) integer arrays of argmax predicted class
            indices for the original and counterfactual inputs.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    if preds_a.shape != preds_b.shape:
        raise ValueError(f"Shape mismatch: {preds_a.shape} vs {preds_b.shape}")
    if preds_a.ndim != 1:
        raise ValueError("Expected 1D array of class indices")
    return float(np.mean(preds_a == preds_b))


def ccr_from_logits(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Convenience wrapper: takes (N, num_classes) logits/probs directly."""
    preds_a = np.argmax(logits_a, axis=-1)
    preds_b = np.argmax(logits_b, axis=-1)
    return counterfactual_consistency_rate(preds_a, preds_b)


def demographic_parity_gap(preds: np.ndarray, groups: np.ndarray, positive_class: int = 1) -> float:
    """
    Demographic Parity Gap: max difference in positive-prediction rate
    across demographic groups.

        DPG = max_g P(pred=positive | group=g) - min_g P(pred=positive | group=g)

    The manuscript (Table 4) defines DPG only descriptively as "demographic
    parity gap across attribute groups"; this is the standard formalization
    used in the fairness literature and is what Eq. (16)-style aggregation
    is then applied to across languages.

    Args:
        preds: (N,) integer array of predicted class indices
        groups: (N,) array of group/attribute labels (e.g. demographic
            attribute a vs b, or language)
        positive_class: which class index counts as the "positive" outcome
    """
    preds = np.asarray(preds)
    groups = np.asarray(groups)
    if preds.shape != groups.shape:
        raise ValueError("preds and groups must have the same shape")
    rates = {}
    for g in np.unique(groups):
        mask = groups == g
        rates[g] = float(np.mean(preds[mask] == positive_class))
    if len(rates) < 2:
        raise ValueError("DPG requires at least 2 groups")
    return max(rates.values()) - min(rates.values())


def instance_weighted_global_metric(per_language_values: dict[str, float], per_language_n: dict[str, int]) -> float:
    """
    General form of Eq. (16)/(20):

        M_global = sum_l (n_l * M_l) / sum_l n_l

    Applies identically to F1, BTS, CCR, DPG, Leakage -- any per-language
    scalar metric aggregated with instance weighting, as specified in
    Section 5.2 ("this identical aggregation process is used for all
    fairness metrics").
    """
    if set(per_language_values) != set(per_language_n):
        raise ValueError("Language keys must match between values and counts")
    num = sum(per_language_n[l] * per_language_values[l] for l in per_language_values)
    den = sum(per_language_n.values())
    if den == 0:
        raise ValueError("Total instance count is zero")
    return num / den


def unweighted_group_average(per_language_values: dict[str, float], languages_in_group: list[str]) -> float:
    """
    Per-resource-group / per-typology averaging described in Section 5.2:
    "For per-language analysis ... unweighted averaging over languages is
    performed" -- e.g. computing the HR/MR/LR bars in Fig. 6 or the
    typology categories in Fig. 12.
    """
    vals = [per_language_values[l] for l in languages_in_group if l in per_language_values]
    if not vals:
        raise ValueError("No matching languages found in per_language_values")
    return float(np.mean(vals))


def resource_category(token_count: float) -> str:
    """
    Sec 3.1 resource categorization:
        HR: T_l > 1e9
        MR: 1e8 < T_l <= 1e9
        LR: T_l <= 1e8
    """
    if token_count > 1e9:
        return "HR"
    elif token_count > 1e8:
        return "MR"
    else:
        return "LR"
