"""
Bias Transfer Score (BTS)

Implements Eq. (4), (15), (21) of the manuscript:

    Delta(x, a, b) = D_TV( P_theta(.|x^(a)), P_theta(.|x^(b)) )
                   = 0.5 * sum_y |P_theta(y|x^(a)) - P_theta(y|x^(b))|

    BTS(theta) = E_{x,a,b}[ Delta(x, a, b) ]

This module is intentionally decoupled from any specific model framework.
It operates on predictive probability distributions (already produced by a
forward pass), so it works identically whether those distributions came
from a PyTorch model, a NumPy array, or a mocked test fixture. The
PyTorch-specific wrapper (for use inside a training loop with autograd) is
provided separately in `bts_torch.py` so that this file has zero heavy
dependencies and can be unit-tested without torch installed.

No result in this file is hardcoded. Every value is computed from the
`p_a`, `p_b` arrays passed in by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _validate_distributions(p_a: np.ndarray, p_b: np.ndarray, atol: float = 1e-3) -> None:
    if p_a.shape != p_b.shape:
        raise ValueError(f"Shape mismatch: p_a={p_a.shape}, p_b={p_b.shape}")
    if p_a.ndim != 2:
        raise ValueError(f"Expected 2D array (batch, num_classes), got shape {p_a.shape}")
    for name, p in (("p_a", p_a), ("p_b", p_b)):
        if np.any(p < -1e-6) or np.any(p > 1 + 1e-6):
            raise ValueError(f"{name} contains values outside [0, 1]")
        row_sums = p.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=atol):
            bad = np.where(~np.isclose(row_sums, 1.0, atol=atol))[0]
            raise ValueError(
                f"{name} rows must sum to 1 (got {row_sums[bad][:5]} at rows {bad[:5]})"
            )


def total_variation_distance(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """
    Instance-level total variation distance, Eq. (15):

        Delta(x_i) = 0.5 * sum_y |P(y|x_i^(a)) - P(y|x_i^(b))|

    Args:
        p_a: (batch, num_classes) predictive distribution under attribute a
        p_b: (batch, num_classes) predictive distribution under attribute b

    Returns:
        (batch,) array of per-instance TV distances in [0, 1]
    """
    _validate_distributions(p_a, p_b)
    delta = 0.5 * np.abs(p_a - p_b).sum(axis=-1)
    # Numerical stability: TV distance is mathematically bounded in [0, 1]
    return np.clip(delta, 0.0, 1.0)


@dataclass
class BTSResult:
    """Container for a BTS computation, keeping per-instance data for
    downstream use (e.g. IBADR ranking, per-language aggregation)."""

    per_instance: np.ndarray          # Delta(x_i, a, b) for each instance
    mean: float                       # BTS(theta) = E[Delta]
    std: float
    n: int

    def as_dict(self) -> dict:
        return {"bts_mean": self.mean, "bts_std": self.std, "n": self.n}


def compute_bts(
    p_a: np.ndarray,
    p_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> BTSResult:
    """
    Batch/global Bias Transfer Score, Eq. (4):

        BTS(theta) = E_{x,a,b}[ D_TV(P_theta(.|x^(a)), P_theta(.|x^(b))) ]

    Args:
        p_a, p_b: (N, num_classes) predictive distributions for the original
            and counterfactual (attribute-swapped) inputs respectively.
        weights: optional (N,) instance weights for weighted expectation
            (e.g. instance-weighted aggregation across languages, Eq. 16/20).

    Returns:
        BTSResult with per-instance divergence and the aggregate score.
    """
    delta = total_variation_distance(p_a, p_b)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != delta.shape[0]:
            raise ValueError("weights must match number of instances")
        w = weights / weights.sum()
        mean = float(np.sum(w * delta))
        # Weighted variance
        std = float(np.sqrt(np.sum(w * (delta - mean) ** 2)))
    else:
        mean = float(delta.mean())
        std = float(delta.std())
    return BTSResult(per_instance=delta, mean=mean, std=std, n=delta.shape[0])


def instance_weighted_global_bts(per_language_bts: dict[str, float], per_language_n: dict[str, int]) -> float:
    """
    Eq. (16)/(20) generalized to BTS: instance-weighted aggregation across
    languages.

        BTS_global = sum_l (n_l * BTS_l) / sum_l n_l
    """
    if set(per_language_bts) != set(per_language_n):
        raise ValueError("per_language_bts and per_language_n must have the same language keys")
    num = sum(per_language_n[lang] * per_language_bts[lang] for lang in per_language_bts)
    den = sum(per_language_n.values())
    if den == 0:
        raise ValueError("Total instance count is zero")
    return num / den


def semantic_preservation_score(cos_sim: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    """
    Eq. (3): boolean mask of samples satisfying the semantic preservation
    constraint  CosSim(h(x), h(x^(b))) >= 0.85.
    """
    return cos_sim >= threshold
