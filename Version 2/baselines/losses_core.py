"""
Pure-NumPy reference implementations of the baseline loss functions,
kept separate from their torch counterparts (baselines/*.py) so the
underlying math can be unit-tested without a torch/GPU environment, the
same pattern used throughout this repo (fairness/bias_transfer_score.py
vs bts_torch.py).

MFC's contrastive losses are implemented here EXACTLY per Lin, He, Tang,
Zhou & Yang (2023/2026), "Model and Evaluation: Towards Fairness in
Multilingual Text Classification," arXiv:2303.15697 (published version
cited by the ADAPT-BTS manuscript as Lin et al. 2026, IJMLC), Eq. (2)-(5):

    L_lf_i = -sum_{t in T} log( exp(sim(v_i,v_t)/tau) / sum_{k != i} exp(sim(v_i,v_k)/tau) )
    T = {t : y_t = y_i, l_t != l_i, t != i}   (same label, different language)

    L_td_i = -sum_{q in Q} log( exp(sim(v_i,v_q)/tau) / sum_{k != i} exp(sim(v_i,v_k)/tau) )
    Q = {q : y_q = y_i, s_q != s_i, q != i}   (same label, different sensitive attribute)

Both are instances of the same "supervised contrastive" pattern (pull
same-label samples that differ along one nuisance axis together, using
all other batch samples as the softmax denominator), so
`supervised_contrastive_loss` below implements the pattern once and both
L_lf and L_td are just different choices of the positive-set mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, N) pairwise cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def supervised_contrastive_loss(
    embeddings: np.ndarray,
    positive_mask: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """
    Generic form of MFC's Eq. (2)-(5): for each anchor i with at least
    one positive, accumulate -log( exp(sim(i,pos)/tau) / sum_{k!=i} exp(sim(i,k)/tau) )
    over all its positives, then average over anchors that had >=1
    positive (anchors with none contribute 0, matching an empty sum in
    the original formulation).

    Args:
        embeddings: (N, D) batch of representations v_i
        positive_mask: (N, N) boolean; positive_mask[i, j] = True iff j
            is a positive for anchor i (e.g. same label + different
            language for L_lf, or same label + different attribute for
            L_td). Diagonal is ignored regardless of its value.
        temperature: tau in the manuscript's notation
    """
    n = embeddings.shape[0]
    if positive_mask.shape != (n, n):
        raise ValueError(f"positive_mask must be ({n},{n}), got {positive_mask.shape}")

    sim = cosine_similarity_matrix(embeddings) / temperature
    # Numerical stability: subtract row-max before exponentiating
    sim_stable = sim - sim.max(axis=1, keepdims=True)
    exp_sim = np.exp(sim_stable)

    not_self = ~np.eye(n, dtype=bool)
    denom = (exp_sim * not_self).sum(axis=1)  # (N,) sum over k != i

    pos_mask = positive_mask & not_self
    log_probs = sim_stable - np.log(denom)[:, None]  # log(exp(sim)/denom) per (i,k)

    per_anchor_loss = -(log_probs * pos_mask).sum(axis=1)  # sum over positives
    n_positives = pos_mask.sum(axis=1)

    has_positives = n_positives > 0
    if not has_positives.any():
        return 0.0
    return float(per_anchor_loss[has_positives].mean())


def build_language_fusion_positive_mask(labels: np.ndarray, languages: np.ndarray) -> np.ndarray:
    """T = {t : y_t = y_i, l_t != l_i}, Eq. (2)'s positive set."""
    same_label = labels[:, None] == labels[None, :]
    diff_language = languages[:, None] != languages[None, :]
    return same_label & diff_language


def build_debiasing_positive_mask(labels: np.ndarray, attributes: np.ndarray) -> np.ndarray:
    """Q = {q : y_q = y_i, s_q != s_i}, Eq. (4)'s positive set."""
    same_label = labels[:, None] == labels[None, :]
    diff_attribute = attributes[:, None] != attributes[None, :]
    return same_label & diff_attribute


# ---------------------------------------------------------------------------
# MAGNET proxy: inverse-token-frequency loss reweighting.
#
# Real MAGNET (Ahia et al. 2024) is a BYTE-LEVEL tokenization architecture
# with learned per-script boundary predictors -- fundamentally different
# from the subword-tokenizer (SentencePiece/mT5, BPE/XLM-R) models used
# throughout this repo. It cannot be faithfully reproduced by a
# training-loop-only change. What CAN be done in this pipeline is a
# same-SPIRIT proxy: since MAGNET's goal is equitable effective sequence
# length / compute across languages, inverse-frequency loss reweighting
# approximates "give under-tokenized-in-practice languages more training
# signal" without touching the tokenizer. This is explicitly a proxy, not
# MAGNET, and is labeled as such everywhere it's used.
# ---------------------------------------------------------------------------

def language_inverse_frequency_weights(language_token_counts: dict[str, float]) -> dict[str, float]:
    """
    Per-language weight proportional to 1/T_l, normalized to mean 1.0
    across languages (so the proxy reweights relative emphasis without
    changing the overall loss scale).
    """
    langs = list(language_token_counts.keys())
    counts = np.array([language_token_counts[l] for l in langs], dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("All token counts must be positive")
    inv = 1.0 / counts
    normalized = inv / inv.mean()
    return dict(zip(langs, normalized.tolist()))


# ---------------------------------------------------------------------------
# Grad-Unl proxy: selects which samples to apply a gradient-ASCENT
# (unlearning) step to, based on per-sample bias divergence.
# ---------------------------------------------------------------------------

def unlearning_selection_mask(per_sample_bts: np.ndarray, top_fraction: float = 0.1) -> np.ndarray:
    """
    Selects the top `top_fraction` highest-BTS-divergence samples in a
    batch as unlearning targets, mirroring the "fast model debias with
    machine unlearning" idea (Chen et al. 2023): identify the samples
    most associated with biased predictions and apply a negative
    (ascent) gradient step on them, rather than the whole dataset.
    """
    n = len(per_sample_bts)
    k = max(1, int(np.ceil(top_fraction * n)))
    order = np.argsort(-per_sample_bts)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask
