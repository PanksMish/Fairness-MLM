"""
MFC baseline (Table 3): "Multilingual fairness-aware classification,"
Lin, He, Tang, Zhou & Yang -- "Model and Evaluation: Towards Fairness in
Multilingual Text Classification," arXiv:2303.15697 (the version cited
by the ADAPT-BTS manuscript as Lin et al. 2026, IJMLC).

Faithfully implements the paper's four-module architecture and EXACT
loss formulas (Eq. 1-5 of the source paper):

    1. Multilingual representation module: v_i = Encoder(x_i)
       (paper uses mBERT; we substitute this repo's shared
       MultilingualEncoder -- mT5 or XLM-R per configs/*.yaml -- so the
       comparison in Table 5 uses the same backbone across all methods,
       which the paper's own mBERT choice would not give us; this
       substitution is a deliberate, documented deviation for fair
       comparison, not an error.)
    2. Language fusion module: contrastive loss pulling same-label,
       different-LANGUAGE samples together (Eq. 2-3)
    3. Text debiasing module: contrastive loss pulling same-label,
       different-ATTRIBUTE samples together (Eq. 4-5)
    4. Text classification module: standard task head + cross-entropy

Total loss: L = L_task + alpha_lf * L_lf + alpha_td * L_td (the source
paper does not specify numeric values for alpha_lf/alpha_td in the
excerpt available to us; treat these as tunable hyperparameters, default
1.0 each here, matching an unweighted sum which is what Eq. 1-5 imply
when no explicit weighting is given).

The contrastive math itself (baselines/losses_core.py's
supervised_contrastive_loss) is unit-tested against hand-computed values
independent of torch; this file is the torch wrapper that computes it
inside a real forward/backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("baselines/mfc.py requires PyTorch.") from e

from model.classifier import SentimentModel
from optimization.losses import sentiment_task_loss


@dataclass
class MFCConfig:
    temperature: float = 0.1     # tau in Eq. 2/4
    alpha_lf: float = 1.0        # weight on the language-fusion loss
    alpha_td: float = 1.0        # weight on the text-debiasing loss


def supervised_contrastive_loss_torch(
    embeddings: "torch.Tensor", positive_mask: "torch.Tensor", temperature: float = 0.1,
) -> "torch.Tensor":
    """Differentiable torch mirror of losses_core.supervised_contrastive_loss
    (Eq. 2-5's shared pattern). embeddings: (N, D); positive_mask: (N, N) bool."""
    n = embeddings.size(0)
    normalized = F.normalize(embeddings, dim=-1)
    sim = (normalized @ normalized.T) / temperature

    not_self = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    sim_stable = sim - sim.max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim_stable)
    denom = (exp_sim * not_self).sum(dim=1)

    pos_mask = positive_mask & not_self
    log_probs = sim_stable - torch.log(denom).unsqueeze(1)
    per_anchor_loss = -(log_probs * pos_mask).sum(dim=1)
    n_positives = pos_mask.sum(dim=1)

    has_positives = n_positives > 0
    if not has_positives.any():
        return torch.tensor(0.0, device=embeddings.device)
    return per_anchor_loss[has_positives].mean()


def build_language_fusion_mask(labels: "torch.Tensor", language_ids: "torch.Tensor") -> "torch.Tensor":
    """T = {t : y_t = y_i, l_t != l_i}, Eq. (2)."""
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_language = language_ids.unsqueeze(0) != language_ids.unsqueeze(1)
    return same_label & diff_language


def build_debiasing_mask(labels: "torch.Tensor", attribute_ids: "torch.Tensor") -> "torch.Tensor":
    """Q = {q : y_q = y_i, s_q != s_i}, Eq. (4)."""
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_attribute = attribute_ids.unsqueeze(0) != attribute_ids.unsqueeze(1)
    return same_label & diff_attribute


def mfc_loss(
    model: SentimentModel,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    labels: "torch.Tensor",
    language_ids: "torch.Tensor",
    attribute_ids: "torch.Tensor",
    config: MFCConfig,
) -> dict[str, "torch.Tensor"]:
    """
    Computes MFC's full composite loss for one batch. Requires the batch
    to include language and (declared or detected) attribute ids for
    every sample -- unlike ADAPT-BTS, MFC does not need explicit
    counterfactual pairs (its contrastive losses operate directly on
    naturally-occurring batch diversity in language/attribute), so this
    can run on a plain SentimentDataset batch augmented with attribute
    labels, not just PairedSentimentDataset.
    """
    token_reprs, pooled = model.encoder(input_ids, attention_mask)
    logits = model.head(pooled)

    task_loss = sentiment_task_loss(logits, labels)

    lf_mask = build_language_fusion_mask(labels, language_ids)
    lf_loss = supervised_contrastive_loss_torch(pooled, lf_mask, config.temperature)

    td_mask = build_debiasing_mask(labels, attribute_ids)
    td_loss = supervised_contrastive_loss_torch(pooled, td_mask, config.temperature)

    total = task_loss + config.alpha_lf * lf_loss + config.alpha_td * td_loss

    return {
        "total_loss": total, "task_loss": task_loss,
        "language_fusion_loss": lf_loss, "text_debiasing_loss": td_loss,
        "logits": logits,
    }
