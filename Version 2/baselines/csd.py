"""
CSD baseline (Table 3): "Contrastive self-debiasing," Xu, Chen, Tang, Li,
Hu, Chu, Ren, Zheng & Lu (2025b), "Mitigating social bias in large
language models: A multi-objective approach within a multi-agent
framework," AAAI 2025 -- and the closely related contrastive-debiasing
line Li, Du, Song, Wang, Sun & Wang (2024), "Mitigating social biases of
pre-trained language models via contrastive self-debiasing with double
data augmentation," Artificial Intelligence 332.

I was not able to fetch the exact loss formula from either paper's full
text in this session (only abstracts/citations were checked). Rather
than guess at a specific numbered equation and present it as verified,
this implements the well-established GENERAL mechanism both papers'
titles and abstracts describe -- supervised contrastive learning applied
to (original, counterfactually-augmented) pairs, pulling each instance's
representation close to its own counterfactual variant while pushing it
apart from other instances in the batch (the same InfoNCE/SimCLR-style
pattern as MFC's Eq. 2/4, just with (x, x^(b)) pairs as the positive set
instead of (same-label, different-language/attribute) pairs). This is a
faithful-to-the-general-approach, not faithful-to-the-exact-published-
equations, implementation -- flagged explicitly rather than presented as
a verified reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("baselines/csd.py requires PyTorch.") from e

from baselines.mfc import supervised_contrastive_loss_torch  # reuses the same InfoNCE-style core
from model.classifier import SentimentModel
from optimization.losses import sentiment_task_loss


@dataclass
class CSDConfig:
    temperature: float = 0.1
    alpha_contrastive: float = 1.0


def csd_loss(
    model: SentimentModel,
    input_ids: "torch.Tensor", attention_mask: "torch.Tensor", labels: "torch.Tensor",
    input_ids_cf: "torch.Tensor", attention_mask_cf: "torch.Tensor",
    config: CSDConfig,
) -> dict[str, "torch.Tensor"]:
    """
    Requires paired counterfactual data (PairedSentimentDataset), unlike
    MFC. For a batch of size N, this constructs a 2N-sample batch of
    [originals; counterfactuals] and treats each original's own
    counterfactual as its sole positive -- everything else (including
    OTHER instances' counterfactuals) is a negative, which is the
    standard self-supervised contrastive setup.
    """
    _, pooled_orig = model.encoder(input_ids, attention_mask)
    _, pooled_cf = model.encoder(input_ids_cf, attention_mask_cf)

    logits = model.head(pooled_orig)
    task_loss = sentiment_task_loss(logits, labels)

    n = pooled_orig.size(0)
    combined = torch.cat([pooled_orig, pooled_cf], dim=0)  # (2N, D)

    # positive_mask[i, i+N] = True and positive_mask[i+N, i] = True (each
    # original's sole positive is its own counterfactual, and vice versa)
    positive_mask = torch.zeros(2 * n, 2 * n, dtype=torch.bool, device=pooled_orig.device)
    idx = torch.arange(n, device=pooled_orig.device)
    positive_mask[idx, idx + n] = True
    positive_mask[idx + n, idx] = True

    contrastive_loss = supervised_contrastive_loss_torch(combined, positive_mask, config.temperature)

    total = task_loss + config.alpha_contrastive * contrastive_loss
    return {"total_loss": total, "task_loss": task_loss, "contrastive_loss": contrastive_loss, "logits": logits}
