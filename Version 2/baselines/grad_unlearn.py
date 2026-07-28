"""
Grad-Unl baseline (Table 3): "Gradient-based removal of demographic
signals," Chen, Yang, Xiong, Bai, Hu, Hao, Feng, Zhou, Wu & Liu (2023),
"Fast model debias with machine unlearning," NeurIPS 2023.

I was not able to fetch this paper's exact algorithm in this session.
The general "machine unlearning for debiasing" mechanism its title
describes -- identify samples associated with a biased behavior and
apply a corrective (typically gradient-ASCENT, i.e. negative-gradient)
update on those samples to reduce the model's reliance on whatever
pattern they represent -- is implemented here using this repo's own BTS
signal to select the targets (the most natural per-sample bias score
available in this pipeline; Chen et al.'s paper likely uses a different
selection criterion specific to their setup, since it predates BTS).
Flagged as a faithful-to-the-general-mechanism, not verified-exact,
reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("baselines/grad_unlearn.py requires PyTorch.") from e

from model.classifier import SentimentModel
from optimization.losses import sentiment_task_loss
from fairness.bts_torch import compute_bts_torch


@dataclass
class GradUnlearnConfig:
    top_fraction: float = 0.10     # fraction of batch selected as unlearning targets
    unlearn_weight: float = 0.5    # scales the negative-gradient (ascent) term


def grad_unlearn_step(
    model: SentimentModel,
    input_ids: "torch.Tensor", attention_mask: "torch.Tensor", labels: "torch.Tensor",
    input_ids_cf: "torch.Tensor", attention_mask_cf: "torch.Tensor",
    config: GradUnlearnConfig,
) -> dict[str, "torch.Tensor"]:
    """
    Computes:
        L = L_task(all samples) - unlearn_weight * L_task(top-BTS subset)

    The subtraction on the high-BTS subset is the "unlearning" gradient:
    minimizing L_task normally DEscends toward the biased pattern; the
    negative term pushes gradient descent to partially UNDO fitting on
    exactly the samples where the model's predictions are most sensitive
    to the demographic transformation (highest per-sample BTS, Eq. 15),
    without a separate unlearning phase after the fact -- it's folded
    into the same optimizer step for simplicity, unlike a true two-phase
    "train then unlearn" pipeline Chen et al.'s title suggests. A
    two-phase variant would call this AFTER normal training convergence
    instead of every step; that scheduling choice is left to the caller.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_cf = model(input_ids=input_ids_cf, attention_mask=attention_mask_cf)

    task_loss = sentiment_task_loss(logits, labels)

    bts_result = compute_bts_torch(logits, logits_cf)
    per_sample_bts = bts_result.per_instance.detach()  # selection uses a detached signal

    n = per_sample_bts.size(0)
    k = max(1, int(round(config.top_fraction * n)))
    top_indices = torch.topk(per_sample_bts, k=k).indices

    unlearn_loss = sentiment_task_loss(logits[top_indices], labels[top_indices])

    total = task_loss - config.unlearn_weight * unlearn_loss
    return {
        "total_loss": total, "task_loss": task_loss, "unlearn_loss": unlearn_loss,
        "n_unlearn_targets": k, "logits": logits,
    }
