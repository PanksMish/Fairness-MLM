"""
Task losses L_task(theta), the first term of Eq. (5)/(12)/(18).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("optimization/losses.py requires PyTorch.") from e


def sentiment_task_loss(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """Standard cross-entropy over the 3-class sentiment label space."""
    return F.cross_entropy(logits, labels)


def ner_task_loss(logits: "torch.Tensor", label_ids: "torch.Tensor", ignore_index: int = -100) -> "torch.Tensor":
    """
    Per-token cross-entropy, ignoring padded positions and non-first
    subwords (label_ids == -100, per datasets/tokenizer.py's alignment
    convention).

    Args:
        logits: (batch, seq_len, num_labels)
        label_ids: (batch, seq_len)
    """
    num_labels = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, num_labels), label_ids.reshape(-1), ignore_index=ignore_index,
    )
