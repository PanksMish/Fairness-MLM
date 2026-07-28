"""
Optimizer/scheduler factories matching Sec 5.2:

    "Training is done using 4xA100 GPUs, BF16 precision, batch size of
    128, and learning rate of 5e-5."

These are just thin, testable-by-inspection wrappers around
torch.optim.AdamW and HF's get_linear_schedule_with_warmup -- there is no
novel math here (unlike fairness_controller.py's lambda update), so
correctness rests on using the standard library calls correctly rather
than on anything worth separately unit-testing without a live model.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch.optim import AdamW
except ImportError as e:  # pragma: no cover
    raise ImportError("optimization/optimizer.py requires PyTorch.") from e


def build_optimizer(model, learning_rate: float = 5e-5, weight_decay: float = 0.01) -> "AdamW":
    """Sec 5.2: lr=5e-5. Standard weight decay exclusion for bias/LayerNorm
    params (not specified by the manuscript, but standard practice for
    transformer fine-tuning; omitting it is not a manuscript-deviation
    since the paper doesn't specify either way)."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def build_scheduler(optimizer, num_training_steps: int, num_warmup_steps: Optional[int] = None,
                     warmup_ratio: float = 0.06):
    """Linear warmup + linear decay, a standard transformer fine-tuning
    schedule. The manuscript doesn't specify a warmup ratio, so this uses
    the common 6% default (used e.g. in RoBERTa fine-tuning recipes) --
    treat as a tunable hyperparameter, not a manuscript-derived value."""
    from transformers import get_linear_schedule_with_warmup
    if num_warmup_steps is None:
        num_warmup_steps = int(warmup_ratio * num_training_steps)
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    )
