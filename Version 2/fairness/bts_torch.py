"""
Differentiable PyTorch implementation of the Bias Transfer Score, for use
inside the training loop (Algorithm 2, line 6: "Estimate fairness metric
BTS_t on batch B_t").

This mirrors `bias_transfer_score.py` exactly but keeps everything as
torch tensors so gradients flow back through BTS into the model
parameters (required because the composite loss in Eq. 5/12/18 is
L = L_task + lambda * BTS(theta), and theta must receive gradient from
both terms).

Requires: torch. Not imported by the rest of the `fairness` package so
that pure-math components remain testable without a torch/GPU
environment.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "bts_torch.py requires PyTorch. Install with `pip install torch` "
        "(see requirements.txt). The pure-NumPy BTS implementation in "
        "bias_transfer_score.py has no such dependency."
    ) from e


@dataclass
class TorchBTSResult:
    per_instance: torch.Tensor  # (batch,) differentiable
    mean: torch.Tensor          # scalar, differentiable -> feeds into L_total


def total_variation_distance_torch(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """
    Delta(x, a, b) = 0.5 * sum_y |P(y|x^a) - P(y|x^b)|,  Eq. (15)

    Args:
        logits_a, logits_b: (batch, num_classes) raw model logits for the
            original and counterfactual inputs. Softmax is applied here so
            the caller can pass either sentiment logits or, for NER,
            per-token logits reshaped to (batch * seq_len, num_labels).
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"Shape mismatch: {logits_a.shape} vs {logits_b.shape}")
    p_a = F.softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    delta = 0.5 * torch.abs(p_a - p_b).sum(dim=-1)
    return torch.clamp(delta, 0.0, 1.0)


def compute_bts_torch(logits_a: torch.Tensor, logits_b: torch.Tensor) -> TorchBTSResult:
    """Differentiable batch BTS, Eq. (4), for use as the fairness term in
    Eq. (5)/(12)/(18): L = L_task + lambda_t * (BTS - tau)."""
    per_instance = total_variation_distance_torch(logits_a, logits_b)
    return TorchBTSResult(per_instance=per_instance, mean=per_instance.mean())


def counterfactual_forward_pass(model, batch_original, batch_counterfactual):
    """
    Runs the (1 + k) forward passes described in Sec. 3.4 / Eq. (6):

        cost = O((1 + k) * C_forward)

    Here k=1 (one counterfactual per original instance, the common case
    used throughout the manuscript's experiments; Sec 3.4 notes k <= 2 in
    general). For k=2, call this twice with the second counterfactual
    batch and average the resulting per_instance BTS terms.

    Args:
        model: a callable taking a batch dict (e.g. HF-style
            {"input_ids": ..., "attention_mask": ...}) and returning logits.
        batch_original: dict of tensors for the original inputs x^(a)
        batch_counterfactual: dict of tensors for x^(b)

    Returns:
        (logits_a, logits_b, TorchBTSResult)
    """
    logits_a = model(**batch_original)
    logits_b = model(**batch_counterfactual)
    bts = compute_bts_torch(logits_a, logits_b)
    return logits_a, logits_b, bts
