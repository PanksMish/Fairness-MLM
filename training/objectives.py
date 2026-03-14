"""
objectives.py
=============
Training objectives for ADAPT-BTS.

Implements:
  - Task loss (cross-entropy for sentiment / NER)
  - BTS fairness regularization term
  - Composite Lagrangian-relaxed objective (Equation 26):
        L(θ, λ_t) = L_task(θ) + λ_t · (BTS(θ) - τ)

Also provides utilities for:
  - Multi-objective Pareto analysis (utility U1 and fairness U2, Equations 15-16)
  - Gradient clipping
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SentimentTaskLoss(nn.Module):
    """
    Cross-entropy loss for 3-class multilingual sentiment classification.
    Supports label smoothing for regularization.
    """

    def __init__(self, num_labels: int = 3, label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, C]
        labels : [B]
        """
        return self.loss_fn(logits, labels)


class NERTaskLoss(nn.Module):
    """
    Cross-entropy loss for token-level NER, ignoring padded positions.
    """

    def __init__(self, num_labels: int = 7, label_smoothing: float = 0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, L, C]
        labels : [B, L]  (with -100 for ignored positions)
        """
        B, L, C = logits.shape
        return self.loss_fn(logits.view(B * L, C), labels.view(B * L))


class AdaptBTSObjective(nn.Module):
    """
    Full ADAPT-BTS training objective.

    L(θ, λ_t) = L_task(θ) + λ_t · (BTS(θ) − τ)    [Equation 26]

    This is the Lagrangian relaxation of the constrained problem:
        min_θ L_task(θ)  s.t. BTS(θ) ≤ τ    [Equation 25]

    The fairness weight λ_t is supplied externally by the FAPC controller.

    Parameters
    ----------
    task : str
        "sentiment" or "ner"
    num_labels : int
    tau : float
        Fairness tolerance τ.
    label_smoothing : float
    """

    def __init__(
        self,
        task: str = "sentiment",
        num_labels: int = 3,
        tau: float = 0.40,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.task = task
        self.tau = tau
        self.num_labels = num_labels

        # Task loss
        if task == "sentiment":
            self.task_loss = SentimentTaskLoss(num_labels, label_smoothing)
        elif task == "ner":
            self.task_loss = NERTaskLoss(num_labels, label_smoothing)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(
        self,
        logits_orig: torch.Tensor,
        labels: torch.Tensor,
        logits_cf: Optional[torch.Tensor],
        lambda_t: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite training loss.

        Parameters
        ----------
        logits_orig : logits for original samples
        labels : ground-truth labels
        logits_cf : logits for counterfactual samples (or None)
        lambda_t : current λ from FAPC controller

        Returns
        -------
        losses : dict with keys "total", "task", "bts", "constraint"
        """
        # === Task Loss ===
        l_task = self.task_loss(logits_orig, labels)

        # === BTS Fairness Loss ===
        bts_value = torch.tensor(0.0, device=logits_orig.device, requires_grad=False)

        if logits_cf is not None and lambda_t > 0.0:
            bts_value = self._compute_bts(logits_orig, logits_cf)

        # === Lagrangian Constraint: BTS(θ) - τ ===
        constraint = bts_value - self.tau

        # === Composite Loss ===
        total = l_task + lambda_t * constraint

        return {
            "total": total,
            "task": l_task,
            "bts": bts_value,
            "constraint": constraint,
        }

    def _compute_bts(
        self, logits_orig: torch.Tensor, logits_cf: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute differentiable BTS = E[D_TV(P^(a), P^(b))].

        BTS needs to be differentiable w.r.t. model parameters θ so that
        ∇_θ BTS can be computed for the gradient update (Equation 19).
        """
        if self.task == "ner":
            B, L, C = logits_orig.shape
            probs_orig = F.softmax(logits_orig.view(B * L, C), dim=-1)
            probs_cf = F.softmax(logits_cf.view(B * L, C), dim=-1)
        else:
            probs_orig = F.softmax(logits_orig, dim=-1)
            probs_cf = F.softmax(logits_cf, dim=-1)

        # Total variation distance
        tv = 0.5 * torch.abs(probs_orig - probs_cf).sum(dim=-1)
        return tv.mean()

    def compute_multi_objective(
        self,
        logits_orig: torch.Tensor,
        labels: torch.Tensor,
        logits_cf: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute utility metrics U1 (task) and U2 (fairness) from
        Equations 15-16 for Pareto analysis.

        U1(θ) = -L_task(θ)
        U2(θ) = -BTS(θ)
        """
        l_task = self.task_loss(logits_orig, labels)
        bts_value = torch.tensor(0.0, device=logits_orig.device)
        if logits_cf is not None:
            bts_value = self._compute_bts(logits_orig, logits_cf)

        return {
            "U1": -l_task,   # Equation 15
            "U2": -bts_value,  # Equation 16
            "task_loss": l_task,
            "bts": bts_value,
        }
