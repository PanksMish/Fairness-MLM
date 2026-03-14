"""
bias_transfer_score.py
======================
Bias Transfer Score (BTS) — the core fairness metric for ADAPT-BTS.

Defined in Section 3.7 of the paper as the expected total variation
divergence between predictive distributions of counterfactual pairs:

    BTS(θ) = E_{x~X} E_{a,b∈D} [ (1/2) Σ_y |P^(a)_θ(y|x) - P^(b)_θ(y|x)| ]
           = E_{x,a,b} [ D_TV(P^(a)_θ, P^(b)_θ) ]    [Equation 12-13]

Properties (from paper Section 3.7):
  - BTS(θ) ∈ [0, 1]  (bounded, normalized)
  - Equivalent to expected total variation distance
  - Proposition 1: |P(ŷ=1|a) - P(ŷ=1|b)| ≤ BTS(θ)
    → minimizing BTS reduces Demographic Parity Gap and Equalized Odds

The module also provides:
  - Per-sample BTS computation (Equation 29) for IBADR
  - Batch BTS computation for training loss
  - Analytical utilities for the Lagrangian update
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BiasTransferScore(nn.Module):
    """
    Differentiable implementation of the Bias Transfer Score.

    This module computes BTS as a differentiable loss term that can be
    directly incorporated into the multi-objective training objective:

        L(θ) = L_task(θ) + λ · BTS(θ)    [Equation 18]

    The gradient ∇_θ BTS is computed automatically via PyTorch autograd,
    enabling feedback-controlled optimization (Section 4.4).

    Parameters
    ----------
    num_labels : int
        Number of output classes |Y|.
    reduction : str
        "mean" (default) or "sum" over the batch dimension.
    """

    def __init__(self, num_labels: int = 3, reduction: str = "mean"):
        super().__init__()
        self.num_labels = num_labels
        self.reduction = reduction

    def forward(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BTS for a batch of counterfactual pairs.

        Parameters
        ----------
        logits_a : torch.Tensor of shape [B, C]
            Logits for original demographic attribute a.
        logits_b : torch.Tensor of shape [B, C]
            Logits for counterfactual attribute b.

        Returns
        -------
        bts : torch.Tensor (scalar)
            Batch BTS value.
        """
        # Convert logits to probability distributions P^(a) and P^(b)
        probs_a = F.softmax(logits_a, dim=-1)  # [B, C]
        probs_b = F.softmax(logits_b, dim=-1)  # [B, C]

        return self.compute_from_probs(probs_a, probs_b)

    def compute_from_probs(
        self,
        probs_a: torch.Tensor,
        probs_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BTS directly from probability distributions.

        Total variation distance: D_TV(P, Q) = (1/2) Σ_y |P(y) - Q(y)|
        BTS = E[D_TV(P^(a), P^(b))]    [Equation 12]

        Parameters
        ----------
        probs_a : [B, C]
        probs_b : [B, C]

        Returns
        -------
        bts : scalar tensor
        """
        # Element-wise absolute difference
        abs_diff = torch.abs(probs_a - probs_b)  # [B, C]

        # Total variation distance per sample
        tv_per_sample = 0.5 * abs_diff.sum(dim=-1)  # [B]

        # BTS = E[D_TV] (Equation 13)
        if self.reduction == "mean":
            return tv_per_sample.mean()
        elif self.reduction == "sum":
            return tv_per_sample.sum()
        else:
            return tv_per_sample  # [B], per-sample

    def compute_per_sample(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-sample BTS values (Equation 29, used by IBADR).

        Returns
        -------
        bts_per_sample : [B] tensor
        """
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        bts_module = BiasTransferScore(num_labels=self.num_labels, reduction="none")
        return bts_module.compute_from_probs(probs_a, probs_b)

    @staticmethod
    def theoretical_upper_bound(bts_value: float) -> float:
        """
        Proposition 1 (paper Section 4.6):
        |P(ŷ=1|a) - P(ŷ=1|b)| ≤ BTS(θ)

        This returns the theoretical upper bound on Demographic Parity Gap
        given the current BTS value.
        """
        return float(bts_value)

    @staticmethod
    def check_bounded(bts_value: float, tol: float = 1e-6) -> bool:
        """Verify BTS ∈ [0, 1] (Equation 14)."""
        return (-tol <= bts_value <= 1.0 + tol)


class CompositeObjective(nn.Module):
    """
    Multi-objective training loss combining task loss and BTS fairness term.

    L(θ) = L_task(θ) + λ · BTS(θ)    [Equation 18]

    The gradient of this composite loss is:
        ∇_θ L = ∇_θ L_task + λ · ∇_θ BTS    [Equation 19]

    Parameters
    ----------
    num_labels : int
    task : str
        "sentiment" or "ner"
    label_smoothing : float
        Optional label smoothing for the task loss.
    """

    def __init__(
        self,
        num_labels: int = 3,
        task: str = "sentiment",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.task = task
        self.bts_module = BiasTransferScore(num_labels=num_labels)

        # Task-specific loss
        if task == "sentiment":
            self.task_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif task == "ner":
            # NER: ignore -100 (padding / subword continuation) labels
            self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logits_cf: Optional[torch.Tensor] = None,
        lambda_fairness: float = 0.0,
        tau: float = 0.4,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Parameters
        ----------
        logits : [B, C] or [B, L, C]
            Predictions for original samples.
        labels : [B] or [B, L]
            Ground-truth task labels.
        logits_cf : [B, C] or [B, L, C], optional
            Predictions for counterfactual samples (required if λ > 0).
        lambda_fairness : float
            Current λ_t from the feedback controller.
        tau : float
            Fairness tolerance τ (for logging).

        Returns
        -------
        Dict with keys: "total", "task", "bts", "lambda"
        """
        # Task loss
        if self.task == "ner":
            # NER: reshape for CrossEntropyLoss [B*L, C] vs [B*L]
            B, L, C = logits.shape
            task_loss = self.task_loss_fn(logits.view(B * L, C), labels.view(B * L))
        else:
            task_loss = self.task_loss_fn(logits, labels)

        # BTS fairness loss (only computed if λ > 0 and counterfactuals available)
        bts_loss = torch.tensor(0.0, device=logits.device)
        if lambda_fairness > 0.0 and logits_cf is not None:
            if self.task == "ner":
                B, L, C = logits.shape
                bts_loss = self.bts_module(
                    logits.view(B * L, C),
                    logits_cf.view(B * L, C),
                )
            else:
                bts_loss = self.bts_module(logits, logits_cf)

        # Combined loss: L = L_task + λ · (BTS - τ)  [Equation 26, Lagrangian form]
        fairness_constraint = bts_loss - tau
        total_loss = task_loss + lambda_fairness * fairness_constraint

        return {
            "total": total_loss,
            "task": task_loss,
            "bts": bts_loss,
            "lambda": torch.tensor(lambda_fairness),
            "constraint": fairness_constraint,
        }


class FairnessMetricsComputer:
    """
    Computes all fairness-related metrics from a set of predictions.

    Metrics computed (Section 5.9 of paper):
      - BTS: Bias Transfer Score (distributional divergence)
      - CCR: Counterfactual Consistency Rate
      - DPG: Demographic Parity Gap
      - EOD: Equalized Odds Difference
    """

    def __init__(self, num_labels: int = 3):
        self.num_labels = num_labels
        self.bts_module = BiasTransferScore(num_labels=num_labels, reduction="mean")

    def compute_bts(
        self,
        probs_a: torch.Tensor,
        probs_b: torch.Tensor,
    ) -> float:
        """Compute batch BTS from probability distributions."""
        with torch.no_grad():
            bts = self.bts_module.compute_from_probs(probs_a, probs_b)
        return float(bts.item())

    def compute_ccr(
        self,
        preds_a: torch.Tensor,
        preds_b: torch.Tensor,
    ) -> float:
        """
        Counterfactual Consistency Rate (Equation 43):
        CCR = (1/N) Σ_i 1[argmax P(y|x^(a)_i) == argmax P(y|x^(b)_i)]

        Parameters
        ----------
        preds_a : [N] — predicted class for attribute a
        preds_b : [N] — predicted class for attribute b
        """
        consistent = (preds_a == preds_b).float().mean()
        return float(consistent.item())

    def compute_dpg(
        self,
        preds: torch.Tensor,
        dem_attrs: List[str],
        positive_class: int = 2,
    ) -> float:
        """
        Demographic Parity Gap (Equation 44):
        DPG = max_{d1,d2} |P(ŷ=1|d1) - P(ŷ=1|d2)|

        Uses the positive class prediction rate per demographic group.
        """
        groups = list(set(dem_attrs))
        if len(groups) < 2:
            return 0.0

        group_rates = {}
        preds_np = preds.cpu().numpy()
        for group in groups:
            group_mask = [i for i, a in enumerate(dem_attrs) if a == group]
            if len(group_mask) == 0:
                continue
            group_preds = preds_np[group_mask]
            group_rates[group] = (group_preds == positive_class).mean()

        if len(group_rates) < 2:
            return 0.0

        rates = list(group_rates.values())
        dpg = max(abs(rates[i] - rates[j]) for i in range(len(rates)) for j in range(i + 1, len(rates)))
        return float(dpg)

    def compute_eod(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        dem_attrs: List[str],
        positive_class: int = 2,
    ) -> float:
        """
        Equalized Odds Difference (Equation 45):
        EOD = max_{d1,d2} (|TPR_d1 - TPR_d2| + |FPR_d1 - FPR_d2|)
        """
        groups = list(set(dem_attrs))
        if len(groups) < 2:
            return 0.0

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        group_metrics = {}
        for group in groups:
            group_mask = [i for i, a in enumerate(dem_attrs) if a == group]
            if len(group_mask) == 0:
                continue
            g_preds = preds_np[group_mask]
            g_labels = labels_np[group_mask]

            # Binary TPR / FPR for the positive class
            positive_mask = g_labels == positive_class
            negative_mask = ~positive_mask

            tpr = (g_preds[positive_mask] == positive_class).mean() if positive_mask.any() else 0.0
            fpr = (g_preds[negative_mask] == positive_class).mean() if negative_mask.any() else 0.0
            group_metrics[group] = (tpr, fpr)

        if len(group_metrics) < 2:
            return 0.0

        metrics = list(group_metrics.values())
        eod = max(
            abs(metrics[i][0] - metrics[j][0]) + abs(metrics[i][1] - metrics[j][1])
            for i in range(len(metrics))
            for j in range(i + 1, len(metrics))
        )
        return float(eod)

    def compute_all(
        self,
        probs_a: torch.Tensor,
        probs_b: torch.Tensor,
        labels: torch.Tensor,
        dem_attrs: List[str],
    ) -> Dict[str, float]:
        """
        Compute all fairness metrics in a single call.

        Returns
        -------
        Dict with keys: "bts", "ccr", "dpg", "eod"
        """
        preds_a = probs_a.argmax(dim=-1)
        preds_b = probs_b.argmax(dim=-1)

        return {
            "bts": self.compute_bts(probs_a, probs_b),
            "ccr": self.compute_ccr(preds_a, preds_b),
            "dpg": self.compute_dpg(preds_a, dem_attrs),
            "eod": self.compute_eod(preds_a, labels, dem_attrs),
        }
