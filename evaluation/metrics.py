"""
metrics.py
==========
Comprehensive evaluation metrics for ADAPT-BTS.

Implements all fairness and predictive performance metrics from Section 5.9:

Predictive Metrics:
  - Macro-F1 (sentiment classification)
  - Span-level F1 (NER)

Fairness Metrics:
  - BTS: Bias Transfer Score (Equation 42)
  - CCR: Counterfactual Consistency Rate (Equation 43)
  - DPG: Demographic Parity Gap (Equation 44)
  - EOD: Equalized Odds Difference (Equation 45)
  - Leakage: Linear probe accuracy on frozen encoder representations
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from seqeval.metrics import f1_score as seqeval_f1

logger = logging.getLogger(__name__)


class PredictiveMetrics:
    """
    Task performance metrics for sentiment and NER tasks.
    """

    @staticmethod
    def macro_f1(preds: List[int], labels: List[int]) -> float:
        """Macro-averaged F1 score for sentiment classification."""
        return float(f1_score(labels, preds, average="macro", zero_division=0))

    @staticmethod
    def per_class_f1(preds: List[int], labels: List[int], num_classes: int = 3) -> Dict[str, float]:
        """Per-class F1 scores."""
        scores = f1_score(labels, preds, average=None, zero_division=0)
        return {f"class_{i}_f1": float(s) for i, s in enumerate(scores[:num_classes])}

    @staticmethod
    def span_f1_ner(
        pred_tag_sequences: List[List[str]],
        gold_tag_sequences: List[List[str]],
    ) -> float:
        """
        Span-level F1 for NER evaluation (XTREME/seqeval convention).
        Uses conll evaluation for exact span matching.
        """
        try:
            return float(seqeval_f1(gold_tag_sequences, pred_tag_sequences, average="micro"))
        except Exception as e:
            logger.warning(f"seqeval F1 computation failed: {e}. Falling back to token F1.")
            flat_preds = [t for seq in pred_tag_sequences for t in seq]
            flat_gold = [t for seq in gold_tag_sequences for t in seq]
            return float(f1_score(flat_gold, flat_preds, average="micro", zero_division=0))


class FairnessMetrics:
    """
    Fairness metrics computed across demographic groups.

    All metrics follow the definitions in Section 5.9 of the paper.
    """

    def __init__(self, num_labels: int = 3):
        self.num_labels = num_labels

    def bts(
        self,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
    ) -> float:
        """
        Bias Transfer Score (Equation 42):
        BTS = E_x [(1/2) Σ_y |P_θ(y|x^(a)) - P_θ(y|x^(b))|]

        Parameters
        ----------
        probs_a : [N, C] — predicted probabilities for attribute a
        probs_b : [N, C] — predicted probabilities for attribute b
        """
        tv_distances = 0.5 * np.abs(probs_a - probs_b).sum(axis=-1)  # [N]
        return float(tv_distances.mean())

    def ccr(
        self,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
    ) -> float:
        """
        Counterfactual Consistency Rate (Equation 43):
        CCR = (1/N) Σ_i 1[ŷ^(a)_i == ŷ^(b)_i]
        """
        consistent = (preds_a == preds_b).astype(float)
        return float(consistent.mean())

    def dpg(
        self,
        preds: np.ndarray,
        dem_attrs: List[str],
        positive_class: int = 2,
    ) -> float:
        """
        Demographic Parity Gap (Equation 44):
        DPG = max_{d1,d2} |P(ŷ=pos|d1) - P(ŷ=pos|d2)|
        """
        groups = sorted(set(dem_attrs))
        if len(groups) < 2:
            return 0.0

        rates = {}
        for g in groups:
            idx = [i for i, a in enumerate(dem_attrs) if a == g]
            if not idx:
                continue
            g_preds = preds[idx]
            rates[g] = float((g_preds == positive_class).mean())

        if len(rates) < 2:
            return 0.0

        rate_vals = list(rates.values())
        return float(max(
            abs(rate_vals[i] - rate_vals[j])
            for i in range(len(rate_vals))
            for j in range(i + 1, len(rate_vals))
        ))

    def eod(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        dem_attrs: List[str],
        positive_class: int = 2,
    ) -> float:
        """
        Equalized Odds Difference (Equation 45):
        EOD = max_{d1,d2} (|TPR_d1 - TPR_d2| + |FPR_d1 - FPR_d2|)
        """
        groups = sorted(set(dem_attrs))
        if len(groups) < 2:
            return 0.0

        group_stats = {}
        for g in groups:
            idx = [i for i, a in enumerate(dem_attrs) if a == g]
            if not idx:
                continue
            g_preds = preds[idx]
            g_labels = labels[idx]
            pos_mask = g_labels == positive_class
            neg_mask = ~pos_mask
            tpr = float((g_preds[pos_mask] == positive_class).mean()) if pos_mask.any() else 0.0
            fpr = float((g_preds[neg_mask] == positive_class).mean()) if neg_mask.any() else 0.0
            group_stats[g] = (tpr, fpr)

        if len(group_stats) < 2:
            return 0.0

        stat_vals = list(group_stats.values())
        return float(max(
            abs(stat_vals[i][0] - stat_vals[j][0]) + abs(stat_vals[i][1] - stat_vals[j][1])
            for i in range(len(stat_vals))
            for j in range(i + 1, len(stat_vals))
        ))

    def representation_leakage(
        self,
        representations: np.ndarray,
        dem_labels: np.ndarray,
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> float:
        """
        Linear probe accuracy measuring demographic information leakage.

        A logistic regression classifier is trained on frozen encoder
        representations to predict demographic attributes. Higher accuracy
        indicates more demographic information is encoded (higher leakage).

        Method (Section 5.9):
        "The leakage of representation is quantified by a linear probe that
        is trained on frozen encoder representations."

        Parameters
        ----------
        representations : [N, H] — frozen encoder hidden states
        dem_labels : [N] — demographic attribute labels (int-encoded)

        Returns
        -------
        leakage : float — accuracy of linear probe (0=no leakage, 1=full leakage)
        """
        # Remove samples with unknown/neutral demographic
        valid_mask = dem_labels != -1
        if valid_mask.sum() < 20:
            logger.warning("Too few labeled samples for leakage probe.")
            return 0.5  # chance level

        X = representations[valid_mask]
        y = dem_labels[valid_mask]

        # Split 80/20
        n = len(X)
        split = int(0.8 * n)
        perm = np.random.permutation(n)
        X_train, X_test = X[perm[:split]], X[perm[split:]]
        y_train, y_test = y[perm[:split]], y[perm[split:]]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic regression (linear probe)
        probe = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        )
        try:
            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)
            leakage = float(accuracy_score(y_test, y_pred))
        except Exception as e:
            logger.warning(f"Leakage probe failed: {e}")
            leakage = 0.5

        return leakage


class MultilingualEvaluationReport:
    """
    Aggregates per-language evaluation results into a structured report.

    Computes:
      - Global weighted averages across all 101 languages
      - Resource-stratified statistics (HR / MR / LR)
      - Typological category statistics
    """

    def __init__(self, language_tiers: Dict[str, str]):
        """
        Parameters
        ----------
        language_tiers : Dict[str, str]
            Mapping from language code to "HR" / "MR" / "LR".
        """
        self.language_tiers = language_tiers
        self.per_language_results: Dict[str, Dict[str, float]] = {}

    def add_language_result(self, lang: str, metrics: Dict[str, float]):
        """Record evaluation results for a single language."""
        self.per_language_results[lang] = metrics

    def global_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean ± std over all recorded languages.
        Matches the format of Table 3 in the paper.
        """
        if not self.per_language_results:
            return {}

        all_metrics: Dict[str, List[float]] = {}
        for lang_metrics in self.per_language_results.values():
            for metric, value in lang_metrics.items():
                all_metrics.setdefault(metric, []).append(value)

        return {
            metric: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
            for metric, vals in all_metrics.items()
        }

    def stratified_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute per-resource-tier statistics.
        Returns nested dict: tier → metric → {mean, std}.
        """
        tier_groups: Dict[str, Dict[str, List[float]]] = {
            "HR": {}, "MR": {}, "LR": {}
        }

        for lang, metrics in self.per_language_results.items():
            tier = self.language_tiers.get(lang, "LR")
            if tier not in tier_groups:
                continue
            for metric, value in metrics.items():
                tier_groups[tier].setdefault(metric, []).append(value)

        return {
            tier: {
                metric: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
                for metric, vals in metrics_dict.items()
            }
            for tier, metrics_dict in tier_groups.items()
        }

    def delta_f1(self, baseline_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute per-language ΔF1 = F1_ADAPT - F1_baseline.
        Used for Figure 14 in the paper.
        """
        deltas = {}
        for lang, metrics in self.per_language_results.items():
            base = baseline_results.get(lang, {}).get("macro_f1", 0.0)
            ours = metrics.get("macro_f1", 0.0)
            deltas[lang] = ours - base
        return deltas

    def relative_improvement(
        self, baseline_results: Dict[str, Dict[str, float]], metric: str
    ) -> float:
        """
        Compute relative improvement (Equation 46):
        ΔM = (M_ADAPT - M_baseline) / M_baseline × 100%
        """
        adapt_val = np.mean([m.get(metric, 0) for m in self.per_language_results.values()])
        base_val = np.mean([m.get(metric, 0) for m in baseline_results.values()])
        if base_val == 0:
            return 0.0
        return float((adapt_val - base_val) / abs(base_val) * 100)
