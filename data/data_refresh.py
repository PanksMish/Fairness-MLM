"""
data_refresh.py
===============
Iterative Bias-Aware Data Refresh (IBADR) module.

Implements the outer-loop refinement mechanism described in Section 4.5
of the ADAPT-BTS paper.

Core idea:
  As model parameters θ evolve during training, counterfactual pairs that
  were initially well-aligned may exhibit increased distributional divergence
  (measured by per-sample BTS). IBADR periodically identifies the top ρ%
  highest-divergence samples and regenerates them using the current
  counterfactual generation pipeline.

Algorithm 3 (paper):
  for every refresh interval K:
      compute per-sample BTS scores
      select top ρ% divergence samples
      regenerate counterfactual pairs
      replace outdated samples in D_aug

This mechanism allows training data and model parameters to co-evolve
throughout optimization, maintaining alignment between the augmented
dataset and the current model's representation space.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class IterativeBiasAwareDataRefresh:
    """
    IBADR: manages the periodic regeneration of high-divergence counterfactual
    samples during training.

    Parameters
    ----------
    counterfactual_generator : CounterfactualGenerator
        Generator used to produce new counterfactual variants.
    refresh_interval_epochs : int
        K — number of epochs between refresh cycles (paper: K=2).
    top_divergence_fraction : float
        ρ — fraction of samples to regenerate at each cycle (paper: ρ=0.15).
    max_refresh_cycles : int
        Maximum number of refresh cycles (paper: 5).
    """

    def __init__(
        self,
        counterfactual_generator,
        refresh_interval_epochs: int = 2,
        top_divergence_fraction: float = 0.15,
        max_refresh_cycles: int = 5,
    ):
        self.generator = counterfactual_generator
        self.refresh_interval_epochs = refresh_interval_epochs
        self.top_divergence_fraction = top_divergence_fraction
        self.max_refresh_cycles = max_refresh_cycles

        self._refresh_count = 0
        self._last_refresh_epoch = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_refresh(self, current_epoch: int) -> bool:
        """
        Returns True if a data refresh should be triggered at the current epoch.

        Conditions:
          - current_epoch > 0 (don't refresh before training starts)
          - enough epochs have passed since the last refresh
          - haven't exceeded max refresh cycles
        """
        if self._refresh_count >= self.max_refresh_cycles:
            return False
        if current_epoch <= 0:
            return False
        epochs_since_last = current_epoch - self._last_refresh_epoch
        return epochs_since_last >= self.refresh_interval_epochs

    def refresh(
        self,
        aug_data: Dict[str, List],
        model,
        device: str,
        current_epoch: int,
    ) -> Dict[str, List]:
        """
        Execute a refresh cycle:
          1. Compute per-sample BTS for all augmented counterfactual pairs.
          2. Select top ρ% by BTS score (highest divergence).
          3. Regenerate counterfactual variants for selected indices.
          4. Replace old samples in aug_data.

        Parameters
        ----------
        aug_data : Dict[str, List]
            Dict with keys: "texts", "labels", "languages", "dem_attrs",
            "cf_texts" (counterfactual texts), "cf_attrs" (cf attributes).
        model : MultilingualModel
            Current model (for computing per-sample BTS).
        device : str
        current_epoch : int

        Returns
        -------
        Updated aug_data dict.
        """
        if not self.should_refresh(current_epoch):
            return aug_data

        logger.info(
            f"[IBADR] Epoch {current_epoch}: starting refresh cycle "
            f"{self._refresh_count + 1}/{self.max_refresh_cycles}"
        )

        n_samples = len(aug_data["texts"])
        n_refresh = max(1, int(n_samples * self.top_divergence_fraction))

        # Step 1: Compute per-sample BTS scores
        per_sample_bts = self._compute_per_sample_bts(aug_data, model, device)

        # Step 2: Select top ρ% highest divergence indices
        sorted_indices = np.argsort(per_sample_bts)[::-1]  # descending
        refresh_indices = sorted_indices[:n_refresh].tolist()

        logger.info(
            f"[IBADR] Refreshing {n_refresh} / {n_samples} samples "
            f"(avg BTS of top ρ%: {np.mean(per_sample_bts[sorted_indices[:n_refresh]]):.4f})"
        )

        # Step 3: Regenerate counterfactual pairs for selected indices
        n_regenerated = 0
        for idx in refresh_indices:
            original_text = aug_data["texts"][idx]
            label = aug_data["labels"][idx]
            lang = aug_data["languages"][idx]
            dem_attr = aug_data["dem_attrs"][idx]

            if dem_attr not in ("male", "female"):
                continue

            cf = self.generator.generate_counterfactual(
                text=original_text,
                original_attribute=dem_attr,
                language=lang,
                label=label,
            )

            if cf.accepted and cf.counterfactual_text != aug_data.get("cf_texts", [""] * n_samples)[idx]:
                # Replace with new counterfactual
                if "cf_texts" in aug_data:
                    aug_data["cf_texts"][idx] = cf.counterfactual_text
                    aug_data["cf_attrs"][idx] = cf.counterfactual_attribute
                n_regenerated += 1

        self._refresh_count += 1
        self._last_refresh_epoch = current_epoch

        logger.info(
            f"[IBADR] Refresh cycle {self._refresh_count} complete. "
            f"Regenerated {n_regenerated} samples."
        )

        return aug_data

    def reset(self):
        """Reset refresh state (e.g., for a new training run)."""
        self._refresh_count = 0
        self._last_refresh_epoch = -1

    # ------------------------------------------------------------------
    # Per-Sample BTS Computation (Equation 29)
    # ------------------------------------------------------------------

    def _compute_per_sample_bts(
        self, aug_data: Dict[str, List], model, device: str
    ) -> np.ndarray:
        """
        Compute per-sample BTS scores for all samples that have a
        counterfactual pair.

        BTS_i = (1/2) Σ_y |P_θ(y|x^(a)_i) - P_θ(y|x^(b)_i)|  [Equation 29]

        Falls back to random scores if no counterfactual texts are available
        (e.g., early in training before full augmentation).
        """
        n_samples = len(aug_data["texts"])

        if "cf_texts" not in aug_data or len(aug_data["cf_texts"]) != n_samples:
            # No counterfactual texts stored yet → random BTS as proxy
            logger.debug("[IBADR] No cf_texts available, using random BTS proxy.")
            return np.random.uniform(0.2, 0.8, size=n_samples)

        model.eval()
        per_sample_bts = np.zeros(n_samples, dtype=np.float32)

        # Process in mini-batches
        batch_size = 32
        tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None

        if tokenizer is None:
            logger.warning("[IBADR] Model has no tokenizer attribute, using random BTS proxy.")
            return np.random.uniform(0.2, 0.8, size=n_samples)

        import unicodedata
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                orig_texts = [
                    unicodedata.normalize("NFKC", aug_data["texts"][i])
                    for i in range(start, end)
                ]
                cf_texts = [
                    unicodedata.normalize("NFKC", aug_data["cf_texts"][i])
                    for i in range(start, end)
                ]

                try:
                    # Tokenize both sets
                    orig_enc = tokenizer(
                        orig_texts,
                        max_length=256,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)

                    cf_enc = tokenizer(
                        cf_texts,
                        max_length=256,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)

                    # Forward pass
                    orig_logits = model(
                        input_ids=orig_enc["input_ids"],
                        attention_mask=orig_enc["attention_mask"],
                    )
                    cf_logits = model(
                        input_ids=cf_enc["input_ids"],
                        attention_mask=cf_enc["attention_mask"],
                    )

                    # Convert to probabilities
                    orig_probs = torch.softmax(orig_logits, dim=-1)  # [B, C]
                    cf_probs = torch.softmax(cf_logits, dim=-1)       # [B, C]

                    # Total variation distance (Equation 29)
                    tv_dist = 0.5 * torch.abs(orig_probs - cf_probs).sum(dim=-1)  # [B]
                    per_sample_bts[start:end] = tv_dist.cpu().numpy()

                except Exception as e:
                    logger.debug(f"[IBADR] BTS computation failed for batch [{start}:{end}]: {e}")
                    per_sample_bts[start:end] = np.random.uniform(0.2, 0.8, size=end - start)

        model.train()
        return per_sample_bts


class AugmentedDataStore:
    """
    In-memory store for the augmented dataset D_aug.

    Tracks original samples, counterfactual variants, and per-sample
    metadata required for IBADR refresh operations.

    This is used as the mutable dataset during training; at each IBADR
    cycle, the counterfactual entries are updated in-place.
    """

    def __init__(self):
        self.texts: List[str] = []
        self.cf_texts: List[str] = []
        self.labels: List[int] = []
        self.languages: List[str] = []
        self.dem_attrs: List[str] = []
        self.cf_attrs: List[str] = []
        self._has_cf: List[bool] = []  # whether each sample has a valid CF pair

    def add_original(
        self, text: str, label: int, language: str, dem_attr: str
    ):
        """Add an original sample without a counterfactual."""
        self.texts.append(text)
        self.cf_texts.append(text)  # placeholder
        self.labels.append(label)
        self.languages.append(language)
        self.dem_attrs.append(dem_attr)
        self.cf_attrs.append(dem_attr)
        self._has_cf.append(False)

    def add_counterfactual_pair(
        self,
        original_text: str,
        cf_text: str,
        label: int,
        language: str,
        original_attr: str,
        cf_attr: str,
    ):
        """Add an original + counterfactual pair."""
        self.texts.append(original_text)
        self.cf_texts.append(cf_text)
        self.labels.append(label)
        self.languages.append(language)
        self.dem_attrs.append(original_attr)
        self.cf_attrs.append(cf_attr)
        self._has_cf.append(True)

    def to_dict(self) -> Dict[str, List]:
        return {
            "texts": self.texts,
            "cf_texts": self.cf_texts,
            "labels": self.labels,
            "languages": self.languages,
            "dem_attrs": self.dem_attrs,
            "cf_attrs": self.cf_attrs,
        }

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_counterfactual_pairs(self) -> int:
        return sum(self._has_cf)

    @property
    def augmentation_rate(self) -> float:
        total = len(self)
        if total == 0:
            return 0.0
        return self.n_counterfactual_pairs / total
