"""
Iterative Bias-Aware Data Refresh (IBADR), Algorithm 3:

    1: Set refresh interval K and selection ratio rho
    2: for every interval K do
    3:     Compute per-sample divergence BTS(x_i) for current data
    4:     Select top rho% samples with highest divergence
    5:     Regenerate counterfactuals for selected samples
    6:     Replace or augment entries in D_aug
    7: end for
    8: return updated dataset D_aug

This module implements the scheduling and selection logic (steps 1-4, 6)
in a framework-agnostic way. Step 5 ("regenerate counterfactuals") calls
back into `counterfactual_generation.py`, which the caller injects as a
function -- this keeps IBADR decoupled from any specific generation
backend (lexicon-based / cross-lingual embedding / pivot-translation, per
Sec 4.1's resource-tiered strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Any

import numpy as np


@dataclass
class IBADRConfig:
    refresh_interval: int = 500   # K: refresh every K training steps
    selection_ratio: float = 0.10  # rho: top 10% highest-divergence samples


@dataclass
class RefreshEvent:
    step: int
    n_selected: int
    selected_indices: np.ndarray
    mean_bts_selected: float
    mean_bts_all: float


class IBADRScheduler:
    """
    Tracks training step count and per-sample BTS scores, and triggers
    selective counterfactual regeneration on the configured interval.
    """

    def __init__(self, config: IBADRConfig | None = None):
        self.config = config or IBADRConfig()
        self.events: list[RefreshEvent] = []

    def should_refresh(self, step: int) -> bool:
        return step > 0 and step % self.config.refresh_interval == 0

    def select_top_divergence(self, per_sample_bts: np.ndarray) -> np.ndarray:
        """
        Algorithm 3, line 4: select top rho% samples with highest
        divergence. Returns the selected indices, sorted descending by
        BTS score.
        """
        per_sample_bts = np.asarray(per_sample_bts)
        n = len(per_sample_bts)
        k = max(1, int(np.ceil(self.config.selection_ratio * n)))
        # argsort descending, take top k
        order = np.argsort(-per_sample_bts)
        return order[:k]

    def refresh_step(
        self,
        step: int,
        dataset: Sequence[Any],
        per_sample_bts: np.ndarray,
        regenerate_fn: Callable[[Any], Any],
        replace: bool = True,
    ) -> tuple[list[Any], RefreshEvent | None]:
        """
        Runs one potential refresh cycle.

        Args:
            step: current global training step
            dataset: the current D_aug, as a sequence of samples (any type
                the caller's `regenerate_fn` understands -- e.g. dicts
                with (x, y, a) tuples)
            per_sample_bts: (len(dataset),) array of per-sample BTS
                (instance-level divergence, Eq. 15) computed by the caller
                from the current model's predictions
            regenerate_fn: callable that takes one sample and returns a
                newly-generated counterfactual sample (delegates to
                fairness/counterfactual_generation.py in the full
                pipeline)
            replace: if True, replace the selected entries in-place
                (Algorithm 3, line 6: "Replace or augment"); if False,
                append the newly generated samples instead of replacing.

        Returns:
            (possibly updated dataset, RefreshEvent or None if no refresh
            occurred this step)
        """
        if not self.should_refresh(step):
            return list(dataset), None

        if len(dataset) != len(per_sample_bts):
            raise ValueError(
                f"dataset length {len(dataset)} must match per_sample_bts length {len(per_sample_bts)}"
            )

        selected = self.select_top_divergence(per_sample_bts)
        new_dataset = list(dataset)
        for idx in selected:
            regenerated = regenerate_fn(dataset[idx])
            if replace:
                new_dataset[idx] = regenerated
            else:
                new_dataset.append(regenerated)

        event = RefreshEvent(
            step=step,
            n_selected=len(selected),
            selected_indices=selected,
            mean_bts_selected=float(np.mean(per_sample_bts[selected])),
            mean_bts_all=float(np.mean(per_sample_bts)),
        )
        self.events.append(event)
        return new_dataset, event
