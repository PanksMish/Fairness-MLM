"""
Representation Leakage (Table 4):

    "Leakage is evaluated by means of a linear probing model which is
    trained on demographic labels from encoder-freezed representations."

Standard linear-probe methodology: freeze the task model's encoder,
extract pooled representations h(x) for a held-out set of examples with
known demographic attribute labels, train a simple linear classifier to
predict the attribute FROM the representation, and report that
classifier's accuracy (or macro-F1) on a held-out split. High probe
accuracy = the representation still encodes demographic information
("leakage") even after fairness-aware training; low probe accuracy
(near chance) = the representation has been successfully scrubbed of
that signal.

Uses sklearn's LogisticRegression, available in this sandbox, so the
probe training/evaluation logic here is real and testable with
synthetic representation vectors -- it just needs real h(x) vectors from
model/encoder.py at actual-run time, which this sandbox can't produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class LeakageResult:
    probe_accuracy: float
    probe_macro_f1: float
    chance_accuracy: float     # 1 / num_classes, for comparison
    n_train: int
    n_test: int


def compute_leakage(
    representations: np.ndarray,
    attribute_labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 1000,
) -> LeakageResult:
    """
    Args:
        representations: (N, hidden_dim) pooled encoder representations
            h(x), extracted with the task model's encoder FROZEN (no
            gradient should have flowed into the encoder from this
            probe -- that's the caller's responsibility when extracting
            `representations`, e.g. via `torch.no_grad()`).
        attribute_labels: (N,) integer-encoded demographic attribute
            labels (e.g. 0="m", 1="f").

    Returns:
        LeakageResult with probe accuracy/F1 on a held-out split, plus
        the chance-level baseline for context (a probe scoring near
        chance indicates low leakage; scoring well above chance
        indicates the representation still linearly encodes the
        attribute).
    """
    representations = np.asarray(representations)
    attribute_labels = np.asarray(attribute_labels)
    if representations.shape[0] != attribute_labels.shape[0]:
        raise ValueError("representations and attribute_labels must have the same number of rows")

    num_classes = len(np.unique(attribute_labels))
    if num_classes < 2:
        raise ValueError("Leakage probe requires at least 2 distinct attribute classes")

    X_train, X_test, y_train, y_test = train_test_split(
        representations, attribute_labels, test_size=test_size, random_state=seed,
        stratify=attribute_labels,
    )

    probe = LogisticRegression(max_iter=max_iter, random_state=seed)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    return LeakageResult(
        probe_accuracy=float(accuracy_score(y_test, y_pred)),
        probe_macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        chance_accuracy=1.0 / num_classes,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def leakage_above_chance(result: LeakageResult, margin: float = 0.05) -> bool:
    """Convenience check: does the probe do meaningfully better than
    chance (by more than `margin`)? Used for a quick pass/fail read
    rather than requiring the caller to compare floats manually."""
    return result.probe_accuracy > result.chance_accuracy + margin
