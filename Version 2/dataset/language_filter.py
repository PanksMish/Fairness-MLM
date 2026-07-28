"""
Language verification and resource-tier bookkeeping used during dataset
building (the "Language verification" step of the preprocessing pipeline,
and the HR/MR/LR categorization of Sec. 3.1).

Language identification itself needs a real model (the manuscript
specifies FastText language detection). We do NOT vendor a language-ID
model here -- instead this module defines a small Protocol so any backend
(fasttext's `lid.176.bin`, `langdetect`, `py3langid`, a HF pipeline, etc.)
can be plugged in without changing the filtering logic. This keeps the
filtering logic itself (which IS deterministic and testable) decoupled
from the model dependency (which is not available in this sandbox).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


class LanguageDetector(Protocol):
    def __call__(self, text: str) -> tuple[str, float]:
        """Return (predicted_language_code, confidence)."""
        ...


@dataclass
class LanguageFilterConfig:
    min_confidence: float = 0.7


def filter_by_declared_language(
    records: Iterable[dict],
    detector: LanguageDetector,
    lang_field: str = "language",
    text_field: str = "text",
    config: LanguageFilterConfig | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Verifies that each record's declared language matches what the
    detector predicts from its text, at or above `min_confidence`.
    Mismatches are common in scraped multilingual corpora (e.g. CC100
    documents mis-tagged by crawl metadata) and are exactly what this
    step is meant to catch, per the pipeline description.

    Args:
        detector: any callable implementing LanguageDetector. In
            production this should wrap a real FastText/langid model;
            for tests we inject a trivial stub (see tests/).

    Returns:
        (kept, rejected) -- two lists of records.
    """
    config = config or LanguageFilterConfig()
    kept, rejected = [], []
    for rec in records:
        declared = rec.get(lang_field)
        predicted, confidence = detector(rec[text_field])
        if predicted == declared and confidence >= config.min_confidence:
            kept.append(rec)
        else:
            rec_with_reason = dict(rec)
            rec_with_reason["_reject_reason"] = f"predicted={predicted} (conf={confidence:.2f}), declared={declared}"
            rejected.append(rec_with_reason)
    return kept, rejected


# ---------------------------------------------------------------------------
# Resource-tier bookkeeping (Sec 3.1): HR / MR / LR by token count.
# Kept here (in addition to evaluation/fairness_metrics.resource_category,
# which is the canonical single source of truth) only as a thin
# re-export so dataset-building code doesn't need to import from
# `evaluation/` and create a layering violation.
# ---------------------------------------------------------------------------

def resource_category(token_count: float) -> str:
    """HR: T_l > 1e9, MR: 1e8 < T_l <= 1e9, LR: T_l <= 1e8 (Sec. 3.1)."""
    if token_count > 1e9:
        return "HR"
    elif token_count > 1e8:
        return "MR"
    else:
        return "LR"


def make_stub_detector(fixed_prediction: str = None, fixed_confidence: float = 1.0) -> LanguageDetector:
    """
    Trivial detector for tests/dry-runs: either always predicts a fixed
    language, or (if fixed_prediction is None) echoes back whatever the
    caller's text hints at via a `lang:XX` prefix convention used only in
    tests. NOT for production use -- swap in a real FastText/langid
    model via the same LanguageDetector signature.
    """
    def _detect(text: str) -> tuple[str, float]:
        if fixed_prediction is not None:
            return fixed_prediction, fixed_confidence
        if text.startswith("lang:"):
            code, _, rest = text.partition(" ")
            return code.split(":", 1)[1], fixed_confidence
        return "unk", 0.0

    return _detect
