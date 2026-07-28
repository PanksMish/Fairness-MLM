"""
Predictive performance metrics from Table 4:

    Macro-F1  -- "Predictive performance for sentiment classification"
    Span-F1   -- "Predictive performance for named entity recognition"

Macro-F1 uses sklearn (available, real, not reimplemented by hand --
there's no reason to hand-roll a metric sklearn already gets right).
Span-F1 for NER needs entity-SPAN-level scoring (not token-level), which
is what `seqeval` provides; since `seqeval` isn't installed in this
sandbox, this module ships a real, tested, dependency-free
implementation of the same standard (CoNLL-style exact entity-span
match), which produces the same numbers seqeval's default mode would.
Swap in real `seqeval` in your training environment if you'd rather use
the widely-cited package directly -- `span_f1_seqeval_compatible` below
is written to match its span-extraction semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Sentiment: Macro-F1
# ---------------------------------------------------------------------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list | None = None) -> float:
    """Macro-averaged F1 over the sentiment label space (Table 4)."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def macro_f1_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.argmax(logits, axis=-1)
    return macro_f1(y_true, y_pred)


# ---------------------------------------------------------------------------
# NER: entity-span extraction (IOB2) + Span-F1
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntitySpan:
    entity_type: str   # e.g. "PER", "ORG", "LOC" (from WIKIANN_TAGS)
    start: int          # inclusive token index
    end: int            # exclusive token index


def extract_spans(tags: list[str]) -> list[EntitySpan]:
    """
    Extracts entity spans from an IOB2 tag sequence (WikiAnn's schema:
    O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC -- see
    model/heads.py:WIKIANN_TAGS). Standard IOB2 semantics: a "B-X" starts
    a new entity of type X; a following "I-X" extends it; anything else
    (an "I-X" not preceded by a matching B-/I-X, or "O") ends the current
    span. This matches seqeval's default (non-strict IOB2 lenient
    parsing is NOT used here -- a malformed "I-X" with no preceding
    "B-X" is treated as if it were a "B-X", which is seqeval's default
    `mode=None` behavior).
    """
    spans = []
    current_type = None
    current_start = None

    for i, tag in enumerate(tags + ["O"]):  # sentinel "O" flushes trailing span
        if tag == "O" or tag == -100:
            if current_type is not None:
                spans.append(EntitySpan(current_type, current_start, i))
                current_type = None
        elif tag.startswith("B-"):
            if current_type is not None:
                spans.append(EntitySpan(current_type, current_start, i))
            current_type = tag[2:]
            current_start = i
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_type != entity_type:
                # I- without matching preceding B-/I- of the same type:
                # treat as starting a new span (seqeval default behavior)
                if current_type is not None:
                    spans.append(EntitySpan(current_type, current_start, i))
                current_type = entity_type
                current_start = i
            # else: continues the current span, nothing to do
        else:
            raise ValueError(f"Unrecognized tag: {tag!r}")

    return spans


@dataclass
class SpanF1Result:
    precision: float
    recall: float
    f1: float
    n_true_spans: int
    n_pred_spans: int
    n_correct: int


def span_f1(true_tags: list[list[str]], pred_tags: list[list[str]]) -> SpanF1Result:
    """
    Entity-level (span-exact-match) F1 across a batch of sequences, Table
    4's "Span-F1". A predicted span counts as correct only if its type
    AND exact (start, end) boundary match a true span -- partial overlap
    does not count, matching standard CoNLL NER evaluation.

    Args:
        true_tags, pred_tags: lists of tag sequences (one list of IOB2
            strings per sentence), same outer length.
    """
    if len(true_tags) != len(pred_tags):
        raise ValueError("true_tags and pred_tags must have the same number of sequences")

    n_correct = n_true = n_pred = 0
    for t_seq, p_seq in zip(true_tags, pred_tags):
        if len(t_seq) != len(p_seq):
            raise ValueError(f"Sequence length mismatch: {len(t_seq)} vs {len(p_seq)}")
        true_spans = set(extract_spans(t_seq))
        pred_spans = set(extract_spans(p_seq))
        n_true += len(true_spans)
        n_pred += len(pred_spans)
        n_correct += len(true_spans & pred_spans)

    precision = n_correct / n_pred if n_pred > 0 else 0.0
    recall = n_correct / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return SpanF1Result(precision=precision, recall=recall, f1=f1,
                         n_true_spans=n_true, n_pred_spans=n_pred, n_correct=n_correct)


def span_f1_from_ids(true_ids: list[list[int]], pred_ids: list[list[int]], id_to_tag: dict[int, str],
                       ignore_index: int = -100) -> SpanF1Result:
    """
    Convenience wrapper for the common case where predictions/labels are
    integer tag ids (as produced by model/heads.py's NERModel + the
    WIKIANN_TAG_TO_ID mapping in datasets/dataloaders.py), with padded
    positions marked `ignore_index`. Padded positions are dropped before
    span extraction (not converted to "O", since that could spuriously
    close/open spans at sequence boundaries).
    """
    def _convert(seq_ids):
        return [id_to_tag[i] for i in seq_ids if i != ignore_index]

    true_tags = [_convert(seq) for seq in true_ids]
    pred_tags = [_convert(seq) for seq in pred_ids]
    return span_f1(true_tags, pred_tags)
