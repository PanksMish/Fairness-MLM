import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from evaluation.metrics import (
    macro_f1, macro_f1_from_logits, extract_spans, EntitySpan,
    span_f1, span_f1_from_ids,
)


# ---------------------------------------------------------------------------
# Macro-F1
# ---------------------------------------------------------------------------

def test_macro_f1_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y_true, y_true) == 1.0


def test_macro_f1_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert macro_f1(y_true, y_pred) == 0.0


def test_macro_f1_from_logits_matches_argmax():
    logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.0, 0.1, 0.9]])
    y_true = np.array([1, 0, 2])
    assert macro_f1_from_logits(logits, y_true) == 1.0


# ---------------------------------------------------------------------------
# extract_spans (IOB2)
# ---------------------------------------------------------------------------

def test_extract_spans_single_entity():
    tags = ["O", "B-PER", "I-PER", "O"]
    spans = extract_spans(tags)
    assert spans == [EntitySpan("PER", 1, 3)]


def test_extract_spans_multiple_entities():
    tags = ["B-PER", "I-PER", "O", "B-LOC", "O"]
    spans = extract_spans(tags)
    assert set(spans) == {EntitySpan("PER", 0, 2), EntitySpan("LOC", 3, 4)}


def test_extract_spans_no_entities():
    tags = ["O", "O", "O"]
    assert extract_spans(tags) == []


def test_extract_spans_adjacent_different_types():
    tags = ["B-PER", "B-ORG"]  # back-to-back B- tags: two separate single-token spans
    spans = extract_spans(tags)
    assert set(spans) == {EntitySpan("PER", 0, 1), EntitySpan("ORG", 1, 2)}


def test_extract_spans_trailing_entity_flushed():
    tags = ["O", "B-LOC", "I-LOC"]  # entity runs to the end of sequence
    spans = extract_spans(tags)
    assert spans == [EntitySpan("LOC", 1, 3)]


def test_extract_spans_malformed_i_tag_without_b_starts_new_span():
    # seqeval-default behavior: an I-X with no preceding B-X/I-X of the
    # same type is treated as starting a new span
    tags = ["O", "I-PER", "I-PER", "O"]
    spans = extract_spans(tags)
    assert spans == [EntitySpan("PER", 1, 3)]


def test_extract_spans_type_change_within_i_tags_splits_span():
    tags = ["B-PER", "I-ORG"]  # I-ORG doesn't match current PER span -> new span
    spans = extract_spans(tags)
    assert set(spans) == {EntitySpan("PER", 0, 1), EntitySpan("ORG", 1, 2)}


# ---------------------------------------------------------------------------
# span_f1
# ---------------------------------------------------------------------------

def test_span_f1_perfect_match():
    true_tags = [["O", "B-PER", "I-PER", "O"]]
    pred_tags = [["O", "B-PER", "I-PER", "O"]]
    result = span_f1(true_tags, pred_tags)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_span_f1_no_predictions():
    true_tags = [["B-PER", "O"]]
    pred_tags = [["O", "O"]]
    result = span_f1(true_tags, pred_tags)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.n_true_spans == 1
    assert result.n_pred_spans == 0


def test_span_f1_partial_boundary_mismatch_counts_as_wrong():
    # true: PER spans tokens [0,2); pred: PER spans tokens [0,1) only ->
    # NOT a match (exact boundary required, standard CoNLL behavior)
    true_tags = [["B-PER", "I-PER", "O"]]
    pred_tags = [["B-PER", "O", "O"]]
    result = span_f1(true_tags, pred_tags)
    assert result.n_correct == 0
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_span_f1_wrong_type_counts_as_wrong():
    true_tags = [["B-PER", "O"]]
    pred_tags = [["B-ORG", "O"]]
    result = span_f1(true_tags, pred_tags)
    assert result.n_correct == 0


def test_span_f1_mixed_batch_aggregates_correctly():
    true_tags = [["B-PER", "O"], ["B-LOC", "I-LOC"]]
    pred_tags = [["B-PER", "O"], ["B-LOC", "O"]]  # second sentence: boundary wrong
    result = span_f1(true_tags, pred_tags)
    assert result.n_true_spans == 2
    assert result.n_pred_spans == 2
    assert result.n_correct == 1
    assert abs(result.precision - 0.5) < 1e-9
    assert abs(result.recall - 0.5) < 1e-9


def test_span_f1_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        span_f1([["B-PER", "O"]], [["B-PER"]])


def test_span_f1_from_ids_with_ignore_index():
    id_to_tag = {0: "O", 1: "B-PER", 2: "I-PER"}
    true_ids = [[-100, 1, 2, -100]]  # e.g. [CLS] B-PER I-PER [SEP]
    pred_ids = [[-100, 1, 2, -100]]
    result = span_f1_from_ids(true_ids, pred_ids, id_to_tag)
    assert result.f1 == 1.0


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
