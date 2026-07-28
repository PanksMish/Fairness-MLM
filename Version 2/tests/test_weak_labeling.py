import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.nrc_lexicon import LanguageLexicon
from datasets.weak_labeling import lexicon_polarity_score, build_weak_labeled_records


def _make_lexicon():
    return LanguageLexicon(
        language="en",
        positive_words={"good", "great", "happy", "love", "excellent"},
        negative_words={"bad", "terrible", "sad", "hate", "awful"},
    )


def test_lexicon_polarity_score_all_positive():
    lex = _make_lexicon()
    result = lexicon_polarity_score(["this", "is", "good", "and", "great"], lex)
    assert result.label == "positive"
    assert result.n_positive_words == 2
    assert result.n_negative_words == 0
    assert result.confidence == 1.0


def test_lexicon_polarity_score_all_negative():
    lex = _make_lexicon()
    result = lexicon_polarity_score(["this", "is", "bad", "and", "terrible"], lex)
    assert result.label == "negative"
    assert result.n_negative_words == 2


def test_lexicon_polarity_score_mixed_near_neutral():
    lex = _make_lexicon()
    result = lexicon_polarity_score(["good", "bad"], lex)
    # score = (1-1)/2 = 0.0, within neutral_band -> neutral
    assert result.label == "neutral"
    assert result.confidence == 0.0


def test_lexicon_polarity_score_no_coverage_is_neutral():
    lex = _make_lexicon()
    result = lexicon_polarity_score(["the", "quick", "fox"], lex)
    assert result.label == "neutral"
    assert result.n_tokens_covered == 0
    assert result.confidence == 0.0


def test_lexicon_polarity_score_case_insensitive():
    lex = _make_lexicon()
    result = lexicon_polarity_score(["GOOD", "Great"], lex)
    assert result.n_positive_words == 2


def test_lexicon_polarity_score_respects_neutral_band():
    lex = _make_lexicon()
    # 3 positive, 1 negative -> score = 0.5, well outside band -> positive
    result = lexicon_polarity_score(["good", "great", "happy", "bad"], lex, neutral_band=0.15)
    assert result.label == "positive"
    # with a very wide neutral band, same sentence should be neutral
    result_wide = lexicon_polarity_score(["good", "great", "happy", "bad"], lex, neutral_band=0.9)
    assert result_wide.label == "neutral"


def test_build_weak_labeled_records_schema_has_required_metadata():
    lex = _make_lexicon()
    texts = ["I feel great and happy today", "This was a terrible and awful experience"]
    records, stats = build_weak_labeled_records(texts, language="en", lexicon=lex)

    assert len(records) == 2
    for rec in records:
        assert rec["label_source"] == "weak_lexicon_nrc"
        assert rec["is_gold_label"] is False
        assert "weak_label_confidence" in rec
        assert "weak_label_coverage" in rec

    assert records[0]["label"] == "positive"
    assert records[1]["label"] == "negative"


def test_build_weak_labeled_records_filters_low_coverage():
    lex = _make_lexicon()
    texts = ["the quick brown fox jumps"]  # zero lexicon coverage
    records, stats = build_weak_labeled_records(texts, language="en", lexicon=lex, min_coverage=1)
    assert len(records) == 0
    assert stats["n_filtered_low_coverage"] == 1


def test_build_weak_labeled_records_filters_low_confidence():
    lex = _make_lexicon()
    texts = ["good bad"]  # score = 0.0, confidence = 0.0
    records, stats = build_weak_labeled_records(texts, language="en", lexicon=lex, min_confidence=0.5, min_coverage=1)
    assert len(records) == 0
    assert stats["n_filtered_low_confidence"] == 1


def test_build_weak_labeled_records_stats_summary_correct():
    lex = _make_lexicon()
    texts = ["I love this, so good", "I hate this, so bad", "the cat sat on the mat"]
    records, stats = build_weak_labeled_records(texts, language="sw", lexicon=lex, min_coverage=1)
    assert stats["n_input_texts"] == 3
    assert stats["n_output_records"] == 2  # third text has zero coverage, filtered
    assert stats["n_filtered_low_coverage"] == 1
    assert stats["label_distribution"]["positive"] == 1
    assert stats["label_distribution"]["negative"] == 1
    assert stats["language"] == "sw"
    assert stats["lexicon_size"] == 10


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
