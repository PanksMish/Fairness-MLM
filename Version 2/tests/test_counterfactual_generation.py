import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairness.counterfactual_generation import (
    LexiconSubstitutor,
    EmbeddingAlignmentTransform,
    PivotTranslationTransform,
    TieredGenerationStrategy,
    LexiconAttributeDetector,
    CounterfactualDataEngine,
    CounterfactualEngineConfig,
)
from fairness.augmentation import build_augmented_dataset
from fairness.semantic_validation import HeuristicGrammarChecker


# ---------------------------------------------------------------------------
# LexiconSubstitutor (HR tier) -- real, no mocks needed
# ---------------------------------------------------------------------------

GENDER_DICT = {
    "m": {"f": {"he": "she", "his": "her", "man": "woman", "himself": "herself"}},
    "f": {"m": {"she": "he", "her": "his", "woman": "man", "herself": "himself"}},
}


def test_lexicon_substitutor_basic_swap():
    sub = LexiconSubstitutor(GENDER_DICT)
    result = sub("He gave his book to the man.", "m", "f")
    assert result == "She gave her book to the woman."


def test_lexicon_substitutor_preserves_capitalization():
    sub = LexiconSubstitutor(GENDER_DICT)
    result = sub("HE was tall.", "m", "f")
    assert "SHE" in result


def test_lexicon_substitutor_word_boundary_safe():
    sub = LexiconSubstitutor(GENDER_DICT)
    # "he" should NOT match inside "the" or "hero"
    result = sub("The hero was there.", "m", "f")
    assert result is None  # no whole-word "he" present, so no match at all


def test_lexicon_substitutor_returns_none_when_no_match():
    sub = LexiconSubstitutor(GENDER_DICT)
    result = sub("They went to the store.", "m", "f")
    assert result is None


def test_lexicon_substitutor_returns_none_for_unknown_attribute_pair():
    sub = LexiconSubstitutor(GENDER_DICT)
    result = sub("He was there.", "m", "nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# EmbeddingAlignmentTransform (MR tier) -- mocked embedding space
# ---------------------------------------------------------------------------

class MockEmbeddingSpace:
    """Tiny hand-built embedding space: "he"/"she" are the only vectors,
    on opposite sides of a 1D line, so nearest_neighbor is trivial and
    deterministic -- enough to test the substitution LOGIC without a
    real cross-lingual embedding model."""

    def __init__(self):
        self.vectors = {"he": [1.0], "she": [-1.0]}

    def get_vector(self, word, language):
        return self.vectors.get(word)

    def nearest_neighbor(self, vector, language, exclude):
        best, best_dist = None, float("inf")
        for w, v in self.vectors.items():
            if w in exclude:
                continue
            dist = abs(v[0] - vector[0])
            if dist < best_dist:
                best, best_dist = w, dist
        return best


def test_embedding_alignment_transform_swaps_seed_word():
    space = MockEmbeddingSpace()
    seeds = {"m": ["he"], "f": ["she"]}
    transform = EmbeddingAlignmentTransform(space, seeds)
    result = transform("he runs fast", "m", "f")
    assert result == "she runs fast"


def test_embedding_alignment_transform_none_when_no_seed_present():
    space = MockEmbeddingSpace()
    seeds = {"m": ["he"], "f": ["she"]}
    transform = EmbeddingAlignmentTransform(space, seeds)
    result = transform("they run fast", "m", "f")
    assert result is None


# ---------------------------------------------------------------------------
# PivotTranslationTransform (LR tier) -- mocked translator
# ---------------------------------------------------------------------------

def mock_translator(text, src_lang, tgt_lang):
    """Fake MT: 'sw:X' <-> 'en:X' round trip, deterministic and reversible
    so we can verify the pivot->substitute->back-translate pipeline."""
    if src_lang == "sw" and tgt_lang == "en":
        return text.replace("sw:", "en:")
    if src_lang == "en" and tgt_lang == "sw":
        return text.replace("en:", "sw:")
    return text


def test_pivot_translation_transform_full_roundtrip():
    pivot_sub = LexiconSubstitutor({"m": {"f": {"en:he": "en:she"}}})
    transform = PivotTranslationTransform(mock_translator, pivot_sub, pivot_lang="en")
    result = transform("sw:he runs", "m", "f", source_lang="sw")
    assert result == "sw:she runs"


def test_pivot_translation_transform_none_when_pivot_substitution_fails():
    pivot_sub = LexiconSubstitutor({"m": {"f": {"en:he": "en:she"}}})
    transform = PivotTranslationTransform(mock_translator, pivot_sub, pivot_lang="en")
    result = transform("sw:they run", "m", "f", source_lang="sw")
    assert result is None


# ---------------------------------------------------------------------------
# TieredGenerationStrategy dispatch
# ---------------------------------------------------------------------------

def test_tiered_strategy_dispatches_hr_by_token_count():
    hr_op = LexiconSubstitutor(GENDER_DICT)
    mr_op = lambda text, a, b: "MR_CALLED"
    lr_op = lambda text, a, b, source_lang: "LR_CALLED"
    strategy = TieredGenerationStrategy(hr_op, mr_op, lr_op, token_counts={"en": 2e9, "sw": 5e7})

    result = strategy.generate("He was there.", "m", "f", language="en")
    assert result == "She was there."


def test_tiered_strategy_dispatches_lr_by_token_count():
    hr_op = lambda text, a, b: "HR_CALLED"
    mr_op = lambda text, a, b: "MR_CALLED"
    lr_op = lambda text, a, b, source_lang: f"LR_CALLED_{source_lang}"
    strategy = TieredGenerationStrategy(hr_op, mr_op, lr_op, token_counts={"sw": 5e7})

    result = strategy.generate("text", "m", "f", language="sw")
    assert result == "LR_CALLED_sw"


def test_tiered_strategy_raises_on_unknown_language():
    strategy = TieredGenerationStrategy(
        lambda t, a, b: None, lambda t, a, b: None, lambda t, a, b, source_lang: None,
        token_counts={"en": 2e9},
    )
    import pytest
    with pytest.raises(ValueError):
        strategy.generate("text", "m", "f", language="xx")


# ---------------------------------------------------------------------------
# LexiconAttributeDetector
# ---------------------------------------------------------------------------

def test_attribute_detector_finds_seed_word():
    detector = LexiconAttributeDetector({"m": ["he", "him"], "f": ["she", "her"]})
    assert detector("He walked home.") == "m"
    assert detector("She walked home.") == "f"


def test_attribute_detector_returns_none_when_no_seeds_present():
    detector = LexiconAttributeDetector({"m": ["he"], "f": ["she"]})
    assert detector("They walked home.") is None


# ---------------------------------------------------------------------------
# Full engine: Algorithm 1 end-to-end with a real lexicon substitutor and a
# trivial mock encoder function (identity-based "embedding")
# ---------------------------------------------------------------------------

def mock_encoder_fn(text: str):
    """A deterministic fake h(x): counts of a few marker words, giving a
    small vector that differs meaningfully between 'he'-text and
    'she'-text so cosine similarity isn't trivially 1.0 or 0.0."""
    words = text.lower().split()
    return [
        words.count("he") + words.count("his"),
        words.count("she") + words.count("her"),
        len(words),
    ]


def _build_engine(gamma=0.3):
    detector = LexiconAttributeDetector({"m": ["he", "his"], "f": ["she", "her"]})
    hr_op = LexiconSubstitutor(GENDER_DICT)
    strategy = TieredGenerationStrategy(
        hr_op, lambda t, a, b: None, lambda t, a, b, source_lang: None,
        token_counts={"en": 2e9},
    )
    return CounterfactualDataEngine(
        detector=detector,
        strategy=strategy,
        encoder_fn=mock_encoder_fn,
        grammar_checker=HeuristicGrammarChecker(),
        config=CounterfactualEngineConfig(gamma=gamma, alpha=1.0, beta=0.5),
        rng_seed=0,
    )


def test_engine_generate_one_produces_accepted_candidate():
    engine = _build_engine(gamma=-10.0)  # very lax threshold so it's accepted
    candidate = engine.generate_one("He gave his book to the man.", "en", all_attributes=["m", "f"])
    assert candidate is not None
    assert candidate.attribute_from == "m"
    assert candidate.attribute_to == "f"
    assert candidate.candidate_text == "She gave her book to the woman."
    assert candidate.accepted


def test_engine_generate_one_returns_none_when_no_attribute_detected():
    engine = _build_engine()
    candidate = engine.generate_one("They went shopping.", "en", all_attributes=["m", "f"])
    assert candidate is None


def test_engine_generate_one_rejects_when_score_below_gamma():
    engine = _build_engine(gamma=100.0)  # impossible threshold
    candidate = engine.generate_one("He gave his book to the man.", "en", all_attributes=["m", "f"])
    assert candidate is not None
    assert not candidate.accepted


def test_engine_uses_declared_attribute_when_given():
    engine = _build_engine(gamma=-10.0)
    # text has no lexical marker the detector would catch on its own for "f"->"m"... use declared attribute override
    candidate = engine.generate_one(
        "She gave her book to the woman.", "en", all_attributes=["m", "f"], attribute_from="f",
    )
    assert candidate is not None
    assert candidate.attribute_from == "f"
    assert candidate.attribute_to == "m"


# ---------------------------------------------------------------------------
# build_augmented_dataset: Eq. (10)
# ---------------------------------------------------------------------------

def test_build_augmented_dataset_adds_accepted_counterfactuals():
    engine = _build_engine(gamma=-10.0)
    dataset = [
        {"text": "He gave his book to the man.", "label": "neutral", "attribute": "m"},
        {"text": "They went shopping.", "label": "neutral", "attribute": None},  # no attribute -> skipped
    ]
    languages = ["en", "en"]
    d_aug, stats = build_augmented_dataset(dataset, engine, languages, all_attributes=["m", "f"])

    assert stats.n_original == 2
    assert stats.n_accepted == 1
    assert len(d_aug) == 3  # original 2 + 1 accepted counterfactual
    new_entry = d_aug[-1]
    assert new_entry["text"] == "She gave her book to the woman."
    assert new_entry["attribute"] == "f"
    assert new_entry["label"] == "neutral"  # label unchanged, assumption A1


def test_build_augmented_dataset_rejects_low_score_candidates():
    engine = _build_engine(gamma=100.0)  # nothing will pass
    dataset = [{"text": "He gave his book to the man.", "label": "neutral", "attribute": "m"}]
    d_aug, stats = build_augmented_dataset(dataset, engine, ["en"], all_attributes=["m", "f"])
    assert stats.n_rejected_score == 1
    assert stats.n_accepted == 0
    assert len(d_aug) == 1  # only the original, no counterfactual added


def test_build_augmented_dataset_length_mismatch_raises():
    engine = _build_engine()
    import pytest
    with pytest.raises(ValueError):
        build_augmented_dataset([{"text": "a", "label": "x"}], engine, ["en", "fr"], all_attributes=["m", "f"])


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
