import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.build_counterfactual_pairs import generate_pairs
from datasets.dataset_utils import write_jsonl, read_jsonl
from fairness.counterfactual_generation import (
    LexiconSubstitutor, TieredGenerationStrategy, LexiconAttributeDetector,
    CounterfactualDataEngine, CounterfactualEngineConfig,
)
from fairness.demographic_dictionaries import ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, EXAMPLE_TOKEN_COUNTS


def _mock_encoder_fn(text: str):
    """Deterministic fake embedding, same trick as
    test_counterfactual_generation.py: counts of gendered marker words
    plus length, so cosine similarity is meaningfully non-trivial."""
    words = text.lower().split()
    return [
        sum(words.count(w) for w in ["he", "him", "his", "man", "father"]),
        sum(words.count(w) for w in ["she", "her", "hers", "woman", "mother"]),
        len(words),
    ]


def _build_test_engine(gamma: float):
    detector = LexiconAttributeDetector(ENGLISH_GENDER_SEEDS)
    hr_op = LexiconSubstitutor(ENGLISH_GENDER_DICT)
    strategy = TieredGenerationStrategy(
        hr_operator=hr_op,
        mr_operator=lambda t, a, b: None,
        lr_operator=lambda t, a, b, source_lang: None,
        token_counts=EXAMPLE_TOKEN_COUNTS,
    )
    return CounterfactualDataEngine(
        detector=detector, strategy=strategy, encoder_fn=_mock_encoder_fn,
        config=CounterfactualEngineConfig(gamma=gamma),
    )


def test_generate_pairs_produces_accepted_pairs_with_correct_schema():
    engine = _build_test_engine(gamma=-10.0)  # lax, everything with a candidate is accepted
    records = [
        {"text": "He gave his book to the father.", "label": "neutral", "attribute": "m"},
        {"text": "They went to the store.", "label": "neutral"},  # no attribute -> skipped
    ]
    pairs, stats = generate_pairs(records, engine, language="en", all_attributes=["m", "f"])

    assert stats["n_input"] == 2
    assert stats["n_accepted"] == 1
    assert stats["n_skipped_no_candidate"] == 1
    assert len(pairs) == 1

    pair = pairs[0]
    assert pair["text"] == "He gave his book to the father."
    assert pair["cf_text"] == "She gave her book to the mother."
    assert pair["attribute"] == "m"
    assert pair["cf_attribute"] == "f"
    assert pair["language"] == "en"
    assert "score" in pair
    assert pair["label"] == "neutral"


def test_generate_pairs_rejects_below_gamma():
    engine = _build_test_engine(gamma=1000.0)  # impossible to pass
    records = [{"text": "He gave his book to the father.", "label": "neutral", "attribute": "m"}]
    pairs, stats = generate_pairs(records, engine, language="en", all_attributes=["m", "f"])
    assert stats["n_accepted"] == 0
    assert stats["n_rejected_score"] == 1
    assert len(pairs) == 0


def test_generate_pairs_yield_rate_computed_correctly():
    engine = _build_test_engine(gamma=-10.0)
    records = [
        {"text": "He gave his book.", "label": "pos", "attribute": "m"},
        {"text": "She gave her book.", "label": "pos", "attribute": "f"},
        {"text": "No attribute here.", "label": "neg"},
        {"text": "Neither does this one.", "label": "neg"},
    ]
    pairs, stats = generate_pairs(records, engine, language="en", all_attributes=["m", "f"])
    assert stats["n_input"] == 4
    assert stats["n_accepted"] == 2
    assert abs(stats["yield_rate"] - 0.5) < 1e-9


def test_generate_pairs_empty_input():
    engine = _build_test_engine(gamma=0.0)
    pairs, stats = generate_pairs([], engine, language="en", all_attributes=["m", "f"])
    assert pairs == []
    assert stats["n_input"] == 0
    assert stats["yield_rate"] == 0.0


def test_generate_pairs_roundtrip_through_jsonl():
    """End-to-end sanity: generate pairs, write to JSONL, read back, and
    verify the schema survives serialization intact (this is the exact
    format PairedSentimentDataset in datasets/dataloaders.py expects)."""
    engine = _build_test_engine(gamma=-10.0)
    records = [{"text": "He gave his book to the man.", "label": "neutral", "attribute": "m"}]
    pairs, _ = generate_pairs(records, engine, language="en", all_attributes=["m", "f"])

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "pairs.jsonl")
        write_jsonl(pairs, path)
        loaded = list(read_jsonl(path))
        assert len(loaded) == 1
        assert loaded[0]["cf_text"] == "She gave her book to the woman."
        assert loaded[0]["cf_attribute"] == "f"
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
