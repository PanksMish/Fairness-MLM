import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.build_ner_counterfactual_pairs import generate_ner_pairs
from fairness.demographic_dictionaries import ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS


def test_generate_ner_pairs_basic():
    records = [
        {"tokens": ["He", "works", "at", "Google"], "tags": ["O", "O", "O", "B-ORG"], "language": "en"},
        {"tokens": ["The", "cat", "sat"], "tags": ["O", "O", "O"], "language": "en"},  # no attribute
    ]
    pairs, stats = generate_ner_pairs(records, ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, ["m", "f"])

    assert stats["n_input"] == 2
    assert stats["n_accepted"] == 1
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["cf_tokens"][0] == "She"
    assert pair["cf_tokens"][3] == "Google"
    assert pair["tags"] == ["O", "O", "O", "B-ORG"]
    assert pair["attribute"] == "m"
    assert pair["cf_attribute"] == "f"


def test_generate_ner_pairs_empty():
    pairs, stats = generate_ner_pairs([], ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, ["m", "f"])
    assert pairs == []
    assert stats["n_input"] == 0
    assert stats["yield_rate"] == 0.0


def test_generate_ner_pairs_deterministic_with_seed():
    records = [{"tokens": ["He", "arrived"], "tags": ["O", "O"], "language": "en"}] * 5
    pairs1, _ = generate_ner_pairs(records, ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, ["m", "f"], seed=7)
    pairs2, _ = generate_ner_pairs(records, ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, ["m", "f"], seed=7)
    assert [p["cf_tokens"] for p in pairs1] == [p["cf_tokens"] for p in pairs2]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
