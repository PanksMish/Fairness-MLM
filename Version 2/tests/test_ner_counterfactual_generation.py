import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairness.ner_counterfactual_generation import (
    substitute_tokens, detect_attribute_in_tokens, generate_token_counterfactual,
)

GENDER_DICT = {
    "m": {"f": {"he": "she", "his": "her", "man": "woman"}},
    "f": {"m": {"she": "he", "her": "his", "woman": "man"}},
}
SEEDS = {"m": ["he", "his", "man"], "f": ["she", "her", "woman"]}


def test_substitute_tokens_preserves_length():
    tokens = ["He", "met", "a", "man", "in", "London"]
    new_tokens, n = substitute_tokens(tokens, {"he": "she", "man": "woman"})
    assert len(new_tokens) == len(tokens)
    assert n == 2


def test_substitute_tokens_preserves_capitalization():
    tokens = ["He", "met", "the", "MAN"]
    new_tokens, n = substitute_tokens(tokens, {"he": "she", "man": "woman"})
    assert new_tokens[0] == "She"
    assert new_tokens[3] == "WOMAN"


def test_substitute_tokens_no_match_unchanged():
    tokens = ["They", "went", "home"]
    new_tokens, n = substitute_tokens(tokens, {"he": "she"})
    assert new_tokens == tokens
    assert n == 0


def test_substitute_tokens_tag_alignment_preserved_by_construction():
    # This is THE critical property: entity tags don't need to move
    # because the token count and positions are unchanged.
    tokens = ["He", "works", "at", "Google"]
    tags = ["O", "O", "O", "B-ORG"]
    new_tokens, n = substitute_tokens(tokens, {"he": "she"})
    assert len(new_tokens) == len(tags)
    assert new_tokens[3] == "Google"  # entity token itself untouched
    assert tags[3] == "B-ORG"  # tag list is literally the same object/values, unaffected


def test_detect_attribute_in_tokens_finds_seed():
    tokens = ["His", "dog", "ran"]
    assert detect_attribute_in_tokens(tokens, SEEDS) == "m"


def test_detect_attribute_in_tokens_none_when_absent():
    tokens = ["The", "dog", "ran"]
    assert detect_attribute_in_tokens(tokens, SEEDS) is None


def test_generate_token_counterfactual_full_flow():
    tokens = ["He", "works", "at", "Google", "in", "Paris"]
    tags = ["O", "O", "O", "B-ORG", "O", "B-LOC"]
    rng = random.Random(0)

    candidate = generate_token_counterfactual(
        tokens, tags, GENDER_DICT, SEEDS, all_attributes=["m", "f"], rng=rng,
    )
    assert candidate is not None
    assert candidate.attribute_from == "m"
    assert candidate.attribute_to == "f"
    assert candidate.cf_tokens[0] == "She"
    assert candidate.cf_tokens[3] == "Google"  # unchanged
    assert candidate.tags == tags  # tags exactly preserved
    assert len(candidate.cf_tokens) == len(tags)
    assert candidate.n_substitutions == 1


def test_generate_token_counterfactual_none_when_no_attribute():
    tokens = ["The", "cat", "sat"]
    tags = ["O", "O", "O"]
    rng = random.Random(0)
    candidate = generate_token_counterfactual(tokens, tags, GENDER_DICT, SEEDS, ["m", "f"], rng)
    assert candidate is None


def test_generate_token_counterfactual_uses_declared_attribute():
    tokens = ["She", "works", "here"]
    tags = ["O", "O", "O"]
    rng = random.Random(0)
    candidate = generate_token_counterfactual(
        tokens, tags, GENDER_DICT, SEEDS, ["m", "f"], rng, attribute_from="f",
    )
    assert candidate is not None
    assert candidate.attribute_from == "f"
    assert candidate.attribute_to == "m"
    assert candidate.cf_tokens[0] == "He"


def test_generate_token_counterfactual_none_when_dict_missing_pair():
    tokens = ["He", "works", "here"]
    tags = ["O", "O", "O"]
    rng = random.Random(0)
    # dictionary only has m->f, not m->x
    candidate = generate_token_counterfactual(
        tokens, tags, GENDER_DICT, SEEDS, ["m", "x"], rng,
    )
    assert candidate is None


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
