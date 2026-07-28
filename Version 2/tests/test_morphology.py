import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairness.morphology import (
    length_ratio_check,
    fix_preceding_article,
    CombinedMorphologyValidator,
)


def test_length_ratio_check_identical_length_passes():
    assert length_ratio_check("the cat sat", "the dog sat")


def test_length_ratio_check_small_deviation_passes():
    assert length_ratio_check("the cat sat down", "the dog sat", max_deviation=0.5)


def test_length_ratio_check_large_deviation_fails():
    assert not length_ratio_check("the cat sat", "the cat sat there quietly all afternoon long", max_deviation=0.5)


def test_length_ratio_check_empty_original_and_candidate():
    assert length_ratio_check("", "")


def test_length_ratio_check_empty_original_nonempty_candidate_fails():
    assert not length_ratio_check("", "something")


def test_fix_preceding_article_spanish_masculine_to_feminine():
    tokens = ["el", "gato", "corre"]
    fixed = fix_preceding_article(tokens, noun_index=1, target_gender="f", language="es")
    assert fixed == ["la", "gato", "corre"]


def test_fix_preceding_article_spanish_preserves_capitalization():
    tokens = ["El", "gato", "corre"]
    fixed = fix_preceding_article(tokens, noun_index=1, target_gender="f", language="es")
    assert fixed[0] == "La"


def test_fix_preceding_article_no_op_for_unsupported_language():
    tokens = ["the", "cat", "runs"]
    fixed = fix_preceding_article(tokens, noun_index=1, target_gender="f", language="en")
    assert fixed == tokens  # unchanged: English not in agreement tables


def test_fix_preceding_article_no_op_when_noun_is_first_token():
    tokens = ["gato", "corre"]
    fixed = fix_preceding_article(tokens, noun_index=0, target_gender="f", language="es")
    assert fixed == tokens


def test_fix_preceding_article_german_lowercase_preserved():
    tokens = ["das", "Kind", "lacht"]
    fixed = fix_preceding_article(tokens, noun_index=1, target_gender="m", language="de")
    assert fixed[0] == "der"


def test_combined_morphology_validator_uses_length_check():
    validator = CombinedMorphologyValidator(max_length_deviation=0.3)
    assert validator("a b c d", "a b c e")
    assert not validator("a b c d", "a b c d e f g h")


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
