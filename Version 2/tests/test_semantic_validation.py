import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fairness.semantic_validation import (
    cosine_similarity,
    semantic_preservation_ok,
    HeuristicGrammarChecker,
    semantic_syntactic_score,
    accept_candidate,
)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_opposite_is_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == 0.0


def test_semantic_preservation_threshold_eq3():
    a = np.array([1.0, 0.0])
    b_close = np.array([0.99, 0.14])   # cos sim > 0.85
    b_far = np.array([0.0, 1.0])       # cos sim = 0
    assert semantic_preservation_ok(a, b_close, threshold=0.85)
    assert not semantic_preservation_ok(a, b_far, threshold=0.85)


def test_heuristic_grammar_checker_flags_doubled_whitespace():
    checker = HeuristicGrammarChecker()
    assert checker("Hello  world") >= 1.0
    assert checker("Hello world") == 0.0


def test_heuristic_grammar_checker_flags_doubled_punctuation():
    checker = HeuristicGrammarChecker()
    assert checker("wait,, what") >= 1.0


def test_heuristic_grammar_checker_flags_repeated_adjacent_word():
    checker = HeuristicGrammarChecker()
    assert checker("the the cat sat") >= 1.0


def test_heuristic_grammar_checker_flags_lowercase_start():
    checker = HeuristicGrammarChecker()
    assert checker("hello world.") >= 1.0
    assert checker("Hello world.") == 0.0


def test_semantic_syntactic_score_eq8_formula():
    h_x = np.array([1.0, 0.0])
    h_xb = np.array([1.0, 0.0])  # cos sim = 1.0

    def zero_grammar_errors(text):
        return 0.0

    score = semantic_syntactic_score(h_x, h_xb, "clean text", zero_grammar_errors, alpha=2.0, beta=3.0)
    # score = 2.0*1.0 - 3.0*0.0 = 2.0
    assert abs(score - 2.0) < 1e-9


def test_semantic_syntactic_score_penalizes_grammar_errors():
    h_x = np.array([1.0, 0.0])
    h_xb = np.array([1.0, 0.0])  # cos sim = 1.0

    def two_grammar_errors(text):
        return 2.0

    score = semantic_syntactic_score(h_x, h_xb, "bad  text", two_grammar_errors, alpha=1.0, beta=0.5)
    # score = 1.0*1.0 - 0.5*2.0 = 0.0
    assert abs(score - 0.0) < 1e-9


def test_accept_candidate_eq9():
    assert accept_candidate(score=0.6, gamma=0.5)
    assert not accept_candidate(score=0.4, gamma=0.5)
    assert accept_candidate(score=0.5, gamma=0.5)  # boundary: >= is accept


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
