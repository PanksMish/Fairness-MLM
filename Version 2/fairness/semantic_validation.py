"""
Semantic-syntactic validation for counterfactual candidates, Eq. (3),
(8), (9):

    Eq 3: CosSim(h(x), h(x^(b))) >= 0.85          (semantic preservation constraint)
    Eq 8: S(x, x^(b)) = alpha*CosSim(h(x),h(x^(b))) - beta*GramErr(x^(b))
    Eq 9: accept iff S(x, x^(b)) >= gamma

CosSim is computed here from plain vectors (pure NumPy, testable). GramErr
needs a real grammar-error model/tool (the manuscript doesn't specify
which); this module defines a `GrammarChecker` protocol so any backend
can be plugged in (LanguageTool via `language-tool-python`, a
fine-tuned GEC model, etc.), plus one concrete, deliberately simple
heuristic checker (`HeuristicGrammarChecker`) that is real, runs with
zero dependencies, and is honestly documented as a weak placeholder --
not a substitute for a real grammar model in a publication-quality run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Eq. 3: cosine similarity between encoder representations h(x), h(x^(b))
# ---------------------------------------------------------------------------

def cosine_similarity(h_x: np.ndarray, h_xb: np.ndarray) -> float:
    """
    Eq. (3):

        CosSim(h(x), h(x^(b))) = (h(x) . h(x^(b))) / (||h(x)|| ||h(x^(b))||)

    Args:
        h_x, h_xb: 1D encoder representation vectors (e.g. the pooled
            output of model/encoder.py's MultilingualEncoder, converted
            to NumPy via `.detach().cpu().numpy()`).
    """
    h_x = np.asarray(h_x, dtype=np.float64)
    h_xb = np.asarray(h_xb, dtype=np.float64)
    if h_x.shape != h_xb.shape:
        raise ValueError(f"Shape mismatch: {h_x.shape} vs {h_xb.shape}")
    norm_x = np.linalg.norm(h_x)
    norm_xb = np.linalg.norm(h_xb)
    if norm_x == 0 or norm_xb == 0:
        return 0.0
    return float(np.dot(h_x, h_xb) / (norm_x * norm_xb))


def semantic_preservation_ok(h_x: np.ndarray, h_xb: np.ndarray, threshold: float = 0.85) -> bool:
    """Eq. (3) as a boolean gate, threshold=0.85 per the manuscript."""
    return cosine_similarity(h_x, h_xb) >= threshold


# ---------------------------------------------------------------------------
# Eq. 8: semantic-syntactic score, combining CosSim and a grammar-error term
# ---------------------------------------------------------------------------

class GrammarChecker(Protocol):
    def __call__(self, text: str) -> float:
        """Returns a non-negative grammar-error score for `text` (higher
        = more errors). Scale is checker-dependent; alpha/beta in Eq. 8
        should be tuned relative to whatever scale your chosen checker
        produces."""
        ...


@dataclass
class HeuristicGrammarChecker:
    """
    NOT a real grammar model. A zero-dependency placeholder that flags a
    handful of surface-level artifacts common to naive lexical
    substitution (the failure mode Sec 4.1/McCutchen et al. specifically
    warn about in morphologically rich languages):

      - doubled whitespace (substitution left an extra space)
      - doubled punctuation
      - a substituted token directly adjacent to itself (substitution
        loop / no-op replacement)
      - capitalization mismatch at sentence start after substitution

    Each triggered pattern adds 1.0 to the score. Replace with a real
    checker (LanguageTool, a GEC model) before treating GramErr as
    meaningful for anything beyond a smoke test.
    """

    def __call__(self, text: str) -> float:
        score = 0.0
        if re.search(r"  +", text):
            score += 1.0
        if re.search(r"([.,!?;:])\1", text):
            score += 1.0
        words = text.split()
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                score += 1.0
        if text and text[0].islower():
            score += 1.0
        return score


def semantic_syntactic_score(
    h_x: np.ndarray,
    h_xb: np.ndarray,
    candidate_text: str,
    grammar_checker: GrammarChecker,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """Eq. (8): S(x, x^(b)) = alpha*CosSim(h(x),h(x^(b))) - beta*GramErr(x^(b))."""
    cos_sim = cosine_similarity(h_x, h_xb)
    gram_err = grammar_checker(candidate_text)
    return alpha * cos_sim - beta * gram_err


def accept_candidate(score: float, gamma: float) -> bool:
    """Eq. (9): accept iff S(x, x^(b)) >= gamma."""
    return score >= gamma
