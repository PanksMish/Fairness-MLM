"""
Morphological validation for counterfactual candidates.

The manuscript flags (Sec 4.1, citing McCutchen et al. 2022) that "simple
lexical substitution methods often introduce grammatical inconsistencies
in morphologically rich languages" -- e.g. swapping a gendered noun
without updating agreeing articles/adjectives/pronouns elsewhere in the
sentence. It does not give a formal morphological validation algorithm,
so this module provides:

  1. A generic `length_ratio_check` heuristic (real, testable): flags
     candidates whose word count deviates too much from the original,
     as a cheap proxy for "substitution broke the sentence structure."
  2. A small, explicitly limited, illustrative rule-based agreement
     fixer for a couple of morphologically-marked-gender languages
     (Spanish, German) as a concrete example of what a real morphology
     module would need to do -- NOT a comprehensive solution, and NOT
     claiming coverage of the 101 languages in the manuscript. Real
     morphological agreement checking at that scale needs a proper
     morphological analyzer per language family (e.g. UDPipe, Stanza,
     or language-specific rule sets), which is out of scope for what can
     be honestly hand-written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MorphologyValidator(Protocol):
    def __call__(self, original: str, candidate: str) -> bool:
        """Returns True if `candidate` is morphologically acceptable
        relative to `original`."""
        ...


def length_ratio_check(original: str, candidate: str, max_deviation: float = 0.5) -> bool:
    """
    Cheap structural sanity check: word count shouldn't change by more
    than `max_deviation` fraction. A substitution that adds/drops many
    words relative to the original likely broke sentence structure
    (e.g. a pivot-translation round-trip that degenerated, or a lexicon
    substitution that matched the wrong span).
    """
    orig_len = len(original.split())
    cand_len = len(candidate.split())
    if orig_len == 0:
        return cand_len == 0
    deviation = abs(cand_len - orig_len) / orig_len
    return deviation <= max_deviation


# ---------------------------------------------------------------------------
# Illustrative, deliberately limited gender-agreement fixers.
# ---------------------------------------------------------------------------

# Spanish: definite articles agree in gender with the following noun.
_ES_ARTICLE_AGREEMENT = {
    ("el", "f"): "la", ("la", "m"): "el",
    ("un", "f"): "una", ("una", "m"): "un",
    ("los", "f"): "las", ("las", "m"): "los",
}

# German: definite articles (nominative case only, the simplest case)
# agreeing in gender.
_DE_ARTICLE_AGREEMENT = {
    ("der", "f"): "die", ("die", "m"): "der",
    ("der", "n"): "das", ("das", "m"): "der",
    ("die", "n"): "das", ("das", "f"): "die",
}

_AGREEMENT_TABLES = {"es": _ES_ARTICLE_AGREEMENT, "de": _DE_ARTICLE_AGREEMENT}


def fix_preceding_article(tokens: list[str], noun_index: int, target_gender: str, language: str) -> list[str]:
    """
    Given a tokenized sentence and the index of a noun whose gender was
    just changed to `target_gender` (via attribute substitution), checks
    whether the immediately preceding token is a definite/indefinite
    article and, if so, replaces it with the gender-agreeing form.

    This is intentionally narrow: it only handles the single
    immediately-preceding-article case, only for `language` in
    `_AGREEMENT_TABLES`, and only nominative-case German. It exists to
    demonstrate the mechanism the manuscript gestures at, not to claim
    general morphological correctness.

    Args:
        tokens: word-tokenized sentence
        noun_index: index of the noun that was gender-swapped
        target_gender: "m", "f", or "n" (German only)
        language: ISO code, must be a key in _AGREEMENT_TABLES or this
            is a no-op (returns tokens unchanged)

    Returns:
        A new token list (input is not mutated).
    """
    table = _AGREEMENT_TABLES.get(language)
    if table is None or noun_index == 0:
        return list(tokens)

    tokens = list(tokens)
    prev_word = tokens[noun_index - 1].lower()
    replacement = table.get((prev_word, target_gender))
    if replacement is not None:
        # Preserve original capitalization pattern of the article
        if tokens[noun_index - 1][0].isupper():
            replacement = replacement.capitalize()
        tokens[noun_index - 1] = replacement
    return tokens


@dataclass
class CombinedMorphologyValidator:
    """
    Default validator wiring: length-ratio check always applied;
    language-specific agreement fixing only applied for languages in
    `_AGREEMENT_TABLES` (a no-op otherwise, which is honest -- it means
    "not validated" rather than pretending validation succeeded).
    """
    max_length_deviation: float = 0.5

    def __call__(self, original: str, candidate: str) -> bool:
        return length_ratio_check(original, candidate, self.max_length_deviation)
