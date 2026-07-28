"""
Counterfactual generation for the NER task, operating on WikiAnn's
already-tokenized (tokens, tags) format rather than raw text.

This is simpler than the sentiment-side text substitution
(fairness/counterfactual_generation.py's LexiconSubstitutor, which needs
regex word-boundary matching over free text) because WikiAnn's tokens are
already split into words: a per-TOKEN dictionary lookup preserves the
token count exactly, so the tag sequence needs no realignment at all --
tags[i] still describes tokens[i] after substitution, for every i.

Semantic invariance (assumption A1, Sec 3.2) here extends to per-token
label invariance: swapping "he"->"she" or "his"->"her" doesn't change
whether a token is part of a PER/ORG/LOC span, so cf_tags = tags
unchanged is the correct behavior for our gender-pronoun/kinship-term
dictionary specifically. This would NOT hold in general for substitution
dictionaries that swap actual entity mentions (e.g. replacing a person's
name) -- flagged here since it's a real constraint on which
dictionaries are safe to use with this function, not a limitation of the
function itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenCounterfactualCandidate:
    tokens: list[str]
    cf_tokens: list[str]
    tags: list[str]        # unchanged from input, by construction (see module docstring)
    attribute_from: str
    attribute_to: str
    n_substitutions: int


def substitute_tokens(
    tokens: list[str],
    attribute_dict: dict[str, str],
) -> tuple[list[str], int]:
    """
    Per-token, case-preserving substitution using a flat {source_word:
    target_word} dictionary (one direction of e.g.
    fairness/demographic_dictionaries.ENGLISH_GENDER_DICT["m"]["f"]).

    Args:
        tokens: word-tokenized sequence (WikiAnn's `tokens` field)
        attribute_dict: lowercase source word -> replacement word

    Returns:
        (new_tokens, n_substitutions) -- new_tokens has the SAME LENGTH
        as tokens (critical: this is what keeps tag alignment trivial).
    """
    new_tokens = []
    n_subs = 0
    for tok in tokens:
        replacement = attribute_dict.get(tok.lower())
        if replacement is None:
            new_tokens.append(tok)
            continue
        n_subs += 1
        if tok.isupper():
            new_tokens.append(replacement.upper())
        elif tok[0].isupper():
            new_tokens.append(replacement.capitalize())
        else:
            new_tokens.append(replacement)
    return new_tokens, n_subs


def detect_attribute_in_tokens(tokens: list[str], attribute_seed_words: dict[str, list[str]]) -> Optional[str]:
    """Same detection logic as
    fairness/counterfactual_generation.LexiconAttributeDetector, adapted
    for an already-tokenized input (no regex tokenization needed)."""
    token_set = set(t.lower() for t in tokens)
    for attribute, seeds in attribute_seed_words.items():
        if token_set & set(s.lower() for s in seeds):
            return attribute
    return None


def generate_token_counterfactual(
    tokens: list[str],
    tags: list[str],
    all_attribute_dicts: dict[str, dict[str, dict[str, str]]],
    attribute_seed_words: dict[str, list[str]],
    all_attributes: list[str],
    rng,
    attribute_from: Optional[str] = None,
) -> Optional[TokenCounterfactualCandidate]:
    """
    Full per-sample NER counterfactual generation, mirroring
    fairness/counterfactual_generation.CounterfactualDataEngine.generate_one's
    control flow (detect attribute -> pick target -> substitute) but for
    the tokens/tags representation and WITHOUT the semantic-syntactic
    scoring step (Eq. 8/9): there is no free-text grammar checker
    equivalent for a token list, and cosine-similarity-based semantic
    validation would need a sentence embedding of the reconstructed text
    anyway (i.e. it reduces to the sentiment-side check on
    " ".join(tokens) if you want it -- left to the caller to add via
    fairness/semantic_validation.py on the joined text if desired).

    Returns None if no attribute is detected or no substitution occurs.
    """
    attribute_from = attribute_from or detect_attribute_in_tokens(tokens, attribute_seed_words)
    if attribute_from is None:
        return None

    candidates = [a for a in all_attributes if a != attribute_from]
    if not candidates:
        return None
    attribute_to = rng.choice(candidates)

    attr_dict = all_attribute_dicts.get(attribute_from, {}).get(attribute_to)
    if not attr_dict:
        return None

    cf_tokens, n_subs = substitute_tokens(tokens, attr_dict)
    if n_subs == 0:
        return None

    return TokenCounterfactualCandidate(
        tokens=tokens, cf_tokens=cf_tokens, tags=tags,
        attribute_from=attribute_from, attribute_to=attribute_to, n_substitutions=n_subs,
    )
