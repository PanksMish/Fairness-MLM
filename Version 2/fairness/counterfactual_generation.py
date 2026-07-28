"""
Counterfactual Data Engine, implementing the transformation operator
T_{a->b} (Eq. 2/7) and Algorithm 1 ("Counterfactual Data Engine"):

    1: Initialize D_aug <- D
    2: Define similarity threshold gamma
    3: for each sample (x, y, a) in D do
    4:     Identify attribute a in x
    5:     Select target attribute b != a
    6:     Generate counterfactual candidate x^(b)
    7:     Compute representations h(x) and h(x^(b))
    8:     Compute semantic similarity S(x, x^(b))
    9:     if S(x, x^(b)) >= gamma then
    10:        Add (x^(b), y, b) to D_aug
    11:    end if
    12: end for
    13: return D_aug

Sec 4.1's cross-lingual generation strategy dispatches by resource tier:
    HR: lexicon-based substitution from curated demographic dictionaries
    MR: cross-lingual embedding space alignment
    LR: pivot-based translation with back-translation

Each tier is a `TransformationOperator` implementation below. The HR
(lexicon) path is real, deterministic, and fully testable with zero
model dependencies. The MR (embedding-alignment) and LR
(pivot-translation) paths need a real embedding model / MT system
respectively -- they're implemented against injected callables
(protocols), so the *dispatch and engine logic* is testable with mocked
embeddings/translation, even though a real run needs real models plugged
in.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from evaluation.fairness_metrics import resource_category
from fairness.semantic_validation import (
    cosine_similarity, semantic_syntactic_score, accept_candidate,
    GrammarChecker, HeuristicGrammarChecker,
)
from fairness.morphology import MorphologyValidator, CombinedMorphologyValidator


# ---------------------------------------------------------------------------
# Transformation operator T_{a->b}: Eq. (2)/(7)
# ---------------------------------------------------------------------------

class TransformationOperator(Protocol):
    def __call__(self, text: str, attribute_from: str, attribute_to: str) -> Optional[str]:
        """Returns the transformed text x^(b), or None if no
        transformation could be generated (e.g. no matching lexicon
        entries found in the input)."""
        ...


@dataclass
class LexiconSubstitutor:
    """
    High-resource tier (Sec 4.1): "lexicon-based substitutions based on
    the curated demographic dictionaries."

    `dictionaries[attribute_from][attribute_to]` maps a source word to
    its counterfactual replacement (e.g. dictionaries["m"]["f"] =
    {"he": "she", "his": "her", "man": "woman", ...}). Substitution is
    whole-word, case-preserving, and word-boundary-safe (won't match
    "he" inside "the"). This is real, deterministic logic with no model
    dependency.
    """
    dictionaries: dict[str, dict[str, dict[str, str]]]

    def __call__(self, text: str, attribute_from: str, attribute_to: str) -> Optional[str]:
        table = self.dictionaries.get(attribute_from, {}).get(attribute_to)
        if not table:
            return None

        result = text
        any_match = False
        for src_word, tgt_word in table.items():
            pattern = re.compile(rf"\b{re.escape(src_word)}\b", flags=re.IGNORECASE)

            def _replace(m: re.Match) -> str:
                nonlocal any_match
                any_match = True
                matched = m.group(0)
                if matched.isupper():
                    return tgt_word.upper()
                if matched[0].isupper():
                    return tgt_word.capitalize()
                return tgt_word

            result = pattern.sub(_replace, result)

        return result if any_match else None


class EmbeddingSpace(Protocol):
    def get_vector(self, word: str, language: str) -> Optional["list[float]"]:
        ...

    def nearest_neighbor(self, vector: "list[float]", language: str, exclude: set[str]) -> Optional[str]:
        ...


@dataclass
class EmbeddingAlignmentTransform:
    """
    Medium-resource tier (Sec 4.1): "cross-lingual embedding space
    alignment to perform demographic attribute transfer."

    Requires a real cross-lingual embedding space (e.g. aligned fastText
    vectors, or a multilingual sentence-embedding model) injected via the
    `EmbeddingSpace` protocol. This class implements the *substitution
    logic* (find the demographic-attribute-bearing word, look up its
    nearest neighbor under the target attribute's embedding subspace) --
    it does not implement or vendor an embedding model itself.

    `attribute_seed_words[attribute]` gives a small seed vocabulary per
    attribute (e.g. attribute_seed_words["m"] = ["he", "him", "man"]),
    used to detect which words in `text` carry the source attribute and
    are therefore substitution candidates.
    """
    embedding_space: EmbeddingSpace
    attribute_seed_words: dict[str, list[str]]

    def __call__(self, text: str, attribute_from: str, attribute_to: str) -> Optional[str]:
        seeds_from = set(w.lower() for w in self.attribute_seed_words.get(attribute_from, []))
        if not seeds_from:
            return None

        words = text.split()
        changed = False
        new_words = []
        for w in words:
            bare = re.sub(r"[^\w]", "", w).lower()
            if bare in seeds_from:
                vec = self.embedding_space.get_vector(bare, language="und")
                if vec is not None:
                    neighbor = self.embedding_space.nearest_neighbor(
                        vec, language="und", exclude=seeds_from,
                    )
                    if neighbor is not None:
                        new_words.append(neighbor if w.islower() else neighbor.capitalize())
                        changed = True
                        continue
            new_words.append(w)

        return " ".join(new_words) if changed else None


class Translator(Protocol):
    def __call__(self, text: str, src_lang: str, tgt_lang: str) -> str:
        ...


@dataclass
class PivotTranslationTransform:
    """
    Low-resource tier (Sec 4.1): "pivot-based translations with
    subsequent back-translation."

    Pipeline: source_lang -> pivot_lang (e.g. English) -> attribute
    substitution in the pivot language (reusing LexiconSubstitutor, since
    the pivot is typically high-resource) -> back-translate to
    source_lang. Requires a real MT system injected via the `Translator`
    protocol; this class only implements the orchestration.
    """
    translator: Translator
    pivot_substitutor: LexiconSubstitutor
    pivot_lang: str = "en"

    def __call__(self, text: str, attribute_from: str, attribute_to: str,
                 source_lang: str = "und") -> Optional[str]:
        pivot_text = self.translator(text, src_lang=source_lang, tgt_lang=self.pivot_lang)
        substituted = self.pivot_substitutor(pivot_text, attribute_from, attribute_to)
        if substituted is None:
            return None
        back_translated = self.translator(substituted, src_lang=self.pivot_lang, tgt_lang=source_lang)
        return back_translated


# ---------------------------------------------------------------------------
# Resource-tiered dispatch
# ---------------------------------------------------------------------------

@dataclass
class TieredGenerationStrategy:
    """
    Dispatches to the appropriate TransformationOperator based on the
    language's resource tier (Sec 3.1's HR/MR/LR categorization, reused
    from evaluation/fairness_metrics.resource_category).
    """
    hr_operator: TransformationOperator
    mr_operator: TransformationOperator
    lr_operator: Callable  # PivotTranslationTransform needs source_lang kwarg
    token_counts: dict[str, float]  # language -> T_l, for resource_category()

    def generate(self, text: str, attribute_from: str, attribute_to: str, language: str) -> Optional[str]:
        token_count = self.token_counts.get(language)
        if token_count is None:
            raise ValueError(f"No token count configured for language '{language}'; "
                              "cannot determine resource tier.")
        tier = resource_category(token_count)
        if tier == "HR":
            return self.hr_operator(text, attribute_from, attribute_to)
        elif tier == "MR":
            return self.mr_operator(text, attribute_from, attribute_to)
        else:
            return self.lr_operator(text, attribute_from, attribute_to, source_lang=language)


# ---------------------------------------------------------------------------
# Attribute detection (Algorithm 1, line 4)
# ---------------------------------------------------------------------------

class AttributeDetector(Protocol):
    def __call__(self, text: str) -> Optional[str]:
        """Returns the detected demographic/dialectal attribute label
        present in `text`, or None if undetectable."""
        ...


@dataclass
class LexiconAttributeDetector:
    """
    Simplest real implementation: detects an attribute by checking
    whether any of its seed words (same seed lists used by
    EmbeddingAlignmentTransform) appear in the text. A production system
    would use a trained classifier for attribute detection where lexical
    cues are insufficient (Sec 4.1 doesn't specify one); this is the
    lexicon-only baseline.
    """
    attribute_seed_words: dict[str, list[str]]

    def __call__(self, text: str) -> Optional[str]:
        words = set(w.lower() for w in re.findall(r"\w+", text))
        for attribute, seeds in self.attribute_seed_words.items():
            if words & set(s.lower() for s in seeds):
                return attribute
        return None


# ---------------------------------------------------------------------------
# Algorithm 1: Counterfactual Data Engine
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualCandidate:
    original_text: str
    candidate_text: str
    attribute_from: str
    attribute_to: str
    score: float
    accepted: bool


@dataclass
class CounterfactualEngineConfig:
    gamma: float = 0.5     # Eq. 9 acceptance threshold
    alpha: float = 1.0     # Eq. 8
    beta: float = 0.5      # Eq. 8
    morphology_check: bool = True


class CounterfactualDataEngine:
    """
    Algorithm 1, wired to real (or injected-mock) components:

        detector: AttributeDetector           (line 4)
        strategy: TieredGenerationStrategy      (line 6)
        encoder_fn: text -> h(x) vector          (line 7; wraps
            model/encoder.py's MultilingualEncoder in production, or any
            embedding function for testing)
        grammar_checker: GrammarChecker          (used inside Eq. 8)
        morphology_validator: MorphologyValidator (extra gate beyond Eq.
            9, per Sec 4.1's morphological-consistency concern; not
            itself part of the Eq. 8/9 formula, applied as an additional
            AND-ed acceptance condition)
    """

    def __init__(
        self,
        detector: AttributeDetector,
        strategy: TieredGenerationStrategy,
        encoder_fn: Callable[[str], "list[float]"],
        grammar_checker: Optional[GrammarChecker] = None,
        morphology_validator: Optional[MorphologyValidator] = None,
        config: Optional[CounterfactualEngineConfig] = None,
        rng_seed: int = 0,
    ):
        self.detector = detector
        self.strategy = strategy
        self.encoder_fn = encoder_fn
        self.grammar_checker = grammar_checker or HeuristicGrammarChecker()
        self.morphology_validator = morphology_validator or CombinedMorphologyValidator()
        self.config = config or CounterfactualEngineConfig()
        self.rng = random.Random(rng_seed)

    def _select_target_attribute(self, attribute_from: str, all_attributes: list[str]) -> str:
        """Algorithm 1, line 5: select target attribute b != a."""
        candidates = [a for a in all_attributes if a != attribute_from]
        if not candidates:
            raise ValueError(f"No valid target attribute distinct from '{attribute_from}' in {all_attributes}")
        return self.rng.choice(candidates)

    def generate_one(
        self, text: str, language: str, all_attributes: list[str],
        attribute_from: Optional[str] = None,
    ) -> Optional[CounterfactualCandidate]:
        """
        Runs Algorithm 1's per-sample body (lines 4-11) for a single
        (x, y, a) sample (y is not needed by this method -- it's carried
        through unchanged by the caller building D_aug, see
        fairness/augmentation.py).
        """
        attribute_from = attribute_from or self.detector(text)
        if attribute_from is None:
            return None  # line 4: attribute not identifiable, skip sample

        attribute_to = self._select_target_attribute(attribute_from, all_attributes)  # line 5
        candidate_text = self.strategy.generate(text, attribute_from, attribute_to, language)  # line 6
        if candidate_text is None:
            return None

        if self.config.morphology_check and not self.morphology_validator(text, candidate_text):
            return CounterfactualCandidate(text, candidate_text, attribute_from, attribute_to, score=float("-inf"), accepted=False)

        h_x = self.encoder_fn(text)               # line 7
        h_xb = self.encoder_fn(candidate_text)     # line 7

        score = semantic_syntactic_score(          # line 8, Eq. 8
            h_x, h_xb, candidate_text, self.grammar_checker,
            alpha=self.config.alpha, beta=self.config.beta,
        )
        accepted = accept_candidate(score, self.config.gamma)  # line 9, Eq. 9

        return CounterfactualCandidate(text, candidate_text, attribute_from, attribute_to, score, accepted)
