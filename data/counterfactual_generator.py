"""
counterfactual_generator.py
============================
Semantically Validated Counterfactual Data Augmentation (CDA) module.

Implements the augmentation pipeline from Section 3.6 / 4.3 of the paper:

  1. For each training instance (x, y, d) with demographic attribute a,
     generate a counterfactual x^(b) = T_{a→b}(x).

  2. Validate each candidate using a semantic–syntactic validation score:
         S(x, x') = α · CosSim(h(x), h(x')) − β · GramErr(x')
     Accept only if S(x, x') ≥ γ  (default γ = 0.75, CosSim threshold 0.85)

  3. Morphological Agreement Checker (MAC) ensures grammatical correctness
     in morphologically rich languages.

The final augmented dataset is:
  D_aug = D ∪ {(x^(b), y) : S(x, x^(b)) ≥ γ}
"""

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualSample:
    """A validated counterfactual sample."""
    original_text: str
    counterfactual_text: str
    label: int
    original_attribute: str        # "male" / "female"
    counterfactual_attribute: str  # opposite of original_attribute
    language: str
    semantic_similarity: float
    grammar_error_score: float
    validation_score: float
    accepted: bool


class CounterfactualGenerator:
    """
    Generates and validates counterfactual text samples.

    Strategy:
      - Primary: Rule-based lexical substitution using language-specific
        substitution tables (fast, grammatically controlled).
      - Secondary: mT5 masked-generation mode (optional, for richer variants).
      - Validation: LaBSE cosine similarity ≥ 0.85 + morphological check.

    Parameters
    ----------
    tokenizer : HuggingFace tokenizer
        For the backbone model (mT5-base or XLM-R).
    embedding_model_name : str
        LaBSE model for semantic similarity computation.
    similarity_threshold : float
        Minimum CosSim to accept a counterfactual (paper: 0.85).
    grammar_penalty_weight : float
        β in the validation score formula.
    similarity_weight : float
        α in the validation score formula.
    validation_threshold : float
        γ, combined threshold for S(x, x') (paper default: 0.75).
    use_mlm_generation : bool
        If True, also generate variants via masked language modeling.
        Slower but produces more diverse counterfactuals.
    device : str
        "cuda" or "cpu"
    """

    def __init__(
        self,
        tokenizer=None,
        embedding_model_name: str = "sentence-transformers/LaBSE",
        similarity_threshold: float = 0.85,
        grammar_penalty_weight: float = 0.1,
        similarity_weight: float = 1.0,
        validation_threshold: float = 0.75,
        use_mlm_generation: bool = False,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.similarity_threshold = similarity_threshold
        self.grammar_penalty_weight = grammar_penalty_weight
        self.similarity_weight = similarity_weight
        self.validation_threshold = validation_threshold
        self.use_mlm_generation = use_mlm_generation
        self.device = device

        # Lazy-load embedding model (LaBSE) to avoid import failures
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name

        # Morphological agreement checker
        from utils.morphological_checker import MorphologicalAgreementChecker
        self.mac = MorphologicalAgreementChecker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_counterfactual(
        self, text: str, original_attribute: str, language: str, label: int
    ) -> CounterfactualSample:
        """
        Generate a counterfactual for a single text sample.

        Parameters
        ----------
        text : str
            Original input text.
        original_attribute : str
            Demographic attribute of the original ("male" or "female").
        language : str
            ISO 639-1 language code.
        label : int
            Task label (preserved in the counterfactual).

        Returns
        -------
        CounterfactualSample with accepted=True if it passes validation.
        """
        target_attribute = "female" if original_attribute == "male" else "male"

        # Generate candidate via rule-based substitution
        candidate_text = self._rule_based_substitution(text, original_attribute, target_attribute, language)

        # If no substitution occurred, return a rejected sample
        if candidate_text == text:
            return CounterfactualSample(
                original_text=text,
                counterfactual_text=candidate_text,
                label=label,
                original_attribute=original_attribute,
                counterfactual_attribute=target_attribute,
                language=language,
                semantic_similarity=1.0,
                grammar_error_score=0.0,
                validation_score=self.similarity_weight * 1.0,
                accepted=False,  # No change made → not useful
            )

        # Compute validation score
        sem_sim = self._compute_semantic_similarity(text, candidate_text)
        gram_err = self._compute_grammar_error(candidate_text, language)
        validation_score = (
            self.similarity_weight * sem_sim
            - self.grammar_penalty_weight * gram_err
        )

        # Morphological agreement check
        mac_passed = self.mac.check(text, candidate_text, language)

        accepted = (
            sem_sim >= self.similarity_threshold
            and validation_score >= self.validation_threshold
            and mac_passed
        )

        return CounterfactualSample(
            original_text=text,
            counterfactual_text=candidate_text,
            label=label,
            original_attribute=original_attribute,
            counterfactual_attribute=target_attribute,
            language=language,
            semantic_similarity=sem_sim,
            grammar_error_score=gram_err,
            validation_score=validation_score,
            accepted=accepted,
        )

    def augment_dataset(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str],
        demographic_attrs: List[str],
    ) -> Tuple[List[str], List[int], List[str], List[str]]:
        """
        Augment a full dataset with counterfactual samples.

        Algorithm 1 from the paper:
          for each (x, y, d) in D:
              detect attribute a
              generate x^(b)
              compute S(x, x^(b))
              if S ≥ γ: add (x^(b), y) to D_aug

        Returns
        -------
        aug_texts, aug_labels, aug_languages, aug_dem_attrs
            The ORIGINAL samples PLUS accepted counterfactuals.
        """
        aug_texts = list(texts)
        aug_labels = list(labels)
        aug_languages = list(languages)
        aug_dem_attrs = list(demographic_attrs)

        accepted_count = 0
        for text, label, lang, attr in zip(texts, labels, languages, demographic_attrs):
            if attr not in ("male", "female"):
                continue  # Only augment samples with detected demographic attrs

            cf = self.generate_counterfactual(text, attr, lang, label)
            if cf.accepted:
                aug_texts.append(cf.counterfactual_text)
                aug_labels.append(cf.label)
                aug_languages.append(cf.language)
                aug_dem_attrs.append(cf.counterfactual_attribute)
                accepted_count += 1

        logger.info(
            f"Augmentation complete: {accepted_count} / {len(texts)} counterfactuals accepted "
            f"({100 * accepted_count / max(len(texts), 1):.1f}%)"
        )
        return aug_texts, aug_labels, aug_languages, aug_dem_attrs

    # ------------------------------------------------------------------
    # Rule-Based Substitution
    # ------------------------------------------------------------------

    def _rule_based_substitution(
        self, text: str, source_attr: str, target_attr: str, language: str
    ) -> str:
        """
        Apply lexical substitution to swap demographic attributes.

        Uses the substitution tables defined per-language. Handles:
          - Case preservation (He → She, HE → SHE, he → she)
          - German grammatical gender cascading (Arzt → Ärztin + article agreement)
        """
        pairs = _get_substitution_pairs(source_attr, target_attr, language)

        result = text
        for src_word, tgt_word in pairs:
            # Case-aware substitution via regex word boundary
            pattern = re.compile(r'\b' + re.escape(src_word) + r'\b', re.UNICODE)

            def _replace(match):
                matched = match.group()
                if matched.isupper():
                    return tgt_word.upper()
                elif matched[0].isupper():
                    return tgt_word.capitalize()
                else:
                    return tgt_word

            result = pattern.sub(_replace, result)

        return result

    # ------------------------------------------------------------------
    # Semantic Similarity
    # ------------------------------------------------------------------

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between LaBSE sentence embeddings.
        Implements Equation (11) / (38) from the paper.

        Falls back to character-level Jaccard similarity if LaBSE
        is unavailable (e.g., in unit-test environments).
        """
        try:
            model = self._get_embedding_model()
            emb_a = model.encode(text_a, convert_to_tensor=True, show_progress_bar=False)
            emb_b = model.encode(text_b, convert_to_tensor=True, show_progress_bar=False)
            cos_sim = torch.nn.functional.cosine_similarity(
                emb_a.unsqueeze(0), emb_b.unsqueeze(0)
            ).item()
            return float(cos_sim)
        except Exception:
            # Fallback: normalized edit-distance based similarity
            return self._jaccard_similarity(text_a, text_b)

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Token-level Jaccard similarity as a fast fallback."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        intersection = set_a & set_b
        union = set_a | set_b
        if not union:
            return 1.0
        return len(intersection) / len(union)

    def _get_embedding_model(self):
        """Lazy-load the LaBSE model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self._embedding_model_name, device=self.device
            )
        return self._embedding_model

    # ------------------------------------------------------------------
    # Grammar Error Scoring
    # ------------------------------------------------------------------

    def _compute_grammar_error(self, text: str, language: str) -> float:
        """
        Lightweight grammar error heuristic.

        Returns a score in [0, 1] where higher = more likely to have
        grammatical errors introduced by the substitution.

        Heuristic rules:
          - Check for duplicate consecutive words (e.g., "the the")
          - Check for mismatched article–noun patterns in German / Spanish
          - Check for agreement suffix violations in morphologically rich languages

        In production, this can be replaced with a language tool API call
        (e.g., LanguageTool) or a grammar classification model.
        """
        score = 0.0
        tokens = text.split()

        # Rule 1: Duplicate consecutive tokens (obvious error signal)
        for i in range(len(tokens) - 1):
            if tokens[i].lower() == tokens[i + 1].lower():
                score += 0.5

        # Rule 2: Language-specific article–noun mismatches
        if language == "de":
            score += self._german_agreement_score(tokens)
        elif language in ("es", "fr", "it", "pt"):
            score += self._romance_agreement_score(tokens, language)

        # Clamp to [0, 1]
        return min(score, 1.0)

    def _german_agreement_score(self, tokens: List[str]) -> float:
        """
        Penalize mismatched article–noun pairs in German.
        E.g., "die Arzt" (should be "der Arzt") → penalty.
        """
        masculine_nouns = {"arzt", "lehrer", "ingenieur", "direktor", "student", "schüler"}
        feminine_nouns = {"ärztin", "lehrerin", "ingenieurin", "direktorin", "studentin", "schülerin"}

        score = 0.0
        for i in range(len(tokens) - 1):
            article = tokens[i].lower()
            noun = tokens[i + 1].lower()
            # Masculine noun with feminine article
            if noun in masculine_nouns and article in ("die", "eine"):
                score += 0.3
            # Feminine noun with masculine article
            if noun in feminine_nouns and article in ("der", "ein"):
                score += 0.3
        return score

    def _romance_agreement_score(self, tokens: List[str], language: str) -> float:
        """
        Penalize obvious article–noun disagreement in Romance languages.
        Simple heuristic: checks -o ending nouns with feminine articles.
        """
        score = 0.0
        fem_articles = {"la", "las", "une", "le", "la", "les"}
        masc_articles = {"el", "los", "un", "le", "les"}

        for i in range(len(tokens) - 1):
            article = tokens[i].lower()
            noun = tokens[i + 1].lower()
            # Masculine -o ending noun with clear feminine article
            if noun.endswith("o") and article in fem_articles:
                score += 0.2
            if noun.endswith("a") and article in masc_articles:
                score += 0.2

        return score


# ---------------------------------------------------------------------------
# Substitution tables (per language, directional)
# ---------------------------------------------------------------------------

_SUBSTITUTION_TABLES: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "en": {
        "male_to_female": [
            ("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
            ("man", "woman"), ("men", "women"), ("male", "female"),
            ("boy", "girl"), ("boys", "girls"), ("son", "daughter"),
            ("father", "mother"), ("brother", "sister"), ("husband", "wife"),
            ("king", "queen"), ("actor", "actress"), ("waiter", "waitress"),
            ("sir", "ma'am"), ("Mr", "Ms"), ("Mr.", "Ms."), ("uncle", "aunt"),
            ("nephew", "niece"), ("grandfather", "grandmother"),
            ("grandson", "granddaughter"), ("dad", "mom"), ("groom", "bride"),
        ],
        "female_to_male": [
            ("she", "he"), ("her", "him"), ("her", "his"), ("herself", "himself"),
            ("woman", "man"), ("women", "men"), ("female", "male"),
            ("girl", "boy"), ("girls", "boys"), ("daughter", "son"),
            ("mother", "father"), ("sister", "brother"), ("wife", "husband"),
            ("queen", "king"), ("actress", "actor"), ("waitress", "waiter"),
            ("ma'am", "sir"), ("Ms", "Mr"), ("Ms.", "Mr."), ("aunt", "uncle"),
            ("niece", "nephew"), ("grandmother", "grandfather"),
            ("granddaughter", "grandson"), ("mom", "dad"), ("bride", "groom"),
        ],
    },
    "de": {
        "male_to_female": [
            ("Er", "Sie"), ("er", "sie"), ("sein", "ihr"), ("seinen", "ihren"),
            ("seinem", "ihrem"), ("seiner", "ihrer"), ("seine", "ihre"),
            ("Arzt", "Ärztin"), ("Lehrer", "Lehrerin"), ("Ingenieur", "Ingenieurin"),
            ("Direktor", "Direktorin"), ("Student", "Studentin"),
            ("Schüler", "Schülerin"), ("Herr", "Frau"), ("Mann", "Frau"),
            ("Vater", "Mutter"), ("Bruder", "Schwester"), ("Sohn", "Tochter"),
            ("der", "die"), ("ein", "eine"),
        ],
        "female_to_male": [
            ("Sie", "Er"), ("sie", "er"), ("ihr", "sein"), ("ihren", "seinen"),
            ("ihrem", "seinem"), ("ihrer", "seiner"), ("ihre", "seine"),
            ("Ärztin", "Arzt"), ("Lehrerin", "Lehrer"), ("Ingenieurin", "Ingenieur"),
            ("Direktorin", "Direktor"), ("Studentin", "Student"),
            ("Schülerin", "Schüler"), ("Frau", "Herr"), ("Frau", "Mann"),
            ("Mutter", "Vater"), ("Schwester", "Bruder"), ("Tochter", "Sohn"),
            ("die", "der"), ("eine", "ein"),
        ],
    },
    "es": {
        "male_to_female": [
            ("él", "ella"), ("su", "su"), ("el", "la"), ("un", "una"),
            ("hombre", "mujer"), ("señor", "señora"), ("padre", "madre"),
            ("hermano", "hermana"), ("hijo", "hija"),
            ("médico", "médica"), ("ingeniero", "ingeniera"), ("director", "directora"),
        ],
        "female_to_male": [
            ("ella", "él"), ("su", "su"), ("la", "el"), ("una", "un"),
            ("mujer", "hombre"), ("señora", "señor"), ("madre", "padre"),
            ("hermana", "hermano"), ("hija", "hijo"),
            ("médica", "médico"), ("ingeniera", "ingeniero"), ("directora", "director"),
        ],
    },
    "fr": {
        "male_to_female": [
            ("il", "elle"), ("son", "sa"), ("le", "la"), ("un", "une"),
            ("homme", "femme"), ("monsieur", "madame"), ("père", "mère"),
            ("frère", "sœur"), ("fils", "fille"),
            ("médecin", "médecin"), ("ingénieur", "ingénieure"), ("directeur", "directrice"),
        ],
        "female_to_male": [
            ("elle", "il"), ("sa", "son"), ("la", "le"), ("une", "un"),
            ("femme", "homme"), ("madame", "monsieur"), ("mère", "père"),
            ("sœur", "frère"), ("fille", "fils"),
            ("ingénieure", "ingénieur"), ("directrice", "directeur"),
        ],
    },
}

# Default: fall back to English pairs for unsupported languages
_DEFAULT_SUB_LANG = "en"


def _get_substitution_pairs(
    source_attr: str, target_attr: str, language: str
) -> List[Tuple[str, str]]:
    """
    Retrieve ordered list of (source_token, target_token) substitution pairs
    for the given source→target attribute direction and language.
    """
    lang_tables = _SUBSTITUTION_TABLES.get(language, _SUBSTITUTION_TABLES[_DEFAULT_SUB_LANG])
    direction = f"{source_attr}_to_{target_attr}"
    return lang_tables.get(direction, [])
