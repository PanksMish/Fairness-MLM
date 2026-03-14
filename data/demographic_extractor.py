"""
demographic_extractor.py
========================
Hybrid demographic attribute extractor as described in paper Section 3.5.

Implements a two-component detection pipeline:
  1. Rule-based lexicon: language-specific gendered word lists,
     occupational nouns, and morphological markers.
  2. Supervised classifier stub: a lightweight logistic regression trained
     on annotated attribute labels (for production; falls back to rule-based).

The extractor returns:
  - detected_attribute:  str  (e.g., "male", "female", "neutral")
  - attribute_tokens:    List[Tuple[int, str]]  — (token_index, token) pairs
  - confidence:          float in [0, 1]
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DemographicDetectionResult:
    """Result returned by the DemographicExtractor."""
    detected_attribute: str          # "male", "female", "neutral", "unknown"
    attribute_tokens: List[Tuple[int, str]] = field(default_factory=list)
    confidence: float = 0.0
    language: str = "en"


# ---------------------------------------------------------------------------
# Language-specific lexicons
# These cover gender-marked tokens for morphologically diverse languages.
# Extended to handle agreement patterns where modifying one token requires
# cascading changes (e.g., German determiners + nouns + adjectives).
# ---------------------------------------------------------------------------

GENDER_LEXICONS: Dict[str, Dict[str, List[str]]] = {
    # English
    "en": {
        "male": [
            "he", "him", "his", "himself", "man", "men", "male", "boy", "boys",
            "son", "father", "brother", "husband", "king", "actor", "waiter",
            "gentleman", "gentlemen", "sir", "mr", "mr.", "uncle", "nephew",
            "grandfather", "grandson", "dad", "groom", "bachelor",
        ],
        "female": [
            "she", "her", "hers", "herself", "woman", "women", "female", "girl", "girls",
            "daughter", "mother", "sister", "wife", "queen", "actress", "waitress",
            "lady", "ladies", "ma'am", "ms", "ms.", "mrs", "mrs.", "miss", "aunt",
            "niece", "grandmother", "granddaughter", "mom", "bride", "bachelorette",
        ],
    },
    # German — gendered nouns and determiners
    "de": {
        "male": [
            "er", "ihn", "ihm", "sein", "seiner", "seinem", "seinen", "seine",
            "mann", "männer", "herr", "vater", "bruder", "sohn", "onkel",
            "arzt", "lehrer", "ingenieur", "direktor", "student", "schüler",
            "der",  # masculine definite article (context-dependent)
            "ein",  # masculine indefinite article
        ],
        "female": [
            "sie", "ihr", "ihrem", "ihren", "ihre", "ihrer",
            "frau", "frauen", "mutter", "schwester", "tochter", "tante",
            "ärztin", "lehrerin", "ingenieurin", "direktorin", "studentin", "schülerin",
            "die",  # feminine definite article
            "eine",  # feminine indefinite article
        ],
    },
    # Spanish — gendered nouns and articles
    "es": {
        "male": [
            "él", "su", "sus", "el",
            "hombre", "hombres", "señor", "padre", "hermano", "hijo",
            "médico", "ingeniero", "director", "estudiante",
            "un",
        ],
        "female": [
            "ella", "su", "sus", "la",
            "mujer", "mujeres", "señora", "madre", "hermana", "hija",
            "médica", "ingeniera", "directora",
            "una",
        ],
    },
    # French
    "fr": {
        "male": [
            "il", "lui", "son", "sa", "ses",
            "homme", "hommes", "monsieur", "père", "frère", "fils",
            "médecin", "ingénieur", "directeur", "étudiant",
            "le", "un",
        ],
        "female": [
            "elle", "lui", "son", "sa", "ses",
            "femme", "femmes", "madame", "mère", "sœur", "fille",
            "médecin", "ingénieure", "directrice", "étudiante",
            "la", "une",
        ],
    },
    # Arabic (romanized check for surface forms)
    "ar": {
        "male": ["هو", "رجل", "ولد", "أب", "أخ", "ابن", "طبيب", "مهندس"],
        "female": ["هي", "امرأة", "بنت", "أم", "أخت", "ابنة", "طبيبة", "مهندسة"],
    },
    # Hindi (Devanagari)
    "hi": {
        "male": ["वह", "उसने", "उसका", "पुरुष", "लड़का", "पिता", "भाई"],
        "female": ["वह", "उसने", "उसकी", "महिला", "लड़की", "माता", "बहन"],
    },
    # Turkish — no grammatical gender; detect by lexical forms only
    "tr": {
        "male": ["adam", "erkek", "baba", "erkek kardeş", "oğul", "bay"],
        "female": ["kadın", "kız", "anne", "kız kardeş", "kız çocuğu", "bayan"],
    },
}

# For languages not explicitly in the lexicon, fall back to English
_DEFAULT_LANG = "en"


class DemographicExtractor:
    """
    Hybrid demographic attribute extractor.

    Detection pipeline (Section 3.5):
      1. Normalize text with Unicode NFKC.
      2. Tokenize into whitespace-delimited words.
      3. Apply language-specific lexicon matching (case-insensitive).
      4. For morphologically rich languages, also check suffix patterns.
      5. Aggregate evidence and return the dominant attribute + confidence.

    Attributes
    ----------
    languages : List[str]
        Languages to support.
    use_suffix_rules : bool
        If True, apply suffix-based agreement markers for morphologically
        rich languages (Turkish, Finnish, Arabic, etc.).
    """

    def __init__(self, languages: Optional[List[str]] = None, use_suffix_rules: bool = True):
        self.languages = languages
        self.use_suffix_rules = use_suffix_rules

        # Build compiled regex patterns for each language lexicon
        self._patterns: Dict[str, Dict[str, re.Pattern]] = {}
        for lang, attrib_dict in GENDER_LEXICONS.items():
            self._patterns[lang] = {}
            for attr, words in attrib_dict.items():
                # Word-boundary match, case-insensitive
                pattern = r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b'
                self._patterns[lang][attr] = re.compile(pattern, re.IGNORECASE | re.UNICODE)

    def detect(self, text: str, language: str = "en") -> DemographicDetectionResult:
        """
        Detect the primary demographic attribute in the given text.

        Parameters
        ----------
        text : str
            Input text (will be NFKC-normalized internally).
        language : str
            ISO 639-1 language code.

        Returns
        -------
        DemographicDetectionResult
        """
        # Normalize
        text_norm = unicodedata.normalize("NFKC", text)

        # Resolve lexicon language (fall back to English)
        lex_lang = language if language in self._patterns else _DEFAULT_LANG

        patterns = self._patterns[lex_lang]
        scores: Dict[str, float] = {"male": 0.0, "female": 0.0}
        attribute_tokens: List[Tuple[int, str]] = []

        tokens = text_norm.split()

        for attr, pattern in patterns.items():
            matches = list(pattern.finditer(text_norm))
            for match in matches:
                scores[attr] += 1.0
                # Map character offset back to approximate token index
                char_start = match.start()
                char_so_far = 0
                for tok_idx, tok in enumerate(tokens):
                    char_so_far += len(tok) + 1
                    if char_so_far > char_start:
                        attribute_tokens.append((tok_idx, match.group()))
                        break

        # Optional suffix-based rules for morphologically rich languages
        if self.use_suffix_rules:
            suffix_scores = self._suffix_check(tokens, language)
            for attr in scores:
                scores[attr] += suffix_scores.get(attr, 0.0)

        # Determine dominant attribute
        total = scores["male"] + scores["female"]
        if total == 0:
            return DemographicDetectionResult(
                detected_attribute="neutral",
                attribute_tokens=[],
                confidence=1.0,
                language=language,
            )

        dominant = max(scores, key=lambda k: scores[k])
        confidence = scores[dominant] / total

        return DemographicDetectionResult(
            detected_attribute=dominant,
            attribute_tokens=attribute_tokens,
            confidence=confidence,
            language=language,
        )

    def _suffix_check(self, tokens: List[str], language: str) -> Dict[str, float]:
        """
        Apply suffix-based heuristics for morphologically rich languages.

        For example:
          - Turkish: -ın / -in / -ün (genitive masculine contexts)
          - Slavic languages: feminine noun endings (-а, -я, -ка)
          - Semitic: feminine suffix -ah / -at

        Returns a score increment dict for "male" and "female".
        """
        scores: Dict[str, float] = {"male": 0.0, "female": 0.0}

        # Slavic feminine noun endings (Russian, Polish, Ukrainian, Serbian, etc.)
        if language in ["ru", "pl", "uk", "sr", "bg", "cs", "sk", "hr"]:
            for tok in tokens:
                # Feminine: ends in -а or -я (Cyrillic) or -a / -ia (Latin)
                if re.search(r'[аяАЯ]$', tok) or re.search(r'[aä]$', tok, re.IGNORECASE):
                    scores["female"] += 0.3
                # Masculine: ends in hard consonant or -й / -ий
                elif re.search(r'[йЙ]$|ий$|ый$', tok):
                    scores["male"] += 0.3

        # Finnish: -nen ending often indicates feminine-associated professions
        if language == "fi":
            for tok in tokens:
                if tok.endswith("nen") and len(tok) > 5:
                    scores["female"] += 0.2

        # Arabic: feminine marker taa marbouta -ة
        if language == "ar":
            for tok in tokens:
                if tok.endswith("ة"):
                    scores["female"] += 0.5
                elif tok.endswith("ون") or tok.endswith("ين"):
                    scores["male"] += 0.3

        # Hebrew: feminine endings -ה or -ת
        if language == "he":
            for tok in tokens:
                if tok.endswith("ה") or tok.endswith("ת"):
                    scores["female"] += 0.3

        return scores

    def get_substitution_pairs(self, language: str = "en") -> List[Tuple[str, str]]:
        """
        Return the list of (male_token, female_token) substitution pairs
        for a given language. Used by the counterfactual generator.
        """
        lex_lang = language if language in GENDER_LEXICONS else _DEFAULT_LANG
        lexicon = GENDER_LEXICONS[lex_lang]
        male_words = lexicon.get("male", [])
        female_words = lexicon.get("female", [])
        # Zip into pairs (trimmed to shortest list)
        pairs = list(zip(male_words[:len(female_words)], female_words[:len(male_words)]))
        return pairs

    def batch_detect(
        self, texts: List[str], languages: List[str]
    ) -> List[DemographicDetectionResult]:
        """
        Run detect() over a batch of texts.

        Parameters
        ----------
        texts : List[str]
        languages : List[str]  — must be same length as texts

        Returns
        -------
        List[DemographicDetectionResult]
        """
        assert len(texts) == len(languages), "texts and languages must have same length"
        return [self.detect(t, l) for t, l in zip(texts, languages)]
