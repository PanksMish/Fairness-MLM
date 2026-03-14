"""
language_utils.py
=================
Language metadata utilities for ADAPT-BTS.

Provides:
  - Language family classification (Indo-European, Niger-Congo, etc.)
  - Typological category (analytic, fusional, agglutinative, highly inflectional)
  - Resource tier stratification (HR / MR / LR)
  - Linguistic Coverage Score (LCS) computation (Equation 37)
"""

from typing import Dict, List, Optional

# Language family groups (Figure 1b in paper)
LANGUAGE_FAMILIES: Dict[str, str] = {
    # Indo-European (38 languages)
    "en": "Indo-European", "de": "Indo-European", "fr": "Indo-European",
    "es": "Indo-European", "it": "Indo-European", "pt": "Indo-European",
    "ru": "Indo-European", "pl": "Indo-European", "uk": "Indo-European",
    "nl": "Indo-European", "sv": "Indo-European", "cs": "Indo-European",
    "sk": "Indo-European", "hr": "Indo-European", "sr": "Indo-European",
    "bg": "Indo-European", "ro": "Indo-European", "el": "Indo-European",
    "hi": "Indo-European", "bn": "Indo-European", "ur": "Indo-European",
    "mr": "Indo-European", "gu": "Indo-European", "ne": "Indo-European",
    "si": "Indo-European", "fa": "Indo-European", "ps": "Indo-European",
    "is": "Indo-European", "lv": "Indo-European", "lt": "Indo-European",
    "hy": "Indo-European", "be": "Indo-European", "mk": "Indo-European",
    "gl": "Indo-European", "ca": "Indo-European", "eu": "Indo-European",  # Basque isolate
    "sq": "Indo-European", "bs": "Indo-European", "cy": "Indo-European",
    "ga": "Indo-European", "fo": "Indo-European", "co": "Indo-European",
    # Niger-Congo (18 languages)
    "sw": "Niger-Congo", "yo": "Niger-Congo", "ha": "Niger-Congo",
    "ig": "Niger-Congo", "sn": "Niger-Congo", "zu": "Niger-Congo",
    "xh": "Niger-Congo", "so": "Niger-Congo", "am": "Niger-Congo",
    "ti": "Niger-Congo", "st": "Niger-Congo", "rw": "Niger-Congo",
    "wo": "Niger-Congo", "mg": "Niger-Congo", "af": "Niger-Congo",
    # Sino-Tibetan (14 languages)
    "zh": "Sino-Tibetan", "ja": "Sino-Tibetan", "km": "Sino-Tibetan",
    "lo": "Sino-Tibetan", "mn": "Sino-Tibetan", "brx": "Sino-Tibetan",
    # Afroasiatic (12)
    "ar": "Afroasiatic", "he": "Afroasiatic", "mt": "Afroasiatic",
    "ku": "Afroasiatic",
    # Austronesian (11)
    "id": "Austronesian", "ms": "Austronesian", "tl": "Austronesian",
    "jv": "Austronesian", "su": "Austronesian", "mi": "Austronesian",
    "sm": "Austronesian",
    # Dravidian (8)
    "ta": "Dravidian", "te": "Dravidian", "kn": "Dravidian",
    # Turkic
    "tr": "Turkic", "az": "Turkic", "kk": "Turkic", "uz": "Turkic",
    "tk": "Turkic", "ug": "Turkic",
    # Others
    "vi": "Austroasiatic", "th": "Tai-Kadai", "ko": "Koreanic",
    "fi": "Uralic", "hu": "Uralic", "et": "Uralic",
    "ka": "Kartvelian", "ht": "Creole",
    "gn": "Tupian", "qu": "Quechuan",
    "kl": "Eskimo-Aleut", "lb": "Germanic",
    "new": "Sino-Tibetan", "dgo": "Indo-European", "tg": "Turkic",
}

# Typological categories (Section 5.3)
TYPOLOGICAL_CATEGORIES: Dict[str, str] = {
    # Analytic: minimal inflectional morphology
    "zh": "analytic", "vi": "analytic", "th": "analytic", "km": "analytic",
    "lo": "analytic", "id": "analytic", "ms": "analytic", "tl": "analytic",
    "jv": "analytic", "su": "analytic",
    # Fusional: moderate inflection combining grammatical features
    "en": "fusional", "de": "fusional", "fr": "fusional", "es": "fusional",
    "it": "fusional", "pt": "fusional", "nl": "fusional", "sv": "fusional",
    "ru": "fusional", "pl": "fusional", "uk": "fusional", "be": "fusional",
    "cs": "fusional", "sk": "fusional", "hr": "fusional", "sr": "fusional",
    "bg": "fusional", "ro": "fusional", "el": "fusional", "hi": "fusional",
    "bn": "fusional", "ur": "fusional", "mr": "fusional", "gu": "fusional",
    "ne": "fusional", "fa": "fusional", "he": "fusional", "ar": "fusional",
    "lv": "fusional", "lt": "fusional", "is": "fusional", "hy": "fusional",
    "mk": "fusional", "bs": "fusional", "af": "fusional", "ca": "fusional",
    "gl": "fusional", "ga": "fusional", "cy": "fusional", "fo": "fusional",
    "sq": "fusional", "mt": "fusional", "co": "fusional",
    # Agglutinative: extensive affixation with separable morphemes
    "tr": "agglutinative", "fi": "agglutinative", "hu": "agglutinative",
    "et": "agglutinative", "az": "agglutinative", "kk": "agglutinative",
    "uz": "agglutinative", "tk": "agglutinative", "ug": "agglutinative",
    "sw": "agglutinative", "yo": "agglutinative", "ha": "agglutinative",
    "ig": "agglutinative", "zu": "agglutinative", "xh": "agglutinative",
    "sn": "agglutinative", "st": "agglutinative", "so": "agglutinative",
    "am": "agglutinative", "ti": "agglutinative", "rw": "agglutinative",
    "wo": "agglutinative", "tl": "agglutinative", "mi": "agglutinative",
    "sm": "agglutinative", "ja": "agglutinative", "ko": "agglutinative",
    "ka": "agglutinative", "lb": "agglutinative", "jv": "agglutinative",
    "tg": "agglutinative", "gn": "agglutinative", "qu": "agglutinative",
    "kl": "agglutinative", "ht": "analytic",
    # Highly inflectional: complex agreement systems and rich morphology
    "ar": "highly_inflectional", "fi": "highly_inflectional",
    "si": "highly_inflectional", "kn": "highly_inflectional",
    "ta": "highly_inflectional", "te": "highly_inflectional",
    "mn": "highly_inflectional", "ps": "highly_inflectional",
    "ku": "highly_inflectional",
}


def get_language_family(lang: str) -> str:
    """Return the language family for a given language code."""
    return LANGUAGE_FAMILIES.get(lang, "Other")


def get_typological_category(lang: str) -> str:
    """Return the typological category (analytic/fusional/agglutinative/highly_inflectional)."""
    return TYPOLOGICAL_CATEGORIES.get(lang, "fusional")


def compute_linguistic_coverage_score(languages: List[str]) -> float:
    """
    Compute the Linguistic Coverage Score (LCS) from Equation 37:

        LCS = (1/K) Σ_{k=1}^{K} n_k / N

    where K = number of typological categories, n_k = languages in category k,
    N = total languages.

    A perfectly balanced dataset would have LCS = 1/K for each category
    (uniform distribution across typological types).
    """
    from collections import Counter
    if not languages:
        return 0.0

    categories = [get_typological_category(l) for l in languages]
    counts = Counter(categories)
    K = len(counts)
    N = len(languages)

    lcs = (1.0 / K) * sum(n / N for n in counts.values())
    return float(lcs)


def get_languages_by_family() -> Dict[str, List[str]]:
    """Group all languages by their family."""
    from collections import defaultdict
    groups: Dict[str, List[str]] = defaultdict(list)
    for lang, family in LANGUAGE_FAMILIES.items():
        groups[family].append(lang)
    return dict(groups)


def get_languages_by_typology() -> Dict[str, List[str]]:
    """Group all languages by typological category."""
    from collections import defaultdict
    groups: Dict[str, List[str]] = defaultdict(list)
    for lang, typ in TYPOLOGICAL_CATEGORIES.items():
        groups[typ].append(lang)
    return dict(groups)
