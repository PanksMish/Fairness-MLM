"""
Starter demographic attribute dictionaries for LexiconSubstitutor (HR
tier, Sec 4.1: "lexicon-based substitutions based on the curated
demographic dictionaries").

HONESTLY SCOPED: this is a small, illustrative gender-attribute
dictionary for English only, meant to make the pipeline runnable
end-to-end on at least one real language. The manuscript's actual
experiments would need curated dictionaries for every high-resource
language (Sec 3.1's HR tier, ~18 languages) and ideally more attribute
dimensions than binary gender (dialect/sociolect markers, per Sec 3.2's
definition of attribute set A). Building that out is real linguistic
curation work, not something to fabricate here -- treat this as a
starting template to extend, not a claim of completeness.
"""

from __future__ import annotations

# English binary-gender pronoun/noun substitution pairs.
ENGLISH_GENDER_DICT = {
    "m": {
        "f": {
            "he": "she", "him": "her", "his": "her", "himself": "herself",
            "man": "woman", "men": "women", "boy": "girl", "boys": "girls",
            "father": "mother", "dad": "mom", "husband": "wife",
            "brother": "sister", "son": "daughter", "king": "queen",
            "actor": "actress", "gentleman": "lady", "sir": "madam",
            "mr": "ms", "uncle": "aunt", "nephew": "niece",
            "grandfather": "grandmother", "waiter": "waitress",
        },
    },
    "f": {
        "m": {
            "she": "he", "her": "his", "hers": "his", "herself": "himself",
            "woman": "man", "women": "men", "girl": "boy", "girls": "boys",
            "mother": "father", "mom": "dad", "wife": "husband",
            "sister": "brother", "daughter": "son", "queen": "king",
            "actress": "actor", "lady": "gentleman", "madam": "sir",
            "ms": "mr", "aunt": "uncle", "niece": "nephew",
            "grandmother": "grandfather", "waitress": "waiter",
        },
    },
}

# Attribute seed words: the minimal lexical cues used both for
# ATTRIBUTE DETECTION (LexiconAttributeDetector) and for the MR-tier
# EmbeddingAlignmentTransform's candidate-word identification. Kept
# smaller than the full substitution dictionary above, deliberately,
# since detection only needs a few strong signals, not exhaustive
# coverage.
ENGLISH_GENDER_SEEDS = {
    "m": ["he", "him", "his", "man", "father", "husband", "brother"],
    "f": ["she", "her", "hers", "woman", "mother", "wife", "sister"],
}

# Per-language token counts used by TieredGenerationStrategy /
# resource_category() dispatch, matching the illustrative starter set in
# configs/default_config.yaml. These are NOT measured from a real CC100
# corpus -- they're placeholder orders-of-magnitude consistent with
# Sec 3.1's HR/MR/LR thresholds, for pipeline smoke-testing only. Replace
# with real per-language token counts measured from your actual corpus.
EXAMPLE_TOKEN_COUNTS = {
    "en": 5.0e9,   # HR
    "de": 3.0e9,   # HR
    "fr": 2.5e9,   # HR
    "sw": 4.0e7,   # LR
    "hi": 3.0e8,   # MR
}
