"""
============================================================================
READ THIS BEFORE USING ANYTHING IN THIS FILE
============================================================================

This module produces WEAK, HEURISTIC sentiment labels via lexicon-based
polarity scoring (a real, documented technique in low-resource NLP --
see e.g. Taboada et al. 2011, "Lexicon-Based Methods for Sentiment
Analysis," Computational Linguistics) applied to unlabeled CC100 text,
for languages where no real labeled sentiment dataset exists (i.e. all
but the ~6 languages `datasets/download_sentiment.py`'s Amazon
Reviews/TweetEval sources actually cover -- see README.md's dataset
scope note).

WHAT THIS IS: a way to get SOME training signal in languages with zero
real labels, using word-level positive/negative associations from the
NRC Emotion Lexicon's machine-translated word lists (datasets/nrc_lexicon.py).

WHAT THIS IS NOT:
  - NOT gold-standard sentiment labels. Nobody read these sentences and
    judged their sentiment. A sentence gets "positive" because it
    happened to contain more lexicon-flagged positive words than
    negative ones -- this misses negation, sarcasm, mixed sentiment,
    domain-specific usage, and anything the (machine-translated, not
    human-verified per language) lexicon doesn't cover.
  - NOT comparable to the manuscript's reported results. Table 5's
    85.6 Macro-F1 etc. were (presumably) measured against real human
    labels. A model evaluated against WEAK labels measures agreement
    with a heuristic, not real sentiment classification accuracy.
    Reporting a Macro-F1 computed this way as if it were comparable to
    Table 5 would be a serious misrepresentation.
  - NOT sufficient by itself to claim "101-language reproduction."

EVERY record this module produces carries `"label_source":
"weak_lexicon_nrc"` and `"is_gold_label": False` so it can never be
silently confused with real labeled data downstream -- check for this
field before trusting any Macro-F1 computed on this data. Output paths
default to a separate `*_weak` directory precisely so it can't collide
with `datasets/build_sentiment.py`'s real `train.jsonl`.
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets.nrc_lexicon import LanguageLexicon


@dataclass
class WeakLabelResult:
    label: str                    # "positive" | "negative" | "neutral"
    confidence: float             # 0.0-1.0, NOT a calibrated probability -- see docstring below
    n_positive_words: int
    n_negative_words: int
    n_tokens_covered: int         # tokens that matched the lexicon at all
    n_tokens_total: int


def lexicon_polarity_score(
    tokens: list[str],
    lexicon: LanguageLexicon,
    neutral_band: float = 0.15,
) -> WeakLabelResult:
    """
    Scores a tokenized sentence's polarity by counting lexicon-flagged
    positive vs. negative words.

        score = (n_pos - n_neg) / (n_pos + n_neg)   if n_pos + n_neg > 0
        score = 0                                    otherwise (no coverage)

    label = "positive" if score > neutral_band
          = "negative" if score < -neutral_band
          = "neutral"  otherwise (includes zero-coverage sentences)

    `confidence` is `|score|` -- a rough measure of how lopsided the
    lexicon-word count was, NOT a calibrated probability of correctness.
    A sentence with 1 positive word and 0 negative words gets
    confidence=1.0 despite that being extremely weak evidence; callers
    wanting to filter low-quality weak labels should probably threshold
    on `n_tokens_covered` (coverage) as much as or more than on
    `confidence` (lopsidedness).
    """
    token_set_lower = [t.lower() for t in tokens]
    n_pos = sum(1 for t in token_set_lower if t in lexicon.positive_words)
    n_neg = sum(1 for t in token_set_lower if t in lexicon.negative_words)
    n_covered = n_pos + n_neg

    if n_covered == 0:
        score = 0.0
    else:
        score = (n_pos - n_neg) / n_covered

    if score > neutral_band:
        label = "positive"
    elif score < -neutral_band:
        label = "negative"
    else:
        label = "neutral"

    return WeakLabelResult(
        label=label, confidence=abs(score),
        n_positive_words=n_pos, n_negative_words=n_neg,
        n_tokens_covered=n_covered, n_tokens_total=len(tokens),
    )


def build_weak_labeled_records(
    texts: list[str],
    language: str,
    lexicon: LanguageLexicon,
    min_coverage: int = 1,
    min_confidence: float = 0.0,
    neutral_band: float = 0.15,
) -> tuple[list[dict], dict]:
    """
    Turns a list of raw (unlabeled) CC100 text paragraphs into
    weak-labeled records, filtering out sentences with too little
    lexicon coverage to trust at all (`min_coverage`) and optionally too
    little polarity lopsidedness (`min_confidence`).

    Every returned record has the loud, unmissable metadata described in
    this module's top-level docstring.

    Returns:
        (records, stats) -- stats summarizes coverage/label distribution
        so you can see, before training anything, how much signal this
        language's lexicon actually provided (a language where the
        NRC translation has poor coverage will produce mostly
        "neutral"/filtered-out records, and that's a sign this
        language's weak labels are especially unreliable, not a bug).
    """
    records = []
    n_filtered_low_coverage = 0
    n_filtered_low_confidence = 0
    label_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for text in texts:
        tokens = text.split()
        result = lexicon_polarity_score(tokens, lexicon, neutral_band=neutral_band)

        if result.n_tokens_covered < min_coverage:
            n_filtered_low_coverage += 1
            continue
        if result.confidence < min_confidence:
            n_filtered_low_confidence += 1
            continue

        label_counts[result.label] += 1
        records.append({
            "text": text,
            "label": result.label,
            "language": language,
            "label_source": "weak_lexicon_nrc",
            "is_gold_label": False,
            "weak_label_confidence": result.confidence,
            "weak_label_coverage": result.n_tokens_covered,
        })

    stats = {
        "language": language,
        "n_input_texts": len(texts),
        "n_output_records": len(records),
        "n_filtered_low_coverage": n_filtered_low_coverage,
        "n_filtered_low_confidence": n_filtered_low_confidence,
        "label_distribution": label_counts,
        "lexicon_size": len(lexicon),
    }
    return records, stats
