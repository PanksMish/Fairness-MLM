"""
Full sentiment dataset build pipeline (the "SENTIMENT DATASET" pipeline
diagram: Download -> Cleaning -> Deduplication -> Unicode normalization
-> Language verification -> Tokenization -> Split -> JSONL).

    python datasets/build_sentiment.py \\
        --languages en de fr sw hi \\
        --out-dir data/processed/sentiment

Tokenization is deliberately left as a separate, later step
(datasets/tokenizer.py, applied at dataloader time) rather than baked
into the stored JSONL -- storing raw cleaned text and tokenizing
on-the-fly is what lets the same processed dataset serve both mT5 and
XLM-R (configs/mt5.yaml vs configs/xlmr.yaml) without rebuilding data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.download_sentiment import build_raw_sentiment_corpus
from datasets.dataset_utils import clean_pipeline, train_val_test_split, write_splits_jsonl, SplitRatios
from datasets.language_filter import filter_by_declared_language, LanguageFilterConfig

logger = logging.getLogger(__name__)


def get_default_language_detector():
    """
    Wires in a real FastText language-ID model if available, otherwise
    raises with an actionable message rather than silently skipping
    language verification (which the manuscript's pipeline treats as a
    required step, not optional).

    Requires: pip install fasttext-langdetect  (or swap in your own
    LanguageDetector-compatible callable, e.g. py3langid).
    """
    try:
        from ftlangdetect import detect  # fasttext-langdetect package
    except ImportError as e:
        raise ImportError(
            "No language detector available. Install one of:\n"
            "  pip install fasttext-langdetect\n"
            "  pip install py3langid\n"
            "and wire it into get_default_language_detector() to match the "
            "LanguageDetector protocol in datasets/language_filter.py "
            "(text -> (lang_code, confidence))."
        ) from e

    def _detector(text: str) -> tuple[str, float]:
        result = detect(text=text.replace("\n", " "), low_memory=True)
        return result["lang"], float(result["score"])

    return _detector


def build(
    languages: list[str],
    out_dir: str,
    cache_dir: str = "data/cache/sentiment",
    split_ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
    skip_language_verification: bool = False,
) -> dict:
    raw = build_raw_sentiment_corpus(languages, cache_dir)
    logger.info("Raw: %d records", len(raw))

    cleaned = clean_pipeline(raw, text_field="text", min_chars=3)
    logger.info("After cleaning + dedup: %d records (%d removed)", len(cleaned), len(raw) - len(cleaned))

    if skip_language_verification:
        logger.warning("Skipping language verification (--skip-language-verification set)")
        verified = cleaned
    else:
        detector = get_default_language_detector()
        verified, rejected = filter_by_declared_language(
            cleaned, detector, config=LanguageFilterConfig(min_confidence=0.7)
        )
        logger.info("After language verification: %d kept, %d rejected", len(verified), len(rejected))

    train, val, test = train_val_test_split(
        verified, split_ratios, seed=seed, stratify_field="language"
    )
    counts = write_splits_jsonl(train, val, test, out_dir)
    logger.info("Wrote splits to %s: %s", out_dir, counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the multilingual sentiment dataset.")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/processed/sentiment")
    parser.add_argument("--cache-dir", default="data/cache/sentiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-language-verification", action="store_true",
                         help="Skip the FastText-based language check (faster, but "
                              "not what the manuscript's pipeline specifies -- use "
                              "only for quick smoke tests).")
    args = parser.parse_args()

    counts = build(
        languages=args.languages,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        skip_language_verification=args.skip_language_verification,
    )
    print(counts)
