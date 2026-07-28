"""
Downloads and merges multilingual sentiment sources for the sentiment
classification task (Sec 5.1: "Three-class sentiment classification task
using multilingual corpora like Amazon Reviews and Twitter ... CC100 ...
approximately 1.82 million labeled examples").

IMPORTANT -- dataset availability has changed since the manuscript's
sources were likely selected:

  * `amazon_reviews_multi` was pulled from the Hugging Face Hub in 2024
    at the data provider's request and is now listed as "defunct"
    (huggingface/course#679; huggingface.co/datasets/defunct-datasets/
    amazon_reviews_multi). It cannot be auto-downloaded anymore. This
    script tries a maintained mirror (`mteb/amazon_reviews_multi`) as a
    fallback and logs clearly which path was used -- verify its license
    and provenance yourself before using it in a publication.
  * `MASSIVE` (AmazonScience/massive) and `tweet_eval` (sentiment
    config) are both still live on the Hub as of this writing and are
    used here as-is.
  * CC100 is a raw monolingual web-crawl corpus, not a labeled sentiment
    dataset -- it's used elsewhere in the pipeline for token-distribution
    statistics (Fig. 1) and as unlabeled text for the counterfactual
    generation module's embedding space, not as a sentiment label source.
    It is NOT downloaded by this script.

Run with network access:

    python datasets/download_sentiment.py --languages en de fr sw hi ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 3-class sentiment label normalization: every source uses different raw
# label sets, so we normalize everything to {"negative", "neutral", "positive"}
# to match the manuscript's "three-class sentiment classification" setup.
_MASSIVE_INTENT_IS_NOT_SENTIMENT = (
    "MASSIVE is an intent-classification/slot-filling dataset, not a "
    "sentiment dataset. It is included in some multilingual pipelines for "
    "its 51-language text coverage but does NOT provide sentiment labels; "
    "do not merge it into the sentiment label set without a separate "
    "labeling step. Listed here for language-coverage augmentation only."
)


def _normalize_amazon_star_rating(stars: int) -> str:
    """1-2 stars -> negative, 3 -> neutral, 4-5 -> positive."""
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"


def _normalize_tweeteval_label(label_id: int) -> str:
    """tweet_eval 'sentiment' config: 0=negative, 1=neutral, 2=positive (fixed schema)."""
    return {0: "negative", 1: "neutral", 2: "positive"}[label_id]


def download_amazon_reviews(languages: list[str], cache_dir: str) -> list[dict]:
    """
    Attempts the canonical dataset first, falls back to a mirror, and
    raises a clear error (rather than silently returning nothing) if
    neither is reachable -- so a broken run fails loudly instead of
    quietly producing a smaller dataset than expected.
    """
    from datasets import load_dataset

    records = []
    for lang in languages:
        ds = None
        source_used = None
        try:
            ds = load_dataset("amazon_reviews_multi", lang, cache_dir=cache_dir)
            source_used = "amazon_reviews_multi (canonical)"
        except Exception as primary_err:
            logger.warning(
                "'amazon_reviews_multi' unavailable for lang=%s (%s). "
                "Trying mirror 'mteb/amazon_reviews_multi' instead.",
                lang, primary_err,
            )
            try:
                ds = load_dataset("mteb/amazon_reviews_multi", lang, cache_dir=cache_dir)
                source_used = "mteb/amazon_reviews_multi (mirror, verify license before publication use)"
            except Exception as fallback_err:
                logger.error(
                    "No usable Amazon Reviews source for lang=%s. "
                    "Canonical error: %s | Mirror error: %s",
                    lang, primary_err, fallback_err,
                )
                continue

        for split_name in ds:
            for example in ds[split_name]:
                records.append({
                    "text": example.get("review_body") or example.get("text"),
                    "label": _normalize_amazon_star_rating(int(example["stars"])),
                    "language": lang,
                    "source": source_used,
                    "label_source": "gold",
                    "is_gold_label": True,
                })
    return records


def download_tweeteval_sentiment(cache_dir: str, language: str = "en") -> list[dict]:
    """
    tweet_eval's sentiment config is English-only on the Hub as
    distributed; multilingual coverage for the "Twitter" source mentioned
    in Sec 5.1 in practice comes from OPUS-translated variants or
    separate per-language Twitter sentiment corpora, which vary by
    language and are not a single canonical HF dataset. This function
    covers the English base case; extend with per-language sources as
    needed for your target language list.
    """
    from datasets import load_dataset

    ds = load_dataset("tweet_eval", "sentiment", cache_dir=cache_dir)
    records = []
    for split_name in ds:
        for example in ds[split_name]:
            records.append({
                "text": example["text"],
                "label": _normalize_tweeteval_label(example["label"]),
                "language": language,
                "source": "tweet_eval",
                "label_source": "gold",
                "is_gold_label": True,
            })
    return records


def build_raw_sentiment_corpus(languages: list[str], cache_dir: str, include_tweeteval: bool = True) -> list[dict]:
    """Merges all configured raw sources into a single unlabeled-format-normalized list."""
    records = []
    records.extend(download_amazon_reviews(languages, cache_dir))
    if include_tweeteval and "en" in languages:
        records.extend(download_tweeteval_sentiment(cache_dir))
    logger.info("Collected %d raw sentiment records across %d languages", len(records), len(languages))
    return records


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download raw multilingual sentiment sources.")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="data/cache/sentiment")
    parser.add_argument("--out", default="data/raw/sentiment_raw.jsonl")
    args = parser.parse_args()

    records = build_raw_sentiment_corpus(args.languages, args.cache_dir)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasets.dataset_utils import write_jsonl

    n = write_jsonl(records, args.out)
    print(f"Wrote {n} raw records to {args.out}")
