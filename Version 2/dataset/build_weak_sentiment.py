"""
============================================================================
Builds WEAK, HEURISTIC sentiment labels for languages that have no real
labeled sentiment data (i.e. everything outside the ~6 languages
Amazon Reviews Multilingual actually covers -- see README.md).

READ datasets/weak_labeling.py's module docstring before using this.
Output is written to a SEPARATE directory
(data/processed/sentiment_weak/{lang}/, never data/processed/sentiment/)
specifically so it can never be silently mixed up with real labeled
data by scripts/build_sentiment.py or scripts/train.py's default paths.
============================================================================

    python datasets/build_weak_sentiment.py \\
        --languages sw am --lexicon-file /path/to/extracted/NRC-lexicon.csv \\
        --lexicon-language-names Swahili Amharic \\
        --max-docs-per-language 5000

Requires: network access (for CC100 via datasets/download_cc100.py), and
the NRC Emotion Lexicon already downloaded + extracted yourself via
datasets/nrc_lexicon.py's download_lexicon_zip/extract_lexicon (kept as
a separate manual step rather than auto-chained here, so you
consciously go through the license terms at
https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm once, not on
every script run).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.download_cc100 import download_cc100_language
from datasets.dataset_utils import clean_pipeline, write_jsonl
from datasets.nrc_lexicon import parse_multilingual_lexicon_csv
from datasets.weak_labeling import build_weak_labeled_records

logger = logging.getLogger(__name__)

WEAK_DATA_ROOT = "data/processed/sentiment_weak"


def build_one_language(
    lang_code: str,
    lexicon_language_name: str,
    lexicon_file: str,
    max_docs: int,
    cc100_cache_dir: str,
    min_coverage: int,
    min_confidence: float,
) -> dict:
    logger.info("Downloading CC100 text for '%s' (max %d docs)...", lang_code, max_docs)
    raw_texts = download_cc100_language(lang_code, cc100_cache_dir, max_docs=max_docs)

    cleaned_records = clean_pipeline(
        [{"text": t} for t in raw_texts], text_field="text", min_chars=5,
    )
    cleaned_texts = [r["text"] for r in cleaned_records]
    logger.info("After cleaning: %d/%d texts remain", len(cleaned_texts), len(raw_texts))

    lexicon = parse_multilingual_lexicon_csv(lexicon_file, lexicon_language_name)
    logger.info("Loaded lexicon for '%s': %d positive + %d negative words",
                lexicon_language_name, len(lexicon.positive_words), len(lexicon.negative_words))

    records, stats = build_weak_labeled_records(
        cleaned_texts, language=lang_code, lexicon=lexicon,
        min_coverage=min_coverage, min_confidence=min_confidence,
    )

    out_dir = Path(WEAK_DATA_ROOT) / lang_code
    n_written = write_jsonl(records, out_dir / "weak_train.jsonl")

    manifest_path = out_dir / "WEAK_LABEL_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "warning": (
                "These are WEAK, HEURISTIC labels from lexicon-based polarity "
                "scoring, NOT human-annotated gold labels. Do not report metrics "
                "computed on this data as comparable to the manuscript's Table 5. "
                "See datasets/weak_labeling.py's module docstring for full detail."
            ),
            "stats": stats,
        }, f, indent=2)

    logger.info("Wrote %d weak-labeled records for '%s' to %s (manifest: %s)",
                n_written, lang_code, out_dir, manifest_path)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Build WEAK lexicon-based sentiment labels for CC100 text. "
                    "See module docstring -- these are NOT gold labels."
    )
    parser.add_argument("--languages", nargs="+", required=True, help="CC100 language codes")
    parser.add_argument("--lexicon-language-names", nargs="+", required=True,
                         help="NRC lexicon language names, same order/count as --languages "
                              "(e.g. --languages sw am --lexicon-language-names Swahili Amharic)")
    parser.add_argument("--lexicon-file", required=True,
                         help="Path to the extracted NRC multilingual lexicon CSV/TSV")
    parser.add_argument("--max-docs-per-language", type=int, default=5000)
    parser.add_argument("--cc100-cache-dir", default="data/cache/cc100")
    parser.add_argument("--min-coverage", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    if len(args.languages) != len(args.lexicon_language_names):
        raise ValueError("--languages and --lexicon-language-names must have the same length")

    print("=" * 70)
    print("WARNING: This produces WEAK, non-gold sentiment labels.")
    print("See datasets/weak_labeling.py for what this data is and is not suitable for.")
    print("=" * 70)

    all_stats = {}
    for lang_code, lex_name in zip(args.languages, args.lexicon_language_names):
        all_stats[lang_code] = build_one_language(
            lang_code, lex_name, args.lexicon_file, args.max_docs_per_language,
            args.cc100_cache_dir, args.min_coverage, args.min_confidence,
        )

    print(json.dumps(all_stats, indent=2))
