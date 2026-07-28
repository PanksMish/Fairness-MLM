"""
Builds the full collated sentiment dataset: 101 languages, targeting
~2.5M total entries, allocated per-language via datasets/collation.py's
quota logic (Table 2's tier ratios scaled to the target total).

    python datasets/build_collated_dataset.py \\
        --config configs/default_config.yaml \\
        --lexicon-file /path/to/extracted/NRC-lexicon.csv \\
        --lexicon-language-names sw=Swahili am=Amharic ... \\
        --target-total 2500000 \\
        --out-dir data/processed/sentiment_collated

Writes:
    {out_dir}/collated_train.jsonl   -- ALL languages, gold + weak, one file
    {out_dir}/collation_report.csv   -- per-language quota vs actual, with shortfall reasons
    {out_dir}/summary.json           -- aggregate stats

Every record in collated_train.jsonl carries label_source/is_gold_label
(from datasets/download_sentiment.py and datasets/weak_labeling.py's
existing tagging), so the gold/weak composition survives the merge.

Requires network access, `datasets`, and the NRC lexicon already
downloaded. Not executable in this sandbox -- syntax-checked only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Oversampling factor for weak-language CC100 requests: since not every
# downloaded paragraph will pass the lexicon's min-coverage/min-confidence
# filters (build_weak_labeled_records drops uncovered/low-confidence
# text), requesting more raw documents than the quota partially
# compensates. This is a heuristic multiplier, not a guarantee -- a
# language with genuinely poor lexicon coverage can still fall short,
# and that shortfall gets reported, not hidden.
WEAK_OVERSAMPLE_FACTOR = 3


def build_gold_language(lang: str, quota: int, cache_dir: str):
    from datasets.download_sentiment import download_amazon_reviews, download_tweeteval_sentiment
    from datasets.dataset_utils import clean_pipeline
    from datasets.collation import collate_language

    records = download_amazon_reviews([lang], cache_dir)
    if lang == "en":
        records += download_tweeteval_sentiment(cache_dir)
    cleaned = clean_pipeline(records, text_field="text", min_chars=3)

    shortfall_reason = "Amazon Reviews Multilingual / TweetEval have fewer real examples than quota for this language"
    selected, result = collate_language(lang, "?", "gold", cleaned, quota, shortfall_reason)
    return selected, result


def build_weak_language(lang: str, lexicon_name: str, lexicon_file: str, quota: int,
                          cc100_cache_dir: str, min_coverage: int, min_confidence: float):
    from datasets.download_cc100 import download_cc100_language
    from datasets.dataset_utils import clean_pipeline
    from datasets.nrc_lexicon import parse_multilingual_lexicon_csv
    from datasets.weak_labeling import build_weak_labeled_records
    from datasets.collation import collate_language

    raw_docs = download_cc100_language(lang, cc100_cache_dir, max_docs=quota * WEAK_OVERSAMPLE_FACTOR)
    cleaned = clean_pipeline([{"text": t} for t in raw_docs], text_field="text", min_chars=5)
    cleaned_texts = [r["text"] for r in cleaned]

    lexicon = parse_multilingual_lexicon_csv(lexicon_file, lexicon_name)
    records, weak_stats = build_weak_labeled_records(
        cleaned_texts, language=lang, lexicon=lexicon,
        min_coverage=min_coverage, min_confidence=min_confidence,
    )

    shortfall_reason = (
        f"CC100 text requested (x{WEAK_OVERSAMPLE_FACTOR} oversample) but lexicon coverage "
        f"yielded only {weak_stats['n_output_records']}/{weak_stats['n_input_texts']} usable records"
    )
    selected, result = collate_language(lang, "?", "weak", records, quota, shortfall_reason)
    return selected, result


def run(config_path: str, lexicon_file: str, lexicon_language_names: dict[str, str],
        target_total: int, out_dir: str, cache_dir: str, min_coverage: int, min_confidence: float):
    import yaml
    from datasets.dataset_utils import write_jsonl
    from datasets.collation import (
        compute_language_quotas, write_collation_report, summarize_results,
    )
    from datasets.build_full_sentiment_dataset import GOLD_LABELED_LANGUAGES

    with open(config_path) as f:
        config = yaml.safe_load(f)

    quotas = compute_language_quotas(config, target_total=target_total)

    tier_by_lang = {}
    for tier_key, tier_label in [
        ("high_resource", "HR"), ("medium_resource", "MR"), ("low_resource", "LR"),
    ]:
        for lang in config["languages"][tier_key]:
            tier_by_lang[lang] = tier_label

    all_records = []
    all_results = []

    for lang in config["languages"]["all"]:
        quota = quotas[lang]
        tier = tier_by_lang[lang]

        if lang in GOLD_LABELED_LANGUAGES:
            logger.info("[%s] gold, quota=%d", lang, quota)
            selected, result = build_gold_language(lang, quota, cache_dir)
        else:
            if lang not in lexicon_language_names:
                logger.warning("[%s] no lexicon language name given -- skipping entirely", lang)
                continue
            logger.info("[%s] weak, quota=%d", lang, quota)
            selected, result = build_weak_language(
                lang, lexicon_language_names[lang], lexicon_file, quota,
                cache_dir, min_coverage, min_confidence,
            )

        result.resource_tier = tier
        all_records.extend(selected)
        all_results.append(result)

        if result.shortfall > 0:
            logger.warning("[%s] SHORTFALL: got %d/%d (%s)", lang, result.actual_count, quota, result.shortfall_reason)

    out_dir_path = Path(out_dir)
    n_written = write_jsonl(all_records, out_dir_path / "collated_train.jsonl")
    report_path = write_collation_report(all_results, out_dir_path / "collation_report.csv")
    summary = summarize_results(all_results, target_total)

    with open(out_dir_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote %d records to %s", n_written, out_dir_path / "collated_train.jsonl")
    logger.info("Summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the full collated 101-language sentiment dataset.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--lexicon-file", required=True)
    parser.add_argument("--lexicon-language-names", nargs="+", default=[],
                         help="lang_code=LexiconName pairs for the 95 weak-labeled languages")
    parser.add_argument("--target-total", type=int, default=2_500_000)
    parser.add_argument("--out-dir", default="data/processed/sentiment_collated")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--min-coverage", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    lexicon_names = {}
    for pair in args.lexicon_language_names:
        code, name = pair.split("=", 1)
        lexicon_names[code] = name

    print("=" * 70)
    print(f"Targeting {args.target_total:,} total entries across 101 languages.")
    print("6 languages get REAL gold labels; ~95 get WEAK lexicon-heuristic labels.")
    print("Actual output may fall short of target -- see collation_report.csv")
    print("for exactly where and why, per language. Nothing here is fabricated")
    print("to hit the target number artificially.")
    print("=" * 70)

    summary = run(
        args.config, args.lexicon_file, lexicon_names, args.target_total,
        args.out_dir, args.cache_dir, args.min_coverage, args.min_confidence,
    )
    print(json.dumps(summary, indent=2))
