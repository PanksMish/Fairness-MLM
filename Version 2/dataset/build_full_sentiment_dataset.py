"""
Builds sentiment data across the FULL configured language list (101
languages as of configs/default_config.yaml's 2026-07 update) by
combining:

  - REAL, gold-labeled data (datasets/download_sentiment.py) for the
    ~6 languages Amazon Reviews Multilingual + TweetEval actually cover
  - WEAK, lexicon-heuristic labels (datasets/build_weak_sentiment.py)
    for every other configured language

This is the combiner the earlier per-source scripts were building
toward -- it's what makes "101 languages" achievable at all, at the
cost of ~95 of those languages' labels being heuristic, not
human-annotated. Every record in the combined output carries
`label_source` ("gold" or "weak_lexicon_nrc") and `is_gold_label`
(True/False) so this composition is never invisible downstream.

    python datasets/build_full_sentiment_dataset.py \\
        --config configs/default_config.yaml \\
        --lexicon-file /path/to/extracted/NRC-lexicon.csv \\
        --lexicon-language-names en=English de=German sw=Swahili ... \\
        --out-dir data/processed/sentiment_101 \\
        --max-weak-docs-per-language 5000

Requires network access, `datasets`, and the NRC lexicon already
downloaded (see datasets/nrc_lexicon.py). Not executable in this
sandbox -- syntax-checked only, same as every other network-dependent
script in this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Amazon Reviews Multilingual's actual real coverage (confirmed via the
# dataset's own documentation, not assumed) -- see
# datasets/download_sentiment.py's module docstring.
GOLD_LABELED_LANGUAGES = {"en", "de", "es", "fr", "ja", "zh"}


def build_language_plan(config: dict) -> dict[str, str]:
    """
    Returns {language_code: "gold" | "weak"} for every language in the
    config's flattened `languages.all` list. Pure logic, no I/O -- this
    is what tests/test_build_full_sentiment_dataset.py actually
    exercises without needing network access.
    """
    plan = {}
    for lang in config["languages"]["all"]:
        plan[lang] = "gold" if lang in GOLD_LABELED_LANGUAGES else "weak"
    return plan


def coverage_report(plan: dict[str, str], config: dict) -> list[dict]:
    """
    Builds the per-language coverage table: language, resource tier,
    label source. This is exactly the shape of a CSV a person would
    want to look at to understand what "101 languages" actually means
    here before training anything.
    """
    tier_by_lang = {}
    for tier_key, tier_label in [
        ("high_resource", "HR"), ("medium_resource", "MR"), ("low_resource", "LR"),
    ]:
        for lang in config["languages"][tier_key]:
            tier_by_lang[lang] = tier_label

    rows = []
    for lang, source in sorted(plan.items()):
        rows.append({
            "language_code": lang,
            "resource_tier": tier_by_lang.get(lang, "?"),
            "label_source": source,
            "is_gold_label": source == "gold",
        })
    return rows


def run(config_path: str, lexicon_file: str, lexicon_language_names: dict[str, str],
        out_dir: str, max_weak_docs: int, min_coverage: int, min_confidence: float) -> dict:
    import yaml
    from datasets.build_sentiment import build as build_gold_sentiment
    from datasets.build_weak_sentiment import build_one_language as build_weak_language

    with open(config_path) as f:
        config = yaml.safe_load(f)

    plan = build_language_plan(config)
    gold_langs = [l for l, s in plan.items() if s == "gold"]
    weak_langs = [l for l, s in plan.items() if s == "weak"]

    logger.info("Plan: %d gold-labeled languages, %d weak-labeled languages", len(gold_langs), len(weak_langs))

    out_dir_path = Path(out_dir)
    results = {"gold": {}, "weak": {}}

    if gold_langs:
        logger.info("Building GOLD-labeled data for: %s", gold_langs)
        gold_counts = build_gold_sentiment(
            languages=gold_langs, out_dir=str(out_dir_path / "gold"),
            skip_language_verification=False,
        )
        results["gold"] = gold_counts

    for lang in weak_langs:
        if lang not in lexicon_language_names:
            logger.warning(
                "No NRC lexicon language name provided for '%s' -- skipping "
                "(pass --lexicon-language-names to cover it).", lang,
            )
            continue
        logger.info("Building WEAK-labeled data for: %s", lang)
        stats = build_weak_language(
            lang, lexicon_language_names[lang], lexicon_file, max_weak_docs,
            cc100_cache_dir="data/cache/cc100", min_coverage=min_coverage, min_confidence=min_confidence,
        )
        results["weak"][lang] = stats

    report = coverage_report(plan, config)
    report_path = out_dir_path / "coverage_report.csv"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language_code", "resource_tier", "label_source", "is_gold_label"])
        writer.writeheader()
        writer.writerows(report)
    logger.info("Wrote coverage report to %s", report_path)

    return {"plan": plan, "results": results, "coverage_report_path": str(report_path)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build sentiment data across all 101 configured languages.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--lexicon-file", required=True)
    parser.add_argument("--lexicon-language-names", nargs="+", default=[],
                         help="lang_code=LexiconName pairs, e.g. sw=Swahili am=Amharic ...")
    parser.add_argument("--out-dir", default="data/processed/sentiment_101")
    parser.add_argument("--max-weak-docs-per-language", type=int, default=5000)
    parser.add_argument("--min-coverage", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    lexicon_names = {}
    for pair in args.lexicon_language_names:
        if "=" not in pair:
            raise ValueError(f"Invalid --lexicon-language-names entry '{pair}', expected lang_code=LexiconName")
        code, name = pair.split("=", 1)
        lexicon_names[code] = name

    print("=" * 70)
    print("Building sentiment data across the full configured language list.")
    print(f"Gold-labeled (real annotations): {sorted(GOLD_LABELED_LANGUAGES)}")
    print("Everything else: WEAK lexicon-heuristic labels. See")
    print("datasets/weak_labeling.py before trusting or reporting on this data.")
    print("=" * 70)

    result = run(
        args.config, args.lexicon_file, lexicon_names, args.out_dir,
        args.max_weak_docs_per_language, args.min_coverage, args.min_confidence,
    )
    print(json.dumps({"plan": result["plan"]}, indent=2))
    print(f"Coverage report: {result['coverage_report_path']}")
