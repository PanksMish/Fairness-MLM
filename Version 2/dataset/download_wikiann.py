"""
Downloads WikiAnn (via the XTREME benchmark's NER subset) for the
languages configured in configs/default_config.yaml, per Sec. 5.1:

    "the NER task has been evaluated using WikiAnn dataset from the
    XTREME benchmark ... providing approximately 1.2 million training
    examples, 150,000 validation examples, and 150,000 test examples"

Run this on a machine with network access:

    python datasets/download_wikiann.py --config configs/default_config.yaml

NOT executed by me in this session -- no network access in this sandbox.
The logic below uses the real `datasets` library API as documented; you
should sanity-check package versions (`datasets>=2.18` recommended,
since older versions load wikiann via a now-flagged trust_remote_code
loading script) before running at scale.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# WikiAnn label schema (IOB2), fixed by the dataset itself.
WIKIANN_TAGS = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
]


def download_wikiann_language(lang_code: str, cache_dir: str) -> "datasets.DatasetDict":
    """
    Downloads WikiAnn for a single language code (ISO 639-1/2, matching
    the codes used by the `wikiann` config names on the Hub, e.g. "en",
    "sw", "hi").

    Some languages in the manuscript's 101-language set (Sec 5.1) are not
    covered by WikiAnn/XTREME's NER subset -- Sec 5.2 explicitly notes:
    "NER evaluation is performed only on the subset of languages present
    in the XTREME/WikiAnn dataset. Therefore, results of NER cannot be
    compared in the same way with those of the 101-language sentiment
    analysis." Callers should skip/log missing languages rather than
    treat them as errors.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("wikiann", lang_code, cache_dir=cache_dir)
    except Exception as e:
        logger.warning("WikiAnn has no split for language '%s': %s", lang_code, e)
        return None
    return ds


def convert_split_to_records(split, lang_code: str) -> list[dict]:
    """
    Converts a HF `datasets.Dataset` split (WikiAnn schema: `tokens`,
    `ner_tags` as integer ids, `langs`) into the JSONL record schema used
    throughout this repo:

        {"tokens": [...], "tags": [...], "language": "en"}

    with `tags` as string IOB2 labels rather than integer ids, so
    downstream code (evaluation/metrics.py span-F1 via seqeval) doesn't
    need to carry the WikiAnn-specific id<->label mapping around.
    """
    tag_names = split.features["ner_tags"].feature.names  # dataset-provided id->label map
    records = []
    for example in split:
        records.append({
            "tokens": example["tokens"],
            "tags": [tag_names[t] for t in example["ner_tags"]],
            "language": lang_code,
        })
    return records


def download_all(languages: list[str], out_dir: str, cache_dir: str) -> dict[str, dict[str, int]]:
    """
    Downloads + converts WikiAnn for every language in `languages`,
    writing per-language JSONL files to
        {out_dir}/{lang}/train.jsonl, validation.jsonl, test.jsonl

    Returns a summary dict of split sizes per language, and separately
    logs (does not silently drop) any language WikiAnn doesn't cover.
    """
    from datasets.dataset_utils import write_splits_jsonl  # local package, not the HF `datasets` lib

    summary = {}
    missing = []
    out_root = Path(out_dir)

    for lang in languages:
        ds = download_wikiann_language(lang, cache_dir=cache_dir)
        if ds is None:
            missing.append(lang)
            continue

        train = convert_split_to_records(ds["train"], lang)
        val = convert_split_to_records(ds["validation"], lang)
        test = convert_split_to_records(ds["test"], lang)

        counts = write_splits_jsonl(train, val, test, out_root / lang)
        summary[lang] = counts
        logger.info("WikiAnn[%s]: %s", lang, counts)

    if missing:
        logger.warning(
            "%d/%d languages had no WikiAnn coverage and were skipped: %s",
            len(missing), len(languages), missing,
        )
        (out_root / "_missing_languages.json").write_text(json.dumps(missing, indent=2))

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download and preprocess WikiAnn for NER.")
    parser.add_argument("--languages", nargs="+", required=True,
                         help="ISO language codes, e.g. en de fr sw hi ...")
    parser.add_argument("--out-dir", default="data/processed/ner")
    parser.add_argument("--cache-dir", default="data/cache/wikiann")
    args = parser.parse_args()

    summary = download_all(args.languages, args.out_dir, args.cache_dir)
    print(json.dumps(summary, indent=2))
