"""
Downloads raw (unlabeled) monolingual text from CC100, the corpus the
manuscript cites for token-distribution statistics (Fig. 1) and which
Sec 5.1 also lists as a sentiment-data source alongside Amazon
Reviews/Twitter -- see datasets/weak_labeling.py for why CC100 itself
needs a separate labeling step before it can serve that second role.

STATUS CHECKED LIVE (not from training data): the original
`load_dataset("cc100")` config uses a loading SCRIPT that fetches raw
files from statmt.org, and those URLs have been unstable/occasionally
503-ing for years (github.com/huggingface/datasets, issues #1679,
#3632). The maintained alternative is `statmt/cc100` on the Hub, a
Parquet-format mirror published by the statmt organization itself
(not a random third party), which avoids the broken loading-script
path entirely. This downloader targets `statmt/cc100`. If that ever
breaks too, `xu-song/cc100-samples` is a smaller (first 10k lines per
language) sample version useful for a quick smoke test while you sort
out a full-size source.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# CC100's valid language codes (from the corpus's own homepage,
# https://data.statmt.org/cc-100/, cross-checked against
# xu-song/cc100-samples' documented VALID_CODES list).
CC100_VALID_CODES = {
    "am", "ar", "as", "az", "be", "bg", "bn", "bn_rom", "br", "bs", "ca", "cs", "cy",
    "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fr", "fy", "ga",
    "gd", "gl", "gn", "gu", "ha", "he", "hi", "hi_rom", "hr", "ht", "hu", "hy", "id",
    "ig", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lg",
    "li", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "my_zaw",
    "ne", "nl", "no", "ns", "om", "or", "pa", "pl", "ps", "pt", "qu", "rm", "ro", "ru",
    "sa", "si", "sc", "sd", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta",
    "ta_rom", "te", "te_rom", "th", "tl", "tn", "tr", "ug", "uk", "ur", "ur_rom", "uz",
    "vi", "wo", "xh", "yi", "yo", "zh-Hans", "zh-Hant", "zu",
}


def download_cc100_language(lang_code: str, cache_dir: str, max_docs: int | None = None) -> list[str]:
    """
    Downloads raw text paragraphs for one language from `statmt/cc100`.

    Args:
        lang_code: a code from CC100_VALID_CODES
        max_docs: if set, only take the first N paragraphs (CC100 is
            huge -- billions of tokens for high-resource languages, so
            capping this is usually necessary for anything beyond a
            full production run)

    Returns:
        List of raw text paragraphs (unlabeled -- see
        datasets/weak_labeling.py to turn these into sentiment-labeled
        examples).
    """
    if lang_code not in CC100_VALID_CODES:
        raise ValueError(
            f"'{lang_code}' is not a recognized CC100 language code. "
            f"See https://data.statmt.org/cc-100/ for the full list."
        )

    from datasets import load_dataset

    try:
        ds = load_dataset("statmt/cc100", lang=lang_code, split="train",
                           streaming=(max_docs is not None), cache_dir=cache_dir)
    except Exception as e:
        logger.error(
            "Failed to load statmt/cc100 for lang='%s': %s. "
            "Try xu-song/cc100-samples for a smaller/more-likely-to-work "
            "alternative, or check https://huggingface.co/datasets/statmt/cc100 "
            "directly for the current dataset status.", lang_code, e,
        )
        raise

    texts = []
    for i, example in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        texts.append(example["text"])
    return texts


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download raw CC100 text for one or more languages.")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--max-docs", type=int, default=10000,
                         help="Cap per language (CC100 is huge; default keeps this a quick smoke test)")
    parser.add_argument("--cache-dir", default="data/cache/cc100")
    parser.add_argument("--out-dir", default="data/raw/cc100")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for lang in args.languages:
        texts = download_cc100_language(lang, args.cache_dir, max_docs=args.max_docs)
        out_path = out_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t, "language": lang}, ensure_ascii=False) + "\n")
        summary[lang] = len(texts)
        logger.info("Wrote %d CC100 paragraphs for '%s' to %s", len(texts), lang, out_path)

    print(summary)
