"""
Top-level entrypoint: builds both the sentiment and NER datasets from a
YAML config listing the target languages.

    python datasets/download_data.py --config configs/default_config.yaml

Requires network access and `pip install -r requirements.txt`
(specifically: datasets, fasttext-langdetect or py3langid). None of this
has been executed in the sandbox this repo was built in -- run it
yourself and check the printed summary counts against Table 2 / Sec 5.1's
expected orders of magnitude (~1.82M sentiment examples, ~1.2M/150K/150K
NER train/val/test) as a sanity check, not an exact-match target.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def load_language_list(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download and build all datasets.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-ner", action="store_true")
    args = parser.parse_args()

    cfg = load_language_list(args.config)
    languages = cfg["languages"]["all"]
    ner_languages = cfg["languages"].get("ner_subset", languages)

    if not args.skip_sentiment:
        from datasets.build_sentiment import build as build_sentiment
        logger.info("Building sentiment dataset for %d languages...", len(languages))
        sentiment_counts = build_sentiment(
            languages=languages,
            out_dir=cfg["paths"]["sentiment_processed"],
            cache_dir=cfg["paths"]["cache_dir"] + "/sentiment",
        )
        logger.info("Sentiment build complete: %s", sentiment_counts)

    if not args.skip_ner:
        from datasets.download_wikiann import download_all
        logger.info("Building NER dataset for %d languages...", len(ner_languages))
        ner_counts = download_all(
            ner_languages,
            out_dir=cfg["paths"]["ner_processed"],
            cache_dir=cfg["paths"]["cache_dir"] + "/wikiann",
        )
        logger.info("NER build complete: %s", ner_counts)

    print("Done. Check data/processed/{sentiment,ner}/ for output JSONL files.")
