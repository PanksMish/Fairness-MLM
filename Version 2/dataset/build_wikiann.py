"""
Thin wrapper matching build_sentiment.py's CLI shape, for symmetry and
so `scripts/run_all.sh` can call both builders identically. The actual
download+convert logic lives in download_wikiann.py since NER doesn't
need the clean/dedup/language-filter steps sentiment does (WikiAnn is
already curated, tokenized into words, and language-labeled by
construction).

    python datasets/build_wikiann.py --languages en de fr sw hi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.download_wikiann import download_all

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the WikiAnn NER dataset.")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/processed/ner")
    parser.add_argument("--cache-dir", default="data/cache/wikiann")
    args = parser.parse_args()

    summary = download_all(args.languages, args.out_dir, args.cache_dir)
    print(json.dumps(summary, indent=2))
