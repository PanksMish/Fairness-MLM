"""
Builds paired NER counterfactual data, the NER analog of
datasets/build_counterfactual_pairs.py: for each WikiAnn record where
token-level substitution succeeds, writes a paired record with both
`tokens`/`tags` and `cf_tokens` (tags are identical for both, since
substitution preserves token count and position -- see
fairness/ner_counterfactual_generation.py's docstring for why this is
correct for the gender-pronoun dictionary specifically).

    python datasets/build_ner_counterfactual_pairs.py \\
        --input data/processed/ner/en/train.jsonl \\
        --output data/processed/ner/en/train_pairs.jsonl \\
        --language en
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.dataset_utils import read_jsonl, write_jsonl
from fairness.ner_counterfactual_generation import generate_token_counterfactual
from fairness.demographic_dictionaries import ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS

logger = logging.getLogger(__name__)


def generate_ner_pairs(
    records: list[dict],
    all_attribute_dicts: dict,
    attribute_seed_words: dict,
    all_attributes: list[str],
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Core, framework-agnostic orchestration (mirrors
    datasets/build_counterfactual_pairs.generate_pairs's structure for
    the sentiment task)."""
    rng = random.Random(seed)
    pairs = []
    n_accepted = n_skipped = 0

    for rec in records:
        candidate = generate_token_counterfactual(
            rec["tokens"], rec["tags"], all_attribute_dicts, attribute_seed_words,
            all_attributes, rng, attribute_from=rec.get("attribute"),
        )
        if candidate is None:
            n_skipped += 1
            continue

        n_accepted += 1
        pair = dict(rec)
        pair["cf_tokens"] = candidate.cf_tokens
        pair["attribute"] = candidate.attribute_from
        pair["cf_attribute"] = candidate.attribute_to
        pairs.append(pair)

    stats = {
        "n_input": len(records), "n_accepted": n_accepted, "n_skipped": n_skipped,
        "yield_rate": n_accepted / len(records) if records else 0.0,
    }
    return pairs, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build paired NER counterfactual data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-attributes", nargs="+", default=["m", "f"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    pairs, stats = generate_ner_pairs(
        records, ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, args.all_attributes, seed=args.seed,
    )
    n_written = write_jsonl(pairs, args.output)
    logger.info("Wrote %d pairs to %s. Stats: %s", n_written, args.output, stats)
    print(stats)
