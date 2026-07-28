"""
Builds the paired counterfactual training data that
optimization/trainer.py consumes: for each input JSONL record where
counterfactual generation succeeds AND is accepted (Eq. 9), writes out a
record containing BOTH the original and counterfactual text side by
side (rather than as two separate flat entries, which is what
fairness/augmentation.build_augmented_dataset produces for Eq. 10's
D_aug). This paired format is what's needed for Algorithm 2 line 6's
per-batch BTS estimation, which requires evaluating the model on both
x^(a) and x^(b) for the same underlying instance.

    python datasets/build_counterfactual_pairs.py \\
        --input data/processed/sentiment/train.jsonl \\
        --output data/processed/sentiment/train_pairs.jsonl \\
        --language en

The core orchestration logic (`generate_pairs`) is decoupled from any
specific encoder, so it's testable with a mock `encoder_fn` (see
tests/test_build_counterfactual_pairs.py). The CLI's default encoder_fn
uses a real pretrained sentence-embedding model (LaBSE, via
sentence-transformers) for the semantic-preservation check (Eq. 3) --
this is a natural choice since Eq. 3's h(.) is meant to capture semantic
content independent of the (not-yet-trained) task model, but requires
`pip install sentence-transformers` + a model download, neither of which
is available in this sandbox.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.dataset_utils import read_jsonl, write_jsonl
from fairness.counterfactual_generation import (
    LexiconSubstitutor, TieredGenerationStrategy, LexiconAttributeDetector,
    CounterfactualDataEngine, CounterfactualEngineConfig,
)
from fairness.demographic_dictionaries import (
    ENGLISH_GENDER_DICT, ENGLISH_GENDER_SEEDS, EXAMPLE_TOKEN_COUNTS,
)

logger = logging.getLogger(__name__)


def generate_pairs(
    records: list[dict],
    engine: CounterfactualDataEngine,
    language: str,
    all_attributes: list[str],
    text_field: str = "text",
    attribute_field: str = "attribute",
) -> tuple[list[dict], dict]:
    """
    Core, encoder-agnostic orchestration logic: runs Algorithm 1 over
    every record and keeps only accepted (x, x^(b)) pairs, in the paired
    schema trainer.py's dataloaders need:

        {"text": x, "label": y, "attribute": a,
         "cf_text": x^(b), "cf_attribute": b,
         "language": lang, "score": S(x, x^(b))}

    Returns (pairs, stats) where stats is a plain dict summary (counts of
    accepted / rejected / skipped), for logging and sanity-checking
    against expected order-of-magnitude yields.
    """
    pairs = []
    n_accepted = n_rejected_score = n_rejected_morphology = n_skipped = 0

    for rec in records:
        text = rec[text_field]
        declared_attribute = rec.get(attribute_field)

        candidate = engine.generate_one(text, language, all_attributes, attribute_from=declared_attribute)
        if candidate is None:
            n_skipped += 1
            continue
        if candidate.score == float("-inf"):
            n_rejected_morphology += 1
            continue
        if not candidate.accepted:
            n_rejected_score += 1
            continue

        n_accepted += 1
        pair = dict(rec)
        pair["cf_text"] = candidate.candidate_text
        pair["cf_attribute"] = candidate.attribute_to
        pair["attribute"] = candidate.attribute_from
        pair["language"] = language
        pair["score"] = candidate.score
        pairs.append(pair)

    stats = {
        "n_input": len(records),
        "n_accepted": n_accepted,
        "n_rejected_score": n_rejected_score,
        "n_rejected_morphology": n_rejected_morphology,
        "n_skipped_no_candidate": n_skipped,
        "yield_rate": n_accepted / len(records) if records else 0.0,
    }
    return pairs, stats


def build_default_hr_engine(gamma: float = 0.5, encoder_fn: Optional[Callable] = None) -> CounterfactualDataEngine:
    """
    Convenience constructor wiring the starter English gender dictionary
    (fairness/demographic_dictionaries.py) into an HR-only engine (MR/LR
    tiers raise if dispatched to, since no real embedding/MT backend is
    configured here -- extend `strategy` yourself once you have one).
    """
    detector = LexiconAttributeDetector(ENGLISH_GENDER_SEEDS)
    hr_op = LexiconSubstitutor(ENGLISH_GENDER_DICT)

    def _mr_unavailable(text, a, b):
        raise NotImplementedError(
            "MR-tier counterfactual generation requires a real cross-lingual "
            "embedding model. Construct your own fairness.counterfactual_generation."
            "EmbeddingAlignmentTransform with a real EmbeddingSpace implementation "
            "and pass it to TieredGenerationStrategy instead of using this default."
        )

    def _lr_unavailable(text, a, b, source_lang):
        raise NotImplementedError(
            "LR-tier counterfactual generation requires a real MT system. "
            "Construct your own PivotTranslationTransform with a real Translator "
            "implementation and pass it to TieredGenerationStrategy instead of "
            "using this default."
        )

    strategy = TieredGenerationStrategy(
        hr_operator=hr_op, mr_operator=_mr_unavailable, lr_operator=_lr_unavailable,
        token_counts=EXAMPLE_TOKEN_COUNTS,
    )

    if encoder_fn is None:
        def encoder_fn(text: str):
            raise RuntimeError(
                "No encoder_fn supplied. Semantic validation (Eq. 3) requires a "
                "real embedding function. For a quick real option: "
                "`pip install sentence-transformers` then:\n"
                "    from sentence_transformers import SentenceTransformer\n"
                "    _m = SentenceTransformer('sentence-transformers/LaBSE')\n"
                "    encoder_fn = lambda text: _m.encode(text)\n"
                "and pass that as build_default_hr_engine(encoder_fn=encoder_fn)."
            )

    return CounterfactualDataEngine(
        detector=detector, strategy=strategy, encoder_fn=encoder_fn,
        config=CounterfactualEngineConfig(gamma=gamma),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build paired counterfactual training data.")
    parser.add_argument("--input", required=True, help="Input JSONL (e.g. train.jsonl)")
    parser.add_argument("--output", required=True, help="Output paired JSONL path")
    parser.add_argument("--language", required=True, help="ISO language code for all records in --input")
    parser.add_argument("--gamma", type=float, default=0.5, help="Eq. 9 acceptance threshold")
    parser.add_argument("--all-attributes", nargs="+", default=["m", "f"])
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/LaBSE")
        encoder_fn = lambda text: _model.encode(text)
        logger.info("Using LaBSE (sentence-transformers) for semantic validation encoder.")
    except ImportError:
        logger.error(
            "sentence-transformers not installed; cannot run without a real "
            "encoder_fn. Install with `pip install sentence-transformers`."
        )
        raise

    engine = build_default_hr_engine(gamma=args.gamma, encoder_fn=encoder_fn)
    records = list(read_jsonl(args.input))
    pairs, stats = generate_pairs(records, engine, args.language, args.all_attributes)

    n_written = write_jsonl(pairs, args.output)
    logger.info("Wrote %d pairs to %s. Stats: %s", n_written, args.output, stats)
    print(stats)
