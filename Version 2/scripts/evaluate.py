"""
Evaluation entrypoint.

    python scripts/evaluate.py \\
        --model-config configs/mt5.yaml \\
        --task-config configs/sentiment.yaml \\
        --eval-config configs/evaluation.yaml \\
        --checkpoint checkpoints/sentiment/final_model.pt \\
        --split test

Loads a trained checkpoint, runs evaluation/evaluator.py's
evaluate_sentiment over the requested split, and writes the resulting
EvaluationReport as JSON (matching Table 5's column layout via
EvaluationReport.as_table_row()).

Requires torch + a real checkpoint. Not executable in this sandbox --
syntax-checked only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config_utils import load_and_merge
from scripts.train import build_model  # reuse the same model factory logic

logger = logging.getLogger(__name__)


def build_eval_dataloader(config: dict, split: str, use_pairs: bool):
    """
    If `use_pairs` is True and a paired file exists for this split (only
    train_pairs_path is guaranteed to exist by default -- a
    validation/test paired file would need its own
    build_counterfactual_pairs.py run), evaluation includes BTS/CCR/DPG.
    Otherwise falls back to the plain (unpaired) split for Macro-F1 only.
    """
    from datasets.dataloaders import (
        SentimentDataset, sentiment_collate_fn,
        PairedSentimentDataset, paired_sentiment_collate_fn,
        build_dataloader,
    )
    from datasets.tokenizer import TextTokenizer

    tokenizer = TextTokenizer(config["model"]["name_or_path"], max_length=config["model"].get("max_seq_length", 128))

    pairs_key = f"{split}_pairs_path"
    if use_pairs and config["data"].get(pairs_key) and Path(config["data"][pairs_key]).exists():
        dataset = PairedSentimentDataset(config["data"][pairs_key], tokenizer)
        collate_fn = paired_sentiment_collate_fn
        logger.info("Evaluating with paired counterfactual data (BTS/CCR/DPG will be computed).")
    else:
        split_key = f"{split}_path"
        dataset = SentimentDataset(config["data"][split_key], tokenizer)
        collate_fn = sentiment_collate_fn
        logger.info("Evaluating without counterfactual pairs (Macro-F1/Leakage only, no BTS/CCR/DPG).")

    return build_dataloader(dataset, collate_fn, batch_size=config["evaluation"]["batch_size"], shuffle=False)


def build_eval_dataloader_ner(config: dict, split: str, use_pairs: bool):
    from datasets.dataloaders import (
        WikiAnnDataset, ner_collate_fn,
        PairedWikiAnnDataset, paired_ner_collate_fn,
        build_dataloader,
    )
    from datasets.tokenizer import TextTokenizer

    tokenizer = TextTokenizer(config["model"]["name_or_path"], max_length=config["model"].get("max_seq_length", 128))

    pairs_key = f"{split}_pairs_path"
    if use_pairs and config["data"].get(pairs_key) and Path(config["data"][pairs_key]).exists():
        dataset = PairedWikiAnnDataset(config["data"][pairs_key], tokenizer)
        collate_fn = paired_ner_collate_fn
        logger.info("Evaluating NER with paired counterfactual data (BTS/CCR will be computed, globally only).")
    else:
        split_key = f"{split}_path"
        dataset = WikiAnnDataset(config["data"][split_key], tokenizer)
        collate_fn = ner_collate_fn
        logger.info("Evaluating NER without counterfactual pairs (Span-F1 only, no BTS/CCR).")

    return build_dataloader(dataset, collate_fn, batch_size=config["evaluation"]["batch_size"], shuffle=False)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate a trained ADAPT-BTS checkpoint.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--eval-config", default="configs/evaluation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pairs", action="store_true", help="Skip BTS/CCR/DPG even if paired data exists")
    args = parser.parse_args()

    config = load_and_merge(args.model_config, args.task_config, args.eval_config)

    if config["task"] not in ("sentiment", "ner"):
        raise ValueError(f"Unknown task '{config['task']}', expected 'sentiment' or 'ner'")

    import torch

    model = build_model(config)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    if config["task"] == "sentiment":
        from evaluation.evaluator import evaluate_sentiment
        dataloader = build_eval_dataloader(config, args.split, use_pairs=not args.no_pairs)
        report = evaluate_sentiment(
            model, dataloader, device=args.device,
            compute_leakage_probe=config["evaluation"]["compute_leakage_probe"],
        )
    else:
        from evaluation.evaluator import evaluate_ner
        from model.heads import WIKIANN_TAGS
        id_to_tag = {i: tag for i, tag in enumerate(WIKIANN_TAGS)}
        dataloader = build_eval_dataloader_ner(config, args.split, use_pairs=not args.no_pairs)
        report = evaluate_ner(model, dataloader, id_to_tag, device=args.device)

    output_path = Path(config["evaluation"]["output_report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "split": args.split,
            "global": report.as_table_row(),
            "per_language": {
                lang: {
                    "n": r.n, "task_metric": r.task_metric, "bts": r.bts,
                    "ccr": r.ccr, "dpg": r.dpg, "leakage": r.leakage,
                }
                for lang, r in report.per_language.items()
            },
        }, f, indent=2)

    logger.info("Wrote evaluation report to %s", output_path)
    print(json.dumps(report.as_table_row(), indent=2))


if __name__ == "__main__":
    main()
