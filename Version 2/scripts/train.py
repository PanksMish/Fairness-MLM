"""
Training entrypoint.

    python scripts/train.py \\
        --model-config configs/mt5.yaml \\
        --task-config configs/sentiment.yaml \\
        --set trainer.num_epochs=5 trainer.learning_rate=3e-5

Wires together:
    configs/*.yaml          -> merged config (scripts/config_utils.py)
    model/mt5.py|xlmr.py    -> model construction
    datasets/dataloaders.py -> PairedSentimentDataset + DataLoader
    optimization/trainer.py -> ADAPTBTSTrainer.train()

Requires torch + transformers + a GPU (or CPU, slowly) + the JSONL data
produced by datasets/build_sentiment.py and
datasets/build_counterfactual_pairs.py to already exist on disk. Not
executable in the sandbox this was authored in -- syntax-checked only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config_utils import load_and_merge, apply_cli_overrides

logger = logging.getLogger(__name__)


def build_model(config: dict):
    model_name = config["model"]["name_or_path"]
    max_seq_length = config["model"].get("max_seq_length", 128)
    dropout = config["model"].get("dropout", 0.1)
    freeze_embeddings = config["model"].get("freeze_embeddings", False)
    num_labels = config["data"]["num_labels"]

    is_t5 = "t5" in model_name.lower()
    task = config["task"]

    if task == "sentiment":
        if is_t5:
            from model.mt5 import build_mt5_sentiment_model
            return build_mt5_sentiment_model(
                max_seq_length=max_seq_length, num_labels=num_labels,
                dropout=dropout, freeze_embeddings=freeze_embeddings,
            )
        else:
            from model.xlmr import build_xlmr_sentiment_model
            return build_xlmr_sentiment_model(
                max_seq_length=max_seq_length, num_labels=num_labels,
                dropout=dropout, model_name=model_name, freeze_embeddings=freeze_embeddings,
            )
    elif task == "ner":
        if is_t5:
            from model.mt5 import build_mt5_ner_model
            return build_mt5_ner_model(
                max_seq_length=max_seq_length, num_labels=num_labels,
                dropout=dropout, freeze_embeddings=freeze_embeddings,
            )
        else:
            from model.xlmr import build_xlmr_ner_model
            return build_xlmr_ner_model(
                max_seq_length=max_seq_length, num_labels=num_labels,
                dropout=dropout, model_name=model_name, freeze_embeddings=freeze_embeddings,
            )
    else:
        raise ValueError(f"Unknown task '{task}', expected 'sentiment' or 'ner'")


def build_sentiment_dataloader(config: dict):
    from datasets.dataloaders import PairedSentimentDataset, paired_sentiment_collate_fn, build_dataloader
    from datasets.tokenizer import TextTokenizer

    tokenizer = TextTokenizer(config["model"]["name_or_path"], max_length=config["model"].get("max_seq_length", 128))
    pairs_path = config["data"].get("train_pairs_path")
    if not pairs_path or not Path(pairs_path).exists():
        raise FileNotFoundError(
            f"No paired counterfactual data found at '{pairs_path}'. Run "
            "datasets/build_counterfactual_pairs.py first, or set "
            "data.train_pairs_path in your task config."
        )

    dataset = PairedSentimentDataset(pairs_path, tokenizer)
    return build_dataloader(
        dataset, paired_sentiment_collate_fn,
        batch_size=config["trainer"]["batch_size"], shuffle=True,
    )


def build_ner_dataloader(config: dict):
    from datasets.dataloaders import PairedWikiAnnDataset, paired_ner_collate_fn, build_dataloader
    from datasets.tokenizer import TextTokenizer

    tokenizer = TextTokenizer(config["model"]["name_or_path"], max_length=config["model"].get("max_seq_length", 128))
    pairs_path = config["data"].get("train_pairs_path")
    if not pairs_path or not Path(pairs_path).exists():
        raise FileNotFoundError(
            f"No paired NER counterfactual data found at '{pairs_path}'. Run "
            "datasets/build_ner_counterfactual_pairs.py first, or set "
            "data.train_pairs_path in your task config."
        )

    dataset = PairedWikiAnnDataset(pairs_path, tokenizer)
    return build_dataloader(
        dataset, paired_ner_collate_fn,
        batch_size=config["trainer"]["batch_size"], shuffle=True,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train an ADAPT-BTS model.")
    parser.add_argument("--model-config", required=True, help="e.g. configs/mt5.yaml")
    parser.add_argument("--task-config", required=True, help="e.g. configs/sentiment.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="dotted.path=value overrides")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_and_merge(args.model_config, args.task_config)
    config = apply_cli_overrides(config, args.set)
    logger.info("Merged config: %s", config)

    if config["task"] not in ("sentiment", "ner"):
        raise ValueError(f"Unknown task '{config['task']}', expected 'sentiment' or 'ner'")

    import torch
    from optimization.trainer import ADAPTBTSTrainer, TrainerConfig

    model = build_model(config)
    if config["task"] == "sentiment":
        dataloader = build_sentiment_dataloader(config)
    else:
        dataloader = build_ner_dataloader(config)

    trainer_cfg = TrainerConfig(
        num_epochs=config["trainer"]["num_epochs"],
        learning_rate=config["trainer"]["learning_rate"],
        batch_size=config["trainer"]["batch_size"],
        tau=config["trainer"]["tau"],
        eta_lambda=config["trainer"]["eta_lambda"],
        lambda_init=config["trainer"]["lambda_init"],
        lambda_min=config["trainer"]["lambda_min"],
        lambda_max=config["trainer"]["lambda_max"],
        ibadr_refresh_interval=config["trainer"]["ibadr_refresh_interval"],
        ibadr_selection_ratio=config["trainer"]["ibadr_selection_ratio"],
        use_amp=config["trainer"]["use_amp"],
        log_every=config["trainer"]["log_every"],
        task=config["task"],
    )

    trainer = ADAPTBTSTrainer(model, trainer_cfg, device=args.device)
    trained_model = trainer.train(dataloader)

    output_dir = Path(config["checkpoint"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "final_model.pt"
    torch.save(trained_model.state_dict(), checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
