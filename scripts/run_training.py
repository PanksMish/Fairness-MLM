"""
run_training.py
===============
Entry point for training the ADAPT-BTS framework.

Usage:
    python scripts/run_training.py \
        --config configs/default_config.yaml \
        --task sentiment \
        --backbone google/mt5-base \
        --languages en de fr es hi tr \
        --output_dir outputs/run1 \
        --seed 42

This script:
  1. Loads and validates the configuration.
  2. Builds the multilingual model and tokenizer.
  3. Loads and preprocesses multilingual training/validation data.
  4. Runs Phase 1: Bias-Aware Counterfactual Augmentation.
  5. Runs Phase 2: Adaptive Fairness-Constrained Optimization.
  6. Saves the best checkpoint and training logs.
"""

import argparse
import logging
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import torch

from data.dataset_loader import (
    MultilingualDatasetLoader, get_dataloader,
    ALL_LANGUAGES, LANGUAGE_RESOURCE_TIERS,
)
from data.demographic_extractor import DemographicExtractor
from data.counterfactual_generator import CounterfactualGenerator
from data.data_refresh import IterativeBiasAwareDataRefresh, AugmentedDataStore
from models.multilingual_model import build_model
from training.trainer import AdaptBTSTrainer
from utils.logging_utils import setup_logging, ExperimentLogger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ADAPT-BTS framework")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--task", type=str, choices=["sentiment", "ner"], default="sentiment")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Override backbone model (e.g., google/mt5-base)")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Subset of languages to train on (default: all 101)")
    parser.add_argument("--output_dir", type=str, default="outputs/run1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable counterfactual augmentation (ablation)")
    parser.add_argument("--no_fapc", action="store_true",
                        help="Disable fairness controller (ablation)")
    parser.add_argument("--no_ibadr", action="store_true",
                        help="Disable iterative data refresh (ablation)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / cpu (auto-detected by default)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_arg_overrides(config: dict, args) -> dict:
    """Override config values with command-line arguments."""
    if args.task:
        config["training"]["task"] = args.task
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.no_augmentation:
        config["augmentation"]["enabled"] = False
        logger.info("[Ablation] Counterfactual augmentation DISABLED.")
    if args.no_fapc:
        config["fairness"]["enabled"] = False
        logger.info("[Ablation] Fairness controller DISABLED.")
    if args.no_ibadr:
        config["ibadr"]["enabled"] = False
        logger.info("[Ablation] IBADR refresh DISABLED.")
    return config


def run_training(args):
    """Main training routine."""
    # === Setup ===
    setup_logging(args.output_dir)
    exp_logger = ExperimentLogger(args.output_dir, experiment_name="adapt_bts")

    config = load_config(args.config)
    config = apply_arg_overrides(config, args)
    exp_logger.log_config(config)

    set_seed(args.seed)
    config["training"]["seed"] = args.seed

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Language selection
    languages = args.languages or ALL_LANGUAGES
    logger.info(f"Training on {len(languages)} languages")

    # === Build Model ===
    backbone = config["model"]["backbone"]
    num_labels = config["model"]["num_labels"]
    task = config["training"]["task"]

    logger.info(f"Loading backbone: {backbone}")
    model, tokenizer = build_model(
        backbone=backbone,
        num_labels=num_labels,
        task=task,
        dropout=config["model"].get("dropout", 0.1),
        device=device,
    )

    # === Load Data ===
    logger.info("Loading multilingual datasets...")
    data_loader = MultilingualDatasetLoader(
        tokenizer=tokenizer,
        languages=languages,
        task=task,
        max_length=config["model"]["max_seq_length"],
    )
    train_ds, val_ds, test_ds = data_loader.load()
    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # === Phase 1: Counterfactual Augmentation ===
    augmentation_enabled = config.get("augmentation", {}).get("enabled", True)
    aug_cfg = config.get("augmentation", {})

    if augmentation_enabled:
        logger.info("Phase 1: Generating counterfactual augmentations...")
        cf_gen = CounterfactualGenerator(
            tokenizer=tokenizer,
            embedding_model_name=aug_cfg.get("embedding_model", "sentence-transformers/LaBSE"),
            similarity_threshold=aug_cfg.get("similarity_threshold", 0.85),
            grammar_penalty_weight=aug_cfg.get("grammar_penalty_weight", 0.1),
            similarity_weight=aug_cfg.get("similarity_weight", 1.0),
            validation_threshold=aug_cfg.get("validation_threshold", 0.75),
            device=device,
        )

        # Augment training dataset
        orig_texts = [train_ds.texts[i] for i in range(len(train_ds))]
        orig_labels = [int(train_ds.labels[i]) for i in range(len(train_ds))]
        orig_langs = train_ds.languages
        orig_attrs = train_ds.demographic_attrs

        aug_texts, aug_labels, aug_langs, aug_attrs = cf_gen.augment_dataset(
            texts=orig_texts,
            labels=orig_labels,
            languages=orig_langs,
            demographic_attrs=orig_attrs,
        )

        logger.info(
            f"Augmentation complete: {len(orig_texts)} → {len(aug_texts)} samples "
            f"({len(aug_texts) - len(orig_texts)} counterfactuals added)"
        )

        # Rebuild training dataset with augmented data
        from data.dataset_loader import SentimentDataset
        train_ds = SentimentDataset(
            texts=aug_texts,
            labels=aug_labels,
            languages=aug_langs,
            demographic_attrs=aug_attrs,
            tokenizer=tokenizer,
            max_length=config["model"]["max_seq_length"],
        )
    else:
        logger.info("Phase 1: Augmentation disabled.")
        cf_gen = None

    # === Build DataLoaders ===
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("dataloader_num_workers", 4)

    train_loader = get_dataloader(train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
    val_loader = get_dataloader(val_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    # === Phase 2: ADAPT-BTS Training ===
    logger.info("Phase 2: Starting ADAPT-BTS training...")

    # Disable fairness if ablation requested
    if not config.get("fairness", {}).get("enabled", True):
        config["fairness"]["lambda_init"] = 0.0
        config["fairness"]["eta_lambda"] = 0.0

    trainer = AdaptBTSTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        device=device,
    )

    # === IBADR: Set up iterative refresh ===
    ibadr_enabled = config.get("ibadr", {}).get("enabled", True)
    if ibadr_enabled and cf_gen is not None:
        ibadr_cfg = config.get("ibadr", {})
        ibadr = IterativeBiasAwareDataRefresh(
            counterfactual_generator=cf_gen,
            refresh_interval_epochs=ibadr_cfg.get("refresh_interval_epochs", 2),
            top_divergence_fraction=ibadr_cfg.get("top_divergence_fraction", 0.15),
            max_refresh_cycles=ibadr_cfg.get("max_refresh_cycles", 5),
        )
        trainer._ibadr = ibadr
        logger.info("[IBADR] Iterative data refresh enabled.")

    # Run training
    results = trainer.train()

    # Log all epochs
    for record in trainer.training_log:
        exp_logger.log_epoch(
            epoch=record["epoch"],
            train_metrics={k: v for k, v in record.items() if k.startswith("train_")},
            val_metrics={k: v for k, v in record.items() if k.startswith("val_")},
        )

    # Save logs
    exp_logger.save()
    exp_logger.print_summary()

    logger.info(f"\nTraining complete. Best F1: {results['best_val_f1']:.4f}")
    logger.info(f"Best BTS: {results['best_bts']:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
