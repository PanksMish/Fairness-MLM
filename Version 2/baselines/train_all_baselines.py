"""
Trains all six baselines (Table 3) plus mT5-FT, one after another, for a
given task/config, so Table 5/6-style comparisons can be generated from
a single command.

    python baselines/train_all_baselines.py \\
        --model-config configs/mt5.yaml --task-config configs/sentiment.yaml \\
        --baselines mt5_ft mfc csd madl grad_unlearn magnet

Each baseline has a different loss signature (see mfc.py, csd.py,
madl.py, grad_unlearn.py, magnet.py, mt5_ft.py above), so this script
dispatches to baseline-specific training loops rather than trying to
force them through one generic interface -- that would either lose
fidelity to each method's actual mechanism or require a much heavier
abstraction than six training loops warrant.

Requires torch/transformers/GPU + real data on disk. Not executable in
this sandbox -- syntax-checked only, same as scripts/train.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config_utils import load_and_merge
from scripts.train import build_model, build_sentiment_dataloader

logger = logging.getLogger(__name__)

AVAILABLE_BASELINES = ["mt5_ft", "mfc", "csd", "madl", "grad_unlearn", "magnet"]


def train_mt5_ft(model, dataloader, config, device):
    from optimization.trainer import ADAPTBTSTrainer
    from baselines.mt5_ft import mt5_ft_trainer_config

    trainer_cfg = mt5_ft_trainer_config(
        num_epochs=config["trainer"]["num_epochs"],
        learning_rate=config["trainer"]["learning_rate"],
        batch_size=config["trainer"]["batch_size"],
        use_amp=config["trainer"]["use_amp"],
        log_every=config["trainer"]["log_every"],
        task=config["task"],
    )
    trainer = ADAPTBTSTrainer(model, trainer_cfg, device=device)
    return trainer.train(dataloader)


def train_generic_baseline(model, dataloader, config, device, loss_fn, extra_batch_fields: list[str]):
    """
    Shared loop for baselines whose loss function has the signature
    `loss_fn(model, input_ids, attention_mask, labels, ...) -> dict with
    "total_loss"`. `extra_batch_fields` lists which additional batch keys
    (beyond input_ids/attention_mask/labels) to forward positionally --
    varies per baseline (e.g. MFC needs language/attribute ids, CSD needs
    the counterfactual pair, Grad-Unl needs the counterfactual pair).
    """
    import torch
    from optimization.optimizer import build_optimizer, build_scheduler

    num_steps = len(dataloader) * config["trainer"]["num_epochs"]
    optimizer = build_optimizer(model, learning_rate=config["trainer"]["learning_rate"])
    scheduler = build_scheduler(optimizer, num_steps)

    model = model.to(device)
    model.train()
    step = 0
    for epoch in range(config["trainer"]["num_epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            args = [
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["labels"].to(device),
            ]
            for field in extra_batch_fields:
                val = batch[field]
                args.append(val.to(device) if hasattr(val, "to") else val)

            result = loss_fn(model, *args)
            result["total_loss"].backward()
            optimizer.step()
            scheduler.step()

            if step % config["trainer"]["log_every"] == 0:
                logger.info("[epoch %d step %d] total_loss=%.4f", epoch, step, float(result["total_loss"]))
            step += 1
    return model


def build_sentiment_dataloader_with_vocabs(config: dict):
    """
    Like scripts.train.build_sentiment_dataloader, but also builds
    language/attribute LabelVocabs from the training data (datasets/vocab.py)
    and passes them into PairedSentimentDataset so the collate function
    emits integer language_ids/attribute_ids/cf_attribute_ids -- required
    by MFC (baselines/mfc.py) and MADL (baselines/madl.py), neither of
    which can consume raw string labels.
    """
    from datasets.dataloaders import PairedSentimentDataset, paired_sentiment_collate_fn, build_dataloader
    from datasets.dataset_utils import read_jsonl
    from datasets.tokenizer import TextTokenizer
    from datasets.vocab import build_vocabs_from_records

    tokenizer = TextTokenizer(config["model"]["name_or_path"], max_length=config["model"].get("max_seq_length", 128))
    pairs_path = config["data"].get("train_pairs_path")
    if not pairs_path or not Path(pairs_path).exists():
        raise FileNotFoundError(
            f"No paired counterfactual data found at '{pairs_path}'. Run "
            "datasets/build_counterfactual_pairs.py first."
        )

    records = list(read_jsonl(pairs_path))
    vocabs = build_vocabs_from_records(records)

    dataset = PairedSentimentDataset(
        pairs_path, tokenizer,
        language_vocab=vocabs["language"], attribute_vocab=vocabs["attribute"],
    )
    dataloader = build_dataloader(
        dataset, paired_sentiment_collate_fn,
        batch_size=config["trainer"]["batch_size"], shuffle=True,
    )
    return dataloader, vocabs


def train_mfc(model, config, device):
    from baselines.mfc import mfc_loss, MFCConfig
    from optimization.optimizer import build_optimizer, build_scheduler

    dataloader, vocabs = build_sentiment_dataloader_with_vocabs(config)
    num_steps = len(dataloader) * config["trainer"]["num_epochs"]
    optimizer = build_optimizer(model, learning_rate=config["trainer"]["learning_rate"])
    scheduler = build_scheduler(optimizer, num_steps)

    model = model.to(device)
    model.train()
    step = 0
    for epoch in range(config["trainer"]["num_epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            result = mfc_loss(
                model,
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["labels"].to(device), batch["language_ids"].to(device),
                batch["attribute_ids"].to(device),
                MFCConfig(),
            )
            result["total_loss"].backward()
            optimizer.step()
            scheduler.step()
            if step % config["trainer"]["log_every"] == 0:
                logger.info(
                    "[MFC epoch %d step %d] total=%.4f task=%.4f lf=%.4f td=%.4f",
                    epoch, step, float(result["total_loss"]), float(result["task_loss"]),
                    float(result["language_fusion_loss"]), float(result["text_debiasing_loss"]),
                )
            step += 1
    return model


def train_madl(model, config, device):
    from baselines.madl import MADLModel, MADLConfig
    from optimization.optimizer import build_optimizer, build_scheduler

    dataloader, vocabs = build_sentiment_dataloader_with_vocabs(config)
    num_attribute_classes = len(vocabs["attribute"])
    if num_attribute_classes < 2:
        raise ValueError(
            f"MADL needs >=2 attribute classes in the training data's vocab, "
            f"found {num_attribute_classes}. Check your paired data's `attribute` field."
        )

    madl_model = MADLModel(model, num_classes_per_attribute={"attribute": num_attribute_classes}, config=MADLConfig())
    madl_model = madl_model.to(device)
    madl_model.train()

    num_steps = len(dataloader) * config["trainer"]["num_epochs"]
    optimizer = build_optimizer(madl_model, learning_rate=config["trainer"]["learning_rate"])
    scheduler = build_scheduler(optimizer, num_steps)

    step = 0
    for epoch in range(config["trainer"]["num_epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            result = madl_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                attribute_labels={"attribute": batch["attribute_ids"].to(device)},
            )
            result["total_loss"].backward()
            optimizer.step()
            scheduler.step()
            if step % config["trainer"]["log_every"] == 0:
                logger.info(
                    "[MADL epoch %d step %d] total=%.4f task=%.4f adversarial=%.4f",
                    epoch, step, float(result["total_loss"]), float(result["task_loss"]),
                    float(result["adversarial_loss"]),
                )
            step += 1
    return madl_model
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train all ADAPT-BTS baselines.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--baselines", nargs="+", default=AVAILABLE_BASELINES, choices=AVAILABLE_BASELINES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_and_merge(args.model_config, args.task_config)
    if config["task"] != "sentiment":
        raise NotImplementedError("baselines/train_all_baselines.py currently supports the sentiment task only.")

    for baseline_name in args.baselines:
        logger.info("=== Training baseline: %s ===", baseline_name)
        model = build_model(config)

        if baseline_name == "mt5_ft":
            dataloader = build_sentiment_dataloader(config)
            train_mt5_ft(model, dataloader, config, args.device)
        elif baseline_name == "csd":
            dataloader = build_sentiment_dataloader(config)
            from baselines.csd import csd_loss, CSDConfig
            train_generic_baseline(
                model, dataloader, config, args.device,
                loss_fn=lambda m, ii, am, lb, iicf, amcf: csd_loss(m, ii, am, lb, iicf, amcf, CSDConfig()),
                extra_batch_fields=["input_ids_cf", "attention_mask_cf"],
            )
        elif baseline_name == "grad_unlearn":
            dataloader = build_sentiment_dataloader(config)
            from baselines.grad_unlearn import grad_unlearn_step, GradUnlearnConfig
            train_generic_baseline(
                model, dataloader, config, args.device,
                loss_fn=lambda m, ii, am, lb, iicf, amcf: grad_unlearn_step(m, ii, am, lb, iicf, amcf, GradUnlearnConfig()),
                extra_batch_fields=["input_ids_cf", "attention_mask_cf"],
            )
        elif baseline_name == "mfc":
            train_mfc(model, config, args.device)
        elif baseline_name == "madl":
            train_madl(model, config, args.device)
        elif baseline_name == "magnet":
            dataloader = build_sentiment_dataloader(config)
            from baselines.magnet import magnet_proxy_loss, MagnetProxyConfig
            from fairness.demographic_dictionaries import EXAMPLE_TOKEN_COUNTS
            logger.warning("Training the MAGNET-PROXY (loss reweighting), not real MAGNET -- see baselines/magnet.py docstring.")

            def _magnet_loss_fn(m, ii, am, lb):
                return magnet_proxy_loss(m, ii, am, lb, languages=["en"] * ii.size(0),
                                          config=MagnetProxyConfig(EXAMPLE_TOKEN_COUNTS))
            train_generic_baseline(model, dataloader, config, args.device, loss_fn=_magnet_loss_fn, extra_batch_fields=[])

        logger.info("=== Finished baseline: %s ===", baseline_name)


if __name__ == "__main__":
    main()
