"""
Trainer implementing Algorithm 2 end-to-end:

    1: Initialize model parameters theta_0 and controller lambda_0
    2: Set fairness threshold tau
    3: for each iteration t do
    4:     Sample mini-batch B_t from D_aug
    5:     Compute task loss L_task(theta_t)
    6:     Estimate fairness metric BTS_t on B_t
    7:     Compute total loss L = L_task + lambda_t * (BTS_t - tau)
    8:     Update model parameters theta_t via gradient descent
    9:     Update controller lambda_{t+1} = lambda_t + eta_lambda*(BTS_t - tau)
    10:    Project lambda_{t+1} to [lambda_min, lambda_max]
    11: end for
    12: return trained parameters theta

plus the IBADR refresh (Algorithm 3), invoked every K steps per
fairness/ibadr.py's IBADRScheduler.

This file requires torch and a real dataset with counterfactual pairs
already attached (i.e. each batch must supply both the original and
counterfactual input_ids/attention_mask -- produced upstream by
fairness/counterfactual_generation.py, still unimplemented; see README).
Until that module exists, `batch["input_ids_cf"]` /
`batch["attention_mask_cf"]` below are placeholders for whatever that
module will eventually attach to each dataset item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover
    raise ImportError("optimization/trainer.py requires PyTorch.") from e

from fairness.fairness_controller import FairnessController, ControllerConfig
from fairness.bts_torch import compute_bts_torch
from fairness.ibadr import IBADRScheduler, IBADRConfig
from optimization.optimizer import build_optimizer, build_scheduler
from optimization.losses import sentiment_task_loss, ner_task_loss
from optimization.trainer_config import TrainerConfig, StepLog


class ADAPTBTSTrainer:
    """
    Wires together:
      - a model (model/mt5.py or model/xlmr.py factory output)
      - a FairnessController (Eq. 12-14, 17-18)
      - the differentiable BTS computation (fairness/bts_torch.py, Eq. 4/15)
      - an IBADRScheduler (Algorithm 3) for periodic data refresh

    The regenerate_fn passed to IBADRScheduler.refresh_step must be
    supplied by the caller once fairness/counterfactual_generation.py
    exists; this trainer only calls the scheduler, it doesn't implement
    generation itself (kept as an injected dependency, matching
    ibadr.py's design).
    """

    def __init__(self, model, config: TrainerConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.controller = FairnessController(ControllerConfig(
            tau=config.tau, eta_lambda=config.eta_lambda,
            lambda_init=config.lambda_init, lambda_min=config.lambda_min,
            lambda_max=config.lambda_max,
        ))
        self.ibadr = IBADRScheduler(IBADRConfig(
            refresh_interval=config.ibadr_refresh_interval,
            selection_ratio=config.ibadr_selection_ratio,
        ))
        self.task_loss_fn = sentiment_task_loss if config.task == "sentiment" else ner_task_loss

        self.history: list[StepLog] = []
        self.global_step = 0

    def _compute_task_loss(self, logits: "torch.Tensor", batch: dict) -> "torch.Tensor":
        if self.config.task == "sentiment":
            return self.task_loss_fn(logits, batch["labels"].to(self.device))
        else:
            return self.task_loss_fn(logits, batch["label_ids"].to(self.device))

    def _forward_logits_for_bts(self, logits: "torch.Tensor", logits_cf: "torch.Tensor", batch: dict):
        """For sentiment, logits are already (batch, num_labels) -- ready
        for BTS directly. For NER, flatten to (n_valid_tokens, num_labels)
        first (model/heads.py:NERModel.flatten_for_bts), masking out
        padding so BTS's expectation (Eq. 4) isn't diluted by padded
        positions."""
        if self.config.task == "sentiment":
            return logits, logits_cf
        else:
            from model.heads import NERModel
            mask = batch["attention_mask"].to(self.device)
            return (
                NERModel.flatten_for_bts(logits, mask),
                NERModel.flatten_for_bts(logits_cf, mask),
            )

    def train_step(self, batch: dict, optimizer, scheduler, scaler: Optional["torch.cuda.amp.GradScaler"] = None) -> StepLog:
        """
        One iteration of Algorithm 2 (lines 4-10).

        `batch` must contain the original input_ids/attention_mask AND
        the counterfactual input_ids_cf/attention_mask_cf, produced by
        the counterfactual generation pipeline -- UNLESS
        `config.use_counterfactual_augmentation` is False (the "-CDA"
        ablation variant), in which case only the task loss is computed
        and input_ids_cf is never touched (a plain SentimentDataset/
        WikiAnnDataset batch without `_cf` fields works fine in that
        mode).
        """
        self.model.train()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        if not self.config.use_counterfactual_augmentation:
            # "-CDA" ablation: no counterfactual pair, no BTS term --
            # the composite loss collapses to task loss alone, and the
            # controller/lambda are frozen at whatever they were
            # initialized to (there's nothing for them to respond to).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.use_amp):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                total_loss = self._compute_task_loss(logits, batch)

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            log = StepLog(
                step=self.global_step, task_loss=float(total_loss.detach().cpu()),
                bts=float("nan"), lam=self.controller.lam, total_loss=float(total_loss.detach().cpu()),
            )
            self.history.append(log)
            self.global_step += 1
            return log

        input_ids_cf = batch["input_ids_cf"].to(self.device)
        attention_mask_cf = batch["attention_mask_cf"].to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.use_amp):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits_cf = self.model(input_ids=input_ids_cf, attention_mask=attention_mask_cf)

            task_loss = self._compute_task_loss(logits, batch)

            bts_logits, bts_logits_cf = self._forward_logits_for_bts(logits, logits_cf, batch)
            bts_result = compute_bts_torch(bts_logits, bts_logits_cf)  # Eq. 4/15

            # Eq. 12 (Lagrangian form of Algorithm 2 line 7)
            lam = self.controller.lam
            total_loss = task_loss + lam * (bts_result.mean - self.config.tau)

        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Algorithm 2, lines 9-10: controller update happens on the
        # DETACHED scalar BTS value (the controller is not part of the
        # autograd graph -- lambda is a control variable, not a learned
        # parameter, matching Eq. 13's plain arithmetic update).
        #
        # "-FAPC" ablation: skip the update entirely, so lambda stays
        # pinned at lambda_init for the whole run (no adaptive
        # feedback), while the BTS term still contributes to the loss
        # at that fixed weight -- this isolates "fixed fairness
        # weighting" from "no fairness weighting at all" (-CDA above).
        bts_value = float(bts_result.mean.detach().cpu())
        if self.config.use_adaptive_controller:
            self.controller.update(bts_value)

        log = StepLog(
            step=self.global_step,
            task_loss=float(task_loss.detach().cpu()),
            bts=bts_value,
            lam=lam,
            total_loss=float(total_loss.detach().cpu()),
        )
        self.history.append(log)
        self.global_step += 1
        return log

    def train(self, dataloader: "DataLoader", regenerate_fn: Optional[Callable] = None,
               dataset_for_refresh: Optional[list] = None):
        """
        Full training loop over `config.num_epochs` epochs. IBADR refresh
        (Algorithm 3) is triggered via `self.ibadr.should_refresh(step)`
        inside the loop; actual regeneration requires `regenerate_fn` and
        `dataset_for_refresh` to be supplied once the counterfactual
        generation module exists -- until then, IBADR refresh is skipped
        with a log message rather than silently doing nothing unnoticed.
        """
        num_training_steps = len(dataloader) * self.config.num_epochs
        optimizer = build_optimizer(self.model, learning_rate=self.config.learning_rate)
        scheduler = build_scheduler(optimizer, num_training_steps)

        for epoch in range(self.config.num_epochs):
            for batch in dataloader:
                log = self.train_step(batch, optimizer, scheduler)

                if self.global_step % self.config.log_every == 0:
                    print(
                        f"[epoch {epoch} step {log.step}] "
                        f"task_loss={log.task_loss:.4f} bts={log.bts:.4f} "
                        f"lambda={log.lam:.4f} total_loss={log.total_loss:.4f}"
                    )

                if self.config.use_ibadr and self.ibadr.should_refresh(self.global_step):
                    if regenerate_fn is None or dataset_for_refresh is None:
                        print(
                            f"[step {self.global_step}] IBADR refresh due but skipped: "
                            "no regenerate_fn/dataset_for_refresh supplied "
                            "(counterfactual_generation.py not yet wired in)."
                        )
                    else:
                        # Caller is responsible for computing per-sample
                        # BTS on the current dataset beforehand and
                        # passing it in -- left as an exercise for the
                        # full data pipeline since it requires a full
                        # forward pass over the (possibly large) dataset,
                        # not just the current minibatch.
                        raise NotImplementedError(
                            "Full-dataset IBADR refresh requires computing "
                            "per-sample BTS over the whole dataset first; "
                            "wire this up once counterfactual_generation.py exists."
                        )

        return self.model
