"""
Full evaluation pipeline: runs a trained model over a DataLoader, collects
predictions per-language, and computes every metric in Table 4/5:
Macro-F1 or Span-F1, BTS, CCR, DPG, Leakage -- per-language AND globally
via instance-weighted aggregation (Eq. 16/20, Sec 5.2: "measures per
language are calculated independently ... global measures are calculated
using instance-weighted averaging").

Requires torch + a real trained model. Not runnable in this sandbox (no
torch, no GPU, no trained checkpoint) -- syntax-checked only. The metrics
this calls into (evaluation/metrics.py, evaluation/leakage.py,
evaluation/fairness_metrics.py, fairness/bias_transfer_score.py) are all
independently unit-tested with synthetic data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("evaluation/evaluator.py requires PyTorch.") from e

import numpy as np

from evaluation.metrics import macro_f1, span_f1_from_ids, span_f1
from evaluation.leakage import compute_leakage
from evaluation.fairness_metrics import (
    counterfactual_consistency_rate, demographic_parity_gap,
    instance_weighted_global_metric,
)
from fairness.bias_transfer_score import compute_bts


@dataclass
class PerLanguageResults:
    language: str
    n: int
    task_metric: float          # Macro-F1 or Span-F1, depending on task
    bts: Optional[float] = None
    ccr: Optional[float] = None
    dpg: Optional[float] = None
    leakage: Optional[float] = None


@dataclass
class EvaluationReport:
    per_language: dict[str, PerLanguageResults] = field(default_factory=dict)
    global_task_metric: Optional[float] = None
    global_bts: Optional[float] = None
    global_ccr: Optional[float] = None
    global_dpg: Optional[float] = None
    global_leakage: Optional[float] = None

    def as_table_row(self) -> dict:
        """Matches Table 5's column layout for easy CSV/DataFrame export
        once evaluation/report.py (not yet written) needs to render it."""
        return {
            "Macro-F1 or Span-F1": self.global_task_metric,
            "BTS": self.global_bts,
            "CCR (%)": self.global_ccr * 100 if self.global_ccr is not None else None,
            "DPG": self.global_dpg,
            "Leakage": self.global_leakage,
        }


@torch.no_grad()
def collect_predictions_sentiment(model, dataloader, device: str = "cuda") -> dict:
    """
    Runs the model over a SentimentDataset-based DataLoader (or
    PairedSentimentDataset, if counterfactual pairs are available for
    BTS/CCR/DPG -- see below) and collects everything needed for
    evaluation, grouped by language.
    """
    model.eval()
    by_language = defaultdict(lambda: {
        "y_true": [], "y_pred": [], "logits": [], "logits_cf": [], "pooled": [], "attributes": [],
    })

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        # Also grab pooled encoder representations for the leakage probe
        _, pooled = model.encoder(input_ids, attention_mask)
        pooled = pooled.cpu().numpy()

        has_cf = "input_ids_cf" in batch
        if has_cf:
            input_ids_cf = batch["input_ids_cf"].to(device)
            attention_mask_cf = batch["attention_mask_cf"].to(device)
            logits_cf = model(input_ids=input_ids_cf, attention_mask=attention_mask_cf)

        for i, lang in enumerate(batch["languages"]):
            entry = by_language[lang]
            entry["y_true"].append(int(labels[i]))
            entry["y_pred"].append(int(preds[i]))
            entry["logits"].append(logits[i].cpu().numpy())
            entry["pooled"].append(pooled[i])
            entry["attributes"].append(batch["attributes"][i])
            if has_cf:
                entry["logits_cf"].append(logits_cf[i].cpu().numpy())

    return dict(by_language)


@torch.no_grad()
def collect_predictions_ner(model, dataloader, id_to_tag: dict[int, str], device: str = "cuda") -> dict:
    """
    NER analog of collect_predictions_sentiment. Runs the model over a
    PairedWikiAnnDataset-based DataLoader (or WikiAnnDataset if no
    counterfactual pairs are available -- BTS/CCR simply won't be
    populated in that case), grouped by language.

    Per-sample (per-sequence) BTS is computed by masking each sample's
    own valid (non-padded) tokens BEFORE flattening -- i.e. one BTS
    scalar per sequence (mean TV distance over that sequence's tokens,
    Eq. 15 generalized from per-token to per-sequence by averaging), so
    each sample can be attributed to its own language and per-language
    aggregation (Eq. 16/20) works the same way it does for sentiment.
    This replaces an earlier version of this function that flattened
    BTS across the whole batch before any language split, which could
    only report a global figure.
    """
    from fairness.bias_transfer_score import total_variation_distance

    model.eval()
    by_language = defaultdict(lambda: {
        "true_tags": [], "pred_tags": [], "per_sample_bts": [], "per_sample_ccr": [],
    })

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_ids = batch["label_ids"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (B, T, C)
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        label_ids_np = label_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()

        has_cf = "input_ids_cf" in batch
        if has_cf:
            input_ids_cf = batch["input_ids_cf"].to(device)
            attention_mask_cf = batch["attention_mask_cf"].to(device)
            logits_cf = model(input_ids=input_ids_cf, attention_mask=attention_mask_cf)
            logits_np = logits.cpu().numpy()
            logits_cf_np = logits_cf.cpu().numpy()
            attention_mask_cf_np = attention_mask_cf.cpu().numpy()
            pred_ids_cf = logits_cf.argmax(dim=-1).cpu().numpy()

        for i, lang in enumerate(batch["languages"]):
            entry = by_language[lang]
            mask_i = label_ids_np[i] != -100
            true_seq = [id_to_tag[t] for t in label_ids_np[i][mask_i]]
            pred_seq = [id_to_tag[p] for p in pred_ids[i][mask_i]]
            entry["true_tags"].append(true_seq)
            entry["pred_tags"].append(pred_seq)

            if has_cf:
                # Per-SEQUENCE BTS: softmax + TV distance over this
                # sample's own valid tokens only, then averaged into one
                # scalar -- keeps the sample (and therefore its
                # language) identifiable, unlike flattening the whole
                # batch first.
                valid_a = attention_mask_np[i].astype(bool)
                valid_b = attention_mask_cf_np[i].astype(bool)
                logits_a_i = logits_np[i][valid_a]
                logits_b_i = logits_cf_np[i][valid_b]
                if len(logits_a_i) > 0 and len(logits_b_i) > 0:
                    n = min(len(logits_a_i), len(logits_b_i))  # truncate to shorter side if lengths differ
                    probs_a = np.exp(logits_a_i[:n]) / np.exp(logits_a_i[:n]).sum(axis=-1, keepdims=True)
                    probs_b = np.exp(logits_b_i[:n]) / np.exp(logits_b_i[:n]).sum(axis=-1, keepdims=True)
                    per_token_tv = total_variation_distance(probs_a, probs_b)
                    entry["per_sample_bts"].append(float(per_token_tv.mean()))

                    preds_a_i = pred_ids[i][valid_a][:n]
                    preds_b_i = pred_ids_cf[i][valid_b][:n]
                    entry["per_sample_ccr"].append(float(np.mean(preds_a_i == preds_b_i)))

    return dict(by_language)


def evaluate_ner(
    model, dataloader, id_to_tag: dict[int, str], device: str = "cuda",
) -> EvaluationReport:
    """
    Table 5-style evaluation for NER: per-language AND instance-weighted
    global Span-F1, BTS, and CCR (Eq. 16/20) -- all three are now
    populated per-language, not just globally, since
    collect_predictions_ner computes BTS/CCR per sequence.
    """
    by_language = collect_predictions_ner(model, dataloader, id_to_tag, device)

    per_language_results = {}
    per_language_n = {}
    per_language_task = {}
    per_language_bts = {}
    per_language_ccr = {}

    for lang, data in by_language.items():
        result = span_f1(data["true_tags"], data["pred_tags"])
        n = len(data["true_tags"])

        bts_val = float(np.mean(data["per_sample_bts"])) if data["per_sample_bts"] else None
        ccr_val = float(np.mean(data["per_sample_ccr"])) if data["per_sample_ccr"] else None

        per_language_results[lang] = PerLanguageResults(
            language=lang, n=n, task_metric=result.f1, bts=bts_val, ccr=ccr_val,
        )
        per_language_n[lang] = n
        per_language_task[lang] = result.f1
        if bts_val is not None:
            per_language_bts[lang] = bts_val
        if ccr_val is not None:
            per_language_ccr[lang] = ccr_val

    report = EvaluationReport(per_language=per_language_results)
    report.global_task_metric = instance_weighted_global_metric(per_language_task, per_language_n)

    if per_language_bts:
        n_subset = {k: per_language_n[k] for k in per_language_bts}
        report.global_bts = instance_weighted_global_metric(per_language_bts, n_subset)
    if per_language_ccr:
        n_subset = {k: per_language_n[k] for k in per_language_ccr}
        report.global_ccr = instance_weighted_global_metric(per_language_ccr, n_subset)

    return report


def evaluate_sentiment(
    model, dataloader, device: str = "cuda", compute_leakage_probe: bool = True,
) -> EvaluationReport:
    """
    Full Table 5-style evaluation for the sentiment task. BTS/CCR/DPG are
    only computed for languages where the dataloader supplied
    counterfactual pairs (`input_ids_cf`) -- i.e. this should typically
    be run over a DataLoader wrapping PairedSentimentDataset for the
    fairness metrics to be populated, or SentimentDataset (no `_cf`
    fields) if you only want Macro-F1/Leakage.
    """
    by_language = collect_predictions_sentiment(model, dataloader, device)

    per_language_results = {}
    per_language_n = {}
    per_language_task = {}
    per_language_bts = {}
    per_language_ccr = {}
    per_language_dpg = {}
    per_language_leakage = {}

    for lang, data in by_language.items():
        n = len(data["y_true"])
        task_metric = macro_f1(np.array(data["y_true"]), np.array(data["y_pred"]))

        bts_val = ccr_val = dpg_val = leakage_val = None

        if data["logits_cf"]:
            logits = np.array(data["logits"])
            logits_cf = np.array(data["logits_cf"])
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            probs_cf = np.exp(logits_cf) / np.exp(logits_cf).sum(axis=-1, keepdims=True)
            bts_result = compute_bts(probs, probs_cf)
            bts_val = bts_result.mean

            preds_a = np.argmax(logits, axis=-1)
            preds_b = np.argmax(logits_cf, axis=-1)
            ccr_val = counterfactual_consistency_rate(preds_a, preds_b)

            attrs = np.array(data["attributes"])
            if len(set(attrs.tolist())) >= 2:
                dpg_val = demographic_parity_gap(preds_a, attrs, positive_class=int(np.max(preds_a)))

        if compute_leakage_probe and data["attributes"] and len(set(a for a in data["attributes"] if a is not None)) >= 2:
            attr_to_id = {a: i for i, a in enumerate(sorted(set(data["attributes"])))}
            attr_ids = np.array([attr_to_id[a] for a in data["attributes"]])
            leakage_result = compute_leakage(np.array(data["pooled"]), attr_ids)
            leakage_val = leakage_result.probe_accuracy

        per_language_results[lang] = PerLanguageResults(
            language=lang, n=n, task_metric=task_metric,
            bts=bts_val, ccr=ccr_val, dpg=dpg_val, leakage=leakage_val,
        )
        per_language_n[lang] = n
        per_language_task[lang] = task_metric
        if bts_val is not None:
            per_language_bts[lang] = bts_val
        if ccr_val is not None:
            per_language_ccr[lang] = ccr_val
        if dpg_val is not None:
            per_language_dpg[lang] = dpg_val
        if leakage_val is not None:
            per_language_leakage[lang] = leakage_val

    report = EvaluationReport(per_language=per_language_results)
    report.global_task_metric = instance_weighted_global_metric(per_language_task, per_language_n)
    if per_language_bts:
        n_subset = {k: per_language_n[k] for k in per_language_bts}
        report.global_bts = instance_weighted_global_metric(per_language_bts, n_subset)
    if per_language_ccr:
        n_subset = {k: per_language_n[k] for k in per_language_ccr}
        report.global_ccr = instance_weighted_global_metric(per_language_ccr, n_subset)
    if per_language_dpg:
        n_subset = {k: per_language_n[k] for k in per_language_dpg}
        report.global_dpg = instance_weighted_global_metric(per_language_dpg, n_subset)
    if per_language_leakage:
        n_subset = {k: per_language_n[k] for k in per_language_leakage}
        report.global_leakage = instance_weighted_global_metric(per_language_leakage, n_subset)

    return report
