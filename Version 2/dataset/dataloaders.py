"""
PyTorch Dataset / collate_fn implementations for both tasks, reading the
JSONL files produced by datasets/build_sentiment.py and
datasets/build_wikiann.py.

Requires torch. The tokenization/label-alignment logic these call into
(datasets/tokenizer.py) is separately unit-tested without torch; this
file just wires that logic to torch tensors and HF tokenizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover
    raise ImportError("model/dataloaders.py requires PyTorch.") from e

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.dataset_utils import read_jsonl
from datasets.tokenizer import TextTokenizer, align_labels_with_subwords, dynamic_pad_sequences
from datasets.vocab import LabelVocab
from model.classifier import SENTIMENT_LABELS
from model.heads import WIKIANN_TAGS


SENTIMENT_LABEL_TO_ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
WIKIANN_TAG_TO_ID = {tag: i for i, tag in enumerate(WIKIANN_TAGS)}


class SentimentDataset(Dataset):
    """
    Reads a sentiment JSONL split (train/validation/test.jsonl from
    build_sentiment.py). Each item also carries the raw `language` and
    `attribute` (demographic attribute `a`, if present) fields, needed
    downstream for per-language evaluation (Eq. 16/20) and for building
    counterfactual pairs (x, x^(b)) during training.
    """

    def __init__(self, jsonl_path: str, tokenizer: TextTokenizer):
        self.records = list(read_jsonl(jsonl_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        encoding = self.tokenizer.encode_text(rec["text"])
        return {
            "input_ids": encoding["input_ids"],
            "label": SENTIMENT_LABEL_TO_ID[rec["label"]],
            "language": rec.get("language"),
            "attribute": rec.get("attribute"),
            "raw_text": rec["text"],
        }


def sentiment_collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    input_ids_list = [item["input_ids"] for item in batch]
    padded, masks = dynamic_pad_sequences(input_ids_list, pad_value=pad_token_id)
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "languages": [item["language"] for item in batch],
        "attributes": [item["attribute"] for item in batch],
    }


class PairedSentimentDataset(Dataset):
    """
    Reads the paired JSONL produced by
    datasets/build_counterfactual_pairs.py -- each record has BOTH
    `text`/`label` and `cf_text`/`cf_attribute`. This is what
    optimization/trainer.py's train_step actually consumes (it expects
    `input_ids_cf`/`attention_mask_cf` in every batch, for the per-batch
    BTS estimation in Algorithm 2 line 6).

    Note this is a DIFFERENT (smaller) dataset than the flat
    SentimentDataset above: only samples where counterfactual generation
    succeeded AND was accepted (Eq. 9) appear here, whereas
    SentimentDataset can read the full train/validation/test splits
    (including samples with no usable counterfactual). A full training
    setup would likely use SentimentDataset for general task-loss
    training and PairedSentimentDataset specifically for the fairness
    term -- how to combine the two sampling streams is a training-loop
    design decision left to configs/*.yaml, not fixed here.

    Optional `language_vocab`/`attribute_vocab` (datasets/vocab.py):
    when supplied, __getitem__ also emits integer `language_id`/
    `attribute_id`/`cf_attribute_id` fields, which baselines/mfc.py's
    contrastive masks and baselines/madl.py's attribute discriminators
    require (they can't consume raw strings as tensor inputs). Without a
    vocab, those fields are omitted and the item is only usable with
    optimization/trainer.py (ADAPT-BTS itself never needed integer
    attribute ids, since BTS/IBADR operate on model predictions, not on
    the attribute label directly).
    """

    def __init__(self, jsonl_path: str, tokenizer: TextTokenizer,
                 language_vocab: "LabelVocab | None" = None,
                 attribute_vocab: "LabelVocab | None" = None):
        self.records = list(read_jsonl(jsonl_path))
        self.tokenizer = tokenizer
        self.language_vocab = language_vocab
        self.attribute_vocab = attribute_vocab

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        enc = self.tokenizer.encode_text(rec["text"])
        enc_cf = self.tokenizer.encode_text(rec["cf_text"])
        item = {
            "input_ids": enc["input_ids"],
            "input_ids_cf": enc_cf["input_ids"],
            "label": SENTIMENT_LABEL_TO_ID[rec["label"]],
            "language": rec.get("language"),
            "attribute": rec.get("attribute"),
            "cf_attribute": rec.get("cf_attribute"),
        }
        if self.language_vocab is not None:
            item["language_id"] = self.language_vocab.encode(rec.get("language"))
        if self.attribute_vocab is not None:
            item["attribute_id"] = self.attribute_vocab.encode(rec.get("attribute"))
            item["cf_attribute_id"] = self.attribute_vocab.encode(rec.get("cf_attribute"))
        return item


def paired_sentiment_collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Produces exactly the batch schema optimization/trainer.py's
    train_step expects: input_ids/attention_mask for the original AND
    input_ids_cf/attention_mask_cf for the counterfactual, padded
    independently (original and counterfactual texts can have different
    lengths after a lexicon substitution changes word count).

    If the underlying PairedSentimentDataset was built with
    language_vocab/attribute_vocab (datasets/vocab.py), this also
    collates `language_ids`/`attribute_ids`/`cf_attribute_ids` as long
    tensors -- required by baselines/mfc.py's contrastive masks and
    baselines/madl.py's attribute discriminators, neither of which can
    consume the raw string labels."""
    orig_padded, orig_masks = dynamic_pad_sequences([b["input_ids"] for b in batch], pad_value=pad_token_id)
    cf_padded, cf_masks = dynamic_pad_sequences([b["input_ids_cf"] for b in batch], pad_value=pad_token_id)
    out = {
        "input_ids": torch.tensor(orig_padded, dtype=torch.long),
        "attention_mask": torch.tensor(orig_masks, dtype=torch.long),
        "input_ids_cf": torch.tensor(cf_padded, dtype=torch.long),
        "attention_mask_cf": torch.tensor(cf_masks, dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "languages": [b["language"] for b in batch],
        "attributes": [b["attribute"] for b in batch],
        "cf_attributes": [b["cf_attribute"] for b in batch],
    }
    if "language_id" in batch[0]:
        out["language_ids"] = torch.tensor([b["language_id"] for b in batch], dtype=torch.long)
    if "attribute_id" in batch[0]:
        out["attribute_ids"] = torch.tensor([b["attribute_id"] for b in batch], dtype=torch.long)
        out["cf_attribute_ids"] = torch.tensor([b["cf_attribute_id"] for b in batch], dtype=torch.long)
    return out


class WikiAnnDataset(Dataset):
    """Reads a per-language WikiAnn JSONL split from
    build_wikiann.py/download_wikiann.py's output."""

    def __init__(self, jsonl_path: str, tokenizer: TextTokenizer,
                 label_all_subword_tokens: bool = False):
        self.records = list(read_jsonl(jsonl_path))
        self.tokenizer = tokenizer
        self.label_all_subword_tokens = label_all_subword_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        encoding, word_ids = self.tokenizer.encode_pretokenized(rec["tokens"])
        aligned = align_labels_with_subwords(
            word_ids, rec["tags"], label_all_subword_tokens=self.label_all_subword_tokens,
        )
        label_ids = [
            WIKIANN_TAG_TO_ID[t] if t != -100 else -100
            for t in aligned
        ]
        return {
            "input_ids": encoding["input_ids"],
            "label_ids": label_ids,
            "language": rec.get("language"),
        }


def ner_collate_fn(batch: list[dict], pad_token_id: int = 0, label_pad: int = -100) -> dict:
    input_ids_list = [item["input_ids"] for item in batch]
    label_ids_list = [item["label_ids"] for item in batch]

    padded_inputs, masks = dynamic_pad_sequences(input_ids_list, pad_value=pad_token_id)
    padded_labels, _ = dynamic_pad_sequences(label_ids_list, pad_value=label_pad,
                                              max_length=len(padded_inputs[0]))
    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "label_ids": torch.tensor(padded_labels, dtype=torch.long),
        "languages": [item["language"] for item in batch],
    }


class PairedWikiAnnDataset(Dataset):
    """
    NER analog of PairedSentimentDataset: reads the paired JSONL produced
    by datasets/build_ner_counterfactual_pairs.py (tokens/tags AND
    cf_tokens, tags identical for both by construction -- see
    fairness/ner_counterfactual_generation.py). Produces
    input_ids/input_ids_cf plus a single label_ids (shared by both, since
    the tag sequence doesn't change under substitution), matching what
    optimization/trainer.py's NER branch needs for BTS computation via
    model/heads.py:NERModel.flatten_for_bts.
    """

    def __init__(self, jsonl_path: str, tokenizer: TextTokenizer,
                 label_all_subword_tokens: bool = False):
        self.records = list(read_jsonl(jsonl_path))
        self.tokenizer = tokenizer
        self.label_all_subword_tokens = label_all_subword_tokens

    def __len__(self):
        return len(self.records)

    def _encode_with_labels(self, tokens: list[str], tags: list[str]) -> tuple[list[int], list[int]]:
        encoding, word_ids = self.tokenizer.encode_pretokenized(tokens)
        aligned = align_labels_with_subwords(
            word_ids, tags, label_all_subword_tokens=self.label_all_subword_tokens,
        )
        label_ids = [WIKIANN_TAG_TO_ID[t] if t != -100 else -100 for t in aligned]
        return encoding["input_ids"], label_ids

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        input_ids, label_ids = self._encode_with_labels(rec["tokens"], rec["tags"])
        # cf_tokens share the SAME tag sequence (tags unchanged by
        # construction), but subword tokenization can still split
        # cf_tokens differently than tokens (e.g. "She" vs "He" have
        # different subword counts in some tokenizers), so label
        # alignment is re-run independently on the cf side.
        input_ids_cf, label_ids_cf = self._encode_with_labels(rec["cf_tokens"], rec["tags"])
        return {
            "input_ids": input_ids,
            "input_ids_cf": input_ids_cf,
            "label_ids": label_ids,
            "label_ids_cf": label_ids_cf,
            "language": rec.get("language"),
            "attribute": rec.get("attribute"),
            "cf_attribute": rec.get("cf_attribute"),
        }


def paired_ner_collate_fn(batch: list[dict], pad_token_id: int = 0, label_pad: int = -100) -> dict:
    orig_inputs = [b["input_ids"] for b in batch]
    cf_inputs = [b["input_ids_cf"] for b in batch]
    orig_labels = [b["label_ids"] for b in batch]

    padded_inputs, masks = dynamic_pad_sequences(orig_inputs, pad_value=pad_token_id)
    padded_inputs_cf, masks_cf = dynamic_pad_sequences(cf_inputs, pad_value=pad_token_id)
    padded_labels, _ = dynamic_pad_sequences(orig_labels, pad_value=label_pad, max_length=len(padded_inputs[0]))

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "input_ids_cf": torch.tensor(padded_inputs_cf, dtype=torch.long),
        "attention_mask_cf": torch.tensor(masks_cf, dtype=torch.long),
        "label_ids": torch.tensor(padded_labels, dtype=torch.long),
        "languages": [b["language"] for b in batch],
        "attributes": [b["attribute"] for b in batch],
        "cf_attributes": [b["cf_attribute"] for b in batch],
    }


class MultilingualDataset(Dataset):
    """
    Wraps either a SentimentDataset or WikiAnnDataset with per-language
    weighting metadata, for use with a WeightedRandomSampler if
    up/down-sampling low-resource languages is desired during training
    (an implementation choice left to configs/*.yaml -- the manuscript's
    Sec 5.2 instance-weighting is applied at EVALUATION aggregation time,
    Eq. 16/20, not necessarily at training sampling time; this class just
    exposes the language labels needed either way).
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]

    def language_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in range(len(self.base)):
            lang = self.base[i].get("language")
            counts[lang] = counts.get(lang, 0) + 1
        return counts


class CombinedDataset(Dataset):
    """
    Concatenates multiple per-language or per-task datasets into one,
    tagging each item with its source dataset index -- used when a
    single training loop needs to draw batches across all 101 languages
    (or across sentiment+NER jointly, if a multi-task run is configured).
    """

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for ds in datasets:
            self.offsets.append(total)
            total += len(ds)
        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, idx: int):
        # binary search over offsets would be more efficient for many
        # datasets; linear scan is fine for the language counts here (<=101)
        for i in range(len(self.datasets) - 1, -1, -1):
            if idx >= self.offsets[i]:
                return self.datasets[i][idx - self.offsets[i]]
        raise IndexError(idx)


def build_dataloader(dataset: Dataset, collate_fn, batch_size: int = 32,
                      shuffle: bool = True, num_workers: int = 2) -> "DataLoader":
    """Standard DataLoader factory. AMP/DDP are applied at the trainer
    level (optimization/trainer.py), not here."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers,
    )
