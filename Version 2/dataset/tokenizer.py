"""
Tokenizer wrapper for the sentiment and NER tasks.

The label-alignment logic (mapping word-level NER tags onto subword
tokens produced by a Fast tokenizer's `word_ids()`) is pure Python/stdlib
and is factored into `align_labels_with_subwords`, which is deliberately
independent of any specific tokenizer instance -- it only needs the
`word_ids` list a fast tokenizer's `BatchEncoding.word_ids(i)` produces.
This lets it be unit-tested with a hand-constructed `word_ids` list,
without needing a real HF tokenizer/model download.

The actual `TextTokenizer` class wraps a real HF tokenizer and IS
torch/transformers-dependent, per model/encoder.py's `load_tokenizer`.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Pure logic (testable without transformers installed)
# ---------------------------------------------------------------------------

def align_labels_with_subwords(
    word_ids: list[Optional[int]],
    word_labels: list[str],
    label_all_subword_tokens: bool = False,
    ignore_label: int = -100,
) -> list:
    """
    Standard NER subword-label-alignment scheme (as used throughout the
    HF token-classification examples): a word tokenized into multiple
    subword pieces gets its label assigned to the FIRST subword only by
    default; subsequent subwords and special tokens ([CLS], [SEP], pad)
    get `ignore_label` so they don't contribute to the loss/metrics.

    Args:
        word_ids: one entry per subword token in the tokenized sequence;
            `None` for special tokens, otherwise the index into the
            original `word_labels` list that subword came from. This is
            exactly what a HF Fast tokenizer's `encoding.word_ids()`
            returns.
        word_labels: the original per-WORD label strings (WikiAnn's
            `tags`, e.g. from datasets/download_wikiann.py's converted
            records), NOT yet mapped to integer ids.
        label_all_subword_tokens: if True, all subwords of a multi-piece
            word get the label (with B- converted to I- on continuation
            pieces, matching common IOB2 conventions); if False
            (default), only the first subword is labeled.

    Returns:
        A list, same length as word_ids, of either label strings or
        `ignore_label` (int) for ignored positions. Caller is responsible
        for converting label strings to integer ids via the task's tag
        vocabulary (e.g. WIKIANN_TAGS in model/heads.py).
    """
    aligned = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            aligned.append(ignore_label)
        elif wid != prev_word_id:
            aligned.append(word_labels[wid])
        else:
            if label_all_subword_tokens:
                label = word_labels[wid]
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                aligned.append(label)
            else:
                aligned.append(ignore_label)
        prev_word_id = wid
    return aligned


def dynamic_pad_sequences(sequences: list[list[int]], pad_value: int = 0, max_length: Optional[int] = None):
    """
    Dynamic padding: pads a batch of variable-length integer sequences to
    the longest sequence in the batch (or to `max_length` if given, with
    truncation for anything longer). Returns (padded, attention_masks) as
    plain Python lists of lists -- the torch-specific collate function
    (dataloaders.py) converts these to tensors.
    """
    if max_length is None:
        max_length = max(len(s) for s in sequences)

    padded, masks = [], []
    for seq in sequences:
        seq = seq[:max_length]
        pad_len = max_length - len(seq)
        padded.append(seq + [pad_value] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return padded, masks


# ---------------------------------------------------------------------------
# Real tokenizer wrapper (requires transformers)
# ---------------------------------------------------------------------------

class TextTokenizer:
    """Thin, task-agnostic wrapper around a HF fast tokenizer, used by
    both SentimentDataset and WikiAnnDataset in dataloaders.py."""

    def __init__(self, model_name_or_path: str, max_length: int = 128):
        from model.encoder import load_tokenizer  # deferred import: requires transformers
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.max_length = max_length

    def encode_text(self, text: str) -> dict:
        """For sentiment: standard single-sequence encoding."""
        return self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False,
        )

    def encode_pretokenized(self, tokens: list[str]) -> tuple[dict, list]:
        """
        For NER: WikiAnn records are already word-tokenized (see
        datasets/download_wikiann.py's `tokens` field). `is_split_into_words=True`
        makes the fast tokenizer subword-tokenize each pre-split word while
        still exposing `word_ids()` for label alignment.
        """
        encoding = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            max_length=self.max_length, padding=False,
        )
        word_ids = encoding.word_ids()
        return encoding, word_ids
