"""
Framework-agnostic dataset utilities shared by both the sentiment and
WikiAnn/NER builders: cleaning -> dedup -> unicode normalization ->
train/val/test split -> JSONL output (the pipeline described in the
"SENTIMENT DATASET" section of the implementation brief).

Deliberately has zero dependencies beyond the stdlib, so it can be
exercised by unit tests without network access or heavy ML packages
installed.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Unicode normalization step of the preprocessing pipeline."""
    return unicodedata.normalize(form, text)


def clean_text(text: str) -> str:
    """
    Basic cleaning: unicode-normalize, collapse whitespace, strip control
    characters. Deliberately conservative -- does not strip punctuation or
    lowercase, since sentiment/NER labels can depend on both, and many of
    the 101 languages are not well served by ASCII-centric cleaning rules.
    """
    text = normalize_unicode(text)
    # Strip non-printable control characters (category Cc), but keep
    # whitespace-class separators (categories Zs, Zl, Zp) which get
    # collapsed below instead.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t")
    text = " ".join(text.split())
    return text.strip()


def is_empty_or_too_short(text: str, min_chars: int = 2) -> bool:
    return len(text.strip()) < min_chars


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _hash_key(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def deduplicate(records: Iterable[dict], text_field: str = "text") -> list[dict]:
    """
    Exact-match deduplication on normalized text (case-insensitive,
    whitespace-stripped). For near-duplicate detection (e.g. minor
    punctuation variants), a MinHash/SimHash pass could be layered on top
    of this -- not implemented here since it needs a decision on
    similarity threshold that isn't specified in the manuscript.
    """
    seen: set[str] = set()
    out = []
    for rec in records:
        key = _hash_key(rec[text_field])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

@dataclass
class SplitRatios:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self):
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def train_val_test_split(
    records: list[dict],
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
    stratify_field: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Deterministic (seeded) split. If `stratify_field` is given (e.g.
    "label" or "language"), splitting is performed independently within
    each stratum so class/language proportions are preserved across
    train/val/test -- important here since label and language
    distributions are both highly imbalanced (Sec 3.1, Fig. 1).
    """
    rng = random.Random(seed)

    def _split_flat(items: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios.train)
        n_val = int(n * ratios.validation)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    if stratify_field is None:
        return _split_flat(records)

    groups: dict = {}
    for rec in records:
        groups.setdefault(rec.get(stratify_field), []).append(rec)

    train, val, test = [], [], []
    for _, group_items in groups.items():
        tr, va, te = _split_flat(group_items)
        train.extend(tr)
        val.extend(va)
        test.extend(te)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_splits_jsonl(
    train: list[dict], val: list[dict], test: list[dict], out_dir: str | Path
) -> dict[str, int]:
    out_dir = Path(out_dir)
    counts = {
        "train": write_jsonl(train, out_dir / "train.jsonl"),
        "validation": write_jsonl(val, out_dir / "validation.jsonl"),
        "test": write_jsonl(test, out_dir / "test.jsonl"),
    }
    return counts


# ---------------------------------------------------------------------------
# Full cleaning pipeline for a single raw record
# ---------------------------------------------------------------------------

def clean_pipeline(records: Iterable[dict], text_field: str = "text", min_chars: int = 2) -> list[dict]:
    """
    Cleaning -> dedup step of the pipeline diagram in the brief
    ("Download -> Cleaning -> Deduplication -> Unicode normalization ->
    Language verification -> Tokenization -> Split -> JSONL"). Language
    verification and tokenization are separate steps (language_filter.py,
    tokenizer.py) since they need pluggable model backends.
    """
    cleaned = []
    for rec in records:
        text = clean_text(rec[text_field])
        if is_empty_or_too_short(text, min_chars=min_chars):
            continue
        new_rec = dict(rec)
        new_rec[text_field] = text
        cleaned.append(new_rec)
    return deduplicate(cleaned, text_field=text_field)
