"""
Integer-id vocabularies for categorical string fields (language,
demographic attribute) that need to become tensor-friendly ids for
MFC's language/attribute contrastive masks (baselines/mfc.py) and MADL's
attribute discriminators (baselines/madl.py) -- both need integer class
indices, not the raw strings PairedSentimentDataset's collate function
currently returns.

Deliberately simple (sorted-unique-value -> index mapping, built once
from the training data and reused at eval time) rather than a general
vocabulary framework, since the closed categorical sets here (a
handful of languages, 2-3 attribute values) don't need anything fancier.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LabelVocab:
    """Bidirectional string<->int mapping for a closed categorical field."""
    label_to_id: dict[str, int] = field(default_factory=dict)
    id_to_label: dict[int, str] = field(default_factory=dict)

    @classmethod
    def build(cls, labels: list) -> "LabelVocab":
        """Builds a vocab from a list of (possibly repeated, possibly
        containing None) label values. None/missing values are excluded
        from the vocab; encode() maps them to -1 (see encode_missing_as)."""
        unique = sorted({l for l in labels if l is not None})
        label_to_id = {label: i for i, label in enumerate(unique)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        return cls(label_to_id=label_to_id, id_to_label=id_to_label)

    def __len__(self) -> int:
        return len(self.label_to_id)

    def encode(self, label, missing_id: int = -1) -> int:
        if label is None:
            return missing_id
        if label not in self.label_to_id:
            raise KeyError(f"Label '{label}' not in vocab (known labels: {sorted(self.label_to_id)})")
        return self.label_to_id[label]

    def encode_batch(self, labels: list, missing_id: int = -1) -> list[int]:
        return [self.encode(l, missing_id=missing_id) for l in labels]

    def decode(self, idx: int) -> str | None:
        return self.id_to_label.get(idx)

    def as_dict(self) -> dict:
        """For serialization alongside a checkpoint, so eval-time
        encoding uses the exact same id mapping training used."""
        return {"label_to_id": self.label_to_id}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelVocab":
        label_to_id = d["label_to_id"]
        id_to_label = {i: label for label, i in label_to_id.items()}
        return cls(label_to_id=label_to_id, id_to_label=id_to_label)


def build_vocabs_from_records(records: list[dict], language_field: str = "language",
                                attribute_field: str = "attribute") -> dict[str, LabelVocab]:
    """Convenience: scans a list of JSONL-style records and builds both
    the language and attribute vocabularies in one pass, matching the
    fields PairedSentimentDataset's records carry (see
    datasets/build_counterfactual_pairs.py's output schema)."""
    languages = [r.get(language_field) for r in records]
    attributes = [r.get(attribute_field) for r in records]
    return {
        "language": LabelVocab.build(languages),
        "attribute": LabelVocab.build(attributes),
    }
