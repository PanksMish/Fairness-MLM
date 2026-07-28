"""
NER tagging head: per-token classification over the WikiAnn IOB2 label
schema (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC -- see
datasets/download_wikiann.py:WIKIANN_TAGS).

Unlike the sentiment head (model/classifier.py), this consumes the
encoder's full token-level representations (batch, seq_len, hidden),
not the pooled sequence vector, since NER needs a per-token prediction.
This matches the manuscript's note (Sec 5.3): "Unlike sentiment
classification, NER requires precise contextual representations for
identifying entity boundaries and entity types."

For BTS computation on NER (Eq. 15), the per-token logits are reshaped to
(batch * seq_len, num_labels) before being passed to
fairness/bts_torch.py's total_variation_distance_torch, treating each
token position as one "instance" -- padded positions should be masked out
of the BTS expectation via the attention mask before averaging.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("model/heads.py requires PyTorch.") from e

WIKIANN_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class NERTaggingHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = len(WIKIANN_TAGS), dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, token_reprs: "torch.Tensor") -> "torch.Tensor":
        """Args: token_reprs (batch, seq_len, hidden_size).
        Returns: logits (batch, seq_len, num_labels)."""
        return self.classifier(self.dropout(token_reprs))


class NERModel(nn.Module):
    """Full model = encoder + tagging head. forward() returns per-token
    logits, shape (batch, seq_len, num_labels)."""

    def __init__(self, encoder, num_labels: int = len(WIKIANN_TAGS), dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = NERTaggingHead(encoder.hidden_size, num_labels, dropout)

    def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        token_reprs, _ = self.encoder(input_ids, attention_mask)
        return self.head(token_reprs)

    @staticmethod
    def flatten_for_bts(logits: "torch.Tensor", attention_mask: "torch.Tensor"):
        """
        Reshapes (batch, seq_len, num_labels) logits + mask into a flat
        (n_valid_tokens, num_labels) tensor with padding removed, suitable
        for fairness/bts_torch.py's total_variation_distance_torch, which
        expects a plain (n, num_labels) shape (Eq. 15 applied per-token
        rather than per-sequence for NER).
        """
        num_labels = logits.size(-1)
        flat_logits = logits.reshape(-1, num_labels)
        flat_mask = attention_mask.reshape(-1).bool()
        return flat_logits[flat_mask]
