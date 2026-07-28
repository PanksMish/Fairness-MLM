"""
Sentiment classification head: maps the encoder's pooled representation
h(x) to logits over the 3-class label space Y = {negative, neutral,
positive}, i.e. produces the logits that get softmax'd into P_theta(y|x)
(Sec 3.2).

Kept as a separate module (rather than fused into encoder.py) so the same
MultilingualEncoder backbone can be shared between this and the NER
tagging head (model/heads.py) -- both the sentiment and NER tasks reuse
one encoder per the manuscript's setup ("Two downstream tasks have been
selected", Sec 5.1), rather than training two independent models with no
shared structure. Whether weights are actually shared across tasks or two
separate encoder instances are trained is a training-script decision
(configs/*.yaml), not fixed here.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("model/classifier.py requires PyTorch.") from e

SENTIMENT_LABELS = ["negative", "neutral", "positive"]  # fixed 3-class schema, Sec 5.1


class SentimentClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: "torch.Tensor") -> "torch.Tensor":
        """Args: pooled (batch, hidden_size) from MultilingualEncoder.
        Returns: logits (batch, num_labels)."""
        return self.classifier(self.dropout(pooled))


class SentimentModel(nn.Module):
    """Full model = encoder + classification head, exposing a single
    forward(input_ids, attention_mask) -> logits interface, which is what
    fairness/bts_torch.py's `counterfactual_forward_pass` expects when
    called with `model(**batch)`."""

    def __init__(self, encoder, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = SentimentClassificationHead(encoder.hidden_size, num_labels, dropout)

    def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        _, pooled = self.encoder(input_ids, attention_mask)
        return self.head(pooled)
