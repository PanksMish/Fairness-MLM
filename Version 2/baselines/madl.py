"""
MADL baseline (Table 3): "Multi-attribute adversarial debiasing," Park &
Cho (2025), "Improving fairness of abusive language detection with
multi-attribute learning," Expert Systems with Applications.

I was not able to fetch this paper's full text/exact equations in this
session. Adversarial debiasing via a gradient-reversal layer (GRL) is a
well-established, standard technique (Ganin & Lempitsky, 2015,
"Unsupervised Domain Adaptation by Backpropagation" -- the same
mechanism cited generically by the ADAPT-BTS manuscript's related-work
section for "adversarial debiasing," e.g. Sweeney & Najafian 2020),
extended here to multiple simultaneous attribute discriminators for the
"multi-attribute" part of MADL's name. This is a faithful implementation
of the general adversarial-debiasing mechanism the paper's title
describes, not a verified reproduction of Park & Cho's specific
architecture/hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("baselines/madl.py requires PyTorch.") from e

from model.classifier import SentimentModel
from optimization.losses import sentiment_task_loss


class _GradientReversalFunction(torch.autograd.Function):
    """Standard GRL (Ganin & Lempitsky 2015): identity in the forward
    pass, negates (and scales by `lambd`) the gradient in the backward
    pass. This is what lets a single optimizer step simultaneously (a)
    train the attribute discriminator to predict the attribute from the
    representation, while (b) training the ENCODER to make the
    representation less predictive of that attribute -- the two
    objectives are adversarial without needing a separate min-max loop."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: "torch.Tensor", lambd: float = 1.0) -> "torch.Tensor":
    return _GradientReversalFunction.apply(x, lambd)


class AttributeDiscriminator(nn.Module):
    """A single-attribute linear/MLP classifier sitting behind the GRL.
    "Multi-attribute" (MADL) means one of these per demographic
    attribute dimension (e.g. one for gender, one for dialect), each
    contributing its own adversarial loss term."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


@dataclass
class MADLConfig:
    grl_lambda: float = 1.0
    alpha_adversarial: float = 1.0


class MADLModel(nn.Module):
    """
    Wraps a SentimentModel with one or more AttributeDiscriminators
    attached via GRL to the pooled encoder representation. `num_classes_per_attribute`
    maps attribute name -> number of classes for that attribute (e.g.
    {"gender": 2}) so multiple attribute dimensions can be adversarially
    debiased simultaneously, per MADL's "multi-attribute" framing.
    """

    def __init__(self, sentiment_model: SentimentModel, num_classes_per_attribute: dict[str, int],
                 config: MADLConfig | None = None):
        super().__init__()
        self.sentiment_model = sentiment_model
        self.config = config or MADLConfig()
        self.discriminators = nn.ModuleDict({
            attr: AttributeDiscriminator(sentiment_model.encoder.hidden_size, n_classes)
            for attr, n_classes in num_classes_per_attribute.items()
        })

    def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor",
                labels: "torch.Tensor", attribute_labels: dict[str, "torch.Tensor"]) -> dict:
        token_reprs, pooled = self.sentiment_model.encoder(input_ids, attention_mask)
        logits = self.sentiment_model.head(pooled)
        task_loss = sentiment_task_loss(logits, labels)

        adversarial_loss = torch.tensor(0.0, device=pooled.device)
        per_attribute_losses = {}
        for attr, discriminator in self.discriminators.items():
            if attr not in attribute_labels:
                continue
            reversed_pooled = gradient_reversal(pooled, self.config.grl_lambda)
            attr_logits = discriminator(reversed_pooled)
            attr_loss = F.cross_entropy(attr_logits, attribute_labels[attr])
            per_attribute_losses[attr] = attr_loss
            adversarial_loss = adversarial_loss + attr_loss

        total = task_loss + self.config.alpha_adversarial * adversarial_loss
        return {
            "total_loss": total, "task_loss": task_loss,
            "adversarial_loss": adversarial_loss, "per_attribute_losses": per_attribute_losses,
            "logits": logits,
        }
