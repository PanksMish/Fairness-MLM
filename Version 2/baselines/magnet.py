"""
MAGNET baseline (Table 3): "Adaptive tokenization for multilingual
fairness," Ahia, Kumar, Gonen, Hofmann, Limisiewicz, Tsvetkov & Smith
(2024), "MAGNET: Improving the Multilingual Fairness of Language Models
with Adaptive Gradient-Based Tokenization," NeurIPS 2024.

*** THIS IS NOT A FAITHFUL REPRODUCTION OF MAGNET. ***

I checked the actual MAGNET paper (arXiv:2407.08818) via web search
before writing this, specifically to avoid guessing at a mechanism I
wasn't sure of. Real MAGNET:

  - operates on BYTE-LEVEL input (not subword tokens)
  - learns per-language-script "boundary predictor" sub-modules INSIDE
    the model that dynamically predict segmentation boundaries between
    bytes, trained via stochastic reparameterization jointly with the
    language-modeling objective
  - fundamentally changes the model's input representation and
    architecture, not just its training loss

This repository's models (model/mt5.py, model/xlmr.py) use pretrained
mT5/XLM-R with their standard SentencePiece/BPE SUBWORD tokenizers
(datasets/tokenizer.py). Retrofitting real MAGNET onto this pipeline
would mean replacing the tokenizer and adding trainable boundary-
predictor modules to the encoder -- a substantial architecture change,
not a baseline that can be dropped into the existing training loop.

What this file provides instead is a SAME-SPIRIT, clearly-labeled proxy:
inverse-token-frequency loss reweighting (baselines/losses_core.py's
`language_inverse_frequency_weights`), which shares MAGNET's high-level
GOAL (give languages that are disadvantaged by tokenization more
effective training signal) without touching tokenization at all. Do NOT
report results from this as "MAGNET" in any comparison intended to be
read against the manuscript's Table 5 MAGNET row -- report it as
"MAGNET-proxy (loss-reweighting)" or similar, and note the limitation.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("baselines/magnet.py requires PyTorch.") from e

from model.classifier import SentimentModel
from optimization.losses import sentiment_task_loss
from baselines.losses_core import language_inverse_frequency_weights


@dataclass
class MagnetProxyConfig:
    language_token_counts: dict[str, float]  # T_l per language, Sec 3.1


def magnet_proxy_loss(
    model: SentimentModel,
    input_ids: "torch.Tensor", attention_mask: "torch.Tensor", labels: "torch.Tensor",
    languages: list[str],
    config: MagnetProxyConfig,
) -> dict[str, "torch.Tensor"]:
    """Per-sample loss reweighted by inverse language token-frequency."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask)

    weights_by_lang = language_inverse_frequency_weights(config.language_token_counts)
    sample_weights = torch.tensor(
        [weights_by_lang.get(lang, 1.0) for lang in languages],
        dtype=torch.float32, device=logits.device,
    )

    per_sample_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted_loss = (per_sample_loss * sample_weights).mean()

    return {"total_loss": weighted_loss, "task_loss": per_sample_loss.mean(), "logits": logits}
