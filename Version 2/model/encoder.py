"""
Shared multilingual encoder, implementing the parametric predictive model
f_theta : X -> Delta(Y) described in Sec 3.2:

    "The predictive model is defined as a parametric function:
     f_theta: X -> Delta(Y), P_theta(y|x)"

This wraps a Hugging Face pretrained encoder (mT5's encoder stack, or
XLM-R directly) and returns pooled sequence representations h(x), which
both the classification head (model/classifier.py) and the NER tagging
head (model/heads.py) consume, and which the counterfactual semantic
validation constraint (Eq. 3, CosSim(h(x), h(x^(b))) >= 0.85) also needs.

Requires torch + transformers. Not runnable in the sandbox this file was
authored in (no GPU, no torch/transformers installed, no network to pull
pretrained weights) -- see README.md "Status" table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "model/encoder.py requires PyTorch. See requirements.txt "
        "(torch>=2.1, transformers>=4.38)."
    ) from e

from transformers import AutoModel, AutoTokenizer, AutoConfig


@dataclass
class EncoderConfig:
    model_name_or_path: str = "google/mt5-base"   # or "xlm-roberta-base"
    max_seq_length: int = 128
    freeze_embeddings: bool = False
    pooling: str = "mean"  # "mean" | "cls" | "last_token"


class MultilingualEncoder(nn.Module):
    """
    Encapsulates:
      - mT5: uses only the encoder stack (T5EncoderModel-equivalent via
        AutoModel picking up the encoder from the full seq2seq model,
        since ADAPT-BTS uses mT5 as a representation extractor for
        classification/tagging, not for generation -- Sec 5.2: "All
        models are trained using the mT5-base architecture" for
        sentiment/NER, not text generation).
      - XLM-R: a standard encoder-only model, used directly.

    Both paths expose the same interface: forward(input_ids,
    attention_mask) -> (batch, seq_len, hidden_dim) token representations,
    plus a pooled (batch, hidden_dim) sequence representation h(x).
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self._is_t5_family = "t5" in config.model_name_or_path.lower()

        hf_config = AutoConfig.from_pretrained(config.model_name_or_path)
        if self._is_t5_family:
            from transformers import T5EncoderModel
            self.backbone = T5EncoderModel.from_pretrained(config.model_name_or_path)
        else:
            self.backbone = AutoModel.from_pretrained(config.model_name_or_path)

        self.hidden_size = hf_config.hidden_size if hasattr(hf_config, "hidden_size") else hf_config.d_model

        if config.freeze_embeddings:
            for p in self.backbone.get_input_embeddings().parameters():
                p.requires_grad = False

    def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor"):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_reprs = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled = self._pool(token_reprs, attention_mask)
        return token_reprs, pooled

    def _pool(self, token_reprs: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        """Produces h(x), the sequence representation used in Eq. (3)'s
        semantic preservation constraint and as the classification head's
        input."""
        if self.config.pooling == "cls":
            return token_reprs[:, 0, :]
        elif self.config.pooling == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(token_reprs.size(0), device=token_reprs.device)
            return token_reprs[batch_idx, seq_lengths, :]
        else:  # mean pooling over non-padded tokens (default; robust across languages/lengths)
            mask = attention_mask.unsqueeze(-1).to(token_reprs.dtype)
            summed = (token_reprs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts


def load_tokenizer(model_name_or_path: str):
    """Thin wrapper so callers don't need to remember `use_fast=True`
    (required for the word_ids()-based NER label alignment in
    datasets/tokenizer.py)."""
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
