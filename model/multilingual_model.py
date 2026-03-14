"""
multilingual_model.py
=====================
Backbone multilingual transformer model wrapper for ADAPT-BTS.

Wraps mT5-base or XLM-RoBERTa-base with a classification head:
    Pθ(y|x) = Softmax(W · h(x) + b)

where h(x) = Encoderθ(x) is the contextual representation (Equation 8-9).

Supports both:
  - Sentiment classification (3-class)
  - Named Entity Recognition (token-level)

The model's backbone is frozen or fine-tuned depending on the training
configuration. The classification head is always trained.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    MT5EncoderModel,
    XLMRobertaModel,
)

logger = logging.getLogger(__name__)


class MultilingualClassificationModel(nn.Module):
    """
    Multilingual transformer with a linear classification head.

    Architecture:
      Encoder (mT5-base or XLM-R) → CLS/mean pooling → Linear → logits

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name (e.g., "google/mt5-base", "xlm-roberta-base").
    num_labels : int
        Number of output classes (3 for sentiment, 7 for NER IOB2).
    task : str
        "sentiment" (sequence classification) or "ner" (token classification).
    dropout : float
        Dropout probability on the hidden representation.
    freeze_encoder : bool
        If True, freeze the encoder backbone and only train the head.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/mt5-base",
        num_labels: int = 3,
        task: str = "sentiment",
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.num_labels = num_labels
        self.task = task

        # Load pretrained config and encoder backbone
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = self._load_encoder(model_name_or_path)

        # Get hidden size from config
        hidden_size = self._get_hidden_size()

        # Classification head (Equation 9 in paper)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder frozen. Training classification head only.")

        # Store reference to tokenizer (set externally after loading)
        self.tokenizer = None

        logger.info(
            f"Initialized MultilingualClassificationModel: "
            f"backbone={model_name_or_path}, task={task}, num_labels={num_labels}, "
            f"hidden_size={hidden_size}"
        )

    def _load_encoder(self, model_name_or_path: str) -> nn.Module:
        """
        Load the encoder portion of the pretrained model.
        For mT5, loads only the encoder stack (not the full seq2seq model).
        """
        if "mt5" in model_name_or_path.lower():
            try:
                return MT5EncoderModel.from_pretrained(model_name_or_path)
            except Exception:
                # Fallback: load generic AutoModel
                logger.warning("MT5EncoderModel load failed, falling back to AutoModel.")
                return AutoModel.from_pretrained(model_name_or_path)
        else:
            return AutoModel.from_pretrained(model_name_or_path)

    def _get_hidden_size(self) -> int:
        """Determine hidden size from model config."""
        # Different models use different attribute names
        for attr in ["d_model", "hidden_size", "dim"]:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        return 768  # default fallback

    def get_encoder_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the encoder and return the pooled representation h(x).

        For sequence classification:  mean-pool over non-padding positions.
        For NER:                       return full token-level hidden states.

        Returns
        -------
        hidden : torch.Tensor
            [batch_size, hidden_size] for sequence tasks
            [batch_size, seq_len, hidden_size] for token tasks
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Handle different output formats (BaseModelOutput vs tuple)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state  # [B, L, H]
        else:
            hidden_states = outputs[0]  # [B, L, H]

        if self.task == "sentiment":
            # Mean pooling over non-padding tokens (Equation 8)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [B, H]
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)      # [B, 1]
            return sum_hidden / sum_mask  # [B, H]
        else:
            # NER: return full token-level representations
            return hidden_states  # [B, L, H]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder + classification head.

        Parameters
        ----------
        input_ids : [B, L]
        attention_mask : [B, L]
        labels : [B] for sentiment, [B, L] for NER (optional, for loss computation)

        Returns
        -------
        logits : [B, num_labels] for sentiment, [B, L, num_labels] for NER
        """
        hidden = self.get_encoder_output(input_ids, attention_mask)  # [B, H] or [B, L, H]
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)  # Equation 9
        return logits

    def get_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns Pθ(y|x) — the softmax probability distribution.

        Returns
        -------
        probs : [B, num_labels] for sentiment, [B, L, num_labels] for NER
        """
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)

    def get_frozen_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns encoder hidden states with gradient computation disabled.
        Used for linear probe (representation leakage evaluation).
        """
        with torch.no_grad():
            return self.get_encoder_output(input_ids, attention_mask)

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """
    Load the tokenizer associated with the backbone model.
    Uses SentencePiece tokenizer (as described in paper Section 3.5).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    logger.info(f"Loaded tokenizer: {model_name_or_path} (vocab size: {tokenizer.vocab_size})")
    return tokenizer


def build_model(
    backbone: str = "google/mt5-base",
    num_labels: int = 3,
    task: str = "sentiment",
    dropout: float = 0.1,
    device: str = "cpu",
) -> Tuple[MultilingualClassificationModel, AutoTokenizer]:
    """
    Convenience factory that creates both the model and its tokenizer.

    Returns
    -------
    model : MultilingualClassificationModel
    tokenizer : AutoTokenizer
    """
    tokenizer = load_tokenizer(backbone)
    model = MultilingualClassificationModel(
        model_name_or_path=backbone,
        num_labels=num_labels,
        task=task,
        dropout=dropout,
    )
    model.tokenizer = tokenizer
    model = model.to(device)

    param_counts = model.count_parameters()
    logger.info(
        f"Model parameters: total={param_counts['total']:,}, "
        f"trainable={param_counts['trainable']:,}"
    )
    return model, tokenizer
