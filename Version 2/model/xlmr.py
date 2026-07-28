"""
XLM-R factory functions. The manuscript's main results table (Table 5)
uses mT5-base exclusively (Sec 5.2), but XLM-R is cited (Kumar and
Albuquerque, 2021) as a comparable multilingual backbone and
configs/xlmr.yaml is provided in the repository structure for anyone
wanting to check whether ADAPT-BTS's gains are backbone-specific or
general -- an ablation the manuscript itself does not report.
"""

from __future__ import annotations

from model.encoder import MultilingualEncoder, EncoderConfig
from model.classifier import SentimentModel
from model.heads import NERModel


def build_xlmr_sentiment_model(max_seq_length: int = 128, num_labels: int = 3, dropout: float = 0.1,
                                 model_name: str = "xlm-roberta-base", freeze_embeddings: bool = False):
    encoder = MultilingualEncoder(EncoderConfig(
        model_name_or_path=model_name,
        max_seq_length=max_seq_length,
        freeze_embeddings=freeze_embeddings,
        pooling="cls",  # XLM-R (RoBERTa-family) conventionally pools via [CLS]
    ))
    return SentimentModel(encoder, num_labels=num_labels, dropout=dropout)


def build_xlmr_ner_model(max_seq_length: int = 128, num_labels: int = 7, dropout: float = 0.1,
                           model_name: str = "xlm-roberta-base", freeze_embeddings: bool = False):
    encoder = MultilingualEncoder(EncoderConfig(
        model_name_or_path=model_name,
        max_seq_length=max_seq_length,
        freeze_embeddings=freeze_embeddings,
        pooling="cls",
    ))
    return NERModel(encoder, num_labels=num_labels, dropout=dropout)
