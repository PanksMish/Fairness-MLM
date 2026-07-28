"""
mT5-base factory functions, matching Sec 5.2: "All models are trained
using the mT5-base architecture." Only the encoder stack is used (see
model/encoder.py's docstring) -- ADAPT-BTS is a classification/tagging
framework, not a generation framework, despite mT5 being seq2seq.
"""

from __future__ import annotations

from model.encoder import MultilingualEncoder, EncoderConfig
from model.classifier import SentimentModel
from model.heads import NERModel


def build_mt5_sentiment_model(max_seq_length: int = 128, num_labels: int = 3, dropout: float = 0.1,
                                freeze_embeddings: bool = False):
    encoder = MultilingualEncoder(EncoderConfig(
        model_name_or_path="google/mt5-base",
        max_seq_length=max_seq_length,
        freeze_embeddings=freeze_embeddings,
        pooling="mean",
    ))
    return SentimentModel(encoder, num_labels=num_labels, dropout=dropout)


def build_mt5_ner_model(max_seq_length: int = 128, num_labels: int = 7, dropout: float = 0.1,
                          freeze_embeddings: bool = False):
    encoder = MultilingualEncoder(EncoderConfig(
        model_name_or_path="google/mt5-base",
        max_seq_length=max_seq_length,
        freeze_embeddings=freeze_embeddings,
        pooling="mean",  # unused by NERModel (uses token_reprs directly), kept for interface symmetry
    ))
    return NERModel(encoder, num_labels=num_labels, dropout=dropout)
