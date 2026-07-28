"""
Real Translator implementation for the LR-tier PivotTranslationTransform
(fairness/counterfactual_generation.py, Sec 4.1: "pivot-based
translations with subsequent back-translation").

Wraps Hugging Face's translation pipeline, defaulting to the Helsinki-NLP
OPUS-MT model family (`Helsinki-NLP/opus-mt-{src}-{tgt}`), which covers
a wide range of language pairs with small, fast models well-suited to
being called twice per counterfactual sample (source->pivot,
pivot->source) as Sec 4.1's pipeline requires. For language pairs OPUS-MT
doesn't cover, swap `model_name_template` for a multilingual model like
`facebook/nllb-200-distilled-600M` (set via the `nllb_mode` flag, which
changes the call signature to NLLB's `src_lang`/`tgt_lang` codes instead
of separate per-pair models).

Requires torch + transformers + (for the first call per language pair)
network access to download the model weights. Not executable in this
sandbox -- syntax-checked only, same as everything else in model/ and
optimization/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    from transformers import pipeline
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "fairness/translation_backends.py requires transformers "
        "(pip install transformers) and a torch backend."
    ) from e


@dataclass
class HFTranslator:
    """Implements fairness.counterfactual_generation.Translator's
    protocol: `__call__(text, src_lang, tgt_lang) -> str`."""

    device: str = "cpu"
    model_name_template: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    max_length: int = 256
    _pipelines: dict = field(default_factory=dict, repr=False)

    def _get_pipeline(self, src_lang: str, tgt_lang: str):
        key = (src_lang, tgt_lang)
        if key not in self._pipelines:
            model_name = self.model_name_template.format(src=src_lang, tgt=tgt_lang)
            self._pipelines[key] = pipeline(
                "translation", model=model_name,
                device=0 if self.device == "cuda" else -1,
            )
        return self._pipelines[key]

    def __call__(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == tgt_lang:
            return text
        translator = self._get_pipeline(src_lang, tgt_lang)
        result = translator(text, max_length=self.max_length)
        return result[0]["translation_text"]


@dataclass
class NLLBTranslator:
    """
    Alternative backed by a single multilingual NLLB model instead of
    per-language-pair OPUS-MT models -- covers ~200 languages from one
    loaded model, at the cost of a larger model and needing NLLB's
    specific language codes (e.g. "eng_Latn", "swh_Latn") rather than
    plain ISO 639-1 codes. Callers are responsible for mapping their
    ISO codes to NLLB codes (e.g. via a small lookup table) before
    calling this -- not done here since the mapping is large and easy
    to get wrong silently; better to make it an explicit caller
    responsibility than guess.
    """
    model_name: str = "facebook/nllb-200-distilled-600M"
    device: str = "cpu"
    max_length: int = 256
    _pipeline: Optional[object] = field(default=None, repr=False)

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = pipeline(
                "translation", model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
        return self._pipeline

    def __call__(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """`src_lang`/`tgt_lang` must already be NLLB-format codes
        (e.g. "eng_Latn"), not plain ISO 639-1 -- see class docstring."""
        if src_lang == tgt_lang:
            return text
        translator = self._get_pipeline()
        result = translator(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=self.max_length)
        return result[0]["translation_text"]
