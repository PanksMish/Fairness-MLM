"""
Downloads and parses the NRC Emotion Lexicon (EmoLex), Mohammad & Turney
2010/2012, which provides positive/negative word associations for
English plus machine-translated versions in 100+ languages (Google
Translate, last updated August 2022). This is a REAL, citable, widely
used resource (SemEval systems, sentiment/emotion research broadly) --
not something invented for this pipeline -- but the translations are
machine-translated, not human-verified per language, so labels derived
from it are weak/noisy by the lexicon's own documentation. See
datasets/weak_labeling.py for how this gets turned into (weak) sentiment
labels, and the loud warnings there about what that data is and isn't
suitable for.

LICENSE / TERMS OF USE (from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm):
  - Free for non-commercial research and educational use.
  - Must cite: Mohammad, S.M., Turney, P.D. (2013). "Crowdsourcing a
    Word-Emotion Association Lexicon." Computational Intelligence,
    29(3), 436-465.
  - "No Redistribution: Do not redistribute the data." This module
    therefore downloads fresh from the official URL each time and does
    NOT bundle/cache the lexicon file inside this repository -- respect
    this term yourself if you build anything on top of this that might
    otherwise redistribute the file.
  - Commercial use requires contacting the authors separately.

I have NOT verified the exact column layout of the multilingual
translation file by downloading it myself (no network in the sandbox
this was written in). The parser below is deliberately defensive: it
looks up columns BY NAME (case-insensitive substring match for
"positive"/"negative" and for the target language name) rather than by
a hardcoded column index, and raises a clear error listing the columns
it actually found if the expected ones aren't there. If the file's real
layout differs from what this expects, you'll get an informative
KeyError, not a silent wrong answer.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

OFFICIAL_DOWNLOAD_URL = "http://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
CITATION = (
    "Mohammad, S.M., Turney, P.D. (2013). Crowdsourcing a Word-Emotion "
    "Association Lexicon. Computational Intelligence, 29(3), 436-465."
)


def download_lexicon_zip(dest_dir: str | Path) -> Path:
    """
    Fetches the lexicon zip fresh from the official URL (does not cache
    it long-term or bundle it in this repo, per the "No Redistribution"
    term). Requires network access and `requests` (or falls back to
    urllib, both stdlib-adjacent, no extra heavy dependency needed for
    a single file download).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "NRC-Emotion-Lexicon.zip"

    import urllib.request
    logger.info("Downloading NRC Emotion Lexicon from %s (see module docstring for license terms)", OFFICIAL_DOWNLOAD_URL)
    urllib.request.urlretrieve(OFFICIAL_DOWNLOAD_URL, dest_path)
    return dest_path


def extract_lexicon(zip_path: str | Path, extract_dir: str | Path) -> Path:
    extract_dir = Path(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


@dataclass
class LanguageLexicon:
    """Positive/negative word sets for one language, ready for
    datasets/weak_labeling.py's lexicon_polarity_score."""
    language: str
    positive_words: set[str]
    negative_words: set[str]

    def __len__(self) -> int:
        return len(self.positive_words) + len(self.negative_words)


def parse_multilingual_lexicon_csv(path: str | Path, language_column_name: str) -> LanguageLexicon:
    """
    Parses the multilingual translation file (expected as a CSV/TSV with
    a header row containing a "Positive"/"Negative" association column
    pair and one translated-word column per language) into a
    LanguageLexicon for one target language.

    Args:
        path: path to the extracted multilingual lexicon file
        language_column_name: the language name as it appears in the
            file's header (e.g. "Spanish", "Swahili") -- see
            https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
            for the full list of covered language names.

    Raises:
        KeyError with the actual column names found, if the expected
        columns can't be located -- see module docstring for why this
        is deliberately defensive rather than assuming a fixed layout.
    """
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")  # sniff delimiter (csv/tsv both plausible)
    columns_lower = {c.lower(): c for c in df.columns}

    def _find_column(name_substring: str) -> str:
        matches = [orig for lower, orig in columns_lower.items() if name_substring.lower() in lower]
        if not matches:
            raise KeyError(
                f"No column matching '{name_substring}' found. "
                f"Available columns: {list(df.columns)}"
            )
        if len(matches) > 1:
            logger.warning("Multiple columns matched '%s': %s -- using the first.", name_substring, matches)
        return matches[0]

    positive_col = _find_column("positive")
    negative_col = _find_column("negative")
    word_col = _find_column(language_column_name)

    positive_words = set(
        df.loc[df[positive_col].astype(str).isin(["1", "1.0", "True", "true"]), word_col].dropna().astype(str)
    )
    negative_words = set(
        df.loc[df[negative_col].astype(str).isin(["1", "1.0", "True", "true"]), word_col].dropna().astype(str)
    )

    return LanguageLexicon(language=language_column_name, positive_words=positive_words, negative_words=negative_words)


def parse_english_wordlevel_txt(path: str | Path) -> LanguageLexicon:
    """
    Parses the standard English-only word-level format (10 rows per
    word: 8 emotions + positive + negative, tab-separated
    word/category/0-or-1), for the common case of just needing English
    positive/negative words without the multilingual translation file.
    """
    positive_words, negative_words = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, category, flag = parts
            if flag != "1":
                continue
            if category == "positive":
                positive_words.add(word)
            elif category == "negative":
                negative_words.add(word)
    return LanguageLexicon(language="en", positive_words=positive_words, negative_words=negative_words)
