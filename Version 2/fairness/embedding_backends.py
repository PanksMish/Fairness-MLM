"""
Real EmbeddingSpace implementations for the MR-tier
EmbeddingAlignmentTransform (fairness/counterfactual_generation.py,
Sec 4.1: "cross-lingual embedding space alignment to perform demographic
attribute transfer").

`FastTextEmbeddingSpace` loads the standard FastText/word2vec plain-text
`.vec` format (first line "n_words dim", then one "word v1 v2 ... vd"
line per word) -- this is the format fastText's `.vec` releases and
aligned cross-lingual embeddings (e.g. MUSE/VecMap-aligned spaces) both
use, so this loader works for real pretrained files, not a bespoke
format. It's pure NumPy, has no torch dependency, and is genuinely
testable here by writing a small synthetic .vec file (real file I/O and
real cosine-similarity nearest-neighbor search, just with made-up
vectors instead of ones downloaded from fasttext.cc).

For real cross-lingual alignment specifically, download aligned vectors
(e.g. from https://fasttext.cc/docs/en/aligned-vectors.html, which
provides vectors for 44 languages already aligned into a single space)
rather than monolingual vectors per language -- this class is agnostic
to which .vec file you point it at, but the counterfactual-generation
use case needs an ALIGNED space for cross-lingual nearest-neighbor
lookups to be meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FastTextEmbeddingSpace:
    """Implements fairness.counterfactual_generation.EmbeddingSpace's
    protocol (get_vector, nearest_neighbor) by loading a FastText/
    word2vec-format .vec file into memory."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = vectors
        self._words = list(vectors.keys())
        if self._words:
            self._matrix = np.stack([vectors[w] for w in self._words])
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            self._normalized_matrix = self._matrix / np.clip(norms, 1e-12, None)
        else:
            self._matrix = np.empty((0, 0))
            self._normalized_matrix = self._matrix

    @classmethod
    def from_vec_file(cls, path: str | Path, max_words: Optional[int] = None) -> "FastTextEmbeddingSpace":
        """
        Loads a .vec file in the standard format:

            <n_words> <dim>
            word1 v1 v2 ... vd
            word2 v1 v2 ... vd
            ...

        Args:
            max_words: if set, only load the first N word vectors
                (useful for a quick smoke test against a huge file
                without loading gigabytes into memory).
        """
        path = Path(path)
        vectors: dict[str, np.ndarray] = {}
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            header = f.readline().split()
            declared_n, dim = int(header[0]), int(header[1])
            for i, line in enumerate(f):
                if max_words is not None and i >= max_words:
                    break
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) != dim:
                    continue  # skip malformed lines rather than crash on one bad row
                vectors[word] = vec
        return cls(vectors)

    def get_vector(self, word: str, language: str = "und") -> Optional[list[float]]:
        """`language` is accepted for protocol compatibility (an aligned
        multilingual space doesn't need per-language lookup tables --
        the word itself, in whatever script, is the key) but unused."""
        vec = self._vectors.get(word)
        return None if vec is None else vec.tolist()

    def nearest_neighbor(self, vector: list[float], language: str = "und",
                          exclude: Optional[set[str]] = None) -> Optional[str]:
        """Brute-force cosine-similarity nearest neighbor. Fine for
        vocabularies up to ~1M words on modern hardware; swap in an ANN
        index (faiss, annoy) if you need this at larger scale or higher
        query throughput than a linear scan gives you."""
        if len(self._words) == 0:
            return None
        exclude = exclude or set()
        query = np.asarray(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None
        query_normalized = query / query_norm

        sims = self._normalized_matrix @ query_normalized
        order = np.argsort(-sims)
        for idx in order:
            word = self._words[idx]
            if word not in exclude:
                return word
        return None

    def __len__(self) -> int:
        return len(self._words)
