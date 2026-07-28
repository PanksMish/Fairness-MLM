import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fairness.embedding_backends import FastTextEmbeddingSpace


def _write_synthetic_vec_file(path, entries: dict[str, list[float]]):
    dim = len(next(iter(entries.values())))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(entries)} {dim}\n")
        for word, vec in entries.items():
            f.write(word + " " + " ".join(str(v) for v in vec) + "\n")


def test_from_vec_file_loads_correct_vectors():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.vec")
        _write_synthetic_vec_file(path, {
            "he": [1.0, 0.0, 0.0],
            "she": [0.0, 1.0, 0.0],
            "man": [0.9, 0.1, 0.0],
        })
        space = FastTextEmbeddingSpace.from_vec_file(path)
        assert len(space) == 3
        vec = space.get_vector("he")
        assert vec == [1.0, 0.0, 0.0]
    finally:
        shutil.rmtree(tmpdir)


def test_get_vector_returns_none_for_unknown_word():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.vec")
        _write_synthetic_vec_file(path, {"he": [1.0, 0.0]})
        space = FastTextEmbeddingSpace.from_vec_file(path)
        assert space.get_vector("nonexistent") is None
    finally:
        shutil.rmtree(tmpdir)


def test_nearest_neighbor_finds_closest_vector():
    space = FastTextEmbeddingSpace({
        "he": np.array([1.0, 0.0]),
        "she": np.array([0.99, 0.14]),   # very close to "he"
        "unrelated": np.array([-1.0, 0.0]),  # opposite direction
    })
    neighbor = space.nearest_neighbor([1.0, 0.0], exclude={"he"})
    assert neighbor == "she"  # closest among non-excluded


def test_nearest_neighbor_respects_exclude_set():
    space = FastTextEmbeddingSpace({
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.9, 0.1]),
        "c": np.array([0.5, 0.5]),
    })
    neighbor = space.nearest_neighbor([1.0, 0.0], exclude={"a", "b"})
    assert neighbor == "c"


def test_nearest_neighbor_empty_space_returns_none():
    space = FastTextEmbeddingSpace({})
    assert space.nearest_neighbor([1.0, 0.0]) is None


def test_nearest_neighbor_zero_query_vector_returns_none():
    space = FastTextEmbeddingSpace({"a": np.array([1.0, 0.0])})
    assert space.nearest_neighbor([0.0, 0.0]) is None


def test_from_vec_file_respects_max_words():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.vec")
        _write_synthetic_vec_file(path, {f"word{i}": [float(i), 0.0] for i in range(10)})
        space = FastTextEmbeddingSpace.from_vec_file(path, max_words=3)
        assert len(space) == 3
    finally:
        shutil.rmtree(tmpdir)


def test_from_vec_file_skips_malformed_lines():
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "test.vec")
        with open(path, "w") as f:
            f.write("2 3\n")
            f.write("good 1.0 2.0 3.0\n")
            f.write("bad 1.0 2.0\n")  # wrong dimensionality -- should be skipped
        space = FastTextEmbeddingSpace.from_vec_file(path)
        assert len(space) == 1
        assert space.get_vector("good") is not None
        assert space.get_vector("bad") is None
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
