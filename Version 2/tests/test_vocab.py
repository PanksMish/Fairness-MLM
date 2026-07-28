import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.vocab import LabelVocab, build_vocabs_from_records


def test_vocab_build_assigns_sorted_ids():
    vocab = LabelVocab.build(["fr", "en", "en", "de", "fr"])
    assert vocab.label_to_id == {"de": 0, "en": 1, "fr": 2}
    assert len(vocab) == 3


def test_vocab_build_excludes_none():
    vocab = LabelVocab.build(["m", None, "f", None])
    assert None not in vocab.label_to_id
    assert len(vocab) == 2


def test_vocab_encode_roundtrip():
    vocab = LabelVocab.build(["m", "f"])
    idx = vocab.encode("m")
    assert vocab.decode(idx) == "m"


def test_vocab_encode_missing_returns_missing_id():
    vocab = LabelVocab.build(["m", "f"])
    assert vocab.encode(None) == -1
    assert vocab.encode(None, missing_id=99) == 99


def test_vocab_encode_unknown_label_raises():
    import pytest
    vocab = LabelVocab.build(["m", "f"])
    with pytest.raises(KeyError):
        vocab.encode("unknown_attribute")


def test_vocab_encode_batch():
    vocab = LabelVocab.build(["en", "de", "fr"])
    ids = vocab.encode_batch(["de", "en", None, "fr"])
    assert ids == [vocab.encode("de"), vocab.encode("en"), -1, vocab.encode("fr")]


def test_vocab_serialization_roundtrip():
    vocab = LabelVocab.build(["m", "f"])
    d = vocab.as_dict()
    restored = LabelVocab.from_dict(d)
    assert restored.label_to_id == vocab.label_to_id
    assert restored.decode(vocab.encode("m")) == "m"


def test_build_vocabs_from_records():
    records = [
        {"language": "en", "attribute": "m"},
        {"language": "en", "attribute": "f"},
        {"language": "fr", "attribute": "m"},
        {"language": "de", "attribute": None},
    ]
    vocabs = build_vocabs_from_records(records)
    assert len(vocabs["language"]) == 3
    assert len(vocabs["attribute"]) == 2  # None excluded
    assert vocabs["language"].encode("de") in {0, 1, 2}


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
