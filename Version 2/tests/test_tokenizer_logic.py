import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.tokenizer import align_labels_with_subwords, dynamic_pad_sequences


def test_align_labels_simple_one_subword_per_word():
    # Every word maps to exactly one subword: [CLS] w0 w1 w2 [SEP]
    word_ids = [None, 0, 1, 2, None]
    word_labels = ["B-PER", "O", "B-LOC"]
    aligned = align_labels_with_subwords(word_ids, word_labels)
    assert aligned == [-100, "B-PER", "O", "B-LOC", -100]


def test_align_labels_multi_subword_default_ignores_continuation():
    # word 0 splits into 2 subwords, word 1 is single subword
    word_ids = [None, 0, 0, 1, None]
    word_labels = ["B-ORG", "O"]
    aligned = align_labels_with_subwords(word_ids, word_labels, label_all_subword_tokens=False)
    assert aligned == [-100, "B-ORG", -100, "O", -100]


def test_align_labels_multi_subword_label_all_converts_b_to_i():
    word_ids = [None, 0, 0, 0, None]
    word_labels = ["B-LOC"]
    aligned = align_labels_with_subwords(word_ids, word_labels, label_all_subword_tokens=True)
    assert aligned == [-100, "B-LOC", "I-LOC", "I-LOC", -100]


def test_align_labels_custom_ignore_value():
    word_ids = [None, 0, None]
    word_labels = ["O"]
    aligned = align_labels_with_subwords(word_ids, word_labels, ignore_label=-1)
    assert aligned == [-1, "O", -1]


def test_dynamic_pad_pads_to_longest_in_batch():
    sequences = [[1, 2, 3], [4, 5], [6]]
    padded, masks = dynamic_pad_sequences(sequences, pad_value=0)
    assert padded == [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
    assert masks == [[1, 1, 1], [1, 1, 0], [1, 0, 0]]


def test_dynamic_pad_respects_max_length_truncation():
    sequences = [[1, 2, 3, 4, 5]]
    padded, masks = dynamic_pad_sequences(sequences, pad_value=-1, max_length=3)
    assert padded == [[1, 2, 3]]
    assert masks == [[1, 1, 1]]


def test_dynamic_pad_respects_max_length_padding_when_shorter():
    sequences = [[1, 2]]
    padded, masks = dynamic_pad_sequences(sequences, pad_value=0, max_length=5)
    assert padded == [[1, 2, 0, 0, 0]]
    assert masks == [[1, 1, 0, 0, 0]]


def test_dynamic_pad_empty_batch_of_equal_length():
    sequences = [[7, 8], [9, 10]]
    padded, masks = dynamic_pad_sequences(sequences)
    assert padded == [[7, 8], [9, 10]]
    assert masks == [[1, 1], [1, 1]]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
