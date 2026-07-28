import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.dataset_utils import (
    normalize_unicode,
    clean_text,
    is_empty_or_too_short,
    deduplicate,
    SplitRatios,
    train_val_test_split,
    write_jsonl,
    read_jsonl,
    write_splits_jsonl,
    clean_pipeline,
)
from datasets.language_filter import (
    filter_by_declared_language,
    resource_category,
    make_stub_detector,
    LanguageFilterConfig,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n\n  world  ") == "hello world"


def test_clean_text_strips_control_chars_but_keeps_content():
    text = "hello\x00\x01world"
    cleaned = clean_text(text)
    assert "\x00" not in cleaned and "\x01" not in cleaned
    assert "hello" in cleaned and "world" in cleaned


def test_normalize_unicode_nfkc():
    # full-width vs half-width digit should normalize to same form
    fullwidth = "１２３"
    assert normalize_unicode(fullwidth, form="NFKC") == "123"


def test_is_empty_or_too_short():
    assert is_empty_or_too_short("")
    assert is_empty_or_too_short(" a ")
    assert not is_empty_or_too_short("hello")


def test_deduplicate_removes_exact_case_insensitive_dupes():
    records = [
        {"text": "Hello World", "label": "pos"},
        {"text": "hello world", "label": "pos"},  # dup (case-insensitive)
        {"text": "  Hello World  ", "label": "pos"},  # dup (whitespace)
        {"text": "Different text", "label": "neg"},
    ]
    deduped = deduplicate(records)
    assert len(deduped) == 2


def test_split_ratios_must_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        SplitRatios(train=0.5, validation=0.3, test=0.3)


def test_train_val_test_split_proportions_approximately_correct():
    records = [{"text": f"item{i}"} for i in range(1000)]
    train, val, test = train_val_test_split(records, SplitRatios(0.8, 0.1, 0.1), seed=0)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    # no overlap
    all_texts = [r["text"] for r in train + val + test]
    assert len(set(all_texts)) == 1000


def test_train_val_test_split_deterministic_with_seed():
    records = [{"text": f"item{i}"} for i in range(200)]
    t1, v1, te1 = train_val_test_split(records, seed=7)
    t2, v2, te2 = train_val_test_split(records, seed=7)
    assert [r["text"] for r in t1] == [r["text"] for r in t2]


def test_train_val_test_split_stratified_preserves_per_group_ratio():
    # 900 "en" records, 100 "sw" records -- stratify by language so both
    # get proportional representation in every split (not all "sw" ending
    # up in test, etc.)
    records = [{"text": f"en_{i}", "language": "en"} for i in range(900)]
    records += [{"text": f"sw_{i}", "language": "sw"} for i in range(100)]
    train, val, test = train_val_test_split(
        records, SplitRatios(0.8, 0.1, 0.1), seed=1, stratify_field="language"
    )
    for split in (train, val, test):
        langs = [r["language"] for r in split]
        assert "en" in langs and "sw" in langs  # both present in every split


def test_jsonl_write_read_roundtrip():
    tmpdir = tempfile.mkdtemp()
    try:
        records = [{"text": "héllo wörld", "label": "pos", "n": 1}, {"text": "日本語", "label": "neg", "n": 2}]
        path = os.path.join(tmpdir, "out.jsonl")
        n_written = write_jsonl(records, path)
        assert n_written == 2
        loaded = list(read_jsonl(path))
        assert loaded == records
    finally:
        shutil.rmtree(tmpdir)


def test_write_splits_jsonl_creates_three_files():
    tmpdir = tempfile.mkdtemp()
    try:
        train = [{"text": "a"}]
        val = [{"text": "b"}]
        test = [{"text": "c"}]
        counts = write_splits_jsonl(train, val, test, tmpdir)
        assert counts == {"train": 1, "validation": 1, "test": 1}
        assert os.path.exists(os.path.join(tmpdir, "train.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "validation.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "test.jsonl"))
    finally:
        shutil.rmtree(tmpdir)


def test_clean_pipeline_removes_short_and_duplicate_records():
    records = [
        {"text": "This is a valid review"},
        {"text": "this is a valid review"},  # dup
        {"text": "x"},  # too short
        {"text": "Another valid one here"},
    ]
    cleaned = clean_pipeline(records, min_chars=3)
    assert len(cleaned) == 2


def test_resource_category_matches_evaluation_module():
    assert resource_category(2e9) == "HR"
    assert resource_category(5e8) == "MR"
    assert resource_category(5e7) == "LR"


def test_language_filter_keeps_matching_high_confidence():
    detector = make_stub_detector(fixed_prediction="en", fixed_confidence=0.95)
    records = [{"text": "hello", "language": "en"}, {"text": "hola", "language": "es"}]
    kept, rejected = filter_by_declared_language(records, detector)
    assert len(kept) == 1  # only the "en"-declared one matches stub's fixed prediction
    assert len(rejected) == 1
    assert "predicted=en" in rejected[0]["_reject_reason"]


def test_language_filter_rejects_low_confidence():
    detector = make_stub_detector(fixed_prediction="en", fixed_confidence=0.5)
    records = [{"text": "hello", "language": "en"}]
    kept, rejected = filter_by_declared_language(
        records, detector, config=LanguageFilterConfig(min_confidence=0.7)
    )
    assert len(kept) == 0
    assert len(rejected) == 1


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
