import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.config_utils import merge_configs, merge_all, apply_cli_overrides, load_and_merge, load_yaml


def test_merge_configs_override_wins_on_scalar_conflict():
    base = {"a": 1, "b": 2}
    override = {"b": 99}
    result = merge_configs(base, override)
    assert result == {"a": 1, "b": 99}


def test_merge_configs_deep_merges_nested_dicts():
    base = {"trainer": {"lr": 5e-5, "batch_size": 128}}
    override = {"trainer": {"lr": 1e-4}}
    result = merge_configs(base, override)
    assert result == {"trainer": {"lr": 1e-4, "batch_size": 128}}


def test_merge_configs_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    merge_configs(base, override)
    assert base == {"a": {"x": 1}}  # unchanged
    assert override == {"a": {"y": 2}}  # unchanged


def test_merge_configs_replaces_lists_not_merges():
    base = {"languages": ["en", "de"]}
    override = {"languages": ["fr"]}
    result = merge_configs(base, override)
    assert result == {"languages": ["fr"]}


def test_merge_all_applies_left_to_right():
    c1 = {"a": 1, "b": 1}
    c2 = {"b": 2, "c": 2}
    c3 = {"c": 3}
    result = merge_all(c1, c2, c3)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_apply_cli_overrides_sets_nested_value():
    config = {"trainer": {"learning_rate": 5e-5}}
    result = apply_cli_overrides(config, ["trainer.learning_rate=0.0001"])
    assert result["trainer"]["learning_rate"] == 0.0001


def test_apply_cli_overrides_infers_int():
    config = {"trainer": {"num_epochs": 3}}
    result = apply_cli_overrides(config, ["trainer.num_epochs=10"])
    assert result["trainer"]["num_epochs"] == 10
    assert isinstance(result["trainer"]["num_epochs"], int)


def test_apply_cli_overrides_infers_bool():
    config = {"trainer": {"use_amp": True}}
    result = apply_cli_overrides(config, ["trainer.use_amp=false"])
    assert result["trainer"]["use_amp"] is False


def test_apply_cli_overrides_infers_string_when_not_numeric_or_bool():
    config = {"model": {"name_or_path": "google/mt5-base"}}
    result = apply_cli_overrides(config, ["model.name_or_path=xlm-roberta-base"])
    assert result["model"]["name_or_path"] == "xlm-roberta-base"


def test_apply_cli_overrides_creates_new_nested_path():
    config = {}
    result = apply_cli_overrides(config, ["a.b.c=42"])
    assert result == {"a": {"b": {"c": 42}}}


def test_apply_cli_overrides_rejects_malformed_entry():
    import pytest
    with pytest.raises(ValueError):
        apply_cli_overrides({}, ["no_equals_sign"])


def test_load_and_merge_real_yaml_files():
    tmpdir = tempfile.mkdtemp()
    try:
        model_path = os.path.join(tmpdir, "model.yaml")
        task_path = os.path.join(tmpdir, "task.yaml")
        with open(model_path, "w") as f:
            f.write("model:\n  name_or_path: google/mt5-base\n  max_seq_length: 128\n")
        with open(task_path, "w") as f:
            f.write("task: sentiment\ntrainer:\n  batch_size: 128\n")

        merged = load_and_merge(model_path, task_path)
        assert merged["model"]["name_or_path"] == "google/mt5-base"
        assert merged["task"] == "sentiment"
        assert merged["trainer"]["batch_size"] == 128
    finally:
        shutil.rmtree(tmpdir)


def test_actual_repo_configs_load_and_merge_correctly():
    """Sanity-checks the real configs/mt5.yaml + configs/sentiment.yaml
    shipped in this repo actually parse and merge without key collisions
    silently clobbering something unexpected."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    mt5_path = os.path.join(repo_root, "configs", "mt5.yaml")
    sentiment_path = os.path.join(repo_root, "configs", "sentiment.yaml")
    merged = load_and_merge(mt5_path, sentiment_path)

    assert merged["model"]["name_or_path"] == "google/mt5-base"
    assert merged["task"] == "sentiment"
    assert merged["trainer"]["batch_size"] == 128
    assert merged["trainer"]["tau"] == 0.40
    assert "data" in merged and "train_path" in merged["data"]


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
