"""
Config loading/merging for the CLI scripts. Deliberately simple (plain
dict deep-merge over YAML files, no Hydra/OmegaConf dependency) since the
config files here are flat and few enough not to need a config framework.

`merge_configs` is pure Python/dict logic and is unit-tested directly;
`load_and_merge` just adds the YAML-reading I/O around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merges `override` into `base` (override wins on conflicts),
    recursing into nested dicts but replacing (not merging) lists and
    scalars. Returns a NEW dict; does not mutate either input.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def merge_all(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merges any number of config dicts left-to-right (later configs
    override earlier ones)."""
    result: dict[str, Any] = {}
    for cfg in configs:
        result = merge_configs(result, cfg)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_and_merge(*paths: str | Path) -> dict[str, Any]:
    """Loads and merges multiple YAML config files, e.g.:

        load_and_merge("configs/mt5.yaml", "configs/sentiment.yaml")

    lets a model config and a task config compose without duplicating
    shared keys, matching the pattern scripts/train.py uses.
    """
    configs = [load_yaml(p) for p in paths]
    return merge_all(*configs)


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """
    Applies dotted-path CLI overrides like `trainer.learning_rate=1e-4`
    on top of a loaded config, with basic type inference (int/float/bool
    parsed from the string; everything else stays a string). Used by
    scripts/train.py's `--set key=value` flag for quick experimentation
    without editing YAML files.
    """
    result = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key.path=value")
        key_path, raw_value = item.split("=", 1)
        value = _infer_type(raw_value)
        _set_nested(result, key_path.split("."), value)
    return result


def _infer_type(raw: str):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_nested(d: dict, keys: list[str], value) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
