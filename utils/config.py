"""Safe YAML configuration with relative base inheritance and deep overrides."""

from copy import deepcopy
from pathlib import Path

import yaml


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"Cyclic configuration inheritance: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    bases = config.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ValueError("_base_ must be a path or list of paths")
    merged = {}
    for base in bases:
        merged = merge_config(merged, load_config(path.parent / base, (*_stack, path)))
    return merge_config(merged, config)
