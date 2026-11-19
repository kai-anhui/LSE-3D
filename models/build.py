"""Construct the model from an experiment mapping or YAML path."""

from collections.abc import Mapping
from pathlib import Path

from utils.config import load_config
from .config import ModelConfig
from .lse3d import LSE3D


def build_model(config):
    if isinstance(config, (str, Path)):
        config = load_config(config)
    if isinstance(config, ModelConfig):
        return LSE3D(config)
    if not isinstance(config, Mapping):
        raise TypeError("Expected experiment mapping, ModelConfig, or YAML path")
    return LSE3D(ModelConfig(**config.get("model", {})))
