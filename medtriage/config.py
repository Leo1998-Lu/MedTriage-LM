"""Load a YAML config file into the project's dataclasses."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Tuple

import yaml

from .data.preprocess import PreprocessConfig
from .models.medtriage_lm import MedTriageLMConfig
from .trainer import TrainConfig


def _filter(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that are fields of the dataclass ``cls``."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in valid}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_configs(
    raw: Dict[str, Any]
) -> Tuple[PreprocessConfig, MedTriageLMConfig, TrainConfig]:
    pre = PreprocessConfig(**_filter(PreprocessConfig, raw.get("preprocess", {})))
    model = MedTriageLMConfig(**_filter(MedTriageLMConfig, raw.get("model", {})))
    train = TrainConfig(**_filter(TrainConfig, raw.get("train", {})))
    return pre, model, train


def configs_from_file(
    path: str,
) -> Tuple[PreprocessConfig, MedTriageLMConfig, TrainConfig]:
    return make_configs(load_config(path))
