"""Configuration loader from YAML files."""
import os
import yaml
from dataclasses import asdict

from model.model import ModelConfig


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    model_cfg = ModelConfig(**raw.get("model", {}))
    raw["model"] = asdict(model_cfg)
    return raw


def save_config(config: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
