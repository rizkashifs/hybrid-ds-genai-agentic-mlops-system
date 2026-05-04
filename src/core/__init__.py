"""Core utilities shared across all layers."""

import yaml
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Load and return the project config as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)
