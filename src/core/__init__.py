"""Core utilities shared across all layers."""

import json
import logging
import os
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

# Standard LogRecord attributes we never want to surface as extra fields
_STDLIB_LOG_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _STDLIB_LOG_KEYS and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-structured logger writing to stderr. Idempotent."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Load and return the project config as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)
