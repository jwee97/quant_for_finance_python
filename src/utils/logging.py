"""Structured logging for research runs.

Every stage script logs to stdout and to ``reports/logs/<stage>.log`` so that
a result can be traced back to the exact sequence of steps that produced it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED: set[str] = set()
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and, optionally, to a file."""
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger


def stage_logger(stage: str, root: str | Path | None = None) -> logging.Logger:
    """Logger for a numbered pipeline stage, with its own log file."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return get_logger(f"stage.{stage}", base / "reports" / "logs" / f"{stage}.log")


def banner(logger: logging.Logger, text: str) -> None:
    logger.info("=" * 78)
    logger.info(text)
    logger.info("=" * 78)
