"""Configuration loading.

Engineering addition (not prescribed by the book). Every number that affects a
research result lives in ``config/*.yaml`` rather than in code, so that a
result can be reproduced from a commit hash plus a config hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = (
    "universe",
    "data",
    "strategies",
    "portfolio",
    "risk",
    "backtest",
)


def project_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "config"


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class Config:
    """Container for the six configuration namespaces.

    Access is dictionary-like with dotted paths::

        cfg.get("portfolio.covariance.method")
        cfg["data"]["start"]
    """

    universe: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    strategies: dict = field(default_factory=dict)
    portfolio: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    root: Path = field(default_factory=project_root)

    # -- construction -----------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path | None = None, overrides: dict | None = None) -> "Config":
        directory = Path(directory) if directory is not None else config_dir()
        payload: dict[str, Any] = {}
        for name in CONFIG_FILES:
            path = directory / f"{name}.yaml"
            payload[name] = load_yaml(path) if path.exists() else {}
        local = directory / "local.yaml"
        if local.exists():  # developer overrides, never committed
            _deep_update(payload, load_yaml(local))
        if overrides:
            _deep_update(payload, overrides)
        return cls(root=directory.parent, **payload)

    # -- access -----------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in CONFIG_FILES}

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.as_dict()
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    # -- derived helpers --------------------------------------------------
    @property
    def tickers(self) -> list[str]:
        return [a["ticker"] for a in self.universe.get("assets", [])]

    @property
    def asset_class_map(self) -> dict[str, str]:
        return {a["ticker"]: a["asset_class"] for a in self.universe.get("assets", [])}

    @property
    def group_map(self) -> dict[str, str]:
        return {a["ticker"]: a.get("group", a["asset_class"]) for a in self.universe.get("assets", [])}

    def path(self, key: str) -> Path:
        """Resolve one of the configured data paths to an absolute path."""
        rel = self.get(f"data.paths.{key}")
        if rel is None:
            raise KeyError(f"unknown data path '{key}'")
        out = self.root / rel
        out.mkdir(parents=True, exist_ok=True)
        return out

    def reports_dir(self, sub: str | None = None) -> Path:
        out = self.root / "reports" / (sub or "")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def fingerprint(self) -> str:
        """Stable hash of the full configuration, stamped into experiments."""
        blob = json.dumps(self.as_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(directory: str | Path | None = None, **overrides: Any) -> Config:
    return Config.load(directory, overrides or None)
