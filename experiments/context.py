"""Shared context for the numbered stage scripts.

Every stage takes the same inputs (config, cleaned market data, registry,
figure/table directories) and writes into the same places, so the plumbing
lives here instead of being repeated eleven times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loader import MarketData
from src.utils.config import Config, load_config
from src.utils.experiments import ExperimentRegistry
from src.utils.logging import stage_logger


@dataclass
class StageContext:
    config: Config
    registry: ExperimentRegistry
    figures: Path
    tables: Path
    processed: Path
    stage: str

    @property
    def tickers(self) -> list[str]:
        return self.config.tickers

    def figure(self, name: str) -> Path:
        return self.figures / name

    def save_table(self, frame: pd.DataFrame, name: str, index: bool = True, float_format: str = "%.6f") -> Path:
        path = self.tables / name
        frame.to_csv(path, index=index, float_format=float_format)
        return path

    def load_table(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.tables / name, **kwargs)

    def market_data(self, rebuild: bool = False) -> MarketData:
        """Cleaned dataset. Reads ``data/processed`` unless asked to rebuild."""
        cfg = self.config
        prices = self.processed / "prices_adjusted.csv"
        if prices.exists() and not rebuild:
            return MarketData.from_processed(
                self.processed, cfg.path("metadata"), cfg.asset_class_map, cfg.group_map
            )
        market, _, cleaned = MarketData.build(
            cfg.path("raw"), cfg.tickers, cfg.data, cfg.path("metadata"),
            cfg.asset_class_map, cfg.group_map,
        )
        cleaned.write(self.processed)
        return market


def build_context(stage: str, rebuild_config: dict | None = None, generation: int = 1):
    """Create the context and a stage logger.

    ``generation=2`` stamps the registry with the fingerprint of the core plus
    Generation 2 configuration, ``generation=3`` and ``generation=4`` with the
    fingerprints that add the Generation 3 and 4 files: a result's identity is the hash of exactly the
    configuration it depends on. Generation 1 stages keep the core fingerprint
    their report cites.
    """
    config = load_config(overrides=rebuild_config) if rebuild_config else load_config()
    from src.data.download import data_version

    registry = ExperimentRegistry(
        data_version=data_version(config.path("metadata")),
        config_fingerprint=config.fingerprint({1: "core", 2: "all", 3: "gen3", 4: "gen4"}.get(generation, "gen5")),
    )
    if generation >= 2:
        replaced = registry.drop_stage(stage)
        if replaced:
            stage_logger(stage, config.root).info(
                "re-run: replacing %d earlier registry entries for %s", replaced, stage)
    context = StageContext(
        config=config,
        registry=registry,
        figures=config.reports_dir("figures"),
        tables=config.reports_dir("tables"),
        processed=config.path("processed"),
        stage=stage,
    )
    return context, stage_logger(stage, config.root)
