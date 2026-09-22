"""Run the whole research pipeline, in the order the specification fixes.

    python -m experiments.run_all                    # everything
    python -m experiments.run_all --from 6 --to 9    # a range of stages
    python -m experiments.run_all --only 1 --download --fresh

Each stage is independent and writes its own tables, figures and registry
entries, so a stage can be re-run on its own after a change. ``--fresh``
clears the derived artefacts first, which is what makes the "reproducible
from a commit hash plus a config fingerprint" claim testable rather than
aspirational.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import time
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import stage_logger

STAGES = [
    (1, "stage01_data", "Financial data infrastructure (Ch. 7)"),
    (2, "stage02_eda", "Exploratory analysis and PCA (Ch. 8)"),
    (3, "stage03_momentum", "Momentum alpha research (Ch. 22)"),
    (4, "stage04_mean_reversion", "Mean-reversion alpha research (Ch. 22)"),
    (5, "stage05_expected_returns", "Expected returns and IC (Ch. 20)"),
    (6, "stage06_backtest", "Backtesting the alpha models (Ch. 22)"),
    (7, "stage07_portfolio", "Portfolio construction (Ch. 19)"),
    (8, "stage08_covariance", "Volatility and covariance (Ch. 20 §20.2)"),
    (9, "stage09_risk", "Risk management (Ch. 21)"),
    (10, "stage10_combination", "Combining strategies (Ch. 22 §22.5)"),
    (11, "stage11_validation", "Walk-forward, robustness and leakage"),
    (12, "stage12_results", "Final comparison and results"),
    (13, "stage13_ml", "Machine learning extension (Ch. 23)"),
    # Optional research branch: deliberately outside the headline ladder, so
    # it runs only when asked for (--to 14, or --only 14).
    (14, "stage14_extensions", "Pairs trading and PCA stat-arb (Ch. 22 §22.3.3-7)"),
]
DEFAULT_LAST_STAGE = 13


def clear_derived(config, logger) -> None:
    """Delete everything that is rebuilt, never the immutable raw data."""
    targets = [
        config.path("processed"), config.path("features"),
        config.reports_dir("figures"), config.reports_dir("tables"),
        config.root / "experiments" / "registry.jsonl",
    ]
    for target in targets:
        if target.is_dir():
            for path in target.iterdir():
                if path.name == ".gitkeep":
                    continue
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            logger.info("cleared %s", target)
        elif target.exists():
            target.unlink()
            logger.info("removed %s", target)
    logger.info("raw data left untouched: %s", config.path("raw"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the research pipeline")
    parser.add_argument("--from", dest="start", type=int, default=1, help="first stage")
    parser.add_argument("--to", dest="end", type=int, default=DEFAULT_LAST_STAGE,
                        help=f"last stage (default {DEFAULT_LAST_STAGE}; pass 14 for the "
                             "optional pairs / PCA stat-arb branch)")
    parser.add_argument("--only", type=int, nargs="*", help="run only these stages")
    parser.add_argument("--download", action="store_true", help="fetch raw data in stage 1")
    parser.add_argument("--fresh", action="store_true", help="clear derived artefacts first")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going if a stage fails")
    args = parser.parse_args(argv)

    config = load_config()
    logger = stage_logger("run_all", config.root)
    logger.info("#" * 78)
    logger.info("SYSTEMATIC MULTI-ASSET RESEARCH PIPELINE")
    logger.info("config fingerprint: %s", config.fingerprint())
    logger.info("#" * 78)

    if args.fresh:
        clear_derived(config, logger)

    selected = (
        [s for s in STAGES if s[0] in set(args.only)] if args.only
        else [s for s in STAGES if args.start <= s[0] <= args.end]
    )
    if not selected:
        logger.error("no stages selected")
        return 1

    results, started = [], time.time()
    for number, module_name, description in selected:
        logger.info("")
        logger.info("=" * 78)
        logger.info("STAGE %d | %s", number, description)
        logger.info("=" * 78)
        stage_started = time.time()
        try:
            module = importlib.import_module(f"experiments.{module_name}")
            argv_stage = ["--download"] if (number == 1 and args.download) else []
            code = module.main(argv_stage)
            elapsed = time.time() - stage_started
            status = "ok" if code == 0 else f"exit {code}"
            results.append((number, module_name, status, elapsed))
        except Exception as exc:
            elapsed = time.time() - stage_started
            logger.exception("stage %d failed: %s", number, exc)
            results.append((number, module_name, f"FAILED: {exc}", elapsed))
            if not args.continue_on_error:
                break

    logger.info("")
    logger.info("=" * 78)
    logger.info("PIPELINE SUMMARY (%.1f minutes total)", (time.time() - started) / 60.0)
    logger.info("=" * 78)
    for number, name, status, elapsed in results:
        logger.info("  stage %2d  %-26s %-12s %6.1fs", number, name, status, elapsed)

    failed = [r for r in results if r[2].startswith("FAILED")]
    if failed:
        logger.error("%d stage(s) failed", len(failed))
        return 1
    logger.info("all stages completed. Reports in %s", config.reports_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
